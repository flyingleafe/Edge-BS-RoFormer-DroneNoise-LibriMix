"""Stage-0 helpers: per-rotor mean spectra, relative transfer functions (RTF),
and validation of the free-field ``1/r + delay`` propagation model against the
DREGON constant-speed single-motor recordings.

These recordings spin **one** motor at a fixed fundamental frequency while the
full 8-mic array records, so each file is a (near) single-source, 8-channel
observation. That makes them ideal for:

* measuring how a single rotor's spectrum looks from each microphone,
* estimating the rotor->mic relative transfer functions, and
* testing whether ``y_m(t) = (ref/r_m) * s(t - r_m/c)`` (the model in
  ``src/models/generative/positional_harmonic_gen.py``) actually holds.

All heavy lifting lives here so the companion notebook stays readable and this
code can be unit-tested without a kernel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import csd, welch

SPEED_OF_SOUND = 343.0  # m/s, matches positional_harmonic_gen.SPEED_OF_SOUND
MOTOR_IDS = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Data location + loading
# ---------------------------------------------------------------------------
def find_dregon_dir(start: Path | str | None = None) -> Path:
    """Walk upward from ``start`` (or CWD) until a ``data/DREGON`` dir is found.

    Works from a git worktree nested inside the main checkout: the parent walk
    reaches the checkout root that actually holds the (git-ignored) dataset.
    """
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for base in [here, *here.parents]:
        cand = base / "data" / "DREGON"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Could not locate data/DREGON by walking up from "
        f"{here}. Set it explicitly or run `dload pull DREGON`."
    )


def motors_dir(dregon_dir: Path) -> Path:
    return dregon_dir / "DREGON_individual_motors_recordings"


def available_speeds(dregon_dir: Path) -> dict[int, list[int]]:
    """``{motor_id: sorted [fundamental-Hz speeds]}`` present on disk."""
    out: dict[int, list[int]] = {m: [] for m in MOTOR_IDS}
    for wav in motors_dir(dregon_dir).rglob("Motor*.wav"):
        m = re.match(r"Motor(\d)_(\d+)\.wav", wav.name)
        if m:
            out[int(m.group(1))].append(int(m.group(2)))
    return {k: sorted(v) for k, v in out.items() if v}


def load_motor(
    dregon_dir: Path, motor_id: int, speed: int, max_seconds: float | None = None
) -> tuple[np.ndarray, int]:
    """Return ``(x, sr)`` with ``x`` shaped ``(8, N)`` float64.

    ``speed`` is the filename integer, which we verify below equals the tonal
    fundamental in Hz (harmonics land at ``k * speed``). ``max_seconds`` truncates
    the (stationary) recording to keep the spectral estimation snappy — a ~20 s
    window still gives hundreds of Welch segments.
    """
    matches = list(motors_dir(dregon_dir).rglob(f"Motor{motor_id}_{speed}.wav"))
    if not matches:
        raise FileNotFoundError(f"No Motor{motor_id}_{speed}.wav under {dregon_dir}")
    frames = int(max_seconds * sf.info(str(matches[0])).samplerate) if max_seconds else -1
    audio, sr = sf.read(str(matches[0]), frames=frames)
    return np.asarray(audio, dtype=np.float64).T, int(sr)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def distance_matrix(mic_pos: np.ndarray, rotor_pos: np.ndarray) -> np.ndarray:
    """Euclidean rotor->mic distances, shape ``(n_rotor, n_mic)`` in metres."""
    return np.linalg.norm(mic_pos[None, :, :] - rotor_pos[:, None, :], axis=2)


def rotate_z(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate ``(..., 3)`` points about the +z axis by ``degrees``."""
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points @ rot.T


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------
def mean_spectrum(x: np.ndarray, sr: int, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel. Returns ``(freqs[F], psd[C, F])`` (V^2/Hz)."""
    freqs, psd = welch(x, fs=sr, nperseg=nperseg, axis=-1)
    return np.asarray(freqs), np.asarray(psd)


def harmonic_freqs(speed: int, fmax: float) -> np.ndarray:
    """Tonal harmonic frequencies ``k * speed`` up to ``fmax`` Hz."""
    kmax = int(fmax // speed)
    return speed * np.arange(1, kmax + 1)


# ---------------------------------------------------------------------------
# Relative Transfer Functions (RTF)
# ---------------------------------------------------------------------------
def estimate_rtf(
    x: np.ndarray, sr: int, ref: int, nperseg: int = 8192
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured RTF of every channel relative to ``ref``.

    For a single source, ``X_m(f) = H_m(f) S(f) + noise``. The least-squares /
    Wiener estimate of the *ratio* ``H_m/H_ref`` that is robust to uncorrelated
    per-mic noise is the cross-spectrum over the reference auto-spectrum::

        RTF_m(f) = S_{x_m, x_ref}(f) / S_{x_ref, x_ref}(f)

    (Welch-averaged over frames -> the expectation). We also return the
    magnitude-squared coherence, which says *where* the RTF is trustworthy:
    coherence near 1 => the two mics see the same coherent source at that
    frequency; low coherence => uncorrelated/broadband/scattered energy where no
    linear single-source model (free-field or measured) can reproduce mic ``m``.

    Returns ``(freqs[F], rtf[C, F] complex, coherence[C, F])``.
    """
    freqs, s_rr = welch(x[ref], fs=sr, nperseg=nperseg)
    rtf = np.empty((x.shape[0], freqs.size), dtype=np.complex128)
    coh = np.empty((x.shape[0], freqs.size), dtype=np.float64)
    for m in range(x.shape[0]):
        # Reuse the same Welch spectra for RTF and coherence (one csd + one
        # auto-spectrum per mic) instead of a separate `coherence` call, which
        # would recompute both auto-spectra and the cross-spectrum again.
        _, s_mr = csd(x[m], x[ref], fs=sr, nperseg=nperseg)
        _, s_mm = welch(x[m], fs=sr, nperseg=nperseg)
        rtf[m] = s_mr / s_rr
        coh[m] = np.abs(s_mr) ** 2 / (s_mm * s_rr + 1e-30)
    return np.asarray(freqs), rtf, coh


def freefield_rtf(
    freqs: np.ndarray, dist_to_rotor: np.ndarray, ref: int, c: float = SPEED_OF_SOUND
) -> np.ndarray:
    """Free-field prediction of the RTF ``H_m/H_ref`` for one rotor.

    Magnitude is the ``1/r`` ratio ``r_ref / r_m`` (frequency-independent) and
    phase is the pure propagation-delay term ``-2*pi*f*(r_m - r_ref)/c``.

    ``dist_to_rotor`` is the length-``C`` vector of mic distances for the rotor.
    Returns ``rtf_ff[C, F]`` complex.
    """
    r = np.asarray(dist_to_rotor, dtype=np.float64)
    mag = (r[ref] / r)[:, None]
    phase = -2.0 * np.pi * freqs[None, :] * (r[:, None] - r[ref]) / c
    return mag * np.exp(1j * phase)


def coherence_weighted_mag_error(
    rtf: np.ndarray,
    rtf_ff: np.ndarray,
    coh: np.ndarray,
    freqs: np.ndarray,
    band: tuple[float, float] = (80.0, 2500.0),
) -> np.ndarray:
    """Per-mic coherence-weighted mean |RTF| error vs free-field, in dB.

    Weighting by coherence focuses the comparison on frequencies where the RTF
    is actually well-defined. Returns length-``C`` array (dB).
    """
    sel = (freqs >= band[0]) & (freqs <= band[1])
    err_db = 20.0 * np.log10(np.abs(rtf[:, sel]) + 1e-12) - 20.0 * np.log10(
        np.abs(rtf_ff[:, sel]) + 1e-12
    )
    w = coh[:, sel]
    return np.sum(err_db * w, axis=1) / np.maximum(np.sum(w, axis=1), 1e-9)


# ---------------------------------------------------------------------------
# Time-difference-of-arrival (delay validation)
# ---------------------------------------------------------------------------
def gcc_phat_tdoa(a: np.ndarray, b: np.ndarray, max_lag: int = 400) -> float:
    """GCC-PHAT time delay of ``a`` relative to ``b`` in samples (sub-sample).

    Positive => ``a`` lags ``b`` (source farther from mic ``a``). PHAT whitening
    makes this robust for broadband content; parabolic interpolation refines the
    integer peak. Used to validate the free-field delay ``(r_a - r_b)/c``.
    """
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    spec = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    spec /= np.abs(spec) + 1e-12
    cc = np.fft.irfft(spec, n)
    cc = np.concatenate((cc[-max_lag:], cc[: max_lag + 1]))
    lags = np.arange(-max_lag, max_lag + 1)
    i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1:
        denom = cc[i - 1] - 2 * cc[i] + cc[i + 1]
        frac = 0.5 * (cc[i - 1] - cc[i + 1]) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    return float(lags[i] + frac)


def measured_tdoa_row(x: np.ndarray, ref: int, max_lag: int = 400) -> np.ndarray:
    """GCC-PHAT TDOA of every channel relative to ``ref`` (samples)."""
    return np.array([gcc_phat_tdoa(x[m], x[ref], max_lag) for m in range(x.shape[0])])


def freefield_tdoa_row(
    dist_to_rotor: np.ndarray, ref: int, sr: int, c: float = SPEED_OF_SOUND
) -> np.ndarray:
    """Free-field predicted TDOA (samples) of each mic relative to ``ref``."""
    r = np.asarray(dist_to_rotor, dtype=np.float64)
    return (r - r[ref]) / c * sr


@dataclass
class FrameAlignment:
    """Result of aligning the mic frame to the rotor frame via TDOA."""

    best_degrees: float
    best_corr: float
    identity_corr: float
    angles: np.ndarray
    corr_curve: np.ndarray


def align_mic_frame(
    x_by_rotor: dict[int, np.ndarray],
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    sr: int,
    ref: int = 0,
    n_angles: int = 361,
) -> FrameAlignment:
    """Find the z-rotation of the mic frame that best matches measured TDOAs.

    DREGON ships ``micPos`` and ``rotorsPos`` that (empirically) disagree by a
    ~180 deg rotation about z: the identity geometry yields *anti-correlated*
    TDOAs. We sweep a z-rotation applied to ``mic_pos`` and correlate the
    predicted TDOA matrix against the GCC-PHAT-measured one across all rotors.
    """
    measured = np.vstack([measured_tdoa_row(x_by_rotor[r], ref) for r in sorted(x_by_rotor)])
    rotor_order = sorted(x_by_rotor)

    def predicted(mp: np.ndarray) -> np.ndarray:
        dist = distance_matrix(mp, rotor_pos)[rotor_order]
        return np.vstack([freefield_tdoa_row(dist[i], ref, sr) for i in range(len(rotor_order))])

    angles = np.linspace(0.0, 360.0, n_angles)
    corrs = np.array(
        [
            np.corrcoef(predicted(rotate_z(mic_pos, a)).ravel(), measured.ravel())[0, 1]
            for a in angles
        ]
    )
    identity = float(np.corrcoef(predicted(mic_pos).ravel(), measured.ravel())[0, 1])
    i = int(np.nanargmax(corrs))
    return FrameAlignment(float(angles[i]), float(corrs[i]), identity, angles, corrs)
