"""Strong noise-chunk augmentations for the online mixer (G6).

Motivation. Every criterion-2.3 arm overfits (train falls, val ~doubles within
~20 epochs of best), and the mixture-level augmentation family is provably
weak for this task: ``random_polarity`` is an exact no-op for magnitude and
instantaneous-frequency front-ends, ``random_gain`` is a log-magnitude offset,
and only ``channel_drop`` plus the mild +-12% ``noise_time_warp`` change
anything the model can see. G5 showed the two-stage schedule (50k unaugmented
warmup, then augs) is load-bearing — augs-from-sample-0 made val WORSE (117.9
vs 63.7) — so G6 keeps the E12 schedule and replaces the *content* of the
augmented stage with the six transforms below.

These are **noise-chunk** augmentations: they run on the sampled noise Frame
*before* speech mixing (unlike ``policy.augmentations``, which is post-mix on
the mixture), because two of them need the RPS labels — ``freq_scale``
rescales them, ``tooth_dropout`` reads them to find the comb teeth. They are
declared under ``policy.noise_augmentations`` with exactly the
``probability`` + ``choices`` schema of the mixture-level block::

    noise_augmentations:
      probability: 0.7
      choices:
        - freq_scale: {alpha_low: 0.75, alpha_high: 1.3}
        - spectral_recolor: {}
        - random_reverb: {}
        - tooth_dropout: {}
        - spec_mask: {}
        - floor_inject: {}

The six transforms (all take/return an ``(audio (C, T), rps_label (R, L))``
pair; STFT-domain ones use the model's own 2048/512 grid):

* ``freq_scale`` — resample the chunk by ``alpha ~ U(0.75, 1.3)`` *without*
  preserving duration (a pitch/comb scale), multiply the RPS labels by
  ``alpha`` (and time-compress them consistently), crop/zero-pad to the chunk
  length (the padded tail gets rps=0 — silence at zero rotor speed is the
  project's own amplitude convention). The key one: it manufactures genuinely
  new (audio, RPS) pairs.
* ``spectral_recolor`` — multiply the STFT magnitude by a smooth random curve
  (gains ``U(-8, +8)`` dB at 10 log-spaced anchors 30 Hz..8 kHz, interpolated
  over bins in log-frequency, independent per channel). Labels untouched.
* ``random_reverb`` — FFT-convolve each channel with a random synthetic RIR
  from a lazily-built in-memory bank (exponential-decay colored-noise tails:
  RT60 ~ U(0.1, 0.8) s, direct-to-reverb ratio ~ U(3, 15) dB; diffuse tails
  are drawn independently per channel, matching the incoherence of real
  diffuse reverb across mics). Output renormalized to the pre-aug global RMS.
* ``tooth_dropout`` — label-aware harmonic masking: 1-4 random (rotor r,
  harmonic k<=25) pairs, each zeroing the +-2 STFT bins around the
  time-varying tooth frequency ``k * rps_r(t)`` (the project's ``f0 = rps``
  comb convention). Forces redundancy across teeth.
* ``spec_mask`` — SpecAugment-style: 1-3 random frequency bands (width
  U(50, 400) Hz) and 0-2 time masks (<=100 ms) zeroed in the STFT.
* ``floor_inject`` — add ``1/f^tilt`` colored noise (tilt ~ U(0, 2)) at
  ``U(-20, 0)`` dB relative to the chunk's global RMS. Washes out weak teeth
  the way a real broadband floor does.

Wiring: :func:`maybe_apply_noise_augmentation` is called by
``OnlineMixIterableDataset._generate_rps_sample`` right after the (optional)
time-warp. On a hit it returns a fresh Frame in :mod:`.time_warp`'s output
convention — audio exactly ``target_len`` long plus a clean uniform-grid
``rps`` label track — so downstream extraction/mixing/target interpolation are
untouched. The fire decision consumes RNG only when the key is present
(mirrors ``_apply_one_augmentation``), keeping un-augmented streams
byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import scipy.signal as sps
import tdseries as td

from data_processing.dregon import clean_command_spikes
from data_processing.time_warp import DEFAULT_LABEL_RATE_HZ, _resolve_rps_track

#: STFT grid for the spectral augmentations — the model's own analysis grid,
#: so masks/curves act on exactly the bins the predictor sees.
AUG_N_FFT = 2048
AUG_HOP = 512

_AUG_NAMES = (
    "freq_scale",
    "spectral_recolor",
    "random_reverb",
    "tooth_dropout",
    "spec_mask",
    "floor_inject",
)


# ── (audio, label) <-> Frame plumbing ───────────────────────────────────────


def _as_ct(audio_data: np.ndarray) -> tuple[np.ndarray, bool]:
    data = np.asarray(audio_data, dtype=np.float32)
    if data.ndim == 1:
        return data[None, :], True
    return data, False


def _fit_len(x: np.ndarray, n: int) -> np.ndarray:
    """Crop/zero-pad the last axis to exactly ``n``."""
    if x.shape[-1] > n:
        return x[..., :n]
    if x.shape[-1] < n:
        pad = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
        return np.pad(x, pad)
    return x


def extract_pair(
    frame: td.Frame,
    *,
    target_len: int,
    sample_rate: int,
    label_rate_hz: float = DEFAULT_LABEL_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Frame -> ``(audio (C, target_len), label (R, L), audio_was_mono)``.

    The label is the frame's rotor track (cleaned when it is a raw
    ``motors_command``/``motors_measured``) interpolated onto a uniform
    ``label_rate_hz`` grid over the chunk — the same label convention
    :func:`data_processing.time_warp.apply_time_warp` emits.
    """
    audio = cast(td.Series, frame["audio"])
    data, is_mono = _as_ct(np.asarray(audio.data))
    data = _fit_len(data, target_len)

    n_label = int(np.ceil(target_len / float(sample_rate) * label_rate_hz)) + 1
    t_label = np.arange(n_label, dtype=np.float64) / float(label_rate_hz)
    rps_key, needs_clean = _resolve_rps_track(frame)
    motor = cast(td.Series, frame[rps_key])
    if motor.data is None or motor.dim_size("time") == 0:
        return data, np.zeros((4, n_label), dtype=np.float32), is_mono
    vals = np.asarray(motor.data, dtype=np.float64)
    if needs_clean:
        vals = clean_command_spikes(vals)
    ti = motor.tindex
    if isinstance(ti, td.StampIndex):
        times = np.asarray(ti.abs_stamps, dtype=np.float64)
    else:
        times = np.asarray(cast(td.GridIndex, ti).sample_times(), dtype=np.float64)
    times = times - float(audio.t_start)
    label = np.empty((vals.shape[0], n_label), dtype=np.float32)
    for r in range(vals.shape[0]):
        label[r] = np.interp(t_label, times, vals[r]).astype(np.float32)
    return data, label, is_mono


def build_frame(
    audio: np.ndarray,
    label: np.ndarray,
    *,
    sample_rate: int,
    label_rate_hz: float,
    source: td.Frame,
    audio_was_mono: bool,
) -> td.Frame:
    """Rebuild a minimal (audio + clean ``rps`` + meta) Frame, timeline at 0 —
    the exact output convention of :func:`~data_processing.time_warp.apply_time_warp`."""
    out_audio = audio[0] if audio_was_mono else audio
    dims = cast(td.Series, source["audio"]).dims
    entries: dict[str, Any] = {
        "audio": td.uniform(
            np.ascontiguousarray(out_audio, dtype=np.float32), sample_rate, dims=dims, t_start=0.0
        ),
        "rps": td.uniform(
            np.ascontiguousarray(label, dtype=np.float32),
            int(round(label_rate_hz)),
            dims=("rotor", "time"),
            t_start=0.0,
        ),
    }
    if "meta" in source:
        entries["meta"] = source["meta"]
    return td.Frame(entries)


# ── STFT helpers (COLA-exact round trip on the model grid) ──────────────────


def _stft(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Z = sps.stft(audio, fs=sample_rate, nperseg=AUG_N_FFT, noverlap=AUG_N_FFT - AUG_HOP)
    return f, t, Z


def _istft(Z: np.ndarray, sample_rate: int, target_len: int) -> np.ndarray:
    _, x = sps.istft(Z, fs=sample_rate, nperseg=AUG_N_FFT, noverlap=AUG_N_FFT - AUG_HOP)
    return _fit_len(np.asarray(x, dtype=np.float32), target_len)


# ── The six augmentations ───────────────────────────────────────────────────


def _freq_scale(
    audio: np.ndarray,
    label: np.ndarray,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    sample_rate: int,
    label_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    import soxr

    alpha = float(
        rng.uniform(float(params.get("alpha_low", 0.75)), float(params.get("alpha_high", 1.3)))
    )
    T = audio.shape[-1]
    # Playback at alpha x speed: all frequencies (and the comb) scale by alpha,
    # duration scales by 1/alpha. soxr takes (T, C) float32 and float rates.
    y = soxr.resample(np.ascontiguousarray(audio.T), float(sample_rate), sample_rate / alpha).T
    out = _fit_len(np.asarray(y, dtype=np.float32), T)

    # Labels: r_new(t) = alpha * r_old(alpha * t); the zero-padded audio tail
    # (alpha > 1) is silence => rps 0 there, consistent with the zero-rps ==
    # silence amplitude convention used by the synthetic pools.
    L = label.shape[-1]
    t_label = np.arange(L, dtype=np.float64) / float(label_rate_hz)
    dur = T / float(sample_rate)
    src_t = alpha * t_label
    in_range = src_t <= dur + 1e-9
    new_label = np.empty_like(label)
    for r in range(label.shape[0]):
        new_label[r] = np.where(
            in_range, alpha * np.interp(src_t, t_label, label[r].astype(np.float64)), 0.0
        ).astype(np.float32)
    return out, new_label


def _spectral_recolor(
    audio: np.ndarray,
    label: np.ndarray,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    sample_rate: int,
    label_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    gain_db = float(params.get("gain_db", 8.0))
    n_anchors = int(params.get("n_anchors", 10))
    f_lo = float(params.get("f_low", 30.0))
    f_hi = float(params.get("f_high", 8000.0))
    T = audio.shape[-1]
    f, _, Z = _stft(audio, sample_rate)
    anchors = np.geomspace(f_lo, f_hi, n_anchors)
    log_f = np.log(np.maximum(f, f_lo * 0.5))  # sub-30 Hz bins clamp to the first anchor
    for ch in range(Z.shape[0]):
        gains = rng.uniform(-gain_db, gain_db, size=n_anchors)
        curve_db = np.interp(log_f, np.log(anchors), gains)  # clamps beyond the anchor span
        Z[ch] *= (10.0 ** (curve_db / 20.0))[:, None]
    return _istft(Z, sample_rate, T), label


_RIR_BANK: dict[tuple[Any, ...], list[np.ndarray]] = {}

#: Fixed bank seed: the RIR bank is a deterministic module resource (like a
#: dataset), not part of the per-sample RNG stream — workers all build the
#: same ~10 MB bank in-process (<1 s; no disk cache needed) and per-chunk
#: variety comes from the bank *index* draws.
_RIR_BANK_SEED = 20260724


def _rir_bank(
    sample_rate: int,
    n_rirs: int,
    rt60_lo: float,
    rt60_hi: float,
    drr_lo: float,
    drr_hi: float,
) -> list[np.ndarray]:
    key = (sample_rate, n_rirs, rt60_lo, rt60_hi, drr_lo, drr_hi)
    bank = _RIR_BANK.get(key)
    if bank is not None:
        return bank
    gen = np.random.default_rng(_RIR_BANK_SEED)
    bank = []
    for _ in range(n_rirs):
        rt60 = float(gen.uniform(rt60_lo, rt60_hi))
        drr_db = float(gen.uniform(drr_lo, drr_hi))
        tilt = float(gen.uniform(0.0, 1.0))  # mild pink-ish tail coloration
        n = max(int(round(rt60 * sample_rate)), 8)
        white = gen.standard_normal(n)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        shape = np.ones_like(freqs)
        nz = freqs > 0
        shape[nz] = freqs[nz] ** (-tilt / 2.0)
        shape[0] = 0.0
        colored = np.fft.irfft(spec * shape, n)
        t = np.arange(n) / float(sample_rate)
        tail = colored * np.exp(-6.9078 * t / rt60)  # amplitude -60 dB at RT60
        e_tail = float(np.sum(tail**2)) or 1.0
        direct = np.sqrt(e_tail * 10.0 ** (drr_db / 10.0))
        rir = np.concatenate([[direct], tail]).astype(np.float64)
        bank.append(rir / np.sqrt(np.sum(rir**2)))
    _RIR_BANK[key] = bank
    return bank


def _random_reverb(
    audio: np.ndarray,
    label: np.ndarray,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    sample_rate: int,
    label_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    bank = _rir_bank(
        sample_rate,
        int(params.get("n_rirs", 200)),
        float(params.get("rt60_low", 0.1)),
        float(params.get("rt60_high", 0.8)),
        float(params.get("drr_low_db", 3.0)),
        float(params.get("drr_high_db", 15.0)),
    )
    T = audio.shape[-1]
    rms_in = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) or 1.0
    out = np.empty_like(audio)
    for ch in range(audio.shape[0]):
        rir = bank[int(rng.integers(0, len(bank)))]
        out[ch] = sps.fftconvolve(audio[ch].astype(np.float64), rir)[:T].astype(np.float32)
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) or 1.0
    return (out * np.float32(rms_in / rms_out)), label


def _tooth_dropout(
    audio: np.ndarray,
    label: np.ndarray,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    sample_rate: int,
    label_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    max_teeth = int(params.get("max_teeth", 4))
    max_harm = int(params.get("max_harmonic", 25))
    halfwidth = int(params.get("halfwidth_bins", 2))
    T = audio.shape[-1]
    R = label.shape[0]
    teeth = params.get("teeth")  # explicit [[rotor, k], ...] override (tests)
    if teeth is None:
        n_teeth = int(rng.integers(1, max_teeth + 1))
        teeth = [
            (int(rng.integers(0, R)), int(rng.integers(1, max_harm + 1))) for _ in range(n_teeth)
        ]
    f, t, Z = _stft(audio, sample_rate)
    df = float(f[1] - f[0])
    t_label = np.arange(label.shape[-1], dtype=np.float64) / float(label_rate_hz)
    for r, k in teeth:
        # Tooth frequency track on the STFT frame times (f0 = rps convention).
        ft = k * np.interp(t, t_label, label[int(r)].astype(np.float64))
        mask = np.abs(f[:, None] - ft[None, :]) <= (halfwidth + 0.5) * df
        Z[:, mask] = 0.0
    return _istft(Z, sample_rate, T), label


def _spec_mask(
    audio: np.ndarray,
    label: np.ndarray,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    sample_rate: int,
    label_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    max_f = int(params.get("max_freq_masks", 3))
    w_lo, w_hi = (float(v) for v in params.get("freq_width_hz", (50.0, 400.0)))
    max_t = int(params.get("max_time_masks", 2))
    t_max_ms = float(params.get("time_width_ms", 100.0))
    T = audio.shape[-1]
    f, t, Z = _stft(audio, sample_rate)
    n_f = int(rng.integers(1, max_f + 1))
    for _ in range(n_f):
        w = float(rng.uniform(w_lo, w_hi))
        f0 = float(rng.uniform(0.0, max(float(f[-1]) - w, 0.0)))
        Z[:, (f >= f0) & (f <= f0 + w), :] = 0.0
    n_t = int(rng.integers(0, max_t + 1))
    for _ in range(n_t):
        wt = float(rng.uniform(0.0, t_max_ms / 1000.0))
        t0 = float(rng.uniform(0.0, max(float(t[-1]) - wt, 0.0)))
        Z[:, :, (t >= t0) & (t <= t0 + wt)] = 0.0
    return _istft(Z, sample_rate, T), label


def _floor_inject(
    audio: np.ndarray,
    label: np.ndarray,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    sample_rate: int,
    label_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    tilt = float(
        rng.uniform(float(params.get("tilt_low", 0.0)), float(params.get("tilt_high", 2.0)))
    )
    level_db = float(
        rng.uniform(
            float(params.get("level_low_db", -20.0)), float(params.get("level_high_db", 0.0))
        )
    )
    T = audio.shape[-1]
    rms_in = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) or 1.0
    target = rms_in * 10.0 ** (level_db / 20.0)
    freqs = np.fft.rfftfreq(T, 1.0 / sample_rate)
    shape = np.ones_like(freqs)
    nz = freqs > 0
    shape[nz] = freqs[nz] ** (-tilt / 2.0)
    shape[0] = 0.0
    out = audio.copy()
    for ch in range(audio.shape[0]):
        spec = np.fft.rfft(rng.standard_normal(T)) * shape
        sig = np.fft.irfft(spec, T)
        sig /= float(np.sqrt(np.mean(sig**2))) + 1e-12
        out[ch] += (sig * target).astype(np.float32)
    return out, label


_AUG_FUNCS = {
    "freq_scale": _freq_scale,
    "spectral_recolor": _spectral_recolor,
    "random_reverb": _random_reverb,
    "tooth_dropout": _tooth_dropout,
    "spec_mask": _spec_mask,
    "floor_inject": _floor_inject,
}
assert set(_AUG_FUNCS) == set(_AUG_NAMES)


# ── Policy entry point ──────────────────────────────────────────────────────


def maybe_apply_noise_augmentation(
    frame: td.Frame,
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
    *,
    target_len: int,
    sample_rate: int,
    label_rate_hz: float = DEFAULT_LABEL_RATE_HZ,
) -> td.Frame:
    """Fire-and-apply one noise-chunk augmentation from ``spec``.

    Mirrors ``_apply_one_augmentation``'s fire/choice contract: absent spec or
    ``probability <= 0`` consumes no RNG and returns the frame unchanged; on a
    hit, one uniformly-drawn choice runs on the extracted ``(audio, label)``
    pair and a fresh time-warp-convention Frame is returned.
    """
    if not spec:
        return frame
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return frame
    choices = list(spec.get("choices", []))
    if not choices:
        return frame
    choice = choices[int(rng.integers(0, len(choices)))]
    if isinstance(choice, str):
        name, params = choice, cast(Mapping[str, Any], {})
    elif isinstance(choice, Mapping):
        if len(choice) != 1:
            raise ValueError(f"noise augmentation choice must have one key, got {choice!r}")
        name, params = next(iter(choice.items()))
        params = params or {}
    else:
        raise ValueError(f"unsupported noise augmentation choice: {choice!r}")
    func = _AUG_FUNCS.get(str(name))
    if func is None:
        raise ValueError(f"unsupported noise augmentation: {name!r} (known: {_AUG_NAMES})")

    audio, label, was_mono = extract_pair(
        frame, target_len=target_len, sample_rate=sample_rate, label_rate_hz=label_rate_hz
    )
    audio, label = func(
        audio, label, params, rng, sample_rate=sample_rate, label_rate_hz=label_rate_hz
    )
    return build_frame(
        audio,
        label,
        sample_rate=sample_rate,
        label_rate_hz=label_rate_hz,
        source=frame,
        audio_was_mono=was_mono,
    )
