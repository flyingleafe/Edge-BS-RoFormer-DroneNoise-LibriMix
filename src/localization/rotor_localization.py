"""
Rotor localization: estimate each rotor's position relative to a mic array.

Inputs
------
- ``audio``         : ``(C, N)`` multichannel recording of the drone noise.
- ``mic_positions`` : ``(C, 3)`` *approximate* microphone coordinates (metres).
- ``sr``            : sample rate.

Output: estimated 3-D position of each rotor, in the microphone frame.

Two modes
---------
1. **Audio-only** (``rps=None``).  Build one near-field SRP-PHAT map over a 3-D
   grid and take the ``n_rotors`` strongest, spatially separated peaks.  This is
   the general case but the four rotors are coherent, near-field, and close
   together, so peak resolution is limited.

2. **RPS-aided** (``rps`` given).  Each rotor has its own instantaneous speed,
   hence its own harmonic series ``k * f0_r(t)``.  We build a soft harmonic mask
   per rotor in the STFT domain, apply it to every channel, and localize each
   rotor's *isolated* signal with its own SRP-PHAT map (a single dominant peak).
   This leverages the rotor-speed telemetry that the rest of this project is
   built around.  It separates rotors well only when their speeds differ enough
   for the harmonic masks to be distinct (poor for steady symmetric hover).

The two modes share the same near-field SRP-PHAT engine in :mod:`.srp_phat`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .srp_phat import (
    SPEED_OF_SOUND,
    Grid,
    extract_peaks,
    make_grid,
    phat_cross_spectrum,
    refine_peak,
    srp_power,
)


@dataclass
class RotorLocalizationResult:
    positions: np.ndarray  # (n_rotors, 3) estimated rotor positions (metres)
    powers: np.ndarray  # (n_rotors,) SRP power at each estimate
    grid: Grid  # the coarse search grid (for plotting)
    coarse_maps: list[np.ndarray]  # per-estimate coarse SRP map, reshaped (nx,ny,nz)


def default_search_bounds(
    mic_positions: np.ndarray,
    *,
    margin: float = 0.30,
    z_min: float = -0.05,
    z_max: float = 0.40,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """A search box centred on the array, large enough to contain the rotors.

    Rotors sit slightly outside the mic array and (on a typical quadcopter) above
    it; the default box spans ``+/- (aperture/2 + margin)`` in x/y and an
    explicit z range.
    """
    c = mic_positions.mean(0)
    half = np.abs(mic_positions - c).max() + margin
    return (
        (float(c[0] - half), float(c[0] + half)),
        (float(c[1] - half), float(c[1] + half)),
        (z_min, z_max),
    )


# ---------------------------------------------------------------------------
# RPS -> harmonic STFT mask
# ---------------------------------------------------------------------------


def _frame_times(n_samples: int, sr: float, hop: int) -> np.ndarray:
    """Centre time (s) of each ``center=True`` STFT frame."""
    n_frames = n_samples // hop + 1
    return np.arange(n_frames) * (hop / sr)


def harmonic_mask(
    f0_track: np.ndarray,  # (T_frames,) fundamental Hz per frame
    freqs: np.ndarray,  # (F,) bin centre Hz
    *,
    n_harmonics: int = 20,
    rel_width: float = 0.02,
    min_width_hz: float = 15.0,
) -> np.ndarray:
    """Soft Gaussian mask around ``k * f0(t)`` for one rotor.

    Returns ``(F, T)`` in ``[0, 1]``.  Each harmonic contributes a Gaussian bump
    whose width grows with frequency (``rel_width`` of the harmonic, floored at
    ``min_width_hz``) to absorb small speed jitter.
    """
    f0 = np.asarray(f0_track, dtype=np.float64)[None, :]  # (1, T)
    fb = np.asarray(freqs, dtype=np.float64)[:, None]  # (F, 1)
    mask = np.zeros((freqs.shape[0], f0_track.shape[0]), dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        center = k * f0  # (1, T)
        width = np.maximum(rel_width * center, min_width_hz)
        mask += np.exp(-0.5 * ((fb - center) / width) ** 2)
    return np.clip(mask, 0.0, 1.0)


def _interp_rps_to_frames(
    rps_rotor: np.ndarray,  # (M,) fundamental Hz over the clip
    n_samples: int,
    sr: float,
    hop: int,
) -> np.ndarray:
    """Linearly resample a rotor speed track onto STFT frame centre times.

    ``rps_rotor`` is assumed to span the clip uniformly (telemetry is ~uniform).
    """
    ft = _frame_times(n_samples, sr, hop)
    src_t = np.linspace(0.0, ft[-1] if len(ft) else 0.0, len(rps_rotor))
    return np.interp(ft, src_t, rps_rotor)


def _masked_cross_spectrum(
    audio: torch.Tensor,
    sr: float,
    mask: np.ndarray,  # (F_full, T)
    *,
    n_fft: int,
    hop: int,
    fmin: float,
    fmax: float,
    eps: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PHAT cross-spectrum of ``audio`` after applying a per-bin STFT ``mask``."""
    window = torch.hann_window(n_fft, device=device)
    X = torch.stft(
        audio.to(device),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )  # (C, F, T)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(device)
    m = torch.as_tensor(mask, dtype=X.real.dtype, device=device)
    # mask may have a slightly different T than X (frame-count rounding); align.
    t = min(m.shape[-1], X.shape[-1])
    Xm = X[..., :t] * m[None, :, :t]
    band = (freqs >= fmin) & (freqs <= fmax)
    Xm = Xm[:, band, :]
    fb = freqs[band]
    Xhat = Xm / (Xm.abs() + eps)
    R = torch.einsum("ift,jft->ijf", Xhat, Xhat.conj())
    return R, fb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def localize_rotors(
    audio: np.ndarray | torch.Tensor,
    mic_positions: np.ndarray,
    sr: float,
    *,
    n_rotors: int = 4,
    rps: np.ndarray | None = None,
    search_bounds: tuple | None = None,
    coarse_step: float = 0.02,
    refine: bool = True,
    n_fft: int = 4096,
    hop: int | None = None,
    fmin: float = 150.0,
    fmax: float = 4000.0,
    n_harmonics: int = 20,
    c: float = SPEED_OF_SOUND,
    suppression_radius: float = 0.12,
    eps: float = 1e-8,
    device: str | torch.device = "cpu",
) -> RotorLocalizationResult:
    """Estimate rotor positions from multichannel drone audio.

    Parameters
    ----------
    audio : (C, N)
        Multichannel recording, channel-first.
    mic_positions : (C, 3)
        Approximate microphone coordinates (metres).
    sr : float
        Sample rate.
    n_rotors : int
        Number of rotors to localize.
    rps : (n_rotors, M) or None
        Per-rotor speed in rev/s (== fundamental Hz), uniformly spanning the
        clip.  If given, each rotor is isolated by a harmonic STFT mask and
        localized separately.  If ``None``, the rotors are taken as the top
        ``n_rotors`` peaks of a single combined SRP-PHAT map.
    search_bounds : ((x0,x1),(y0,y1),(z0,z1)) or None
        Search box (metres).  Defaults to :func:`default_search_bounds`.
    coarse_step : float
        Coarse grid spacing (metres).
    refine : bool
        If True, refine each coarse estimate with a fine local grid search.
    """
    x: torch.Tensor = (
        audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio))
    )
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    n_samples = x.shape[-1]
    hop = n_fft // 4 if hop is None else hop

    mic_positions = np.asarray(mic_positions, dtype=np.float64)
    if search_bounds is None:
        search_bounds = default_search_bounds(mic_positions)
    grid = make_grid(search_bounds, coarse_step, device=device, dtype=torch.float64)

    positions: list[np.ndarray] = []
    powers: list[float] = []
    coarse_maps: list[np.ndarray] = []

    if rps is None:
        # ---- audio-only: one map, top-K separated peaks --------------------
        R, freqs = phat_cross_spectrum(
            x, sr, n_fft=n_fft, hop=hop, fmin=fmin, fmax=fmax, eps=eps, device=device
        )
        power = srp_power(R, freqs, mic_positions, grid, c=c)
        peaks, vals = extract_peaks(power, grid, n_rotors, suppression_radius=suppression_radius)
        for k in range(peaks.shape[0]):
            pos = peaks[k]
            val = float(vals[k].item())
            if refine:
                pos, val = refine_peak(R, freqs, mic_positions, pos, c=c, device=device)
            positions.append(pos.cpu().numpy())
            powers.append(val)
        coarse_maps.append(grid.reshape_map(power).cpu().numpy())
    else:
        # ---- RPS-aided: isolate each rotor, localize independently ---------
        rps_arr = np.asarray(rps, dtype=np.float64)
        if rps_arr.ndim != 2:
            raise ValueError(f"rps must be (n_rotors, M), got shape {rps_arr.shape}")
        freqs_full = torch.fft.rfftfreq(n_fft, d=1.0 / sr).cpu().numpy()
        for r in range(rps_arr.shape[0]):
            f0 = _interp_rps_to_frames(rps_arr[r], n_samples, sr, hop)
            mask = harmonic_mask(f0, freqs_full, n_harmonics=n_harmonics)
            R, freqs = _masked_cross_spectrum(
                x, sr, mask, n_fft=n_fft, hop=hop, fmin=fmin, fmax=fmax, eps=eps, device=device
            )
            power = srp_power(R, freqs, mic_positions, grid, c=c)
            i = int(torch.argmax(power).item())
            pos, val = grid.points[i], float(power[i].item())
            if refine:
                pos, val = refine_peak(R, freqs, mic_positions, pos, c=c, device=device)
            positions.append(pos.cpu().numpy())
            powers.append(val)
            coarse_maps.append(grid.reshape_map(power).cpu().numpy())

    return RotorLocalizationResult(
        positions=np.stack(positions),
        powers=np.asarray(powers),
        grid=grid,
        coarse_maps=coarse_maps,
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def match_and_score(
    estimated: np.ndarray,  # (K, 3)
    ground_truth: np.ndarray,  # (K, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match estimates to ground-truth rotors; return (perm, errors).

    ``perm[k]`` is the ground-truth index assigned to estimate ``k``;
    ``errors[k]`` is the Euclidean distance (m) of that pairing.
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(estimated[:, None, :] - ground_truth[None, :, :], axis=-1)
    row, col = linear_sum_assignment(cost)
    errors = cost[row, col]
    return col, errors
