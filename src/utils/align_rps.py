"""
RPS–audio alignment via variable-phasor (VP) projection energy.

Problem
-------
Michael's DJI CSVs have timestamp errors:
1. A clock-reset *jump* mid-recording (the largest Δt in the series).
2. A clock-rate mismatch — the effective CSV sample period is not exactly
   1 true-second per 1 CSV-timestamp-second, causing ~0.5 s drift by end of
   recording.
3. A global offset between CSV time and audio time (empirical ``time_offset``).

The fix we optimise
-------------------
Given raw CSV timestamps ``ts_raw`` (already filtered by the rough
``time_offset`` from `data_processing.michaels`), we:

1. Detect the jump: ``jump_idx = argmax(diff(ts_raw))``.
2. Replace the pre-jump segment (0 … jump_idx+1) with a linear ramp from
   ``ts_raw[0]`` to ``ts_raw[jump_idx+2]``.
3. Scale post-jump deltas by a *rate* factor ``r`` (near 1.0).
4. Add a global *offset* ``o`` (seconds).

::

    ts_fixed[:jump_idx+2]  = linspace(ts_raw[0],  ts_raw[jump_idx+2], jump_idx+2)
    ts_fixed[jump_idx+2:]   = ts_raw[jump_idx+2:]
    ts_corrected[:jump_idx+2] = ts_fixed[:jump_idx+2] + o
    ts_corrected[jump_idx+2:] = (ts_fixed[jump_idx+1] +
                                 (ts_raw[jump_idx+2:] - ts_raw[jump_idx+2]) * r + o)

Then RPS values at CSV times are linearly interpolated to every audio sample,
converted to harmonic frequency series, and projected onto the audio with
``VP_transform``.  The total energy ``sum(|V|²)`` measures how well the
harmonic model matches the recording.  We maximise it over ``(o, r)`` with
L-BFGS-B (bounded, derivative-free via finite differences).

Usage
-----
::

    from data_processing.michaels import _load_raw
    from src.utils.align_rps import align_michaels_recording

    wav, ts_raw, motor_rps = _load_raw(wav_path, csv_path, time_offset, sr)
    result = align_michaels_recording(ts_raw, motor_rps, wav, sr)
    print(result.offset, result.rate, result.energy)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from src.models.generative.dsp import harmonic_freq_series
from src.models.generative.harmonic_transform import VP_transform

# ---------------------------------------------------------------------------
# Timestamp correction
# ---------------------------------------------------------------------------


def _find_jump(ts: np.ndarray) -> int:
    """Index *into* the diff array where the largest timestamp jump occurs."""
    return int(np.argmax(np.diff(ts)))


def correct_timestamps(
    ts_raw: np.ndarray,
    jump_idx: int,
    offset: float = 0.0,
    rate: float = 1.0,
) -> np.ndarray:
    """Return corrected timestamps given offset (s) and post-jump rate factor.

    Parameters
    ----------
    ts_raw : (M,) float64
        Raw CSV timestamps after ``time_offset`` filtering.
    jump_idx : int
        Index of the largest Δt (i.e. ``argmax(diff(ts_raw))``).
    offset : float
        Global time shift applied to *all* timestamps (seconds).
    rate : float
        Multiplier on post-jump time-deltas.  ``1.0`` = no correction.

    Returns
    -------
    ts_corrected : (M,) float64
    """
    split = jump_idx + 2  # first *post-jump* index (where the "to" side of the jump lands)

    # Pre-jump: linear ramp covering indices 0 … split (split+1 elements).
    # The last element equals ts_raw[split] (the anchor).
    pre = np.linspace(ts_raw[0], ts_raw[split], split + 1, dtype=np.float64)

    # Post-jump: scale deltas from the anchor.
    # post[0] == anchor == pre[-1], so we drop it to avoid a duplicate.
    anchor = ts_raw[split]
    post_deltas = ts_raw[split:] - anchor
    post = anchor + post_deltas * rate  # (M - split,)  elements

    ts_fixed = np.concatenate([pre, post[1:]])  # total: (split+1) + (M-split-1) = M
    return ts_fixed + offset


# ---------------------------------------------------------------------------
# RPS → full-audio-grid interpolation
def _interpolate_rps_full(
    motor_rps: np.ndarray,  # (n_motors, M)
    ts_corrected: np.ndarray,  # (M,)
    n_audio: int,  # total audio samples
    sr: int,
) -> torch.Tensor:
    """Linearly interpolate motor RPS to a FIXED uniform audio-rate grid.

    The audio grid is ``[0, 1/sr, …, (n_audio-1)/sr]`` — anchored at WAV
    time zero, independent of ``ts_corrected``.  ``ts_corrected`` shifts
    relative to this grid, so offset changes actually move the RPS in time.

    Returns
    -------
    rps_grid : (n_motors, n_audio) float32 tensor
    """
    audio_t = np.linspace(0.0, (n_audio - 1) / sr, n_audio, dtype=np.float64)

    n_motors = motor_rps.shape[0]
    rps_grid = np.empty((n_motors, n_audio), dtype=np.float32)
    for m in range(n_motors):
        rps_grid[m] = np.interp(audio_t, ts_corrected, motor_rps[m]).astype(np.float32)

    return torch.from_numpy(rps_grid)


# ---------------------------------------------------------------------------
# VP-transform energy
# ---------------------------------------------------------------------------


def compute_vp_energy(
    audio: torch.Tensor,  # (T,)  or  (1, T)
    freq_series: torch.Tensor,  # (C, T)  or  (M, H, T)
    window_len: int = 2048,
    hop_len: int = 512,
) -> float:
    """Total VP-projection energy = sum(|V|²) over all frames and components.

    Parameters
    ----------
    audio : (T,) or (1, T)
        Single-channel audio waveform.
    freq_series : (C, T) or (M, H, T)
        Instantaneous frequencies for each component at each audio sample.
        If 3‑D it is reshaped to (M*H, T).
    window_len, hop_len : int
        STFT‑like frame parameters for the VP transform.

    Returns
    -------
    energy : float  (Python scalar)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)
    if freq_series.dim() == 3:
        C = freq_series.shape[0] * freq_series.shape[1]
        freq_series = freq_series.reshape(C, freq_series.shape[-1])
    freq_series = freq_series.unsqueeze(0)  # (1, C, T)

    V = VP_transform(freq_series, audio, window_len=window_len, hop_len=hop_len)
    return torch.sum(torch.abs(V) ** 2).item()


# ---------------------------------------------------------------------------
# Full objective (for scipy) — operates on the *entire* recording
# ---------------------------------------------------------------------------


def _make_objective(
    ts_raw: np.ndarray,
    jump_idx: int,
    motor_rps: np.ndarray,  # (n_motors, M)
    audio_full: np.ndarray,  # (T,)  — full single-channel audio
    sr: int,
    window_len: int,
    hop_len: int,
    n_harmonics: int,
) -> object:
    """Return a callable ``f([offset, rate]) -> -energy`` for scipy.

    Interpolates RPS to the full audio grid and computes VP energy over the
    entire recording so the objective measures *global* alignment quality.
    """
    n_audio = len(audio_full)
    audio_tensor = torch.from_numpy(audio_full.astype(np.float32))

    def objective(x: np.ndarray) -> float:
        offset, rate = float(x[0]), float(x[1])

        ts_corr = correct_timestamps(ts_raw, jump_idx, offset=offset, rate=rate)
        rps_full = _interpolate_rps_full(motor_rps, ts_corr, n_audio, sr)
        # rps_full: (n_motors, n_audio)

        freqs = harmonic_freq_series(rps_full, n_harmonics)
        # freqs: (n_motors, n_harmonics, n_audio)

        energy = compute_vp_energy(audio_tensor, freqs, window_len=window_len, hop_len=hop_len)
        return -energy

    return objective


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    offset: float  # optimal global time shift (seconds)
    rate: float  # optimal post-jump rate factor
    energy: float  # VP energy at optimum
    success: bool  # optimizer convergence flag
    nit: int  # number of iterations
    message: str  # optimizer status message


def align_michaels_recording(
    ts_raw: np.ndarray,
    motor_rps: np.ndarray,  # (n_motors, M)  — Hz
    wav: np.ndarray,  # (n_channels, n_samples)  or  (n_samples,)
    sr: int = 16000,
    *,
    # VP parameters
    n_harmonics: int = 15,
    window_len: int = 2048,
    hop_len: int = 1024,
    # Initial guess
    offset_0: float = 0.15,
    rate_0: float = 1.0,
    # Bounds
    offset_bounds: tuple[float, float] = (-0.5, 0.5),
    rate_bounds: tuple[float, float] = (0.9, 1.1),
) -> AlignmentResult:
    """Find the (offset, rate) that best aligns CSV RPS with audio harmonics.

    Optimises over the **entire recording** using VP-transform energy as the
    objective, so the returned parameters are globally optimal.

    Parameters
    ----------
    ts_raw : (M,) float64
        Raw CSV timestamps from ``_load_raw``.
    motor_rps : (n_motors, M) float32
        Motor speeds in revolutions per second (Hz).
    wav : (n_channels, n_samples) or (n_samples,)
        Audio waveform (uses channel 0 if multichannel).
    sr : int
        Audio sample rate.
    n_harmonics : int
        Number of harmonics per motor.
    window_len, hop_len : int
        VP‑transform frame parameters.
    offset_0, rate_0 : float
        Initial guess.
    offset_bounds, rate_bounds : tuple
        (lo, hi) bounds for L‑BFGS‑B.

    Returns
    -------
    AlignmentResult
    """
    audio_full = wav[0].astype(np.float32) if wav.ndim > 1 else wav.astype(np.float32)

    jump_idx = _find_jump(ts_raw)

    objective = _make_objective(
        ts_raw=ts_raw,
        jump_idx=jump_idx,
        motor_rps=motor_rps,
        audio_full=audio_full,
        sr=sr,
        window_len=window_len,
        hop_len=hop_len,
        n_harmonics=n_harmonics,
    )

    x0 = np.array([offset_0, rate_0], dtype=np.float64)
    bounds = [offset_bounds, rate_bounds]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-8, "gtol": 1e-6},
    )

    return AlignmentResult(
        offset=float(result.x[0]),
        rate=float(result.x[1]),
        energy=float(-result.fun),
        success=bool(result.success),
        nit=int(result.nit),
        message=str(result.message),
    )
