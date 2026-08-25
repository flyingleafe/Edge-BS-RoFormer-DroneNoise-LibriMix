"""Two tacholess order-tracking baselines, adapted to four shafts.

Both methods come from the tacholess (encoderless) order-tracking literature,
where the shaft speed is recovered from the vibration/acoustic signal itself.
Both are **single-shaft** methods as published: they assume one rotating source
and extract one instantaneous-frequency (IF) trace. A quadcopter carries four
independent shafts, so each method is extended here with the same greedy
peel-off convention the restored classical baselines of
:mod:`experiments.classical_rps.predictors` use — find one trajectory, suppress
its harmonic comb, search again — until ``N_ROTORS`` trajectories exist.

Methods implemented
-------------------
1. ``ridge_tracker``  - classical tacholess IF ridge extraction. The
   log-magnitude STFT is read on the blade-pass frequency (BPF) of every
   candidate shaft rate; the best rate trajectory is the max-sum Viterbi path
   over that emission surface, with an L1 penalty on the rate change between
   adjacent frames. Dividing the tracked order back by the blade count gives
   the shaft rate. Single-shaft in the literature; the greedy comb peel-off
   is the four-shaft adaptation.
2. ``iavkf_tracker`` - iterative adaptive Vold-Kalman filtering (Jiang, Chen &
   Wang, IEEE TII 2024). The published scheme seeds a Vold-Kalman order
   extraction with an external wide-capture IF estimate and iterates, adapting
   the filter bandwidth from the extracted envelope. Here the seed is
   ``ridge_tracker``'s four trajectories and the extraction is the project's
   coupled multi-shaft Vold-Kalman tracker (:func:`tracking.vk_track`) with
   ``bw_adapt=True``, which is that bandwidth adaptation. The four-shaft
   adaptation is therefore in the extraction itself: one coupled solve carries
   all four combs, instead of four independent single-shaft filters.

Both obey the interface of the other classical baselines::

    predictions = method(audio, sr=16000)  -> ndarray, shape (4, n_frames)

with ``audio`` a single-channel waveform and the output in rev/s on the
2048/512 STFT frame grid at 16 kHz. Row order is the greedy discovery order —
strongest comb first — the same "descending by evidence" convention as
``nmf_tracker``. The protocol evaluation in
:mod:`experiments.classical_rps.valid_eval` matches rows per frame with the
Hungarian assignment, so the row order does not change any reported number.
"""

from __future__ import annotations

import warnings

import numpy as np

from experiments.classical_rps.predictors import (
    HARM_BW_BINS,
    HOP_LENGTH,
    N_BLADES,
    N_FFT,
    N_ROTORS,
    RPS_MAX,
    RPS_MIN,
    SR,
    _frame_spectra,
)
from tracking.pipelines import comb_teeth, surface_contrast, viterbi_lattice
from tracking.vk_tracking import VKConfig, vk_track

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of candidate shaft rates spanning [RPS_MIN, RPS_MAX]. 401 points is
#: a 0.25 rev/s step, i.e. a 0.5 Hz step on the blade-pass frequency — about
#: 1/16 of an FFT bin (7.8 Hz), so the emission surface is limited by the STFT
#: resolution and not by the grid.
RIDGE_N_GRID = 401

#: Viterbi transition weight, as a multiple of the surface contrast (the median
#: over frames of ``max - median`` node score). ``gamma = MULT * contrast``
#: with the lattice grid in rev/s, so a 1 rev/s jump between adjacent frames
#: costs ``MULT`` times the typical peak-to-floor emission difference — a
#: modest one. This is the same convention and the same value as
#: ``tracking.pipelines.VIT_GAMMA_MULT``.
RIDGE_GAMMA_MULT = 0.3

#: Floor value written into the log spectrogram when a comb is peeled off.
_LOG_FLOOR_EPS = 1e-10

#: IAVKF cost control. ``k_max=25`` covers the comb to ~3.7 kHz at the top of
#: the rate range; three outer rounds are enough for a ridge seed that is
#: already inside the capture basin (the seed error is a fraction of a rev/s,
#: far below the per-round ``max_step`` clip).
IAVKF_K_MAX = 25
IAVKF_N_OUTER = 3

#: Incremented every time :func:`iavkf_tracker` falls back to its ridge seed.
#: A module-level counter (rather than an exception) keeps a failed clip from
#: killing a whole gridrun sweep while still leaving the failure countable.
IAVKF_FALLBACK_COUNT = 0


# ---------------------------------------------------------------------------
# 1. Tacholess IF ridge extraction
# ---------------------------------------------------------------------------


def _log_spectrogram(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-magnitude STFT as ``(n_freqs, n_frames)``, plus the bin width in Hz."""
    specs, _ = _frame_spectra(audio)  # (n_frames, n_freqs)
    lm = np.log(np.maximum(specs.T.astype(np.float64), _LOG_FLOOR_EPS))
    return lm, SR / N_FFT


def _ridge_surface(lm: np.ndarray, bin_hz: float, rate_grid: np.ndarray) -> np.ndarray:
    """Emission surface ``(n_frames, n_rates)``: the log magnitude on each BPF.

    Candidate shaft rate ``r`` puts the blade-pass frequency at
    ``N_BLADES * r``; the value there is linearly interpolated between the two
    neighbouring FFT bins by :func:`tracking.pipelines.comb_teeth`.
    """
    teeth = N_BLADES * np.asarray(rate_grid, dtype=np.float64)
    values = comb_teeth(lm, bin_hz, teeth, f_min=0.0, f_max=SR / 2.0)  # (n_rates, n_frames)
    return np.nan_to_num(values, nan=float(lm.min())).T


def _suppress_comb(
    lm: np.ndarray,
    bin_hz: float,
    rate_track: np.ndarray,
    floor: float,
    half_width: int = HARM_BW_BINS,
) -> np.ndarray:
    """Peel one comb off the log spectrogram: every ``k * BPF`` up to Nyquist.

    ``rate_track`` is one rate per frame, so the suppressed bins follow the
    trajectory. Bins within ``half_width`` of a harmonic are set to ``floor``.
    """
    lm = lm.copy()
    n_freqs, n_frames = lm.shape
    nyquist = (n_freqs - 1) * bin_hz
    bpf = N_BLADES * np.asarray(rate_track, dtype=np.float64)
    lowest = float(np.min(bpf))
    if lowest <= 0.0 or not np.isfinite(lowest):
        return lm
    cols = np.arange(n_frames)
    for k in range(1, int(np.floor(nyquist / lowest)) + 1):
        centers = np.round(k * bpf / bin_hz).astype(int)
        for offset in range(-half_width, half_width + 1):
            idx = centers + offset
            ok = (idx >= 0) & (idx < n_freqs)
            lm[idx[ok], cols[ok]] = floor
    return lm


def ridge_tracker(audio: np.ndarray, sr: int = SR, n_rotors: int = N_ROTORS) -> np.ndarray:
    """Tacholess IF ridge extraction, greedily extended to four shafts.

    The textbook single-shaft method tracks the ridge of one dominant order on
    a time-frequency representation and divides by that order to get the shaft
    speed. Here the tracked order is the blade-pass frequency (order
    ``N_BLADES``), the ridge is the max-sum Viterbi path over the log-magnitude
    surface with an L1 rate-change penalty, and the four-shaft adaptation is
    the greedy peel-off: after each ridge, the whole harmonic comb of the found
    trajectory is written down to the spectrogram floor and the search runs
    again.

    Parameters
    ----------
    audio : ndarray, shape (n_samples,)
    sr : int, default 16000
    n_rotors : int, default 4

    Returns
    -------
    preds : ndarray, shape (n_rotors, n_frames), float32
        Shaft rates in rev/s, rows in greedy discovery order.
    """
    lm, bin_hz = _log_spectrogram(audio)
    floor = float(lm.min())
    rate_grid = np.linspace(RPS_MIN, RPS_MAX, RIDGE_N_GRID)

    preds = np.zeros((n_rotors, lm.shape[1]), dtype=np.float32)
    work = lm
    for r in range(n_rotors):
        surface = _ridge_surface(work, bin_hz, rate_grid)
        gamma = RIDGE_GAMMA_MULT * surface_contrast(surface)
        track = viterbi_lattice(surface, rate_grid, max(gamma, 0.0))
        preds[r] = track.astype(np.float32)
        work = _suppress_comb(work, bin_hz, track, floor)
    return preds


# ---------------------------------------------------------------------------
# 2. Iterative adaptive Vold-Kalman filtering (IAVKF)
# ---------------------------------------------------------------------------


def iavkf_tracker(audio: np.ndarray, sr: int = SR, n_rotors: int = N_ROTORS) -> np.ndarray:
    """Iterative adaptive Vold-Kalman filtering, seeded by the ridge tracker.

    Jiang, Chen & Wang (IEEE TII 2024) run a Vold-Kalman order extraction from
    an external wide-capture IF estimate and iterate, letting the filter
    bandwidth adapt to the extracted envelope (their eqs 25-27) so the passband
    narrows as the estimate locks. Their formulation is single-shaft. The
    adaptation to four shafts is twofold: the seed is the four-trajectory
    greedy output of :func:`ridge_tracker`, and the extraction is the coupled
    multi-shaft Vold-Kalman solve of :func:`tracking.vk_track` — one system
    carrying all four combs, so near-degenerate line pairs are attributed
    instead of collapsing onto their mean — with ``bw_adapt=True``, which is
    exactly the paper's bandwidth adaptation.

    If the solve raises or returns non-finite values, the ridge seed is
    returned unchanged and :data:`IAVKF_FALLBACK_COUNT` is incremented: a
    baseline must not kill a sweep.

    Parameters
    ----------
    audio : ndarray, shape (n_samples,)
    sr : int, default 16000
    n_rotors : int, default 4

    Returns
    -------
    preds : ndarray, shape (n_rotors, n_frames), float32
        Shaft rates in rev/s, clipped to ``[RPS_MIN, RPS_MAX]`` as the other
        classical baselines are, rows in the ridge seed's order.
    """
    global IAVKF_FALLBACK_COUNT

    seed = ridge_tracker(audio, sr, n_rotors)
    n_frames = seed.shape[1]
    frame_times = np.arange(n_frames, dtype=np.float64) * HOP_LENGTH / SR
    cfg = VKConfig(
        fs=float(SR),
        k_max=IAVKF_K_MAX,
        n_outer=IAVKF_N_OUTER,
        bw_adapt=True,
    )

    try:
        result = vk_track(
            np.asarray(audio, dtype=np.float64).reshape(-1),
            seed.astype(np.float64),
            frame_times,
            cfg,
        )
        refined = np.asarray(result.r_refined, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - a baseline must never kill the sweep
        IAVKF_FALLBACK_COUNT += 1
        warnings.warn(
            f"iavkf_tracker: vk_track failed ({exc!r}); using the ridge seed", stacklevel=2
        )
        return seed

    if refined.shape != seed.shape or not np.isfinite(refined).all():
        IAVKF_FALLBACK_COUNT += 1
        warnings.warn(
            "iavkf_tracker: vk_track returned a bad trajectory "
            f"(shape {refined.shape}, finite={bool(np.isfinite(refined).all())}); "
            "using the ridge seed",
            stacklevel=2,
        )
        return seed

    return np.clip(refined, RPS_MIN, RPS_MAX).astype(np.float32)
