"""The two tacholess order-tracking baselines (ridge extraction and IAVKF).

Both are published as single-shaft methods; here they carry four shafts, so the
tests pin down what a four-shaft adaptation can honestly promise: the interface
contract (shape, dtype, finiteness, search band), recovery of the STRONGEST
comb on a synthetic four-rotor mixture, the fact that the Vold-Kalman
refinement does not make its own ridge seed worse, and survival on a degenerate
(all-zero) waveform. The weaker three rotors are left unpinned on purpose — the
greedy peel-off loses them, and that is the baseline's known limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.classical_rps.predictors import (
    CLASSICAL_TRACKERS,
    HOP_LENGTH,
    N_BLADES,
    N_ROTORS,
    RPS_MAX,
    RPS_MIN,
    SR,
)
from experiments.classical_rps.tacholess import iavkf_tracker, ridge_tracker

#: Rotor rates in rev/s, strongest first, paired with a relative comb amplitude.
#: None of them is a 2:1 multiple of another inside the search band, so the
#: strongest comb cannot be confused with its own second harmonic.
FOUR_ROTORS: tuple[tuple[float, float], ...] = (
    (88.0, 1.00),
    (78.0, 0.60),
    (70.0, 0.45),
    (62.0, 0.35),
)


def _four_comb(
    duration_s: float = 4.0, n_harm: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Four slowly drifting blade-pass combs plus light noise.

    Returns the waveform and the ground-truth rates on the STFT frame grid,
    row 0 being the strongest rotor.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * SR)) / SR
    x = 0.01 * rng.standard_normal(t.size)
    rates = []
    for i, (rate_0, amp) in enumerate(FOUR_ROTORS):
        rate = rate_0 + 1.5 * np.sin(2 * np.pi * 0.15 * t + i)
        rates.append(rate)
        phase = 2 * np.pi * np.cumsum(N_BLADES * rate) / SR
        for k in range(1, n_harm + 1):
            x += amp * (1.0 / k) * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
    audio = (x / np.max(np.abs(x))).astype(np.float32)

    frame_times = np.arange(len(audio) // HOP_LENGTH + 1) * HOP_LENGTH / SR
    target = np.stack([np.interp(frame_times, t, r) for r in rates])
    return audio, target


def _median_abs_err(pred_row: np.ndarray, target_row: np.ndarray) -> float:
    return float(np.median(np.abs(pred_row - target_row)))


@pytest.mark.parametrize("tracker", [ridge_tracker, iavkf_tracker])
def test_a_tacholess_tracker_returns_a_full_rotor_track_inside_the_search_band(tracker):
    audio, _ = _four_comb()
    pred = tracker(audio)

    assert pred.shape == (N_ROTORS, len(audio) // HOP_LENGTH + 1)
    assert pred.dtype.kind == "f"
    assert np.isfinite(pred).all()
    assert (pred >= RPS_MIN).all()
    assert (pred <= RPS_MAX).all()


def test_ridge_recovers_the_strongest_rotor():
    audio, target = _four_comb()
    pred = ridge_tracker(audio)
    assert _median_abs_err(pred[0], target[0]) < 1.0


def test_iavkf_does_not_degrade_its_own_ridge_seed():
    audio, target = _four_comb()
    seed = ridge_tracker(audio)
    refined = iavkf_tracker(audio)

    seed_err = _median_abs_err(seed[0], target[0])
    refined_err = _median_abs_err(refined[0], target[0])
    # A tolerance of 0.05 rev/s absorbs the interpolation of the refined
    # trajectory back onto the frame grid; the refinement should in fact win.
    assert refined_err <= seed_err + 0.05


@pytest.mark.parametrize("tracker", [ridge_tracker, iavkf_tracker])
def test_a_degenerate_waveform_does_not_raise(tracker):
    audio = np.zeros(2 * SR, dtype=np.float32)
    pred = tracker(audio)

    assert pred.shape == (N_ROTORS, len(audio) // HOP_LENGTH + 1)
    assert np.isfinite(pred).all()


def test_both_methods_are_registered():
    for name in ("ridge", "iavkf"):
        assert name in CLASSICAL_TRACKERS
        pred = CLASSICAL_TRACKERS[name](np.zeros(SR, dtype=np.float32))
        assert pred.shape == (N_ROTORS, SR // HOP_LENGTH + 1)
