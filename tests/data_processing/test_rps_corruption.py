"""Tests for the conditional-refiner corruption sampler
(``data_processing.rps_corruption``): determinism, zero-span preservation,
corruption statistics, and the aligned-GT (swap) contract."""

from __future__ import annotations

import numpy as np
import pytest

from data_processing.online_mixing import make_rng
from data_processing.rps_corruption import RPSCorruption, corrupt_rps

FRAME_RATE = 16000.0 / 512.0


def _clean_track(n_frames: int = 251, zero_head: int = 40) -> np.ndarray:
    """A plausible clean track: ground span, then distinct cruise plateaus."""
    rps = np.zeros((4, n_frames), dtype=np.float32)
    t = np.arange(n_frames - zero_head) / FRAME_RATE
    for r in range(4):
        rps[r, zero_head:] = 78.0 + 2.0 * r + 0.5 * np.sin(2 * np.pi * 0.3 * t + r)
    return rps


def test_deterministic_per_seed_and_id():
    rps = _clean_track()
    c = RPSCorruption(seed=123)
    a_cond, a_gt = c(rps, 42)
    b_cond, b_gt = c(rps, 42)
    assert np.array_equal(a_cond, b_cond)
    assert np.array_equal(a_gt, b_gt)
    # Different id or seed -> different corruption (overwhelmingly likely).
    d_cond, _ = c(rps, 43)
    assert not np.array_equal(a_cond, d_cond)
    e_cond, _ = RPSCorruption(seed=124)(rps, 42)
    assert not np.array_equal(a_cond, e_cond)


def test_zero_spans_stay_zero_and_nonnegative():
    rps = _clean_track(zero_head=60)
    c = RPSCorruption(seed=7)
    for sid in range(50):
        cond, gt = c(rps, sid)
        assert np.all(cond[np.abs(gt) <= 1e-6] == 0.0)
        assert np.all(cond >= 0.0)
        assert cond.dtype == np.float32 and gt.dtype == np.float32
        assert cond.shape == rps.shape and gt.shape == rps.shape


def test_gt_aligned_is_row_permutation_of_input():
    rps = _clean_track()
    c = RPSCorruption(seed=99)
    n_swapped = 0
    for sid in range(400):
        _, gt = c(rps, sid)
        # gt must always be a row permutation of the input GT.
        matched = [any(np.array_equal(gt[i], rps[j]) for j in range(4)) for i in range(4)]
        assert all(matched)
        if not np.array_equal(gt, rps):
            n_swapped += 1
    # Swap branch fires at pair_event_prob * 0.5 = 0.075; 400 draws ->
    # expect ~30, allow a generous binomial band.
    assert 10 <= n_swapped <= 60, n_swapped


def test_corruption_statistics_in_range():
    rps = _clean_track(zero_head=0)  # no zero spans: pure-noise statistics
    c = RPSCorruption(seed=5)
    devs = []
    step_sizes = []
    n_offset_like = 0
    for sid in range(200):
        cond, gt = c(rps, sid)
        d = cond.astype(np.float64) - gt.astype(np.float64)
        devs.append(d)
        step_sizes.append(np.abs(np.diff(cond.astype(np.float64), axis=-1)))
        if np.any(np.abs(d.mean(axis=-1)) > 1.0):
            n_offset_like += 1
    dev = np.concatenate([d.ravel() for d in devs])
    # Overall deviation scale: OU sigma U(0.1,1.5) + offsets U(-2.5,2.5)@0.7
    # -> RMS well within [0.3, 3.5].
    assert 0.3 < float(np.sqrt(np.mean(dev**2))) < 3.5
    # Smoothness: OU with tau >= 0.5 s at 31.25 Hz moves far less per frame
    # than its stationary sigma.
    mean_step = float(np.mean(np.concatenate([s.ravel() for s in step_sizes])))
    assert mean_step < 0.5, mean_step
    # Constant offsets (|mean dev| > 1 on some rotor) fire often (p=0.7/rotor).
    assert n_offset_like > 100, n_offset_like


def test_twin_capture_present():
    """Some draws must set one rotor's conditioning onto another's track."""
    rps = _clean_track(zero_head=0)
    c = RPSCorruption(seed=11)
    n_twin = 0
    for sid in range(400):
        cond, gt = c(rps, sid)
        d = cond.astype(np.float64) - gt.astype(np.float64)
        # Twin capture: one row's deviation tracks the difference to ANOTHER
        # rotor's plateau (~2 rev/s apart per rotor index step) plus noise —
        # detect as a row whose mean |dev| exceeds the offset cap.
        if np.any(np.abs(d.mean(axis=-1)) > 3.0):
            n_twin += 1
    assert n_twin > 0


def test_corrupt_rps_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(R, F\)"):
        corrupt_rps(np.zeros(10, dtype=np.float32), make_rng(0, 0))


def test_from_config_none_and_frame_rate_default():
    assert RPSCorruption.from_config(None) is None
    assert RPSCorruption.from_config({}) is None
    c = RPSCorruption.from_config({"seed": 3}, frame_rate_hz=31.25)
    assert c is not None
    assert c.seed == 3 and c.params["frame_rate_hz"] == 31.25
    # Config pins win over the dataset-provided rate.
    c2 = RPSCorruption.from_config({"seed": 3, "frame_rate_hz": 10.0}, frame_rate_hz=31.25)
    assert c2 is not None and c2.params["frame_rate_hz"] == 10.0
