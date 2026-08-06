"""The tachometer label-noise transform and the arm dataset that switches it on.

These pin the *signature* of the device the Phase-7 experiment attributes its
verdict to (docs/experiments/generator-label-sensitivity.md): the quantization
step, the refresh rate, the constant scale, and the fact that only the label —
never the target audio — moves between arms.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_processing.frame_datasets import StaticCombGenDataset
from data_processing.rps_corruption import (
    DREGON_TACH_REFRESH_HZ,
    DREGON_TACH_STEP,
    presmooth_track,
    tachometer_corrupt,
)

SR = 16000


def _ramp(duration_s: float = 1.0, lo: float = 78.0, hi: float = 84.0) -> np.ndarray:
    n = int(duration_s * SR)
    return np.linspace(lo, hi, n)


def test_output_lives_on_the_quantization_lattice():
    out = tachometer_corrupt(_ramp(), SR)
    residual = out / DREGON_TACH_STEP
    assert np.allclose(residual, np.rint(residual), atol=1e-9)


def test_refresh_rate_sets_the_number_of_plateaus():
    out = tachometer_corrupt(_ramp(duration_s=2.0), SR, step=0.0)
    # step=0 keeps the hold but drops quantization, so every interval boundary
    # of a monotone ramp is a distinct transition.
    transitions = int(np.count_nonzero(np.abs(np.diff(out)) > 1e-12))
    assert transitions == pytest.approx(2.0 * DREGON_TACH_REFRESH_HZ, abs=2)


def test_plateaus_are_flat_and_start_on_the_refresh_grid():
    out = tachometer_corrupt(_ramp(), SR, step=0.0)
    edges = np.flatnonzero(np.abs(np.diff(out)) > 1e-12) + 1
    times = edges / SR
    # each edge sits within one sample of an integer multiple of the interval
    on_grid = np.abs(times * DREGON_TACH_REFRESH_HZ - np.rint(times * DREGON_TACH_REFRESH_HZ))
    assert np.all(on_grid < 2.0 * DREGON_TACH_REFRESH_HZ / SR)


def test_constant_scale_is_applied_before_the_device():
    const = np.full(SR, 80.0)
    out = tachometer_corrupt(const, SR, scale=0.99458)
    # a constant survives the interval average untouched, so only the lattice
    # separates the output from scale * truth
    assert np.all(out == out[0])
    assert abs(float(out[0]) - 80.0 * 0.99458) <= DREGON_TACH_STEP / 2 + 1e-9


def test_constant_input_on_the_lattice_is_returned_exactly():
    value = 40.0 * DREGON_TACH_STEP
    out = tachometer_corrupt(np.full(SR, value), SR)
    assert np.allclose(out, value)


def test_error_magnitude_is_the_quantization_floor_for_a_slow_input():
    slow = 80.0 + 0.2 * np.sin(2 * np.pi * 0.5 * np.arange(SR) / SR)
    out = tachometer_corrupt(slow, SR)
    rms = float(np.sqrt(np.mean((out - slow) ** 2)))
    assert rms == pytest.approx(DREGON_TACH_STEP / np.sqrt(12), rel=0.25)


def test_disabling_refresh_leaves_a_pure_quantizer():
    slow = _ramp()
    out = tachometer_corrupt(slow, SR, refresh_hz=0.0)
    assert np.allclose(out, np.rint(slow / DREGON_TACH_STEP) * DREGON_TACH_STEP)


def test_leading_axes_are_preserved():
    stack = np.stack([_ramp(), _ramp(lo=60.0, hi=62.0)])
    out = tachometer_corrupt(stack, SR)
    assert out.shape == stack.shape
    assert np.allclose(out[0], tachometer_corrupt(stack[0], SR))


def test_presmooth_removes_more_staircase_than_signal():
    truth = 80.0 + 1.5 * np.sin(2 * np.pi * 0.7 * np.arange(4 * SR) / (4 * SR))
    tach = tachometer_corrupt(truth, SR)
    smoothed = presmooth_track(tach, SR, cut_hz=5.0)
    err_tach = float(np.sqrt(np.mean((tach - truth) ** 2)))
    err_smooth = float(np.sqrt(np.mean((smoothed - truth) ** 2)))
    distortion = float(np.sqrt(np.mean((presmooth_track(truth, SR, cut_hz=5.0) - truth) ** 2)))
    assert err_smooth < err_tach
    assert distortion < err_smooth


def test_presmooth_zero_cutoff_is_the_identity():
    x = _ramp()
    assert np.allclose(presmooth_track(x, SR, cut_hz=0.0), x)


# --- the arm dataset -------------------------------------------------------


def _ds(mode: str, **kw):
    return StaticCombGenDataset(n_samples=4, split="train", label_mode=mode, seed=3, **kw)


def test_arms_share_the_target_and_differ_only_in_the_label():
    exact, tach = _ds("exact"), _ds("tach")
    a0 = np.asarray(exact[0]["audio"].data)
    a1 = np.asarray(tach[0]["audio"].data)
    assert np.allclose(a0, a1), "the target must not depend on the label arm"
    r0 = np.asarray(exact[0]["rps"].data)
    r1 = np.asarray(tach[0]["rps"].data)
    assert not np.allclose(r0, r1)


def test_exact_arm_label_is_the_truth():
    ds = _ds("exact")
    assert np.allclose(np.asarray(ds[0]["rps"].data)[0], ds.true_rps(0), atol=1e-4)


def test_scale_arm_is_a_pure_gain():
    ds = _ds("scale", label_scale=0.99458)
    assert np.allclose(np.asarray(ds[0]["rps"].data)[0], ds.true_rps(0) * 0.99458, atol=1e-4)


def test_samples_are_deterministic_across_instances():
    a = np.asarray(_ds("tach")[1]["audio"].data)
    b = np.asarray(_ds("tach")[1]["audio"].data)
    assert np.array_equal(a, b)


def test_frame_shapes_match_the_noise_generation_contract():
    frame = _ds("tach")[0]
    assert np.asarray(frame["audio"].data).shape == (1, SR)
    assert np.asarray(frame["rps"].data).shape == (1, SR)
    assert np.asarray(frame["mic_pos"].data).shape == (1, 3)
    assert np.asarray(frame["rotor_pos"].data).shape == (1, 3)


def test_target_level_is_clip_independent():
    ds = _ds("exact")
    rms = [float(np.sqrt(np.mean(np.asarray(ds[i]["audio"].data) ** 2))) for i in range(4)]
    assert np.std(rms) / np.mean(rms) < 0.05


def test_unknown_label_mode_is_rejected():
    with pytest.raises(ValueError, match="label_mode"):
        StaticCombGenDataset(n_samples=1, label_mode="nope")
