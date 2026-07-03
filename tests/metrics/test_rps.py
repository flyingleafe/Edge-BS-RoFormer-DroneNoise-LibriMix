"""Tests for metrics.rps (PIT-aware RPS statistics, the ONE implementation)."""

import numpy as np
import tdseries as td

from metrics.rps import RPSMetric, rps_mae_clip, rps_mae_frame, rps_mse, rps_r2, rps_rmse


def _rps_frame(entry: str, rps: np.ndarray) -> td.Frame:
    return td.Frame({entry: td.uniform(rps.astype(np.float32), 100, dims=("rotor", "time"))})


def test_rps_mse_zero_on_identical_signal():
    rng = np.random.default_rng(0)
    x = rng.random((4, 30))
    assert rps_mse(x, x) == 0.0
    assert rps_rmse(x, x) == 0.0
    assert rps_mae_frame(x, x) == 0.0
    assert rps_mae_clip(x, x) == 0.0


def test_pit_alignment_finds_permuted_rotors():
    rng = np.random.default_rng(1)
    target = rng.random((4, 40))
    perm = [3, 1, 0, 2]
    pred = target[perm]

    # Without PIT alignment, comparing in raw order is wrong (nonzero error).
    assert rps_mse(pred, target, pit=False) > 0.1
    # With PIT alignment, the permutation is recovered exactly.
    assert rps_mse(pred, target, pit=True) < 1e-8
    assert rps_mae_frame(pred, target, pit=True) < 1e-6
    assert rps_mae_clip(pred, target, pit=True) < 1e-6


def test_rps_r2_perfect_prediction_is_one():
    rng = np.random.default_rng(2)
    target = rng.random((3, 50)) * 100
    assert abs(rps_r2(target, target) - 1.0) < 1e-6


def test_rps_r2_worse_than_mean_baseline_is_negative():
    rng = np.random.default_rng(3)
    target = rng.random((2, 50)) * 100
    pred = target[::-1] * 3.0 + 500  # deliberately bad, unaligned-ish prediction
    r2 = rps_r2(pred, target, pit=False)
    assert r2 < 0.0


def test_rps_r2_nan_when_target_constant():
    target = np.ones((2, 10)) * 5.0
    pred = np.ones((2, 10)) * 5.0
    assert np.isnan(rps_r2(pred, target))


def test_rmse_is_sqrt_of_mse():
    rng = np.random.default_rng(4)
    pred = rng.random((4, 20))
    target = rng.random((4, 20))
    assert abs(rps_rmse(pred, target) ** 2 - rps_mse(pred, target)) < 1e-9


def test_rps_metric_frame_adapter():
    rng = np.random.default_rng(5)
    target_np = rng.random((4, 25))
    perm = [2, 0, 3, 1]
    pred_np = target_np[perm]

    pred = _rps_frame("rps_pred", pred_np)
    target = _rps_frame("rps", target_np)

    mse_metric = RPSMetric("mse")
    r2_metric = RPSMetric("r2")

    assert mse_metric(pred, target) < 1e-8
    assert r2_metric(pred, target) > 0.99
    assert "rps_pred" in mse_metric.requires_pred.entries
    assert "rps" in mse_metric.requires_target.entries


def test_rps_metric_rejects_unknown_stat():
    try:
        RPSMetric("not_a_stat")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown stat")
