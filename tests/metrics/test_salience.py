"""Tests for metrics.salience.SalienceBCEMetric — the validation-time
sibling of losses.salience.SalienceRPSBCELoss (same grid/derivation, one
sample at a time, returns a plain float)."""

from __future__ import annotations

import numpy as np
import tdseries as td

from losses.salience import SalienceRPSBCELoss, auto_pos_weight
from metrics.salience import SalienceBCEMetric


def _sample_pair(*, n_bins: int, n_grid: int, num_rotors: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n_bins, n_grid)).astype(np.float32)
    t_stft = n_grid + 3
    rps = (rng.uniform(30.0, 90.0, size=(num_rotors, t_stft))).astype(np.float32)

    pred = td.Frame({"salience": td.uniform(logits, 100, dims=("freq", "time"), t_start=0.0)})
    target = td.Frame({"rps": td.uniform(rps, 1000, dims=("rotor", "time"), t_start=0.0)})
    return pred, target


def test_salience_bce_metric_returns_finite_float():
    pred, target = _sample_pair(n_bins=60, n_grid=15)
    metric = SalienceBCEMetric(fmin=32.7, n_octaves=1, over_sample=5, blur_bins=1, pos_weight=2.0)
    value = metric(pred, target)
    assert isinstance(value, float)
    assert value >= 0.0


def test_salience_bce_metric_matches_batched_loss_for_one_sample():
    pred, target = _sample_pair(n_bins=60, n_grid=15, seed=1)
    metric = SalienceBCEMetric(fmin=32.7, n_octaves=1, over_sample=5, blur_bins=1, pos_weight=2.0)
    metric_value = metric(pred, target)

    # Same computation, but through the batched Frame-adapter loss (batch=1)
    # -- the metric and the loss must agree exactly for equivalent params.
    loss_fn = SalienceRPSBCELoss(fmin=32.7, n_octaves=1, over_sample=5, blur_bins=1, pos_weight=2.0)
    pred_b = td.Frame(
        {
            "salience": td.uniform(
                np.asarray(pred["salience"].data)[None], 100, dims=("batch", "freq", "time")
            )
        }
    )
    target_b = td.Frame(
        {
            "rps": td.uniform(
                np.asarray(target["rps"].data)[None], 1000, dims=("batch", "rotor", "time")
            )
        }
    )
    loss_value = float(loss_fn(pred_b, target_b).item())
    assert metric_value == loss_value


def test_salience_bce_metric_auto_pos_weight():
    metric = SalienceBCEMetric(
        fmin=32.7, n_octaves=1, over_sample=5, n_bins=100, blur_bins=1, pos_weight="auto"
    )
    assert metric.pos_weight == auto_pos_weight(100, num_rotors=4, blur_bins=1)


def test_salience_bce_metric_linear_output_grid():
    pred, target = _sample_pair(n_bins=360, n_grid=8)
    metric = SalienceBCEMetric(out_fmin=55.0, out_fmax=110.0, out_bins=360, blur_bins=2)
    value = metric(pred, target)
    assert np.isfinite(value)


def test_salience_bce_metric_spec_entries_are_batched_like_rps_metric():
    # Matches metrics.rps.RPSMetric's convention: requires_pred/requires_target
    # declare BATCHED dims (validate_config checks metrics against the batched
    # task/dataset spec) even though __call__ itself is unbatched per-sample.
    from tasks.spec import SeriesSpec

    metric = SalienceBCEMetric()
    pred_spec = metric.requires_pred.entries["salience"]
    target_spec = metric.requires_target.entries["rps"]
    assert isinstance(pred_spec, SeriesSpec)
    assert isinstance(target_spec, SeriesSpec)
    assert pred_spec.dims == ("batch", "freq", "time")
    assert target_spec.dims == ("batch", "rotor", "time")
