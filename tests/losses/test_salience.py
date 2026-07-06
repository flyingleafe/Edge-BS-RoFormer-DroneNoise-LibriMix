"""Tests for losses.salience (BCE-on-salience with pos_weight)."""

import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F

from losses.salience import SalienceBCELoss, SalienceRPSBCELoss, auto_pos_weight, salience_bce_loss
from models.multif0.utils import cqt_freq_grid, salience_target_from_resampled_rps


def test_auto_pos_weight_matches_hand_computation():
    # n_bins=100, 2 rotors, blur_bins=1 -> active = 2 * (2*1+1) = 6
    pw = auto_pos_weight(n_bins=100, num_rotors=2, blur_bins=1)
    assert pw == (100 - 6) / 6


def test_auto_pos_weight_guards_against_zero_active():
    pw = auto_pos_weight(n_bins=10, num_rotors=0, blur_bins=0)
    assert np.isfinite(pw)


def test_salience_bce_loss_matches_torch_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 3)
    target = torch.randint(0, 2, (2, 5, 3)).float()
    ours = salience_bce_loss(logits, target)
    reference = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert torch.allclose(ours, reference)


def test_salience_bce_loss_with_pos_weight_differs_from_unweighted():
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 3)
    target = torch.ones(2, 5, 3)  # all-positive target makes pos_weight matter
    unweighted = salience_bce_loss(logits, target)
    weighted = salience_bce_loss(logits, target, pos_weight=5.0)
    assert not torch.allclose(unweighted, weighted)


def test_salience_bce_loss_frame_adapter():
    torch.manual_seed(2)
    logits = torch.randn(6, 4).numpy()
    target_sal = (torch.rand(6, 4) > 0.5).float().numpy()

    pred = td.Frame({"salience": td.uniform(logits.astype(np.float32), 100, dims=("freq", "time"))})
    target = td.Frame(
        {"salience": td.uniform(target_sal.astype(np.float32), 100, dims=("freq", "time"))}
    )

    loss_fn = SalienceBCELoss(pos_weight=2.0)
    value = loss_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert value.item() >= 0.0
    assert "salience" in loss_fn.requires_pred.entries
    assert "salience" in loss_fn.requires_target.entries


# ─── SalienceRPSBCELoss: derives its target from target["rps"] ───────────────


def _make_rps_pred_target(*, n_bins: int, n_grid: int, batch: int = 2, num_rotors: int = 4):
    torch.manual_seed(3)
    logits_np = torch.randn(batch, n_bins, n_grid).numpy().astype(np.float32)
    # RPS in Hz on the STFT grid (a different T than n_grid, on purpose —
    # the loss must resample to match pred's actual frame count).
    t_stft = n_grid + 5
    rps_np = torch.rand(batch, num_rotors, t_stft).numpy().astype(np.float32) * 50.0 + 30.0

    pred = td.Frame(
        {"salience": td.uniform(logits_np, 100, dims=("batch", "freq", "time"), t_start=0.0)}
    )
    target = td.Frame(
        {"rps": td.uniform(rps_np, 1000, dims=("batch", "rotor", "time"), t_start=0.0)}
    )
    return pred, target, logits_np, rps_np


def test_salience_rps_bce_loss_derives_target_from_rps_matches_shared_function():
    pred, target, logits_np, rps_np = _make_rps_pred_target(n_bins=60, n_grid=17)

    loss_fn = SalienceRPSBCELoss(fmin=32.7, n_octaves=1, over_sample=5, blur_bins=1, pos_weight=2.0)
    value = loss_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert value.item() >= 0.0

    # Hand-compute the expected target via the exact same shared function the
    # loss delegates to, proving no math was duplicated/drifted.
    freqs = cqt_freq_grid(fmin=32.7, n_octaves=1, over_sample=5)
    rps_t = torch.from_numpy(rps_np)
    n_grid = logits_np.shape[-1]
    rps_grid = F.interpolate(rps_t, size=n_grid, mode="linear", align_corners=False)
    expected_target = salience_target_from_resampled_rps(rps_grid, freqs, blur_bins=1)
    expected_loss = salience_bce_loss(torch.from_numpy(logits_np), expected_target, pos_weight=2.0)
    assert torch.allclose(value, expected_loss)


def test_salience_rps_bce_loss_auto_pos_weight_matches_formula():
    pred, target, _logits_np, _rps_np = _make_rps_pred_target(n_bins=100, n_grid=10)
    loss_fn = SalienceRPSBCELoss(
        fmin=32.7, n_octaves=1, over_sample=5, n_bins=100, blur_bins=1, pos_weight="auto"
    )
    assert loss_fn.pos_weight == auto_pos_weight(100, num_rotors=4, blur_bins=1)


def test_salience_rps_bce_loss_linear_output_grid():
    pred, target, _logits_np, _rps_np = _make_rps_pred_target(n_bins=360, n_grid=12)
    loss_fn = SalienceRPSBCELoss(out_fmin=55.0, out_fmax=110.0, out_bins=360, blur_bins=2)
    value = loss_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert torch.isfinite(value)


def test_salience_rps_bce_loss_requires_out_grid_params_together():
    import pytest

    with pytest.raises(ValueError):
        SalienceRPSBCELoss(out_fmin=55.0, out_fmax=110.0)  # missing out_bins


def test_salience_rps_bce_loss_spec_entries():
    loss_fn = SalienceRPSBCELoss()
    assert "salience" in loss_fn.requires_pred.entries
    assert "rps" in loss_fn.requires_target.entries
