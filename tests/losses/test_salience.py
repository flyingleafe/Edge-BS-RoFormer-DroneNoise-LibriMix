"""Tests for losses.salience (BCE-on-salience with pos_weight)."""

import numpy as np
import tdseries as td
import torch

from losses.salience import SalienceBCELoss, auto_pos_weight, salience_bce_loss


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
