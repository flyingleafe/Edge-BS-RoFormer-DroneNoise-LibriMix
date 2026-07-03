"""Tests for losses.pit (pairwise_mse, pit_mse_loss, segmented_pit_mse, Frame adapters)."""

import numpy as np
import tdseries as td
import torch

from losses.pit import (
    PITMSELoss,
    SegmentedPITMSELoss,
    pairwise_mse,
    pit_mse_loss,
    segmented_pit_mse,
)

SR = 16000


def _rps_frame(entry: str, rps: np.ndarray) -> td.Frame:
    return td.Frame({entry: td.uniform(rps.astype(np.float32), 100, dims=("rotor", "time"))})


def test_pairwise_mse_diagonal_zero_for_matched_rotors():
    est = torch.zeros(1, 3, 10)
    est[0, 0] = 1.0
    est[0, 1] = 2.0
    est[0, 2] = 3.0
    target = est.clone()
    pw = pairwise_mse(est, target)
    assert pw.shape == (1, 3, 3)
    assert torch.allclose(torch.diagonal(pw[0]), torch.zeros(3), atol=1e-6)


def test_pit_mse_loss_finds_permuted_match():
    torch.manual_seed(0)
    b, k, t = 2, 4, 50
    target = torch.rand(b, k, t)
    # est is target with rotors permuted (a fixed non-identity permutation)
    perm = [2, 0, 3, 1]
    est = target[:, perm, :]

    loss, best_idx = pit_mse_loss(est, target, return_indices=True)
    assert loss.item() < 1e-6  # perfect match under the right permutation
    assert best_idx.shape == (b,)

    # Un-permuted (standard) MSE should be much larger for a random permutation.
    std_loss = torch.nn.functional.mse_loss(est, target)
    assert std_loss.item() > loss.item()


def test_pit_mse_loss_identity_when_already_aligned():
    torch.manual_seed(1)
    target = torch.rand(3, 4, 20)
    loss = pit_mse_loss(target, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() < 1e-6


def test_segmented_pit_mse_matches_global_when_mask_none():
    torch.manual_seed(2)
    pred = torch.rand(1, 3, 30)
    target = pred.clone()
    loss = segmented_pit_mse(pred, target, merge_mask=None)
    assert loss.item() < 1e-6


def test_segmented_pit_mse_independent_segments():
    # Two segments separated by one merge frame; rotor identity swaps between them.
    pred = torch.zeros(1, 2, 7)
    pred[0, 0, :3] = 1.0
    pred[0, 1, :3] = 2.0
    pred[0, 0, 4:] = 2.0
    pred[0, 1, 4:] = 1.0

    target = torch.zeros(1, 2, 7)
    target[0, 0, :3] = 1.0
    target[0, 1, :3] = 2.0
    target[0, 0, 4:] = 2.0
    target[0, 1, 4:] = 1.0

    merge_mask = torch.zeros(1, 7, dtype=torch.bool)
    merge_mask[0, 3] = True

    loss = segmented_pit_mse(pred, target, merge_mask)
    assert loss.item() < 1e-6


def test_pit_mse_loss_frame_adapter():
    torch.manual_seed(3)
    target_np = np.random.default_rng(3).random((4, 20)).astype(np.float32)
    perm = [1, 2, 3, 0]
    pred_np = target_np[perm]

    pred = _rps_frame("rps_pred", pred_np)
    target = _rps_frame("rps", target_np)

    loss_fn = PITMSELoss()
    value = loss_fn(pred, target)
    assert isinstance(value, torch.Tensor)
    assert value.item() < 1e-6
    assert "rps_pred" in loss_fn.requires_pred.entries
    assert "rps" in loss_fn.requires_target.entries


def test_segmented_pit_mse_loss_frame_adapter_without_merge_mask():
    rng = np.random.default_rng(4)
    rps = rng.random((3, 15)).astype(np.float32)
    pred = _rps_frame("rps_pred", rps)
    target = _rps_frame("rps", rps)

    loss_fn = SegmentedPITMSELoss()
    value = loss_fn(pred, target)
    assert value.item() < 1e-6
