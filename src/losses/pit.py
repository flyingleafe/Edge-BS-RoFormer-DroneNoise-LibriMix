"""Permutation-invariant losses for per-rotor RPS prediction.

Ported from ``train_rps_predictor.py`` (``pairwise_mse``, ``pit_mse_loss``)
and ``src/models/multif0/utils.py`` (``segmented_pit_mse``). RPS predictors
are permutation-invariant over rotor identity — the model has no way to know
which output row corresponds to which physical rotor — so training and
evaluation both search rotor permutations for the best match before scoring.
"""

from __future__ import annotations

import itertools
from functools import cache

import tdseries as td
import torch

from losses._common import get_tensor, rps_series_spec
from tasks.spec import FrameSpec

# ─── Pure tensor functions ───────────────────────────────────────────────────


# Rotor counts are physically small (quadrotor = 4); k! permutations are
# materialized, so an absurd k means the caller passed the wrong axis
# (e.g. an unbatched (K, T) tensor read as (B, K)) — fail fast instead of
# allocating k! tuples and taking the machine down.
_MAX_PIT_SOURCES = 8


@cache
def _permutations_tensor(k: int) -> torch.Tensor:
    """All ``k!`` permutations of ``range(k)`` as a ``(k!, k)`` long tensor."""
    if k > _MAX_PIT_SOURCES:
        raise ValueError(
            f"PIT over k={k} sources needs k! = {k}! permutations; refusing "
            f"(max {_MAX_PIT_SOURCES}). Check the tensor layout — expected "
            "(B, K, T) with K on axis 1."
        )
    return torch.tensor(list(itertools.permutations(range(k))), dtype=torch.long)


def pairwise_mse(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise MSE between each estimated and target rotor.

    Args:
        est: (B, K, T) predicted RPS.
        target: (B, K, T) ground-truth RPS.

    Returns:
        (B, K, K) pairwise MSE matrix where ``[b, i, j] = MSE(est[b,i], target[b,j])``.
    """
    diff = est.unsqueeze(2) - target.unsqueeze(1)
    return diff.pow(2).mean(dim=-1)  # (B, K, K)


def pit_mse_loss(
    est: torch.Tensor,
    target: torch.Tensor,
    perms: torch.Tensor | None = None,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Permutation-invariant MSE loss for RPS prediction.

    Finds the best 1-to-1 matching between predicted and target rotors by
    minimizing total MSE over all ``K!`` permutations (``K`` inferred from
    ``est``/``target``'s rotor axis — the original hardcoded ``K=4``).

    Args:
        est: (B, K, T) predicted RPS.
        target: (B, K, T) ground-truth RPS.
        perms: Pre-computed permutation tensor (P, K). If None, all ``K!``
            permutations are used (cached).
        return_indices: If True, also return the best permutation index per
            batch element (so callers can recover the full permutation for
            other metrics like MAE / R²).

    Returns:
        Scalar loss, or (loss, best_perm_idx) with shape (B,) if return_indices.
    """
    if est.dim() != 3 or target.dim() != 3:
        raise ValueError(
            f"pit_mse_loss expects (B, K, T) tensors, got est {tuple(est.shape)} "
            f"and target {tuple(target.shape)}; unbatched (K, T) input would be "
            "misread as K=T and explode combinatorially"
        )
    k = est.size(1)
    if perms is None:
        perms = _permutations_tensor(k).to(est.device)
    elif perms.device != est.device:
        perms = perms.to(est.device)

    pw = pairwise_mse(est, target)  # (B, K, K)

    b = pw.size(0)
    p = perms.size(0)

    src_idx = torch.arange(k, device=pw.device).view(1, 1, k)
    tgt_idx = perms.view(1, p, k)
    b_idx = torch.arange(b, device=pw.device).view(b, 1, 1)
    perm_losses = pw[b_idx, src_idx, tgt_idx]  # (B, P, K)
    perm_losses = perm_losses.sum(dim=-1)  # (B, P)

    best_loss, best_idx = perm_losses.min(dim=1)  # (B,)
    loss = best_loss.mean() / k
    if return_indices:
        return loss, best_idx
    return loss


def _segment_boundaries(merge_mask) -> list[tuple[int, int]]:
    """Convert a frame-level merge mask into (start, end) inclusive segment indices.

    A segment is a contiguous range of frames with no merge points; each merge
    point forms its own 1-frame segment where rotor identity is lost. Ported
    verbatim from ``src/models/multif0/utils.py``.
    """
    t = len(merge_mask)
    if t == 0:
        return []

    segments = []
    start = 0
    for i in range(t):
        if merge_mask[i]:
            if i > start:
                segments.append((start, i - 1))
            segments.append((i, i))
            start = i + 1
    if start < t:
        segments.append((start, t - 1))

    return segments


def segmented_pit_mse(
    rps_pred: torch.Tensor,
    rps_gt: torch.Tensor,
    merge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Segment-based PIT-MSE loss.

    Unlike :func:`pit_mse_loss`, which finds a single global permutation, this
    splits the timeline at merge points (frames where two rotors' rotation
    speeds cross, so a downstream tracker could plausibly swap identity) and
    finds the best permutation **independently within each segment**. Used by
    the salience-map RPS baselines, where trajectory identity is only
    meaningful between merge points.

    Args:
        rps_pred: (B, K, T) predicted RPS.
        rps_gt: (B, K, T) ground-truth RPS.
        merge_mask: (B, T) bool, or None. If None, degenerates to global PIT
            (one segment covering all frames).

    Returns:
        Scalar MSE loss.
    """
    b, k, t = rps_pred.shape
    device = rps_pred.device
    perms = list(itertools.permutations(range(k)))

    if merge_mask is None:
        merge_mask = torch.zeros(b, t, dtype=torch.bool, device=device)

    total_se = torch.tensor(0.0, device=device)
    total_count = 0

    for bi in range(b):
        merge_np = merge_mask[bi].cpu().numpy()
        segments = _segment_boundaries(merge_np)

        for start, end in segments:
            pred_seg = rps_pred[bi, :, start : end + 1]
            gt_seg = rps_gt[bi, :, start : end + 1]

            best_err = float("inf")
            for perm in perms:
                err = ((pred_seg - gt_seg[perm, :]) ** 2).sum()
                if err < best_err:
                    best_err = err

            total_se = total_se + best_err
            total_count = total_count + (end - start + 1)

    return total_se / (total_count * k)


def _batched_rps(x: torch.Tensor) -> torch.Tensor:
    """Normalize an RPS tensor to batched ``(B, K, T)``: a per-sample Frame
    entry is ``(K, T)`` and gets a singleton batch axis."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"expected (K, T) or (B, K, T) RPS tensor, got {tuple(x.shape)}")
    return x


# ─── Frame adapters ──────────────────────────────────────────────────────────


class PITMSELoss:
    """Frame adapter around :func:`pit_mse_loss`.

    Compares ``pred[pred_key]`` (default ``"rps_pred"``) against
    ``target[target_key]`` (default ``"rps"``).
    """

    def __init__(
        self,
        *,
        rate: tuple[int, int] | None = None,
        pred_key: str = "rps_pred",
        target_key: str = "rps",
    ) -> None:
        self.pred_key = pred_key
        self.target_key = target_key
        spec = rps_series_spec(rate)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        est = _batched_rps(get_tensor(pred, self.pred_key))
        tgt = _batched_rps(get_tensor(target, self.target_key))
        loss = pit_mse_loss(est, tgt)
        assert isinstance(loss, torch.Tensor)
        return loss


class SegmentedPITMSELoss:
    """Frame adapter around :func:`segmented_pit_mse`.

    ``target[merge_mask_key]`` is optional; when absent this degenerates to
    global PIT over the whole clip (same as :class:`PITMSELoss` but without
    the ``/K`` normalisation quirk — see :func:`segmented_pit_mse`).
    """

    def __init__(
        self,
        *,
        rate: tuple[int, int] | None = None,
        pred_key: str = "rps_pred",
        target_key: str = "rps",
        merge_mask_key: str = "merge_mask",
    ) -> None:
        self.pred_key = pred_key
        self.target_key = target_key
        self.merge_mask_key = merge_mask_key
        spec = rps_series_spec(rate)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec(
            {target_key: spec, merge_mask_key: rps_series_spec(rate)},
            optional=frozenset({merge_mask_key}),
        )

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        est = _batched_rps(get_tensor(pred, self.pred_key))
        tgt = _batched_rps(get_tensor(target, self.target_key))
        merge_mask = None
        if self.merge_mask_key in target:
            mask = get_tensor(target, self.merge_mask_key).bool()
            merge_mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        return segmented_pit_mse(est, tgt, merge_mask)


class RPSMSELoss:
    """Plain (non-PIT) MSE between ``pred[pred_key]`` and ``target[target_key]``.

    For the conditional RPS *refiner* (``simple_conv_v2_ckla_phaseonly_cond``):
    the model's output row ``i`` corresponds to its conditioning row ``i`` by
    construction (bounded residual on the conditioning track), and the
    corruption seam (``data_processing.rps_corruption``) emits the ground
    truth already permuted to the conditioning order — so no permutation
    search is wanted: a PIT loss would silently forgive identity swaps the
    refiner is supposed to be pinned against.
    """

    def __init__(
        self,
        *,
        rate: tuple[int, int] | None = None,
        pred_key: str = "rps_pred",
        target_key: str = "rps",
    ) -> None:
        self.pred_key = pred_key
        self.target_key = target_key
        spec = rps_series_spec(rate)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        est = _batched_rps(get_tensor(pred, self.pred_key))
        tgt = _batched_rps(get_tensor(target, self.target_key))
        if est.shape != tgt.shape:
            raise ValueError(
                f"RPSMSELoss shape mismatch: pred {tuple(est.shape)} vs target {tuple(tgt.shape)}"
            )
        return torch.nn.functional.mse_loss(est, tgt)


__all__ = [
    "pairwise_mse",
    "pit_mse_loss",
    "segmented_pit_mse",
    "PITMSELoss",
    "RPSMSELoss",
    "SegmentedPITMSELoss",
]
