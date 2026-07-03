"""BCE-on-salience loss with positive-class weighting.

Ported from the inline training step for salience-map RPS baselines
(``multif0_salience`` / ``basic_pitch_salience``) in ``train_rps_predictor.py``:
``F.binary_cross_entropy_with_logits(logits, sal_target, pos_weight=...)`` plus
the "auto" ``pos_weight`` heuristic (roughly #negative-bins / #positive-bins
per frame).
"""

from __future__ import annotations

import tdseries as td
import torch
import torch.nn.functional as F

from losses._common import get_tensor
from tasks.spec import FrameSpec, SeriesSpec

# ─── Pure tensor functions ───────────────────────────────────────────────────


def auto_pos_weight(n_bins: int, num_rotors: int, blur_bins: int) -> float:
    """Heuristic positive-class weight for salience BCE.

    ``active`` is the number of positive bins per frame (each of ``num_rotors``
    active harmonics/bins gets ``2 * blur_bins + 1`` bins after target
    blurring); the weight is roughly (#negative bins) / (#positive bins).
    """
    active = num_rotors * (2 * blur_bins + 1)
    return (n_bins - active) / max(active, 1)


def salience_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy on per-bin salience logits.

    Args:
        logits: (B, n_bins, T) raw model output (pre-sigmoid).
        target: (B, n_bins, T) target salience (binary or blurred/soft).
        pos_weight: scalar or per-bin positive-class weight, or None.
    """
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


# ─── Frame adapter ────────────────────────────────────────────────────────────


class SalienceBCELoss:
    """Frame adapter around :func:`salience_bce_loss`.

    Compares ``pred[pred_key]`` (default ``"salience"``, raw logits) against
    ``target[target_key]`` (default ``"salience"``, the RPS-derived target
    built by the salience-RPS task codec).
    """

    def __init__(
        self,
        *,
        pos_weight: float | None = None,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "salience",
    ) -> None:
        self.pos_weight = pos_weight
        self.pred_key = pred_key
        self.target_key = target_key
        spec = SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        logits = get_tensor(pred, self.pred_key)
        tgt = get_tensor(target, self.target_key)
        return salience_bce_loss(logits, tgt, pos_weight=self.pos_weight)


__all__ = ["auto_pos_weight", "salience_bce_loss", "SalienceBCELoss"]
