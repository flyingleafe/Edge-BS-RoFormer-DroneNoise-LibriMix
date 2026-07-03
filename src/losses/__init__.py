"""Consolidated loss functions (docs/refactor-unified-framework.md § "Losses").

Every public loss is a small class/nn.Module declaring ``requires_pred`` /
``requires_target`` (``tasks.spec.FrameSpec``) and
``__call__(pred: td.Frame, target: td.Frame) -> torch.Tensor`` — see
:class:`losses._common.Loss`. Pure tensor-level implementations (no Frame
dependency) are also exported for direct use/testing.
"""

from __future__ import annotations

from losses._common import Loss
from losses.composite import CompositeLoss, LossTerm
from losses.masked import MaskedLoss, masked_loss
from losses.pit import (
    PITMSELoss,
    SegmentedPITMSELoss,
    pairwise_mse,
    pit_mse_loss,
    segmented_pit_mse,
)
from losses.regularizers import SmoothnessPenalty, smoothness_penalty
from losses.salience import SalienceBCELoss, auto_pos_weight, salience_bce_loss
from losses.spectral import (
    AuraMRSTFTLoss,
    MultiScaleSTFT,
    MultiScaleSTFTLoss,
    multistft_reshape,
)

__all__ = [
    "Loss",
    "CompositeLoss",
    "LossTerm",
    "MaskedLoss",
    "masked_loss",
    "PITMSELoss",
    "SegmentedPITMSELoss",
    "pairwise_mse",
    "pit_mse_loss",
    "segmented_pit_mse",
    "SmoothnessPenalty",
    "smoothness_penalty",
    "SalienceBCELoss",
    "auto_pos_weight",
    "salience_bce_loss",
    "AuraMRSTFTLoss",
    "MultiScaleSTFT",
    "MultiScaleSTFTLoss",
    "multistft_reshape",
]
