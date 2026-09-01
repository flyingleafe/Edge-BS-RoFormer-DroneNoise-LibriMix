"""Consolidated loss functions (docs/refactor-unified-framework.md § "Losses").

Every public loss is a small class/nn.Module declaring ``requires_pred`` /
``requires_target`` (``framespec.FrameSpec``) and
``__call__(pred: td.Frame, target: td.Frame) -> torch.Tensor`` — see
:class:`losses._common.Loss`. Pure tensor-level implementations (no Frame
dependency) are also exported for direct use/testing.
"""

from __future__ import annotations

from losses._common import Loss
from losses.amplitude_target import (
    AmplitudeTarget,
    AmplitudeTargetLoss,
    band_powers,
)
from losses.composite import CompositeLoss, LossTerm
from losses.masked import MaskedLoss, masked_loss
from losses.pit import (
    PITMSELoss,
    RPSMSELoss,
    SegmentedPITMSELoss,
    align_rps_to_gt,
    pairwise_mse,
    pit_mse_loss,
    segmented_pit_mse,
)
from losses.regularizers import SmoothnessPenalty, smoothness_penalty
from losses.salience import (
    SalienceBCELoss,
    SalienceRPSBCELoss,
    auto_pos_weight,
    salience_bce_loss,
)
from losses.salience_layers import (
    LayerPITSalienceBCELoss,
    layer_pit_bce,
)
from losses.sisdr import SISDRLoss, si_sdr_loss
from losses.spatial_likelihood import (
    SpatialLikelihood,
    SpatialLikelihoodLoss,
    spatial_whittle_nll,
    steering_vectors,
)
from losses.spectral import (
    AuraMRSTFTLoss,
    MultiScaleSTFT,
    MultiScaleSTFTLoss,
    multistft_reshape,
)
from losses.spectral_likelihood import (
    SpectralLikelihood,
    SpectralLikelihoodLoss,
    rice_nll,
    split_coherence,
)

__all__ = [
    "Loss",
    "AmplitudeTarget",
    "AmplitudeTargetLoss",
    "band_powers",
    "CompositeLoss",
    "LossTerm",
    "MaskedLoss",
    "masked_loss",
    "PITMSELoss",
    "RPSMSELoss",
    "SegmentedPITMSELoss",
    "align_rps_to_gt",
    "pairwise_mse",
    "pit_mse_loss",
    "segmented_pit_mse",
    "SmoothnessPenalty",
    "smoothness_penalty",
    "SISDRLoss",
    "si_sdr_loss",
    "SalienceBCELoss",
    "SalienceRPSBCELoss",
    "LayerPITSalienceBCELoss",
    "layer_pit_bce",
    "auto_pos_weight",
    "salience_bce_loss",
    "AuraMRSTFTLoss",
    "MultiScaleSTFT",
    "MultiScaleSTFTLoss",
    "SpatialLikelihood",
    "SpatialLikelihoodLoss",
    "spatial_whittle_nll",
    "steering_vectors",
    "SpectralLikelihood",
    "SpectralLikelihoodLoss",
    "rice_nll",
    "split_coherence",
    "multistft_reshape",
]
