"""One ``build_model(name, **params) -> nn.Module`` entry point.

Re-exports two pre-existing model registries behind a single name lookup:

- the **RPS-model** registry, a verbatim copy of ``train_rps_predictor.py``'s
  ``MODEL_REGISTRY`` (imported directly from ``models.rps_predictor`` /
  ``models.multif0.rps_predictor`` / ``models.salience_rps`` rather than from
  ``train_rps_predictor.py`` itself, since that root script is deleted in a
  later refactor wave — see docs/refactor-unified-framework.md § "Execution
  waves");
- the **legacy** registry (``utils.get_model_from_config``'s ``model_type``
  dispatch — DCUNet/DCCRN/MDX23C/htdemucs/... 28+ types), reached by name
  through :func:`build_legacy_model`.

``build_model`` only covers the RPS registry (it takes plain kwargs, no
config file); the legacy registry needs a YAML config alongside the
``model_type`` string, so ``src/training/config.py`` calls
:func:`build_legacy_model` directly for that path instead of going through
here. Both are exposed from this module so callers have one place to look.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from models.multif0.rps_predictor import MultiF0RPSPredictor
from models.rps_predictor import (
    SimpleConv,
    SimpleConvAttnPool,
    SimpleConvBiGRU,
    SimpleConvBiGRUV2,
    SimpleConvMagPhaseBiGRU,
    SimpleConvMultiScale,
    SimpleConvSENext,
    SimpleConvTCN,
    SimpleConvV2,
    SimpleConvV2CausalGRU,
    SimpleConvV2CausalGRU96,
    SimpleConvV2CausalTCN,
    SimpleConvV2DualPool,
    SimpleConvV2GRU96,
    SimpleConvV2LocalAttention,
    SimpleConvV2MagPhase,
    SimpleConvV2MultiRes,
    SimpleConvV2SMoLBiGRU,
    SimpleConvV2SMoLCausalTCN,
    SimpleConvV2SMoLTCN,
    SimpleConvV2TCN,
    SimpleConvV2Transformer,
    SimpleConvV2UniGRU,
    SimpleConvV2UniGRU64NormDO03,
    SimpleConvV2UniGRU96NormDO02,
    SimpleConvV2UniGRU96NormDO03,
    SimpleConvV2UniGRU128,
    SimpleConvV2UniGRU128Norm,
    SimpleConvV2UniGRU128NormDO03,
    SimpleConvV2Wavelet,
    SimpleConvWide,
    SMoLnetRPSCausalTCN,
    SMoLnetRPSSimpleHead,
    SMoLnetRPSTCN,
)
from models.salience_rps import BasicPitchSalience, LateDeepSalience

# Verbatim copy of train_rps_predictor.py::MODEL_REGISTRY, minus the two
# DCUNet/DCCRN-encoder-+-RPS-head classes defined inline in that script
# (DCUNetEncRPS / DCCRNEncRPS) — those never moved into src/models/, so they
# are out of scope for this refactor wave (not used by any conf/model/*.yaml
# added here); add them here first if a future experiment needs them.
RPS_MODEL_REGISTRY: dict[str, Any] = {
    "simple_conv": SimpleConv,
    "simple_conv_v2": SimpleConvV2,
    "simple_conv_v2_tcn": SimpleConvV2TCN,
    "simple_conv_v2_causal_tcn": SimpleConvV2CausalTCN,
    "simple_conv_v2_smol_tcn": SimpleConvV2SMoLTCN,
    "simple_conv_v2_smol_causal_tcn": SimpleConvV2SMoLCausalTCN,
    "simple_conv_v2_smol_bigru": SimpleConvV2SMoLBiGRU,
    "smolnet_rps_tcn": SMoLnetRPSTCN,
    "smolnet_rps_simple_head": SMoLnetRPSSimpleHead,
    "smolnet_rps_causal_tcn": SMoLnetRPSCausalTCN,
    "simple_conv_v2_uni_gru": SimpleConvV2UniGRU,
    "simple_conv_v2_uni_gru128": SimpleConvV2UniGRU128,
    "simple_conv_v2_uni_gru128_norm": SimpleConvV2UniGRU128Norm,
    "simple_conv_v2_uni_gru128_norm_do03": SimpleConvV2UniGRU128NormDO03,
    "simple_conv_v2_uni_gru96_norm_do03": SimpleConvV2UniGRU96NormDO03,
    "simple_conv_v2_uni_gru96_norm_do02": SimpleConvV2UniGRU96NormDO02,
    "simple_conv_v2_uni_gru64_norm_do03": SimpleConvV2UniGRU64NormDO03,
    "simple_conv_v2_causal_gru": SimpleConvV2CausalGRU,
    "simple_conv_v2_causal_gru96": SimpleConvV2CausalGRU96,
    "simple_conv_v2_transformer": SimpleConvV2Transformer,
    "simple_conv_v2_local_attn": SimpleConvV2LocalAttention,
    "simple_conv_v2_multires": SimpleConvV2MultiRes,
    "simple_conv_v2_dwt": SimpleConvV2Wavelet,
    "simple_conv_v2_magphase": SimpleConvV2MagPhase,
    "simple_conv_v2_dual_pool": SimpleConvV2DualPool,
    "simple_conv_v2_gru96": SimpleConvV2GRU96,
    "simple_conv_wide": SimpleConvWide,
    "simple_conv_tcn": SimpleConvTCN,
    "simple_conv_multiscale": SimpleConvMultiScale,
    "simple_conv_bigru": SimpleConvBiGRU,
    "simple_conv_bigru_v2": SimpleConvBiGRUV2,
    "simple_conv_magphase_bigru": SimpleConvMagPhaseBiGRU,
    "simple_conv_attn_pool": SimpleConvAttnPool,
    "simple_conv_se_next": SimpleConvSENext,
    "multif0_rps": MultiF0RPSPredictor,
    "multif0_salience": LateDeepSalience,
    "basic_pitch_salience": BasicPitchSalience,
}


def build_model(name: str, **params: Any) -> nn.Module:
    """Build an RPS-family model by name (``RPS_MODEL_REGISTRY``).

    For the legacy (``utils.get_model_from_config``) registry use
    :func:`build_legacy_model` instead — it needs a config file, not bare
    kwargs.
    """
    if name not in RPS_MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose one of {sorted(RPS_MODEL_REGISTRY)}")
    return RPS_MODEL_REGISTRY[name](**params)


def build_legacy_model(model_type: str, config_path: str) -> nn.Module:
    """Build a model through the legacy ``utils.get_model_from_config`` registry.

    Discards the returned ``DictConfig`` — in the unified framework, task/data
    parameters come from the Hydra ``conf/`` tree, not the legacy YAML; the
    legacy YAML is only consulted for the architecture hyperparameters the
    model class itself needs (e.g. DCUNet's ``audio.n_fft``).
    """
    from utils import get_model_from_config

    model, _config = get_model_from_config(model_type, config_path)
    return model


__all__ = ["RPS_MODEL_REGISTRY", "build_model", "build_legacy_model"]
