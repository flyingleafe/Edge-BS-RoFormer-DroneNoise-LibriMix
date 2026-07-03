"""One ``build_model(name, **params) -> nn.Module`` entry point.

Re-exports three pre-existing model registries behind a single name lookup:

- the **RPS-model** registry, a verbatim copy of the former
  ``train_rps_predictor.py``'s ``MODEL_REGISTRY`` (imported directly from
  ``models.rps_predictor`` / ``models.multif0.rps_predictor`` /
  ``models.salience_rps`` — that root script has been deleted per
  docs/refactor-unified-framework.md § "Execution waves");
- the **noise-generation** registry, a verbatim copy of the former
  ``train_noise_generation.py``'s ``MODEL_REGISTRY``/``get_model``/
  ``build_loss`` (that script is also deleted; no ``conf/model`` wiring for
  the ``noise_generation`` task exists yet — see docs/refactor-unified-framework.md
  § "Future expansions" — but report/notebook figure scripts still reconstruct
  a trained generator + its loss to load a checkpoint, e.g.
  ``notebooks/noise_gen_real_vs_generated.ipynb``);
- the **legacy** registry (``utils.get_model_from_config``'s ``model_type``
  dispatch — DCUNet/DCCRN/MDX23C/htdemucs/... 28+ types), reached by name
  through :func:`build_legacy_model`.

``build_model`` only covers the RPS registry (it takes plain kwargs, no
config file); the legacy registry needs a YAML config alongside the
``model_type`` string, so ``src/training/config.py`` calls
:func:`build_legacy_model` directly for that path instead of going through
here. All three are exposed from this module so callers have one place to look.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from models.generative import MultiScaleSTFT, PositionalHarmonicNoiseGen
from models.multif0.rps_predictor import MultiF0RPSPredictor
from models.rps_predictor import (
    DCCRNEncRPS,
    DCUNetEncRPS,
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

# Verbatim copy of the former train_rps_predictor.py::MODEL_REGISTRY.
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
    "dcunet_enc_rps": DCUNetEncRPS,
    "dccrn_enc_rps": lambda **kw: DCCRNEncRPS(lite=False, **kw),
    "dccrn_lite_rps": lambda **kw: DCCRNEncRPS(lite=True, **kw),
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


def get_rps_model(
    model_name: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    num_rotors: int = 4,
    hcqt_fmin: float | None = None,
    fused_branches: bool = False,
    stacked_hcqt: bool = False,
    salience_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Build an RPS-family model with the salience-model config overrides.

    Verbatim port of the former ``train_rps_predictor.py::get_model`` (kept
    under a different name here since :func:`build_model` already owns the
    plain ``name, **params`` contract). Still needed by report/slide figure
    scripts that reconstruct a narrow-input/super-resolution salience model
    to load a checkpoint (e.g. ``writing/reports/2026-06-15/prepare_narrow.py``).

    ``hcqt_fmin`` overrides the HCQT base frequency for ``multif0_salience``
    (default in the model is A0 = 27.5 Hz); ``fused_branches`` runs LateDeep's
    mag/phase branches as one grouped stack; ``stacked_hcqt`` uses the
    single-CQT + harmonic-shift front-end. ``salience_cfg`` is a dict of
    optional narrow-input / super-resolution-output overrides (keys ignored
    when ``None``). All ignored by non-salience models.
    """
    if model_name not in RPS_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(RPS_MODEL_REGISTRY)}")
    kwargs: dict[str, Any] = dict(n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors)
    cfg = salience_cfg or {}

    def _merge(keys: list[str]) -> None:
        for k in keys:
            if cfg.get(k) is not None:
                kwargs[k] = cfg[k]

    if model_name == "multif0_salience":
        if hcqt_fmin is not None:
            kwargs["fmin"] = hcqt_fmin
        if fused_branches:
            kwargs["fused_branches"] = True
        if stacked_hcqt:
            kwargs["stacked"] = True  # rides through LateDeepSalience -> build_frontend
        _merge(
            [
                "n_octaves",
                "over_sample",
                "harmonics",
                "superres_out",
                "out_fmin",
                "out_fmax",
                "out_bins",
            ]
        )
    elif model_name == "basic_pitch_salience":
        _merge(
            [
                "bp_fmin",
                "bins_per_semitone",
                "n_contour_semitones",
                "superres_out",
                "out_fmin",
                "out_fmax",
                "out_bins",
            ]
        )
    return RPS_MODEL_REGISTRY[model_name](**kwargs)


# Verbatim copy of the former train_noise_generation.py::MODEL_REGISTRY.
NOISE_GEN_MODEL_REGISTRY: dict[str, Any] = {
    "positional_harmonic_gen": PositionalHarmonicNoiseGen,
}


class _CodebookConditionedNoiseGen(nn.Module):
    """Bundles a position-aware generator with its external per-drone codebook.

    The deleted ``train_noise_generation.py`` trainer kept
    ``tasks.noise_generation.DroneCodebook`` fully external to "the model"
    (its own optimizer param group, its own bundle-file entry). The unified
    ``training.loop.run_training`` has a narrower single-model contract —
    one ``optimizer = get_optimizer(model, ...)`` over ``model.parameters()``,
    one checkpoint = ``model.state_dict()`` (see
    ``training/loop.py``/``eval.py``) — so a codebook that needs to be
    *trained* and *persisted* through that contract must be a submodule of
    the instantiated model. This wrapper is that composition:
    :meth:`forward` resolves each sample's conditioning code ``z`` from its
    drone *name* via the codebook, then calls the generator — matching
    :class:`tasks.codecs.NoiseGenerationCodec`'s ``conditioned=True`` call
    convention (``model(rps, rel_pos, drone_names)``).
    """

    def __init__(self, generator: nn.Module, codebook: nn.Module) -> None:
        super().__init__()
        self.generator = generator
        self.codebook = codebook

    def forward(
        self,
        rps: Any,
        rel_pos: Any,
        drone_names: list[str],
        **kwargs: Any,
    ) -> Any:
        z = self.codebook(list(drone_names))
        return self.generator(rps, rel_pos, z=z, **kwargs)


def build_noise_gen_model(
    model_name: str,
    *,
    sample_rate: int = 16000,
    n_harmonics: int = 100,
    use_diff_noise: bool = True,
    cond_dim: int = 0,
    drone_names: list[str] | None = None,
) -> nn.Module:
    """Construct a noise-generation model by name (``NOISE_GEN_MODEL_REGISTRY``).

    Verbatim port of the former ``train_noise_generation.py::get_model``,
    plus ``drone_names``: when ``cond_dim > 0``, the returned model is a
    :class:`_CodebookConditionedNoiseGen` wrapping the generator and a fresh
    ``tasks.noise_generation.DroneCodebook(cond_dim, names=drone_names)`` —
    see that class's docstring for why the codebook now lives inside the
    model rather than external to it. ``cond_dim == 0`` (single-drone,
    unconditioned) returns the bare generator, ``drone_names`` ignored.
    """
    if model_name not in NOISE_GEN_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {sorted(NOISE_GEN_MODEL_REGISTRY)}"
        )
    generator = NOISE_GEN_MODEL_REGISTRY[model_name](
        sample_rate=sample_rate,
        n_harmonics=n_harmonics,
        use_diff_noise=use_diff_noise,
        cond_dim=cond_dim,
    )
    if cond_dim <= 0:
        return generator
    if not drone_names:
        raise ValueError("cond_dim > 0 requires drone_names (DroneCodebook keys)")

    from tasks.noise_generation import DroneCodebook

    codebook = DroneCodebook(cond_dim, names=list(drone_names))
    return _CodebookConditionedNoiseGen(generator, codebook)


def build_noise_gen_loss(
    *,
    stft_sizes: list[int] | None = None,
    log_weight: float = 1.0,
    loss_type: str = "L1",
) -> MultiScaleSTFT:
    """Build the multi-scale STFT loss used to train/score noise-generation models.

    Verbatim port of the former ``train_noise_generation.py::build_loss``, minus
    the ``argparse.Namespace`` indirection — takes the three loss knobs directly.
    """
    return MultiScaleSTFT(
        n_ffts=list(stft_sizes or [2048, 1024, 512, 256, 128]),
        log_weight=log_weight,
        loss_type=loss_type,
    )


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


def build_legacy_inline(model_type: str, config: Any) -> nn.Module:
    """Build a legacy-registry model from an **inline** config (the Hydra-native
    replacement for ``build_legacy_model``'s file + ``legacy_config_path``).

    ``config`` is the ZFTurbo-style tree (``audio`` / ``model`` / ``training``
    sections) inlined directly into a ``conf/model/*.yaml`` under ``params`` —
    identical content to the former ``configs/*.yaml`` file, just no separate
    file. Routes through the exact same construction dispatch
    (:func:`utils.build_model_from_config`) as the file-based path, so the
    resulting module is bit-for-bit identical to the legacy build."""
    from omegaconf import DictConfig, OmegaConf

    from utils import build_model_from_config

    cfg = config if isinstance(config, DictConfig) else OmegaConf.create(config)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"inline model config must be a mapping, got {type(cfg).__name__}")
    return build_model_from_config(model_type, cfg)


__all__ = [
    "RPS_MODEL_REGISTRY",
    "NOISE_GEN_MODEL_REGISTRY",
    "build_model",
    "build_legacy_model",
    "build_legacy_inline",
    "get_rps_model",
    "build_noise_gen_model",
    "build_noise_gen_loss",
]
