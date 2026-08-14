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
- the **legacy** registry (:data:`LEGACY_MODEL_BUILDERS` — the dict-ified
  former ``utils.build_model_from_config`` if/elif chain; DCUNet/DCCRN/
  htdemucs/sgmse/... 10 live types), reached through
  :func:`build_legacy_model` (config file path) or
  :func:`build_legacy_inline` (inline config).

``build_model`` only covers the RPS registry (it takes plain kwargs, no
config file); the legacy registry needs a YAML config alongside the
``model_type`` string, so ``src/training/config.py`` calls
:func:`build_legacy_model` directly for that path instead of going through
here. All three are exposed from this module so callers have one place to look;
:func:`model_types` is the unified name listing across all of them (plus the
direct ``conf/model`` ``_target_`` factories).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from models.generative import (
    MultiScaleSTFT,
    PositionalHarmonicNoiseGen,
    PositionalHarmonicPlusWindGen,
)

if TYPE_CHECKING:
    from models.generative.codebook import DroneCodebook
    from models.generative.propagation import MicEQ
from models.ckla import SimpleConvV2CKLA, SimpleConvV2CKLACond
from models.fkla import FKLARPSModel
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
    SimpleConvV2TransformerComb,
    SimpleConvV2TransformerHCQT,
    SimpleConvV2TransformerIF,
    SimpleConvV2TransformerPyramid,
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
def _build_edge_bs_rof_rps(**kw: Any):
    """Lazy builder — the roformer stack imports beartype/einops/
    rotary_embedding_torch, which the rest of the registry never needs."""
    from models.edge_bs_rof.rps import BSRoformerRPS

    return BSRoformerRPS(**kw)


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
    "simple_conv_v2_transformer_hcqt": SimpleConvV2TransformerHCQT,
    "simple_conv_v2_transformer_if": SimpleConvV2TransformerIF,
    "simple_conv_v2_transformer_comb": SimpleConvV2TransformerComb,
    "simple_conv_v2_transformer_pyramid": SimpleConvV2TransformerPyramid,
    # CKLA head arms (docs/ckla-design.md §3): default stft_mag_if front-end,
    # plus a stft_mag variant (P1 ablation 5 — front-end interaction check).
    "simple_conv_v2_ckla": SimpleConvV2CKLA,
    "simple_conv_v2_ckla_mag": lambda **kw: SimpleConvV2CKLA(frontend_key="stft_mag", **kw),
    # rotation=False controls (design §5 item 1): the scan degenerates to the
    # exact real-KLA recursion — decides whether the complex path is
    # load-bearing (P0b eval-time ablation measured a null delta).
    "simple_conv_v2_ckla_norot": lambda **kw: SimpleConvV2CKLA(rotation=False, **kw),
    "simple_conv_v2_ckla_mag_norot": lambda **kw: SimpleConvV2CKLA(
        frontend_key="stft_mag", rotation=False, **kw
    ),
    # Phase-differential readout arms: feed the mix layer the angle first
    # differential arg(y_t·conj(y_{t−1})) — the state phasor's angular
    # velocity, i.e. the tracked instantaneous frequency — instead of (or in
    # addition to) the raw [Re y, Im y] quadratures (ckla.py::
    # phase_diff_features). Mechanistic candidate for why rotation-on lost
    # to plain KLA: the mean readout discards the phase velocity and passes
    # the ω-oscillation through as feature noise.
    "simple_conv_v2_ckla_phasediff": lambda **kw: SimpleConvV2CKLA(readout="phase_diff", **kw),
    "simple_conv_v2_ckla_phaseonly": lambda **kw: SimpleConvV2CKLA(readout="phase_only", **kw),
    "simple_conv_v2_ckla_phaseunit": lambda **kw: SimpleConvV2CKLA(readout="phase_unit", **kw),
    # Conditional RPS refiner (ckla.py::SimpleConvV2CKLACond): phase-only
    # CKLA backbone + corrupted-track conditioning (concat before the head)
    # + bounded residual output — forward(audio, cond), plain non-PIT loss.
    "simple_conv_v2_ckla_phaseonly_cond": lambda **kw: SimpleConvV2CKLACond(
        readout="phase_only", **kw
    ),
    # Vendored flat-KLA (kla-loglinear@11e5a39, src/models/fkla/) plain-KLA
    # arm — cross-implementation companion to the norot controls.
    "simple_conv_v2_fkla": FKLARPSModel,
    # Edge-BS-RoFormer trunk adapted to RPS (models/edge_bs_rof/rps.py):
    # the paper's rotary-embedding harmonic-tracking claim, tested on the
    # task where the target IS the harmonic-line trajectory. Lazy import —
    # the roformer stack pulls beartype/einops/rotary_embedding_torch.
    "edge_bs_rof_rps": _build_edge_bs_rof_rps,
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

    For the legacy registry (:data:`LEGACY_MODEL_BUILDERS`) use
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
    "positional_harmonic_wind_gen": PositionalHarmonicPlusWindGen,
}


class _CodebookConditionedNoiseGen(nn.Module):
    """Bundles a position-aware generator with its external per-drone codebook.

    The deleted ``train_noise_generation.py`` trainer kept
    ``models.generative.codebook.DroneCodebook`` fully external to "the model"
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

    ``z_noise_std`` (opt-in, default off) is **vicinal conditioning noise**:
    with only 2 codebook entries the decoder sees exactly 2 points in z-space,
    so nothing constrains its behaviour *around* each code — later vicinal
    sampling / interpolation between drones would step into unregularised
    territory. When ``z_noise_std > 0`` and the module is in training mode,
    ``eps ~ N(0, (z_noise_std * RMS(z))^2 I)`` is added to each sample's code
    before it reaches the emitter's FiLM. The scale is **relative** (a fraction
    of the per-sample code RMS, ``||z|| / sqrt(d)``): ``DroneCodebook`` codes
    are initialised tiny (``init_std=0.01``) and grow freely during training,
    so an *absolute* std would start out dominating the code and end up
    negligible — a relative one keeps the vicinal ball a constant fraction of
    the code's own magnitude throughout. Recommended value 0.1 (a 10%
    perturbation). The RMS factor is detached, so the noise scale does not
    feed gradients back into the code. Off at eval (no override — vicinal
    noise is purely a training-time smoothness prior).

    ``learn_rps_jitter_sigma`` (opt-in) makes the OU jitter linewidth a
    **learnable per-drone** parameter instead of the single global scalar baked
    into the emitter. The physical jitter amplitude differs by airframe/throttle
    regime (a DREGON-calibrated sigma did not transfer to the M100 at idle in
    E6), so one sigma per codebook entry lets each drone's linewidth be fit
    jointly with everything else. A raw scalar per drone name is kept here (next
    to the codebook, so it is trained and persisted through the single-model
    contract) and mapped to a positive sigma via ``softplus``; the resolved
    per-sample sigma ``[B]`` is threaded into the generator forward
    (``rps_jitter_sigma=``), folded onto the rotor axis by
    :meth:`~models.generative.PositionalHarmonicNoiseGen.emit`, and replaces the
    emitter's scalar. Like the codebook it is **name-keyed** (few-shot: freeze
    everything, add a drone's code + sigma, fit just those). ``sigma`` gradients
    flow because :meth:`_apply_rps_jitter` factors sigma out of the (gradient-free)
    OU innovation path. It respects the same train/eval gate as the scalar jitter:
    active only when the emitter would apply jitter (training, or an explicit
    ``rps_jitter=True`` override at eval).
    """

    def __init__(
        self,
        generator: nn.Module,
        codebook: nn.Module,
        *,
        z_noise_std: float = 0.0,
        learn_rps_jitter_sigma: bool = False,
        rps_jitter_sigma_init: float = 0.6,
        per_rotor_deltas: bool = False,
        cond_dim: int = 0,
        n_rotors: int = 4,
        amp_calibration: bool = False,
        n_mics: int = 8,
        noise_floor_bands: int = 0,
        mic_eq_knots: int = 0,
        eq_f_min: float = 20.0,
        eq_f_max: float = 8000.0,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.codebook = codebook
        self.z_noise_std = float(z_noise_std)
        # Per-rotor sub-embeddings: the emitter code becomes z_r = z_drone + δz_r,
        # where δz_r is a learnable per-rotor delta SHARED across drones (rotor
        # identity — position/manufacturing timbre — is drone-independent). Zero
        # init ⇒ starts identical to the per-clip model, a strict generalisation;
        # each rotor's delta then diverges under its own position/RPS gradients.
        self.per_rotor_deltas = bool(per_rotor_deltas)
        self.rotor_deltas: nn.Parameter | None = None
        if self.per_rotor_deltas:
            if cond_dim <= 0:
                raise ValueError("per_rotor_deltas requires cond_dim > 0")
            self.rotor_deltas = nn.Parameter(torch.zeros(n_rotors, cond_dim))
        self.learn_rps_jitter_sigma = bool(learn_rps_jitter_sigma)
        self.log_jitter_sigma: nn.ParameterDict | None = None
        if self.learn_rps_jitter_sigma:
            # softplus^{-1}(init) so the initial resolved sigma == the calibrated
            # scalar; one raw scalar per codebook name (few-shot / name-keyed).
            cb = cast("DroneCodebook", codebook)
            init = float(rps_jitter_sigma_init)
            raw0 = float(math.log(math.expm1(init))) if init > 0 else -10.0
            self.log_jitter_sigma = nn.ParameterDict(
                {cb._key(name): nn.Parameter(torch.tensor(raw0)) for name in cb.names()}
            )
        # ── absolute-level calibration (the amplitude-target path) ───────────
        # The decomposition targets live in the RECORDING's absolute units; the
        # emitter's output does not, and nothing else in the objective anchors
        # it. Three name-keyed, zero-initialised (unity) log-gains close that
        # gap, all constant in time so none of them can leak level DYNAMICS:
        #   log_gain        one per drone — the coherent branch's unit constant;
        #   log_mic_gain    M per drone — real arrays deviate from a pure 1/r law
        #                   (per-microphone sensitivity, directivity, room), and
        #                   the learned values double as an array-calibration
        #                   readout;
        #   log_gain_noise  one per drone, applied in the POWER domain — the
        #                   broadband branch's magnitude response feeds a
        #                   unit-variance excitation, so its target-side
        #                   normalization constant is its own and must not be
        #                   forced through the coherent branch's gain.
        # They are applied to EVERY prediction path (rendered audio included), so
        # a calibrated model renders at the recording's level rather than needing
        # a post-hoc scale. Off by default => existing configs are unchanged.
        self.amp_calibration = bool(amp_calibration)
        self.n_mics = int(n_mics)
        self.noise_floor_bands = int(noise_floor_bands)
        # ── the propagation head's frequency-dependent half ──────────────────
        # `mic_eq_knots > 0` REPLACES the coherent branch's frequency-flat
        # per-microphone scalar with a smooth learnable magnitude response
        # EQ_c(f), per rig, per microphone, SHARED across rotors — the full
        # amplitude-only propagation law is
        #   A_obs[r,k,c](t) = A_src[r,k](t) * (1/dist_{r,c}) * EQ_c(f_k(t)).
        # A room's reverberant transfer is frequency-structured and a rate-swept
        # harmonic samples it, so the measured envelopes carry an rps-correlated
        # ripple a flat scalar cannot express (see
        # models.generative.propagation and
        # docs/experiments/amplitude-target-training.md). The two are mutually
        # exclusive because a flat EQ IS the scalar: keeping both would leave
        # the level split between two redundant parameters.
        self.mic_eq_knots = int(mic_eq_knots)
        self.mic_eq: nn.Module | None = None
        self.log_gain: nn.ParameterDict | None = None
        self.log_mic_gain: nn.ParameterDict | None = None
        self.log_gain_noise: nn.ParameterDict | None = None
        self.log_mic_gain_noise: nn.ParameterDict | None = None
        self.log_floor_psd: nn.ParameterDict | None = None
        if self.mic_eq_knots > 0 and not self.amp_calibration:
            raise ValueError("mic_eq_knots > 0 requires amp_calibration=True")
        if self.amp_calibration:
            cb = cast("DroneCodebook", codebook)
            keys = [cb._key(name) for name in cb.names()]
            self.log_gain = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in keys})
            self.log_gain_noise = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in keys})
            if self.mic_eq_knots > 0:
                from models.generative.propagation import MicEQ

                self.mic_eq = MicEQ(
                    keys,
                    self.n_mics,
                    n_knots=self.mic_eq_knots,
                    f_min=eq_f_min,
                    f_max=eq_f_max,
                )
            else:
                self.log_mic_gain = nn.ParameterDict(
                    {k: nn.Parameter(torch.zeros(self.n_mics)) for k in keys}
                )
            # The broadband branch gets its OWN per-microphone gain. Measured
            # reason (docs/experiments/residual-attribution.md): the residual is
            # 70-90 % per-microphone INCOHERENT and its per-mic energy spread on
            # DREGON is 8.54 dB, while four equal 1/r rotor sources can produce
            # at most 1.59 dB — per-rotor attribution of the residual is refuted.
            # Sharing one per-mic gain with the coherent branch would force the
            # rotor-propagated broadband term to absorb a pattern it structurally
            # cannot make, and would corrupt the coherent gains (the readout that
            # is supposed to be an array calibration) doing it.
            self.log_mic_gain_noise = nn.ParameterDict(
                {k: nn.Parameter(torch.zeros(self.n_mics)) for k in keys}
            )
            if self.noise_floor_bands > 0:
                # An explicit per-microphone, per-band STATIC floor, added in the
                # power domain. This is the part of the residual that is not a
                # propagated rotor source at all (self-noise / flow noise), so it
                # is modelled as what it is instead of being pushed through the
                # 1/r law. Init -12 nats (6e-6) sits about a quarter of a typical
                # residual band power on these recordings, so it is neither inert
                # nor dominant at step 0; fitted per-mic floors D_m(f) exist in
                # results/residual_attribution/real_*.json if a better warm start
                # is ever wanted.
                self.log_floor_psd = nn.ParameterDict(
                    {
                        k: nn.Parameter(torch.full((self.n_mics, self.noise_floor_bands), -12.0))
                        for k in keys
                    }
                )

    def _resolve_calibration(
        self, drone_names: list[str], n_mics: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(amplitude gain [B, M], noise POWER gain [B, M])``, or ``None``.

        Each branch carries its own per-microphone gain and its own scalar unit
        constant — see the constructor for why the two must not share one. With
        a :class:`~models.generative.propagation.MicEQ` fitted, the COHERENT
        branch's per-microphone term is the EQ curve instead of a scalar, so the
        gain returned here is that branch's unit constant alone; the caller
        multiplies the EQ in on the axis it is evaluated on.
        """
        if self.log_gain is None or self.log_gain_noise is None:
            return None
        assert self.log_mic_gain_noise is not None
        cb = cast("DroneCodebook", self.codebook)
        keys = [cb._key(n) for n in drone_names]
        if n_mics > self.n_mics:
            raise ValueError(
                f"amp calibration was built for {self.n_mics} microphones, batch has {n_mics}"
            )
        mic_n = torch.stack([self.log_mic_gain_noise[k][:n_mics] for k in keys], dim=0)  # [B, M]
        glob = torch.stack([self.log_gain[k] for k in keys], dim=0).unsqueeze(-1)  # [B, 1]
        noise = torch.stack([self.log_gain_noise[k] for k in keys], dim=0).unsqueeze(-1)  # [B, 1]
        if self.log_mic_gain is None:
            mic = glob.new_zeros((len(keys), n_mics))  # the EQ carries the mic axis
        else:
            mic = torch.stack([self.log_mic_gain[k][:n_mics] for k in keys], dim=0)  # [B, M]
        return torch.exp(mic + glob), torch.exp(2.0 * mic_n + noise)

    def _add_noise_floor(self, psd: torch.Tensor, drone_names: list[str]) -> torch.Tensor:
        """Add the static per-microphone incoherent floor to ``[B, M, t, F]``."""
        if self.log_floor_psd is None:
            return psd
        cb = cast("DroneCodebook", self.codebook)
        n_mics, n_bands = int(psd.shape[1]), int(psd.shape[-1])
        if n_bands != self.noise_floor_bands:
            raise ValueError(
                f"noise_floor_bands={self.noise_floor_bands} does not match the branch's "
                f"{n_bands} bands; set it to the emitter's `noise_amps` band count"
            )
        floor = torch.stack(
            [self.log_floor_psd[cb._key(n)][:n_mics] for n in drone_names], dim=0
        )  # [B, M, F]
        return psd + torch.exp(floor).unsqueeze(-2)

    def _resolve_jitter_sigma(self, drone_names: list[str]) -> torch.Tensor | None:
        """Per-sample learnable linewidth ``[B]`` (softplus of the per-drone raw)."""
        if self.log_jitter_sigma is None:
            return None
        cb = cast("DroneCodebook", self.codebook)
        raws = torch.stack([self.log_jitter_sigma[cb._key(n)] for n in drone_names], dim=0)
        return nn.functional.softplus(raws)  # [B], strictly positive

    def _resolve_conditioning(self, drone_names: list[str], kwargs: dict[str, Any]) -> torch.Tensor:
        """Per-sample code ``z`` (plus the per-drone jitter sigma, threaded into
        ``kwargs``) — shared by :meth:`forward` and :meth:`spectral_stats` so the
        two cannot drift apart."""
        z = self.codebook(list(drone_names))
        if self.z_noise_std > 0.0 and self.training:
            rms = z.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
            z = z + torch.randn_like(z) * (self.z_noise_std * rms)
        if self.rotor_deltas is not None:
            # z_drone [B, d] -> per-rotor z_r [B, R, d] = z_drone + δz_r
            z = z.unsqueeze(1) + self.rotor_deltas.unsqueeze(0)
        sigma = self._resolve_jitter_sigma(list(drone_names))
        if sigma is not None:
            kwargs.setdefault("rps_jitter_sigma", sigma)
        return z

    @staticmethod
    def _n_mics(rel_pos: Any) -> int:
        """Observer count of a ``[B, R, 3]`` (=1) or ``[B, M, R, 3]`` geometry."""
        return 1 if rel_pos.dim() == 3 else int(rel_pos.shape[1])

    def _eq_filter(self, audio: torch.Tensor, drone_names: list[str]) -> torch.Tensor:
        """Apply the per-microphone EQ to a rendered ``[B, M, T]`` waveform.

        The amplitude path multiplies each line by ``EQ_c(f_k(t))``, so a
        rendering must carry the same response or a checkpoint would sound
        different from what it was fit to.
        """
        if self.mic_eq is None:
            return audio
        eq = cast("MicEQ", self.mic_eq)
        sr = float(cast(int, getattr(self.generator, "sample_rate", 16000)))
        return eq.filter_audio(audio, list(drone_names), sr)

    def forward(
        self,
        rps: Any,
        rel_pos: Any,
        drone_names: list[str],
        **kwargs: Any,
    ) -> Any:
        z = self._resolve_conditioning(list(drone_names), kwargs)
        out = self.generator(rps, rel_pos, z=z, **kwargs)
        cal = self._resolve_calibration(list(drone_names), self._n_mics(rel_pos))
        if cal is None:
            return out
        # Rendered audio is [B, M, T], or [B, T] for a single-observer geometry.
        gain, _ = cal
        scale = gain.unsqueeze(-1) if rel_pos.dim() == 4 else gain[:, 0].reshape(-1, 1)

        def _apply(audio: torch.Tensor) -> torch.Tensor:
            single = audio.dim() == 2  # [B, T] — one observer, microphone 0
            a = audio.unsqueeze(1) if single else audio
            a = self._eq_filter(a, list(drone_names))
            return a.squeeze(1) if single else a

        if isinstance(out, dict):
            out = dict(out)
            out["audio"] = _apply(out["audio"] * scale)
            return out
        return _apply(out * scale)

    def spatial_stats(
        self,
        rps: Any,
        rel_pos: Any,
        drone_names: list[str],
        **kwargs: Any,
    ) -> Any:
        """Conditioned passthrough to the generator's SPATIAL statistics (per-rotor
        source power and per-mic wind power kept apart) — see
        ``losses.spatial_likelihood``."""
        fn = getattr(self.generator, "spatial_stats", None)
        if fn is None:
            raise TypeError(
                f"wrapped generator {type(self.generator).__name__} has no `spatial_stats`"
            )
        z = self._resolve_conditioning(list(drone_names), kwargs)
        return fn(rps, rel_pos, z=z, **kwargs)

    def spectral_stats(
        self,
        rps: Any,
        rel_pos: Any,
        drone_names: list[str],
        **kwargs: Any,
    ) -> Any:
        """Conditioned passthrough to the generator's distributional prediction
        (see ``models.generative`` ``spectral_stats``), so a codebook-conditioned
        model can be trained by :class:`losses.SpectralLikelihoodLoss`."""
        fn = getattr(self.generator, "spectral_stats", None)
        if fn is None:
            raise TypeError(
                f"wrapped generator {type(self.generator).__name__} has no `spectral_stats`"
            )
        z = self._resolve_conditioning(list(drone_names), kwargs)
        out = dict(fn(rps, rel_pos, z=z, **kwargs))
        cal = self._resolve_calibration(list(drone_names), self._n_mics(rel_pos))
        if cal is not None:
            gain, power = cal
            out["coherent"] = self._eq_filter(
                out["coherent"] * gain.unsqueeze(-1), list(drone_names)
            )
            out["noise_psd"] = out["noise_psd"] * power[:, :, None, None]
        out["noise_psd"] = self._add_noise_floor(out["noise_psd"], list(drone_names))
        return out

    def amp_stats(
        self,
        rps: Any,
        rel_pos: Any,
        drone_names: list[str],
        **kwargs: Any,
    ) -> Any:
        """Conditioned passthrough to the generator's AMPLITUDE-envelope
        prediction (see
        :meth:`models.generative.PositionalHarmonicNoiseGen.amp_stats`), with the
        absolute-level calibration applied — the training path of the
        amplitude-target objective (``losses.AmplitudeTargetLoss``)."""
        fn = getattr(self.generator, "amp_stats", None)
        if fn is None:
            raise TypeError(f"wrapped generator {type(self.generator).__name__} has no `amp_stats`")
        z = self._resolve_conditioning(list(drone_names), kwargs)
        # The OU linewidth never reaches this path: `amp_stats` reads the
        # amplitude network's output, which is computed BEFORE the jitter is
        # applied to the carrier. Dropping the resolved sigma here keeps that
        # explicit instead of letting it surface as an unexpected-keyword error.
        kwargs.pop("rps_jitter_sigma", None)
        out = dict(fn(rps, rel_pos, z=z, **kwargs))
        cal = self._resolve_calibration(list(drone_names), self._n_mics(rel_pos))
        if cal is not None:
            gain, power = cal
            out["amp"] = out["amp"] * gain[:, :, None, None, None]
            out["noise_psd"] = out["noise_psd"] * power[:, :, None, None]
        if self.mic_eq is not None:
            # The frequency-dependent half of the propagation law, evaluated at
            # each cell's own frequency f = k * rps_r(t) — the term a per-mic
            # SCALAR cannot express. Shared across rotors: room response and
            # capsule sensitivity belong to the receiver, not to the source.
            eq = cast("MicEQ", self.mic_eq)
            names = list(drone_names)
            n_mics = int(out["amp"].shape[1])
            out["amp"] = out["amp"] * eq.gain(out["freq"], names, n_mics=n_mics)
            # The knot curve rides out as a prediction so its curvature penalty
            # is an ordinary composite-loss term (losses.SmoothnessPenalty).
            out["mic_eq"] = eq.curve(names, n_mics=n_mics)
        out["noise_psd"] = self._add_noise_floor(out["noise_psd"], list(drone_names))
        return out


def build_noise_gen_model(
    model_name: str,
    *,
    sample_rate: int = 16000,
    n_harmonics: int = 100,
    use_diff_noise: bool = True,
    cond_dim: int = 0,
    drone_names: list[str] | None = None,
    use_random_phases: bool = False,
    rps_jitter_sigma: float = 0.0,
    rps_jitter_tau: float = 0.05,
    learn_rps_jitter_sigma: bool = False,
    z_noise_std: float = 0.0,
    film_spectral_norm: bool = False,
    silence_fade_rps: float = 10.0,
    per_rotor_deltas: bool = False,
    n_rotors: int = 4,
    wind_uniform_exposure: bool = False,
    amp_calibration: bool = False,
    n_mics: int = 8,
    noise_floor_bands: int = 0,
    mic_eq_knots: int = 0,
    eq_f_min: float = 20.0,
    eq_f_max: float = 8000.0,
) -> nn.Module:
    """Construct a noise-generation model by name (``NOISE_GEN_MODEL_REGISTRY``).

    Verbatim port of the former ``train_noise_generation.py::get_model``,
    plus ``drone_names``: when ``cond_dim > 0``, the returned model is a
    :class:`_CodebookConditionedNoiseGen` wrapping the generator and a fresh
    ``models.generative.codebook.DroneCodebook(cond_dim, names=drone_names)`` —
    see that class's docstring for why the codebook now lives inside the
    model rather than external to it. ``cond_dim == 0`` (single-drone,
    unconditioned) returns the bare generator, ``drone_names`` ignored.

    The three emitter augmentation knobs are forwarded to the wrapped
    :class:`~models.generative.HarmonicNoiseGenNew` (via
    :class:`~models.generative.PositionalHarmonicNoiseGen`'s ``**kwargs``):
    ``use_random_phases`` (STFT-phase scrambling of the harmonic bank) and the
    ``rps_jitter_sigma``/``rps_jitter_tau`` Ornstein-Uhlenbeck RPS perturbation
    (``sigma`` in rev/s, ``tau`` in seconds; calibrated by
    ``scripts/calibrate_rps_jitter.py``). All are training-time augmentations
    (off at eval unless explicitly overridden per-call).

    Two opt-in **latent-space regularisers** smooth the decoder around the (few)
    codebook codes, so later vicinal sampling / interpolation in z-space behaves:

    - ``z_noise_std``: vicinal conditioning noise — relative-scale Gaussian
      perturbation of the code ``z`` during training only; see
      :class:`_CodebookConditionedNoiseGen` (requires ``cond_dim > 0``;
      recommended 0.1).
    - ``film_spectral_norm``: wraps the emitter's FiLM generator Linear
      (``z -> (gamma, beta)``) in ``torch.nn.utils.parametrizations.spectral_norm``,
      bounding the Lipschitz constant of the z-conditioning path. **Changes
      state-dict keys** — new-training-only, not loadable into/from plain
      checkpoints.

    ``learn_rps_jitter_sigma`` promotes the OU linewidth from the single global
    ``rps_jitter_sigma`` scalar to a **learnable per-drone** parameter (one per
    codebook entry, initialised from ``rps_jitter_sigma`` and fit jointly). The
    per-drone raws live on the :class:`_CodebookConditionedNoiseGen` wrapper (see
    its docstring); ``rps_jitter_tau`` stays shared. Requires ``cond_dim > 0``.
    Adds new state-dict keys (``log_jitter_sigma.*``) — not loadable into/from a
    fixed-sigma checkpoint, so train from scratch (or ``strict=False``).

    ``amp_calibration`` adds the absolute-level gains the amplitude-target
    objective needs (per-drone global, per-drone per-microphone, and a separate
    power-domain constant for the broadband branch) — see
    :class:`_CodebookConditionedNoiseGen`. ``n_mics`` sizes the per-microphone
    vectors (8 = the DREGON array, the widest in use). ``noise_floor_bands`` (0 =
    off) additionally gives the broadband branch a static per-microphone,
    per-band floor with that many bands — set it to the emitter's ``noise_amps``
    band count (60 by default); see the class docstring for the measurement that
    motivates it. All of these require ``cond_dim > 0`` and add state-dict keys,
    so they are new-training-only.

    ``mic_eq_knots`` (0 = off, requires ``amp_calibration``) upgrades the
    coherent branch's per-microphone term from a frequency-FLAT scalar to a
    smooth learnable magnitude response ``EQ_c(f)`` — ``mic_eq_knots`` control
    points, log-spaced between ``eq_f_min`` and ``eq_f_max``, one curve per
    (rig, microphone), shared across rotors. It REPLACES ``log_mic_gain``
    (a flat EQ is that scalar), which is why the two are never both built. See
    :class:`models.generative.propagation.MicEQ`.
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
        use_random_phases=use_random_phases,
        rps_jitter_sigma=rps_jitter_sigma,
        rps_jitter_tau=rps_jitter_tau,
        film_spectral_norm=film_spectral_norm,
        silence_fade_rps=silence_fade_rps,
        # Only the wind composite accepts this; it is the wake-model control
        # (uniform per-mic exposure, same capacity, no geometry gating).
        **({"wind_uniform_exposure": True} if wind_uniform_exposure else {}),
    )
    if cond_dim <= 0:
        if z_noise_std > 0.0:
            raise ValueError("z_noise_std > 0 requires cond_dim > 0 (a conditioning code)")
        if learn_rps_jitter_sigma:
            raise ValueError("learn_rps_jitter_sigma requires cond_dim > 0 (per-drone codebook)")
        return generator
    if not drone_names:
        raise ValueError("cond_dim > 0 requires drone_names (DroneCodebook keys)")

    from models.generative.codebook import DroneCodebook

    codebook = DroneCodebook(cond_dim, names=list(drone_names))
    return _CodebookConditionedNoiseGen(
        generator,
        codebook,
        z_noise_std=z_noise_std,
        learn_rps_jitter_sigma=learn_rps_jitter_sigma,
        rps_jitter_sigma_init=rps_jitter_sigma,
        per_rotor_deltas=per_rotor_deltas,
        cond_dim=cond_dim,
        n_rotors=n_rotors,
        amp_calibration=amp_calibration,
        n_mics=n_mics,
        noise_floor_bands=noise_floor_bands,
        mic_eq_knots=mic_eq_knots,
        eq_f_min=eq_f_min,
        eq_f_max=eq_f_max,
    )


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


# ─── Legacy (ZFTurbo-style) model registry ────────────────────────────────────
#
# Dict-ified from the former ``utils.build_model_from_config`` if/elif chain
# (docs/refactor-2026-08-plan.md § "Phase 3" item 1), which made ``utils`` a
# leaf package. Each builder takes the ZFTurbo-style config tree (``audio`` /
# ``model`` / ``training`` sections) and keeps its model import LAZY, exactly
# as the chain did — these stacks are heavy and at most one is needed per
# process. The 13 dead chain branches (mdx23c, segm_models, torchseg,
# mel_band_roformer, swin_upernet, bandit, bandit_v2, scnet_unofficial, scnet,
# apollo, bs_mamba2, experimental_mdx23c_stht, dprnn — zero ``conf/`` /
# ``scripts/`` / ``tests/`` references; implementation modules already
# deleted) were dropped rather than ported.


def _roformer_kwargs(config: DictConfig) -> dict[str, Any]:
    """``config.model`` as plain kwargs for (Mel)BandRoformer, restoring the
    tuple type of ``freqs_per_bands`` / ``multi_stft_resolutions_window_sizes``.
    The legacy YAML tagged these ``!!python/tuple``; BSRoformer's beartype hints
    require ``tuple[int, ...]``, but container coercion turns them into lists."""
    mp = cast(dict[str, Any], OmegaConf.to_container(config.model, resolve=True))
    for k in ("freqs_per_bands", "multi_stft_resolutions_window_sizes"):
        v = mp.get(k)
        if isinstance(v, list):
            mp[k] = tuple(v)
    return mp


def _build_legacy_htdemucs(config: DictConfig) -> nn.Module:
    from models.demucs4ht import get_model

    return get_model(config)


def _build_legacy_edge_bs_rof(config: DictConfig) -> nn.Module:
    from models.edge_bs_rof import BSRoformer

    return BSRoformer(**_roformer_kwargs(config))


def _build_legacy_dcunet(config: DictConfig) -> nn.Module:
    from models.dcunet import DCUNet

    return DCUNet(cast(dict[str, Any], OmegaConf.to_container(config, resolve=True)))


def _build_legacy_dcunet_refactored(config: DictConfig) -> nn.Module:
    # RPS conditioning is decoder-side only.
    from models.dcunet_refactored import DCUNetRefactored

    return DCUNetRefactored(config)


def _build_legacy_dccrn(config: DictConfig) -> nn.Module:
    from models.dccrn import DCCRN

    return DCCRN(config)


def _build_legacy_dccrn_refactored(config: DictConfig) -> nn.Module:
    # RPS conditioning is decoder-side only.
    from models.dcunet_refactored import DCCRNRefactored

    return DCCRNRefactored(config)


def _build_legacy_rps_predictor(config: DictConfig) -> nn.Module:
    # Standalone RPS predictor (real-valued encoder on log-mag spectrograms).
    from models.rps_predictor import SimpleConv as RPSPredictor

    return RPSPredictor(
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        num_rotors=getattr(config, "num_rotors", 4),
    )


def _build_legacy_dptnet(config: DictConfig) -> nn.Module:
    from models.dptnet.dpt_net import DPTNet

    return DPTNet(config)


def _build_legacy_diffusion_buffer(config: DictConfig) -> nn.Module:
    from models.diffusion_buffer import DiffusionBufferModel

    return DiffusionBufferModel(config)


def _build_legacy_sgmse(config: DictConfig) -> nn.Module:
    from models.sgmse import SGMSEModel

    return SGMSEModel(config)


LEGACY_MODEL_BUILDERS: dict[str, Callable[[DictConfig], nn.Module]] = {
    "htdemucs": _build_legacy_htdemucs,
    "edge_bs_rof": _build_legacy_edge_bs_rof,
    "dcunet": _build_legacy_dcunet,
    "dcunet_refactored": _build_legacy_dcunet_refactored,
    "dccrn": _build_legacy_dccrn,
    "dccrn_refactored": _build_legacy_dccrn_refactored,
    "rps_predictor": _build_legacy_rps_predictor,
    "dptnet": _build_legacy_dptnet,
    "diffusion_buffer": _build_legacy_diffusion_buffer,
    "sgmse": _build_legacy_sgmse,
}


def _dispatch_legacy(model_type: str, config: DictConfig) -> nn.Module:
    try:
        builder = LEGACY_MODEL_BUILDERS[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown legacy model type {model_type!r}; "
            f"choose one of {sorted(LEGACY_MODEL_BUILDERS)}"
        ) from None
    return builder(config)


def build_legacy_model(model_type: str, config_path: str) -> nn.Module:
    """Build a model through the legacy registry from a ZFTurbo-style YAML file.

    In the unified framework, task/data parameters come from the Hydra
    ``conf/`` tree, not the legacy YAML; the legacy YAML is only consulted for
    the architecture hyperparameters the model class itself needs (e.g.
    DCUNet's ``audio.n_fft``).
    """
    try:
        cfg = OmegaConf.load(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}") from None
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Expected a mapping config at {config_path}, got {type(cfg).__name__}")
    return _dispatch_legacy(model_type, cfg)


def build_legacy_inline(model_type: str, config: Any) -> nn.Module:
    """Build a legacy-registry model from an **inline** config (the Hydra-native
    replacement for ``build_legacy_model``'s file + ``legacy_config_path``).

    ``config`` is the ZFTurbo-style tree (``audio`` / ``model`` / ``training``
    sections) inlined directly into a ``conf/model/*.yaml`` under ``params`` —
    identical content to the former ``configs/*.yaml`` file, just no separate
    file. Routes through the exact same :data:`LEGACY_MODEL_BUILDERS` dispatch
    as the file-based path, so the resulting module is bit-for-bit identical
    to the legacy build."""
    cfg = config if isinstance(config, DictConfig) else OmegaConf.create(config)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"inline model config must be a mapping, got {type(cfg).__name__}")
    return _dispatch_legacy(model_type, cfg)


# ─── Unified listing ──────────────────────────────────────────────────────────

# Direct plain-kwargs factory functions referenced by ``conf/model/*.yaml``
# ``_target_`` entries that do not route through any registry dict above.
# Values are the dotted paths as they appear in the YAML; kept as strings so
# listing them never triggers the (heavy) module imports.
DIRECT_FACTORY_TYPES: dict[str, str] = {
    "tfgridnet": "models.tfgridnet.build_tfgridnet",
    "mpsenet": "models.mpsenet.build_mpsenet",
    "htdemucs_ft": "models.htdemucs_ft.build_htdemucs_ft",
}


def model_types() -> dict[str, dict[str, str]]:
    """Merged name listing over every model registry in the project.

    Returns ``{name: {"kind": ..., "ref": ...}}`` where ``kind`` is one of
    ``"rps"`` / ``"legacy"`` / ``"noise_gen"`` / ``"factory"`` and ``ref`` is
    the dotted path of the registry dict (or factory function) that builds the
    model. Pure dict merge — no model imports happen at call time.
    """
    listing: dict[str, dict[str, str]] = {}
    for name in RPS_MODEL_REGISTRY:
        listing[name] = {"kind": "rps", "ref": "models.registry.RPS_MODEL_REGISTRY"}
    for name in LEGACY_MODEL_BUILDERS:
        listing[name] = {"kind": "legacy", "ref": "models.registry.LEGACY_MODEL_BUILDERS"}
    for name in NOISE_GEN_MODEL_REGISTRY:
        listing[name] = {"kind": "noise_gen", "ref": "models.registry.NOISE_GEN_MODEL_REGISTRY"}
    for name, ref in DIRECT_FACTORY_TYPES.items():
        listing[name] = {"kind": "factory", "ref": ref}
    return listing


__all__ = [
    "RPS_MODEL_REGISTRY",
    "NOISE_GEN_MODEL_REGISTRY",
    "LEGACY_MODEL_BUILDERS",
    "DIRECT_FACTORY_TYPES",
    "model_types",
    "build_model",
    "build_legacy_model",
    "build_legacy_inline",
    "get_rps_model",
    "build_noise_gen_model",
    "build_noise_gen_loss",
]
