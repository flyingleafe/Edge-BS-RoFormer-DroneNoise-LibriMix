"""FKLA RPS model — plain-KLA (real, non-rotating) cross-implementation ablation.

Uses the vendored kla-loglinear flat-KLA layer (``models.fkla.layer``, an
independent implementation of the same KLA substrate CKLA generalizes) inside
EXACTLY the ``SimpleConvV2CKLA`` wrapper (``models.ckla``): same
``stft_mag_if`` front-end, same 6-block ResidualConvBlock2d encoder, same
``FrequencyAttentionPool``, same head wiring (Linear in-proj → blocks →
RMSNorm → linear read-out), same defaults d_model=128, n_layers=2, n_state=16.
The ONLY difference is the temporal mixer: ``FlatKLABlock`` (real OU state,
no complex rotation, learned fold weight) instead of ``CKLABlock``.

Registry key ``simple_conv_v2_fkla``; the companion in-repo control is
``simple_conv_v2_ckla_pnoise_norot`` (CKLA scan with rotation disabled) —
this arm checks the same claim through an independent codebase.
"""

from __future__ import annotations

from torch import Tensor, nn

from models.fkla.layer import FlatKLABlock
from models.rps_predictor import (
    FrequencyAttentionPool,
    ResidualConvBlock2d,
    _remap_legacy_state_dict,
)


class TemporalFKLAHead(nn.Module):
    """Temporal head over pooled trunk features: (B, in_ch, T) → (B, R, T).

    Mirror of ``models.ckla.TemporalCKLAHead`` with ``FlatKLABlock`` mixers:
    Linear in-projection → ``n_layers`` blocks over time → RMSNorm → linear
    read-out. No output activation or scaling (raw per-frame RPS predictions),
    so training configs transfer unchanged.
    """

    def __init__(
        self,
        in_ch: int = 128,
        d_model: int = 128,
        num_rotors: int = 4,
        n_layers: int = 2,
        n_state: int = 16,
        p_init: float = 0.01,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, d_model)
        self.blocks = nn.ModuleList(
            FlatKLABlock(d_model, n_state=n_state, p_init=p_init) for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, num_rotors)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, T) → (B, num_rotors, T)."""
        h = self.in_proj(x.transpose(1, 2))  # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        return self.proj(self.norm(h)).transpose(1, 2)


class FKLARPSModel(nn.Module):
    """``SimpleConvV2CKLA`` wrapper with a ``TemporalFKLAHead``.

    Trunk verbatim from ``SimpleConvV2Transformer``/``SimpleConvV2CKLA``:
    front-end → 6× ``ResidualConvBlock2d`` encoder → ``FrequencyAttentionPool``
    → (B, 128, T). Default front-end ``stft_mag_if`` — identical to the CKLA
    arms, so any delta is attributable to the temporal mixer.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        frontend: nn.Module | None = None,
        frontend_key: str = "stft_mag_if",
        d_model: int = 128,
        n_layers: int = 2,
        n_state: int = 16,
        p_init: float = 0.01,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend(frontend_key, n_fft=n_fft, hop_length=hop_length)
        self.frontend: nn.Module = frontend

        # First block adapts to the front-end's channel count (2 for the
        # default stft_mag_if).
        in_ch = getattr(frontend, "out_channels", 1)
        enc_spec = [
            (in_ch, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, kern, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, kern, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = TemporalFKLAHead(
            in_ch=128,
            d_model=d_model,
            num_rotors=num_rotors,
            n_layers=n_layers,
            n_state=n_state,
            p_init=p_init,
        )

    def forward(self, audio: Tensor) -> Tensor:
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, num_rotors, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap (SimpleConv family
        convention — harmless for FKLA checkpoints, which are all new)."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)
