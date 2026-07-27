"""BSRoformerRPS — the Edge-BS-RoFormer trunk adapted to RPS prediction.

The original Edge-BS-RoFormer (Liu et al., Drones 2025; `edge_bs_rof.py`)
claims its rotary time/freq positional embeddings help the axial
transformers *track harmonic lines* for speech enhancement. RPS prediction
is the sharper test of that claim: the target IS the harmonic-line
trajectory. This adaptation keeps the paper's trunk byte-identical —
internal complex STFT (re/im per band ⇒ phase-aware by construction) →
`BandSplit` over the 62-band layout → depth × (time-RoPE transformer +
freq-RoPE transformer) → `final_norm` — and replaces only the per-stem
`MaskEstimator`s with a band-attention pool + linear read-out on the
hop-512 frame grid, so the output contract matches the SimpleConv family:
audio (B, N) → RPS (B, num_rotors, T), T = N // hop + 1.

Trunk reuse is by *composition*: we instantiate a full ``BSRoformer`` with
``num_stems=1, mask_estimator_depth=1`` and drop its ``mask_estimators``
(so checkpoints stay self-describing and the trunk construction code is not
duplicated), then run the trunk portion of its forward here (the STFT /
band-split / axial-transformer plumbing mirrors ``BSRoformer.forward`` up
to the mask-estimation stage). Campaign context:
`docs/experiments/ckla.md` § comparison ladder.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from einops import pack, rearrange, unpack
from torch import Tensor, nn

from models.edge_bs_rof.edge_bs_rof import BSRoformer


class BandAttentionPool(nn.Module):
    """Attention-pool the band axis: (B, T, F_bands, D) → (B, T, D).

    Same idea as the SimpleConv family's ``FrequencyAttentionPool``, applied
    to the roformer's band tokens: a learned query attends over bands per
    frame, so rotor-band evidence is weighted rather than averaged away.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)

    def forward(self, x: Tensor) -> Tensor:
        b, t, f, d = x.shape
        tokens = x.reshape(b * t, f, d)
        q = self.query.expand(b * t, 1, d)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return pooled.reshape(b, t, d)


class BSRoformerRPS(nn.Module):
    """Edge-BS-RoFormer trunk + band-pool RPS head (registry ``edge_bs_rof_rps``).

    Defaults mirror the F1 `a1_edge_bs_rof_fa_rope48` speech-enhancement
    configuration (dim 48, depth 3, RoPE on, flash attention) at 16 kHz /
    n_fft 2048 / hop 512 — the exact trunk whose rotary embeddings carry
    the harmonic-tracking claim.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        dim: int = 48,
        depth: int = 3,
        dim_head: int = 48,
        heads: int = 6,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_rotary_pos: bool = True,
        flash_attn: bool = True,
        pool_heads: int = 4,
        **roformer_kwargs: Any,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors

        core = BSRoformer(
            dim=dim,
            depth=depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            stft_n_fft=n_fft,
            stft_hop_length=hop_length,
            stft_win_length=n_fft,
            mask_estimator_depth=1,
            use_rotary_pos=use_rotary_pos,
            **roformer_kwargs,
        )
        # The head replaces mask estimation entirely; dropping the estimators
        # removes their parameters from the module tree (they would dominate
        # the budget and never receive gradients).
        del core.mask_estimators
        self.core = core

        self.pool = BandAttentionPool(dim, heads=pool_heads)
        self.head = nn.Linear(dim, num_rotors)

    def forward(self, audio: Tensor) -> Tensor:
        """audio (B, N) or (B, 1, N) → (B, num_rotors, T), T = N // hop + 1.

        Mirrors ``BSRoformer.forward`` up to the mask-estimation stage
        (mono path, no checkpointing/skip variants — the F1 config uses
        neither), then pools the band axis and reads out per-frame RPS.
        """
        core = self.core
        if audio.ndim == 3:
            audio = audio.squeeze(1)

        window = core.stft_window_fn(device=audio.device)
        stft_repr = torch.stft(
            audio, **core.stft_kwargs, window=window, return_complex=True, center=True
        )
        stft_repr = torch.view_as_real(stft_repr)  # (B, F, T, 2)
        x = rearrange(stft_repr, "b f t c -> b t (f c)")
        x = core.band_split(x)  # (B, T, F_bands, D)

        for transformer_block in core.layers:
            block = cast(nn.ModuleList, transformer_block)
            time_transformer, freq_transformer = block[0], block[1]

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")

            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        x = core.final_norm(x)  # (B, T, F_bands, D)
        h = self.pool(x)  # (B, T, D)
        return self.head(h).transpose(1, 2)  # (B, R, T)
