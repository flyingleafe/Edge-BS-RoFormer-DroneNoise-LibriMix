"""hFT-Transformer (Toyama et al., ISMIR 2023) as a rotor-rate salience model.

WHY THIS ARCHITECTURE IS THE INTERESTING ONE. HarmoF0 and HPPNet carry an
explicit harmonic prior (a shift on a log axis) that this project replaces with
a gather. hFT carries NO harmonic prior at all — and it does not need one,
because its decoder already holds ONE OUTPUT TOKEN PER NOTE and lets that token
CROSS-ATTEND to the frequency tokens, learning from data which bins belong to
which note. Its own comment names the tensor:

    attention = [batch_size, n_frame, n_heads, n_note, n_bin]

That is a per-hypothesis, learned read of the spectrum. The one substitution
this package makes turns it into a per-hypothesis, COMPUTED read:

    output tokens  = 88 piano notes        ->  ``n_grid`` CANDIDATE RATES
    key/value set  = all ``n_bin`` bins    ->  the K harmonics of THAT rate,
                                               gathered at ``k * r`` from the
                                               LINEAR STFT by ``CombGather``

The gather is therefore used as a STRUCTURED SPARSITY PRIOR on hFT's existing
cross-attention, not as a new organ bolted on. Each rate token attends only to
the bins its own hypothesis predicts; nothing else is visible to it.

WHY THAT IS ALSO WHAT MAKES THE MODEL AFFORDABLE. hFT's ``n_bin`` is 256
because its input is a log-scale spectrogram. `docs/harmonic-ports-design.md`
rejects a log axis here (the separation-to-bandwidth ratio of two rotors ``D``
apart is ``D / (r * (2^(1/B) - 1))``, in which ``k`` cancels, so a rotor pair is
resolved at every harmonic or at none) and that conclusion is settled. The
linear STFT this port must use has 2049 bins, so hFT's frequency SELF-attention
would be 2049^2 = 4.20e6 attention entries per frame, and its decoder's
cross-attention 300 x 2049 = 6.15e5. The gather replaces both with
300 rates x 32 harmonics = 9.6e3 reads per frame — 437x fewer than the
self-attention, 64x fewer than the dense cross-attention.

WHAT IS KEPT FROM hFT
  * The decoder: learned per-output-token embeddings as the initial queries,
    a stack of cross-attention layers onto the evidence, then the SAtime stage
    (self-attention along TIME, one sequence per output token), then a linear
    head. Post-LayerNorm residual blocks and the position-wise feedforward are
    hFT's, structurally unchanged.
  * The temporal context convolution in front of the tokens (hFT's
    ``Encoder.conv``).
  * hFT's own hyperparameters as the defaults: ``hid_dim`` 256, ``n_layers`` 3,
    ``n_heads`` 4, ``pf_dim`` 512, ``cnn_channel`` 4.

DELIBERATE DEVIATIONS, each forced and each measured or arithmetic

1. THE FREQUENCY SELF-ATTENTION ENCODER IS GONE. Under hard sparsity a rate
   token can only ever see its own K harmonics, so a global representation of
   the other 2017 bins is never read. Deleting the encoder removes the 4.2e6
   entries per frame that the linear axis makes unaffordable, and removes
   nothing the decoder could have used. (Variant (ii) of the design note —
   a reduced frequency token set plus a learned harmonic BIAS instead of a
   hard mask — is the setting that would restore it; it is not implemented.)

2. THE KEY AND VALUE ARE AFFINE IN THE READING, NOT FULL PROJECTIONS. A
   frequency token in hFT is a ``hid_dim`` vector, so its K/V projections cost
   ``n_bin`` vectors per (batch, frame). Here the number of evidence tokens is
   ``n_grid * k_max`` per frame — 9600 against 256 — and giving each one a
   ``hid_dim`` vector would materialize ``B*T*G*K*hid_dim`` floats: at B=16,
   T=32, G=300, K=32, hid_dim=256 that is 1.26e9 floats, 5.0 GB for ONE tensor.
   It is also unnecessary, because a harmonic reading is a SCALAR (one
   log-elevation), not a learned embedding. So the key of harmonic ``k`` is
   ``Ak[k] + gain[k] * (projection of its C context channels)`` and its value
   is ``(projection of its C context channels) + O[k]``: affine in the reading,
   with a learned per-harmonic-order term. Every contraction then runs over C
   (=4) or K (=32) and the largest tensor is the attention map itself,
   ``B*T*G*K*n_heads``. The attention stays fully data-dependent and fully
   per-harmonic; only the K/V PARAMETERIZATION is rank-restricted.

3. THE TEMPORAL CONTEXT IS A CONVOLUTION, NOT A FLATTENED WINDOW. hFT unfolds
   a 65-frame window per frequency bin and flattens ``cnn_channel * 61``
   features into the token embedding. Here that would multiply the largest
   tensor by 61. A stride-1 convolution of ``cnn_kernel`` taps with
   ``cnn_channel`` channels supplies the local context instead, and the
   long-range time context comes from the SAtime stage, which hFT also has.

4. SAtime RUNS IN BLOCKS. hFT is defined on a fixed ``n_frame``-frame segment.
   Here the clip length is whatever the dataset gives (32 frames at training,
   ~251 at validation), so the time self-attention partitions the axis into
   blocks of ``n_frame`` and attends inside each. That is hFT's own segment
   structure, applied as a partition instead of as an input constraint, and it
   keeps the time attention linear rather than quadratic in clip length.

5. ONLY THE MPE (FRAME) HEAD SURVIVES. hFT emits onset, offset, frame and
   velocity, twice (a frequency-stage output A and a time-stage output B).
   Onset/offset/velocity are piano-specific: a rotor has no note-on, no
   note-off and no MIDI velocity. Only the frame/MPE head is kept, and only its
   time-stage (B) output, because the framework's ``salience_rps`` contract and
   ``losses.SalienceRPSBCELoss`` take exactly ONE map. hFT's two-stage deep
   supervision is therefore not reproduced.

6. LOGITS, NOT SIGMOID. hFT applies a sigmoid inside the model; the
   ``salience_rps`` task expects raw logits.

7. MULTI-PITCH BY MULTI-HOT. One map, four Gaussian-blurred bumps per frame —
   which is what a polyphonic MPE head already is, so this needs no framework
   change. Per-rotor maps would need a permutation-invariant loss that does not
   exist yet.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.comb_salience import CombGather, local_floor_torch
from models.multif0.utils import linear_freq_grid
from models.salience_rps import SalienceRPSPredictor

__all__ = [
    "HFTRPS",
    "HarmonicCrossAttentionLayer",
    "BiasedCrossAttentionLayer",
    "FrequencyEncoder",
    "TimeSelfAttentionLayer",
]


def _block_pad_mask(n_real: int, blk: int, rows: int, device) -> torch.Tensor | None:
    """Key-padding mask for a sequence of ``n_real`` split into ``blk`` blocks.

    ``True`` marks a position that only exists because the axis was padded up to
    a whole number of blocks. Without it the padding is a real key, and the last
    block of an axis that does not divide evenly is attended differently from
    every block that does.
    """
    pad = (-n_real) % blk
    if not pad:
        return None
    mask = torch.zeros(n_real + pad, dtype=torch.bool, device=device)
    mask[n_real:] = True
    return mask.view(-1, blk).repeat(rows, 1)


class PositionwiseFeedforward(nn.Module):
    """hFT's ``PositionwiseFeedforwardLayer``, verbatim."""

    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_2(self.dropout(torch.relu(self.fc_1(x))))


class HarmonicCrossAttentionLayer(nn.Module):
    """hFT's ``DecoderLayer_Zero``/``DecoderLayer`` with a HARD-SPARSE key set.

    ``x`` are the rate tokens ``(N, D)`` with ``N = B*T*G`` flattened, and ``u``
    are the per-harmonic context features ``(N, K, C)`` of the SAME rate. The
    attention that comes out is ``(N, K, n_heads)`` — hFT's
    ``[batch, frame, heads, note, bin]`` with ``bin`` restricted to the K
    harmonics of ``note``'s own hypothesis.

    ``self_attention`` reproduces hFT's ``DecoderLayer``: layers after the first
    let output tokens talk to each other before reading the evidence again. Here
    "each other" means the OTHER CANDIDATE RATES in the same frame, which is the
    explain-away channel — one rotor's comb accounts for lines a competing
    hypothesis would otherwise claim. It runs over the rate axis in blocks
    (``rate_block``) because ``n_grid`` is 300 and a dense 300x300 map per
    (batch, frame) is 1.8e8 entries at B=16, T=32, 4 heads.
    """

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        k_max: int,
        cnn_channel: int,
        self_attention: bool = False,
        rate_block: int = 64,
    ):
        super().__init__()
        if hid_dim % n_heads:
            raise ValueError("hid_dim must divide by n_heads")
        self.hid_dim, self.n_heads = hid_dim, n_heads
        self.head_dim = hid_dim // n_heads
        self.k_max, self.cnn_channel = k_max, cnn_channel
        self.rate_block = int(rate_block)
        self.scale = math.sqrt(self.head_dim)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        # KEY: the query is projected onto C "reading directions" (M), the
        # harmonic's C context channels select among them, and `gain` is the
        # per-(harmonic, head) weight on that data term. `Ak` is the
        # harmonic-IDENTITY key — query-dependent but reading-independent — so
        # a head can prefer, say, harmonics 8-20 regardless of what they read.
        self.key_proj = nn.Parameter(torch.empty(cnn_channel, n_heads, self.head_dim))
        # `key_gain` starts at ONE, not zero. At zero the data term vanishes and
        # so does every gradient reaching `fc_q` and `key_proj` (measured: both
        # were exactly zero after one backward), leaving the queries dead at
        # initialization. At one the attention is mildly data-dependent from the
        # first step while still close to uniform, so an untrained token reads
        # roughly the mean log-elevation over its harmonics — the classical
        # Whittle comb score.
        self.key_gain = nn.Parameter(torch.ones(k_max, n_heads))
        self.key_ident = nn.Parameter(torch.empty(k_max, n_heads, self.head_dim))
        # VALUE: the C context channels projected (N), plus a per-harmonic
        # value offset (O) so attending to a particular order carries
        # information even when its reading is flat.
        self.val_proj = nn.Parameter(torch.empty(cnn_channel, n_heads, self.head_dim))
        self.val_ident = nn.Parameter(torch.zeros(k_max, n_heads, self.head_dim))
        nn.init.normal_(self.key_proj, std=hid_dim**-0.5)
        nn.init.normal_(self.val_proj, std=hid_dim**-0.5)
        nn.init.normal_(self.key_ident, std=hid_dim**-0.5)

        self.self_attn = (
            nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
            if self_attention
            else None
        )
        self.ff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.ln_self = nn.LayerNorm(hid_dim) if self_attention else None
        self.ln_attn = nn.LayerNorm(hid_dim)
        self.ln_ff = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def _rate_self_attention(self, x: torch.Tensor, n_g: int) -> torch.Tensor:
        """``x`` ``(N, D)`` with ``N = B*T*G`` -> blocked self-attention over G."""
        assert self.self_attn is not None
        n, d = x.shape
        rows = n // n_g
        blk = min(self.rate_block, n_g)
        pad = (-n_g) % blk
        y = x.view(rows, n_g, d)
        if pad:
            y = F.pad(y, (0, 0, 0, pad))
        n_blk = y.shape[1] // blk
        y = y.reshape(rows * n_blk, blk, d)
        y, _ = self.self_attn(y, y, y, key_padding_mask=_block_pad_mask(n_g, blk, rows, y.device),
                              need_weights=False)
        y = y.reshape(rows, -1, d)[:, :n_g]
        return y.reshape(n, d)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        key_mask: torch.Tensor,
        any_valid: torch.Tensor,
        n_g: int,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``x`` ``(N, D)``, ``u`` ``(N, K, C)``, ``key_mask`` ``(N, K)`` bool."""
        if self.self_attn is not None:
            assert self.ln_self is not None
            x = self.ln_self(x + self.dropout(self._rate_self_attention(x, n_g)))

        n = x.shape[0]
        q = self.fc_q(x).view(n, self.n_heads, self.head_dim)
        # data term: (N,C,H) query readout, then contract the harmonic's C channels
        q_read = torch.einsum("nhd,chd->nch", q, self.key_proj)
        e = torch.bmm(u, q_read) * self.key_gain  # (N, K, H)
        e = e + torch.einsum("nhd,khd->nkh", q, self.key_ident)
        e = e / self.scale
        # HARD SPARSITY. Out-of-band harmonics are removed from the softmax, not
        # down-weighted: a rate token has no access to them at all.
        e = e.masked_fill(~key_mask.unsqueeze(-1), -1e4)
        attn = torch.softmax(e, dim=1) * any_valid.view(n, 1, 1)
        attn_out = attn if return_attention else None
        attn = self.dropout(attn)

        w = torch.bmm(u.transpose(1, 2), attn)  # (N, C, H)
        o = torch.einsum("nch,chd->nhd", w, self.val_proj)
        o = o + torch.einsum("nkh,khd->nhd", attn, self.val_ident)
        o = self.fc_o(o.reshape(n, self.hid_dim))

        x = self.ln_attn(x + self.dropout(o))
        x = self.ln_ff(x + self.dropout(self.ff(x)))
        return x, attn_out


class TimeSelfAttentionLayer(nn.Module):
    """hFT's SAtime stage: one sequence per output token, along TIME.

    hFT reshapes ``[batch*n_frame, n_note, hid]`` to ``[batch*n_note, n_frame,
    hid]`` and runs plain encoder layers over the frame axis. Identical here,
    with ``n_note`` becoming ``n_grid``.
    """

    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.ln_attn = nn.LayerNorm(hid_dim)
        self.ln_ff = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln_attn(x + self.dropout(a))
        return self.ln_ff(x + self.dropout(self.ff(x)))


class FrequencyEncoder(nn.Module):
    """hFT's ``Encoder_SPEC2MIDI``, on a POOLED linear STFT (variant (ii) only).

    hFT's encoder gives every one of its ``n_bin`` log-frequency bins a token
    and runs self-attention across frequency. On the linear axis this port must
    use, ``n_bin`` is 2049 and that attention is 4.2e6 entries per frame. The
    pooling is what buys it back: contiguous groups of ``2048 / n_bin``
    linear bins are averaged into ``n_bin`` tokens, which at the default 256
    restores hFT's own ``n_bin`` exactly. Pooling is lossy about WHERE inside a
    group a line sits — which is why variant (i), the hard gather, does not
    pool — but variant (ii) does not ask the token to carry that: the position
    information is supplied separately, as the learned bias.
    """

    def __init__(self, n_bin: int, hid_dim: int, n_layers: int, n_heads: int,
                 pf_dim: int, dropout: float, cnn_channel: int, cnn_kernel: int):
        super().__init__()
        self.n_bin = int(n_bin)
        self.ctx = nn.Conv1d(1, cnn_channel, cnn_kernel, padding=cnn_kernel // 2)
        with torch.no_grad():
            self.ctx.weight.zero_()
            self.ctx.weight[0, 0, cnn_kernel // 2] = 1.0
            self.ctx.bias.zero_()
            if cnn_channel > 1:
                nn.init.normal_(self.ctx.weight[1:], std=0.2)
        self.tok_embedding = nn.Linear(cnn_channel, hid_dim)
        self.pos_embedding = nn.Embedding(self.n_bin, hid_dim)
        self.scale = math.sqrt(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TimeSelfAttentionLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        """``(B, n_bin, T)`` log-elevation -> ``(B*T, n_bin, D)`` frequency tokens."""
        b, f, t = zf.shape
        u = self.ctx(zf.reshape(b * f, 1, t))                       # (B*F, C, T)
        u = u.view(b, f, -1, t).permute(0, 3, 1, 2)                 # (B, T, F, C)
        x = self.tok_embedding(u.reshape(b * t, f, -1)) * self.scale
        x = self.dropout(x + self.pos_embedding.weight.unsqueeze(0))
        for layer in self.layers:
            x = layer(x)
        return x


class BiasedCrossAttentionLayer(nn.Module):
    """hFT's ``DecoderLayer``, unchanged, plus a learned HARMONIC BIAS.

    Variant (ii) of `docs/harmonic-ports-design.md`: keep the attention over a
    reduced frequency token set and add a learned bias favouring each rate's
    own harmonics, instead of masking everything else away. The bias is
    ``(n_heads, n_grid, n_bin)`` and is initialized from the SAME read positions
    ``CombGather`` computes — ``bias_scale * log1p(number of harmonics of rate g
    falling in token j)`` — so the layer starts where variant (i) is forced to
    stay, and can leave it if a rate is better explained by a bin its own comb
    does not predict.
    """

    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float,
                 n_grid: int, n_bin: int, bias_init: torch.Tensor,
                 self_attention: bool = False, rate_block: int = 64):
        super().__init__()
        if hid_dim % n_heads:
            raise ValueError("hid_dim must divide by n_heads")
        self.hid_dim, self.n_heads = hid_dim, n_heads
        self.head_dim = hid_dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.rate_block = int(rate_block)
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.bias = nn.Parameter(bias_init.unsqueeze(0).repeat(n_heads, 1, 1).clone())
        self.self_attn = (
            nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
            if self_attention
            else None
        )
        self.ff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.ln_self = nn.LayerNorm(hid_dim) if self_attention else None
        self.ln_attn = nn.LayerNorm(hid_dim)
        self.ln_ff = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc: torch.Tensor,
                return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``x`` ``(N, G, D)`` rate tokens, ``enc`` ``(N, S, D)`` frequency tokens."""
        if self.self_attn is not None:
            assert self.ln_self is not None
            n, g, d = x.shape
            blk = min(self.rate_block, g)
            pad = (-g) % blk
            y = F.pad(x, (0, 0, 0, pad)) if pad else x
            y = y.reshape(n * (y.shape[1] // blk), blk, d)
            y, _ = self.self_attn(y, y, y, key_padding_mask=_block_pad_mask(g, blk, n, y.device),
                                  need_weights=False)
            y = y.reshape(n, -1, d)[:, :g]
            x = self.ln_self(x + self.dropout(y))

        n, g, d = x.shape
        s = enc.shape[1]
        q = self.fc_q(x).view(n, g, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.fc_k(enc).view(n, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.fc_v(enc).view(n, s, self.n_heads, self.head_dim).transpose(1, 2)
        e = torch.matmul(q, k.transpose(-1, -2)) / self.scale + self.bias.unsqueeze(0)
        attn = torch.softmax(e, dim=-1)
        out = torch.matmul(self.dropout(attn), v)                   # (N, H, G, dh)
        out = self.fc_o(out.transpose(1, 2).reshape(n, g, d))
        x = self.ln_attn(x + self.dropout(out))
        x = self.ln_ff(x + self.dropout(self.ff(x)))
        return x, (attn if return_attention else None)


class HFTRPS(SalienceRPSPredictor):
    """Audio -> rotor-rate salience logits ``(B, G, T)`` on a LINEAR rate grid.

    The ``salience_rps`` contract: ``forward(audio) -> (B, F, T)`` logits,
    ``outputs_salience = True``, BCE against the RPS-derived target, Hungarian
    tracking at eval. The output axis is declared through ``out_freqs`` — the
    hook :class:`SalienceRPSPredictor` already has for a salience grid that is
    not a CQT — so the target construction, the tracker and ``predict_rps`` are
    reused unchanged.

    Args:
        n_fft, hop_length, sr: the linear STFT. 4096 at 16 kHz is 3.906 Hz per
            bin over a 0.256 s window (`docs/harmonic-ports-design.md`).
        r_lo, r_hi, n_grid: the candidate-rate grid in rev/s, and hence the
            output grid. 0-150 in 300 bins is 0.5017 rev/s per bin.
        k_max, f_max, f_min: harmonics per hypothesis, and the band a harmonic
            must fall inside to enter the attention at all. ``f_min`` matters
            because the grid reaches 0: at 1.5 rev/s all 32 harmonics land in
            the window's DC mainlobe and read far above a floor computed from
            that same mainlobe.
        r_min: candidate rates below this get NO valid harmonics and therefore
            no evidence at all. Defaults to one STFT bin (``sr / n_fft`` =
            3.906 rev/s), which is the rate at which consecutive harmonics stop
            being separate bins. See the comment at the mask construction for
            the measurement that forces it. Those output bins emit a constant,
            which is the honest reading: this front end cannot resolve them.
        hid_dim, n_layers, n_heads, pf_dim, dropout: hFT's transformer widths.
            The defaults are hFT's own.
        cnn_channel, cnn_kernel: the temporal context convolution in front of
            the evidence tokens (hFT's ``Encoder.conv``).
        n_frame: SAtime block length, in frames. hFT's fixed segment length,
            applied here as a partition of an arbitrary-length clip.
        rate_block: block length of the rate-axis self-attention in the
            cross-attention layers after the first.
        n_time_layers: SAtime layers; hFT uses ``n_layers`` of them.
        floor_hz: width of the running-median floor along frequency.
        attn_mode: ``"gather"`` is variant (i) of the design note — hard
            sparsity, a rate token sees ONLY its own K harmonics.
            ``"bias"`` is variant (ii) — hFT's frequency self-attention encoder
            is restored over a POOLED ``n_bin``-token spectrum and the gather
            becomes a learned additive bias on the cross-attention instead of a
            mask. (ii) costs more (its score tensor is ``B*T*G*n_bin*heads``
            against ``B*T*G*k_max*heads``, 8x at the defaults) and can attend
            outside a hypothesis's own comb; (i) cannot.
        n_bin, n_enc_layers, bias_scale: variant (ii) only — the pooled
            frequency token count (256 is hFT's own), the encoder depth, and
            the scale of the log-incidence bias initialization.
    """

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 512,
        num_rotors: int = 4,
        sr: int = 16000,
        r_lo: float = 0.0,
        r_hi: float = 150.0,
        n_grid: int = 300,
        k_max: int = 32,
        f_max: float = 7500.0,
        f_min: float = 30.0,
        r_min: float | None = None,
        hid_dim: int = 256,
        n_layers: int = 3,
        n_time_layers: int | None = None,
        n_heads: int = 4,
        pf_dim: int = 512,
        dropout: float = 0.1,
        cnn_channel: int = 4,
        cnn_kernel: int = 5,
        n_frame: int = 128,
        rate_block: int = 64,
        floor_hz: float = 120.0,
        attn_mode: str = "gather",
        n_bin: int = 256,
        n_enc_layers: int = 3,
        bias_scale: float = 2.0,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        self.sr, self.k_max = int(sr), int(k_max)
        self.hid_dim, self.n_heads = int(hid_dim), int(n_heads)
        self.cnn_channel, self.n_frame = int(cnn_channel), int(n_frame)

        grid = linear_freq_grid(r_lo, r_hi, n_grid)
        self.out_freqs = grid
        self.n_bins = self.n_grid = int(n_grid)
        # torch.stft(center=True) emits n // hop + 1 frames and nothing pools
        # along time, so the salience rate IS the STFT rate.
        self.spec_sr, self.spec_hop = int(sr), int(hop_length)

        if attn_mode not in ("gather", "bias"):
            raise ValueError(f"unknown attn_mode {attn_mode!r}")
        self.attn_mode = attn_mode

        df = float(sr) / float(n_fft)
        self.floor_bins = max(3, int(round(floor_hz / df)) | 1)
        self.gather = CombGather(
            k_max=self.k_max, sr=int(sr), n_fft=int(n_fft), f_max=float(f_max), grid=grid
        )
        fk = (
            torch.arange(1, self.k_max + 1, dtype=torch.float64)[:, None]
            * torch.as_tensor(grid, dtype=torch.float64)[None, :]
        )
        # THE UNRESOLVABLE-RATE CUT. Consecutive harmonics of a rate `r` are
        # `r` Hz apart, so below one STFT bin width they land in the SAME bin
        # and the hypothesis is not a comb at all — every one of its K reads is
        # the same handful of bins. Measured on single-comb clips: without the
        # cut the classical read of a 37 rev/s comb peaks at 1.51 rev/s (whose
        # harmonics 20-32 all sit inside the 37 Hz line's own skirt) and a 45
        # rev/s comb peaks at 1.51 as well; with it, 37, 45, 60, 84.5 and 120
        # all recover to within one grid bin. `f_min` alone does not catch this
        # — those reads are at 30-48 Hz, outside the DC mainlobe.
        r_cut = float(sr) / float(n_fft) if r_min is None else float(r_min)
        self.r_min = r_cut
        grid_t = torch.as_tensor(grid, dtype=torch.float64)[None, :]
        band = (fk >= float(f_min)) & (fk < float(f_max)) & (grid_t >= r_cut)  # (K, G)
        self.register_buffer("band", band, persistent=False)
        self.register_buffer("any_band", band.any(dim=0), persistent=False)  # (G,)
        self.register_buffer("window", torch.hann_window(int(n_fft)), persistent=False)

        # hFT's Encoder.conv, reduced to a stride-1 temporal convolution
        # (deviation 3). Channel 0 is initialized to a delta so it passes the
        # raw log-elevation through: an untrained token then reads the same
        # evidence the classical Whittle scan reads.
        # Variant (ii) has its own copy of this convolution inside
        # `FrequencyEncoder`; building it here too would leave a dead parameter.
        self.ctx: nn.Conv1d | None = None
        if attn_mode == "gather":
            self.ctx = nn.Conv1d(1, self.cnn_channel, cnn_kernel, padding=cnn_kernel // 2)
            with torch.no_grad():
                self.ctx.weight.zero_()
                self.ctx.weight[0, 0, cnn_kernel // 2] = 1.0
                self.ctx.bias.zero_()
                if self.cnn_channel > 1:
                    nn.init.normal_(self.ctx.weight[1:], std=0.2)

        # hFT's `pos_embedding_freq(n_note)`: the output tokens' initial query
        # is a pure learned embedding, one per candidate rate. All data enters
        # through the cross-attention.
        self.rate_embedding = nn.Embedding(self.n_grid, self.hid_dim)
        self.time_embedding = nn.Embedding(self.n_frame, self.hid_dim)
        self.scale_time = math.sqrt(self.hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.encoder: FrequencyEncoder | None = None
        # hFT: DecoderLayer_Zero (cross-attention only) then n_layers-1
        # DecoderLayers (self-attention + cross-attention).
        if attn_mode == "gather":
            self.layers = nn.ModuleList(
                [
                    HarmonicCrossAttentionLayer(
                        self.hid_dim,
                        n_heads,
                        pf_dim,
                        dropout,
                        self.k_max,
                        self.cnn_channel,
                        self_attention=(i > 0),
                        rate_block=rate_block,
                    )
                    for i in range(int(n_layers))
                ]
            )
        else:
            n_f = n_fft // 2 + 1
            self.n_bin = int(n_bin)
            self.pool = (n_f - 1) // self.n_bin
            self.encoder = FrequencyEncoder(
                self.n_bin,
                self.hid_dim,
                int(n_enc_layers),
                n_heads,
                pf_dim,
                dropout,
                self.cnn_channel,
                cnn_kernel,
            )
            # THE BIAS INITIALIZATION. `CombGather` has already computed where
            # harmonic k of rate g reads; pooling maps each read to its token.
            # The incidence count is what a hard mask would keep, so
            # `bias_scale * log1p(incidence)` starts the layer at variant (i)'s
            # preference without variant (i)'s prohibition.
            tok = (self.gather.bin_lo // self.pool).clamp(0, self.n_bin - 1)  # (K, G)
            inc = torch.zeros(self.n_grid, self.n_bin)
            inc.scatter_add_(1, tok.t(), band.t().to(inc.dtype))
            bias0 = float(bias_scale) * torch.log1p(inc)
            self.layers = nn.ModuleList(
                [
                    BiasedCrossAttentionLayer(
                        self.hid_dim,
                        n_heads,
                        pf_dim,
                        dropout,
                        self.n_grid,
                        self.n_bin,
                        bias0,
                        self_attention=(i > 0),
                        rate_block=rate_block,
                    )
                    for i in range(int(n_layers))
                ]
            )
        n_t = int(n_layers if n_time_layers is None else n_time_layers)
        self.time_layers = nn.ModuleList(
            [TimeSelfAttentionLayer(self.hid_dim, n_heads, pf_dim, dropout) for _ in range(n_t)]
        )
        self.fc_mpe = nn.Linear(self.hid_dim, 1)
        self.last_attention: torch.Tensor | None = None

    # ── grid ────────────────────────────────────────────────────────────────

    def grid_params(self) -> dict:
        raise NotImplementedError(
            "HFTRPS has no log-spaced CQT grid; its salience axis is the linear "
            "candidate-rate grid exposed as `out_freqs`."
        )

    def output_freqs(self) -> np.ndarray:
        return np.asarray(self.out_freqs, dtype=np.float64)

    def num_grid_frames(self, n_samples: int) -> int:
        return int(n_samples) // self.spec_hop + 1

    # ── front end ───────────────────────────────────────────────────────────

    def spectrum(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio ``(B, T)`` or ``(B, 1, T)`` -> power spectrogram ``(B, F, T)``."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.spec_hop,
            window=cast(torch.Tensor, self.window).to(audio.dtype),
            center=True,
            return_complex=True,
        )
        return spec.real.pow(2) + spec.imag.pow(2)

    def evidence(self, pw: torch.Tensor) -> torch.Tensor:
        """Power ``(B, F, T)`` -> log-elevation ``(B, K, G, T)``.

        Power and floor are gathered SEPARATELY and combined as
        ``log1p(h / floor)`` — what ``CombScoreHead`` does, and what keeps the
        untrained read comparable with the classical scan. The floor is
        detached: it sets the scale, it is not a parameter. Out-of-band
        harmonics are zeroed here and masked out of the softmax later.
        """
        floor = local_floor_torch(pw, self.floor_bins).detach()
        h = self.gather(pw)
        fh = self.gather(floor).clamp_min(1e-12)
        band = cast(torch.Tensor, self.band).to(h.dtype)[None, :, :, None]
        return torch.log1p(h / fh) * band

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        if self.attn_mode == "bias":
            return self._forward_bias(audio, return_attention)
        z = self.evidence(self.spectrum(audio))  # (B, K, G, T)
        b, k, g, t = z.shape

        # Temporal context, shared across every (harmonic, rate) token.
        assert self.ctx is not None
        u = self.ctx(z.reshape(b * k * g, 1, t))  # (B*K*G, C, T)
        u = u.view(b, k, g, self.cnn_channel, t).permute(0, 4, 2, 1, 3).contiguous()
        u = u.reshape(b * t * g, k, self.cnn_channel)  # (N, K, C)

        band = cast(torch.Tensor, self.band)  # (K, G)
        key_mask = band.t().reshape(1, g, k).expand(b * t, g, k).reshape(-1, k)
        any_valid = (
            cast(torch.Tensor, self.any_band).to(z.dtype).view(1, g).expand(b * t, g).reshape(-1)
        )

        idx = torch.arange(g, device=z.device)
        x = self.rate_embedding(idx).view(1, g, self.hid_dim).expand(b * t, g, self.hid_dim)
        x = self.dropout(x).reshape(-1, self.hid_dim)  # (N, D)

        attn = None
        for i, layer in enumerate(self.layers):
            want = return_attention and i == len(self.layers) - 1
            x, a = layer(x, u, key_mask, any_valid, g, return_attention=want)
            if a is not None:
                attn = a.view(b, t, g, k, self.n_heads).permute(0, 1, 4, 2, 3)
        self.last_attention = attn  # (B, T, H, G, K) — hFT's own layout

        # SAtime: one sequence per (clip, candidate rate) along the frame axis,
        # partitioned into hFT-sized blocks.
        y = x.view(b, t, g, self.hid_dim).permute(0, 2, 1, 3).reshape(b * g, t, self.hid_dim)
        y = self._time_stack(y)
        logits = self.fc_mpe(y).view(b, g, t)
        return logits

    def _forward_bias(self, audio: torch.Tensor, return_attention: bool) -> torch.Tensor:
        """Variant (ii): pooled frequency tokens + a learned harmonic bias."""
        assert self.encoder is not None
        pw = self.spectrum(audio)
        floor = local_floor_torch(pw, self.floor_bins).detach()
        b, n_f, t = pw.shape
        used = self.pool * self.n_bin
        zf = torch.log1p(pw[:, :used] / floor[:, :used].clamp_min(1e-12))
        zf = zf.view(b, self.n_bin, self.pool, t).mean(2)             # (B, n_bin, T)
        enc = self.encoder(zf)                                        # (B*T, n_bin, D)

        g = self.n_grid
        x = self.rate_embedding.weight.unsqueeze(0).expand(b * t, g, self.hid_dim)
        x = self.dropout(x)
        attn = None
        for i, layer in enumerate(self.layers):
            want = return_attention and i == len(self.layers) - 1
            x, a = layer(x, enc, return_attention=want)
            if a is not None:
                attn = a.view(b, t, self.n_heads, g, self.n_bin)
        self.last_attention = attn                                    # (B, T, H, G, n_bin)

        y = x.view(b, t, g, self.hid_dim).permute(0, 2, 1, 3).reshape(b * g, t, self.hid_dim)
        return self.fc_mpe(self._time_stack(y)).view(b, g, t)

    def _time_stack(self, y: torch.Tensor) -> torch.Tensor:
        """``(B*G, T, D)`` -> same, self-attended along T in ``n_frame`` blocks."""
        rows, t, d = y.shape
        blk = min(self.n_frame, t)
        pad = (-t) % blk
        if pad:
            y = F.pad(y, (0, 0, 0, pad))
        n_blk = y.shape[1] // blk
        y = y.reshape(rows * n_blk, blk, d)
        pos = self.time_embedding(torch.arange(blk, device=y.device)).unsqueeze(0)
        y = self.dropout(y * self.scale_time + pos)
        # The last block of a clip whose length does not divide `n_frame` is
        # part padding. Without a mask those positions are real keys carrying a
        # position embedding, so the final frames of a validation clip would be
        # attended differently from any frame ever seen in training — a
        # train/eval mismatch that appears only at the clip lengths that differ.
        mask = None
        if pad:
            mask = torch.zeros(n_blk * blk, dtype=torch.bool, device=y.device)
            mask[t:] = True
            mask = mask.view(n_blk, blk).repeat(rows, 1)
        for layer in self.time_layers:
            y = layer(y, key_padding_mask=mask)
        return y.reshape(rows, n_blk * blk, d)[:, :t]
