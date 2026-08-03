"""TF-GridNet (V1) speech-enhancement model — faithful pure-PyTorch port.

Canonical reference: ESPnet ``espnet2/enh/separator/tfgridnet_separator.py``
(the V1 separator, Wang et al., "TF-GridNet: Integrating Full- and Sub-Band
Modeling for Speech Separation", TASLP 2023). This is a single-source
(``n_srcs=1``) enhancement variant doing **direct complex spectral mapping**
(regress the enhanced real/imag STFT, no mask multiply).

Pipeline (mono, ``M = 1``):
    per-utterance RMS norm → STFT → stack (re, im) as 2 channels → Conv2d+GN
    → ``n_layers`` GridNetBlocks (intra full-band BLSTM over freq, inter
    sub-band BLSTM over time, cross-frame multi-head self-attention over time)
    → ConvTranspose2d back to (re, im) → iSTFT → undo RMS norm.

The two custom norms from the reference are both reproduced verbatim:
``LayerNormalization4D`` (over the channel dim only) and
``LayerNormalization4DCF`` (over channel + frequency).

Deliberate deviations from the ESPnet default constructor, all to fit this
project's native-16 kHz SE contract (``forward(x: (B,1,T)) -> (B,T)``):

* **STFT window** is a *square-root Hann* window (``hann_window(win)**0.5``),
  center=True, rather than ESPnet's plain ``"hann"``. sqrt-Hann gives
  analysis·synthesis = Hann → exact COLA reconstruction for enhancement.
* **STFT size** is native 16 kHz: ``n_fft=512, hop=128, win=512`` (F = 257)
  rather than the reference's tiny ``n_fft=128, stride=64``. All configurable.
* Single mic (``n_imics = 1``) and single source (``n_srcs = 1``); returns a
  bare waveform ``(B, T)`` instead of ESPnet's ``([wav_per_src], ilens, {})``.

Everything else — block structure, unfold/BLSTM/ConvTranspose1d intra/inter
paths, per-head Q/K/V conv stacks with PReLU + LN4DCF, ``E = ceil(qk/F)``,
attention scaled by ``sqrt(flattened Q dim)``, GroupNorm(1) encoder norm — is
a line-for-line port of the reference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TFGridNet", "build_tfgridnet"]


class LayerNormalization4D(nn.Module):
    """LayerNorm over the channel dim of a ``[B, C, T, F]`` tensor."""

    def __init__(self, input_dimension: int, eps: float = 1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = nn.Parameter(torch.ones(*param_size))
        self.beta = nn.Parameter(torch.zeros(*param_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNormalization4D expects a 4D tensor, got {x.ndim}D")
        stat_dim = (1,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B, 1, T, F]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        return ((x - mu_) / std_) * self.gamma + self.beta


class LayerNormalization4DCF(nn.Module):
    """LayerNorm over channel + frequency of a ``[B, C, T, F]`` tensor."""

    def __init__(self, input_dimension: tuple[int, int], eps: float = 1e-5):
        super().__init__()
        if len(input_dimension) != 2:
            raise ValueError("LayerNormalization4DCF needs (channels, freqs)")
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = nn.Parameter(torch.ones(*param_size))
        self.beta = nn.Parameter(torch.zeros(*param_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNormalization4DCF expects a 4D tensor, got {x.ndim}D")
        stat_dim = (1, 3)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B, 1, T, 1]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        return ((x - mu_) / std_) * self.gamma + self.beta


class GridNetBlock(nn.Module):
    """One TF-GridNet block: intra BLSTM, inter BLSTM, cross-frame attention."""

    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        n_freqs: int,
        hidden_channels: int,
        n_head: int = 4,
        approx_qk_dim: int = 512,
        eps: float = 1e-5,
    ):
        super().__init__()
        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        e = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # per-head Q/K channel dim
        if emb_dim % n_head != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by n_head ({n_head})")
        self.attn_conv_q = nn.ModuleList()
        self.attn_conv_k = nn.ModuleList()
        self.attn_conv_v = nn.ModuleList()
        for _ in range(n_head):
            self.attn_conv_q.append(
                nn.Sequential(
                    nn.Conv2d(emb_dim, e, 1),
                    nn.PReLU(),
                    LayerNormalization4DCF((e, n_freqs), eps=eps),
                )
            )
            self.attn_conv_k.append(
                nn.Sequential(
                    nn.Conv2d(emb_dim, e, 1),
                    nn.PReLU(),
                    LayerNormalization4DCF((e, n_freqs), eps=eps),
                )
            )
            self.attn_conv_v.append(
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    nn.PReLU(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                )
            )
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.PReLU(),
            LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C, T, Q]`` (C=emb_dim, T=frames, Q=freqs) → same shape."""
        b, c, old_t, old_q = x.shape
        t = math.ceil((old_t - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        q = math.ceil((old_q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, q - old_q, 0, t - old_t))

        # intra RNN (full-band: over frequency)
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(b * t, c, q)  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, 2H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, 2H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([b, t, c, q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # residual

        # inter RNN (sub-band: over time)
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, Q]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(b * q, c, t)  # [BQ, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BQ, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, 2H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, 2H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
        inter_rnn = inter_rnn.view([b, q, c, t])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # residual

        # cross-frame self-attention (over time, per frequency)
        inter_rnn = inter_rnn[..., :old_t, :old_q]
        batch = inter_rnn

        all_q, all_k, all_v = [], [], []
        for ii in range(self.n_head):
            all_q.append(self.attn_conv_q[ii](batch))  # [B, E, T, Q]
            all_k.append(self.attn_conv_k[ii](batch))  # [B, E, T, Q]
            all_v.append(self.attn_conv_v[ii](batch))  # [B, C/n_head, T, Q]

        qh = torch.cat(all_q, dim=0)  # [n_head*B, E, T, Q]
        kh = torch.cat(all_k, dim=0)
        vh = torch.cat(all_v, dim=0)

        qh = qh.transpose(1, 2).flatten(start_dim=2)  # [n_head*B, T, E*Q]
        kh = kh.transpose(1, 2).flatten(start_dim=2)  # [n_head*B, T, E*Q]
        vh = vh.transpose(1, 2)  # [n_head*B, T, C/n_head, Q]
        v_shape = vh.shape
        vh = vh.flatten(start_dim=2)  # [n_head*B, T, (C/n_head)*Q]
        qk_dim = qh.shape[-1]

        attn_mat = torch.matmul(qh, kh.transpose(1, 2)) / (qk_dim**0.5)  # [n_head*B, T, T]
        attn_mat = F.softmax(attn_mat, dim=2)
        vh = torch.matmul(attn_mat, vh)  # [n_head*B, T, (C/n_head)*Q]

        vh = vh.reshape(v_shape)  # [n_head*B, T, C/n_head, Q]
        vh = vh.transpose(1, 2)  # [n_head*B, C/n_head, T, Q]
        v_c = vh.shape[1]

        batch = vh.view([self.n_head, b, v_c, old_t, -1])  # [n_head, B, C/n_head, T, Q]
        batch = batch.transpose(0, 1)  # [B, n_head, C/n_head, T, Q]
        batch = batch.contiguous().view([b, self.n_head * v_c, old_t, -1])  # [B, C, T, Q]

        batch = self.attn_concat_proj(batch)  # [B, C, T, Q]

        return batch + inter_rnn


class TFGridNet(nn.Module):
    """Single-source TF-GridNet speech enhancer (direct complex spectral mapping)."""

    window: torch.Tensor

    def __init__(
        self,
        n_srcs: int = 1,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_layers: int = 6,
        lstm_hidden_units: int = 192,
        attn_n_head: int = 4,
        attn_approx_qk_dim: int = 512,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        eps: float = 1e-5,
    ):
        super().__init__()
        if n_fft % 2 != 0:
            raise ValueError("n_fft must be even")
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
        n_freqs = n_fft // 2 + 1

        # square-root Hann analysis/synthesis window (COLA under hop = n_fft/4)
        window = torch.hann_window(win_length, periodic=True) ** 0.5
        self.register_buffer("window", window)

        ks, padding = (3, 3), (1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList(
            [
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, n_samples]`` → complex ``[B, F, T]``."""
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
            normalized=False,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """``spec``: complex ``[B, F, T]`` → ``[B, length]``."""
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            length=length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: mono mixture ``(B, 1, T)`` → enhanced waveform ``(B, T)``."""
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, T)
        n_samples = x.shape[-1]

        # per-utterance RMS normalization
        std_ = torch.std(x, dim=1, keepdim=True)  # (B, 1)
        x = x / (std_ + self.eps)

        # Complex-safe AMP boundary: torch.stft/istft have no ComplexHalf
        # kernels under autocast, so the transforms always run in fp32; only
        # the real-valued network body below autocasts. This is what lets the
        # F1 configs set amp=true (bfloat16) instead of full-fp32 training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self._stft(x.float())  # [B, F, T] complex
        spec = spec.transpose(1, 2)  # [B, T, F]

        batch = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, T, F]
        batch = self.conv(batch)  # [B, emb_dim, T, F]

        for block in self.blocks:
            batch = block(batch)  # [B, emb_dim, T, F]

        batch = self.deconv(batch)  # [B, 2, T, F]
        with torch.autocast(device_type=x.device.type, enabled=False):
            batch = batch.float()
            est_spec = torch.complex(batch[:, 0], batch[:, 1])  # [B, T, F]
            est_spec = est_spec.transpose(1, 2).contiguous()  # [B, F, T]
            out = self._istft(est_spec, length=n_samples)  # [B, T]
        out = out * (std_ + self.eps)  # undo RMS normalization
        return out


def build_tfgridnet(**params) -> nn.Module:
    """Factory for the Hydra ``_target_`` path — plain kwargs, see conf/model."""
    return TFGridNet(**params)
