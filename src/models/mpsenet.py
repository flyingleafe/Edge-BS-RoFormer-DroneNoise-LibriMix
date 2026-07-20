"""MP-SENet generator — magnitude + phase speech enhancement network.

Faithful PyTorch reimplementation of the **generator** of MP-SENet
(Lu et al., *"MP-SENet: A Speech Enhancement Model with Parallel Denoising of
Magnitude and Phase Spectra"*, Interspeech 2023), following the reference
repository ``github.com/yxlu-0102/MP-SENet`` (``models/model.py`` +
``models/conformer.py``).

Pipeline (all internal, so the model is a drop-in time-domain SE module for
``tasks.codecs.SpeechEnhancementCodec`` — waveform in, waveform out):

    x  ->  STFT (n_fft=400, hop=100, win=400, center)  ->  |X|**0.3 , angle(X)
        ->  DenseEncoder (2->64 ch, freq 201->101)
        ->  4x TSConformerBlock (time-conformer then freq-conformer)
        ->  MaskDecoder  : sigmoid mask * noisy_mag   -> denoised_mag
            PhaseDecoder : atan2(x_i, x_r)            -> denoised_pha
        ->  uncompress mag (** 1/0.3), complex, iSTFT(length=T)  ->  waveform

Deviations from the reference (documented, all param-count-identical):

* **Generator only — no MetricGAN discriminator.** The reference trains the
  generator adversarially against a metric discriminator; the generator is a
  complete standalone enhancement network and is all that this task's training
  loop (a plain reconstruction/spectral loss) needs. The discriminator, the
  multi-term GAN/consistency losses, and the PCS post-filter are not ported.
* **Multi-head attention uses ``batch_first=True``** so self-attention runs over
  the intended sequence axis (time for the time-conformer, frequency for the
  freq-conformer). The reference constructs ``nn.MultiheadAttention`` with the
  default ``batch_first=False`` while feeding it ``(batch, seq, feat)`` tensors,
  which makes attention mix across the batch/other axis — an unintended quirk
  that is harmful when training from scratch (as we do here — no pretrained
  MP-SENet weights are loaded). Same parameters, corrected wiring.
* iSTFT is called with ``length=T`` so the output length exactly matches the
  input, independent of centering/framing.

Reference: Lu, Ai, Ling (2023). Reimplemented here as SE baseline ``f1_mpsenet``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MPSENet", "build_mpsenet"]


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Symmetric 1-D 'same' padding for an odd kernel."""
    return (kernel_size * dilation - dilation) // 2


class LearnableSigmoid2d(nn.Module):
    """Per-frequency learnable-slope sigmoid: ``beta * sigmoid(slope * x)``.

    ``slope`` is a learnable ``(dim, 1)`` parameter (init 1) broadcast over the
    ``(B, F, T)`` mask, so each frequency bin gets its own slope. Output range
    ``[0, beta]`` (``beta=2`` lets the mask exceed 1, as in the reference).
    """

    def __init__(self, dim: int, beta: float = 2.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T); slope: (F, 1) -> broadcasts over batch and time.
        return self.beta * torch.sigmoid(self.slope * x)


class DenseBlock(nn.Module):
    """Dilated dense block over ``(B, C, T, F)`` (depth 4, dilations 1/2/4/8).

    Kernel ``(2, 3)`` in (time, freq). Time is padded *causally* on the left by
    ``dilation`` (``ConstantPad2d((1, 1, dil, 0))``) and freq symmetrically by 1,
    so every conv preserves ``(T, F)`` and the running dense concat stays
    shape-compatible.
    """

    def __init__(
        self, dense_channel: int, kernel_size: tuple[int, int] = (2, 3), depth: int = 4
    ) -> None:
        super().__init__()
        self.depth = depth
        self.pads = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i in range(depth):
            dil = 2**i
            self.pads.append(nn.ConstantPad2d((1, 1, dil, 0), value=0.0))
            self.convs.append(
                nn.Conv2d(
                    dense_channel * (i + 1),
                    dense_channel,
                    kernel_size,
                    dilation=(dil, 1),
                )
            )
            self.norms.append(nn.InstanceNorm2d(dense_channel, affine=True))
            self.acts.append(nn.PReLU(dense_channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out = x
        for i in range(self.depth):
            out = self.pads[i](skip)
            out = self.convs[i](out)
            out = self.norms[i](out)
            out = self.acts[i](out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    """2->``dense_channel`` channels, then halve the frequency axis (201->101)."""

    def __init__(self, dense_channel: int, in_channel: int = 2) -> None:
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class SPConvTranspose2d(nn.Module):
    """Sub-pixel (pixel-shuffle) transposed conv that upsamples freq by ``r``."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], r: int = 1
    ) -> None:
        super().__init__()
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        out = self.conv(x)
        b, c, t, f = out.shape
        out = out.view(b, self.r, c // self.r, t, f)
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view(b, c // self.r, t, f * self.r)
        return out


class MaskDecoder(nn.Module):
    """Magnitude-mask branch: DenseBlock -> upsample freq -> LearnableSigmoid2d."""

    def __init__(
        self, dense_channel: int, n_freq: int, beta: float = 2.0, out_channel: int = 1
    ) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.Conv2d(dense_channel, out_channel, (1, 2)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(n_freq, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # (B, F, T)
        return self.lsigmoid(x)


class PhaseDecoder(nn.Module):
    """Phase branch: DenseBlock -> upsample freq -> atan2(x_i, x_r)."""

    def __init__(self, dense_channel: int, out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x.permute(0, 3, 2, 1).squeeze(-1)  # (B, F, T)


# ── Conformer ────────────────────────────────────────────────────────────────


class FeedForwardModule(nn.Module):
    """Macaron feed-forward: LN -> Linear -> SiLU -> Linear (inner = dim*mult)."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffm(x)


class AttentionModule(nn.Module):
    """LN -> multi-head self-attention (batch_first — see module deviations)."""

    def __init__(self, dim: int, n_head: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        out, _ = self.attn(x, x, x)
        return out


class ConformerConvModule(nn.Module):
    """LN -> pointwise -> GLU -> depthwise(31) -> BN -> SiLU -> pointwise."""

    def __init__(
        self, dim: int, expansion_factor: int = 2, kernel_size: int = 31, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim * expansion_factor
        self.layernorm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            inner_dim, inner_dim, kernel_size, padding=_get_padding(kernel_size), groups=inner_dim
        )
        self.batchnorm = nn.BatchNorm1d(inner_dim)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x = self.layernorm(x)
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)  # (B, N, C)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """Macaron conformer block: 0.5*FFN, MHSA, ConvModule, 0.5*FFN, LN."""

    def __init__(
        self,
        dim: int,
        n_head: int = 4,
        ffm_mult: int = 4,
        ccm_expansion_factor: int = 2,
        ccm_kernel_size: int = 31,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=dropout)
        self.attn = AttentionModule(dim, n_head, dropout=dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        return self.post_norm(x)


class TSConformerBlock(nn.Module):
    """Two-stage conformer over ``(B, C, T, F)``: time axis then frequency axis."""

    def __init__(self, dense_channel: int, n_head: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.time_conformer = ConformerBlock(
            dim=dense_channel, n_head=n_head, ccm_kernel_size=31, dropout=dropout
        )
        self.freq_conformer = ConformerBlock(
            dim=dense_channel, n_head=n_head, ccm_kernel_size=31, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        # Attention over time: fold (batch, freq) together, sequence = time.
        xt = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        xt = self.time_conformer(xt) + xt
        # Attention over frequency: fold (batch, time) together, sequence = freq.
        xf = xt.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        xf = self.freq_conformer(xf) + xf
        return xf.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()


class MPSENet(nn.Module):
    """MP-SENet generator (waveform -> waveform speech enhancement).

    Args:
        n_fft, hop_size, win_size: STFT framing (defaults 400/100/400 @ 16 kHz).
        dense_channel: encoder/decoder channel width (default 64).
        num_tsconformers: number of TSConformerBlocks (default 4).
        compress_factor: power-law magnitude compression exponent (default 0.3).
        beta: LearnableSigmoid2d ceiling for the magnitude mask (default 2.0).
        conformer_dropout: dropout inside each conformer (default 0.2).
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_size: int = 100,
        win_size: int = 400,
        dense_channel: int = 64,
        num_tsconformers: int = 4,
        compress_factor: float = 0.3,
        beta: float = 2.0,
        conformer_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        n_freq = n_fft // 2 + 1

        self.register_buffer("window", torch.hann_window(win_size), persistent=False)

        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.ts_conformers = nn.ModuleList(
            [
                TSConformerBlock(dense_channel, n_head=4, dropout=conformer_dropout)
                for _ in range(num_tsconformers)
            ]
        )
        self.mask_decoder = MaskDecoder(dense_channel, n_freq, beta=beta)
        self.phase_decoder = PhaseDecoder(dense_channel)

    def _stft(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.window
        assert isinstance(window, torch.Tensor)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=window.to(y.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        mag = torch.abs(spec)
        pha = torch.angle(spec)
        mag = torch.pow(mag, self.compress_factor)
        return mag, pha

    def _istft(self, mag: torch.Tensor, pha: torch.Tensor, length: int) -> torch.Tensor:
        window = self.window
        assert isinstance(window, torch.Tensor)
        mag = torch.pow(mag, 1.0 / self.compress_factor)
        spec = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        return torch.istft(
            spec,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=window.to(mag.device),
            center=True,
            normalized=False,
            length=length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) from the SpeechEnhancementCodec (mono).
        if x.dim() == 3:
            x = x.squeeze(1)
        length = x.shape[-1]

        noisy_mag, noisy_pha = self._stft(x)  # each (B, F, T)

        # -> (B, 1, T, F) so channel=1, time and freq are the spatial axes.
        mag_in = noisy_mag.unsqueeze(1).permute(0, 1, 3, 2)
        pha_in = noisy_pha.unsqueeze(1).permute(0, 1, 3, 2)
        feat = torch.cat([mag_in, pha_in], dim=1)  # (B, 2, T, F)

        feat = self.dense_encoder(feat)
        for block in self.ts_conformers:
            feat = block(feat)

        mask = self.mask_decoder(feat)  # (B, F, T)
        denoised_mag = noisy_mag * mask
        denoised_pha = self.phase_decoder(feat)  # (B, F, T)

        wav = self._istft(denoised_mag, denoised_pha, length)  # (B, T)
        return wav


def build_mpsenet(**params: object) -> nn.Module:
    """Factory for the ``_target_`` config path: ``build_mpsenet(**params)``."""
    return MPSENet(**params)  # type: ignore[arg-type]
