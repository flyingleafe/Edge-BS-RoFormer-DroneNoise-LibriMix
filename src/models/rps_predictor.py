#!/usr/bin/env python3
"""
RPS predictor model architectures.

Baseline: SimpleConv (from paper)
Variants:
  - SimpleConvV2: residual + SE + attention pooling + BiGRU head
  - SimpleConvWide: wider/deeper, no fancy components
  - SimpleConvTCN: TCN head with dilated convolutions
  - SimpleConvMultiScale: FPN-style multi-scale feature fusion
  - SimpleConvAttnPool: attention-based frequency pooling
  - SimpleConvBiGRU: BiGRU temporal head
  - SimpleConvSENext: squeeze-excitation + residual
  - SimpleConvComplex: complex-valued STFT input with 2-channel real/imag
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Utilities ───────────────────────────────────────────────────────────────


def stft_time_frames(audio_length: int, hop_length: int, n_fft: int) -> int:
    """Number of STFT time frames for a given audio length (with center padding)."""
    return audio_length // hop_length + 1


class STFTReImFrontEnd(nn.Module):
    """Compressed real/imag STFT front-end for SMoLnet-style RPS models."""

    out_channels = 2

    def __init__(self, n_fft=2048, hop_length=512, center=True, compressed=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.compressed = compressed
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(n_fft), persistent=True)

    def forward(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        x = torch.stack([spec.real, spec.imag], dim=1)
        if self.compressed:
            x = x.sign() * torch.log1p(x.abs())
        return x

    def num_frames(self, n_samples: int) -> int:
        if self.center:
            return n_samples // self.hop_length + 1
        return max(0, (n_samples - self.n_fft) // self.hop_length + 1)


class SqueezeExcitation2d(nn.Module):
    """Squeeze-and-Excitation block for 2D conv features (B, C, F, T)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResidualConvBlock2d(nn.Module):
    """Conv2d + BN + LeakyReLU with optional residual skip and SE."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: tuple,
        stride: tuple,
        padding: tuple,
        use_se: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)
        self.se = SqueezeExcitation2d(out_ch) if use_se else None
        self.has_skip = stride == (1, 1) and in_ch == out_ch
        if not self.has_skip and stride == (1, 1):
            self.skip_proj = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        if self.skip_proj is not None:
            identity = self.skip_proj(identity)
        if self.has_skip or self.skip_proj is not None:
            out = out + identity
        out = self.act(out)
        if self.se is not None:
            out = self.se(out)
        return out


class FrequencyAttentionPool(nn.Module):
    """Learned attention pooling over frequency dimension."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.scale = math.sqrt(channels // num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T)
        Returns: (B, C, T)
        """
        B, C, F, T = x.shape
        # Reshape to (B, T, F, C) then (B*T, F, C)
        x_perm = x.permute(0, 3, 2, 1).reshape(B * T, F, C)
        q = self.query(x_perm).view(B * T, F, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x_perm).view(B * T, F, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x_perm).view(B * T, F, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * T, F, C)
        # Weighted sum over frequency
        out = out.mean(dim=1)  # (B*T, C)
        out = out.view(B, T, C).transpose(1, 2)  # (B, C, T)
        return out


class TCNHead(nn.Module):
    """Temporal Convolutional Network head with dilated convolutions."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_rotors: int = 4,
        num_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch if i == 0 else hidden_ch,
                        hidden_ch,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(hidden_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv1d(hidden_ch, num_rotors, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        Returns: (B, num_rotors, T)
        """
        for layer in self.layers:
            x = layer(x) + x if x.shape == layer(x).shape else layer(x)
        return self.proj(x)


class CausalTCNHead(nn.Module):
    """Left-padded temporal convolution head.

    This keeps the temporal head causal by padding only on the left. It avoids
    BatchNorm/GroupNorm because those normalize across the time axis for Conv1d
    tensors and would leak future frames during training/inference on full clips.
    """

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_rotors: int = 4,
        num_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            conv = nn.Conv1d(
                in_ch if i == 0 else hidden_ch,
                hidden_ch,
                kernel_size,
                padding=0,
                dilation=dilation,
            )
            skip = None
            if i == 0 and in_ch != hidden_ch:
                skip = nn.Conv1d(in_ch, hidden_ch, kernel_size=1)
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv": conv,
                        "act": nn.ReLU(inplace=True),
                        "drop": nn.Dropout(dropout),
                        "skip": skip if skip is not None else nn.Identity(),
                    }
                )
            )
        self.kernel_size = kernel_size
        self.proj = nn.Conv1d(hidden_ch, num_rotors, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T), returns (B, num_rotors, T)."""
        for i, block in enumerate(self.blocks):
            assert isinstance(block, nn.ModuleDict)
            dilation = 2**i
            pad = (self.kernel_size - 1) * dilation
            y = F.pad(x, (pad, 0))
            y = block["conv"](y)
            y = block["act"](y)
            y = block["drop"](y)
            residual = block["skip"](x)
            x = y + residual if y.shape == residual.shape else y
        return self.proj(x)


class SMoLnetRPSDilatedLayer(nn.Module):
    """SMoLnet-style frequency-dilated convolution layer."""

    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3, use_norm=True):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2 * dilation, 0),
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class SMoLnetRPSLateLayer(nn.Module):
    """SMoLnet late square layer; optionally left-padded in time."""

    def __init__(self, channels, kernel_size=3, causal_time=False, use_norm=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal_time = causal_time
        padding = (kernel_size // 2, 0) if causal_time else (kernel_size // 2, kernel_size // 2)
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=(kernel_size, kernel_size), padding=padding
        )
        self.norm = nn.BatchNorm2d(channels) if use_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.causal_time:
            x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        return self.act(self.norm(self.conv(x)))


class SMoLnetRPSBackbone(nn.Module):
    """Frequency-dilated SMoLnet-style 2D convolutional backbone."""

    def __init__(
        self,
        input_ch=2,
        inner_channels=32,
        dilated_layers=8,
        total_layers=11,
        max_dilation=64,
        causal_time=False,
    ):
        super().__init__()
        assert total_layers > dilated_layers
        use_norm = not causal_time
        layers = []
        prev_ch = input_ch
        for idx in range(dilated_layers):
            dilation = 2**idx
            if max_dilation is not None:
                dilation = min(dilation, max_dilation)
            layers.append(
                SMoLnetRPSDilatedLayer(
                    prev_ch, inner_channels, dilation=dilation, use_norm=use_norm
                )
            )
            prev_ch = inner_channels
        for _ in range(total_layers - dilated_layers):
            layers.append(
                SMoLnetRPSLateLayer(inner_channels, causal_time=causal_time, use_norm=use_norm)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SMoLnetRPSTCN(nn.Module):
    """SMoLnet-style re/im STFT backbone with symmetric dilated TCN head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.frontend = frontend if frontend is not None else STFTReImFrontEnd(n_fft, hop_length)
        self.backbone = SMoLnetRPSBackbone(
            input_ch=2, inner_channels=16, dilated_layers=6, total_layers=8, max_dilation=32
        )
        self.head = TCNHead(16, hidden_ch=64, num_rotors=num_rotors, num_layers=4)

    def forward(self, audio):
        x = self.frontend(audio)
        h = self.backbone(x)
        h = h.mean(dim=2)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SMoLnetRPSSimpleHead(SMoLnetRPSTCN):
    """SMoLnet-style re/im STFT backbone with SimpleConv-style Conv1d head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.head = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, num_rotors, kernel_size=1),
        )


class SMoLnetRPSCausalTCN(SMoLnetRPSTCN):
    """SMoLnet-style backbone with left-padded late layers and causal TCN head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.backbone = SMoLnetRPSBackbone(
            input_ch=2,
            inner_channels=16,
            dilated_layers=6,
            total_layers=8,
            max_dilation=32,
            causal_time=True,
        )
        self.head = CausalTCNHead(16, hidden_ch=64, num_rotors=num_rotors, num_layers=4)


class GatedProjection(nn.Module):
    """Voicing-gated output projection: speed * sigmoid(gate).

    Rotor-off is a decision, not a regression to the edge of the output
    range: under MSE an uncertain plain-linear head outputs the conditional
    mean (a 10-30 rev/s hover on silent frames). The gate lets the model
    output an exact zero by classification while the speed branch stays a
    free regression.
    """

    def __init__(self, in_features: int, num_rotors: int):
        super().__init__()
        self.in_features = in_features
        self.num_rotors = num_rotors
        # One Linear to 2*num_rotors; the halves are (speed, gate_logit).
        # Default init leaves the gate at sigmoid(0) ~= 0.5, which scales the
        # speeds by half. This is harmless and learnable.
        self.linear = nn.Linear(in_features, 2 * num_rotors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) → (..., num_rotors)."""
        speed, gate_logit = self.linear(x).chunk(2, dim=-1)
        return speed * torch.sigmoid(gate_logit)


def _output_projection(in_features: int, num_rotors: int, gated: bool) -> nn.Module:
    """The final head projection — plain Linear, or the voicing-gated one."""
    if gated:
        return GatedProjection(in_features, num_rotors)
    return nn.Linear(in_features, num_rotors)


class BiGRUHead(nn.Module):
    """Bidirectional GRU head for temporal modeling."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_rotors: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        gated: bool = False,
    ):
        super().__init__()
        self.prenet = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_ch,
            hidden_ch,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = _output_projection(hidden_ch * 2, num_rotors, gated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        Returns: (B, num_rotors, T)
        """
        x = self.prenet(x)  # (B, hidden, T)
        x = x.transpose(1, 2)  # (B, T, hidden)
        x, _ = self.gru(x)  # (B, T, hidden*2)
        x = self.proj(x)  # (B, T, num_rotors)
        return x.transpose(1, 2)  # (B, num_rotors, T)


class CausalGRUHead(nn.Module):
    """Unidirectional GRU head with causal Conv1d prenet."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_rotors: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 5,
        gated: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.prenet_conv = nn.Conv1d(in_ch, hidden_ch, kernel_size=kernel_size, padding=0)
        self.prenet = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_ch,
            hidden_ch,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = _output_projection(hidden_ch, num_rotors, gated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        Returns: (B, num_rotors, T)
        """
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.prenet(self.prenet_conv(x))  # (B, hidden, T)
        x = x.transpose(1, 2)  # (B, T, hidden)
        x, _ = self.gru(x)  # (B, T, hidden)
        x = self.proj(x)  # (B, T, num_rotors)
        return x.transpose(1, 2)  # (B, num_rotors, T)


class CausalGRUNormHead(nn.Module):
    """Unidirectional GRU head with causal Conv1d + GroupNorm prenet."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 128,
        num_rotors: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        groups = 8 if hidden_ch % 8 == 0 else 1
        self.prenet_conv = nn.Conv1d(in_ch, hidden_ch, kernel_size=kernel_size, padding=0)
        self.prenet = nn.Sequential(
            nn.GroupNorm(groups, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_ch,
            hidden_ch,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_ch, num_rotors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.prenet(self.prenet_conv(x))
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = self.proj(x)
        return x.transpose(1, 2)


class CausalResidualConvBlock2d(nn.Module):
    """Residual Conv2d block with left-only padding along time."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: tuple,
        stride: tuple,
        padding: tuple,
        use_se: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.time_pad = kernel[1] - 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=(padding[0], 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)
        self.se = SqueezeExcitation2d(out_ch) if use_se else None
        self.has_skip = stride == (1, 1) and in_ch == out_ch
        if not self.has_skip and stride == (1, 1):
            self.skip_proj = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.pad(x, (self.time_pad, 0, 0, 0))
        out = self.conv(out)
        out = self.bn(out)
        if self.skip_proj is not None:
            identity = self.skip_proj(identity)
        if self.has_skip or self.skip_proj is not None:
            out = out + identity
        out = self.act(out)
        if self.se is not None:
            out = self.se(out)
        return out


class CausalSTFTMag(nn.Module):
    """Log-magnitude STFT using only past samples for each output frame.

    We left-pad by ``n_fft`` and run ``torch.stft(center=False)`` so the output
    frame count remains ``n_samples // hop_length + 1`` like the comparable
    centered STFT front-end. Frame ``t`` sees audio up to, but not after, the
    frame boundary at ``t * hop_length``.
    """

    out_channels = 1

    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(n_fft))

    def num_frames(self, n_samples: int) -> int:
        return n_samples // self.hop_length + 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        audio = F.pad(audio, (self.n_fft, 0))
        X = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            return_complex=True,
            normalized=True,
        )
        return torch.log1p(X.abs()).unsqueeze(1)


class TemporalTransformerHead(nn.Module):
    """Transformer temporal head for per-frame RPS prediction."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_rotors: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        gated: bool = False,
    ):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.prenet = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_ch,
            nhead=num_heads,
            dim_feedforward=hidden_ch * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = _output_projection(hidden_ch, num_rotors, gated)

    @staticmethod
    def _sinusoidal_positional_encoding(
        length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        Returns: (B, num_rotors, T)
        """
        x = self.prenet(x).transpose(1, 2)  # (B, T, hidden)
        pe = self._sinusoidal_positional_encoding(x.size(1), self.hidden_ch, x.device, x.dtype)
        x = self.transformer(x + pe.unsqueeze(0))
        x = self.proj(x)
        return x.transpose(1, 2)


class LocalTemporalTransformerHead(TemporalTransformerHead):
    """Transformer temporal head constrained to a fixed local attention window."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_rotors: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        local_window: int = 17,
    ):
        super().__init__(
            in_ch=in_ch,
            hidden_ch=hidden_ch,
            num_rotors=num_rotors,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.local_window = local_window

    @staticmethod
    def _local_attention_mask(length: int, local_window: int, device: torch.device) -> torch.Tensor:
        radius = max(0, local_window // 2)
        idx = torch.arange(length, device=device)
        return (idx[:, None] - idx[None, :]).abs() > radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        Returns: (B, num_rotors, T)
        """
        x = self.prenet(x).transpose(1, 2)  # (B, T, hidden)
        pe = self._sinusoidal_positional_encoding(x.size(1), self.hidden_ch, x.device, x.dtype)
        mask = self._local_attention_mask(x.size(1), self.local_window, x.device)
        x = self.transformer(x + pe.unsqueeze(0), mask=mask)
        x = self.proj(x)
        return x.transpose(1, 2)


class MultiScaleFusionHead(nn.Module):
    """FPN-style multi-scale feature fusion + prediction head."""

    def __init__(
        self,
        encoder_channels: list[int],
        target_t: int,
        common_dim: int = 64,
        num_rotors: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_t = target_t
        self.level_projs = nn.ModuleList(
            [nn.Conv1d(ch, common_dim, kernel_size=1) for ch in encoder_channels]
        )
        self.merge_conv = nn.Sequential(
            nn.Conv1d(common_dim, common_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(common_dim, common_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv1d(common_dim, num_rotors, kernel_size=1)

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """
        encoder_features: list of (B, C_i, F_i, T_i) from each encoder level,
                          ordered finest-to-coarsest (level 0 = finest).
        Returns: (B, num_rotors, target_t)
        """
        level_feats = []
        for feat, proj in zip(encoder_features, self.level_projs):
            B, C, F_i, T_i = feat.shape
            pooled = feat.mean(dim=2)  # (B, C, T_i)
            level_feats.append(proj(pooled))  # (B, common_dim, T_i)

        # Bottom-up merge: coarsest → finest
        merged = level_feats[-1]
        for i in range(len(level_feats) - 2, -1, -1):
            finer = level_feats[i]
            if merged.shape[-1] != finer.shape[-1]:
                merged = F.interpolate(
                    merged, size=finer.shape[-1], mode="linear", align_corners=False
                )
            merged = merged + finer

        # Upsample to target STFT frame rate
        if merged.shape[-1] != self.target_t:
            merged = F.interpolate(merged, size=self.target_t, mode="linear", align_corners=False)

        merged = self.merge_conv(merged)
        return self.proj(merged)  # (B, num_rotors, target_t)


# ─── Checkpoint compatibility ────────────────────────────────────────────────


def _remap_legacy_state_dict(state_dict: dict) -> dict:
    """Remap pre-0.13 state dict keys (window → frontend.window).

    Old checkpoints stored the Hann window buffer as ``window`` directly
    on the model.  After the front-end refactor it lives at
    ``frontend.window``.  All other keys are identical.
    """
    if "window" in state_dict and "frontend.window" not in state_dict:
        window_val = state_dict["window"]
        state_dict = {k: v for k, v in state_dict.items() if k != "window"}
        state_dict["frontend.window"] = window_val
    return state_dict


# ─── Baseline: SimpleConv (from paper) ───────────────────────────────────────


class SimpleConv(nn.Module):
    """
    Lightweight CNN on log-magnitude spectrograms for RPS prediction.
    Architecture mirrors DCUNet encoder but uses real-valued convolutions.
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        # Real-valued encoder (mirrors DCUNet channel sizes)
        self.encoder = nn.ModuleList()
        enc_spec = [
            (1, 45, (7, 5), (2, 1), (3, 2)),  # → (B,45, 513, T)
            (45, 90, (7, 5), (2, 1), (3, 2)),  # → (B,90, 257, T)
            (90, 90, (5, 3), (2, 1), (2, 1)),  # → (B,90, 129, T)
            (90, 90, (5, 3), (2, 1), (2, 1)),  # → (B,90,  65, T)
            (90, 90, (5, 3), (2, 1), (2, 1)),  # → (B,90,  33, T)
        ]
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        # Prediction head: pool freq → (B, 4, T)
        self.head = nn.Sequential(
            nn.Conv1d(90, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, num_rotors, kernel_size=1),
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)  # pool frequency (B, 90, T)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Variant 1: SimpleConvV2 (residual + SE + attention pool + BiGRU) ────────


class SimpleConvV2(nn.Module):
    """
    Improved SimpleConv with:
      - Deeper residual encoder (6 blocks)
      - Squeeze-and-Excitation blocks
      - Learned frequency attention pooling
      - BiGRU temporal head

    ``frontend`` also accepts a front-end registry key (a string), which is
    built with this model's ``n_fft``/``hop_length``. ``voicing_gate=True``
    replaces the head's output projection with a ``GatedProjection``.
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None, voicing_gate=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if isinstance(frontend, str):
            from models.frontends import build_frontend

            frontend = build_frontend(frontend, n_fft=n_fft, hop_length=hop_length)
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        # First block adapts to the front-end's channel count (1 for the
        # default stft_mag — weight-identical to earlier checkpoints).
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
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = BiGRUHead(
            128, hidden_ch=64, num_rotors=num_rotors, num_layers=2, gated=voicing_gate
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Frequency-aggregation variants (the G4 question) ───────────────────────
#
# SimpleConvV2's FrequencyAttentionPool ends in `out.mean(dim=1)` over
# frequency, and its attention over the frequency axis carries no positional
# encoding. The pool and everything after it are therefore EXACTLY permutation
# invariant over frequency: shuffle the 17 surviving bands and the predicted
# RPS does not move (verified to 1e-7). The encoder is convolutional and so
# weight-shared over frequency, which leaves absolute frequency position with
# no route into the head at all beyond boundary effects.
#
# That is a strange property for a task whose answer IS an absolute frequency.
# It is also not the only loss along the way. The six encoder blocks each
# stride frequency by two, so the axis goes 1025 -> 17 bins, from 7.8 Hz/bin to
# 470.6 Hz/bin. Rotor speeds of 20 to 90 rev/s put the comb's spacing at 2.6 to
# 11.5 bins at the front end and 0.3 to 1.4 bins by the third block — below the
# frequency axis' own Nyquist. The spacing that carries the answer is aliased
# away early, and whatever survives does so encoded in channels.
#
# The three variants below separate the two losses, so a result says which one
# mattered:
#   freqpos   — position only. Same shapes, same pool, plus a learned embedding
#               over the frequency axis. +2 k parameters, no extra arithmetic.
#   freqcat   — position and per-band identity. The pool is replaced by a
#               linear map over the flattened (channel, frequency) pairs.
#   freqhires — the above plus resolution: the last two blocks stop striding
#               frequency, so 65 bins survive instead of 17.


def _encoder_freq_bins(encoder: nn.ModuleList, frontend: nn.Module, hop_length: int) -> int:
    """Frequency bins surviving `encoder`, measured rather than derived.

    The front end decides how many rows it emits, and not every one emits
    ``n_fft // 2 + 1`` of them — ``comb_if`` emits one row per f0 candidate.
    So the count comes from running silence through the real front end.
    """
    modules = list(encoder.modules()) + list(frontend.modules())
    was_training = [m for m in modules if m.training]
    for m in modules:
        m.eval()
    with torch.no_grad():
        h = frontend(torch.zeros(1, 32 * hop_length))
        for block in encoder:
            h = block(h)
    for m in was_training:
        m.train()
    return int(h.shape[2])


class _FreqVariantBase(nn.Module):
    """SimpleConvV2 with a configurable frequency-aggregation stage."""

    FREQ_STRIDES = (2, 2, 2, 2, 2, 2)

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None, voicing_gate=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None or isinstance(frontend, str):
            from models.frontends import build_frontend

            frontend = build_frontend(frontend or "stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        in_ch = getattr(frontend, "out_channels", 1)
        shapes = [
            (in_ch, 64, (7, 5), (3, 2)),
            (64, 128, (7, 5), (3, 2)),
            (128, 128, (5, 3), (2, 1)),
            (128, 128, (5, 3), (2, 1)),
            (128, 128, (5, 3), (2, 1)),
            (128, 128, (5, 3), (2, 1)),
        ]
        self.encoder = nn.ModuleList(
            ResidualConvBlock2d(ic, oc, k, (fs, 1), p, use_se=True)
            for (ic, oc, k, p), fs in zip(shapes, self.FREQ_STRIDES, strict=True)
        )
        self.n_freq = _encoder_freq_bins(self.encoder, self.frontend, hop_length)
        self._build_aggregator()
        self.head = BiGRUHead(
            128, hidden_ch=64, num_rotors=num_rotors, num_layers=2, gated=voicing_gate
        )

    def _build_aggregator(self) -> None:
        raise NotImplementedError

    def _aggregate(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, audio):
        h = self.frontend(audio)  # (B, C, F, T)
        for block in self.encoder:
            h = block(h)
        return self.head(self._aggregate(h))  # (B, 4, T)


class SimpleConvV2FreqPos(_FreqVariantBase):
    """SimpleConvV2 plus a learned positional embedding over the frequency axis.

    The minimal repair, and the control for the other two: identical shapes,
    identical pooling, 2 k more parameters, and the only new capability is that
    the pool can tell one frequency band from another.
    """

    def _build_aggregator(self) -> None:
        self.freq_pos = nn.Parameter(torch.zeros(1, 128, self.n_freq, 1))
        nn.init.normal_(self.freq_pos, std=0.02)
        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)

    def _aggregate(self, h: torch.Tensor) -> torch.Tensor:
        return self.freq_pool(h + self.freq_pos)


class SimpleConvV2FreqCat(_FreqVariantBase):
    """SimpleConvV2 with the frequency axis kept rather than averaged away.

    The pool is replaced by a linear map over the flattened (channel,
    frequency) pairs, so the head sees WHICH band each feature came from and
    can weight bands differently, instead of receiving their mean.
    """

    def _build_aggregator(self) -> None:
        self.freq_proj = nn.Sequential(
            nn.Linear(128 * self.n_freq, 256), nn.GELU(), nn.Linear(256, 128)
        )

    def _aggregate(self, h: torch.Tensor) -> torch.Tensor:
        B, C, F, T = h.shape
        flat = h.permute(0, 3, 1, 2).reshape(B * T, C * F)
        return self.freq_proj(flat).view(B, T, 128).transpose(1, 2)


class SimpleConvV2FreqHiRes(SimpleConvV2FreqCat):
    """FreqCat, with the last two blocks no longer striding frequency.

    65 bins survive instead of 17, so the head sees frequency at 123 Hz rather
    than 471 Hz. Blocks five and six do their work on a longer axis, which is
    where the extra cost goes.
    """

    FREQ_STRIDES = (2, 2, 2, 2, 1, 1)


# ─── Variant 2: SimpleConvWide (scale up, keep it simple) ────────────────────


class SimpleConvV2TCN(nn.Module):
    """SimpleConvV2 encoder/pool with the existing symmetric dilated TCN head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = TCNHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=4)

    def forward(self, audio):
        x = self.frontend(audio)
        h = x
        for block in self.encoder:
            h = block(h)
        h = self.freq_pool(h)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2CausalTCN(SimpleConvV2TCN):
    """SimpleConvV2 encoder/pool with a left-padded dilated TCN head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.head = CausalTCNHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=4)


class SimpleConvV2SMoLTCN(SimpleConvV2TCN):
    """SimpleConvV2 plus SMoLnet-style frequency-dilated refinement before TCN."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.smol_refine = SMoLnetRPSBackbone(
            input_ch=128, inner_channels=128, dilated_layers=4, total_layers=5, max_dilation=8
        )

    def forward(self, audio):
        x = self.frontend(audio)
        h = x
        for block in self.encoder:
            h = block(h)
        h = self.smol_refine(h)
        h = self.freq_pool(h)
        return self.head(h)


class SimpleConvV2SMoLCausalTCN(SimpleConvV2SMoLTCN):
    """SimpleConvV2 + SMoL refinement with left-padded TCN head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.smol_refine = SMoLnetRPSBackbone(
            input_ch=128,
            inner_channels=128,
            dilated_layers=4,
            total_layers=5,
            max_dilation=8,
            causal_time=True,
        )
        self.head = CausalTCNHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=4)


class SimpleConvV2SMoLBiGRU(SimpleConvV2):
    """SimpleConvV2 + SMoLnet-style frequency-dilated refinement + BiGRU head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.smol_refine = SMoLnetRPSBackbone(
            input_ch=128, inner_channels=128, dilated_layers=4, total_layers=5, max_dilation=8
        )

    def forward(self, audio):
        x = self.frontend(audio)
        h = x
        for block in self.encoder:
            h = block(h)
        h = self.smol_refine(h)
        h = self.freq_pool(h)
        return self.head(h)


class SimpleConvV2UniGRU(nn.Module):
    """SimpleConvV2 encoder/pool with a unidirectional causal GRU head.

    ``frontend`` also accepts a front-end registry key (a string), which is
    built with this model's ``n_fft``/``hop_length``. ``voicing_gate=True``
    replaces the head's output projection with a ``GatedProjection``.
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None, voicing_gate=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if isinstance(frontend, str):
            from models.frontends import build_frontend

            frontend = build_frontend(frontend, n_fft=n_fft, hop_length=hop_length)
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        # First block adapts to the front-end's channel count (1 for the
        # default stft_mag — weight-identical to earlier checkpoints).
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
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = CausalGRUHead(
            128, hidden_ch=64, num_rotors=num_rotors, num_layers=2, gated=voicing_gate
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)
        h = x
        for block in self.encoder:
            h = block(h)
        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2UniGRU128(SimpleConvV2UniGRU):
    """SimpleConvV2 encoder/pool with capacity-matched unidirectional GRU head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None, voicing_gate=False):
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            num_rotors=num_rotors,
            frontend=frontend,
            voicing_gate=voicing_gate,
        )
        self.head = CausalGRUHead(
            128, hidden_ch=128, num_rotors=num_rotors, num_layers=2, gated=voicing_gate
        )


class SimpleConvV2UniGRU128Norm(SimpleConvV2UniGRU):
    """Capacity-matched unidirectional GRU head with normalized prenet."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.head = CausalGRUNormHead(128, hidden_ch=128, num_rotors=num_rotors, num_layers=2)


class SimpleConvV2UniGRU128NormDO03(SimpleConvV2UniGRU):
    """Normalized capacity-matched unidirectional GRU head with stronger dropout."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.head = CausalGRUNormHead(
            128, hidden_ch=128, num_rotors=num_rotors, num_layers=2, dropout=0.3
        )


class SimpleConvV2UniGRU96NormDO03(SimpleConvV2UniGRU):
    """Normalized lower-capacity unidirectional GRU head with stronger dropout."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.head = CausalGRUNormHead(
            128, hidden_ch=96, num_rotors=num_rotors, num_layers=2, dropout=0.3
        )


class SimpleConvV2UniGRU96NormDO02(SimpleConvV2UniGRU):
    """Normalized hidden=96 unidirectional GRU head with moderate dropout."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.head = CausalGRUNormHead(
            128, hidden_ch=96, num_rotors=num_rotors, num_layers=2, dropout=0.2
        )


class SimpleConvV2UniGRU64NormDO03(SimpleConvV2UniGRU):
    """Normalized hidden=64 unidirectional GRU head with stronger dropout."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )
        self.head = CausalGRUNormHead(
            128, hidden_ch=64, num_rotors=num_rotors, num_layers=2, dropout=0.3
        )


class SimpleConvV2CausalGRU(nn.Module):
    """SimpleConvV2-style fully time-causal stack with unidirectional GRU."""

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        num_rotors=4,
        frontend=None,
        hidden_ch=64,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.frontend = frontend or CausalSTFTMag(n_fft=n_fft, hop_length=hop_length)

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(CausalResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = CausalGRUHead(128, hidden_ch=hidden_ch, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, 1, F, T)
        h = x
        for block in self.encoder:
            h = block(h)
        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2CausalGRU96(SimpleConvV2CausalGRU):
    """Time-causal SimpleConvV2 stack with wider unidirectional GRU head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            num_rotors=num_rotors,
            frontend=frontend,
            hidden_ch=96,
        )


class SimpleConvV2Transformer(nn.Module):
    """SimpleConvV2 encoder/pool with a Transformer temporal head replacing BiGRU.

    ``frontend`` also accepts a front-end registry key (a string), which is
    built with this model's ``n_fft``/``hop_length``. ``voicing_gate=True``
    replaces the head's output projection with a ``GatedProjection``.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        num_rotors=4,
        frontend=None,
        voicing_gate=False,
        head_hidden_ch=64,
        head_layers=2,
        head_heads=4,
        head_dropout=0.1,
        width=128,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if isinstance(frontend, str):
            from models.frontends import build_frontend

            frontend = build_frontend(frontend, n_fft=n_fft, hop_length=hop_length)
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        # First block adapts to the front-end's channel count (1 for the
        # default stft_mag — weight-identical to pre-G2 checkpoints).
        in_ch = getattr(frontend, "out_channels", 1)
        # ``width`` scales the whole trunk. At the default 128 the spec is
        # channel-identical to every existing checkpoint; the frequency pool and
        # the head follow it, so one knob widens the encoder end to end.
        w, half = int(width), max(int(width) // 2, 1)
        enc_spec = [
            (in_ch, half, (7, 5), (2, 1), (3, 2)),
            (half, w, (7, 5), (2, 1), (3, 2)),
            (w, w, (5, 3), (2, 1), (2, 1)),
            (w, w, (5, 3), (2, 1), (2, 1)),
            (w, w, (5, 3), (2, 1), (2, 1)),
            (w, w, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(w, num_heads=4)
        # The defaults reproduce every existing checkpoint exactly. The four
        # knobs exist so the temporal head's capacity can be raised without a
        # new class: at the defaults it is 2 layers of width 64, which is small
        # beside the 6-block encoder in front of it.
        self.head = TemporalTransformerHead(
            w,
            hidden_ch=head_hidden_ch,
            num_rotors=num_rotors,
            num_layers=head_layers,
            num_heads=head_heads,
            dropout=head_dropout,
            gated=voicing_gate,
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2TransformerHCQT(SimpleConvV2Transformer):
    """G2a front-end arm: SimpleConvV2Transformer trunk on a harmonic-stacked
    HCQT front-end (VK-parity campaign, criterion 2.3).

    Hypothesis under test: the log-magnitude STFT front-end is the parity
    bottleneck because the trunk has no harmonically *aligned* evidence — the
    HCQT stacks log-spaced CQT copies at ``fmin*h`` so all harmonics of a
    candidate f0 line up on the same frequency bin across channels
    (``models/multif0`` HCQT, nnAudio backend — all-torch on-device,
    including the fixed ``_phase_diff_torch`` phase path; the front-end is a
    few percent of the model's compute).

    Native 16 kHz, no resampling (``sr=input_sr=16000``); ``fmin=32.7`` (C1)
    matches the multif0 checkpoint convention. At 16 kHz / 6 octaves the
    harmonics auto-derive to [1, 2, 3] under Nyquist → 6 channels (mag +
    dphase per harmonic). The HCQT runs on its own 256-sample hop; features
    are linearly interpolated along time onto the model's STFT output grid
    (``n_samples // hop_length + 1``) before the encoder, so the output
    contract and the PIT-MSE target grid are unchanged.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        num_rotors=4,
        frontend=None,
        fmin=32.7,
        n_octaves=6,
        over_sample=5,
        harmonics=None,
        hcqt_hop_length=256,
        phase=True,
    ):
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend(
                "hcqt",
                sr=16000,
                input_sr=16000,
                fmin=fmin,
                n_octaves=n_octaves,
                over_sample=over_sample,
                harmonics=harmonics,
                hop_length=hcqt_hop_length,
                phase=phase,
            )
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )

    def forward(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        t_out = audio.shape[-1] // self.hop_length + 1
        x = self.frontend(audio)  # (B, 2H, F_cqt, T_hcqt)
        if x.shape[-1] != t_out:
            b, c, fbins, t = x.shape
            x = F.interpolate(
                x.reshape(b, c * fbins, t), size=t_out, mode="linear", align_corners=False
            ).reshape(b, c, fbins, t_out)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)


class SimpleConvV2TransformerIF(SimpleConvV2Transformer):
    """G2b front-end arm: SimpleConvV2Transformer trunk on the IF-augmented
    STFT front-end (VK-parity campaign, criterion 2.3).

    Hypothesis under test: the magnitude STFT lacks sub-bin frequency
    precision (one bin at n_fft=2048 / 16 kHz is 7.8 Hz, far coarser than
    the ~0.7 rev/s blind-VK bar). ``stft_mag_if`` channel-concatenates the
    standard instantaneous-frequency estimator (per-hop phase difference,
    wrapped, as deviation from bin center in fractional bins) with the log
    magnitude — same STFT grid as the baseline, so nothing else changes.
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag_if", n_fft=n_fft, hop_length=hop_length)
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )


class SimpleConvV2TransformerLearned(SimpleConvV2Transformer):
    """SimpleConvV2Transformer trunk on a LEARNED time-domain filterbank.

    Hypothesis under test: every front-end in this project hands the trunk a
    function of the STFT magnitude, and magnitude discards the phase. The
    campaign's models converge on a degenerate answer — a fixed rotor spread of
    about 10 rev/s whatever the rotors do — which is what reading a comb's mean
    spacing rather than its individual lines produces. Recovering a speed to the
    precision the signal carries is a phase problem: the classical phase-
    increment refiner reaches 0.0006 rev/s on a single-rotor comb, against this
    trunk's 2.535 comb floor.

    ``learned_conv`` convolves the raw waveform with ``2F`` free filters of
    length ``n_fft`` at stride ``hop_length`` and hands over their raw responses
    (real, imaginary, log-magnitude). The windowed DFT basis IS such a filter
    set, so the STFT is a strict subset of what this can represent, and
    ``init="stft"`` starts there — reproducing ``stft_mag`` to 2e-06 — so the
    arm begins at the baseline representation with phase added and separates
    "do learned filters help" from "can gradient descent find the STFT".
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None, **kwargs):
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("learned_conv", n_fft=n_fft, hop_length=hop_length)
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors,
            frontend=frontend, **kwargs
        )


class SimpleConvV2TransformerComb(SimpleConvV2Transformer):
    """G4 front-end arm: SimpleConvV2Transformer trunk on the whitened comb
    matched-filter + IF-consensus front-end (VK-parity campaign, criterion 2.3).

    G2 verdict (docs/experiments/g1-vk-parity.md): IF phase evidence helps
    (first arm to beat the baseline) but constant-Q harmonic stacking hurts —
    the missing ingredient is harmonic AGGREGATION on the LINEAR frequency
    grid. ``comb_if`` performs exactly that: the trainable analogue of the
    blind VK tracker's whitened comb scan (mean whitened log-mag over comb
    teeth per candidate f0, teeth capped at 1200 Hz), plus a Fisher
    k²-weighted IF frequency-consensus channel and the stage-guard occupancy
    fraction — 3 channels over a 30..120 rev/s ×0.25 f0 grid (361 rows).

    The trunk therefore operates in f0-space where each rotor is a ridge;
    the time grid and output contract are unchanged (same hop-512 frames as
    the baseline / stft_mag_if arms).

    G4b: ``coord_channel`` (default True) rides through to the front-end's
    CoordConv-style 4th channel (row f0 / 100, constant over time) — the G4a
    training refutation showed the trunk cannot read WHERE the ridge is once
    frequency pooling averages the row axis away; the coordinate channel
    makes the position an explicit feature. ``coord_channel=False``
    reproduces the 3-channel G4a model (A/B + G4a checkpoint loading).
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None, coord_channel=True):
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend(
                "comb_if", n_fft=n_fft, hop_length=hop_length, coord_channel=coord_channel
            )
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )


class SimpleConvV2TransformerPyramid(SimpleConvV2Transformer):
    """G8a front-end arm: SimpleConvV2Transformer trunk on the multi-resolution
    STFT pyramid front-end (C1 of docs/g8-hierarchical-frontend-design.md).

    The single-window STFT cannot serve fundamentals (need fine FREQUENCY,
    tolerate slow time) and high harmonics (need fine TIME + IF sub-bin,
    tolerate coarse bins) at once, and constant-Q allocates backwards (G2a
    refuted). ``pyramid_if`` runs four parallel STFTs (n_fft 8192/4096/2048/
    1024, each used only in its band 30-250/250-1000/1000-2000/2000-4000 Hz),
    each contributing log1p-mag + IF channels, resampled onto a common
    log-frequency axis (340 rows, ~48 bins/octave — comb patterns become
    shift-equivariant in f0) and the standard hop-512 time grid.

    The trunk is unchanged (in_ch adapts via the frontend-aware first conv);
    output contract identical to the baseline / stft_mag_if arms.

    G8a2: ``collapse_bands`` (default True) rides through to the front-end —
    the masked per-band tensors are summed into 2 dense channels (in_ch=2)
    instead of the G8a 8-channel concat, whose 6-of-8 exactly-zero channels
    per row trained violently unstably (val 142→659 swings — see
    docs/experiments/g1-vk-parity.md § G8a result). ``collapse_bands=False``
    reproduces the dead G8a model (in_ch=8; A/B + G8a checkpoint loading).
    """

    def __init__(
        self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None, collapse_bands=True
    ):
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend(
                "pyramid_if", hop_length=hop_length, collapse_bands=collapse_bands
            )
        super().__init__(
            n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors, frontend=frontend
        )


class SimpleConvV2LocalAttention(nn.Module):
    """SimpleConvV2 encoder/pool with local-window Transformer temporal attention."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        # 17 STFT frames ≈ 0.54 s at 16 kHz / hop 512: enough for smooth RPS
        # changes while discouraging nonlocal shortcuts on the tiny valid set.
        self.head = LocalTemporalTransformerHead(
            128,
            hidden_ch=64,
            num_rotors=num_rotors,
            num_layers=2,
            num_heads=4,
            local_window=17,
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2MultiRes(nn.Module):
    """SimpleConvV2 with concatenated long/short-window STFT magnitude inputs."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        from models.frontends import build_frontend

        if frontend is None:
            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend
        # Same hop keeps time frames aligned; shorter window trades frequency
        # resolution for better localization of rapid RPS/noise changes.
        short_n_fft = max(256, n_fft // 2)
        self.short_frontend = build_frontend("stft_mag", n_fft=short_n_fft, hop_length=hop_length)

        enc_spec = [
            (2, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = BiGRUHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x_long = self.frontend(audio)  # (B, 1, F_long, T)
        x_short = self.short_frontend(audio)  # (B, 1, F_short, T_short)
        if x_short.shape[-1] != x_long.shape[-1] or x_short.shape[-2] != x_long.shape[-2]:
            x_short = F.interpolate(
                x_short,
                size=(x_long.shape[-2], x_long.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x_long, x_short], dim=1)  # (B, 2, F_long, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2Wavelet(nn.Module):
    """SimpleConvV2 with an added lightweight Haar-like temporal wavelet branch."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.wavelet_scales = (128, 256, 512, 1024)
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend
        self.wavelet_proj = nn.Sequential(
            nn.Conv1d(len(self.wavelet_scales), 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=1),
        )

        enc_spec = [
            (2, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = BiGRUHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

    def _wavelet_features(self, audio: torch.Tensor, target_t: int) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        x = audio.unsqueeze(1)
        feats = []
        for scale in self.wavelet_scales:
            half = scale // 2
            filt = torch.cat(
                [
                    torch.ones(half, device=x.device, dtype=x.dtype),
                    -torch.ones(half, device=x.device, dtype=x.dtype),
                ]
            ).view(1, 1, scale)
            filt = filt / math.sqrt(float(scale))
            y = F.conv1d(x, filt, stride=self.hop_length, padding=scale // 2)
            feats.append(torch.log1p(y.abs()))
        w = torch.cat(feats, dim=1)  # (B, scales, T_w)
        if w.shape[-1] != target_t:
            w = F.interpolate(w, size=target_t, mode="linear", align_corners=False)
        return self.wavelet_proj(w)  # (B, 1, T)

    def forward(self, audio):
        x_mag = self.frontend(audio)  # (B, 1, F, T)
        w = self._wavelet_features(audio, x_mag.shape[-1]).unsqueeze(2)
        w = w.expand(-1, -1, x_mag.shape[-2], -1)
        x = torch.cat([x_mag, w], dim=1)  # (B, 2, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2MagPhase(nn.Module):
    """SimpleConvV2 using log-magnitude plus cosine/sine phase STFT channels."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_magphase", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (3, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = BiGRUHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, 3, F, T) — mag, cos(θ), sin(θ)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2DualPool(nn.Module):
    """SimpleConvV2 concatenating attention and mean frequency pooling before BiGRU."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = BiGRUHead(256, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h_attn = self.freq_pool(h)  # (B, 128, T)
        h_mean = h.mean(dim=2)  # (B, 128, T)
        return self.head(torch.cat([h_attn, h_mean], dim=1))  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2GRU96(nn.Module):
    """SimpleConvV2 with a modestly wider BiGRU temporal head (hidden=96)."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = BiGRUHead(128, hidden_ch=96, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, 4, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvWide(nn.Module):
    """Wider and deeper SimpleConv with residual connections but no attention/GRU."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 256, (5, 3), (2, 1), (2, 1)),
            (256, 256, (5, 3), (2, 1), (2, 1)),
            (256, 256, (5, 3), (2, 1), (2, 1)),
            (256, 256, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        self.head = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, num_rotors, kernel_size=1),
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Variant 3: SimpleConvTCN (dilated conv head) ────────────────────────────


class SimpleConvTCN(nn.Module):
    """SimpleConv with TCN head for long-range temporal dependencies."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        self.head = TCNHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=4)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Variant 4: SimpleConvMultiScale (FPN-style fusion) ──────────────────────


class SimpleConvMultiScale(nn.Module):
    """SimpleConv with multi-scale encoder feature fusion."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        enc_channels = [oc for _, oc, _, _, _ in enc_spec]
        target_t = stft_time_frames(131584, hop_length, n_fft)
        self.head = MultiScaleFusionHead(
            enc_channels, target_t, common_dim=64, num_rotors=num_rotors
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        encoder_features = []
        for block in self.encoder:
            h = block(h)
            encoder_features.append(h)

        return self.head(encoder_features)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Variant 5: SimpleConvBiGRU (encoder + BiGRU head) ───────────────────────


class SimpleConvBiGRU(nn.Module):
    """Baseline encoder with BiGRU head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        self.encoder = nn.ModuleList()
        enc_spec = [
            (1, 45, (7, 5), (2, 1), (3, 2)),
            (45, 90, (7, 5), (2, 1), (3, 2)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        self.head = BiGRUHead(90, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Variant 6: SimpleConvAttnPool (attention pooling only) ────────────────────


class SimpleConvAttnPool(nn.Module):
    """Baseline encoder with attention-based frequency pooling."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        self.encoder = nn.ModuleList()
        enc_spec = [
            (1, 45, (7, 5), (2, 1), (3, 2)),
            (45, 90, (7, 5), (2, 1), (3, 2)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        self.freq_pool = FrequencyAttentionPool(90, num_heads=3)
        self.head = nn.Sequential(
            nn.Conv1d(90, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, num_rotors, kernel_size=1),
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Variant 7: SimpleConvSENext (SE + residual + deeper) ────────────────────


class SimpleConvSENext(nn.Module):
    """Residual + SE blocks, deeper encoder, larger head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, k, s, p, use_se=True))

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, num_rotors, kernel_size=1),
        )

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    # ─── Model factory / registry ────────────────────────────────────────────────


# ─── Variant 8: SimpleConvMagPhaseBiGRU (mag + phase input, BiGRU head) ─────


class SimpleConvMagPhaseBiGRU(nn.Module):
    """
    Uses log-magnitude, cos(phase) and sin(phase) as 3 input channels.
    Phase provides temporal structure complementary to magnitude.
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_magphase", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        self.encoder = nn.ModuleList()
        enc_spec = [
            (3, 45, (7, 5), (2, 1), (3, 2)),
            (45, 90, (7, 5), (2, 1), (3, 2)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        self.head = BiGRUHead(90, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, 3, F, T) — mag, cos(θ), sin(θ)

        h = x
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── Variant 9: SimpleConvBiGRUV2 (deeper encoder + BiGRU) ───────────────────


class SimpleConvBiGRUV2(nn.Module):
    """Deeper/wider encoder (6 blocks, 128 ch) + BiGRU head."""

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

        enc_spec = [
            (1, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        self.head = BiGRUHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)
        return self.head(h)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


# ─── DCUNet/DCCRN complex-conv encoders (standalone RPS prediction) ──────────
#
# Ported from ``train_rps_predictor.py`` (``DCUNetEncRPS`` / ``DCCRNEncRPS``,
# never registered in ``src/models/registry.py::RPS_MODEL_REGISTRY`` until
# now — see docs/refactor-unified-framework.md). ``RPSPredictionHead`` is
# *not* duplicated here: the canonical implementation already lives in
# ``models.dcunet`` (``models.dccrn`` imports it from there too).


class DCUNetEncRPS(nn.Module):
    """DCUNet encoder (complex conv) + ``RPSPredictionHead`` for standalone RPS
    prediction. Faithfully replicates the encoder architecture from ``models/dcunet.py``.
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, num_layers=5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(n_fft))

        from models.dcunet import CBatchNorm2d as DCUNetCBN
        from models.dcunet import CConv2d as DCUNetCConv
        from models.dcunet import RPSPredictionHead

        # DCUNet encoder spec — faithful copy from models/dcunet.py
        enc_spec = [
            (1, 45, (7, 5), (2, 2), (3, 2)),
            (45, 90, (7, 5), (2, 2), (3, 2)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        if num_layers == 6:
            enc_spec.append((90, 90, (5, 3), (2, 1), (2, 1)))

        self.encoders = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoders.append(
                nn.Sequential(
                    DCUNetCConv(ic, oc, k, s, p),
                    DCUNetCBN(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        enc_channels = [oc for _, oc, _, _, _ in enc_spec]
        target_t = stft_time_frames(131584, hop_length, n_fft)  # chunk_size=131584
        self.head = RPSPredictionHead(enc_channels, target_t, num_rotors=num_rotors)

    def forward(self, audio):
        """
        audio: (B, samples) raw mono waveform.
        Returns: (B, 4, T_stft) predicted RPS per STFT frame.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        X = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=True,
        )
        X = torch.view_as_real(X)  # (B, F, T, 2)
        X = X.unsqueeze(1)  # (B, 1, F, T, 2)

        encoder_features = []
        h = X
        for encoder in self.encoders:
            h = encoder(h)
            encoder_features.append(h)

        return self.head(encoder_features)


class DCCRNEncRPS(nn.Module):
    """DCCRN encoder (complex conv) + ``RPSPredictionHead`` for standalone RPS
    prediction. Faithfully replicates the encoder architecture from ``models/dccrn.py``.
    Supports the lite variant (fewer layers/channels).
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, lite=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.lite = lite
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(n_fft))

        from models.dccrn import CBatchNorm2d as DCCRN_CBN
        from models.dccrn import CConv2d as DCCRN_CConv
        from models.dcunet import RPSPredictionHead

        encoder_channels = [16, 32, 64, 128] if lite else [32, 64, 128, 256, 256, 512]

        enc_kernel = (5, 2)
        enc_stride = (2, 1)
        enc_padding = (2, 0)

        in_channels = [1] + encoder_channels[:-1]

        self.encoders = nn.ModuleList()
        for ic, oc in zip(in_channels, encoder_channels):
            self.encoders.append(
                nn.Sequential(
                    DCCRN_CConv(ic, oc, enc_kernel, enc_stride, enc_padding),
                    DCCRN_CBN(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        target_t = stft_time_frames(131584, hop_length, n_fft)
        self.head = RPSPredictionHead(encoder_channels, target_t, num_rotors=num_rotors)

    def forward(self, audio):
        """
        audio: (B, samples) raw mono waveform.
        Returns: (B, 4, T_stft) predicted RPS per STFT frame.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        X = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=True,
        )
        X = torch.view_as_real(X)  # (B, F, T, 2)
        X = X.unsqueeze(1)  # (B, 1, F, T, 2)

        encoder_features = []
        h = X
        for encoder in self.encoders:
            h = encoder(h)
            encoder_features.append(h)

        return self.head(encoder_features)


# ─── Model factory / registry ────────────────────────────────────────────────


RPS_MODEL_REGISTRY = {
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
    "simple_conv_v2_transformer_learned": SimpleConvV2TransformerLearned,
    "simple_conv_v2_transformer_comb": SimpleConvV2TransformerComb,
    "simple_conv_v2_transformer_pyramid": SimpleConvV2TransformerPyramid,
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
}


def get_rps_model(model_name, n_fft=2048, hop_length=512, num_rotors=4):
    if model_name not in RPS_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(RPS_MODEL_REGISTRY.keys())}"
        )
    return RPS_MODEL_REGISTRY[model_name](n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors)
