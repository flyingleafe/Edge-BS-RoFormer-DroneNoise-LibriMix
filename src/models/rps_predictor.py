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


class BiGRUHead(nn.Module):
    """Bidirectional GRU head for temporal modeling."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        num_rotors: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
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
        self.proj = nn.Linear(hidden_ch * 2, num_rotors)

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
        self.proj = nn.Linear(hidden_ch, num_rotors)

    @staticmethod
    def _sinusoidal_positional_encoding(
        length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dim)
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
        pe = self._sinusoidal_positional_encoding(
            x.size(1), self.hidden_ch, x.device, x.dtype
        )
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
        pe = self._sinusoidal_positional_encoding(
            x.size(1), self.hidden_ch, x.device, x.dtype
        )
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
        self.head = BiGRUHead(128, hidden_ch=64, num_rotors=num_rotors, num_layers=2)

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


# ─── Variant 2: SimpleConvWide (scale up, keep it simple) ────────────────────


class SimpleConvV2Transformer(nn.Module):
    """SimpleConvV2 encoder/pool with a Transformer temporal head replacing BiGRU."""

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
        self.head = TemporalTransformerHead(
            128, hidden_ch=64, num_rotors=num_rotors, num_layers=2, num_heads=4
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
        self.short_frontend = build_frontend(
            "stft_mag", n_fft=short_n_fft, hop_length=hop_length
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


# ─── Model factory / registry ────────────────────────────────────────────────


RPS_MODEL_REGISTRY = {
    "simple_conv": SimpleConv,
    "simple_conv_v2": SimpleConvV2,
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
}


def get_rps_model(model_name, n_fft=2048, hop_length=512, num_rotors=4):
    if model_name not in RPS_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(RPS_MODEL_REGISTRY.keys())}"
        )
    return RPS_MODEL_REGISTRY[model_name](n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors)
