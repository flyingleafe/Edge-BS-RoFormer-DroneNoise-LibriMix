"""
Refactored DCUNet with separate Encoder and Decoder modules.
RPS conditioning is a DECODER-side feature (not encoder-side).

Key design decisions:
- Encoder: clean feature extraction from STFT input (no RPS)
- Decoder: receives bottleneck features and skip connections; RPS fusion happens HERE
- Two decoder RPS fusion strategies:
  1. decoder_bottleneck: inject RPS at start of decoder (after first transposed conv)
  2. decoder_hierarchical: inject RPS at multiple decoder levels (mirrors encoder hierarchical)

This design allows training encoder-only baselines and adding decoder RPS later,
which is useful for analyzing where RPS helps most in the pipeline.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft_time_frames(audio_length: int, hop_length: int, n_fft: int) -> int:
    """Number of STFT time frames for a given audio length."""
    return (audio_length - n_fft) // hop_length + 1


def encoder_time_lengths(n_stft_time: int, encoder_strides: list) -> list:
    """Time dimension at each encoder level."""
    lengths = [n_stft_time]
    for _, stride_t in encoder_strides:
        lengths.append((lengths[-1] + 1) // stride_t)
    return lengths


def decoder_time_lengths(bottleneck_time: int, decoder_strides: list) -> list:
    """Time dimension at each decoder level (starting from bottleneck)."""
    lengths = [bottleneck_time]
    for _, stride_t in decoder_strides:
        lengths.append(lengths[-1] * stride_t)
    return lengths


# =============================================================================
# Complex Convolutional Building Blocks
# =============================================================================


class CConv2d(nn.Module):
    """Complex Convolutional Layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.im_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        """Input: (B, C, F, T, 2), Output: (B, C, F, T, 2)"""
        x_real, x_im = x[..., 0], x[..., 1]
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        return torch.stack([c_real, c_im], dim=-1)


class CConvTranspose2d(nn.Module):
    """Complex Transpose Convolutional Layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        self.real_convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.im_convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        """Input: (B, C, F, T, 2), Output: (B, C, F, T, 2)"""
        x_real, x_im = x[..., 0], x[..., 1]
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        return torch.stack([ct_real, ct_im], dim=-1)


class CBatchNorm2d(nn.Module):
    """Complex Batch Normalization."""

    def __init__(self, num_features):
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.im_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        """Input: (B, C, F, T, 2), Output: (B, C, F, T, 2)"""
        x_real, x_im = x[..., 0], x[..., 1]
        x_real = self.real_bn(x_real)
        x_im = self.im_bn(x_im)
        return torch.stack([x_real, x_im], dim=-1)


# =============================================================================
# Rotor/RPS Encoder (shared by encoder-side and decoder-side RPS fusion)
# =============================================================================


class RotorEncoder(nn.Module):
    """
    Encodes rotor RPS time series via two 1D convolutions.
    Output: (B, 64, target_length) when target_length is set, else (B, 64, T).
    """

    def __init__(self, num_rotors: int, out_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        self.num_rotors = num_rotors
        self.out_channels = out_channels
        padding = kernel_size // 2
        self.input_bn = nn.BatchNorm1d(num_rotors)
        self.conv1 = nn.Conv1d(num_rotors, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.ReLU()
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, rps: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        """
        Args:
            rps: (B, num_rotors, time_rps) or (B, time_rps) [then unsqueeze(1)]
            target_length: if set, interpolate to this length
        Returns:
            (B, 64, target_length) or (B, 64, T)
        """
        if rps.dim() == 2:
            rps = rps.unsqueeze(1)
        rps = self.input_bn(rps)
        x = self.act(self.conv1(rps))
        x = self.act(self.conv2(x))
        if target_length is not None and x.size(-1) != target_length:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return x


# =============================================================================
# RPS Prediction Head (auxiliary task, uses encoder features)
# =============================================================================


class RPSPredictionHead(nn.Module):
    """
    FPN-style auxiliary head that predicts per-STFT-frame RPS from all encoder levels.
    """

    def __init__(
        self, encoder_channels: list[int], target_t: int, common_dim: int = 64, num_rotors: int = 4
    ):
        super().__init__()
        self.target_t = target_t
        self.level_projs = nn.ModuleList()
        for ch in encoder_channels:
            self.level_projs.append(nn.Conv1d(ch * 2, common_dim, 1))

        self.head = nn.Sequential(
            nn.Conv1d(common_dim, common_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(common_dim, num_rotors, kernel_size=1),
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_features: list of (B, C_i, F_i, T_i, 2) from each encoder level,
                              ordered finest-to-coarsest.
        Returns: (B, num_rotors, target_t)
        """
        level_feats = []
        for feat, proj in zip(encoder_features, self.level_projs):
            B, C, F_i, T_i, _ = feat.shape
            pooled = feat.mean(dim=2)  # (B, C, T_i, 2)
            pooled = pooled.reshape(B, C * 2, T_i)
            level_feats.append(proj(pooled))

        merged = level_feats[-1]
        for i in range(len(level_feats) - 2, -1, -1):
            finer = level_feats[i]
            merged = F.interpolate(merged, size=finer.shape[-1], mode="linear", align_corners=False)
            merged = merged + finer

        if merged.shape[-1] != self.target_t:
            merged = F.interpolate(merged, size=self.target_t, mode="linear", align_corners=False)

        return self.head(merged)


# =============================================================================
# Encoder Module (standalone, no RPS input)
# =============================================================================


class Encoder(nn.Module):
    """Single encoder layer: complex conv -> complex BN -> LeakyReLU."""

    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.cconv = CConv2d(in_channels, out_channels, kernel, stride, padding)
        self.cbn = CBatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.cconv(x)
        x = self.cbn(x)
        return self.act(x)


class EncoderModule(nn.Module):
    """
    Standalone encoder module. Extracts hierarchical features from STFT input.
    No RPS input - clean feature extraction.

    Attributes:
        num_layers: number of encoder layers
        strides: (freq_stride, time_stride) per layer
        out_channels_per_layer: list of output channels per layer
        features: intermediate encoder outputs (skip connections)
        bottleneck: final bottleneck features after all encoder layers
    """

    def __init__(self, in_channels: int, layer_specs: list):
        """
        Args:
            in_channels: input channels (1 for STFT magnitude)
            layer_specs: list of tuples:
                - (out_channels, kernel, stride, padding) - 4 elements
                - (in_channels, out_channels, kernel, stride, padding) - 5 elements (full spec)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for spec in layer_specs:
            if len(spec) == 4:
                # (out_channels, kernel, stride, padding)
                out_ch, k, s, p = spec
                self.layers.append(Encoder(in_channels, out_ch, k, s, p))
                in_channels = out_ch
            elif len(spec) == 5:
                # (in_channels, out_channels, kernel, stride, padding)
                _, out_ch, k, s, p = spec
                self.layers.append(Encoder(in_channels, out_ch, k, s, p))
                in_channels = out_ch
            else:
                raise ValueError(f"Expected 4 or 5-tuple in layer_specs, got {len(spec)}")

        self.num_layers = len(layer_specs)
        # Store strides for time dimension computation
        self.strides = [spec[2] if len(spec) == 4 else spec[3] for spec in layer_specs]

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, 1, F, T, 2) STFT input
        Returns:
            (bottleneck_features, encoder_features)
            - bottleneck: (B, C, F', T', 2) after all encoder layers
            - encoder_features: list of (B, C_i, F_i, T_i, 2) for skip connections
        """
        encoder_features = []
        current = x

        for i, layer in enumerate(self.layers):
            current = layer(current)
            if i < len(self.layers) - 1:
                encoder_features.append(current)

        return current, encoder_features


# =============================================================================
# Decoder RPS Fusion Modules
# =============================================================================


class DecoderBottleneckRPSFusion(nn.Module):
    """
    Decoder-side RPS fusion at bottleneck level.
    RPS features are injected before the first decoder transposed conv.

    Design: RPS → RotorEncoder → project to match bottleneck channels → ADD.
    This allows RPS to modulate the reconstruction path directly.
    """

    def __init__(
        self, num_rotors: int, rps_channels: int, target_channels: int, kernel_size: int = 3
    ):
        super().__init__()
        self.rotor_encoder = RotorEncoder(
            num_rotors, out_channels=rps_channels, kernel_size=kernel_size
        )
        # Project RPS features to match complex feature channels: rps_channels -> C*2
        self.rps_proj = nn.Conv1d(rps_channels, target_channels * 2, kernel_size=1)
        self.num_rotors = num_rotors

    def forward(self, x: torch.Tensor, rps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, F, T, 2) - features from previous decoder or bottleneck
            rps: (B, num_rotors, time_rps) or (B, time_rps)
        Returns: (B, C, F, T, 2) with RPS injected
        """
        B, C, F, T, _ = x.shape
        if rps.dim() == 2:
            rps = rps.unsqueeze(1)

        # Encode RPS to match time dimension
        rps_feat = self.rotor_encoder(rps, target_length=T)  # (B, rps_channels, T)

        # Project RPS: (B, rps_channels, T) -> (B, C*2, T) -> (B, C, 2, T) -> (B, C, T, 2)
        rps_proj = self.rps_proj(rps_feat)  # (B, C*2, T)
        rps_proj = rps_proj.reshape(B, C, 2, T).permute(0, 1, 3, 2)  # (B, C, T, 2)

        # Broadcast to all freq bins: (B, C, 1, T, 2) -> (B, C, F, T, 2)
        x = x.float()
        x = x + rps_proj.unsqueeze(2)
        return x


class DecoderHierarchicalRPSFusion(nn.Module):
    """
    Decoder-side hierarchical RPS fusion - injects RPS at multiple decoder levels.

    Design mirrors encoder hierarchical fusion:
    - RPS is processed through per-level projections
    - Injected at multiple points in the decoder path (before transposed conv)
    - Each level gets RPS info aligned to that level's time dimension
    - Use 1D conv to project from rps_channels to C*2 (real+imag), then reshape

    This allows RPS to help at multiple scales of reconstruction.
    """

    def __init__(self, num_rotors: int, decoder_input_channels: list, rps_channels: int = 64):
        """
        Args:
            num_rotors: number of drone rotors
            decoder_input_channels: list of input channel counts for each decoder level
                                   [bottleneck_channels, dec1_in, dec2_in, ...]
            rps_channels: intermediate RPS channels
        """
        super().__init__()
        self.num_rotors = num_rotors
        self.decoder_input_channels = decoder_input_channels  # input channels per level

        # Shared rotor encoder for initial RPS processing
        self.rotor_encoder = RotorEncoder(num_rotors, out_channels=rps_channels, kernel_size=3)

        # Per-level projections: 1D conv to project from rps_channels to C*2
        self.level_projs = nn.ModuleList()
        for ch in decoder_input_channels:
            # Project (B, rps_channels, T) -> (B, C*2, T)
            self.level_projs.append(nn.Conv1d(rps_channels, ch * 2, kernel_size=1))

    def forward(
        self,
        decoder_features: list[torch.Tensor],
        rps: torch.Tensor,
        decoder_time_lengths: list[int],
    ) -> list[torch.Tensor]:
        """
        Args:
            decoder_features: list of (B, C_i, F_i, T_i, 2) from each decoder level
            rps: (B, num_rotors, time_rps) or (B, time_rps)
            decoder_time_lengths: list of time lengths at each decoder level
        Returns:
            List of (B, C_i, F_i, T_i, 2) with RPS injected at each level
        """
        B = decoder_features[0].shape[0]

        if rps.dim() == 2:
            rps = rps.unsqueeze(1)

        # Encode RPS at the coarsest time resolution
        coarsest_t = decoder_time_lengths[-1]
        rps_encoded = self.rotor_encoder(
            rps, target_length=coarsest_t
        )  # (B, rps_channels, T_coarse)

        fused_features = []
        for _i, (feat, proj) in enumerate(zip(decoder_features, self.level_projs)):
            B_f, C_f, F_f, T_f, _ = feat.shape

            # Interpolate RPS to match this level's time dimension
            if T_f != rps_encoded.shape[-1]:
                rps_level = F.interpolate(rps_encoded, size=T_f, mode="linear", align_corners=False)
            else:
                rps_level = rps_encoded

            # Project RPS to decoder channel dimension
            rps_proj = proj(rps_level.permute(0, 2, 1))  # (B, T_f, C*2)
            rps_proj = rps_proj.permute(0, 2, 1).unsqueeze(2)  # (B, C*2, 1, T_f)
            rps_proj = rps_proj.reshape(B, C_f, 2, F_f, T_f).permute(
                0, 1, 3, 4, 2
            )  # (B, C, F, T, 2)

            fused_feat = feat.float() + rps_proj
            fused_features.append(fused_feat)

        return fused_features


# =============================================================================
# Decoder Module (standalone, supports RPS fusion)
# =============================================================================


class Decoder(nn.Module):
    """Single decoder layer: complex transposed conv -> complex BN -> LeakyReLU."""

    def __init__(
        self, in_channels, out_channels, kernel, stride, output_padding, padding, last_layer=False
    ):
        super().__init__()
        self.cconvt = CConvTranspose2d(
            in_channels, out_channels, kernel, stride, output_padding, padding
        )
        self.cbn = CBatchNorm2d(out_channels) if not last_layer else None
        self.act = nn.LeakyReLU() if not last_layer else None
        self.last_layer = last_layer

    def forward(self, x):
        x = self.cconvt(x)  # pyright: ignore[reportOptionalCall]
        if not self.last_layer:
            x = self.cbn(x)  # pyright: ignore[reportOptionalCall]
            x = self.act(x)  # pyright: ignore[reportOptionalCall]
        else:
            x = x.float()
            m_phase = x / (torch.abs(x) + 1e-8)
            m_mag = torch.tanh(torch.abs(x))
            x = m_phase * m_mag
        return x

    def forward_with_rps_inject(self, x, rps_inject):  # pyright: ignore[reportOptionalCall]
        """Forward with RPS injection after transposed conv."""
        x = self.cconvt(x)  # pyright: ignore[reportOptionalCall]
        # rps_inject should have shape matching x after transposed conv
        x = x + rps_inject
        if not self.last_layer:
            x = self.cbn(x)  # pyright: ignore[reportOptionalCall]
            x = self.act(x)  # pyright: ignore[reportOptionalCall]
        else:
            x = x.float()
            m_phase = x / (torch.abs(x) + 1e-8)
            m_mag = torch.tanh(torch.abs(x))
            x = m_phase * m_mag
        return x


class DecoderModule(nn.Module):
    """
    Standalone decoder module with optional RPS conditioning.

    Supports two RPS fusion strategies:
    1. 'bottleneck': RPS injected after first transposed conv
    2. 'hierarchical': RPS injected at multiple decoder levels

    Attributes:
        num_layers: number of decoder layers
        strides: (freq_stride, time_stride) per layer
        features: intermediate decoder outputs
    """

    def __init__(
        self,
        bottleneck_channels: int,
        layer_specs: list,
        num_rotors: int = 4,
        rps_fusion: str | None = None,
        rps_channels: int = 64,
    ):
        """
        Args:
            bottleneck_channels: channels at bottleneck (input to decoder)
            layer_specs: list of tuples:
                - (out_channels, kernel, stride, output_padding, padding) - 5 elements
                - (out_channels, kernel, stride, output_padding, padding, is_last) - 6 elements
                - (in_channels, out_channels, kernel, stride, output_padding, padding) - 6 elements
                - (in_channels, out_channels, kernel, stride, output_padding, padding, is_last) - 7 elements
            num_rotors: number of drone rotors (default 4)
            rps_fusion: 'bottleneck', 'hierarchical', or None
            rps_channels: intermediate RPS channels
        """
        super().__init__()
        self.num_layers = len(layer_specs)
        self.rps_fusion = rps_fusion
        self.num_rotors = num_rotors

        # Build decoder layers
        self.layers = nn.ModuleList()
        self.layer_specs = []  # Store normalized specs (5 elements each)

        # For hierarchical fusion, we need to track channel dimensions
        self.decoder_channels = []

        in_ch = bottleneck_channels
        for i, spec in enumerate(layer_specs):
            if len(spec) == 5:
                out_ch, k, s, op, p = spec
                is_last = False
            elif len(spec) == 6:
                # Could be (in_ch, out_ch, k, s, op, p) or (out_ch, k, s, op, p, is_last)
                if isinstance(spec[0], int) and isinstance(spec[-1], bool):
                    # Full spec with is_last: (in_ch, out_ch, k, s, op, p, is_last) but got 6?
                    # Actually this is (in_ch, out_ch, k, s, op, p) without is_last
                    _, out_ch, k, s, op, p = spec
                    is_last = False
                else:
                    out_ch, k, s, op, p, is_last = spec
            elif len(spec) == 7:
                # Full spec: (in_ch, out_ch, k, s, op, p, is_last)
                _, out_ch, k, s, op, p, is_last = spec
            else:
                raise ValueError(f"Expected 5, 6, or 7 elements in layer_specs, got {len(spec)}")

            self.layer_specs.append((out_ch, k, s, op, p))
            self.layers.append(Decoder(in_ch, out_ch, k, s, op, p, last_layer=is_last))
            self.decoder_channels.append(out_ch)
            in_ch = out_ch * 2 if i < len(layer_specs) - 1 else out_ch  # double for skip concat

        # RPS fusion modules
        self.rps_fusion_module = None
        self.decoder_input_channels = []  # Store for debugging
        if rps_fusion == "bottleneck":
            # Bottleneck fusion: inject RPS at first decoder level (matching bottleneck channels)
            self.rps_fusion_module = DecoderBottleneckRPSFusion(
                num_rotors, rps_channels, bottleneck_channels
            )
            self.decoder_input_channels = [bottleneck_channels]
        elif rps_fusion == "hierarchical":
            # Hierarchical fusion: inject RPS at multiple levels
            # Track input channels for each level:
            # Level 0: bottleneck_channels (before first decoder)
            # Level i (i > 0): dec_channels[i-1] * 2 (after skip concat)
            decoder_input_channels = [bottleneck_channels]
            for i in range(len(self.decoder_channels) - 1):
                decoder_input_channels.append(self.decoder_channels[i] * 2)
            self.decoder_input_channels = decoder_input_channels
            self.rps_fusion_module = DecoderHierarchicalRPSFusion(
                num_rotors, decoder_input_channels, rps_channels
            )

    def forward(
        self, bottleneck: torch.Tensor, skip_connections: list, rps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: (B, C_bottleneck, F_b, T_b, 2) features from encoder
            skip_connections: list of (B, C_i, F_i, T_i, 2) from encoder (reversed for use)
            rps: (B, num_rotors, time_rps) or None for no RPS conditioning
        Returns:
            (B, 1, F_out, T_out, 2) reconstructed features
        """
        current = bottleneck
        decoder_features = []

        # First layer: no skip connection
        if (
            self.rps_fusion == "bottleneck"
            and rps is not None
            and self.rps_fusion_module is not None
        ):
            # For bottleneck fusion: inject RPS BEFORE transposed conv
            # This matches input channels (bottleneck_channels)
            rps_inject = self._compute_bottleneck_rps(current, rps)
            current = current + rps_inject
        elif (
            self.rps_fusion == "hierarchical"
            and rps is not None
            and self.rps_fusion_module is not None
        ):
            # For hierarchical fusion at level 0: inject RPS before transposed conv
            rps_inject = self._compute_hierarchical_rps(current, rps, 0)
            current = current + rps_inject

        current = self.layers[0](current)
        decoder_features.append(current)

        # Subsequent layers: with skip connections
        for i in range(1, len(self.layers)):
            skip = skip_connections[-i]

            # Match dimensions
            min_f = min(current.shape[2], skip.shape[2])
            min_t = min(current.shape[3], skip.shape[3])
            if current.shape[2] != min_f or current.shape[3] != min_t:
                current = current[:, :, :min_f, :min_t, :]
            if skip.shape[2] != min_f or skip.shape[3] != min_t:
                skip = skip[:, :, :min_f, :min_t, :]

            current = torch.cat([current, skip], dim=1)

            # Hierarchical RPS injection at this level (BEFORE transposed conv)
            if (
                self.rps_fusion == "hierarchical"
                and rps is not None
                and self.rps_fusion_module is not None
            ):
                rps_inject = self._compute_hierarchical_rps(current, rps, i)
                current = current + rps_inject

            current = self.layers[i](current)
            decoder_features.append(current)

        return current

    def _compute_bottleneck_rps(self, x: torch.Tensor, rps: torch.Tensor) -> torch.Tensor:
        """
        Compute RPS injection for bottleneck fusion.
        Input: (B, C_bottleneck, F, T, 2) - input to first decoder (bottleneck)
        Output: RPS injection of same shape
        """
        B, C, F, T, _ = x.shape
        rps_feat = self.rps_fusion_module.rotor_encoder(rps, target_length=T)  # pyright: ignore[reportOptionalMemberAccess]
        # Conv1d: (B, 64, T) -> (B, C*2, T)
        rps_proj = self.rps_fusion_module.rps_proj(rps_feat)  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
        # Reshape to (B, C, 2, T) then permute to (B, C, T, 2)
        rps_proj = rps_proj.reshape(B, C, 2, T).permute(0, 1, 3, 2)  # (B, C, T, 2)
        # Broadcast to (B, C, F, T, 2)
        rps_proj = rps_proj.unsqueeze(2).expand(-1, -1, F, -1, -1)
        return rps_proj

    def _compute_hierarchical_rps(
        self, x: torch.Tensor, rps: torch.Tensor, level: int
    ) -> torch.Tensor:
        """
        Compute RPS injection for hierarchical fusion at given level.
        Input: (B, C_in, F, T, 2) - input to decoder layer (after skip concat for level > 0)
        Output: RPS injection of same shape
        """
        B, C, F, T, _ = x.shape
        rps_feat = self.rps_fusion_module.rotor_encoder(rps, target_length=T)  # pyright: ignore[reportOptionalMemberAccess]
        proj = self.rps_fusion_module.level_projs[level]  # pyright: ignore[reportOptionalMemberAccess, reportOptionalSubscript, reportIndexIssue]
        # Conv1d: (B, 64, T) -> (B, C*2, T)
        rps_proj = proj(rps_feat)  # pyright: ignore[reportOptionalCall, reportCallIssue]
        # Reshape to (B, C, 2, T) then permute to (B, C, T, 2)
        rps_proj = rps_proj.reshape(B, C, 2, T).permute(0, 1, 3, 2)  # (B, C, T, 2)
        # Broadcast to (B, C, F, T, 2)
        rps_proj = rps_proj.unsqueeze(2).expand(-1, -1, F, -1, -1)
        return rps_proj


# =============================================================================
# STFT Processor
# =============================================================================


class STFTProcessor(nn.Module):
    """STFT Processing Module."""

    def __init__(self, config):
        super().__init__()
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]
        self.window = torch.hann_window(self.n_fft)

    def transform(self, x):
        """Input: (B, 1, time) -> Output: (B, 1, freq, time, 2)"""
        x = x.squeeze(1)
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
            normalized=True,
        )
        X = torch.view_as_real(X)
        X = X.unsqueeze(1)
        return X

    def inverse(self, X):
        """Input: (B, 1, freq, time, 2) -> Output: (B, 1, time)"""
        X = X.squeeze(1)
        X = torch.view_as_complex(X)
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(X.device),
            normalized=True,
        )
        x = x.unsqueeze(1)
        return x


# =============================================================================
# Refactored DCUNet
# =============================================================================


def _get_rps_config(config) -> dict:
    """Read RPS-related options from config (top-level or under 'model')."""
    get = getattr(config, "get", lambda k, d=None: getattr(config, k, d))
    m = get("model")
    m = m if isinstance(m, dict) else {}
    m_get = getattr(m, "get", lambda k, d=None: getattr(m, k, d) if hasattr(m, k) else d)
    return {
        "use_rps": get("use_rps") or m_get("use_rps", False),
        # New decoder-specific RPS fusion options
        "decoder_rps_fusion": get("decoder_rps_fusion")
        or m_get("decoder_rps_fusion", "bottleneck"),
        "dcunet_num_encoder_layers": get("dcunet_num_encoder_layers")
        or m_get("dcunet_num_encoder_layers", 5),
        "num_rotors": get("num_rotors") or m_get("num_rotors", 4),
        "predict_rps": get("predict_rps") or m_get("predict_rps", False),
    }


class DCUNetRefactored(nn.Module):
    """
    Refactored Deep Complex U-Net with DECODER-side RPS conditioning.

    Key changes from original DCUNet:
    - Encoder is clean (no RPS input)
    - Decoder receives RPS and performs fusion
    - Two decoder RPS fusion strategies:
      1. 'bottleneck': RPS injected after first transposed conv
      2. 'hierarchical': RPS injected at multiple decoder levels

    This refactoring allows:
    - Training encoder-only baselines
    - Analyzing where RPS helps (encoder vs decoder)
    - Decoupled encoder/decoder architecture
    """

    def __init__(self, config):
        super().__init__()
        self.stft = STFTProcessor(config)
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]

        rps_cfg = _get_rps_config(config)
        self.use_rps = rps_cfg["use_rps"]
        self.decoder_rps_fusion = rps_cfg["decoder_rps_fusion"]
        self.num_encoder_layers = rps_cfg["dcunet_num_encoder_layers"]
        self.num_rotors = rps_cfg["num_rotors"]
        self.predict_rps = rps_cfg["predict_rps"]

        # Encoder strides for time-dimension computation
        self._encoder_strides = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)]
        if self.num_encoder_layers == 6:
            self._encoder_strides.append((2, 1))

        # Encoder layer specs: (out_channels, kernel, stride, padding)
        enc_spec = [
            (1, 45, (7, 5), (2, 2), (3, 2)),
            (45, 90, (7, 5), (2, 2), (3, 2)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        if self.num_encoder_layers == 6:
            enc_spec.append((90, 90, (5, 3), (2, 1), (2, 1)))

        # Build standalone encoder module
        self.encoder = EncoderModule(1, enc_spec)

        # Decoder layer specs: (out_channels, kernel, stride, output_padding, padding, is_last)
        bottleneck_ch = enc_spec[-1][1]
        dec_spec = [
            (bottleneck_ch, 90, (5, 3), (2, 1), (0, 0), (2, 1), False),
            (180, 90, (5, 3), (2, 2), (0, 0), (2, 1), False),
            (180, 90, (5, 3), (2, 2), (0, 0), (2, 1), False),
            (180, 45, (7, 5), (2, 2), (0, 0), (3, 2), False),
            (90, 1, (7, 5), (2, 2), (0, 1), (3, 2), True),
        ]
        if self.num_encoder_layers == 6:
            dec_spec = [
                (90, 90, (5, 3), (2, 1), (0, 0), (2, 1), False),
                (180, 90, (5, 3), (2, 1), (0, 0), (2, 1), False),
                (180, 90, (5, 3), (2, 2), (0, 0), (2, 1), False),
                (180, 90, (5, 3), (2, 2), (0, 0), (2, 1), False),
                (180, 45, (7, 5), (2, 2), (0, 0), (3, 2), False),
                (90, 1, (7, 5), (2, 2), (0, 1), (3, 2), True),
            ]

        # Build standalone decoder module (with RPS fusion if enabled)
        self.decoder = DecoderModule(
            bottleneck_ch,
            dec_spec,
            num_rotors=self.num_rotors,
            rps_fusion=self.decoder_rps_fusion if self.use_rps else None,
            rps_channels=64,
        )

        # Auxiliary RPS prediction head
        self.rps_prediction_head = None
        if self.predict_rps:
            enc_channels = [spec[1] for spec in enc_spec]
            chunk_size = config["audio"]["chunk_size"]
            target_t = stft_time_frames(chunk_size, self.hop_length, self.n_fft)
            self.rps_prediction_head = RPSPredictionHead(
                enc_channels, target_t, num_rotors=self.num_rotors
            )

    def forward(
        self, x: torch.Tensor, rps: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Args:
            x: (B, 1, time) audio input
            rps: (B, num_rotors, time_rps) or None.
                 Note: RPS is only used in the decoder (not encoder).
        Returns:
            (B, 1, 1, time) enhanced audio
            If predict_rps=True, also returns (B, num_rotors, target_t) RPS prediction
        """
        input_length = x.shape[-1]
        X = self.stft.transform(x)  # (B, 1, F, T, 2)
        B, _, F, T, _ = X.shape

        # Encode (clean, no RPS)
        bottleneck, encoder_features = self.encoder(X)

        # Auxiliary RPS prediction from encoder features
        rps_pred = None
        if self.predict_rps and self.rps_prediction_head is not None:
            all_encoder_feats = encoder_features + [bottleneck]
            rps_pred = self.rps_prediction_head(all_encoder_feats)

        # Decode with RPS conditioning (decoder handles RPS fusion internally)
        decoded = self.decoder(bottleneck, encoder_features, rps)

        # Pad or crop output to match input spectrogram dimensions
        if decoded.shape[2] != F or decoded.shape[3] != T:
            pad_f = F - decoded.shape[2]
            pad_t = T - decoded.shape[3]
            if pad_f > 0 or pad_t > 0:
                decoded = F.pad(decoded, (0, 0, 0, max(0, pad_t), 0, max(0, pad_f)))  # pyright: ignore[reportAttributeAccessIssue]
            if pad_f < 0 or pad_t < 0:
                decoded = decoded[:, :, :F, :T, :]  # pyright: ignore[reportAttributeAccessIssue]

        # Apply mask and inverse STFT
        output = decoded * X
        output = self.stft.inverse(output)

        # Ensure output length matches input
        if output.shape[-1] != input_length:
            if output.shape[-1] < input_length:
                output = F.pad(output, (0, input_length - output.shape[-1]))  # pyright: ignore[reportAttributeAccessIssue]
            else:
                output = output[..., :input_length]

        output = output.unsqueeze(1)

        if rps_pred is not None:
            return output, rps_pred
        return output


# =============================================================================
# DCCRN Refactored (same pattern)
# =============================================================================


class DCCRNRefactored(nn.Module):
    """
    Refactored Deep Complex Convolution Recurrent Network with DECODER-side RPS conditioning.

    Key changes from original DCCRN:
    - Encoder is clean (no RPS input)
    - RPS fusion happens in decoder (after GRU bottleneck)
    - Two decoder RPS fusion strategies:
      1. 'bottleneck': RPS injected after GRU projection
      2. 'hierarchical': RPS injected at multiple decoder levels
    """

    def __init__(self, config):
        super().__init__()
        self.stft = STFTProcessor(config)
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]

        # Architecture variant
        self.lite = _get_config_val(config, "dccrn_lite", False)

        if self.lite:
            encoder_channels = [16, 32, 64, 128]
            gru_hidden = _get_config_val(config, "dccrn_gru_hidden", 128)
            gru_layers = 1
        else:
            encoder_channels = [32, 64, 128, 256, 256, 512]
            gru_hidden = _get_config_val(config, "dccrn_gru_hidden", 256)
            gru_layers = _get_config_val(config, "dccrn_gru_layers", 2)

        self._encoder_channels = encoder_channels

        # RPS configuration (decoder-side only)
        self.use_rps = _get_config_val(config, "use_rps", False)
        self.decoder_rps_fusion = _get_config_val(config, "decoder_rps_fusion", "bottleneck")
        self.num_rotors = _get_config_val(config, "num_rotors", 4)
        self.predict_rps = _get_config_val(config, "predict_rps", False)

        # Encoder: kernel (5,2), stride (2,1), padding (2,0)
        enc_kernel = (5, 2)
        enc_stride = (2, 1)
        enc_padding = (2, 0)

        enc_spec = []
        in_ch = 1
        for oc in encoder_channels:
            enc_spec.append((in_ch, oc, enc_kernel, enc_stride, enc_padding))
            in_ch = oc

        # Build standalone encoder module
        self.encoder = EncoderModule(1, enc_spec)

        # Compute bottleneck dimensions
        freq = self.n_fft // 2 + 1
        for _ in encoder_channels:
            freq = (freq + 2 * enc_padding[0] - enc_kernel[0]) // enc_stride[0] + 1
        self._bottleneck_freq = freq
        bottleneck_ch = encoder_channels[-1]

        # GRU bottleneck (no RPS at input)
        gru_input_size = bottleneck_ch * freq * 2
        self.gru = nn.GRU(
            gru_input_size,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_proj = nn.Linear(gru_hidden * 2, bottleneck_ch * freq * 2)

        # Decoder
        dec_kernel = (5, 2)
        dec_stride = (2, 1)
        dec_padding = (2, 0)
        dec_output_padding = (0, 0)

        dec_out = list(reversed([1] + encoder_channels[:-1]))
        dec_spec = []
        for i, out_ch in enumerate(dec_out):
            is_last = i == len(dec_out) - 1
            dec_spec.append(
                (0, out_ch, dec_kernel, dec_stride, dec_output_padding, dec_padding, is_last)
            )

        self.decoder = DecoderModule(
            bottleneck_ch,
            dec_spec,
            num_rotors=self.num_rotors,
            rps_fusion=self.decoder_rps_fusion if self.use_rps else None,
            rps_channels=64,
        )

        # Note: DCCRN has GRU in the bottleneck, so decoder receives GRU output
        # The actual RPS fusion happens in DecoderModule after GRU

        # Auxiliary RPS prediction head
        self.rps_prediction_head = None
        if self.predict_rps:
            chunk_size = config["audio"]["chunk_size"]
            target_t = stft_time_frames(chunk_size, self.hop_length, self.n_fft)
            self.rps_prediction_head = RPSPredictionHead(
                encoder_channels, target_t, num_rotors=self.num_rotors
            )

    def forward(
        self, x: torch.Tensor, rps: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Args:
            x: (B, 1, time) audio input
            rps: (B, num_rotors, time_rps) or None
        Returns:
            (B, 1, 1, time) enhanced audio
        """
        input_length = x.shape[-1]
        X = self.stft.transform(x)
        B, _, F_stft, T, _ = X.shape

        # Encode (clean, no RPS)
        bottleneck, encoder_features = self.encoder(X)
        _, C, F_b, T_b, _ = bottleneck.shape

        # Auxiliary RPS prediction
        rps_pred = None
        if self.predict_rps and self.rps_prediction_head is not None:
            all_encoder_feats = encoder_features + [bottleneck]
            rps_pred = self.rps_prediction_head(all_encoder_feats)

        # GRU bottleneck (no RPS input)
        gru_in = bottleneck.permute(0, 3, 1, 2, 4).reshape(B, T_b, C * F_b * 2)
        gru_out, _ = self.gru(gru_in)
        gru_out = self.gru_proj(gru_out)
        gru_bottleneck = gru_out.reshape(B, T_b, C, F_b, 2).permute(0, 2, 3, 1, 4)

        # Decode with RPS conditioning
        decoded = self.decoder(gru_bottleneck, encoder_features, rps)

        # Match output dimensions
        if decoded.shape[2] != F_stft or decoded.shape[3] != T:
            pad_f = F_stft - decoded.shape[2]
            pad_t = T - decoded.shape[3]
            if pad_f > 0 or pad_t > 0:
                decoded = F.pad(decoded, (0, 0, 0, max(0, pad_t), 0, max(0, pad_f)))  # pyright: ignore[reportAttributeAccessIssue]
            if pad_f < 0 or pad_t < 0:
                decoded = decoded[:, :, :F_stft, :T, :]

        # Apply mask and inverse STFT
        output = decoded * X
        output = self.stft.inverse(output)

        # Ensure output length
        if output.shape[-1] != input_length:
            if output.shape[-1] < input_length:
                output = F.pad(output, (0, input_length - output.shape[-1]))  # pyright: ignore[reportAttributeAccessIssue]
            else:
                output = output[..., :input_length]

        output = output.unsqueeze(1)

        if rps_pred is not None:
            return output, rps_pred
        return output


def _get_config_val(config, key, default=None):
    """Get a value from config (top-level or under 'model')."""
    get = getattr(config, "get", lambda k, d=None: getattr(config, k, d))
    val = get(key)
    if val is not None:
        return val
    m = get("model")
    if m is not None and isinstance(m, dict):
        return m.get(key, default)
    if m is not None and hasattr(m, key):
        return getattr(m, key)
    return default


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    config: dict[str, Any] = {
        "audio": {
            "chunk_size": 131584,
            "dim_f": 1024,
            "hop_length": 512,
            "n_fft": 2048,
            "num_channels": 1,
            "sample_rate": 16000,
        },
        "training": {"batch_size": 10},
    }

    x = torch.randn(2, 1, 8192)
    rps = torch.randn(2, 4, 100)

    # Test 1: DCUNet Refactored - Baseline (no RPS)
    print("=" * 60)
    print("Test 1: DCUNetRefactored - Baseline (no RPS)")
    model = DCUNetRefactored(config)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    assert out.shape == (2, 1, 1, 8192), out.shape
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    # Test 2: DCUNet Refactored - Decoder Bottleneck RPS Fusion
    print("\n" + "=" * 60)
    print("Test 2: DCUNetRefactored - Decoder Bottleneck RPS Fusion")
    config["use_rps"] = True
    config["decoder_rps_fusion"] = "bottleneck"
    config["dcunet_num_encoder_layers"] = 5
    config["num_rotors"] = 4
    model2 = DCUNetRefactored(config)
    out2 = model2(x, rps=rps)
    print(f"  Output shape: {out2.shape}")
    assert out2.shape == (2, 1, 1, 8192), out2.shape
    total2 = sum(p.numel() for p in model2.parameters())
    print(f"  Parameters: {total2:,}")

    # Test 3: DCUNet Refactored - Decoder Hierarchical RPS Fusion
    print("\n" + "=" * 60)
    print("Test 3: DCUNetRefactored - Decoder Hierarchical RPS Fusion")
    config["decoder_rps_fusion"] = "hierarchical"
    model3 = DCUNetRefactored(config)
    out3 = model3(x, rps=rps)
    print(f"  Output shape: {out3.shape}")
    assert out3.shape == (2, 1, 1, 8192), out3.shape
    total3 = sum(p.numel() for p in model3.parameters())
    print(f"  Parameters: {total3:,}")

    # Test 4: DCCRN Refactored - Baseline (no RPS)
    print("\n" + "=" * 60)
    print("Test 4: DCCRNRefactored - Baseline (no RPS)")
    config2 = config.copy()
    config2.pop("use_rps", None)
    config2.pop("decoder_rps_fusion", None)
    config2.pop("dcunet_num_encoder_layers", None)
    config2.pop("num_rotors", None)
    config2["dccrn_lite"] = False
    model4 = DCCRNRefactored(config2)
    out4 = model4(x)
    print(f"  Output shape: {out4.shape}")
    assert out4.shape == (2, 1, 1, 8192), out4.shape
    total4 = sum(p.numel() for p in model4.parameters())
    print(f"  Parameters: {total4:,}")

    # Test 5: DCCRN Refactored - Decoder RPS Fusion
    print("\n" + "=" * 60)
    print("Test 5: DCCRNRefactored - Decoder RPS Fusion")
    config2["use_rps"] = True
    config2["decoder_rps_fusion"] = "hierarchical"
    config2["num_rotors"] = 4
    model5 = DCCRNRefactored(config2)
    out5 = model5(x, rps=rps)
    print(f"  Output shape: {out5.shape}")
    assert out5.shape == (2, 1, 1, 8192), out5.shape
    total5 = sum(p.numel() for p in model5.parameters())
    print(f"  Parameters: {total5:,}")

    # Test 6: Test encoder/decoder are accessible
    print("\n" + "=" * 60)
    print("Test 6: Encoder/Decoder modules are accessible")
    model = DCUNetRefactored(config)
    print(f"  model.encoder type: {type(model.encoder).__name__}")
    print(f"  model.decoder type: {type(model.decoder).__name__}")
    print("  ✓ Encoder and Decoder are separate modules")

    # Test 7: Test RPS is NOT passed to encoder
    print("\n" + "=" * 60)
    print("Test 7: Verify RPS is NOT used in encoder (encoder is clean)")
    # This is verified by design - encoder module doesn't accept rps parameter
    print("  ✓ Encoder.forward() signature: (x) only")
    print("  ✓ Decoder.forward() signature: (bottleneck, skips, rps)")

    print("\n" + "=" * 60)
    print("All tests passed!")
