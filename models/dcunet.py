"""Deep Complex U-Net with optional RPS conditioning.

Model types registered as ``dcunet`` and ``dcunet_enc_rps`` (with RPS prediction).

When ``rps`` is None the model behaves as the baseline (Paper 1) DCUNet.
"""

import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def stft_time_frames(audio_length: int, hop_length: int, n_fft: int) -> int:
    """Number of STFT time frames for a given audio length."""
    return (audio_length - n_fft) // hop_length + 1


def encoder_time_lengths(
    n_stft_time: int,
    encoder_strides: list[tuple[int, int]],
) -> list[int]:
    """Time dimension at each encoder level.

    Level 0 = STFT input, level i = after i-th encoder.
    encoder_strides: list of (stride_f, stride_t) per layer.
    """
    lengths = [n_stft_time]
    for _, stride_t in encoder_strides:
        lengths.append((lengths[-1] + 1) // stride_t)
    return lengths


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

_RPS_FUSION_TYPES = ("bottleneck", "gru", "hierarchical")


def _get_rps_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract RPS options from a flat dict or DictConfig-style config.

    Looks under top-level keys and under an optional ``model`` sub-dict.
    """
    model_cfg: dict[str, Any] = (
        config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    )
    return {
        "use_rps": config.get("use_rps", False) or model_cfg.get("use_rps", False),
        "dcunet_rps_fusion": config.get("dcunet_rps_fusion")
        or model_cfg.get("dcunet_rps_fusion", "bottleneck"),
        "dcunet_num_encoder_layers": config.get("dcunet_num_encoder_layers")
        or model_cfg.get("dcunet_num_encoder_layers", 5),
        "num_rotors": config.get("num_rotors") or model_cfg.get("num_rotors", 4),
    }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RotorEncoder(nn.Module):
    """2-layer 1D convolutional encoder for rotor RPM time series.

    Input:  (B, num_rotors, time_rps)  or  (B, time_rps)
    Output: (B, 64, target_length)     or  (B, 64, T)  (without interpolation)
    """

    def __init__(self, num_rotors: int, out_channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_bn = nn.BatchNorm1d(num_rotors)
        self.conv1 = nn.Conv1d(num_rotors, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.ReLU()
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, rps: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        if rps.dim() == 2:
            rps = rps.unsqueeze(1)
        rps = self.input_bn(rps)
        x = self.act(self.conv1(rps))
        x = self.act(self.conv2(x))
        if target_length is not None and x.size(-1) != target_length:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return x


class RPSPredictionHead(nn.Module):
    """FPN-style auxiliary head: predicts per-STFT-frame RPS from all encoder levels."""

    def __init__(
        self,
        encoder_channels: list[int],
        target_t: int,
        common_dim: int = 64,
        num_rotors: int = 4,
    ) -> None:
        super().__init__()
        self.target_t = target_t
        self.level_projs = nn.ModuleList(
            [
                nn.Conv1d(ch * 2, common_dim, 1)
                for ch in encoder_channels  # ×2 for real+imag
            ]
        )
        self.head = nn.Sequential(
            nn.Conv1d(common_dim, common_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(common_dim, num_rotors, kernel_size=1),
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """encoder_features: finest→coarsest, each (B, C, F, T, 2).  Returns (B, num_rotors, target_t)."""
        level_feats: list[torch.Tensor] = []
        for feat, proj in zip(encoder_features, self.level_projs):
            B, C, _, T_i, _ = feat.shape
            pooled = feat.mean(dim=2)  # (B, C, T_i, 2)
            pooled = pooled.reshape(B, C * 2, T_i)
            level_feats.append(proj(pooled))

        merged = level_feats[-1]
        for i in range(len(level_feats) - 2, -1, -1):
            merged = F.interpolate(
                merged, size=level_feats[i].shape[-1], mode="linear", align_corners=False
            )
            merged = merged + level_feats[i]

        if merged.shape[-1] != self.target_t:
            merged = F.interpolate(merged, size=self.target_t, mode="linear", align_corners=False)

        return self.head(merged)


# ---------------------------------------------------------------------------
# Complex-valued layers
# ---------------------------------------------------------------------------


class CConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.im_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, T, 2) → (B, C, F, T, 2)"""
        r = x[..., 0]
        i = x[..., 1]
        return torch.stack(
            [
                self.real_conv(r) - self.im_conv(i),
                self.im_conv(r) + self.real_conv(i),
            ],
            dim=-1,
        )


class CConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        output_padding: int | tuple[int, int] = 0,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.real_convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.im_convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, T, 2) → (B, C, F, T, 2)"""
        r = x[..., 0]
        i = x[..., 1]
        return torch.stack(
            [
                self.real_convt(r) - self.im_convt(i),
                self.im_convt(r) + self.real_convt(i),
            ],
            dim=-1,
        )


class CBatchNorm2d(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.im_bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, T, 2) → (B, C, F, T, 2)"""
        return torch.stack(
            [
                self.real_bn(x[..., 0]),
                self.im_bn(x[..., 1]),
            ],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.cconv = CConv2d(in_channels, out_channels, kernel, stride, padding)
        self.cbn = CBatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.cbn(self.cconv(x)))


class DecoderBlock(nn.Module):
    """Inner decoder block with complex transpose conv + BN + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        output_padding: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.cconvt = CConvTranspose2d(
            in_channels, out_channels, kernel, stride, output_padding, padding
        )
        self.cbn = CBatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.cbn(self.cconvt(x)))


class DecoderLast(nn.Module):
    """Final decoder block: transposed conv then magnitude/phase in float32."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        output_padding: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.cconvt = CConvTranspose2d(
            in_channels, out_channels, kernel, stride, output_padding, padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cconvt(x)
        # Compute phase and magnitude in float32 for numerical stability
        # under AMP (float16), where eps < 6e-5 is rounded to 0 causing NaN
        x = x.float()
        m_phase = x / (torch.abs(x) + 1e-8)
        m_mag = torch.tanh(torch.abs(x))
        return m_phase * m_mag


# ---------------------------------------------------------------------------
# STFT processor
# ---------------------------------------------------------------------------


class STFTProcessor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]
        self.dim_f = config["audio"]["dim_f"]
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.window: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, 1, F, T, 2)"""
        x = x.squeeze(1)
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
            normalized=True,
        )
        X = torch.view_as_real(X)  # (B, F, T, 2)
        return X.unsqueeze(1)  # (B, 1, F, T, 2)

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        """(B, 1, F, T, 2) → (B, 1, T)"""
        X = X.squeeze(1)  # (B, F, T, 2)
        X = torch.view_as_complex(X)
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(X.device),
            normalized=True,
        )
        return x.unsqueeze(1)  # (B, 1, T)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

# Type alias for encoder/decoder spec tuples
_EncSpec = tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]]
_DecSpec = tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


class DCUNet(nn.Module):
    """Deep Complex U-Net with optional RPS conditioning.

    When ``rps is None`` the model behaves as the baseline DCUNet (Paper 1).
    """

    # ── typed Optional module slots ──
    rotor_encoder: RotorEncoder | None
    rps_bottleneck_proj: nn.Linear | None
    rps_gru: nn.GRU | None
    rps_gru_proj: nn.Linear | None
    rps_hierarchical_blocks: nn.ModuleList | None
    rps_hierarchical_projs: nn.ModuleList | None
    rps_prediction_head: RPSPredictionHead | None

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.stft = STFTProcessor(config)

        self.output_channels: int = config["audio"]["num_channels"]
        self.n_fft: int = config["audio"]["n_fft"]
        self.hop_length: int = config["audio"]["hop_length"]

        # ── RPS config ──
        rps_cfg = _get_rps_config(config)
        self.use_rps: bool = rps_cfg["use_rps"]
        self.rps_fusion: str = rps_cfg["dcunet_rps_fusion"]
        self.num_encoder_layers: int = rps_cfg["dcunet_num_encoder_layers"]
        self.num_rotors: int = rps_cfg["num_rotors"]

        # ── Encoder strides (for hierarchical timing) ──
        self._encoder_strides: list[tuple[int, int]] = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)]
        if self.num_encoder_layers == 6:
            self._encoder_strides.append((2, 1))

        # ── Encoder ──
        enc_spec: list[_EncSpec] = [
            (1, 45, (7, 5), (2, 2), (3, 2)),
            (45, 90, (7, 5), (2, 2), (3, 2)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        if self.num_encoder_layers == 6:
            enc_spec.append((90, 90, (5, 3), (2, 1), (2, 1)))
        self.encoders = nn.ModuleList(
            [Encoder(ic, oc, k, s, padding=p) for ic, oc, k, s, p in enc_spec]
        )

        # ── Decoder ──
        bottleneck_ch = enc_spec[-1][1]
        dec_spec: list[_DecSpec] = [
            (bottleneck_ch, 90, (5, 3), (2, 1), (0, 0), (2, 1)),
            (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
            (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
            (180, 45, (7, 5), (2, 2), (0, 0), (3, 2)),
            (90, 1, (7, 5), (2, 2), (0, 1), (3, 2)),  # last
        ]
        if self.num_encoder_layers == 6:
            dec_spec = [
                (bottleneck_ch, 90, (5, 3), (2, 1), (0, 0), (2, 1)),
                (180, 90, (5, 3), (2, 1), (0, 0), (2, 1)),
                (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
                (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
                (180, 45, (7, 5), (2, 2), (0, 0), (3, 2)),
                (90, 1, (7, 5), (2, 2), (0, 1), (3, 2)),  # last
            ]
        self.decoders = nn.ModuleList(
            [DecoderBlock(*dec_spec[i]) for i in range(len(dec_spec) - 1)]
        )
        last = dec_spec[-1]
        self.decoders.append(
            DecoderLast(last[0], last[1], last[2], last[3], output_padding=last[4], padding=last[5])
        )

        # ── RPS pathway ──
        self.rotor_encoder = None
        self.rps_bottleneck_proj = None
        self.rps_gru = None
        self.rps_gru_proj = None
        self.rps_hierarchical_blocks = None
        self.rps_hierarchical_projs = None

        if self.use_rps:
            self.rotor_encoder = RotorEncoder(self.num_rotors, out_channels=64, kernel_size=3)
            if self.rps_fusion == "bottleneck":
                self.rps_bottleneck_proj = nn.Linear(64, bottleneck_ch * 2)
            elif self.rps_fusion == "gru":
                self._gru_hidden = 256
                self.rps_gru = nn.GRU(
                    64 + bottleneck_ch * 2,
                    self._gru_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                self.rps_gru_proj = nn.Linear(self._gru_hidden * 2, bottleneck_ch * 2)
            elif self.rps_fusion == "hierarchical":
                enc_channels = [45, 90, 90, 90, 90]
                if self.num_encoder_layers == 6:
                    enc_channels.append(90)
                self.rps_hierarchical_blocks = nn.ModuleList()
                self.rps_hierarchical_projs = nn.ModuleList()
                for c in enc_channels:
                    self.rps_hierarchical_blocks.append(
                        nn.Sequential(nn.Conv1d(self.num_rotors, 64, 3, padding=1), nn.ReLU())
                    )
                    self.rps_hierarchical_projs.append(nn.Linear(64, c * 2))

        # ── Auxiliary RPS prediction head ──
        self.predict_rps: bool = rps_cfg.get("predict_rps", False) or config.get(
            "predict_rps", False
        )
        self.rps_prediction_head = None
        if self.predict_rps:
            enc_channels: list[int] = [oc for _, oc, _, _, _ in enc_spec]
            chunk_size: int = config["audio"]["chunk_size"]
            target_t: int = stft_time_frames(chunk_size, self.hop_length, self.n_fft)
            self.rps_prediction_head = RPSPredictionHead(
                enc_channels, target_t, num_rotors=self.num_rotors
            )

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, rps: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, T),  rps: (B, num_rotors, time_rps) or None."""
        input_length = x.shape[-1]
        X = self.stft.transform(x)  # (B, 1, n_freq, n_time, 2)
        B, _, n_freq, n_time, _ = X.shape
        encoder_features: list[torch.Tensor] = []
        current = X

        # ── Encoder with optional hierarchical RPS injection ──
        blocks: nn.ModuleList | None = None
        projs: nn.ModuleList | None = None
        if self.use_rps and rps is not None and self.rps_fusion == "hierarchical":
            assert self.rps_hierarchical_blocks is not None
            assert self.rps_hierarchical_projs is not None
            rps_align = rps if rps.dim() == 3 else rps.unsqueeze(1)
            blocks = self.rps_hierarchical_blocks
            projs = self.rps_hierarchical_projs

        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            if self.use_rps and rps is not None and self.rps_fusion == "hierarchical":
                assert blocks is not None
                assert projs is not None
                level_t = current.shape[3]
                h = blocks[i](rps_align)
                if h.size(-1) != level_t:
                    h = F.interpolate(h, size=level_t, mode="linear", align_corners=False)
                h = h.permute(0, 2, 1)
                h = projs[i](h)
                c_i = current.shape[1]
                h = h.reshape(B, level_t, c_i, 2).permute(0, 2, 1, 3).unsqueeze(2)
                current = current + h
            if i < len(self.encoders) - 1:
                encoder_features.append(current)

        # ── Auxiliary RPS prediction ──
        rps_pred: torch.Tensor | None = None
        if self.predict_rps and self.rps_prediction_head is not None:
            all_encoder_feats = encoder_features + [current]
            rps_pred = self.rps_prediction_head(all_encoder_feats)

        # ── RPS fusion at the bottleneck ──
        if self.use_rps and rps is not None:
            assert self.rotor_encoder is not None
            if self.rps_fusion == "bottleneck":
                assert self.rps_bottleneck_proj is not None
                with torch.autocast("cuda", enabled=False):
                    rotor_feat = self.rotor_encoder(rps.float())
                    rotor_feat = rotor_feat.mean(dim=-1)
                    proj = self.rps_bottleneck_proj(rotor_feat)
                C = current.shape[1]
                proj = proj.view(B, C, 2)
                current = current.float() + proj.unsqueeze(2).unsqueeze(3)
            elif self.rps_fusion == "gru":
                assert self.rps_gru is not None
                assert self.rps_gru_proj is not None
                T_b = current.shape[3]
                rotor_feat = self.rotor_encoder(rps, target_length=T_b)
                # Pool frequency: (B, C, F, T, 2) → (B, C, T, 2) → (B, C*2, T)
                pooled = current.mean(dim=2)
                pooled = pooled.reshape(B, current.shape[1] * 2, T_b).permute(
                    0, 2, 1
                )  # (B, T_b, C*2)
                concat = torch.cat([pooled, rotor_feat.permute(0, 2, 1)], dim=-1)
                gru_out, _ = self.rps_gru(concat)
                back = self.rps_gru_proj(gru_out)
                C, n_freq_b = current.shape[1], current.shape[2]
                current = (
                    back.reshape(B, T_b, C, 2)
                    .permute(0, 2, 3, 1)
                    .unsqueeze(2)
                    .expand(-1, -1, n_freq_b, -1, -1)
                )

        # ── Decoder ──
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                current = decoder(current)
            else:
                skip = encoder_features[-i]
                min_f = min(current.shape[2], skip.shape[2])
                min_t = min(current.shape[3], skip.shape[3])
                if current.shape[2] != min_f or current.shape[3] != min_t:
                    current = current[:, :, :min_f, :min_t, :]
                if skip.shape[2] != min_f or skip.shape[3] != min_t:
                    skip = skip[:, :, :min_f, :min_t, :]
                current = decoder(torch.cat([current, skip], dim=1))

        # ── Match STFT dimensions ──
        if current.shape[2] != n_freq or current.shape[3] != n_time:
            pad_f = n_freq - current.shape[2]
            pad_t = n_time - current.shape[3]
            if pad_f > 0 or pad_t > 0:
                current = torch.nn.functional.pad(
                    current, (0, 0, 0, max(0, pad_t), 0, max(0, pad_f))
                )
            if pad_f < 0 or pad_t < 0:
                current = current[:, :, :n_freq, :n_time, :]

        # ── Mask + ISTFT ──
        output = current * X
        output = self.stft.inverse(output)

        if output.shape[-1] != input_length:
            warnings.warn(
                f"DCUNet output length mismatch: output={output.shape[-1]}, "
                f"input={input_length}, diff={output.shape[-1] - input_length}. "
                "Consider adjusting chunk_size.",
                stacklevel=2,
            )
            if output.shape[-1] < input_length:
                output = torch.nn.functional.pad(output, (0, input_length - output.shape[-1]))
            else:
                output = output[..., :input_length]

        output = output.unsqueeze(1)
        if rps_pred is not None:
            return output, rps_pred
        return output


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

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

    # Baseline (no RPS)
    model = DCUNet(config)
    x = torch.randn(2, 1, 8192)
    out = model(x)
    print("Baseline DCUNet output shape:", out.shape)
    assert out.shape == (2, 1, 1, 8192), out.shape

    # RPS bottleneck (6-layer)
    config["use_rps"] = True
    config["dcunet_rps_fusion"] = "bottleneck"
    config["dcunet_num_encoder_layers"] = 6
    config["num_rotors"] = 4
    model2 = DCUNet(config)
    rps = torch.randn(2, 4, 100)
    out2 = model2(x, rps=rps)
    print("RPS-DCUN6-P output shape:", out2.shape)

    # RPS GRU (5-layer)
    config["dcunet_rps_fusion"] = "gru"
    config["dcunet_num_encoder_layers"] = 5
    model3 = DCUNet(config)
    out3 = model3(x, rps=rps)
    print("RPS-DCUN5 (GRU) output shape:", out3.shape)

    # RPS hierarchical (5-layer)
    config["dcunet_rps_fusion"] = "hierarchical"
    model4 = DCUNet(config)
    out4 = model4(x, rps=rps)
    print("RPS-DCUN5-H output shape:", out4.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nBaseline total parameters: {total_params}")
    print("All smoke tests passed.")
