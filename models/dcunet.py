import torch
import torch.nn as nn
import torch.nn.functional as F


def stft_time_frames(audio_length: int, hop_length: int, n_fft: int) -> int:
    """Number of STFT time frames for a given audio length."""
    return (audio_length - n_fft) // hop_length + 1


def encoder_time_lengths(
    n_stft_time: int,
    encoder_strides: list,
) -> list:
    """
    Time dimension at each encoder level (0 = STFT input, 1..L = after each encoder).
    encoder_strides: list of (stride_f, stride_t) per layer.
    """
    lengths = [n_stft_time]
    for (_, stride_t) in encoder_strides:
        lengths.append((lengths[-1] + 1) // stride_t)
    return lengths


class RotorEncoder(nn.Module):
    """
    Encodes rotor rps time series (B, num_rotors, time_rps) via two 1D convs (kernel 3, 64 ch).
    Optionally interpolates to target_length along time.
    Output: (B, 64, target_length) when target_length is set, else (B, 64, T) after convs.
    """
    def __init__(self, num_rotors: int, out_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        self.num_rotors = num_rotors
        self.out_channels = out_channels
        padding = kernel_size // 2  # same length
        self.input_bn = nn.BatchNorm1d(num_rotors)
        self.conv1 = nn.Conv1d(num_rotors, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.ReLU()
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, rps: torch.Tensor, target_length: int = None) -> torch.Tensor:
        """
        rps: (B, num_rotors, time_rps) or (B, time_rps) [then unsqueeze(1)]
        """
        if rps.dim() == 2:
            rps = rps.unsqueeze(1)  # (B, 1, time_rps)
        rps = self.input_bn(rps)
        x = self.act(self.conv1(rps))
        x = self.act(self.conv2(x))   # (B, 64, time)
        if target_length is not None and x.size(-1) != target_length:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return x


class CConv2d(nn.Module):
    """Complex Convolutional Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.im_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        """
        Input x: (batch, channels, freq, time, 2)
        Output: (batch, channels, freq, time, 2)
        """
        # Separate real and imaginary parts
        x_real, x_im = x[..., 0], x[..., 1]

        # Apply convolution
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        # Combine real and imaginary parts
        return torch.stack([c_real, c_im], dim=-1)

class CConvTranspose2d(nn.Module):
    """Complex Transpose Convolutional Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        self.real_convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                           stride, padding, output_padding)
        self.im_convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride, padding, output_padding)
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        """
        Input x: (batch, channels, freq, time, 2)
        Output: (batch, channels, freq, time, 2)
        """
        # Separate real and imaginary parts
        x_real, x_im = x[..., 0], x[..., 1]

        # Apply transpose convolution
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        # Combine real and imaginary parts
        return torch.stack([ct_real, ct_im], dim=-1)

class CBatchNorm2d(nn.Module):
    """Complex Batch Normalization"""
    def __init__(self, num_features):
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.im_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        """
        Input x: (batch, channels, freq, time, 2)
        Output: (batch, channels, freq, time, 2)
        """
        # Separate real and imaginary parts
        x_real, x_im = x[..., 0], x[..., 1]

        # Apply batch normalization
        x_real = self.real_bn(x_real)
        x_im = self.im_bn(x_im)

        # Combine real and imaginary parts
        return torch.stack([x_real, x_im], dim=-1)

class Encoder(nn.Module):
    """Encoder Module"""
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.cconv = CConv2d(in_channels, out_channels, kernel, stride, padding)
        self.cbn = CBatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.cconv(x)
        x = self.cbn(x)
        return self.act(x)

class Decoder(nn.Module):
    """Decoder Module"""
    def __init__(self, in_channels, out_channels, kernel, stride, output_padding, padding, last_layer=False):
        super().__init__()
        self.cconvt = CConvTranspose2d(in_channels, out_channels, kernel, stride, output_padding, padding)
        self.cbn = CBatchNorm2d(out_channels) if not last_layer else None
        self.act = nn.LeakyReLU() if not last_layer else None
        self.last_layer = last_layer

    def forward(self, x):
        x = self.cconvt(x)
        if not self.last_layer:
            x = self.cbn(x)
            x = self.act(x)
        else:
            # Compute phase and magnitude in float32 for numerical stability
            # under AMP (float16), where eps < 6e-5 is rounded to 0 causing NaN
            x = x.float()
            m_phase = x / (torch.abs(x) + 1e-8)
            m_mag = torch.tanh(torch.abs(x))
            x = m_phase * m_mag
        return x

class STFTProcessor(nn.Module):
    """STFT Processing Module with unified interface for compatibility with other models"""
    def __init__(self, config):
        super().__init__()
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        self.window = torch.hann_window(self.n_fft)
        self.dim_f = config['audio']['dim_f']

    def transform(self, x):
        """
        Input x: (batch, channels, time)
        Output: (batch, 1, freq, time, 2)
        """
        if __name__ == "__main__":
            print(f"STFT input shape: {x.shape}")
        x = x.squeeze(1)  # Remove channel dimension
        if __name__ == "__main__":
            print(f"After removing channel dimension: {x.shape}")

        # Perform STFT
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                      window=self.window.to(x.device), return_complex=True,
                      normalized=True)
        if __name__ == "__main__":
            print(f"After STFT: {X.shape}")

        # Convert to real representation and adjust dimensions
        X = torch.view_as_real(X)  # (batch, freq, time, 2)
        if __name__ == "__main__":
            print(f"After converting to real representation: {X.shape}")
        X = X.unsqueeze(1)  # Add channel dimension (batch, 1, freq, time, 2)
        if __name__ == "__main__":
            print(f"After adding channel dimension: {X.shape}")

        return X

    def inverse(self, X):
        """
        Input X: (batch, 1, freq, time, 2)
        Output: (batch, channels=1, time)
        """
        if __name__ == "__main__":
            print(f"ISTFT input shape: {X.shape}")
        # Adjust dimensions for ISTFT compatibility
        X = X.squeeze(1)  # Remove channel dimension (batch, freq, time, 2)
        if __name__ == "__main__":
            print(f"After removing channel dimension: {X.shape}")
        X = torch.view_as_complex(X)
        if __name__ == "__main__":
            print(f"After converting to complex: {X.shape}")

        x = torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window.to(X.device), normalized=True)
        if __name__ == "__main__":
            print(f"After ISTFT: {x.shape}")

        x = x.unsqueeze(1)  # Add channel dimension (batch, 1, time)
        if __name__ == "__main__":
            print(f"After adding channel dimension: {x.shape}")
        return x

def _get_rps_config(config) -> dict:
    """Read RPS-related options from config (top-level or under 'model')."""
    get = getattr(config, "get", lambda k, d=None: getattr(config, k, d))
    m = get("model")
    m = m if isinstance(m, dict) else {}
    m_get = getattr(m, "get", lambda k, d=None: getattr(m, k, d) if hasattr(m, k) else d)
    return {
        "use_rps": get("use_rps") or m_get("use_rps", False),
        "dcunet_rps_fusion": get("dcunet_rps_fusion") or m_get("dcunet_rps_fusion", "bottleneck"),
        "dcunet_num_encoder_layers": get("dcunet_num_encoder_layers") or m_get("dcunet_num_encoder_layers", 5),
        "num_rotors": get("num_rotors") or m_get("num_rotors", 4),
    }


class DCUNet(nn.Module):
    """Deep Complex U-Net with optional rotor (rps) conditioning. When rps is None, behaves as baseline."""
    def __init__(self, config):
        super().__init__()
        self.stft = STFTProcessor(config)
        self.input_channels = 1
        self.output_channels = config["audio"]["num_channels"]
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]

        rps_cfg = _get_rps_config(config)
        self.use_rps = rps_cfg["use_rps"]
        self.rps_fusion = rps_cfg["dcunet_rps_fusion"]
        self.num_encoder_layers = rps_cfg["dcunet_num_encoder_layers"]
        self.num_rotors = rps_cfg["num_rotors"]

        # Encoder strides for time-dimension helper
        self._encoder_strides = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)]  # 5 layers
        if self.num_encoder_layers == 6:
            self._encoder_strides.append((2, 1))

        # Build encoder list (5 or 6 layers)
        enc_spec = [
            (1, 45, (7, 5), (2, 2), (3, 2)),
            (45, 90, (7, 5), (2, 2), (3, 2)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        if self.num_encoder_layers == 6:
            enc_spec.append((90, 90, (5, 3), (2, 1), (2, 1)))
        self.encoders = nn.ModuleList([
            Encoder(ic, oc, k, s, padding=p) for ic, oc, k, s, p in enc_spec
        ])

        # Decoder list (mirror encoder; skip channels double at concat)
        bottleneck_ch = enc_spec[-1][1]
        dec_spec = [
            (bottleneck_ch, 90, (5, 3), (2, 1), (0, 0), (2, 1)),
            (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
            (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
            (180, 45, (7, 5), (2, 2), (0, 0), (3, 2)),
            (90, 1, (7, 5), (2, 2), (0, 1), (3, 2), True),
        ]
        if self.num_encoder_layers == 6:
            dec_spec = [
                (90, 90, (5, 3), (2, 1), (0, 0), (2, 1)),
                (180, 90, (5, 3), (2, 1), (0, 0), (2, 1)),
                (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
                (180, 90, (5, 3), (2, 2), (0, 0), (2, 1)),
                (180, 45, (7, 5), (2, 2), (0, 0), (3, 2)),
                (90, 1, (7, 5), (2, 2), (0, 1), (3, 2), True),
            ]
        self.decoders = nn.ModuleList()
        for t in dec_spec:
            if len(t) == 7:
                self.decoders.append(Decoder(t[0], t[1], t[2], t[3], output_padding=t[4], padding=t[5], last_layer=t[6]))
            else:
                self.decoders.append(Decoder(t[0], t[1], t[2], t[3], output_padding=t[4], padding=t[5]))

        # RPS pathway and fusion
        self.rotor_encoder = None
        self.rps_bottleneck_proj = None
        self.rps_gru = None
        self.rps_gru_proj = None
        self.rps_hierarchical_blocks = None
        self.rps_hierarchical_projs = None

        if self.use_rps:
            self.rotor_encoder = RotorEncoder(self.num_rotors, out_channels=64, kernel_size=3)
            if self.rps_fusion == "bottleneck":
                # Linear: 64 -> bottleneck_ch * 2 (real/imag)
                self.rps_bottleneck_proj = nn.Linear(64, bottleneck_ch * 2)
            elif self.rps_fusion == "gru":
                self._gru_hidden = 256
                self.rps_gru = nn.GRU(
                    64 + bottleneck_ch * 2,  # concat rotor + flattened complex bottleneck per t
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
                        nn.Sequential(
                            nn.Conv1d(self.num_rotors, 64, 3, padding=1),
                            nn.ReLU(),
                        )
                    )
                    self.rps_hierarchical_projs.append(nn.Linear(64, c * 2))

    def forward(self, x, rps=None):
        """
        Input x: (batch, channels, time)
        rps: (batch, num_rotors, time_rps) or None for baseline behaviour.
        Output: (batch, instruments=1, channels=1, time)
        """
        input_length = x.shape[-1]  # Store original length for output padding
        X = self.stft.transform(x)
        B, _, F, T, _ = X.shape
        encoder_features = []
        current = X

        if self.use_rps and rps is not None and self.rps_fusion == "hierarchical":
            n_stft_time = T
            time_lengths = encoder_time_lengths(n_stft_time, self._encoder_strides)
            rps_align = rps
            if rps.dim() == 2:
                rps_align = rps.unsqueeze(1)

        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            if self.use_rps and rps is not None and self.rps_fusion == "hierarchical":
                level_t = current.shape[3]
                h = self.rps_hierarchical_blocks[i](rps_align)
                if h.size(-1) != level_t:
                    h = F.interpolate(h, size=level_t, mode="linear", align_corners=False)
                h = h.permute(0, 2, 1)
                h = self.rps_hierarchical_projs[i](h)
                c_i = current.shape[1]
                h = h.reshape(B, level_t, c_i, 2).permute(0, 2, 1, 3).unsqueeze(2)
                current = current + h
            if i < len(self.encoders) - 1:
                encoder_features.append(current)

        if self.use_rps and rps is not None:
            if self.rps_fusion == "bottleneck":
                # Run RPS pathway in float32 to avoid float16 overflow
                # (rotor activations can be large before BN fully converges)
                with torch.amp.autocast('cuda', enabled=False):
                    rotor_feat = self.rotor_encoder(rps.float())
                    rotor_feat = rotor_feat.mean(dim=-1)
                    proj = self.rps_bottleneck_proj(rotor_feat)
                C = current.shape[1]
                proj = proj.view(B, C, 2)
                current = current.float() + proj.unsqueeze(2).unsqueeze(3)
            elif self.rps_fusion == "gru":
                T_b = current.shape[3]
                rotor_feat = self.rotor_encoder(rps, target_length=T_b)
                flat = current.permute(0, 3, 1, 2, 4).reshape(B, T_b, -1)
                concat = torch.cat([flat, rotor_feat.permute(0, 2, 1)], dim=-1)
                gru_out, _ = self.rps_gru(concat)
                back = self.rps_gru_proj(gru_out)
                C, F_b = current.shape[1], current.shape[2]
                current = back.reshape(B, T_b, C, 2).permute(0, 2, 3, 1).unsqueeze(2).expand(-1, -1, F_b, -1, -1)
            # hierarchical already fused in the loop

        for i, decoder in enumerate(self.decoders):
            if i == 0:
                current = decoder(current)
            else:
                skip = encoder_features[-i]
                # Match dimensions between current and skip (crop the larger one)
                min_f = min(current.shape[2], skip.shape[2])
                min_t = min(current.shape[3], skip.shape[3])
                if current.shape[2] != min_f or current.shape[3] != min_t:
                    current = current[:, :, :min_f, :min_t, :]
                if skip.shape[2] != min_f or skip.shape[3] != min_t:
                    skip = skip[:, :, :min_f, :min_t, :]
                current = decoder(torch.cat([current, skip], dim=1))

        # Pad or crop output to match input spectrogram dimensions
        if current.shape[2] != F or current.shape[3] != T:
            # Pad if output is smaller, crop if larger
            pad_f = F - current.shape[2]
            pad_t = T - current.shape[3]
            if pad_f > 0 or pad_t > 0:
                # Pad with zeros: (last_dim_pad, ..., first_dim_pad)
                # Shape is (B, C, F, T, 2), pad T (dim 3) and F (dim 2)
                current = torch.nn.functional.pad(current, (0, 0, 0, max(0, pad_t), 0, max(0, pad_f)))
            if pad_f < 0 or pad_t < 0:
                # Crop if larger
                current = current[:, :, :F, :T, :]
        output = current * X
        output = self.stft.inverse(output)

        # Ensure output length matches input length
        output_length = output.shape[-1]
        if output_length != input_length:
            import warnings
            warnings.warn(
                f"DCUNet output length mismatch: output={output_length}, input={input_length}, "
                f"diff={output_length - input_length}. Consider adjusting chunk_size."
            )
            if output_length < input_length:
                output = torch.nn.functional.pad(output, (0, input_length - output_length))
            else:
                output = output[..., :input_length]

        output = output.unsqueeze(1)
        return output


# ---------------------- Simple test case ----------------------
if __name__ == "__main__":
    config = {
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

    # Baseline (no rps)
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
    print("All tests passed.")
