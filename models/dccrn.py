import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dcunet import RPSPredictionHead


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
        x_real, x_im = x[..., 0], x[..., 1]
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
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
        x_real, x_im = x[..., 0], x[..., 1]
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
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
        x_real, x_im = x[..., 0], x[..., 1]
        x_real = self.real_bn(x_real)
        x_im = self.im_bn(x_im)
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
            x = x.float()
            m_phase = x / (torch.abs(x) + 1e-8)
            m_mag = torch.tanh(torch.abs(x))
            x = m_phase * m_mag
        return x


class STFTProcessor(nn.Module):
    """STFT Processing Module"""
    def __init__(self, config):
        super().__init__()
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        self.window = torch.hann_window(self.n_fft)

    def transform(self, x):
        """Input: (batch, channels, time) -> Output: (batch, 1, freq, time, 2)"""
        x = x.squeeze(1)
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window.to(x.device), return_complex=True,
                       normalized=True)
        X = torch.view_as_real(X)
        X = X.unsqueeze(1)
        return X

    def inverse(self, X):
        """Input: (batch, 1, freq, time, 2) -> Output: (batch, 1, time)"""
        X = X.squeeze(1)
        X = torch.view_as_complex(X)
        x = torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                        window=self.window.to(X.device), normalized=True)
        x = x.unsqueeze(1)
        return x


class RotorEncoder(nn.Module):
    """Encodes rotor RPS time series via two 1D convs. Output: (B, 64, target_length)."""
    def __init__(self, num_rotors, out_channels=64, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.input_bn = nn.BatchNorm1d(num_rotors)
        self.conv1 = nn.Conv1d(num_rotors, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.ReLU()
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, rps, target_length=None):
        if rps.dim() == 2:
            rps = rps.unsqueeze(1)
        rps = self.input_bn(rps)
        x = self.act(self.conv1(rps))
        x = self.act(self.conv2(x))
        if target_length is not None and x.size(-1) != target_length:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return x


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


class DCCRN(nn.Module):
    """Deep Complex Convolution Recurrent Network for speech enhancement.

    Architecture follows Gulli et al., "Enhancing drone audition with
    rotor-conditioned deep models", EURASIP J. Audio Speech Music Process. 2025.

    Three variants:
    - DCCRN baseline: 6 complex conv layers (32->512), 2-layer bidir GRU (256)
    - RPS-DCCRN: same + GRU-based rotor fusion (concatenation before GRU)
    - RPS-DCCRNLite: 4 conv layers (16->128), 1-layer bidir GRU (128), same fusion

    Encoder uses kernel (5,2), stride (2,1), padding (2,0) for all layers
    (stride 2 on frequency, stride 1 on time). The decoder mirrors the encoder
    with transposed convolutions and skip connections. A regular (non-complex)
    bidirectional GRU at the bottleneck models temporal dependencies. For
    RPS-informed variants, rotor features are concatenated with the encoded
    spectrogram before the GRU.
    """
    def __init__(self, config):
        super().__init__()
        self.stft = STFTProcessor(config)
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]

        # Architecture variant
        self.lite = _get_config_val(config, "dccrn_lite", False)

        if self.lite:
            # DCCRNLite: 4 layers, channels 16->32->64->128
            encoder_channels = [16, 32, 64, 128]
            gru_hidden = _get_config_val(config, "dccrn_gru_hidden", 128)
            gru_layers = 1
        else:
            # Full DCCRN: 6 layers, channels 32->64->128->256->256->512
            encoder_channels = [32, 64, 128, 256, 256, 512]
            gru_hidden = _get_config_val(config, "dccrn_gru_hidden", 256)
            gru_layers = _get_config_val(config, "dccrn_gru_layers", 2)

        self._encoder_channels = encoder_channels

        # RPS configuration
        self.use_rps = _get_config_val(config, "use_rps", False)
        self.num_rotors = _get_config_val(config, "num_rotors", 4)

        # Encoder: kernel (5,2), stride (2,1), padding (2,0) for all layers
        enc_kernel = (5, 2)
        enc_stride = (2, 1)
        enc_padding = (2, 0)

        in_channels = [1] + encoder_channels[:-1]
        self.encoders = nn.ModuleList()
        for ic, oc in zip(in_channels, encoder_channels):
            self.encoders.append(Encoder(ic, oc, enc_kernel, enc_stride, enc_padding))

        bottleneck_ch = encoder_channels[-1]

        # Compute bottleneck frequency dimension
        freq = self.n_fft // 2 + 1
        for _ in encoder_channels:
            freq = (freq + 2 * enc_padding[0] - enc_kernel[0]) // enc_stride[0] + 1
        self._bottleneck_freq = freq
        self._bottleneck_ch = bottleneck_ch

        # GRU input: flatten C * F_b * 2 (channels, freq, real+imag)
        gru_input_size = bottleneck_ch * freq * 2

        # RPS pathway: rotor features concatenated before GRU
        self.rotor_encoder = None
        if self.use_rps:
            self.rotor_encoder = RotorEncoder(self.num_rotors, out_channels=64, kernel_size=3)
            gru_input_size += 64

        # Bottleneck: regular bidirectional GRU (not complex)
        self.gru = nn.GRU(
            gru_input_size,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_proj = nn.Linear(gru_hidden * 2, bottleneck_ch * freq * 2)

        # Decoder: mirror encoder with skip connections
        # Decoder output channels = reversed encoder input channels
        # dec_out[i] matches encoder_features[-(i+1)] for skip concatenation
        dec_kernel = (5, 2)
        dec_stride = (2, 1)
        dec_padding = (2, 0)
        dec_output_padding = (0, 0)

        dec_out = list(reversed([1] + encoder_channels[:-1]))
        self.decoders = nn.ModuleList()
        for i in range(len(dec_out)):
            if i == 0:
                in_ch = encoder_channels[-1]  # bottleneck channels
            else:
                in_ch = dec_out[i - 1] * 2  # prev decoder output + skip
            out_ch = dec_out[i]
            is_last = (i == len(dec_out) - 1)
            self.decoders.append(
                Decoder(in_ch, out_ch, dec_kernel, dec_stride,
                        output_padding=dec_output_padding, padding=dec_padding,
                        last_layer=is_last)
            )

        # Auxiliary RPS prediction head
        self.predict_rps = _get_config_val(config, "predict_rps", False)
        self.rps_prediction_head = None
        if self.predict_rps:
            self.rps_prediction_head = RPSPredictionHead(
                bottleneck_ch, self._bottleneck_freq, self.num_rotors
            )

    def forward(self, x, rps=None):
        """
        Input x: (batch, channels, time)
        rps: (batch, num_rotors, time_rps) or None for baseline behaviour.
        Output: (batch, instruments=1, channels=1, time)
        """
        input_length = x.shape[-1]
        X = self.stft.transform(x)  # (B, 1, F, T, 2)
        B, _, F_stft, T, _ = X.shape

        # Encoder
        encoder_features = []
        current = X
        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            if i < len(self.encoders) - 1:
                encoder_features.append(current)

        # Bottleneck: (B, C, F_b, T_b, 2)
        _, C, F_b, T_b, _ = current.shape

        # Auxiliary RPS prediction from encoder bottleneck (before RPS fusion)
        rps_pred = None
        if self.predict_rps and self.rps_prediction_head is not None:
            rps_pred = self.rps_prediction_head(current)

        # Flatten for GRU: (B, T_b, C*F_b*2)
        gru_in = current.permute(0, 3, 1, 2, 4).reshape(B, T_b, C * F_b * 2)

        # Concatenate rotor features before GRU (Gulli et al. RPS-DCCRN)
        if self.use_rps and rps is not None:
            rotor_feat = self.rotor_encoder(rps, target_length=T_b)  # (B, 64, T_b)
            rotor_feat = rotor_feat.permute(0, 2, 1)  # (B, T_b, 64)
            gru_in = torch.cat([gru_in, rotor_feat], dim=-1)

        # Bidirectional GRU + projection back to bottleneck shape
        gru_out, _ = self.gru(gru_in)  # (B, T_b, gru_hidden*2)
        gru_out = self.gru_proj(gru_out)  # (B, T_b, C*F_b*2)
        current = gru_out.reshape(B, T_b, C, F_b, 2).permute(0, 2, 3, 1, 4)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                current = decoder(current)
            else:
                skip = encoder_features[-i]
                current = decoder(torch.cat([current, skip], dim=1))

        # Pad or crop output to match input spectrogram dimensions
        if current.shape[2] != F_stft or current.shape[3] != T:
            pad_f = F_stft - current.shape[2]
            pad_t = T - current.shape[3]
            if pad_f > 0 or pad_t > 0:
                current = F.pad(current, (0, 0, 0, max(0, pad_t), 0, max(0, pad_f)))
            if pad_f < 0 or pad_t < 0:
                current = current[:, :, :F_stft, :T, :]

        output = current * X
        output = self.stft.inverse(output)

        # Ensure output length matches input length
        output_length = output.shape[-1]
        if output_length != input_length:
            if output_length < input_length:
                output = F.pad(output, (0, input_length - output_length))
            else:
                output = output[..., :input_length]

        output = output.unsqueeze(1)
        if rps_pred is not None:
            return output, rps_pred
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

    x = torch.randn(2, 1, 8192)
    rps = torch.randn(2, 4, 100)

    # 1. Baseline DCCRN (no RPS)
    model = DCCRN(config)
    out = model(x)
    print("Baseline DCCRN output shape:", out.shape)
    assert out.shape == (2, 1, 1, 8192), out.shape
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    # 2. RPS-DCCRN (full, 6 layers, GRU-based rotor fusion)
    config["use_rps"] = True
    config["num_rotors"] = 4
    model2 = DCCRN(config)
    out2 = model2(x, rps=rps)
    print("RPS-DCCRN output shape:", out2.shape)
    assert out2.shape == (2, 1, 1, 8192), out2.shape
    total2 = sum(p.numel() for p in model2.parameters())
    print(f"  Parameters: {total2:,}")

    # 3. RPS-DCCRNLite (4 layers, lighter GRU)
    config["dccrn_lite"] = True
    model3 = DCCRN(config)
    out3 = model3(x, rps=rps)
    print("RPS-DCCRNLite output shape:", out3.shape)
    assert out3.shape == (2, 1, 1, 8192), out3.shape
    total3 = sum(p.numel() for p in model3.parameters())
    print(f"  Parameters: {total3:,}")

    # 4. DCCRNLite baseline (no RPS)
    config["use_rps"] = False
    del config["dccrn_lite"]
    config["dccrn_lite"] = True
    model4 = DCCRN(config)
    out4 = model4(x)
    print("DCCRNLite baseline output shape:", out4.shape)
    assert out4.shape == (2, 1, 1, 8192), out4.shape
    total4 = sum(p.numel() for p in model4.parameters())
    print(f"  Parameters: {total4:,}")

    print("\nAll tests passed.")
