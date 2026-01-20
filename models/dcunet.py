import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DCUNet(nn.Module):
    """Deep Complex U-Net Main Model"""
    def __init__(self, config):
        super().__init__()
        # Add STFT processor
        self.stft = STFTProcessor(config)

        # Adjust input/output channels
        self.input_channels = 1  # Set to 1, since complex channels are in the last dimension
        self.output_channels = config['audio']['num_channels']

        # Fixed parameters (based on original paper implementation)
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']

        # Encoder - Modified first Encoder input channels to 1
        self.encoders = nn.ModuleList([
            Encoder(1, 45, (7,5), (2,2), padding=(3,2)),  # Modified input channels to 1 and added padding
            Encoder(45, 90, (7,5), (2,2), padding=(3,2)),
            Encoder(90, 90, (5,3), (2,2), padding=(2,1)),
            Encoder(90, 90, (5,3), (2,2), padding=(2,1)),
            Encoder(90, 90, (5,3), (2,1), padding=(2,1))
        ])

        # Decoder
        self.decoders = nn.ModuleList([
            Decoder(90, 90, (5,3), (2,1), output_padding=(0,0), padding=(2,1)),
            Decoder(180, 90, (5,3), (2,2), output_padding=(0,0), padding=(2,1)),
            Decoder(180, 90, (5,3), (2,2), output_padding=(0,0), padding=(2,1)),
            Decoder(180, 45, (7,5), (2,2), output_padding=(0,0), padding=(3,2)),
            Decoder(90, 1, (7,5), (2,2), output_padding=(0,1), padding=(3,2), last_layer=True)
        ])

    def forward(self, x):
        """
        Input x: (batch, channels, time)
        Output: (batch, instruments=1, channels=1, time)
        """
        if __name__ == "__main__":
            print("\n----- Starting forward pass -----")
            print(f"Input shape: {x.shape}")

        # STFT transform
        X = self.stft.transform(x)  # (batch, 1, freq, time, 2)
        if __name__ == "__main__":
            print(f"\n----- Encoder process -----")
            print(f"Shape after STFT: {X.shape}")

        # Encoding process
        encoder_features = []
        current = X

        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            if i < len(self.encoders) - 1:
                encoder_features.append(current)
            if __name__ == "__main__":
                print(f"Encoder {i+1} output: {current.shape}")

        if __name__ == "__main__":
            print(f"\n----- Decoder process -----")
        # Decoding process
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                current = decoder(current)
            else:
                skip_connection = encoder_features[-(i)]
                current = decoder(torch.cat([current, skip_connection], dim=1))
            if __name__ == "__main__":
                print(f"Decoder {i+1} output: {current.shape}")

        # Adjust dimensions to match X for mask operation
        output = current * X
        if __name__ == "__main__":
            print(f"\nShape after masking: {output.shape}")

        # ISTFT transform
        output = self.stft.inverse(output)  # (batch, 1, time)
        # Adjust dimensions (batch, instruments=1, channels=1, time)
        output = output.unsqueeze(1)
        if __name__ == "__main__":
            print(f"Final output shape: {output.shape}")
            print("----- Forward pass complete -----\n")

        return output

# ---------------------- Simple test case ----------------------
if __name__ == "__main__":
    # Simulated configuration
    config = {
        "audio": {
            "chunk_size": 131584,  # Only used for generating test input
            "dim_f": 1024,
            "hop_length": 512,
            "n_fft": 2048,
            "num_channels": 1,     # Set to 1 to match mono STFT logic
            "sample_rate": 16000,
        },
        "training": {
            "batch_size": 10
        }
    }

    # Initialize model
    model = DCUNet(config)
    print("Model structure:")
    print(model)

    # Create test input: (batch_size, 1, time)
    batch_size = config["training"]["batch_size"]
    channels   = config["audio"]["num_channels"]  # Usually set to 1
    time_len   = config["audio"]["chunk_size"]
    x = torch.randn(batch_size, channels, time_len)

    # Forward pass
    print("\nPerforming forward pass...")
    output = model(x)

    # Output shape
    print("\nOutput shape:", output.shape)
    # Print total model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params}")
