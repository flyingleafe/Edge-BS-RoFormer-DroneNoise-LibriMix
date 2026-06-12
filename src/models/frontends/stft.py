"""
STFT-based spectral front-ends.

Produces the same features the original SimpleConv models computed inline,
ensuring numeric equivalence so old checkpoints stay loadable.

``STFTMag`` → log₁₊, 1 channel  (matches SimpleConv, SimpleConvV2, …)
``STFTMagPhase`` → log mag + cos(θ) + sin(θ), 3 channels
                     (matches SimpleConvMagPhaseBiGRU)
"""

import torch
from torch import Tensor

from . import SpectralFrontEnd, register_frontend


@register_frontend
class STFTMag(SpectralFrontEnd):
    """Log-magnitude STFT — the current SimpleConv default.

    Reproduces exactly the expression that was inlined in every old model's
    ``forward()``::

        X = torch.stft(audio, n_fft, hop_length, window=self.window,
                        return_complex=True, normalized=True)
        mag = torch.log1p(X.abs()).unsqueeze(1)   # (B, 1, F, T)
    """

    key = "stft_mag"
    out_channels = 1

    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def num_frames(self, n_samples: int) -> int:
        return n_samples // self.hop_length + 1

    def forward(self, audio: Tensor) -> Tensor:
        # (B, N) or (B, 1, N) → (B, N)
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
        mag = torch.log1p(X.abs())
        return mag.unsqueeze(1)  # (B, 1, F, T)


@register_frontend
class STFTMagPhase(SpectralFrontEnd):
    """Log-magnitude + cos(θ) + sin(θ) — 3-channel STFT input.

    Matches ``SimpleConvMagPhaseBiGRU`` semantics exactly.
    """

    key = "stft_magphase"
    out_channels = 3

    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def num_frames(self, n_samples: int) -> int:
        return n_samples // self.hop_length + 1

    def forward(self, audio: Tensor) -> Tensor:
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
        mag = torch.log1p(X.abs())
        phase = torch.angle(X)
        phase_cos = torch.cos(phase)
        phase_sin = torch.sin(phase)
        return torch.stack([mag, phase_cos, phase_sin], dim=1)  # (B, 3, F, T)
