"""
STFT-based spectral front-ends.

Produces the same features the original SimpleConv models computed inline,
ensuring numeric equivalence so old checkpoints stay loadable.

``STFTMag`` → log₁₊, 1 channel  (matches SimpleConv, SimpleConvV2, …)
``STFTMagPhase`` → log mag + cos(θ) + sin(θ), 3 channels
                     (matches SimpleConvMagPhaseBiGRU)
``STFTMagIF`` → log mag + instantaneous-frequency deviation, 2 channels
                     (G2b front-end arm — sub-bin frequency evidence)
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from . import SpectralFrontEnd, register_frontend


def if_deviation_bins(
    X: Tensor,
    n_fft: int,
    hop_length: int,
    bin_advance: Tensor,
) -> Tensor:
    """Instantaneous-frequency deviation of each STFT bin, in fractional bins.

    The deviation is the phase difference between two consecutive hops, minus
    the phase advance of the bin-center frequency, wrapped to the principal
    value, and scaled to bins::

        dphi[k, t] = angle(X[k, t]) - angle(X[k, t-1])
        dev[k, t]  = wrap(dphi[k, t] - bin_advance[k])
        IF[k, t]   = dev[k, t] * n_fft / (2*pi*hop_length)

    Parameters
    ----------
    X : Tensor
        Complex STFT, shape ``(B, F, T)``.
    n_fft, hop_length : int
        The STFT parameters that made *X*.
    bin_advance : Tensor
        Per-hop phase advance of each bin center, shape ``(F,)``:
        ``2*pi * hop_length * k / n_fft``.

    Returns
    -------
    Tensor
        Shape ``(B, F, T)``. The first frame has no predecessor, thus it is
        zero. All operations stay on-device: the wrap is ``torch.remainder``,
        never a cumulative ``np.unwrap``.
    """
    phase = torch.angle(X)
    dphi = phase[..., 1:] - phase[..., :-1]  # (B, F, T-1)
    dev = dphi - bin_advance[None, :, None]
    dev = torch.remainder(dev + math.pi, 2.0 * math.pi) - math.pi
    if_dev = dev * (n_fft / (2.0 * math.pi * hop_length))
    return F.pad(if_dev, (1, 0))  # first frame has no predecessor


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
        self.window: Tensor
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
        self.window: Tensor
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


@register_frontend
class STFTMagIF(SpectralFrontEnd):
    """Log-magnitude + instantaneous-frequency deviation — 2-channel STFT input.

    G2b front-end arm (VK-parity campaign, criterion 2.3): gives the trunk
    explicit sub-bin frequency evidence that a magnitude spectrogram lacks.

    Channel 0 is the standard ``log1p`` magnitude (identical to ``STFTMag``).
    Channel 1 is the standard IF estimator: the phase difference between
    consecutive hops, principal-value wrapped, expressed as the deviation from
    the bin-center frequency in *fractional bins*::

        dphi[k, t] = angle(X[k, t]) - angle(X[k, t-1])       # per-hop advance
        dev[k, t]  = wrap(dphi[k, t] - 2*pi*hop*k/n_fft)     # wrap to [-pi, pi)
        IF[k, t]   = dev[k, t] * n_fft / (2*pi*hop)          # fractional bins

    A tone at bin-center frequency gives IF = 0; a tone offset by +0.3 bins
    gives IF = +0.3 at the peak bin (verified in tests). With the project
    default ``hop = n_fft/4`` the channel is bounded to [-2, 2) — a
    well-scaled network input without further normalization.

    All-torch and stays on-device: the wrap uses ``torch.remainder``, never a
    cumulative ``np.unwrap`` (which forced a GPU->CPU sync per forward in the
    old multif0 HCQT path — see ``models/multif0/nnaudio_cqt.py``). The first
    frame (no predecessor hop) is zero-padded.
    """

    key = "stft_mag_if"
    out_channels = 2

    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window: Tensor
        self.register_buffer("window", torch.hann_window(n_fft))
        # Expected per-hop phase advance of each bin-center frequency:
        # 2*pi * (k * sr / n_fft) * (hop / sr) = 2*pi * hop * k / n_fft.
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        self.bin_advance: Tensor
        self.register_buffer("bin_advance", 2.0 * math.pi * hop_length / n_fft * k)

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
        mag = torch.log1p(X.abs())  # (B, F, T)
        if_dev = if_deviation_bins(X, self.n_fft, self.hop_length, self.bin_advance)
        return torch.stack([mag, if_dev], dim=1)  # (B, 2, F, T)
