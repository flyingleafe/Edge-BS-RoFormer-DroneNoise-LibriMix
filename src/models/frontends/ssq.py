"""
Synchrosqueezed STFT front-end.

``STFTSSQMag`` → log₁₊ of the synchrosqueezed magnitude, 1 channel
                     (a sharpened drop-in alternative to ``STFTMag``)
"""

import math

import torch
from torch import Tensor

from . import SpectralFrontEnd, register_frontend
from .stft import if_deviation_bins


@register_frontend
class STFTSSQMag(SpectralFrontEnd):
    """Log-magnitude of the synchrosqueezed STFT — 1-channel input.

    Synchrosqueezing moves the energy of each bin to the frequency at which
    the phase of that bin actually advances. The window of the STFT spreads
    a pure tone over many bins, but all of those bins carry the phase advance
    of the same tone. Thus the reassignment collects the leakage back into
    one bin, and a comb of tones becomes a set of sharp ridges::

        P[k, t]      = |X[k, t]|**2
        target[k, t] = clamp(round(k + IF_dev[k, t]), 0, F-1)
        S[j, t]      = sum of P[k, t] over all k with target[k, t] == j
        out[j, t]    = log1p(sqrt(S[j, t]))

    ``IF_dev`` is the instantaneous-frequency deviation in fractional bins,
    identical to channel 1 of ``STFTMagIF`` (``stft.if_deviation_bins``).

    Rotor fundamentals sit at low frequency, where the comb teeth are close
    together and window leakage merges them. This front-end sharpens those
    ridges. It gives 1 channel only, thus it is a cheaper alternative to the
    2-channel ``stft_mag_if``, and the trunk sees the same ``(F, T)`` grid as
    ``stft_mag``.

    The scatter conserves energy exactly, because each bin moves its full
    power to one target bin. The clamp only holds out-of-range mass at the
    edge bins.

    In silence, or in a noise-dominated bin, the deviation is garbage. This
    causes no problem: the power of such a bin is negligible, thus the
    reassignment scatters nothing of importance. An all-zero input gives an
    all-zero output.
    """

    key = "stft_ssq"
    out_channels = 1

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
        self.bin_index: Tensor
        self.register_buffer("bin_index", k.clone())

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
        n_bins = X.shape[-2]
        dev = if_deviation_bins(X, self.n_fft, self.hop_length, self.bin_advance)
        power = X.abs() ** 2  # (B, F, T)

        target = torch.round(self.bin_index[None, :, None] + dev)
        target = target.clamp(0, n_bins - 1).long()

        squeezed = torch.zeros_like(power)
        squeezed.scatter_add_(1, target, power)
        return torch.log1p(squeezed.sqrt()).unsqueeze(1)  # (B, 1, F, T)
