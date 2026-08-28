"""Learned time-domain filterbank front-end.

WHY THIS EXISTS. Every front-end in this package hands the trunk a function of
the STFT MAGNITUDE (or of a magnitude-derived quantity such as the
instantaneous-frequency deviation). Magnitude discards the phase, and the
campaign's models converge on a degenerate answer: they emit a fixed rotor
spread of about 10 rev/s whatever the rotors actually do, which is what a model
that reads a comb's mean spacing rather than its individual lines would produce.
Recovering a rotor's speed to the precision the signal supports is a phase
problem — the classical refiner that reaches 0.0006 rev/s on a single-rotor comb
works entirely in phase increments.

So this front-end learns its own analysis filters on the RAW WAVEFORM and hands
the trunk their raw responses. Phase is then available to the network rather
than removed before it starts.

THE STFT IS A STRICT SUBSET. A convolution of length ``n_fft`` at stride
``hop_length`` with ``2F`` free filters contains the windowed DFT basis exactly:
the cosine and sine rows ARE such filters. ``init="stft"`` starts the bank at
that basis, so training begins bit-near the `stft_mag` baseline (with the phase
channels added) and can only move away from it if that helps. This separates
"do learned filters help" from "can gradient descent find the STFT", which a
random init would confound.

Output channels are ``(real, imag, log-magnitude)``: the magnitude channel is
redundant given the first two, but supplying it means the trunk starts from the
representation every previous arm was given rather than having to synthesize it.
Set ``include_mag=False`` for the 2-channel ablation.

NOTE ON THE BASE-CLASS CONTRACT. `SpectralFrontEnd` documents its forward as
deterministic and parameter-free. This front-end deliberately breaks the second
half of that: its filters are `nn.Parameter` and train with the rest of the
model. It stays deterministic.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import SpectralFrontEnd, register_frontend


def _stft_basis(n_fft: int, n_rows: int) -> Tensor:
    """``(2*n_rows, n_fft)`` windowed DFT rows: cosine bank then sine bank.

    Row ``f`` of each bank is the Hann-windowed sinusoid at bin ``f``, scaled
    the way ``torch.stft(..., normalized=True)`` scales it, so a bank frozen at
    this value reproduces that transform's real and imaginary parts.
    """
    win = torch.hann_window(n_fft, periodic=True)
    t = torch.arange(n_fft, dtype=torch.float64)
    rows = torch.arange(n_rows, dtype=torch.float64)[:, None]
    ang = 2.0 * math.pi * rows * t[None, :] / n_fft
    scale = 1.0 / math.sqrt(n_fft)
    cos = (torch.cos(ang) * win.double()[None, :]) * scale
    sin = (-torch.sin(ang) * win.double()[None, :]) * scale
    return torch.cat([cos, sin], dim=0).float()


@register_frontend
class LearnedConvFrontEnd(SpectralFrontEnd):
    """Free time-domain filterbank: ``(B, N) -> (B, C, F, T)``.

    ``2*n_rows`` filters of length ``n_fft`` at stride ``hop_length``. The
    responses are read as a complex pair per row, so ``F = n_rows`` and the
    time grid matches `stft_mag`'s exactly (centred, ``n_samples//hop + 1``).

    Args:
        n_fft: filter length, and the DFT size ``init="stft"`` reproduces.
        hop_length: stride, so the output shares the `stft_mag` time grid.
        n_rows: frequency rows. Default ``n_fft//2 + 1`` makes the output the
            same shape the trunk already receives, so nothing downstream moves.
        init: ``"stft"`` starts at the windowed DFT basis; ``"random"`` uses
            He-scaled noise, which measures how much of any gain is the
            initialization rather than the freedom.
        include_mag: append ``log1p|z|`` as a third channel.
        trainable: ``False`` freezes the bank, which with ``init="stft"`` is a
            pure control — the same trunk on real/imag/log-mag STFT channels
            with no learning in the front-end at all.
    """

    key = "learned_conv"

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_rows: int | None = None,
        init: str = "stft",
        include_mag: bool = True,
        trainable: bool = True,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_rows = int(n_rows) if n_rows is not None else self.n_fft // 2 + 1
        self.include_mag = bool(include_mag)
        self.out_channels = 3 if self.include_mag else 2

        if init == "stft":
            w = _stft_basis(self.n_fft, self.n_rows)
        elif init == "random":
            w = torch.randn(2 * self.n_rows, self.n_fft) / math.sqrt(self.n_fft)
        else:
            raise ValueError(f"init must be 'stft' or 'random', got {init!r}")
        # (2F, 1, n_fft) — one input channel, the waveform.
        self.filters = nn.Parameter(w.unsqueeze(1), requires_grad=bool(trainable))

    def num_frames(self, n_samples: int) -> int:
        return n_samples // self.hop_length + 1

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        # Centre the analysis window the way torch.stft(center=True) does, so
        # frame t is centred on sample t*hop and the grid matches every other
        # front-end in this package.
        pad = self.n_fft // 2
        x = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect")
        y = F.conv1d(x, self.filters, stride=self.hop_length)  # (B, 2F, T)
        re, im = y[:, : self.n_rows], y[:, self.n_rows :]
        chans = [re, im]
        if self.include_mag:
            chans.append(torch.log1p(torch.sqrt(re * re + im * im + 1e-12)))
        return torch.stack(chans, dim=1)  # (B, C, F, T)
