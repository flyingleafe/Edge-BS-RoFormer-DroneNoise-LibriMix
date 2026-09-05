"""PyTorch port of ``basic_pitch.nn`` (Spotify Basic Pitch).

Shape-only layers (no trainable parameters): harmonic stacking and the two
flatten helpers.  Faithful re-implementations of the TensorFlow originals at
https://github.com/spotify/basic-pitch/blob/main/basic_pitch/nn.py

The TF model works in channels-last ``(B, time, freq, ch)`` layout.  Here we
use PyTorch's channels-first ``(B, ch, time, freq)`` (NCHW with H=time,
W=freq) so that the convolutions in :mod:`model` are ordinary ``nn.Conv2d``.
The shift / concat / slice algebra is identical; only the moved axis differs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEMITONES_PER_OCTAVE = 12


class HarmonicStacking(nn.Module):
    """Harmonic stacking layer.

    Input shape:  ``(B, 1, time, n_freqs)``
    Output shape: ``(B, len(harmonics), time, n_output_freqs)``

    For each harmonic ``h`` the input is shifted along the frequency axis by
    ``round(12 * bins_per_semitone * log2(h))`` bins (filling with zeros), so
    that the energy of the ``h``-th harmonic lines up with the fundamental.
    ``n_freqs`` should be much larger than ``n_output_freqs`` so that the upper
    harmonics still have valid (non-zero-padded) content.
    """

    def __init__(self, bins_per_semitone: int, harmonics: list[float], n_output_freqs: int):
        super().__init__()
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.n_output_freqs = n_output_freqs
        # round-half-to-even to match tf.math.round
        self.shifts = [
            int(
                torch.round(
                    torch.tensor(SEMITONES_PER_OCTAVE * bins_per_semitone * math.log2(float(h)))
                ).item()
            )
            for h in harmonics
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, time, freq)
        assert x.dim() == 4
        channels = []
        for shift in self.shifts:
            if shift == 0:
                padded = x
            elif shift > 0:
                # drop the lowest `shift` bins, zero-pad on the high-freq side
                padded = F.pad(x[:, :, :, shift:], (0, shift))
            else:  # shift < 0
                # zero-pad on the low-freq side, drop the highest `-shift` bins
                padded = F.pad(x[:, :, :, :shift], (-shift, 0))
            channels.append(padded)
        x = torch.cat(channels, dim=1)
        x = x[:, :, :, : self.n_output_freqs]
        return x


def flatten_freq_ch(x: torch.Tensor) -> torch.Tensor:
    """Flatten the channel dimension into the frequency dimension.

    NCHW input ``(B, C, time, freq)`` -> ``(B, time, C * freq)``. The note and
    onset heads are single-channel, so ``C == 1`` reduces to squeezing the
    channel axis, exactly as before.

    THE CHANNEL IS THE SLOW AXIS. A multi-map contour head (``BasicPitch(
    n_maps=R)``, the per-rotor salience layers) stacks its maps here, and the
    consumer is ``models.harmonic_ports.layer_readout.split_maps``, which reads
    ``(B, R*G, T)`` back as ``(B, R, G, T)``. That is map-major, thus each map
    keeps its ``freq`` bins contiguous.
    """
    b, c, t, f = x.shape
    if c == 1:
        return x[:, 0, :, :]
    return x.permute(0, 2, 1, 3).reshape(b, t, c * f)
