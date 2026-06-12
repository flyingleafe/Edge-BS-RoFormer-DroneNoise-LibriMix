"""PyTorch port of ``basic_pitch.layers.signal.NormalizedLog``.

Only ``NormalizedLog`` is used by the model (the ``Stft`` / ``Spectrogram``
layers in the original file are unused by ``models.model``).  Faithful port of
https://github.com/spotify/basic-pitch/blob/main/basic_pitch/layers/signal.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NormalizedLog(nn.Module):
    """Rescale a magnitude spectrogram to dB, normalised per-sample to [0, 1].

    Input/output shape: ``(B, time, freq)``.  Adds ``1e-10`` before the log to
    avoid ``NaN``.  Equivalent to the TF layer's rank-3 path.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # convert magnitude to power, then to dB
        power = inputs.square()
        log_power = 10.0 * torch.log10(power + 1e-10)

        log_power_min = torch.amin(log_power, dim=(1, 2), keepdim=True)
        log_power_offset = log_power - log_power_min
        log_power_offset_max = torch.amax(log_power_offset, dim=(1, 2), keepdim=True)

        # tf.math.divide_no_nan: 0 where the denominator is 0
        out = torch.where(
            log_power_offset_max == 0,
            torch.zeros_like(log_power_offset),
            log_power_offset / log_power_offset_max,
        )
        return out
