"""CQT front-end for Basic Pitch.

Basic Pitch's ``basic_pitch.layers.nnaudio`` is a TensorFlow re-port of the
PyTorch `nnAudio <https://github.com/KinWaiCheuk/nnAudio>`_ library's
``CQT2010v2``.  Since nnAudio is already a project dependency we use it
directly rather than re-porting the TF code; the parameters below reproduce
``basic_pitch.models.get_cqt`` exactly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from nnAudio.features import CQT2010v2

from .signal import NormalizedLog

# basic_pitch.constants
FFT_HOP = 256
ANNOTATIONS_BASE_FREQUENCY = 27.5
ANNOTATIONS_N_SEMITONES = 88
AUDIO_SAMPLE_RATE = 22050
CONTOURS_BINS_PER_SEMITONE = 3
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

MAX_N_SEMITONES = int(
    np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY))
)


def max_n_semitones_for(sr: int = AUDIO_SAMPLE_RATE) -> int:
    """Highest representable semitone count above fmin for a given Nyquist."""
    return int(np.floor(12.0 * np.log2(0.5 * sr / ANNOTATIONS_BASE_FREQUENCY)))


def n_semitones_for(n_harmonics: int, sr: int = AUDIO_SAMPLE_RATE) -> int:
    """Replicates ``get_cqt``'s semitone count derivation.

    ``sr`` only sets the Nyquist ceiling: at 16 kHz the harmonic-stacking CQT
    caps at 98 semitones above 27.5 Hz (vs 103 at 22.05 kHz). The 264-bin
    contour grid (88 semitones) fits either way, so the model's salience grid
    is sample-rate-invariant; only the upper harmonic channels lose content.
    """
    return int(
        np.min(
            [
                int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
                max_n_semitones_for(sr),
            ]
        )
    )


class CQTFrontEnd(nn.Module):
    """``get_cqt`` without the trailing BatchNorm: CQT -> NormalizedLog.

    Input:  ``(B, n_samples)`` audio at ``sr`` Hz (default 22050).
    Output: ``(B, 1, time, n_freq_bins)`` (channels-first, ch=1).

    ``sr`` is configurable so the model can run natively at 16 kHz (matching the
    rest of the RPS pipeline) without resampling. The pretrained ICASSP-2022
    weights, however, assume 22050 — only use ``sr != 22050`` when training from
    scratch.
    """

    def __init__(
        self,
        n_harmonics: int = 8,
        sr: int = AUDIO_SAMPLE_RATE,
        fmin: float = ANNOTATIONS_BASE_FREQUENCY,
        bins_per_semitone: int = CONTOURS_BINS_PER_SEMITONE,
        n_contour_semitones: int = ANNOTATIONS_N_SEMITONES,
    ):
        super().__init__()
        self.sr = sr
        self.fmin = fmin
        self.bins_per_semitone = bins_per_semitone
        # The CQT spans the contour output band plus enough extra semitones above
        # it for the harmonic-stacking shifts (ceil(12·log2(n_harmonics))), capped
        # by Nyquist for this ``fmin``.
        max_semitones = int(np.floor(12.0 * np.log2(0.5 * sr / fmin)))
        cqt_semitones = min(
            int(np.ceil(12.0 * np.log2(max(n_harmonics, 1))) + n_contour_semitones),
            max_semitones,
        )
        self.n_bins = cqt_semitones * bins_per_semitone
        self.contour_bins = n_contour_semitones * bins_per_semitone
        self.cqt = CQT2010v2(
            sr=sr,
            hop_length=FFT_HOP,
            fmin=fmin,
            n_bins=self.n_bins,
            bins_per_octave=12 * bins_per_semitone,
            output_format="Magnitude",
            verbose=False,
        )
        self.normalized_log = NormalizedLog()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, n_samples) -> nnAudio CQT magnitude (B, freq, time)
        x = self.cqt(audio)
        # match basic_pitch layout: (B, time, freq)
        x = x.transpose(1, 2)
        x = self.normalized_log(x)
        # add channel axis -> (B, 1, time, freq)
        return x.unsqueeze(1)
