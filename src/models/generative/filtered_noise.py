"""
Filtered-noise residual generator.

`FilteredNoiseSynth` is a thin port of the DDSP-style filtered-noise synth from
drone_audition. `RPSFilterNet` predicts a time-varying magnitude response from
the rotor-speed (RPS) telemetry — a small 1D-conv network operating at frame
rate, producing per-frame filter magnitudes for the noise synth.

`DroneNoisePlusFilterGen` combines `DroneNoiseGen` (harmonic) with
`FilteredNoiseSynth` (residual) into the full sinusoidal+filter generator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dsp import frequency_filter
from .harmonic_noise_gen import DroneNoiseGen
from .math_utils import exp_sigmoid


class FilteredNoiseSynth(nn.Module):
    """Filter white noise with a (per-frame) magnitude response.

    Args:
        filter_mags: [B, n_frames, n_freqs] — magnitude response per frame on
            a uniform grid 0..Nyquist
        audio_shape: target audio shape, e.g. (B, T)
    """

    def forward(self, filter_mags: torch.Tensor, audio_shape: tuple) -> torch.Tensor:
        noise = (
            torch.rand(audio_shape, device=filter_mags.device, dtype=filter_mags.dtype) * 2.0 - 1.0
        )
        return frequency_filter(noise, filter_mags)


class RPSFilterNet(nn.Module):
    """Predicts per-frame filter magnitudes from RPS telemetry.

    Input: RPS [B, n_motors, T_rps] at some rotor-telemetry rate (e.g. audio
    rate after upsampling, or motor-native ~929 Hz).

    Output: filter magnitudes [B, n_frames, n_freqs]. `n_frames` is determined
    by adaptive average pooling so the filter changes at frame rate
    (independent of the input RPS rate).
    """

    def __init__(
        self,
        n_motors: int = 4,
        n_freqs: int = 65,
        hidden: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_motors = n_motors
        self.n_freqs = n_freqs

        layers: list[nn.Module] = []
        in_ch = n_motors
        for _ in range(n_layers):
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2))
            layers.append(nn.GELU())
            in_ch = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, n_freqs, kernel_size=1)

    def forward(self, rps: torch.Tensor, n_frames: int) -> torch.Tensor:
        """
        Args:
            rps: [B, n_motors, T_rps] in Hz (or any consistent scale; the net
                doesn't care about units)
            n_frames: number of output filter frames
        Returns:
            filter_mags: [B, n_frames, n_freqs]
        """
        # Pool RPS to frame rate first (cheap and stabilises training).
        x = F.adaptive_avg_pool1d(rps, n_frames)  # [B, n_motors, n_frames]
        x = self.body(x)
        x = self.head(x)  # [B, n_freqs, n_frames]
        x = x.transpose(-1, -2)  # [B, n_frames, n_freqs]
        return exp_sigmoid(x)


class DroneNoisePlusFilterGen(nn.Module):
    """Sinusoidal harmonics + filtered noise — full RPS-conditioned generator.

    Forward
    -------
    rps_audio_rate: [B, n_motors, T_audio] RPS upsampled to audio rate (Hz)
    phase_shifts: optional [B, n_motors]. If None, zeros are used.
    audio_length: optional target output length (defaults to T_audio).

    Returns
    -------
    dict with keys:
      - 'audio'     : [B, T_audio]
      - 'harmonic'  : [B, T_audio]
      - 'noise'     : [B, T_audio]
      - 'filter_mags': [B, n_frames, n_freqs]
    """

    def __init__(
        self,
        n_motors: int = 4,
        n_harmonics: int = 50,
        sample_rate: int = 16000,
        use_basis_gain: bool = True,
        filter_n_freqs: int = 65,
        filter_n_frames: int = 64,
        filter_hidden: int = 64,
        filter_n_layers: int = 3,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_motors = n_motors
        self.filter_n_frames = filter_n_frames

        self.harmonic_gen = DroneNoiseGen(
            use_basis_gain=use_basis_gain,
            n_motors=n_motors,
            n_harmonics=n_harmonics,
            sample_rate=sample_rate,
        )
        self.filter_net = RPSFilterNet(
            n_motors=n_motors,
            n_freqs=filter_n_freqs,
            hidden=filter_hidden,
            n_layers=filter_n_layers,
        )
        self.filter_noise = FilteredNoiseSynth()

    def forward(
        self,
        rps_audio_rate: torch.Tensor,
        phase_shifts: torch.Tensor | None = None,
    ) -> dict:
        B, M, T = rps_audio_rate.shape
        assert self.n_motors == M, f"expected {self.n_motors} rotors, got {M}"

        if phase_shifts is None:
            phase_shifts = torch.zeros(
                B, M, device=rps_audio_rate.device, dtype=rps_audio_rate.dtype
            )

        harmonic = self.harmonic_gen(rps_audio_rate, phase_shifts)  # [B, T]

        filter_mags = self.filter_net(rps_audio_rate, self.filter_n_frames)  # [B, F, K]
        noise = self.filter_noise(filter_mags, (B, T))

        audio = harmonic + noise
        return {
            "audio": audio,
            "harmonic": harmonic,
            "noise": noise,
            "filter_mags": filter_mags,
        }
