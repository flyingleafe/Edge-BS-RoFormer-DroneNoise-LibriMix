"""
Harmonic propeller-noise synthesizer from RPS telemetry.

Ported from drone_audition.models.harmonic_noise_gen with the only change
being explicit `sample_rate` arguments (no global settings dependency).

Usage
-----
- `PropellerNoiseGen`: single rotor oscillator bank.
- `DroneNoiseGen`: linear combination of N rotor oscillator banks. Takes
  RPS shaped `[B, n_motors, T]` (Hz, i.e. revolutions per second) at the
  AUDIO sample rate and per-rotor phase shifts `[B, n_motors]`.
"""

from __future__ import annotations

import torch
from torch import nn

from .dsp import harmonic_oscillator_bank

EPS = 1e-8


class PolynomialRegression(nn.Module):
    """Polynomial scalar regression y = sum_k w_k * x^k (+ b)."""

    def __init__(self, max_degree: int, bias: bool = True):
        super().__init__()
        self.poly_coeffs = nn.Parameter(torch.abs(torch.randn(max_degree) * 0.0001))
        self.bias = nn.Parameter(torch.randn(1)) if bias else None

    def forward(self, x):
        inp = torch.stack([x**i for i in torch.arange(1, self.poly_coeffs.shape[0] + 1)], dim=-1)
        out = torch.matmul(inp, self.poly_coeffs.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            out = out + self.bias
        return out


class PropellerNoiseGen(nn.Module):
    """Single propeller (rotor) sinusoidal noise generator.

    Args:
        use_basis_gain: if True, an RPS-dependent global gain modulates the
            harmonic amplitudes (so the spectrum can change shape with speed).
        n_harmonics: number of harmonics generated.
        sample_rate: audio sample rate.
    """

    def __init__(
        self,
        use_basis_gain: bool = True,
        n_harmonics: int = 50,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.basis_gain_function = PolynomialRegression(2, bias=False) if use_basis_gain else None
        self.harmonic_gain_corrections = nn.Parameter(torch.randn(n_harmonics))
        # First harmonic's phase is fixed to phase_shift; corrections apply to
        # harmonics 2..N relative to that reference.
        self.harmonic_phase_corrections = nn.Parameter(torch.randn(n_harmonics - 1) * 0.001)

    def forward(self, speed_rps: torch.Tensor, phase_shift: torch.Tensor):
        """
        Args:
            speed_rps: [..., T] — fundamental rotation frequency in Hz
            phase_shift: [...] — per-rotor initial phase reference (radians)
        Returns:
            [..., T] audio
        """
        assert speed_rps.shape[:-1] == phase_shift.shape
        assert speed_rps.device == phase_shift.device

        # Per-harmonic initial phases: k * phase_shift + per-harmonic correction
        coeffs = torch.arange(
            1, self.n_harmonics + 1, device=speed_rps.device, dtype=speed_rps.dtype
        )
        phase_corrections = torch.zeros(
            self.n_harmonics, device=speed_rps.device, dtype=speed_rps.dtype
        )
        phase_corrections[1:] = self.harmonic_phase_corrections

        harmonic_phase_shifts = (
            torch.matmul(phase_shift.unsqueeze(-1), coeffs.unsqueeze(0)) + phase_corrections
        )

        # gains shape: [..., n_harmonics, T] (broadcasts naturally)
        h_corr = self.harmonic_gain_corrections.unsqueeze(-1)  # [N, 1]
        if self.basis_gain_function is not None:
            # /343 is historical (speed of sound) — what matters is keeping the
            # polynomial input small. Leaves room for the model to learn freely.
            basis_gains = self.basis_gain_function(speed_rps / 343.0)  # [..., T]
            gains = basis_gains.unsqueeze(-2) + h_corr  # [..., N, T]
        else:
            # Broadcast [N, 1] -> [..., N, T] against speed_rps
            gains = h_corr.expand(self.n_harmonics, speed_rps.shape[-1])
            for _ in range(speed_rps.dim() - 1):
                gains = gains.unsqueeze(0)
            gains = gains.expand(*speed_rps.shape[:-1], self.n_harmonics, speed_rps.shape[-1])

        return harmonic_oscillator_bank(
            speed_rps,
            torch.exp(gains) + EPS,
            sr=self.sample_rate,
            initial_phases=harmonic_phase_shifts,
        )


class DroneNoiseGen(nn.Module):
    """Multi-rotor drone noise synthesizer.

    Args:
        use_basis_gain: see PropellerNoiseGen
        n_motors: number of rotors (4 for a typical quadcopter)
        n_harmonics: harmonics per rotor
        sample_rate: audio sample rate

    Forward
    -------
    speed_rps: [B, n_motors, T_audio] — RPS upsampled to audio rate (Hz)
    phase_shifts: [B, n_motors] — per-rotor reference phases (radians)

    Returns: [B, T_audio] mono audio
    """

    def __init__(
        self,
        use_basis_gain: bool = True,
        n_motors: int = 4,
        n_harmonics: int = 50,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_motors = n_motors
        self.propeller = PropellerNoiseGen(use_basis_gain, n_harmonics, sample_rate)
        # Learned per-rotor mixing coefficients (positivity via exp)
        self.log_motor_coeffs = nn.Parameter(torch.rand(n_motors))

    def forward(self, speed_rps: torch.Tensor, phase_shifts: torch.Tensor):
        assert speed_rps.shape[-2] == self.n_motors, (
            f"speed_rps must have shape [..., {self.n_motors}, T], got {tuple(speed_rps.shape)}"
        )
        propeller_outputs = self.propeller(speed_rps, phase_shifts)
        # [B, n_motors, T_audio]  ->  [B, T_audio]
        coeffs = torch.exp(self.log_motor_coeffs)  # [n_motors]
        return (coeffs.view(*([1] * (propeller_outputs.dim() - 2)), -1, 1) * propeller_outputs).sum(
            -2
        )
