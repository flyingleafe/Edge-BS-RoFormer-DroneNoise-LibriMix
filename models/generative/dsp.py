"""
DSP primitives ported from drone_audition.dsp.

Self-contained — no dependency on `env.settings`. Every function that needs
a sample rate accepts one as an explicit argument.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .math_utils import get_fft_size, overlap_and_add, signal_frame


# ---------------------------------------------------------------------------
# Phase / oscillator banks
# ---------------------------------------------------------------------------

def harmonic_freq_series(freq: torch.Tensor, n_harmonics: int):
    """Make N harmonics from a fundamental-frequency series.

    Args:
        freq: [..., T] instantaneous fundamental frequency (Hz)
        n_harmonics: number of harmonics including the fundamental
    Returns:
        [..., N, T] frequencies of each harmonic per time step
    """
    coeffs = torch.arange(1, n_harmonics + 1, device=freq.device, dtype=freq.dtype)
    return torch.matmul(coeffs.unsqueeze(-1), freq.unsqueeze(-2))


def remove_above_nyquist(freqs, amps, sr: int):
    """Zero amplitudes at time steps where frequency exceeds Nyquist."""
    assert freqs.shape == amps.shape
    return torch.where(freqs > sr / 2, torch.zeros_like(amps), amps)


def freqs_to_phasors(freq: torch.Tensor, sr: int):
    """Convert frequency series to rotating complex phasors (cumprod form)."""
    phase_diff = freq * 2 * torch.pi / sr
    complex_diffs = torch.exp(1j * phase_diff)
    return torch.cumprod(complex_diffs, -1)


def oscillator_bank(freqs, amps, initial_phases=None, return_sum=True, sr: int = 16000):
    """Synthesize a sum of sinusoids with time-varying freq & amp.

    Args:
        freqs: [..., N, T]
        amps:  [..., N, T]
        initial_phases: optional [..., N] initial phase offsets
        sr: sample rate
    """
    assert freqs.shape == amps.shape
    amps = remove_above_nyquist(freqs, amps, sr)
    phasors = freqs_to_phasors(freqs, sr)
    if initial_phases is not None:
        phasors = phasors * torch.exp(1j * initial_phases.unsqueeze(-1))
    cosines = phasors.real
    oscillators = amps * cosines
    if return_sum:
        return oscillators.sum(-2)
    return oscillators


def harmonic_oscillator_bank(freqs, amps, sr: int = 16000, **kwargs):
    """Synthesize harmonic series. `freqs` is the fundamental.

    Args:
        freqs: [..., T] fundamental frequency series
        amps:  [..., N, T] amplitudes per harmonic (N = amps.shape[-2])
    """
    harmonic_freqs = harmonic_freq_series(freqs, amps.shape[-2])
    return oscillator_bank(harmonic_freqs, amps, sr=sr, **kwargs)


# ---------------------------------------------------------------------------
# Filtered noise (DDSP-style frequency-domain FIR)
# ---------------------------------------------------------------------------

def crop_and_compensate_delay(audio, audio_size, ir_size, padding, delay_compensation):
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError(f"padding must be 'valid' or 'same', got {padding}")
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = (ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation
    end = crop - start
    return audio[:, start:-end] if end > 0 else audio[:, start:]


def fft_convolve(audio, impulse_response, padding="same", delay_compensation=-1):
    """Convolve audio with time-varying impulse response via overlap-add."""
    batch_size, audio_size = audio.shape
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(-2)
    if ir_shape[0] == 1 and batch_size > 1:
        impulse_response = impulse_response.expand(batch_size, -1, -1)

    batch_size_ir, n_ir_frames, ir_size = impulse_response.shape
    assert batch_size == batch_size_ir, "audio and IR batch must match"

    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    audio_frames = signal_frame(audio, frame_size, hop_size, pad_end=True)
    n_audio_frames = audio_frames.shape[1]
    assert n_audio_frames == n_ir_frames, (
        f"frames mismatch: audio={n_audio_frames}, ir={n_ir_frames}"
    )

    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(impulse_response, fft_size)
    audio_ir_fft = audio_fft * ir_fft
    audio_frames_out = torch.fft.irfft(audio_ir_fft)
    audio_out = overlap_and_add(audio_frames_out, hop_size)
    return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding, delay_compensation)


def apply_window_to_impulse_response(impulse_response, window_size=0, causal=False):
    if causal:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    ir_size = int(impulse_response.shape[-1])
    if window_size <= 0 or window_size > ir_size:
        window_size = ir_size
    window = torch.hann_window(window_size, device=impulse_response.device, dtype=impulse_response.dtype)

    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:], torch.zeros([padding], device=window.device, dtype=window.dtype), window[:half_idx]], dim=0)
    else:
        window = torch.fft.fftshift(window, dim=-1)

    window = window.expand_as(impulse_response)
    impulse_response = window * impulse_response.real

    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat(
            [impulse_response[..., first_half_start:], impulse_response[..., :second_half_end]], dim=-1
        )
    else:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)
    return impulse_response


def frequency_impulse_response(magnitudes, window_size=0):
    """Get windowed IRs from a magnitude-only frequency response (zero phase)."""
    magnitudes = torch.view_as_complex(
        torch.stack([magnitudes, torch.zeros_like(magnitudes)], -1)
    )
    impulse_response = torch.fft.irfft(magnitudes)
    return apply_window_to_impulse_response(impulse_response, window_size)


def frequency_filter(audio, magnitudes, window_size=0, padding="same"):
    """Filter audio with a (possibly time-varying) magnitude-only FIR filter.

    Args:
        audio: [B, T]
        magnitudes: [B, n_frames, n_freqs] or [B, n_freqs] — magnitude response on a
            uniform freq grid from 0 to Nyquist
    """
    impulse_response = frequency_impulse_response(magnitudes, window_size=window_size)
    return fft_convolve(audio, impulse_response, padding=padding)
