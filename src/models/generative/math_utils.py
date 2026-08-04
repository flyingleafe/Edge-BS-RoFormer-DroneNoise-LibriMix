"""Math/DSP helpers ported from drone_audition.utils (subset).

Removes dependency on `env.settings` — everything is parameterized.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len

# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def torch_float32(x):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32, device=x.device)
    return torch.tensor(x, dtype=torch.float32)


def signal_frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """Equivalent of tf.signal.frame. With pad_end=True, pads the signal so the
    number of frames covers the entire input (and no further). When the input
    is exactly aligned, no extra frame is added."""
    signal_length = signal.shape[axis]
    if pad_end:
        # ceil((L - frame_length) / step) + 1 frames after padding
        if signal_length <= frame_length:
            pad_size = frame_length - signal_length
        else:
            rest = (signal_length - frame_length) % frame_step
            pad_size = (frame_step - rest) % frame_step
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    return signal.unfold(axis, frame_length, frame_step)


# Canonical home moved to tracking.harmonic_basis (shared with the VP
# transform); re-exported here for the existing generative-model imports.
from tracking.harmonic_basis import overlap_and_add as overlap_and_add  # noqa: E402


def overlap_add_50pct(windowed: torch.Tensor, hop_size: int) -> torch.Tensor:
    """Fold-free overlap-add specialised to 50%-overlapping frames.

    Numerically identical to ``overlap_and_add(windowed, hop_size)`` when the
    frame length is exactly ``2 * hop_size`` (adjacent frames overlap by one
    hop), but avoids the ``F.fold`` / ``aten::col2im`` kernel — which dominated
    the harmonic-amplitude upsampling (~40% of the whole generator forward) and
    is slow on both CPU and GPU. Each frame splits into two half-hop blocks; the
    overlap-add is then two shifted slice-adds instead of a fold.

    Args:
        windowed: ``[..., N, 2*hop_size]`` windowed frames.
        hop_size: hop between frames (= half the frame length).
    Returns:
        ``[..., (N + 1) * hop_size]`` overlap-added signal.
    """
    frame_length = windowed.shape[-1]
    if frame_length != 2 * hop_size:
        raise ValueError(
            f"overlap_add_50pct expects frame length 2*hop_size={2 * hop_size}, got {frame_length}"
        )
    *batch, n_frames, _ = windowed.shape
    first = windowed[..., :hop_size].reshape(*batch, n_frames * hop_size)
    second = windowed[..., hop_size:].reshape(*batch, n_frames * hop_size)
    out = windowed.new_zeros(*batch, (n_frames + 1) * hop_size)
    out[..., : n_frames * hop_size] += first
    out[..., hop_size:] += second
    return out


def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculate FFT size for efficient convolution."""
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        return int(2 ** np.ceil(np.log2(convolved_frame_size)))
    return int(next_fast_len(convolved_frame_size) or 0)


# ---------------------------------------------------------------------------
# Safe math
# ---------------------------------------------------------------------------


def safe_divide(numerator, denominator, eps=1e-7):
    safe_denominator = torch.where(denominator == 0.0, eps, denominator)
    return numerator / safe_denominator


def safe_log(x, eps=1e-5):
    x = torch_float32(x)
    safe_x = torch.where(x <= 0.0, eps, x)
    return torch.log(safe_x)


def logb(x, base=2.0, eps=1e-5):
    return safe_divide(safe_log(x, eps), safe_log(torch.tensor(base), eps), eps)


def log10(x, eps=1e-5):
    return logb(x, base=10, eps=eps)


# ---------------------------------------------------------------------------
# Frequency/amplitude scales
# ---------------------------------------------------------------------------


def midi_to_hz(notes):
    notes = torch_float32(notes)
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))


def hz_to_midi(frequencies):
    frequencies = torch_float32(frequencies)
    notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
    return torch.where(torch.le(frequencies, 0.0), 0.0, notes)


def unit_to_midi(
    unit,
    midi_min: float | torch.Tensor = 20.0,
    midi_max: float | torch.Tensor = 90.0,
    clip: bool = False,
):
    unit = torch.clamp(unit, 0.0, 1.0) if clip else unit
    return midi_min + (midi_max - midi_min) * unit


def midi_to_unit(
    midi,
    midi_min: float | torch.Tensor = 20.0,
    midi_max: float | torch.Tensor = 90.0,
    clip: bool = False,
):
    unit = (midi - midi_min) / (midi_max - midi_min)
    return torch.clamp(unit, 0.0, 1.0) if clip else unit


def unit_to_hz(
    unit, hz_min: float | torch.Tensor, hz_max: float | torch.Tensor, clip: bool = False
):
    midi = unit_to_midi(unit, midi_min=hz_to_midi(hz_min), midi_max=hz_to_midi(hz_max), clip=clip)
    return midi_to_hz(midi)


def hz_to_unit(hz, hz_min: float | torch.Tensor, hz_max: float | torch.Tensor, clip: bool = False):
    midi = hz_to_midi(hz)
    return midi_to_unit(midi, midi_min=hz_to_midi(hz_min), midi_max=hz_to_midi(hz_max), clip=clip)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity (DDSP)."""
    return (
        max_value
        * torch.sigmoid(x) ** torch.log(torch.tensor(exponent, device=x.device, dtype=x.dtype))
        + threshold
    )
