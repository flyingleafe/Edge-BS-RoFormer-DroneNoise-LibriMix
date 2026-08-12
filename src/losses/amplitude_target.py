"""The amplitude-target objective: fit the VK decomposition, not the waveform.

Why this loss exists
--------------------
The audio-domain multi-scale STFT loss cannot defend mid/high-k harmonics of a
real recording. Real lines decohere INSIDE its analysis windows (shaft-speed
wander sigma ~0.6 rev/s gives ~0.24*k rad of phase wander per 2048-sample
window), so each window's band magnitude fluctuates around a low median, and a
log-L1 term fits that median — measured as a persistent downward gradient on
every steady rendered line above k~25 (docs/experiments/
generator-perrotor-dynamics.md, finding 6).

The Vold-Kalman decomposition removes phase from the problem entirely: it is
comb-COHERENT demodulation, so it yields one amplitude ENVELOPE per (rotor,
harmonic, microphone) plus a broadband residual, at 100 Hz
(``scripts/vk_decompose.py``, published as ``decomp-frames-v1``). Fitting those
envelopes never compares two realizations of a decohering line, and the high-k
amplitude information survives.

The two terms
-------------
``amp``
    ``L1`` on ``log(amplitude + eps)`` over the valid ``(mic, rotor, k, frame)``
    cells. Cell weight is 1 — deliberately: the targets already ISOLATE each
    harmonic, so a high-k line is no longer a hundredth of a band's energy and
    needs no ``k`` weighting to be seen. The model's prediction comes from
    :meth:`models.generative.PositionalHarmonicNoiseGen.amp_stats` (1/r gains,
    no delays, no rotor sum, no synthesis, no RPS jitter).
``psd``
    ``L1`` on the log per-microphone broadband POWER of the residual waveform,
    on the noise branch's own uniform ``0..Nyquist`` band grid. The prediction
    is the power-summed per-rotor broadband envelope; the target is the
    residual's band power, so the two are commensurate up to one constant per
    drone, which the model's ``log_gain_noise`` absorbs.

Alignment and floors
--------------------
The predicted control curves live on the emitter's own frame grid (hop 512, so
31.25 Hz) and the targets on the decomposition's 100 Hz grid. Targets are
resampled onto the prediction grid, treating both as uniform grids over the same
chunk — the residual sub-frame ambiguity is at most half of a 32 ms frame, well
inside an envelope's own bandwidth (the solve's per-track bandwidth is clamped
at 1 Hz).

``eps`` is a floor, not a regularizer: below it the log is flat and the cell
stops contributing. It must sit at the decomposition's own noise floor —
amplitudes of weak bands absorb floor noise proportional to the solved
bandwidth (docs/experiments/vk-decomposition.md, finding 4). ``1.6e-5`` is the
1st percentile of the valid DREGON amplitudes at ``bw_rps=1``, the campaign's
setting (the three published recordings give 1.58e-5 / 1.70e-5 / 2.20e-5, so one
constant covers the dataset). A v2 could instead DEBIAS each track by its noise-equivalent bandwidth
(``Envelopes.bw_track`` is published for exactly that) and drop the fixed floor;
that is deliberately not done here, so v1 has one bias, shared by every band.
"""

from __future__ import annotations

from typing import Any

import tdseries as td
import torch
from torch import nn
from torch.nn import functional as F

from framespec import FrameSpec, SeriesSpec
from losses._common import AUDIO_RATE, audio_series_spec, get_tensor

__all__ = ["AmplitudeTarget", "AmplitudeTargetLoss", "band_powers", "resample_time"]


def resample_time(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Linear-resample the LAST axis of ``[..., T]`` to ``n_frames``.

    Both grids are read as uniform covers of the same span (``align_corners``),
    which is what makes the 100 Hz target grid and the emitter's 31.25 Hz
    control grid comparable without either side committing to a frame offset.
    """
    if x.shape[-1] == n_frames:
        return x
    lead = x.shape[:-1]
    flat = x.reshape(-1, 1, x.shape[-1])
    out = F.interpolate(flat, size=n_frames, mode="linear", align_corners=True)
    return out.reshape(*lead, n_frames)


def band_powers(
    audio: torch.Tensor, n_bands: int, *, n_fft: int = 2048, hop_length: int = 512
) -> torch.Tensor:
    """``[..., T]`` waveform -> ``[..., n_frames, n_bands]`` mean band POWER.

    The bands are the ``n_bands`` equal slices of ``0..Nyquist`` — the grid the
    generator's broadband branch predicts a magnitude response on
    (``models.generative.dsp.frequency_filter`` reads its magnitudes as samples
    of ``|H|`` on a uniform grid), so a band mean of ``|STFT|^2 / ||w||^2`` is
    the matching measurement of the recording. The window-power division is the
    standard periodogram normalization for unit-variance white excitation, the
    same one :mod:`losses.spectral_likelihood` uses.
    """
    lead = audio.shape[:-1]
    x = audio.reshape(-1, audio.shape[-1])
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True
    )
    power = spec.abs().pow(2) / window.pow(2).sum()  # [N, F, frames]
    banded = F.adaptive_avg_pool1d(power.transpose(1, 2), n_bands)  # [N, frames, n_bands]
    return banded.reshape(*lead, banded.shape[-2], banded.shape[-1])


class AmplitudeTarget(nn.Module):
    """The tensor core: log-L1 on amplitudes + log-L1 on residual band power.

    Args:
        eps: amplitude floor (see the module docstring). Absolute, in the
            recording's own units — the model's calibration gains put its
            predictions in those units.
        psd_eps: power floor for the broadband term, in the same units squared.
        psd_weight: weight of the broadband term, measured rather than guessed.
            On a first batch of ``decomp-frames-v1`` the two terms are 7.14 and
            7.93 log-units at initialization — both dominated by the absolute
            unit constant, which the model's global gains absorb within the first
            steps. At that operating point (the one-scalar fit applied) they are
            **1.08 (amp) and 2.54 (psd)**, so ``0.5`` brings the broadband term
            to 1.27 and the two enter the sum within 18 % of each other: neither
            silenced, and the amplitude term — the objective's whole purpose —
            not outvoted by the branch that is only there to stay constrained.
        n_fft / hop_length: analysis of the residual waveform.
    """

    def __init__(
        self,
        *,
        eps: float = 1.6e-5,
        psd_eps: float = 1e-12,
        psd_weight: float = 0.5,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.psd_eps = float(psd_eps)
        self.psd_weight = float(psd_weight)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def amplitude_term(
        self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """``[B, M, R, H, t_a]`` prediction vs ``[B, M, R, K, t_e]`` target.

        ``valid`` is ``[B, R, K, t_e]`` (microphone-independent: a track either
        was solved or was not). The harmonic axes are intersected — the emitter
        may model more harmonics than the decomposition solved — and the target
        is resampled onto the prediction's frame grid. A frame of the coarse
        grid counts as valid only if EVERY target frame under it is, so a
        partially covered edge never leaks an unsolved envelope into the loss.
        """
        n_k = min(int(pred.shape[-2]), int(target.shape[-2]))
        t_a = int(pred.shape[-1])
        pred = pred[..., :n_k, :]
        tgt = resample_time(target[..., :n_k, :], t_a)
        vmask = valid[..., :n_k, :].to(pred.dtype)
        vmask = resample_time(vmask, t_a) >= 1.0 - 1e-6  # [B, R, n_k, t_a]
        vmask = vmask.unsqueeze(1).expand_as(pred)
        if not bool(vmask.any()):
            return pred.new_zeros(())
        diff = (torch.log(pred + self.eps) - torch.log(tgt + self.eps)).abs()
        return (diff * vmask).sum() / vmask.sum().clamp_min(1)

    def psd_term(self, pred: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """``[B, M, t_n, F]`` predicted power vs the residual's band power."""
        target = band_powers(
            residual, int(pred.shape[-1]), n_fft=self.n_fft, hop_length=self.hop_length
        )
        # [B, M, frames, F] -> the prediction's frame count (power averages).
        tgt = resample_time(target.transpose(-1, -2), int(pred.shape[-2])).transpose(-1, -2)
        return (
            (torch.log(pred.clamp_min(0.0) + self.psd_eps) - torch.log(tgt + self.psd_eps))
            .abs()
            .mean()
        )

    def forward(
        self,
        amp_pred: torch.Tensor,
        amp_target: torch.Tensor,
        amp_valid: torch.Tensor,
        psd_pred: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        amp = self.amplitude_term(amp_pred, amp_target, amp_valid)
        psd = self.psd_term(psd_pred, residual)
        return amp + self.psd_weight * psd


class AmplitudeTargetLoss(nn.Module):
    """Frame adapter around :class:`AmplitudeTarget`.

    Consumes the generator's ``amp_pred``/``noise_psd`` (produced by
    :class:`tasks.codecs.NoiseGenerationCodec` in ``amplitude`` mode) and the
    dataset's ``amp``/``amp_valid``/``residual`` entries (published as
    ``decomp-frames-v1``, loaded by
    :class:`data_processing.frame_datasets.DecompFrameDataset`).
    """

    def __init__(
        self,
        *,
        sr: tuple[int, int] = AUDIO_RATE,
        amp_key: str = "amp_pred",
        psd_key: str = "noise_psd",
        target_amp_key: str = "amp",
        valid_key: str = "amp_valid",
        residual_key: str = "residual",
        **core_kwargs: Any,
    ) -> None:
        super().__init__()
        self.core = AmplitudeTarget(**core_kwargs)
        self.amp_key = amp_key
        self.psd_key = psd_key
        self.target_amp_key = target_amp_key
        self.valid_key = valid_key
        self.residual_key = residual_key
        self.requires_pred = FrameSpec(
            {
                amp_key: SeriesSpec(dims=("batch", "mic", "rotor", None, None), time=None),
                psd_key: SeriesSpec(dims=("batch", "mic", None, None), time=None),
            }
        )
        self.requires_target = FrameSpec(
            {
                target_amp_key: SeriesSpec(
                    dims=("batch", "mic", "rotor", "k", "time"), time="grid"
                ),
                valid_key: SeriesSpec(dims=("batch", "rotor", "k", "time"), time="grid"),
                residual_key: audio_series_spec(1, sr),
            }
        )

    def forward(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        return self.core(
            get_tensor(pred, self.amp_key),
            get_tensor(target, self.target_amp_key),
            get_tensor(target, self.valid_key),
            get_tensor(pred, self.psd_key),
            get_tensor(target, self.residual_key),
        )
