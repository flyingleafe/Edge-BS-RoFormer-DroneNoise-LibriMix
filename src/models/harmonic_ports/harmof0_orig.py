"""HarmoF0 (Wei et al., ISMIR 2022) UNMODIFIED, as a rotor-rate salience model.

WHY THIS FILE EXISTS. `models.harmonic_ports.harmof0_rps` is HarmoF0 with its
harmonic organ replaced: `MRDConv`'s log-axis SHIFT became an explicit gather at
``k * r`` on the linear STFT, and the 352-bin semitone grid became a 0-150 rev/s
linear grid. Those two substitutions are the campaign's claim, and a claim needs
a control. This module is that control — the published architecture, kept whole,
wired into the same `salience_rps` task, so the pair
``hb_sal_hf0_orig`` / ``hf0_*`` isolates what the substitutions buy.

WHAT IS KEPT. Everything the paper has:

* `WaveformToLogSpecgram` — a uniform STFT (``n_fft = 2 * n_freq``, Hann
  window, POWER) linearly interpolated onto ``freq_bins`` log-spaced bins at
  ``bins_per_octave`` from ``fmin``, then `AmplitudeToDB(top_db=80)`.
* `MRDConv` — the multi-rate dilated convolution. One 1x1 convolution per
  harmonic ``k``, its output read at an offset of ``round(log2(k) * B)`` bins,
  all summed. On a log axis that offset IS the position of ``k * f``.
* Blocks 2-4 — ``conv 3x3 -> ReLU -> conv [1,3] dilated by B -> ReLU ->
  BatchNorm``, the octave-sized dilated context convolutions.
* ``conv_5``/``conv_6`` — the 128 -> 64 -> 1 pair of 1x1 convolutions that emit
  the salience map.
* The layout ``(B, C, T, freq)``, so every kernel walks the axes the paper says.

The defaults are HarmoF0's own: ``n_freq=512`` (thus ``n_fft=1024``),
``fmin=27.5``, ``bins_per_octave=48``, ``freq_bins=352``, ``n_har=12``,
``channels=(32, 64, 128, 128)``, ``dilation_rates=(48, 48, 48, 48)``.

DEVIATIONS, all forced by the task seam and none of them structural:

1. HOP. The published `PitchTracker` frames the waveform at ``hop = 160``
   (10 ms) OUTSIDE the network and hands it ``[b, num_frames, 1024]``. Here the
   framing is `torch.stft` at ``hop = 512``, this project's frame grid, with
   ``center=True`` so the frame count is ``n // hop + 1`` and matches every
   dataset, loss and metric in the campaign. The per-frame arithmetic is
   identical (Hann window, then squared magnitude of the rfft); only the frame
   POSITIONS move, plus the reflect padding at the two ends that ``center=True``
   adds. ``n_fft`` stays 1024, so the analysis window and the 15.625 Hz bin
   spacing are the paper's.
2. LOGITS. HarmoF0 ends in a sigmoid. `losses.SalienceRPSBCELoss` takes LOGITS
   (``BCEWithLogits``), so the sigmoid moves out of the model and into the loss.
   This is the same model: ``BCEWithLogitsLoss(x) == BCELoss(sigmoid(x))``.
3. BATCH AXIS. The released ``forward`` does ``x = specgram[None, :]``, which
   promotes the BATCH to the channel axis and only computes the paper's model
   when ``b == 1`` (their `PitchTracker` asserts exactly that). Here the batch
   stays the batch and the channel axis is 1, which is what the training code in
   the same repository (``train/src/mono_pitch/nets.py``) does.
4. MULTI-PITCH. HarmoF0 is monophonic — one map, one peak per frame. ``n_maps=1``
   keeps that map and reads it MULTI-HOT: four rotors, four bumps, the shared
   representation every `salience_rps` row uses. ``n_maps>1`` emits one map per
   rotor and reaches `LayerCRFReadout`, exactly as in the ports, so a later
   config can ask for per-rotor layers on this log grid without a code change.

WHAT THE GRID MEANS HERE. Under the project's ``f0 = rps`` convention a bin at
``f`` Hz is a candidate rate of ``f`` rev/s, so the paper's grid spans
27.5-4370 rev/s in 352 log-spaced bins. Rotors live in 0-150 rev/s, that is bins
0-118; the remaining 233 bins are the harmonics' home and carry no fundamental.
The bin spacing is 1.45% of the rate — 0.40 rev/s at 27.5, 0.58 at 40, 2.2 at
150 — against the ports' uniform 0.5017 rev/s. A stopped rotor has no bin at all
(the target builder marks a rotor active only above 0.1 Hz, so it becomes a dark
column), which is the same convention `multif0_salience` and
`basic_pitch_salience` already train under.

THE PUBLISHED INTERPOLATION IS SWAPPED, AND IT IS REPRODUCED. The upstream
weights are ``w_floor = x - floor(x)`` and ``w_ceil = ceil(x) - x``, applied to
the floor and ceiling bins RESPECTIVELY — the two are the wrong way round for
linear interpolation, which needs ``1 - (x - floor(x))`` on the floor bin. The
effect is a deterministic sawtooth warp of the frequency axis, at most one FFT
bin (15.6 Hz) wide, identical at training and at test, and the released
checkpoints were trained through it. Thus it is part of the published
model and is kept. ``published_interp=False`` swaps it back, for the follow-up
that asks whether it costs anything at 16 kHz.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_ports.layer_readout import LayerCRFReadout
from models.multif0.utils import cqt_freq_grid
from models.salience_rps import SalienceRPSPredictor

__all__ = ["HarmoF0Orig", "MRDConv", "WaveformToLogSpecgram", "dila_conv_block"]


class MRDConv(nn.Module):
    """HarmoF0's Multiple Rate Dilated Convolution, verbatim.

    ``(B, C_in, T, F) -> (B, C_out, T, F)``. Harmonic ``k`` owns a 1x1
    convolution and a read offset ``d_k = round(log2(k) * bins_per_octave)``:
    its output at bin ``g`` is taken from bin ``g + d_k``, which on a log axis
    is the frequency ``k`` times the frequency at ``g``. The offsets run off the
    top of the axis, so the last ``d_k`` bins of branch ``k`` are zero.

    The upstream implementation writes this as an in-place accumulation into a
    slice (``y[:, :, :, :n_freq] += x`` after ``x = x[:, :, :, d:]``). The
    out-of-place ``pad`` here is the same arithmetic — padding ``d`` zeros on
    the right of a slice of width ``F - d`` and adding the whole width — and it
    keeps autograd off a mutated convolution output.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation_list: list[int]):
        super().__init__()
        self.dilation_list = [int(d) for d in dilation_list]
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) for _ in self.dilation_list]
        )

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        y = self.conv_list[0](specgram)
        d0 = self.dilation_list[0]
        if d0:  # the upstream pads then crops; for d0 == 0 both are no-ops
            y = F.pad(y, [0, d0])[:, :, :, d0:]
        for conv, d in zip(self.conv_list[1:], self.dilation_list[1:], strict=True):
            x = conv(specgram)[:, :, :, d:]
            y = y + F.pad(x, [0, y.shape[-1] - x.shape[-1]])
        return y


def harmonic_dilation_list(n_har: int, bins_per_octave: int) -> list[int]:
    """``round(log2(k) * bins_per_octave)`` for ``k = 1 .. n_har`` — the offsets.

    HarmoF0 writes this as ``log(k) / log(2 ** (1 / B))``; the two are the same
    number. At ``B = 48`` and ``n_har = 12`` it is
    ``[0, 48, 76, 96, 111, 124, 135, 144, 152, 159, 166, 172]``.
    """
    a = np.log(np.arange(1, int(n_har) + 1)) / np.log(2.0 ** (1.0 / int(bins_per_octave)))
    return [int(v) for v in np.round(a).astype(int)]


def dila_conv_block(
    in_channel: int,
    out_channel: int,
    bins_per_octave: int,
    n_har: int,
    dilation_mode: str,
    dilation_rate: int,
    dil_kernel_size: tuple[int, int] = (1, 3),
    kernel_size: tuple[int, int] = (3, 3),
    padding: tuple[int, int] = (1, 1),
) -> nn.Sequential:
    """HarmoF0's ``dila_conv_block``, the ``log_scale`` and ``fixed`` branches.

    ``conv -> ReLU -> (MRDConv | dilated conv) -> ReLU -> BatchNorm``. The
    ``fixed_causal`` branch of the upstream file is not reachable from
    HarmoF0's own defaults and is not ported.
    """
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channel)
    if dilation_mode == "log_scale":
        second: nn.Module = MRDConv(
            out_channel, out_channel, harmonic_dilation_list(n_har, bins_per_octave)
        )
    elif dilation_mode == "fixed":
        second = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=dil_kernel_size,
            padding=(0, int(dilation_rate)),
            dilation=(1, int(dilation_rate)),
        )
    else:
        raise ValueError(f"unknown dilation mode {dilation_mode!r}")
    return nn.Sequential(conv, nn.ReLU(), second, nn.ReLU(), batch_norm)


class WaveformToLogSpecgram(nn.Module):
    """HarmoF0's log-frequency front end, ``(B, n)`` -> ``(B, T, freq_bins)`` dB.

    The upstream module receives pre-framed audio and calls `torch.fft.fft` on
    each frame; `torch.stft` computes the same windowed rfft, so the framing
    (deviation 1) is the only change. ``AmplitudeToDB(top_db=80)`` is applied to
    the POWER spectrum, matching the upstream call order.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        fmin: float,
        bins_per_octave: int,
        freq_bins: int,
        hop_length: int,
        published_interp: bool = True,
    ):
        super().__init__()
        import torchaudio

        self.n_fft, self.hop_length = int(n_fft), int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

        fre_resolution = float(sample_rate) / float(n_fft)
        idxs = torch.arange(0, int(freq_bins), dtype=torch.float64)
        log_idxs = float(fmin) * (2.0 ** (idxs / float(bins_per_octave))) / fre_resolution
        if float(log_idxs[-1]) > n_fft // 2:
            raise ValueError(
                f"the top log bin lands at FFT index {float(log_idxs[-1]):.1f}, past "
                f"Nyquist index {n_fft // 2} — raise n_freq or lower freq_bins"
            )
        floor = torch.floor(log_idxs).long()
        ceil = torch.ceil(log_idxs).long()
        w_floor = (log_idxs - floor).reshape(1, 1, -1)
        w_ceil = (ceil - log_idxs).reshape(1, 1, -1)
        if not published_interp:  # true linear interpolation; see the module docstring
            w_floor, w_ceil = w_ceil, w_floor
        self.register_buffer("log_idxs_floor", floor, persistent=False)
        self.register_buffer("log_idxs_ceiling", ceil, persistent=False)
        self.register_buffer("log_idxs_floor_w", w_floor, persistent=False)
        self.register_buffer("log_idxs_ceiling_w", w_ceil, persistent=False)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        spec = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=cast(torch.Tensor, self.window).to(waveforms.dtype),
            center=True,
            return_complex=True,
        )  # (B, n_fft // 2 + 1, T)
        power = (spec.real.pow(2) + spec.imag.pow(2)).transpose(1, 2)  # (B, T, F)
        dt = power.dtype
        specgram = power[:, :, cast(torch.Tensor, self.log_idxs_floor)] * cast(
            torch.Tensor, self.log_idxs_floor_w
        ).to(dt) + power[:, :, cast(torch.Tensor, self.log_idxs_ceiling)] * cast(
            torch.Tensor, self.log_idxs_ceiling_w
        ).to(dt)
        return self.amplitude_to_db(specgram)


class HarmoF0Orig(LayerCRFReadout, SalienceRPSPredictor):
    """Audio -> rotor-rate salience logits ``(B, n_maps * 352, T)`` on the LOG grid.

    The `salience_rps` contract: ``forward(audio) -> (B, F, T)`` logits,
    ``outputs_salience = True``, BCE against the RPS-derived target, Hungarian
    tracking at eval. The output axis is the paper's own log-spaced grid, so
    ``output_freqs()`` is ``cqt_freq_grid(fmin=27.5, n_bins=352,
    bins_per_octave=48)`` and `conf/loss/salience_bce_orig.yaml` builds the same
    array from the same three numbers.

    Args:
        n_fft, hop_length: the frame grid `predict_rps` resamples its output
            onto. ``hop_length`` is ALSO the front end's hop, so the salience
            time grid and the target time grid are one grid; ``n_fft`` is
            recorded only, and is kept equal to the front end's ``2 * n_freq``.
        num_rotors: rotors to track at eval.
        sr: sample rate the log-bin table is computed for.
        n_freq: HarmoF0's ``n_freq``; the front end's ``n_fft`` is twice it.
        fmin, bins_per_octave, freq_bins: the log grid. The paper's 27.5 / 48 /
            352 spans 88 semitones from A0.
        n_har: harmonics `MRDConv` reads (the paper's 12).
        channels, dilation_rates: the four blocks' widths and their octave-sized
            frequency dilations (the paper's ``(32, 64, 128, 128)`` and 48).
        published_interp: keep the upstream's swapped interpolation weights.
        n_maps: salience maps emitted. 1 is the shared multi-hot map; > 1 stacks
            per-rotor layers along the output axis for `LayerCRFReadout`.

    CAVEAT ON ``n_maps > 1``. `LayerCRFReadout` sizes its transition band from
    ``np.median(np.diff(out_freqs))``, which on a LOG grid is one number for a
    spacing that runs from 0.40 to 62 Hz. The band then admits far more than
    25 rev/s at the bottom of the axis and far less at the top. The mixin is
    kept so a later config can ask for layers here without a code change, but a
    row that does must size that band itself.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 512,
        num_rotors: int = 4,
        sr: int = 16000,
        n_freq: int = 512,
        fmin: float = 27.5,
        bins_per_octave: int = 48,
        freq_bins: int = 352,
        n_har: int = 12,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        dilation_rates: tuple[int, ...] = (48, 48, 48, 48),
        published_interp: bool = True,
        n_maps: int = 1,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        if len(channels) != 4 or len(dilation_rates) != 4:
            raise ValueError("HarmoF0 has four blocks; give four channels and four dilations")
        self.sr, self.n_maps = int(sr), int(n_maps)

        # Grid descriptor. `over_sample` is bins per SEMITONE, `n_octaves` the
        # (non-integer) span; `n_bins`/`bins_per_octave` are what size
        # the grid, so a non-multiple-of-12 `bins_per_octave` stays describable.
        self.fmin = float(fmin)
        self.bins_per_octave = int(bins_per_octave)
        self.over_sample = int(bins_per_octave) // 12
        self.n_bins = int(freq_bins)
        self.n_octaves = float(freq_bins) / float(bins_per_octave)
        # Also published as an explicit array so `LayerCRFReadout.grid_step()`
        # and the tracker have one grid to read; it equals `grid_params()`'s.
        self.out_freqs = cqt_freq_grid(
            fmin=self.fmin, n_bins=self.n_bins, bins_per_octave=self.bins_per_octave
        )
        self.spec_sr = int(sr)
        self.spec_hop = int(hop_length)

        self.waveform_to_logspecgram = WaveformToLogSpecgram(
            int(sr),
            int(n_freq) * 2,
            float(fmin),
            int(bins_per_octave),
            int(freq_bins),
            int(hop_length),
            published_interp=published_interp,
        )

        bins = int(bins_per_octave)
        self.block_1 = dila_conv_block(
            1,
            channels[0],
            bins,
            n_har=int(n_har),
            dilation_mode="log_scale",
            dilation_rate=int(dilation_rates[0]),
        )
        bins = bins // 2  # the paper halves it here; the `fixed` branch ignores it
        self.block_2 = dila_conv_block(
            channels[0], channels[1], bins, 3, "fixed", int(dilation_rates[1])
        )
        self.block_3 = dila_conv_block(
            channels[1], channels[2], bins, 3, "fixed", int(dilation_rates[2])
        )
        self.block_4 = dila_conv_block(
            channels[2], channels[3], bins, 3, "fixed", int(dilation_rates[3])
        )
        self.conv_5 = nn.Conv2d(channels[3], channels[3] // 2, kernel_size=(1, 1))
        self.conv_6 = nn.Conv2d(channels[3] // 2, self.n_maps, kernel_size=(1, 1))

    # ── grid ────────────────────────────────────────────────────────────────

    def num_grid_frames(self, n_samples: int) -> int:
        return int(n_samples) // self.spec_hop + 1

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        specgram = self.waveform_to_logspecgram(audio).float()  # (B, T, F)
        x = specgram.unsqueeze(1)  # (B, 1, T, F) — deviation 3
        x = self.block_4(self.block_3(self.block_2(self.block_1(x))))
        x = self.conv_6(torch.relu(self.conv_5(x)))  # (B, n_maps, T, F)
        x = x.transpose(2, 3)  # (B, n_maps, F, T)
        b, m, f, t = x.shape
        return x.reshape(b, m * f, t)
