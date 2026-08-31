"""HarmoF0 (Wei et al., ISMIR 2022) as a rotor-rate salience model.

WHAT IS KEPT AND WHAT IS REPLACED. HarmoF0 is four convolution blocks over a
log-frequency spectrogram, then two 1x1 convolutions to one salience map. The
first block is the only one that knows about harmonics: ``MRDConv`` applies a
1x1 convolution to the map, SHIFTS it by ``round(log2(k) * bins_per_octave)``
bins for each harmonic ``k``, and sums. On a log axis that shift IS the gather
at ``k * f``, which is why the architecture works.

This port keeps every block and replaces exactly one thing:

    MRDConv  ->  CombGather at ``k * r`` on the LINEAR STFT, times a learned
                 per-harmonic weight.

The 1x1 convolutions of ``MRDConv`` are that weight and nothing else — one
learned mixing matrix indexed by harmonic order — so the substitution changes
the READ POSITIONS, not the parameterization. `docs/harmonic-ports-design.md`
holds the measurements that reject the log axis here: a log grid's
separation-to-bandwidth ratio for two rotors ``D`` apart is
``D / (r * (2^(1/B) - 1))``, in which the harmonic index CANCELS, so a rotor
pair is resolved at every harmonic or at none; a uniform STFT instead improves
linearly with ``k``, which is where this task's evidence lives.

The axis downstream of the gather is therefore the CANDIDATE RATE, not
frequency. Blocks 2-4 keep their shape.

DELIBERATE DEVIATIONS FROM THE PAPER, all forced by that axis change:

1. DILATIONS. HarmoF0's blocks 2-4 use ``dilation = bins_per_octave`` (48),
   i.e. "look one octave up". On a linear RATE axis an octave is not a fixed
   offset, so an octave-sized dilation has no meaning. These blocks use plain
   dilated context convolutions along rate. The defaults (4, 8, 16 bins on a
   0.5 rev/s grid) give a receptive field of about +-15 rev/s, which is the
   span of one airframe's four rotors — the context that matters here is
   "what else on this airframe", not "the octave above".
2. FRONT END. ``WaveformToLogSpecgram`` (a uniform STFT linearly interpolated
   onto log-spaced bins) is dropped entirely. The input is the STFT power
   spectrogram, flattened by the same running-median floor and ``log1p`` the
   classical scan uses, so an untrained channel of the harmonic block is the
   Whittle comb score.
3. BLOCK 1's PRE-CONVOLUTION. In the paper block 1 is
   ``conv 3x3 -> ReLU -> MRDConv -> ReLU -> BatchNorm``, so ``MRDConv`` mixes
   32 channels and carries a 32x32 weight per harmonic. Here the gather runs on
   the ONE evidence channel and the pre-convolution is dropped, because
   gathering C channels costs ``C * K * G * T`` activations — at 32 channels
   that is 300 MB per second of audio, against 10 MB at one. The harmonic
   weight is therefore ``(C, K)``: one learned scalar per (output channel,
   harmonic), which is what ``MRDConv`` reduces to at one input channel.
4. OUTPUT. HarmoF0 emits a sigmoid; this emits LOGITS, which is what the
   ``salience_rps`` task and ``losses.SalienceRPSBCELoss`` expect.
5. MULTI-PITCH. HarmoF0 is monophonic (one map, one peak). Here the target is
   MULTI-HOT — four rotors, four bumps on one map — which needs no framework
   change. Per-rotor maps (``n_maps > 1``) are a further step that needs a
   permutation-invariant loss; the head can emit them, nothing trains them yet.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from models.comb_salience import CombGather, local_floor_torch
from models.multif0.utils import linear_freq_grid
from models.salience_rps import SalienceRPSPredictor

__all__ = ["HarmoF0RPS", "RateHarmonicBlock", "RateContextBlock"]


class RateHarmonicBlock(nn.Module):
    """The ``MRDConv`` substitution: gather at ``k * r``, weight, sum.

    ``(B, K, G, T)`` harmonic readings -> ``(B, C, G, T)``. The mixing matrix
    ``W`` of shape ``(C, K)`` is exactly what ``MRDConv``'s per-harmonic 1x1
    convolutions are, with one input channel: a learned weight per harmonic
    order, shared across every candidate rate. Weight sharing across the
    hypothesis is the property the log axis was bought for, and the gather
    gives it directly, because the read offsets are proportional to the rate.

    Channel 0's weights are initialized to ``1/K`` and its bias to zero, so
    that channel STARTS as the mean over harmonics of ``log1p(power/floor)``
    — the classical Whittle comb score of ``tracking.comb_seed``. The rest are
    random. Training can only leave the classical score by improving on it.
    """

    def __init__(self, k_max: int, out_channels: int):
        super().__init__()
        self.k_max, self.out_channels = int(k_max), int(out_channels)
        w = torch.randn(self.out_channels, self.k_max) / float(self.k_max) ** 0.5
        w[0] = 1.0 / float(self.k_max)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # (C, K) x (B, K, G, T) -> (B, C, G, T)
        return (
            torch.einsum("ck,bkgt->bcgt", self.weight.to(z.dtype), z)
            + self.bias.to(z.dtype)[None, :, None, None]
        )


class RateContextBlock(nn.Module):
    """HarmoF0's blocks 2-4, with the dilation running along RATE.

    ``conv 3x3 -> ReLU -> dilated conv (3, 1) -> ReLU -> BatchNorm``, verbatim
    from ``dila_conv_block(dilation_mode="fixed")`` apart from which axis the
    dilation walks. In HarmoF0's ``(B, C, T, freq)`` layout the dilated kernel
    is ``[1, 3]`` with ``dilation=[1, 48]``, i.e. along frequency only; here
    the layout is ``(B, C, rate, time)`` and the kernel is ``(3, 1)`` with
    ``dilation=(d, 1)``, i.e. along rate only. The 3x3 conv mixes rate and time
    exactly as it does in the paper.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        d = int(dilation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.dil = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 1), dilation=(d, 1), padding=(d, 0)
        )
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.act(self.dil(self.act(self.conv(x)))))


class HarmoF0RPS(SalienceRPSPredictor):
    """Audio -> rotor-rate salience logits ``(B, G, T)`` on a LINEAR rate grid.

    The ``salience_rps`` contract: ``forward(audio) -> (B, F, T)`` logits,
    ``outputs_salience = True``, BCE against the RPS-derived target, Hungarian
    tracking at eval. The output grid is declared through ``out_freqs``, which
    is the mechanism ``SalienceRPSPredictor`` already has for a salience axis
    decoupled from a log-spaced input CQT — here there is no CQT at all and the
    axis is candidate rate in rev/s, but the plumbing (target construction,
    tracker, ``predict_rps``) is identical and is reused unchanged.

    Args:
        n_fft, hop_length, sr: the linear STFT. 4096 at 16 kHz is 3.906 Hz per
            bin over a 0.256 s window, which is what the design note's
            separability table is computed for.
        r_lo, r_hi, n_grid: the candidate-rate grid, rev/s. 0-150 in 300 bins
            is 0.5 rev/s per bin; the note's error table puts the resulting
            discretization at 0.013 rev/s at 40 dB and 0.13 at 20 dB, under the
            campaign's 0.2 rev/s honest floor.
        k_max, f_max: harmonics per hypothesis, and the frequency past which a
            harmonic is dropped from the gather and from its count.
        f_min: harmonics BELOW this frequency are dropped too. The classical
            scan never needed this because it searches 30-100 rev/s; a grid
            that reaches 0 does, and the reason is measured: at 1.5 rev/s all
            32 harmonics land inside the window's DC mainlobe, every one of
            them reads far above a median floor computed from that same
            mainlobe, and the untrained score peaks there instead of at a real
            37 rev/s comb. Excluding the lowest bins removes the artifact and
            costs nothing real — a rotor whose whole comb sits under 30 Hz has
            no evidence at 3.9 Hz bin spacing anyway.
        channels: HarmoF0's own widths.
        dilations: rate-axis dilations of blocks 2-4 (see the module docstring
            — this is the deliberate deviation).
        floor_hz: width of the running-median floor along frequency.
        n_maps: salience maps emitted. 1 is the multi-hot multi-pitch model the
            framework trains today; > 1 stacks per-rotor maps along the output
            axis and needs a permutation-invariant loss that does not exist yet.
    """

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 512,
        num_rotors: int = 4,
        sr: int = 16000,
        r_lo: float = 0.0,
        r_hi: float = 150.0,
        n_grid: int = 300,
        k_max: int = 32,
        f_max: float = 7500.0,
        f_min: float = 30.0,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        dilations: tuple[int, ...] = (4, 8, 16),
        floor_hz: float = 120.0,
        n_maps: int = 1,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        if len(channels) != 4:
            raise ValueError("HarmoF0 has four blocks; `channels` must have four entries")
        if len(dilations) != 3:
            raise ValueError("blocks 2-4 take three dilations")
        self.sr, self.k_max, self.n_maps = int(sr), int(k_max), int(n_maps)

        # The output axis IS the candidate-rate grid, so one array serves the
        # gather, the BCE target and the tracker. `out_freqs` is the base
        # class's hook for a salience axis that is not a CQT grid.
        grid = linear_freq_grid(r_lo, r_hi, n_grid)
        self.out_freqs = grid
        self.n_bins = int(n_grid)

        # Time grid: torch.stft(center=True) emits n // hop + 1 frames, and the
        # model does not pool along time, so the salience rate is the STFT rate.
        self.spec_sr = int(sr)
        self.spec_hop = int(hop_length)

        df = float(sr) / float(n_fft)
        self.floor_bins = max(3, int(round(floor_hz / df)) | 1)
        self.gather = CombGather(
            k_max=self.k_max, sr=int(sr), n_fft=int(n_fft), f_max=float(f_max), grid=grid
        )
        # The low-frequency exclusion, applied on top of CombGather's own
        # `f_max` mask (that module is shared and stays untouched).
        fk = (
            torch.arange(1, self.k_max + 1, dtype=torch.float64)[:, None]
            * torch.as_tensor(grid, dtype=torch.float64)[None, :]
        )
        band = (fk >= float(f_min)) & (fk < float(f_max))
        self.register_buffer("band", band.to(torch.float64), persistent=False)
        self.register_buffer("window", torch.hann_window(int(n_fft)), persistent=False)

        self.block_1 = nn.Sequential(
            RateHarmonicBlock(self.k_max, channels[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[0]),
        )
        self.block_2 = RateContextBlock(channels[0], channels[1], dilations[0])
        self.block_3 = RateContextBlock(channels[1], channels[2], dilations[1])
        self.block_4 = RateContextBlock(channels[2], channels[3], dilations[2])
        self.conv_5 = nn.Conv2d(channels[3], channels[3] // 2, kernel_size=(1, 1))
        self.conv_6 = nn.Conv2d(channels[3] // 2, self.n_maps, kernel_size=(1, 1))

    # ── grid ────────────────────────────────────────────────────────────────

    def grid_params(self) -> dict:
        raise NotImplementedError(
            "HarmoF0RPS has no log-spaced CQT grid; its salience axis is the "
            "linear candidate-rate grid exposed as `out_freqs`."
        )

    def output_freqs(self) -> np.ndarray:
        return np.asarray(self.out_freqs, dtype=np.float64)

    def num_grid_frames(self, n_samples: int) -> int:
        return int(n_samples) // self.spec_hop + 1

    # ── front end ───────────────────────────────────────────────────────────

    def spectrum(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio ``(B, T)`` or ``(B, 1, T)`` -> power spectrogram ``(B, F, T)``."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.spec_hop,
            window=cast(torch.Tensor, self.window).to(audio.dtype),
            center=True,
            return_complex=True,
        )
        return spec.real.pow(2) + spec.imag.pow(2)

    def evidence(self, pw: torch.Tensor) -> torch.Tensor:
        """Power ``(B, F, T)`` -> per-(harmonic, rate) evidence ``(B, K, G, T)``.

        Gathers the power and the floor SEPARATELY and forms ``log1p(h / floor)``
        from the two, which is what ``CombScoreHead`` does — interpolating in
        the log domain instead would break the correspondence with the
        classical scan. The floor is detached: it sets the scale, it is not a
        parameter. Harmonics past ``f_max`` come back as exactly zero.
        """
        floor = local_floor_torch(pw, self.floor_bins).detach()
        h = self.gather(pw)
        fh = self.gather(floor).clamp_min(1e-12)
        band = cast(torch.Tensor, self.band).to(h.dtype)[None, :, :, None]
        # Divide by the CONSTANT k_max, not by the per-rate count of in-band
        # harmonics. The classical scan divides by the count, which is right in
        # its own 30-100 rev/s search band where the count barely moves; on a
        # grid reaching 0 it is a trap, and the trap was measured. At 1.5 rev/s
        # only three harmonics clear f_min, so a count-normalized mean is the
        # mean of three readings, and one accidental hit on a real line makes
        # that candidate beat the true comb: with count normalization the
        # untrained score misses on 5 of 8 synthetic combs (37 rev/s decodes to
        # 1.51), with k_max normalization it misses on 1 of 8. A candidate
        # whose comb is mostly out of band should score LOW, and dividing by a
        # constant is what says so.
        return torch.log1p(h / fh) * band / float(self.k_max)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.block_1(self.evidence(self.spectrum(audio)))
        x = self.block_4(self.block_3(self.block_2(x)))
        x = self.conv_6(torch.relu(self.conv_5(x)))  # (B, n_maps, G, T)
        b, m, g, t = x.shape
        return x.reshape(b, m * g, t)
