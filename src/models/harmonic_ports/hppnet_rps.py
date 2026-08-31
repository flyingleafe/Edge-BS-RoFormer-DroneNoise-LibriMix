"""HPPNet (Wei et al., ISMIR 2022) as a rotor-rate salience model.

WHAT IS KEPT AND WHAT IS REPLACED. HPPNet is a `CNNTrunk` over a log-frequency
spectrogram followed by a `FreqGroupLSTM` per output head. Exactly one module
in the trunk knows about harmonics: `HarmonicDilatedConv`, eight
`Conv2d(kernel=[1,3], dilation=[1,d])` branches with ``d`` in
``48, 76, 96, 111, 124, 135, 144, 152``. Those are ``round(log2(k) * 48)`` for
``k = 2..9`` — harmonic offsets on a 48-bin-per-octave log axis, where a
harmonic is a fixed SHIFT and a dilated convolution can therefore reach it.

This port replaces that one module:

    HarmonicDilatedConv  ->  CombGather at ``k * r`` on the LINEAR STFT,
                             then one ``[1, 3]`` convolution over the harmonic
                             channels — which is the same sum of eight
                             per-harmonic ``[1, 3]`` convolutions, with the
                             dilated shift replaced by an explicit read.

`docs/harmonic-ports-design.md` holds the measurements that reject the log axis
here, and they are not re-derived: for two rotors ``D`` rev/s apart the
separation-to-bandwidth ratio on a log grid is ``D / (r * (2^(1/B) - 1))``, in
which the harmonic index CANCELS, so a rotor pair is resolved at every harmonic
or at none. A uniform STFT instead improves linearly with ``k``, and this task's
discriminating evidence lives at high ``k`` (DREGON's tightest cruise pair,
0.13 rev/s, separates from ``k >= 31``).

Downstream of the gather the axis is CANDIDATE RATE, not frequency.
`FreqGroupLSTM` is kept unchanged, now grouping over candidate rates rather
than pitches: its job is a recurrence shared across the output axis, which does
not care what that axis means.

DELIBERATE DEVIATIONS FROM THE PAPER
------------------------------------

1. HOP. HPPNet natively runs at 16 kHz with ``hop = 320`` (20 ms). This port
   uses ``hop = 512`` at 16 kHz, which is this project's frame grid — every
   dataset, loss and metric in the RPS campaign is on it, and putting the
   salience on a different grid would make the row incomparable with the
   baselines it exists to be read against. ``n_fft`` is 4096 rather than 2048:
   3.906 Hz per bin over a 0.256 s window, which is the resolution the design
   note's separability table is computed for.

2. FRONT END. `WaveformToLogSpecgram` — a uniform STFT LINEARLY INTERPOLATED
   onto log-spaced bins, then `AmplitudeToDB` — is dropped entirely. The input
   is the STFT power spectrogram flattened by the same running-median floor and
   ``log1p`` the classical scan uses, so an untrained channel of the harmonic
   block is the Whittle comb score of ``tracking.comb_seed``.

3. BLOCK ORDER. HPPNet's `block_1`/`block_2`/`block_2_5` (three ``7x7``
   convolutions, 1 -> 16 -> 16 -> 16 channels) smooth the log-spectrogram
   BEFORE the harmonic convolution. They cannot stay there: the gather consumes
   the raw periodogram, and gathering a 16-channel map at 32 harmonics of 300
   rates would cost 16x the activation memory of gathering one. The three
   blocks are kept verbatim — same count, same kernel, same widths — and moved
   to AFTER the gather, where they smooth in ``(rate, time)`` instead of
   ``(pitch, time)``. `conv_3` keeps its position and its ``16 -> 128``
   widening; only its eight dilated branches collapse to one, because the
   gather has already brought the harmonics into alignment.

4. DILATIONS. `block_4` and `block_5` use ``dilation = [1, 48]`` and
   ``[1, 12]``, i.e. one octave and one semitone-group on a 48-bin/octave axis.
   On a linear RATE axis an octave is not a fixed offset, so an octave-sized
   dilation has no meaning. These use plain dilated context convolutions along
   rate. The defaults (8 and 2 bins on a 0.5 rev/s grid) give a receptive field
   of roughly +-15 rev/s, the span of one airframe's four rotors — the context
   that matters here is "what else is on this airframe", not "the octave above".

5. NO POOLING. HPPNet's `block_4` carries ``MaxPool2d([1, 4])``, which takes
   the 4x-oversampled log axis (352 bins) down to one bin per pitch (88). Here
   the rate axis IS the output grid already (300 bins at 0.5 rev/s), so there is
   nothing to decimate and the pool is removed.

6. HEADS. HPPNet emits four heads — onset, frame, offset and velocity — because
   its task is piano transcription. Only the FRAME (multi-pitch estimation)
   head is ported. Onset/offset are note-boundary events, which a rotor
   trajectory does not have, and velocity is MIDI loudness. The dropped heads
   are the `onset_subnet`/`velocity_subnet` stacks, not part of the trunk.

7. LOGITS. `FreqGroupLSTM` ends in a sigmoid. The framework's
   `losses.SalienceRPSBCELoss` takes LOGITS (``BCEWithLogits``), so the sigmoid
   moves out of the model and into the loss. This is the same model:
   ``BCEWithLogitsLoss(x) == BCELoss(sigmoid(x))`` exactly.

8. MULTI-PITCH. HPPNet's frame head is already polyphonic, one bin per pitch.
   Here the target is MULTI-HOT — four rotors, four bumps on one rate map —
   which needs no framework change. Per-rotor maps (``n_maps > 1``) are a
   further step needing a permutation-invariant loss that does not exist yet;
   the head can emit them, nothing trains them.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from models.comb_salience import CombGather, local_floor_torch
from models.multif0.utils import linear_freq_grid
from models.salience_rps import SalienceRPSPredictor

__all__ = ["HPPNetRPS", "BiLSTM", "FreqGroupLSTM", "HarmonicCombConv", "CombCNNTrunk"]


class BiLSTM(nn.Module):
    """HPPNet's ``hppnet/lstm.py`` BiLSTM: ``(N, T, C_in) -> (N, T, 2*H)``."""

    def __init__(self, input_features: int, recurrent_features: int):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


class FreqGroupLSTM(nn.Module):
    """HPPNet's `FreqGroupLSTM`, unchanged apart from the terminal sigmoid.

    ``[b, c_in, T, n_out] -> [b, c_out, T, n_out]``. Every position on the
    output axis is folded into the batch, so ONE recurrence is shared across
    all of them and the model has no per-position temporal parameters. That is
    the property the module exists for, and it does not depend on the axis
    being pitch: here it is candidate rate.

    ``sigmoid=False`` returns logits (see deviation 7 in the module docstring).
    """

    def __init__(
        self, channel_in: int, channel_out: int, lstm_size: int, sigmoid: bool = False
    ) -> None:
        super().__init__()
        self.channel_out = int(channel_out)
        self.sigmoid = bool(sigmoid)
        self.lstm = BiLSTM(channel_in, lstm_size // 2)
        self.linear = nn.Linear(lstm_size, channel_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c_in, t, n_freq = x.size()
        x = torch.permute(x, [0, 3, 2, 1])  # [b, n_freq, T, c_in]
        x = x.reshape([b * n_freq, t, c_in])
        x = self.lstm(x)  # [(b*n_freq), T, lstm_size]
        x = self.linear(x)  # [(b*n_freq), T, c_out]
        x = x.reshape([b, n_freq, t, self.channel_out])
        x = torch.permute(x, [0, 3, 2, 1])  # [b, c_out, T, n_freq]
        return torch.sigmoid(x) if self.sigmoid else x


class HarmonicCombConv(nn.Module):
    """THE SUBSTITUTION: `HarmonicDilatedConv` with the shifts made explicit.

    ``(B, K, G, T)`` harmonic readings -> ``(B, C, G, T)``.

    HPPNet's module is eight ``Conv2d(c_in, c_out, [1, 3], dilation=[1, d_k])``
    branches summed together. Each branch reads the map at a fixed offset
    ``d_k`` (the log-axis position of harmonic ``k``) with a 3-tap kernel along
    frequency. Once `CombGather` has read the spectrum at ``k * r`` for every
    ``(k, r)`` pair, harmonic ``k`` IS channel ``k`` and the offsets are gone:
    a single ``Conv2d(K, C, [1, 3])`` over those channels computes exactly the
    same sum of eight per-harmonic 3-tap convolutions. The kernel walks the
    RATE axis, which is what the offsets were proportional to.

    Channel 0's center tap is initialized to 1 over every harmonic and the rest
    to zero, so that channel starts as the mean of ``log1p(power / floor)`` —
    the Whittle comb score of ``tracking.comb_seed``, since the caller has
    already divided by the in-band harmonic count. Training can leave the
    classical score only by improving on it.
    """

    def __init__(self, k_max: int, out_channels: int, kernel: int = 3):
        super().__init__()
        self.k_max, self.out_channels = int(k_max), int(out_channels)
        pad = int(kernel) // 2
        # (B, K, G, T): kernel (kernel, 1) walks G (rate), not T. HPPNet's
        # [1, 3] walks frequency in its (B, C, T, freq) layout — the same axis.
        self.conv = nn.Conv2d(
            self.k_max, self.out_channels, kernel_size=(int(kernel), 1), padding=(pad, 0)
        )
        with torch.no_grad():
            self.conv.weight.zero_()
            cast(torch.Tensor, self.conv.bias).zero_()
            self.conv.weight[0, :, pad, 0] = 1.0
            self.conv.weight[1:].normal_(0.0, (1.0 / self.k_max) ** 0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(z))


def _conv_block(
    c_in: int,
    c_out: int,
    kernel: int | tuple[int, int] = (7, 7),
    dilation: tuple[int, int] = (1, 1),
) -> nn.Sequential:
    """HPPNet's ``CNNTrunk.get_conv2d_block``: conv -> ReLU -> InstanceNorm.

    ``padding='same'`` and `InstanceNorm2d` are the paper's, kept verbatim. The
    pooling branch is not ported (deviation 5).
    """
    k = (int(kernel), int(kernel)) if isinstance(kernel, int) else kernel
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=k, padding="same", dilation=dilation),
        nn.ReLU(),
        nn.InstanceNorm2d(c_out),
    )


class CombCNNTrunk(nn.Module):
    """HPPNet's `CNNTrunk` on the ``(B, C, rate, time)`` plane.

    Block map, HPPNet -> port (see the module docstring for the reasons):

    ==================  =========================================================
    ``conv_3``          `HarmonicCombConv` — the gather plus one ``[1,3]`` conv
    ``block_1/2/2_5``   unchanged ``7x7``, 16 channels, MOVED after the gather
    ``block_4``         ``[1,3]`` at ``dilation[0]`` along rate, POOL REMOVED
    ``block_5``         ``[1,3]`` at ``dilation[1]`` along rate
    ``block_6/7/8``     ``[5,1]`` temporal, unchanged
    ==================  =========================================================
    """

    def __init__(
        self,
        k_max: int,
        c_har: int = 16,
        embedding: int = 128,
        dilations: tuple[int, int] = (8, 2),
    ):
        super().__init__()
        self.conv_3 = HarmonicCombConv(k_max, c_har)
        self.block_1 = _conv_block(c_har, c_har, kernel=7)
        self.block_2 = _conv_block(c_har, c_har, kernel=7)
        self.block_2_5 = _conv_block(c_har, c_har, kernel=7)
        self.widen = _conv_block(c_har, embedding, kernel=(3, 1))
        self.block_4 = _conv_block(
            embedding, embedding, kernel=(3, 1), dilation=(int(dilations[0]), 1)
        )
        self.block_5 = _conv_block(
            embedding, embedding, kernel=(3, 1), dilation=(int(dilations[1]), 1)
        )
        self.block_6 = _conv_block(embedding, embedding, kernel=(1, 5))
        self.block_7 = _conv_block(embedding, embedding, kernel=(1, 5))
        self.block_8 = _conv_block(embedding, embedding, kernel=(1, 5))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, K, G, T)`` evidence -> ``(B, embedding, G, T)``."""
        x = self.conv_3(z)
        x = self.block_2_5(self.block_2(self.block_1(x)))
        x = self.widen(x)
        x = self.block_5(self.block_4(x))
        return self.block_8(self.block_7(self.block_6(x)))


class HPPNetRPS(SalienceRPSPredictor):
    """Audio -> rotor-rate salience logits ``(B, G, T)`` on a LINEAR rate grid.

    The ``salience_rps`` contract: ``forward(audio) -> (B, F, T)`` logits,
    ``outputs_salience = True``, BCE against the RPS-derived target, Hungarian
    tracking at eval. The output grid is declared through ``out_freqs``, the
    hook `SalienceRPSPredictor` already has for a salience axis decoupled from
    a log-spaced input CQT — here there is no CQT at all and the axis is
    candidate rate in rev/s, but the plumbing (target construction, tracker,
    ``predict_rps``) is identical and is reused unchanged.

    Args:
        n_fft, hop_length, sr: the linear STFT. 4096 at 16 kHz is 3.906 Hz per
            bin over a 0.256 s window. ``hop_length`` 512 is this project's
            frame grid, not HPPNet's 320 (deviation 1).
        r_lo, r_hi, n_grid: the candidate-rate grid, rev/s. 0-150 in 300 bins
            is 0.5 rev/s per bin; the design note's error table puts the
            resulting discretization at 0.013 rev/s at 40 dB and 0.13 at 20 dB,
            under the campaign's 0.2 rev/s honest floor.
        k_max, f_max: harmonics per hypothesis, and the frequency past which a
            harmonic is dropped from the gather and from its count.
        c_har, embedding: HPPNet's own trunk widths.
        dilations: rate-axis dilations of blocks 4 and 5 (deviation 4).
        lstm_size: `FreqGroupLSTM`'s recurrent width. Its cost is
            ``B * G`` sequences of length ``T``, and ``G`` is 300 here against
            HPPNet's 88 pitches, so this is the model's memory knob.
        floor_hz: width of the running-median floor along frequency.
        n_maps: salience maps emitted (deviation 8).
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
        c_har: int = 16,
        embedding: int = 128,
        dilations: tuple[int, int] = (8, 2),
        lstm_size: int = 64,
        floor_hz: float = 120.0,
        n_maps: int = 1,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        self.sr, self.k_max, self.n_maps = int(sr), int(k_max), int(n_maps)

        # The output axis IS the candidate-rate grid, so one array serves the
        # gather, the BCE target and the tracker. Built with `linear_freq_grid`
        # so it is bit-identical to the grid `losses.SalienceRPSBCELoss` builds
        # from the same (out_fmin, out_fmax, out_bins) in the loss config.
        grid = linear_freq_grid(r_lo, r_hi, n_grid)
        self.out_freqs = grid
        self.n_bins = int(n_grid)

        # torch.stft(center=True) emits n // hop + 1 frames, and nothing in the
        # trunk pools along time, so the salience rate is the STFT rate.
        self.spec_sr = int(sr)
        self.spec_hop = int(hop_length)

        df = float(sr) / float(n_fft)
        self.floor_bins = max(3, int(round(floor_hz / df)) | 1)
        self.gather = CombGather(
            k_max=self.k_max, sr=int(sr), n_fft=int(n_fft), f_max=float(f_max), grid=grid
        )
        self.register_buffer("window", torch.hann_window(int(n_fft)), persistent=False)

        self.trunk = CombCNNTrunk(self.k_max, c_har=c_har, embedding=embedding, dilations=dilations)
        self.head = FreqGroupLSTM(embedding, self.n_maps, lstm_size, sigmoid=False)

    # ── grid ────────────────────────────────────────────────────────────────

    def grid_params(self) -> dict:
        raise NotImplementedError(
            "HPPNetRPS has no log-spaced CQT grid; its salience axis is the "
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
        from the two, which is what `CombScoreHead` does — interpolating in the
        log domain instead would break the correspondence with the classical
        scan. The floor is detached: it sets the scale, it is not a parameter.
        Harmonics past ``f_max`` come back as exactly zero, and dividing by the
        in-band count keeps a rate whose high harmonics fall out of band from
        being scored lower for that reason alone.
        """
        floor = local_floor_torch(pw, self.floor_bins).detach()
        h = self.gather(pw)
        fh = self.gather(floor).clamp_min(1e-12)
        z = torch.log1p(h / fh)
        count = cast(torch.Tensor, self.gather.count)
        return z / count.to(z.dtype)[None, None, :, None]

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.trunk(self.evidence(self.spectrum(audio)))  # (B, C, G, T)
        # `FreqGroupLSTM` reads HPPNet's [b, c, T, n_out] layout.
        x = self.head(x.transpose(2, 3))  # (B, n_maps, T, G)
        x = x.transpose(2, 3)  # (B, n_maps, G, T)
        b, m, g, t = x.shape
        return x.reshape(b, m * g, t)
