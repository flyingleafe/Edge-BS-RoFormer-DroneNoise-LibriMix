"""HPPNet (Wei et al., ISMIR 2022) UNMODIFIED, as a rotor-rate salience model.

WHY THIS FILE EXISTS. `models.harmonic_ports.hppnet_rps` is HPPNet with its
harmonic organ replaced: `HarmonicDilatedConv`'s eight log-axis dilated branches
became an explicit gather at ``k * r`` on the linear STFT, the CQT became a
uniform STFT, and the 352-bin semitone grid became a 0-150 rev/s linear grid.
This module is the control for that claim — the published architecture, kept
whole, on the same `salience_rps` task, so the pair ``hb_sal_hppnet_orig`` /
``hppnet_*`` isolates what the substitutions buy.

WHAT IS KEPT. Everything the paper's frame path has:

* The CQT front end — nnAudio, ``fmin`` 27.5, 352 bins at 48 per octave, then
  `AmplitudeToDB(top_db=80)`, exactly the call in ``hppnet/transcriber.py``.
* `HarmonicDilatedConv` — eight ``Conv2d([1, 3], dilation=[1, d])`` branches at
  ``d = 48, 76, 96, 111, 124, 135, 144, 152``, summed, then ReLU. Those are
  ``round(log2(k) * 48)`` for ``k = 2..9``: harmonic offsets on the log axis.
* `CNNTrunk` — ``block_1/2/2_5`` (7x7, 1 -> 16 -> 16 -> 16), ``conv_3`` (the
  harmonic convolution, 16 -> 128), ``block_4`` (``[1,3]`` dilated by 48),
  ``block_5`` (``[1,3]`` dilated by 12), ``block_6/7/8`` (``[5,1]``, temporal).
  ``conv -> ReLU -> InstanceNorm2d`` throughout, `padding='same'`.
* `FreqGroupLSTM` — every frequency position folded into the batch, one shared
  `BiLSTM` along time, then a `Linear` to the output channels. ONE recurrence
  for the whole axis, which is the module's whole point.
* The layout ``(B, C, T, freq)``.

WHAT IS DROPPED. The onset, offset and velocity heads and the whole
``onset_subnet``: they are note-boundary and MIDI-loudness events, which a
rotor trajectory does not have. Only the FRAME (multi-pitch) head is ported,
which is what the task asks for and what the port already does.

DEVIATIONS, and what forces each:

1. NO FREQUENCY POOL (``freq_pool=1``). HPPNet's ``block_4`` carries
   ``MaxPool2d([1, 4])``, which takes the 4x-oversampled log axis (352 bins)
   down to one bin per SEMITONE (88), because its output is a piano roll. A
   rotor rate is not quantized to semitones, and 88 bins over 88 semitones is
   5.95% per bin — 2.4 rev/s at 40 rev/s, an order of magnitude past the
   campaign's 0.2 rev/s floor, which would make this arm a measurement of the
   pool and not of the log axis. Thus the pool is off and the grid stays at
   352 bins (1.45% per bin). ``freq_pool=4`` restores the published trunk, and
   ``output_freqs()`` follows it to the pooled bin centers.
2. NO TIME POOL (``time_pooling=False``). HPPNet's frame subnet runs
   ``max_pool2d([2, 1])`` on its input and `F.interpolate(mode='bilinear')` on
   its output, so the frame head works at HALF the frame rate. Bilinear
   upsampling a one-bin-wide salience bump makes every odd frame a blend of its
   neighbours, and on this project's 32 ms grid a rotor moves up to 23.8 rev/s
   per frame, so the blend is not a small error. ``time_pooling=True`` restores
   it.
3. WHAT THE POOL TOOK WITH IT. ``block_5`` is dilated by 12, which is one
   OCTAVE on the 12-bin-per-octave axis the pool produces. With the pool off
   it walks a quarter of an octave instead. The published NUMBER is kept,
   because one deviation at a time is what makes the arm readable;
   ``dilations`` ``[48, 48]`` restores the published MEANING, for the
   follow-up that asks which of the two the block was really using.
4. HOP. HPPNet runs at 16 kHz with ``hop = 320`` (20 ms). This uses ``hop =
   512``, this project's frame grid — every dataset, loss and metric in the
   campaign is on it. Same reason as the port's deviation 1.
5. CQT2010v2, NOT nnAudio's plain ``CQT``. ``nnAudio.Spectrogram.CQT`` builds
   ONE kernel for the whole range, and at ``fmin`` 27.5 with 48 bins per octave
   ``Q = 1 / (2 ** (1 / 48) - 1) = 68.8``, so the lowest kernel is 40000 samples
   and nnAudio rounds ``n_fft`` up to 65536. It then reflect-pads by 32768 and
   RAISES on any clip shorter than 4.1 s — including this project's 1 s training
   clips. `CQT2010v2` computes the same grid octave by octave with downsampling
   and runs on 1 s. The consequence is real and belongs in the read-out of this
   arm: a 48-bin-per-octave CQT at 27.5 Hz needs a 2.5 s analysis window at the
   bottom bin, which a 1 s clip does not have.
6. FMIN 27.5 EXACTLY. The transcriber passes ``27.5 / 2 ** (1 / 24)``, half a
   semitone low, so that each group of four bins is centred on a MIDI note
   before the ``[1, 4]`` pool. With the pool off there is no group to centre,
   and 27.5 puts this arm on bit-identical bins to `HarmoF0Orig`, so one
   `conf/loss/salience_bce_orig.yaml` serves both and the two are read on one
   axis.
7. LOGITS. `FreqGroupLSTM` ends in a sigmoid, and the framework's
   `losses.SalienceRPSBCELoss` takes logits, so the sigmoid moves into the loss.
   ``BCEWithLogitsLoss(x) == BCELoss(sigmoid(x))``.
8. MULTI-PITCH. HPPNet's frame head is already polyphonic, one bin per pitch;
   ``n_maps=1`` reads it multi-hot (four rotors, four bumps). ``n_maps>1`` emits
   one map per rotor and reaches `LayerCRFReadout`, as in the ports.

WHAT THE GRID MEANS HERE. Under ``f0 = rps`` the 352 bins span 27.5-4370 rev/s,
of which rotors occupy bins 0-118. See `harmof0_orig` for the full reading of
that axis; the two share it exactly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_ports.layer_readout import LayerCRFReadout
from models.salience_rps import SalienceRPSPredictor

__all__ = [
    "HPPNetOrig",
    "BiLSTM",
    "FreqGroupLSTM",
    "HarmonicDilatedConv",
    "CNNTrunk",
    "CQTLogSpecgram",
]

# ``round(log2(k) * 48)`` for k = 2..9 — HPPNet's eight branch dilations,
# hard-coded in ``hppnet/nets.py`` and reproduced here as data.
HPPNET_DILATIONS: tuple[int, ...] = (48, 76, 96, 111, 124, 135, 144, 152)


class BiLSTM(nn.Module):
    """HPPNet's ``hppnet/lstm.py`` BiLSTM: ``(N, T, C_in) -> (N, T, 2*H)``.

    The upstream module also chunks the sequence at inference to bound
    memory on full piano pieces. That branch is an inference-memory device and
    changes nothing about the model, so it is not ported.
    """

    def __init__(self, input_features: int, recurrent_features: int):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


class FreqGroupLSTM(nn.Module):
    """HPPNet's `FreqGroupLSTM`, verbatim apart from the terminal sigmoid.

    ``(B, C_in, T, F) -> (B, C_out, T, F)``. Every frequency position is folded
    into the batch, so ONE recurrence is shared across the whole axis and the
    model has no per-position temporal parameters.
    """

    def __init__(
        self, channel_in: int, channel_out: int, lstm_size: int, sigmoid: bool = False
    ) -> None:
        super().__init__()
        self.channel_out = int(channel_out)
        self.sigmoid = bool(sigmoid)
        self.lstm = BiLSTM(channel_in, int(lstm_size) // 2)
        self.linear = nn.Linear(int(lstm_size), int(channel_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c_in, t, n_freq = x.size()
        x = torch.permute(x, [0, 3, 2, 1]).reshape(b * n_freq, t, c_in)
        x = self.linear(self.lstm(x))
        x = x.reshape(b, n_freq, t, self.channel_out)
        x = torch.permute(x, [0, 3, 2, 1])
        return torch.sigmoid(x) if self.sigmoid else x


class HarmonicDilatedConv(nn.Module):
    """HPPNet's `HarmonicDilatedConv`, verbatim.

    Eight ``Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, d])``
    branches summed, then ReLU. Branch ``k`` reads the log axis at an offset of
    ``round(log2(k) * 48)`` bins, which is where the ``k``-th harmonic sits.
    """

    def __init__(self, c_in: int, c_out: int, dilations: tuple[int, ...] = HPPNET_DILATIONS):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(c_in, c_out, (1, 3), padding="same", dilation=(1, int(d)))
                for d in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convs[0](x)
        for conv in self.convs[1:]:
            y = y + conv(x)
        return torch.relu(y)


def _conv2d_block(
    channel_in: int,
    channel_out: int,
    kernel_size: int | tuple[int, int] = (1, 3),
    pool_size: tuple[int, int] | None = None,
    dilation: tuple[int, int] = (1, 1),
) -> nn.Sequential:
    """HPPNet's ``CNNTrunk.get_conv2d_block``: conv -> ReLU -> [pool] -> InstanceNorm."""
    conv = nn.Conv2d(
        channel_in, channel_out, kernel_size=kernel_size, padding="same", dilation=dilation
    )
    layers: list[nn.Module] = [conv, nn.ReLU()]
    if pool_size is not None:
        layers.append(nn.MaxPool2d(pool_size))
    layers.append(nn.InstanceNorm2d(channel_out))
    return nn.Sequential(*layers)


class CNNTrunk(nn.Module):
    """HPPNet's `CNNTrunk`, ``(B, 1, T, F) -> (B, embedding, T, F // freq_pool)``.

    ``freq_pool`` is the ``MaxPool2d([1, n])`` of ``block_4``; 1 removes it
    (deviation 1), 4 is the published value.
    """

    def __init__(
        self,
        c_in: int = 1,
        c_har: int = 16,
        embedding: int = 128,
        freq_pool: int = 1,
        dilations: tuple[int, int] = (48, 12),
        harmonic_dilations: tuple[int, ...] = HPPNET_DILATIONS,
    ):
        super().__init__()
        self.freq_pool = int(freq_pool)
        self.block_1 = _conv2d_block(c_in, c_har, kernel_size=7)
        self.block_2 = _conv2d_block(c_har, c_har, kernel_size=7)
        self.block_2_5 = _conv2d_block(c_har, c_har, kernel_size=7)
        self.conv_3 = HarmonicDilatedConv(c_har, embedding, harmonic_dilations)
        self.block_4 = _conv2d_block(
            embedding,
            embedding,
            pool_size=(1, self.freq_pool) if self.freq_pool > 1 else None,
            dilation=(1, int(dilations[0])),
        )
        self.block_5 = _conv2d_block(embedding, embedding, dilation=(1, int(dilations[1])))
        self.block_6 = _conv2d_block(embedding, embedding, (5, 1))
        self.block_7 = _conv2d_block(embedding, embedding, (5, 1))
        self.block_8 = _conv2d_block(embedding, embedding, (5, 1))

    def forward(self, log_gram_db: torch.Tensor) -> torch.Tensor:
        x = self.block_2_5(self.block_2(self.block_1(log_gram_db)))
        x = self.block_4(self.conv_3(x))
        x = self.block_5(x)
        return self.block_8(self.block_7(self.block_6(x)))


class CQTLogSpecgram(nn.Module):
    """HPPNet's CQT front end: nnAudio magnitude CQT -> dB, ``(B, n) -> (B, T, F)``.

    `models.basic_pitch.cqt.CQTFrontEnd` is NOT reused: it fixes the hop at 256,
    derives ``n_bins`` from a harmonic-stacking rule, and ends in basic-pitch's
    `NormalizedLog` rather than `AmplitudeToDB`. Three of the four numbers this
    front end is defined by would have to be overridden, so nnAudio is called
    directly with HPPNet's own arguments.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 512,
        fmin: float = 27.5,
        n_bins: int = 352,
        bins_per_octave: int = 48,
    ):
        super().__init__()
        import torchaudio
        from nnAudio.features.cqt import CQT2010v2

        self.cqt = CQT2010v2(
            sr=int(sample_rate),
            hop_length=int(hop_length),
            fmin=float(fmin),
            n_bins=int(n_bins),
            bins_per_octave=int(bins_per_octave),
            output_format="Magnitude",
            verbose=False,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        mag = self.cqt(audio)  # (B, F, T)
        return self.amplitude_to_db(mag).transpose(1, 2)  # (B, T, F)


class HPPNetOrig(LayerCRFReadout, SalienceRPSPredictor):
    """Audio -> rotor-rate salience logits ``(B, n_maps * 352, T)`` on the LOG grid.

    The `salience_rps` contract: ``forward(audio) -> (B, F, T)`` logits,
    ``outputs_salience = True``, BCE against the RPS-derived target, Hungarian
    tracking at eval. The output axis is the CQT's own grid, so
    ``output_freqs()`` is ``27.5 * 2 ** (i / 48)`` for ``i = 0 .. 351`` — the
    same array `HarmoF0Orig` emits and the same one
    `conf/loss/salience_bce_orig.yaml` builds.

    Args:
        n_fft, hop_length: the frame grid `predict_rps` resamples its output
            onto. ``hop_length`` is ALSO the CQT hop, so the salience time grid
            and the target time grid are one grid; ``n_fft`` is recorded only
            (the CQT has no ``n_fft`` of its own to agree with).
        num_rotors: rotors to track at eval.
        sr, fmin, n_bins, bins_per_octave: the CQT grid.
        c_har, embedding: HPPNet's trunk widths (16 and 128).
        lstm_size: `FreqGroupLSTM`'s recurrent width. Its cost is ``B * F``
            sequences of length ``T``; with the frequency pool off ``F`` is 352
            against HPPNet's 88, so this is the model's memory knob.
        dilations: ``block_4``/``block_5`` frequency dilations. The published
            ``(48, 12)`` is one octave on each block's own axis only when the
            frequency pool is on (deviation 3).
        freq_pool, time_pooling: the two published pools (deviations 1 and 2).
        n_maps: salience maps emitted. See `HarmoF0Orig` for the caveat that
            `LayerCRFReadout`'s transition band carries on a log grid.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        sr: int = 16000,
        fmin: float = 27.5,
        n_bins: int = 352,
        bins_per_octave: int = 48,
        c_har: int = 16,
        embedding: int = 128,
        lstm_size: int = 128,
        dilations: tuple[int, int] = (48, 12),
        freq_pool: int = 1,
        time_pooling: bool = False,
        n_maps: int = 1,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        if int(n_bins) % int(freq_pool):
            raise ValueError(f"n_bins={n_bins} is not divisible by freq_pool={freq_pool}")
        self.sr, self.n_maps = int(sr), int(n_maps)
        self.freq_pool, self.time_pooling = int(freq_pool), bool(time_pooling)

        self.frontend = CQTLogSpecgram(
            sample_rate=int(sr),
            hop_length=int(hop_length),
            fmin=float(fmin),
            n_bins=int(n_bins),
            bins_per_octave=int(bins_per_octave),
        )
        self.trunk = CNNTrunk(
            c_in=1,
            c_har=int(c_har),
            embedding=int(embedding),
            freq_pool=self.freq_pool,
            dilations=dilations,
        )
        self.head = FreqGroupLSTM(int(embedding), self.n_maps, int(lstm_size), sigmoid=False)

        # Grid descriptor. With the pool off these are the CQT's own bins; with
        # ``freq_pool = n`` output bin j takes the max over input bins
        # ``n*j .. n*j + n - 1``, whose centre is ``n*j + (n-1)/2``.
        self.fmin = float(fmin)
        self.bins_per_octave = int(bins_per_octave)
        self.over_sample = int(bins_per_octave) // 12
        self.n_bins = int(n_bins) // self.freq_pool
        self.n_octaves = float(self.n_bins) / float(bins_per_octave) * self.freq_pool
        centers = (
            np.arange(self.n_bins, dtype=np.float64) * self.freq_pool + (self.freq_pool - 1) / 2.0
        )
        self.out_freqs = float(fmin) * 2.0 ** (centers / float(bins_per_octave))
        self.spec_sr = int(sr)
        self.spec_hop = int(hop_length)

    # ── grid ────────────────────────────────────────────────────────────────

    def grid_params(self) -> dict:
        if self.freq_pool != 1:
            raise NotImplementedError(
                "with freq_pool > 1 the output bins are pooled groups, not CQT "
                "bins; read the axis from `output_freqs()` / `out_freqs`"
            )
        return super().grid_params()

    def num_grid_frames(self, n_samples: int) -> int:
        return int(n_samples) // self.spec_hop + 1

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.frontend(audio).unsqueeze(1)  # (B, 1, T, F)
        n_time = x.shape[2]
        if self.time_pooling:
            x = F.max_pool2d(x, (2, 1))
        y = self.head(self.trunk(x))  # (B, n_maps, T', F_out)
        if self.time_pooling:
            y = F.interpolate(y, size=(n_time, y.shape[-1]), mode="bilinear", align_corners=False)
        y = y.transpose(2, 3)  # (B, n_maps, F_out, T)
        b, m, f, t = y.shape
        return y.reshape(b, m * f, t)

    # No `to()` override, unlike `LateDeepSalience`: nnAudio's `CQT2010v2` is an
    # ordinary submodule holding ordinary buffers, so `nn.Module.to` moves it.
