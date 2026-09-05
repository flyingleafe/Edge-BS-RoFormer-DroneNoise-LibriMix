"""Salience-map RPS predictors — multi-pitch models adapted as RPS baselines.

These models output a per-frequency-bin **salience map** (raw logits) instead of
RPS trajectories directly:

    forward(audio) -> (B, n_bins, T_grid)   logits on the model's CQT grid

- **Training** (``train_rps_predictor.py``): BCE against binary/soft salience
  targets derived on the fly from the batch's STFT-grid RPS target (see
  ``salience_target_from_frame_rps``). This keeps every dataloader on the common
  ``(audio, rps)`` task interface; no per-step inverse tracking is involved.
- **Inference / eval**: ``predict_rps()`` does ``sigmoid -> salience_to_rps_segmented``
  (Hungarian tracking) -> resample to the STFT frame grid -> ``(B, num_rotors, T_stft)``,
  so the *existing* global-PIT metrics in ``evaluate()`` apply unchanged.

Models are flagged with ``outputs_salience = True`` so the train/eval loops route
them to the BCE/tracking path. Two concrete baselines:

    LateDeepSalience    — Cuesta et al. ISMIR 2020 LateDeep CNN over an HCQT
                          front-end (native 16 kHz, 3 harmonics, 360-bin grid).
    BasicPitchSalience  — Bittner et al. ICASSP 2022 contour branch (264-bin
                          grid, fmin 27.5, 36 bins/oct). Trained from scratch at
                          native 16 kHz; pretrained/22.05 kHz path is deferred.

PER-ROTOR LAYERS (``n_maps``). Both baselines take ``n_maps=R``, which is the
same option the harmonic ports carry, and it changes the OUTPUT ONLY — the
front end, the trunk and every input grid stay as they are. `models.
salience_crf` measured why it exists: encode real training telemetry into ONE
shared salience map and decode it again — a PERFECT target, no model involved —
and the trajectory comes back 8.24 rev/s away on average, against 2.22e-16 for
Gaussian per-rotor layers read by a CRF plus a log-parabolic fit. 8.24 rev/s is
an oracle floor, thus every shared-map row measured the representation as much
as the architecture.

``n_maps > 1`` widens the OUTPUT HEAD and nothing else — the trunk's final 1x1
map convolution, plus the channel width the super-resolution head carries — and
stacks the maps along the codec's ``(batch, freq, time)`` output axis, width
``n_maps * out_bins``, because a 4-D model output does not type-check through
``tasks.codecs.SalienceRPSCodec``. Pair it with
``conf/loss/salience_layers_r150_h256.yaml`` and
``conf/metrics/salience_layers_r150_h256.yaml`` — the hop-256 twins of the
ports' r150 pair, because both front ends here emit salience at hop 256.
`models.harmonic_ports.layer_readout.LayerCRFReadout` then decodes one CRF best
path per layer, with no threshold and no Hungarian step. It needs an explicit
LINEAR output grid (``superres_out=True``), because the log-parabolic readout
and the CRF band are both defined on a uniform axis. ``n_maps=1`` is the old
model exactly: same parameter names, same shapes, same shared-map decoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_ports.layer_readout import LayerCRFReadout
from models.multif0.utils import (
    cqt_freq_grid,
    linear_freq_grid,
    rps_to_salience,
    salience_target_from_resampled_rps,
    salience_to_rps_segmented,
)


def stft_time_frames(audio_length: int, hop_length: int) -> int:
    """Number of STFT output frames for an audio length (center padding)."""
    return audio_length // hop_length + 1


def _freq_interp_matrix(in_freqs: np.ndarray, out_freqs: np.ndarray) -> np.ndarray:
    """Linear-interpolation resampling matrix ``W`` of shape ``(F_out, F_in)``.

    ``W @ x`` maps a per-input-bin vector on ``in_freqs`` to ``out_freqs`` by
    linear interpolation in Hz (handles log→linear, any ascending grids). Output
    frequencies outside the input range clamp to the nearest input bin.
    """
    in_freqs = np.asarray(in_freqs, dtype=np.float64)
    out_freqs = np.asarray(out_freqs, dtype=np.float64)
    F_in, F_out = len(in_freqs), len(out_freqs)
    hi = np.clip(np.searchsorted(in_freqs, out_freqs), 1, F_in - 1)
    lo = hi - 1
    f_lo, f_hi = in_freqs[lo], in_freqs[hi]
    w_hi = np.clip((out_freqs - f_lo) / (f_hi - f_lo + 1e-12), 0.0, 1.0)
    W = np.zeros((F_out, F_in), dtype=np.float32)
    rows = np.arange(F_out)
    W[rows, lo] = 1.0 - w_hi
    W[rows, hi] = w_hi
    return W


class FreqSuperResHead(nn.Module):
    """Resample salience logits from the model's input grid to a finer output grid,
    then learn to sharpen along frequency (regression-by-classification super-res).

    A fixed linear-interpolation matrix maps the model's native (log-spaced) input
    frequency grid to an arbitrary output grid (e.g. a fine *linear* 55–110 Hz
    grid); a small ``(kernel, 1)`` conv stack then learns to deconvolve/sharpen the
    peak on the fine grid. The time axis is untouched, so output ``T`` == input ``T``.

    ``n_maps`` carries per-rotor salience layers through: the maps ride the
    channel axis of the conv stack, thus the first layer MIXES them (a rotor's
    layer can use what the others hold) and the last layer emits one refined
    map each. At ``n_maps=1`` both convolutions keep their old shapes and
    names, so a checkpoint of the single-map head loads unchanged.
    """

    def __init__(
        self,
        in_freqs: np.ndarray,
        out_freqs: np.ndarray,
        hidden: int = 32,
        kernel: int = 5,
        n_layers: int = 2,
        n_maps: int = 1,
    ):
        super().__init__()
        self.register_buffer("W", torch.from_numpy(_freq_interp_matrix(in_freqs, out_freqs)))
        self.n_maps = int(n_maps)
        pad = kernel // 2
        layers: list[nn.Module] = []
        ch_in = self.n_maps
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(ch_in, hidden, (kernel, 1), padding=(pad, 0)),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
            ch_in = hidden
        layers += [nn.Conv2d(ch_in, self.n_maps, (1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, logits_in: torch.Tensor) -> torch.Tensor:
        """``(B, M, F_in, T)`` salience logits → ``(B, M, F_out, T)`` on the output grid.

        The resampling is per map, thus the maps fold into the batch axis for
        it; at ``M == 1`` that fold is a no-op and the arithmetic is the old
        one, bit for bit.
        """
        b, m, _f_in, t = logits_in.shape
        x = torch.einsum("oi,bit->bot", self.W, logits_in.reshape(b * m, _f_in, t))
        x = self.net(x.reshape(b, m, -1, t))  # (B, M, F_out, T)
        return x


class SalienceRPSPredictor(nn.Module):
    """Base class for salience-map RPS baselines.

    Subclasses set the grid descriptor attributes (``fmin``, ``n_octaves``,
    ``over_sample``, ``n_bins``, ``bins_per_octave``, ``spec_sr``, ``spec_hop``)
    and implement ``forward`` (returning ``(B, n_bins, T_grid)`` logits) and
    ``num_grid_frames``.
    """

    outputs_salience = True

    # Set by subclasses
    fmin: float
    n_octaves: float
    over_sample: int
    n_bins: int
    bins_per_octave: int
    spec_sr: int
    spec_hop: int

    # Optional explicit OUTPUT salience grid (Hz), decoupled from the input CQT
    # grid. When set (by a subclass with a super-resolution head), the salience
    # target and the tracker run on this grid instead of ``grid_params()``.
    out_freqs: np.ndarray | None = None

    def __init__(self, n_fft: int, hop_length: int, num_rotors: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length  # STFT hop (target output time grid)
        self.num_rotors = num_rotors

    # ── grid / target ────────────────────────────────────────────────────────

    def grid_params(self) -> dict:
        """CQT grid descriptor consumed by the rps<->salience helpers."""
        return dict(
            fmin=self.fmin,
            n_octaves=self.n_octaves,
            over_sample=self.over_sample,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

    def output_freqs(self) -> np.ndarray:
        """Frequency grid (Hz) of the salience the model actually emits.

        The explicit ``out_freqs`` grid if set (super-resolution head), else the
        CQT input grid from ``grid_params()``.
        """
        if self.out_freqs is not None:
            return np.asarray(self.out_freqs, dtype=np.float64)
        return cqt_freq_grid(**self.grid_params())

    def _default_max_jump_bins(self) -> int:
        """Frames-to-frame jump cap, scaled to the output grid (~1.5 Hz)."""
        if self.out_freqs is None:
            return 3
        spacing = float(np.median(np.diff(np.asarray(self.out_freqs)))) or 1.0
        return max(3, int(round(1.5 / abs(spacing))))

    def num_grid_frames(self, n_samples: int) -> int:
        """Number of salience time frames the front-end emits for this length."""
        raise NotImplementedError

    def salience_target(
        self,
        rps: torch.Tensor,
        n_samples: int,
        *,
        rps_sr: float = 1000.0,
        blur_bins: int = 0,
    ) -> torch.Tensor:
        """Binary/soft salience target on this model's grid.

        Args:
            rps: ``(4, T_rps)`` or ``(B, 4, T_rps)`` raw RPS (Hz) at ``rps_sr``.
            n_samples: audio length (to size the time grid).
            blur_bins: frequency-axis smoothing half-width (0 = strictly binary).

        Returns:
            ``(n_bins, T_grid)`` or ``(B, n_bins, T_grid)``.
        """
        n_grid = self.num_grid_frames(n_samples)
        if self.out_freqs is not None:
            # Decoupled output grid (e.g. fine linear 55–110 Hz). The time grid
            # (spec_sr/spec_hop) is still the front-end's; only the freq axis changes.
            return rps_to_salience(
                rps,
                n_grid,
                freqs=self.out_freqs,
                hcqt_sr=self.spec_sr,
                hcqt_hop=self.spec_hop,
                rps_sr=rps_sr,
                blur_bins=blur_bins,
            )
        return rps_to_salience(
            rps,
            n_grid,
            **self.grid_params(),
            hcqt_sr=self.spec_sr,
            hcqt_hop=self.spec_hop,
            rps_sr=rps_sr,
            blur_bins=blur_bins,
        )

    def salience_target_from_frame_rps(
        self,
        rps_frames: torch.Tensor,
        n_samples: int,
        *,
        blur_bins: int = 0,
    ) -> torch.Tensor:
        """Build a BCE target from RPS already resampled to the STFT frame grid.

        Training datasets should be able to expose the same ``(audio, rps)``
        batch for every RPS-prediction model.  Direct-regression models consume
        the STFT-grid RPS directly; salience models convert it to their own
        time/frequency grid here, inside the training loop.

        Args:
            rps_frames: ``(4, T_stft)`` or ``(B, 4, T_stft)`` RPS in Hz.
            n_samples: waveform length used to size this model's salience grid.
            blur_bins: frequency-axis smoothing half-width.

        Returns:
            ``(n_bins_out, T_grid)`` or ``(B, n_bins_out, T_grid)``.
        """
        was_batched = rps_frames.dim() == 3
        if not was_batched:
            rps_frames = rps_frames.unsqueeze(0)

        n_grid = self.num_grid_frames(n_samples)
        rps_grid = F.interpolate(
            rps_frames.float(), size=n_grid, mode="linear", align_corners=False
        )
        salience = salience_target_from_resampled_rps(
            rps_grid, self.output_freqs(), blur_bins=blur_bins
        )

        if not was_batched:
            salience = salience.squeeze(0)
        return salience

    # ── forward / inference ──────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return salience **logits** ``(B, n_bins, T_grid)``."""
        raise NotImplementedError

    @torch.no_grad()
    def predict_rps(
        self,
        audio: torch.Tensor,
        *,
        threshold: float = 0.3,
        max_jump_bins: int | None = None,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """Salience -> tracked RPS on the STFT frame grid, ``(B, num_rotors, T_stft)``.

        Hungarian tracking runs on CPU/numpy (slow but validation-only).

        The CNN forward is run in row-chunks of ``chunk_size``. Validation clips
        are typically much longer than training clips, and LateDeep's (360, 1)
        distribution conv has activation memory ~ ``B·T``; forwarding the whole
        flattened multichannel batch (B*C rows) at full length in fp32 OOMs.
        Chunking bounds peak memory without affecting results (rows are
        independent). ``chunk_size <= 0`` disables chunking.
        """
        if chunk_size and chunk_size > 0 and audio.shape[0] > chunk_size:
            logits = torch.cat(
                [
                    self.forward(audio[i : i + chunk_size])
                    for i in range(0, audio.shape[0], chunk_size)
                ],
                dim=0,
            )
        else:
            logits = self.forward(audio)  # (B, n_bins, T_grid)
        salience = torch.sigmoid(logits)
        if max_jump_bins is None:
            max_jump_bins = self._default_max_jump_bins()
        if self.out_freqs is not None:
            rps_grid, _merge = salience_to_rps_segmented(
                salience,
                num_rotors=self.num_rotors,
                freqs=self.out_freqs,
                threshold=threshold,
                max_jump_bins=max_jump_bins,
            )  # (B, num_rotors, T_grid)
        else:
            rps_grid, _merge = salience_to_rps_segmented(
                salience,
                num_rotors=self.num_rotors,
                **self.grid_params(),
                threshold=threshold,
                max_jump_bins=max_jump_bins,
            )  # (B, num_rotors, T_grid)

        # Dark frames already decode to 0.0 inside the tracker (silence == zero
        # rotor speed). This guard only covers a rotor that never gets a peak.
        rps_grid = torch.nan_to_num(rps_grid, nan=0.0)

        # Resample grid frames -> STFT frames. Endpoint-to-endpoint shape-stretch,
        # matching how the GT RPS target is built in DREGONRPSDataset (both cover
        # the same audio span, so the time axes align).
        n_samples = audio.shape[-1]
        t_stft = stft_time_frames(n_samples, self.hop_length)
        if rps_grid.shape[-1] != t_stft:
            rps_grid = F.interpolate(rps_grid, size=t_stft, mode="linear", align_corners=False)
        return rps_grid


class LateDeepSalience(LayerCRFReadout, SalienceRPSPredictor):
    """LateDeep multi-F0 CNN over an HCQT front-end, emitting salience logits.

    Native 16 kHz: with the default ``fmin=27.5`` the HCQT front-end auto-derives
    4 harmonics ``[1,2,3,4]`` (Nyquist 8 kHz, top bin 1760 Hz) on a 360-bin grid
    (60 bins/oct, spanning 27.5 → 1760 Hz).

    ``fmin`` defaults to **27.5 Hz (A0)** — matching basic-pitch's
    ``ANNOTATIONS_BASE_FREQUENCY`` — rather than the multi-F0 paper's 32.7 Hz
    (C1), so the grid reaches low enough to cover rotor fundamentals that dip
    below 32.7 Hz. The grid descriptor (fmin/n_bins/...) is read back from the
    front-end, so lowering it automatically reshapes the salience target and the
    Hungarian tracker — no other changes needed.

    ``n_maps=R`` (the ``_l4`` option, see the module docstring) makes the model
    emit ONE SALIENCE LAYER PER ROTOR instead of one shared map. The INPUT is
    untouched: same HCQT, same trunk, same time grid. Only ``LateDeep``'s final
    1x1 convolution and the super-resolution head widen to ``R`` channels, and
    the output is ``(B, R * out_bins, T)``. It requires ``superres_out=True``,
    and it pairs with `losses.LayerPITSalienceBCELoss` plus the CRF readout
    this class inherits from `models.harmonic_ports.layer_readout`.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        fmin: float = 27.5,  # A0; matches basic-pitch ANNOTATIONS_BASE_FREQUENCY
        fused_branches: bool = False,
        frontend: nn.Module | None = None,
        # Decoupled fine OUTPUT salience grid (super-resolution head). When
        # ``superres_out=True`` the model emits salience on a *linear* grid of
        # ``out_bins`` bins over ``[out_fmin, out_fmax]`` Hz instead of on the
        # (log-spaced) HCQT input grid — see FreqSuperResHead.
        superres_out: bool = False,
        out_fmin: float = 55.0,
        out_fmax: float = 110.0,
        out_bins: int = 360,
        head_hidden: int = 32,
        head_kernel: int = 5,
        n_maps: int = 1,
        **frontend_kwargs,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        self.n_maps = int(n_maps)
        if self.n_maps > 1 and not superres_out:
            raise ValueError(
                "per-rotor layers need an explicit LINEAR output grid: the "
                "log-parabolic readout and the CRF band are both defined on a "
                "uniform axis. Set superres_out=True."
            )
        from typing import cast

        from models.frontends import build_frontend
        from models.frontends.hcqt import HCQTFrontEnd
        from models.multif0.model import LateDeep

        if frontend is None:
            frontend = build_frontend("hcqt", phase=True, fmin=fmin, **frontend_kwargs)
        self.frontend = frontend
        # Grid descriptor reads HCQT-specific attributes; cast for static typing
        # (the frontend must expose fmin/n_octaves/over_sample/n_bins/sr/hop_length).
        fe = cast(HCQTFrontEnd, frontend)

        self.n_harmonics = fe.out_channels // 2 if fe.use_phase else fe.out_channels
        self.cnn = LateDeep(
            n_harmonics=self.n_harmonics, fused_branches=fused_branches, n_maps=self.n_maps
        )

        # Grid descriptor (HCQT input params)
        self.fmin = fe.fmin
        self.n_octaves = fe.n_octaves
        self.over_sample = fe.over_sample
        self.bins_per_octave = 12 * fe.over_sample
        self.n_bins = fe.n_bins
        self.spec_sr = fe.sr
        self.spec_hop = fe.hop_length

        # Super-resolution output head (input log grid -> fine linear output grid).
        self.head: FreqSuperResHead | None = None
        if superres_out:
            in_freqs = cqt_freq_grid(**self.grid_params())
            out_freqs = linear_freq_grid(out_fmin, out_fmax, out_bins)
            self.out_freqs = out_freqs
            self.head = FreqSuperResHead(
                in_freqs, out_freqs, hidden=head_hidden, kernel=head_kernel, n_maps=self.n_maps
            )

    def num_grid_frames(self, n_samples: int) -> int:
        from typing import cast

        from models.frontends.hcqt import HCQTFrontEnd

        return cast(HCQTFrontEnd, self.frontend).num_frames(n_samples)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self.frontend(audio)  # (B, 2H, F, T) or (B, H, F, T)
        H = self.n_harmonics
        if getattr(self.frontend, "use_phase", True):
            mag = feats[:, :H, :, :]
            dphase = feats[:, H:, :, :]
        else:
            mag = feats
            dphase = torch.zeros_like(mag)
        logits = self.cnn(mag, dphase, return_logits=True)  # (B, M, F_in, T)
        if self.head is not None:
            logits = self.head(logits)  # (B, M, F_out, T) on the fine linear grid
        # The codec's wire format is 3-D, thus the maps stack along the output
        # axis; `layer_readout.split_maps` reads them back.
        b, m, g, t = logits.shape
        return logits.reshape(b, m * g, t)

    def to(self, *args, **kwargs):
        # HCQT nnAudio modules are not plain submodules — move them explicitly.
        if hasattr(self.frontend, "to"):
            self.frontend = self.frontend.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class BasicPitchSalience(LayerCRFReadout, SalienceRPSPredictor):
    """Basic Pitch contour branch as a salience-map RPS baseline.

    Uses the 264-bin contour grid (fmin 27.5, 36 bins/oct). Trained from scratch
    at native 16 kHz. The pretrained/22.05 kHz path is stubbed but deferred.

    ``n_maps=R`` (the ``_l4`` option, see the module docstring) makes the model
    emit ONE SALIENCE LAYER PER ROTOR instead of one shared map. The INPUT is
    untouched: same CQT, same harmonic stacking, same contour trunk. Only
    ``BasicPitch.contour_out`` and the super-resolution head widen to ``R``
    channels, and the output is ``(B, R * out_bins, T)``. It requires
    ``superres_out=True``, and it pairs with `losses.LayerPITSalienceBCELoss`
    plus the CRF readout this class inherits from
    `models.harmonic_ports.layer_readout`.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        sr: int = 16000,
        n_harmonics: int = 8,
        pretrained: bool = False,
        freeze: bool = False,
        # Narrow input contour grid (defaults reproduce the original 27.5 Hz /
        # 88-semitone / 3-bins-per-semitone basic-pitch contour grid).
        bp_fmin: float = 27.5,
        bins_per_semitone: int = 3,
        n_contour_semitones: int = 88,
        # Decoupled fine OUTPUT salience grid (super-resolution head).
        superres_out: bool = False,
        out_fmin: float = 55.0,
        out_fmax: float = 110.0,
        out_bins: int = 360,
        head_hidden: int = 32,
        head_kernel: int = 5,
        n_maps: int = 1,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        from models.basic_pitch.cqt import FFT_HOP
        from models.basic_pitch.model import BasicPitch

        self.n_maps = int(n_maps)
        if self.n_maps > 1 and not superres_out:
            raise ValueError(
                "per-rotor layers need an explicit LINEAR output grid: the "
                "log-parabolic readout and the CRF band are both defined on a "
                "uniform axis. Set superres_out=True."
            )

        if pretrained:
            # Deferred: pretrained kernels assume 22.05 kHz CQT input.
            raise NotImplementedError(
                "Zero-shot pretrained Basic Pitch (with 16k->22.05k resampling) "
                "is deferred; train from scratch at native 16 kHz instead."
            )

        self.net = BasicPitch(
            n_harmonics=n_harmonics,
            sr=sr,
            fmin=bp_fmin,
            bins_per_semitone=bins_per_semitone,
            n_contour_semitones=n_contour_semitones,
            n_maps=self.n_maps,
        )
        if freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)

        # Grid descriptor of the contour (input) grid — sample-rate-invariant.
        self.spec_sr = sr
        self.spec_hop = FFT_HOP  # 256
        self.fmin = bp_fmin
        self.over_sample = bins_per_semitone
        self.bins_per_octave = 12 * bins_per_semitone
        self.n_bins = self.net.contour_bins
        self.n_octaves = self.n_bins / self.bins_per_octave  # (unused; n_bins explicit)

        # Super-resolution output head (contour log grid -> fine linear grid).
        self.head: FreqSuperResHead | None = None
        if superres_out:
            in_freqs = cqt_freq_grid(**self.grid_params())
            out_freqs = linear_freq_grid(out_fmin, out_fmax, out_bins)
            self.out_freqs = out_freqs
            self.head = FreqSuperResHead(
                in_freqs, out_freqs, hidden=head_hidden, kernel=head_kernel, n_maps=self.n_maps
            )

    def num_grid_frames(self, n_samples: int) -> int:
        # nnAudio CQT2010v2 emits n_samples // hop + 1 frames (matches the
        # reference 43844-sample -> 172-frame mapping at hop 256).
        return n_samples // self.spec_hop + 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        logits = self.net.contour_logits(audio)  # (B, time, M * contour_bins)
        logits = logits.transpose(1, 2)  # (B, M * contour_bins, time)
        b, mg, t = logits.shape
        x = logits.reshape(b, self.n_maps, mg // self.n_maps, t)
        if self.head is not None:
            x = self.head(x)  # (B, M, F_out, time) on the fine linear grid
        # The codec's wire format is 3-D, thus the maps stack along the output
        # axis; `layer_readout.split_maps` reads them back.
        return x.reshape(b, x.shape[1] * x.shape[2], t)
