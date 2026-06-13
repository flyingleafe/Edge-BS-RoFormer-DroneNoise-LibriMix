"""Salience-map RPS predictors — multi-pitch models adapted as RPS baselines.

These models output a per-frequency-bin **salience map** (raw logits) instead of
RPS trajectories directly:

    forward(audio) -> (B, n_bins, T_grid)   logits on the model's CQT grid

- **Training** (``train_rps_predictor.py``): BCE against ``rps_to_salience()``
  binary/soft targets (see ``salience_target``). The target is deterministic per
  sample, so it is precomputed/cached by the dataset — no per-step tracking.
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multif0.utils import rps_to_salience, salience_to_rps_segmented


def stft_time_frames(audio_length: int, hop_length: int) -> int:
    """Number of STFT output frames for an audio length (center padding)."""
    return audio_length // hop_length + 1


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
        return rps_to_salience(
            rps,
            n_grid,
            **self.grid_params(),
            hcqt_sr=self.spec_sr,
            hcqt_hop=self.spec_hop,
            rps_sr=rps_sr,
            blur_bins=blur_bins,
        )

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
        max_jump_bins: int = 3,
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
                [self.forward(audio[i : i + chunk_size]) for i in range(0, audio.shape[0], chunk_size)],
                dim=0,
            )
        else:
            logits = self.forward(audio)  # (B, n_bins, T_grid)
        salience = torch.sigmoid(logits)
        rps_grid, _merge = salience_to_rps_segmented(
            salience,
            num_rotors=self.num_rotors,
            **self.grid_params(),
            threshold=threshold,
            max_jump_bins=max_jump_bins,
        )  # (B, num_rotors, T_grid)

        # Tracking leaves NaN where a rotor is never assigned a peak.
        rps_grid = torch.nan_to_num(rps_grid, nan=0.0)

        # Resample grid frames -> STFT frames. Endpoint-to-endpoint shape-stretch,
        # matching how the GT RPS target is built in DREGONRPSDataset (both cover
        # the same audio span, so the time axes align).
        n_samples = audio.shape[-1]
        t_stft = stft_time_frames(n_samples, self.hop_length)
        if rps_grid.shape[-1] != t_stft:
            rps_grid = F.interpolate(rps_grid, size=t_stft, mode="linear", align_corners=False)
        return rps_grid


class LateDeepSalience(SalienceRPSPredictor):
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
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        fmin: float = 27.5,  # A0; matches basic-pitch ANNOTATIONS_BASE_FREQUENCY
        fused_branches: bool = False,
        frontend: nn.Module | None = None,
        **frontend_kwargs,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
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
        self.cnn = LateDeep(n_harmonics=self.n_harmonics, fused_branches=fused_branches)

        # Grid descriptor (HCQT params)
        self.fmin = fe.fmin
        self.n_octaves = fe.n_octaves
        self.over_sample = fe.over_sample
        self.bins_per_octave = 12 * fe.over_sample
        self.n_bins = fe.n_bins
        self.spec_sr = fe.sr
        self.spec_hop = fe.hop_length

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
        logits = self.cnn(mag, dphase, return_logits=True)  # (B, 1, F, T)
        return logits.squeeze(1)  # (B, F, T)

    def to(self, *args, **kwargs):
        # HCQT nnAudio modules are not plain submodules — move them explicitly.
        if hasattr(self.frontend, "to"):
            self.frontend = self.frontend.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class BasicPitchSalience(SalienceRPSPredictor):
    """Basic Pitch contour branch as a salience-map RPS baseline.

    Uses the 264-bin contour grid (fmin 27.5, 36 bins/oct). Trained from scratch
    at native 16 kHz. The pretrained/22.05 kHz path is stubbed but deferred.
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
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        from models.basic_pitch.cqt import (
            ANNOTATIONS_BASE_FREQUENCY,
            CONTOURS_BINS_PER_SEMITONE,
            FFT_HOP,
            N_FREQ_BINS_CONTOURS,
        )
        from models.basic_pitch.model import BasicPitch

        if pretrained:
            # Deferred: pretrained kernels assume 22.05 kHz CQT input.
            raise NotImplementedError(
                "Zero-shot pretrained Basic Pitch (with 16k->22.05k resampling) "
                "is deferred; train from scratch at native 16 kHz instead."
            )

        self.net = BasicPitch(n_harmonics=n_harmonics, sr=sr)
        if freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)

        # Grid descriptor (contour grid is sample-rate-invariant)
        self.spec_sr = sr
        self.spec_hop = FFT_HOP  # 256
        self.fmin = ANNOTATIONS_BASE_FREQUENCY  # 27.5
        self.over_sample = CONTOURS_BINS_PER_SEMITONE  # 3 (bins/semitone)
        self.bins_per_octave = 12 * CONTOURS_BINS_PER_SEMITONE  # 36
        self.n_bins = N_FREQ_BINS_CONTOURS  # 264
        self.n_octaves = (
            N_FREQ_BINS_CONTOURS / self.bins_per_octave
        )  # ~7.33 (unused; n_bins explicit)

    def num_grid_frames(self, n_samples: int) -> int:
        # nnAudio CQT2010v2 emits n_samples // hop + 1 frames (matches the
        # reference 43844-sample -> 172-frame mapping at hop 256).
        return n_samples // self.spec_hop + 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        logits = self.net.contour_logits(audio)  # (B, time, 264)
        return logits.transpose(1, 2)  # (B, 264, time)
