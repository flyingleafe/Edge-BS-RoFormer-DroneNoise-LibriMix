"""
HCQT (Harmonic Constant-Q Transform) spectral front-end.

Two backends:
- ``"nnaudio"`` (default, GPU) — uses ``nnAudio.CQT2010v2``, peak-frequencies
  match librosa to 100% for all bins with signal energy.
- ``"librosa"`` (CPU reference) — uses ``librosa.cqt``, identical to the
  original pumpp-based implementation.

When ``phase=True``, magnitude and phase differentials are stacked along the
channel axis (C = 2 * n_harmonics); the consuming model splits them.

Parameters (paper defaults)
---------------------------
sr           = 22050 Hz  (target CQT rate)
input_sr     = 16000 Hz  (input audio rate, resampled automatically)
fmin         = 32.7 Hz (C1)
n_octaves    = 6
over_sample  = 5  → 60 bins/octave (20¢)
hop_length   = 256 samples
harmonics    = [1, 2, 3, 4, 5]
"""

import numpy as np
import torch
import torchaudio.functional as AF
from torch import Tensor

from . import SpectralFrontEnd, register_frontend


@register_frontend
class HCQTFrontEnd(SpectralFrontEnd):
    """HCQT feature extraction front-end.

    Parameters
    ----------
    sr : int
        Target sample rate for CQT.  Input audio is resampled from
        ``input_sr`` to this rate automatically.
    input_sr : int
        Sample rate of the incoming audio (default 16000).
    fmin : float
        Minimum frequency (Hz).  Default C1 = 32.7 Hz.
    n_octaves : int
        Number of octaves.
    over_sample : int
        Oversampling factor → bins_per_octave = 12 * over_sample.
    harmonics : list[int]
        Which harmonics to compute (e.g. [1, 2, 3, 4, 5]).
    hop_length : int
        Hop length in samples.
    phase : bool
        If True, stack magnitude + phase differential (2H channels).
        If False, magnitude only (H channels).
    use_log : bool
        If True, convert magnitude to log scale (dB).
    backend : str
        ``"nnaudio"`` (default) or ``"librosa"``.
    """

    key = "hcqt"

    def __init__(
        self,
        sr: int = 22050,
        input_sr: int = 16000,
        fmin: float = 32.7,
        n_octaves: int = 6,
        over_sample: int = 5,
        harmonics: list[int] = None,
        hop_length: int = 256,
        phase: bool = True,
        use_log: bool = True,
        backend: str = "nnaudio",
    ):
        super().__init__()
        if backend not in ("nnaudio", "librosa"):
            raise ValueError(f"backend must be 'nnaudio' or 'librosa', got {backend!r}")

        self.sr = sr
        self.input_sr = input_sr
        self.fmin = fmin
        self.n_octaves = n_octaves
        self.over_sample = over_sample
        self.harmonics = harmonics or [1, 2, 3, 4, 5]
        self.hop_length = hop_length
        self.use_phase = phase
        self.use_log = use_log
        self.backend = backend

        n_h = len(self.harmonics)
        self.out_channels = 2 * n_h if phase else n_h

        # ── nnAudio backend ──
        if backend == "nnaudio":
            from models.multif0.nnaudio_cqt import HCQT_nnAudio

            self._nn_hcqt = HCQT_nnAudio(
                sr=sr,
                fmin=fmin,
                n_octaves=n_octaves,
                over_sample=over_sample,
                harmonics=list(self.harmonics),
                hop_length=hop_length,
                log_mag=use_log,
            )
        else:
            self._nn_hcqt = None

        # Cached librosa params (built lazily)
        self._cqt_params: dict | None = None

    # ── time grid ────────────────────────────────────────────────────────

    def num_frames(self, n_samples: int) -> int:
        import math

        fmin_h1 = self.fmin * self.harmonics[0]
        filter_length = self.sr / fmin_h1
        filter_length = int(2 ** math.ceil(math.log2(filter_length)))
        return max(1, 1 + (n_samples - filter_length) // self.hop_length)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, audio: Tensor) -> Tensor:
        """
        Args:
            audio: (B, N) raw waveform at ``self.input_sr`` (default 16 kHz).

        Returns:
            (B, C, F, T)  where C = n_harmonics or 2*n_harmonics.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # Resample to CQT target rate if needed
        if self.input_sr != self.sr:
            audio = AF.resample(audio, self.input_sr, self.sr)

        if self.backend == "nnaudio":
            return self._forward_nnaudio(audio)
        else:
            return self._forward_librosa(audio)

    def _forward_nnaudio(self, audio: Tensor) -> Tensor:
        """GPU path: nnAudio CQT2010v2, entirely on-device."""
        mag, dphase = self._nn_hcqt(audio)  # (B, H, F, T) each

        if self.use_phase:
            return torch.cat([mag, dphase], dim=1)  # (B, 2H, F, T)
        else:
            return mag  # (B, H, F, T)

    def _forward_librosa(self, audio: Tensor) -> Tensor:
        """CPU path: librosa CQT, per-sample numpy loop."""
        B = audio.shape[0]
        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        device = audio.device

        from models.multif0.hcqt import compute_hcqt_mag_phase

        if self._cqt_params is None:
            self._cqt_params = _make_cqt_params(
                self.sr,
                self.fmin,
                self.n_octaves,
                self.over_sample,
                self.harmonics,
                self.hop_length,
                self.use_log,
            )

        # nnAudio path already resampled; librosa path receives audio at sr.
        # … but wait — _forward_librosa gets audio already resampled by
        # forward().  The old code had per-sample librosa.resample; now
        # forward() does it once in torch.  So we can skip the per-sample
        # resample loop here.
        mags, dphases = [], []
        for i in range(B):
            m, dp = compute_hcqt_mag_phase(audio_np[i], **self._cqt_params)
            mags.append(torch.from_numpy(m))
            if self.use_phase:
                dphases.append(torch.from_numpy(dp))

        mag = torch.stack(mags).to(device=device, dtype=torch.float32)
        if self.use_phase:
            dphase = torch.stack(dphases).to(device=device, dtype=torch.float32)
            return torch.cat([mag, dphase], dim=1)
        else:
            return mag

    # Expose key params so MultiF0RPSPredictor can infer freq_grid
    @property
    def n_bins(self) -> int:
        return self.n_octaves * 12 * self.over_sample

    def to(self, *args, **kwargs):
        """Move nnAudio CQT modules when model is moved to device."""
        if self._nn_hcqt is not None:
            self._nn_hcqt = self._nn_hcqt.to(*args, **kwargs)
        return super().to(*args, **kwargs)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_cqt_params(
    sr: int,
    fmin: float,
    n_octaves: int,
    over_sample: int,
    harmonics: list[int],
    hop_length: int,
    log: bool,
) -> dict:
    """Build kwargs dict for ``compute_hcqt_mag_phase``."""
    return dict(
        sr=sr,
        fmin=fmin,
        n_octaves=n_octaves,
        over_sample=over_sample,
        harmonics=harmonics,
        hop_length=hop_length,
        log=log,
    )
