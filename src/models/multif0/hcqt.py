"""
Harmonic Constant-Q Transform (HCQT) computation.

Precisely follows the pumpp HCQTPhaseDiff implementation used in the original paper:
    https://github.com/bmcfee/pumpp

Parameters (from paper Section 4.1 and original config):
    sr           = 22050 Hz
    fmin         = 32.7 Hz (C1)
    bins_per_octave = 60 (20 cents per bin)
    n_octaves    = 6
    over_sample  = 5
    hop_length   = 256 samples
    harmonics    = [1, 2, 3, 4, 5]

Output:
    HCQT magnitude:     (n_harmonics, n_bins, n_frames) = (5, 360, T)
    HCQT phase diff:    (n_harmonics, n_bins, n_frames) = (5, 360, T)

    NOTE: this is a "channels_first" layout for direct PyTorch Conv2d input:
        (batch, channels=harmonics, freq=n_bins, time=n_frames)

    For compatibility with the original code's "channels_last" format
    (T, 360, 5), transpose to (2, 1, 0).
"""

import librosa
import numpy as np
from librosa import amplitude_to_db, cqt, magphase
from librosa.util import fix_length

# ── Default parameters (exactly as in the paper) ───────────────────────────


def hcqt_params():
    """Return the HCQT parameters used in the paper."""
    return dict(
        sr=22050,
        fmin=32.7,  # C1
        n_octaves=6,
        over_sample=5,  # → 60 bins/oct = 20 cents/bin
        hop_length=256,
        harmonics=[1, 2, 3, 4, 5],
        log=True,  # magnitude in dB
    )


# ── Frequency / time grids ─────────────────────────────────────────────────


def freq_grid(
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    bins_per_octave: int | None = None,
) -> np.ndarray:
    """HCQT frequency grid (center frequencies of each bin)."""
    if bins_per_octave is None:
        bins_per_octave = 12 * over_sample
    n_bins = n_octaves * bins_per_octave
    return librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)


def time_grid(
    n_frames: int,
    sr: float = 22050.0,
    hop_length: int = 256,
) -> np.ndarray:
    """HCQT time grid (center time of each frame, in seconds)."""
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)


# ── Phase differential ─────────────────────────────────────────────────────


def _phase_diff(phase: np.ndarray, axis: int = 0) -> np.ndarray:
    """Unwrapped phase differential along an axis.

    Mirrors pumpp.feature._utils.phase_diff with axis=0 (channels_last / time axis).
    """
    dphase = np.empty_like(phase)
    zero_idx = [slice(None)] * phase.ndim
    zero_idx[axis] = slice(1)
    else_idx = [slice(None)] * phase.ndim
    else_idx[axis] = slice(1, None)
    zero_idx = tuple(zero_idx)
    else_idx = tuple(else_idx)

    dphase[zero_idx] = phase[zero_idx]
    dphase[else_idx] = np.diff(np.unwrap(phase, axis=axis), axis=axis)
    return dphase


# ── HCQT computation ───────────────────────────────────────────────────────


def compute_hcqt(
    audio: np.ndarray,
    sr: float = 22050.0,
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    harmonics: list[int] = None,
    hop_length: int = 256,
    log: bool = True,
    dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    """Compute HCQT magnitude and raw phase.

    Parameters
    ----------
    audio : np.ndarray, shape (n_samples,)
        Audio waveform at sample rate `sr`.
    sr : float
        Sample rate. Audio is resampled if needed.
    fmin : float
        Minimum frequency for harmonic 1 (Hz).
    n_octaves : int
        Number of octaves.
    over_sample : int
        Bins per semitone.
    harmonics : list of int
        Harmonic indices to compute (default [1,2,3,4,5]).
    hop_length : int
        Hop length for CQT frames.
    log : bool
        If True, convert magnitude to dB (ref=max).
    dtype : np.dtype
        Output data type.

    Returns
    -------
    data : dict
        'mag'   : np.ndarray, shape (n_harmonics, n_bins, n_frames)
                  HCQT magnitude (dB if log=True, linear otherwise).
        'phase' : np.ndarray, shape (n_harmonics, n_bins, n_frames)
                  Raw phase in radians.
    """
    if harmonics is None:
        harmonics = [1, 2, 3, 4, 5]
    else:
        harmonics = list(harmonics)

    bins_per_octave = 12 * over_sample
    n_bins = n_octaves * bins_per_octave
    n_frames = librosa.time_to_frames(
        librosa.get_duration(y=audio, sr=sr),
        sr=sr,
        hop_length=hop_length,
    )

    mags, phases = [], []
    for h in harmonics:
        C = cqt(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin * h,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
        C = fix_length(C, size=n_frames)
        C_mag, C_phase = magphase(C)

        if log:
            C_mag = amplitude_to_db(C_mag, ref=np.max)

        mags.append(C_mag)
        phases.append(C_phase)

    # Stack: list of (n_bins, n_frames) → (n_harmonics, n_bins, n_frames)
    mag = np.asarray(mags, dtype=dtype)
    phase = np.angle(np.asarray(phases, dtype=np.complex64))

    return {"mag": mag, "phase": phase}


def compute_hcqt_mag_phase(
    audio: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute HCQT magnitude and phase differentials.

    This is the exact equivalent of pumpp's HCQTPhaseDiff with
    conv='channels_last' (default), but returns data in PyTorch-friendly
    (n_harmonics, n_bins, n_frames) layout.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform.
    **kwargs
        Passed to compute_hcqt().

    Returns
    -------
    mag : np.ndarray, shape (H, F, T)
        HCQT magnitude (dB).
    dphase : np.ndarray, shape (H, F, T)
        Unwrapped phase differentials along the time axis.
    """
    data = compute_hcqt(audio, **kwargs)

    mag = data["mag"]  # (H, F, T)
    phase = data["phase"]  # (H, F, T)

    # Phase differential along time axis (axis=2 for (H, F, T) → time is last)
    # Original pumpp uses axis=0 on (T, F, H) format → we use axis=2 on (H, F, T)
    dphase = _phase_diff(phase, axis=2)

    return mag, dphase


# ── PyTorch Module wrapper ─────────────────────────────────────────────────


class HCQT:
    """HCQT feature extractor (convenience wrapper, not differentiable).

    Usage
    -----
    >>> extractor = HCQT()
    >>> mag, dphase = extractor(audio)   # audio is np.ndarray
    >>> # mag.shape    = (5, 360, T)
    >>> # dphase.shape = (5, 360, T)
    """

    def __init__(self, **kwargs):
        params = hcqt_params()
        params.update(kwargs)
        self.params = params

    @property
    def n_bins(self) -> int:
        return self.params["n_octaves"] * 12 * self.params["over_sample"]

    @property
    def n_harmonics(self) -> int:
        return len(self.params["harmonics"])

    def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return compute_hcqt_mag_phase(audio, **self.params)

    def freq_grid(self) -> np.ndarray:
        return freq_grid(
            fmin=self.params["fmin"],
            n_octaves=self.params["n_octaves"],
            over_sample=self.params["over_sample"],
        )

    def time_grid(self, n_frames: int) -> np.ndarray:
        return time_grid(
            n_frames,
            sr=self.params["sr"],
            hop_length=self.params["hop_length"],
        )
