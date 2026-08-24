"""Classical (non-learned) baselines for per-rotor RPS prediction.

Provenance: restored, without changes to the estimators, from the deleted
root-level module ``classical_rps_predictors.py`` at commit ``00753c4``. The
numbers it produced are in the report
``writing/reports/2026-05-29_classical-baselines/``. The adaptation is
mechanical only: the module moved into the ``experiments`` sandbox, unused
imports and unused locals were removed, and the module-level caches got a
type-safe form. No estimator changed.

Each predictor obeys a common interface::

    predictions = method(audio, sr=16000)  -> ndarray, shape (n_rotors, n_frames)

where ``audio`` is a single-channel waveform (float32, any length) and
``predictions`` contain the estimated rotor speeds in rev/s at the STFT frame
rate (hop_length=512 at 16 kHz, thus approximately 31.25 Hz).

Methods implemented
-------------------
1. pyin_single_f0    - single-f0 tracker (librosa.pyin); the found f0 is
                       converted to RPS by division by the blade count.
                       Because it can track one source only, the same trace
                       is returned for all four rotors (which makes the
                       limit explicit).
2. cepstral_tracker  - real cepstrum + peak picking. A greedy multi-rotor
                       extension suppresses the comb of the strongest rotor
                       and repeats the search up to four times.
3. hps_tracker       - Harmonic Product Spectrum. The same greedy extension
                       as above.
4. matched_filter_tracker - harmonic-comb template correlation. Templates are
                       built for a dense RPS grid; the best match per frame is
                       extracted greedily.
5. nmf_tracker       - semi-supervised NMF with a fixed harmonic-comb
                       dictionary. The magnitude spectrogram is factorized
                       as V ~ W.H; the four strongest activations per frame
                       give the rotor-speed estimates. This is a true
                       multi-pitch method (no greedy suppression step).

Methods 2-4 share the same :func:`_greedy_multi_rotor` helper, which finds the
best RPS, builds a suppression mask around its harmonics, and runs the search
again on the masked spectrum.

The five methods are also available as the ``CLASSICAL_TRACKERS`` dictionary,
which maps a short name to a callable of one argument (the waveform).
"""

from __future__ import annotations

from collections.abc import Callable

import librosa
import numpy as np
from numpy.fft import ifft

# ---------------------------------------------------------------------------
# Constants shared with the learned model
# ---------------------------------------------------------------------------
SR = 16_000
N_FFT = 2048
HOP_LENGTH = 512
N_BLADES = 2  # DREGON quadcopter
N_ROTORS = 4
N_HARMONICS = 15  # number of harmonics used in comb templates
RPS_MIN = 50.0  # rev/s
RPS_MAX = 150.0  # rev/s
HARM_BW_BINS = 3  # suppression half-width around each harmonic


def _stft_frame_count(audio_len: int) -> int:
    """Number of STFT frames for a given audio length (center=True)."""
    return audio_len // HOP_LENGTH + 1


def _rps_to_harmonic_freqs(rps: float, n_harmonics: int = N_HARMONICS) -> np.ndarray:
    """Return harmonic frequencies (Hz) for a given RPS."""
    f0 = N_BLADES * rps
    return np.arange(1, n_harmonics + 1) * f0


def _freq_to_bin(freq: float, n_fft: int = N_FFT, sr: int = SR) -> int:
    """Convert frequency (Hz) to the closest positive FFT bin."""
    return int(np.round(freq * n_fft / sr))


def _frame_spectra(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude spectra for each STFT frame.

    Returns
    -------
    specs : ndarray, shape (n_frames, n_fft//2+1)
        Linear magnitude spectra.
    times : ndarray, shape (n_frames,)
        Frame center times in seconds.
    """
    specs = np.abs(
        librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True)
    ).T  # -> (n_frames, n_freqs)
    times = librosa.frames_to_time(np.arange(specs.shape[0]), sr=SR, hop_length=HOP_LENGTH)
    return specs, times


def _suppress_harmonics(
    spec: np.ndarray,
    rps: float,
    n_harmonics: int = N_HARMONICS,
    half_width: int = HARM_BW_BINS,
) -> np.ndarray:
    """Set the frequency bins around the harmonics of a given RPS to zero.

    Parameters
    ----------
    spec : ndarray, shape (n_freqs,)
        Single-frame magnitude spectrum.
    rps : float
        Rotor speed whose harmonics must be suppressed.
    """
    spec = spec.copy()
    freqs = _rps_to_harmonic_freqs(rps, n_harmonics)
    for f in freqs:
        cb = _freq_to_bin(f)
        if cb < half_width or cb >= len(spec) - half_width:
            continue
        spec[cb - half_width : cb + half_width + 1] = 0.0
    return spec


def _greedy_multi_rotor(
    specs: np.ndarray,
    estimator: Callable[[np.ndarray], float],
    n_rotors: int = N_ROTORS,
) -> np.ndarray:
    """Greedy multi-rotor extractor.

    For each frame, call ``estimator(masked_spec)`` to get one RPS value,
    suppress its harmonics, and repeat up to ``n_rotors`` times.

    Parameters
    ----------
    specs : ndarray, shape (n_frames, n_freqs)
        Frame-wise magnitude spectra.
    estimator : callable
        ``estimator(spec_frame) -> float``  (RPS in rev/s).

    Returns
    -------
    preds : ndarray, shape (n_rotors, n_frames)
    """
    n_frames = specs.shape[0]
    preds = np.zeros((n_rotors, n_frames), dtype=np.float32)
    for t in range(n_frames):
        spec_frame = specs[t].copy()
        for r in range(n_rotors):
            rps = estimator(spec_frame)
            preds[r, t] = rps
            spec_frame = _suppress_harmonics(spec_frame, rps)
    return preds


# ===========================================================================
# 1. Single-f0 tracker (librosa.pyin)
# ===========================================================================


def pyin_single_f0(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Single-f0 tracking with librosa.pyin.

    Because pyin can follow one source only, the same trace is replicated for
    all four rotors. The frequency range is wide enough for both the rotor
    rotation frequency and the blade-passage frequency; the found f0 is
    clipped to the RPS range.

    Parameters
    ----------
    audio : ndarray, shape (n_samples,)
    sr    : int, default 16000

    Returns
    -------
    preds : ndarray, shape (4, n_frames)
    """
    # pyin returns f0 per frame at the default hop_length=512
    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        audio,
        fmin=50.0,
        fmax=400.0,
        sr=sr,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )
    # f0 shape -> (n_frames,)
    n_frames = _stft_frame_count(len(audio))
    if f0.shape[0] < n_frames:
        pad = np.full(n_frames - f0.shape[0], np.nan)
        f0 = np.concatenate([f0, pad])
    elif f0.shape[0] > n_frames:
        f0 = f0[:n_frames]

    # Replace NaN with the median voiced value
    voiced = f0[~np.isnan(f0)]
    if len(voiced) > 0:
        median_f0 = np.median(voiced)
    else:
        median_f0 = np.nanmean(f0) if not np.all(np.isnan(f0)) else 100.0
    f0 = np.where(np.isnan(f0), median_f0, f0)

    # pyin sometimes tracks the blade-passage frequency (2 x RPS) and
    # sometimes the motor-rotation frequency (RPS). Heuristic: if the found
    # value is more than 1.5 x the top of the RPS range, divide it by the
    # blade count; if it is not, keep it as it is.
    rps = np.where(f0 > 1.5 * RPS_MAX, f0 / N_BLADES, f0)
    rps = np.clip(rps, RPS_MIN, RPS_MAX)

    return np.tile(rps.astype(np.float32), (N_ROTORS, 1))


# ===========================================================================
# 2. Cepstral analysis
# ===========================================================================


def _cepstral_rps_estimate(spec_frame: np.ndarray) -> float:
    """Estimate one RPS from a single-frame magnitude spectrum with the cepstrum.

    The real cepstrum is ``IFFT(log(|FFT|))``. A peak in the quefrency domain
    is the period of the fundamental. The search covers the quefrency range
    that maps to RPS_MIN ... RPS_MAX.
    """
    # Prevent log(0)
    log_spec = np.log(np.maximum(spec_frame, 1e-12))
    cepstrum = np.real(ifft(log_spec))

    n_freqs = len(spec_frame)
    # Quefrency in samples: q = 0 ... n_freqs-1
    # Period T = q samples  =>  freq = sr / q  =>  RPS = freq / b = sr / (q * b)
    # RPS in [RPS_MIN, RPS_MAX]  =>  q in [sr/(b*RPS_MAX), sr/(b*RPS_MIN)]
    q_min = int(SR / (N_BLADES * RPS_MAX))
    q_max = int(SR / (N_BLADES * RPS_MIN))
    q_max = min(q_max, n_freqs - 1)

    if q_max <= q_min:
        return (RPS_MIN + RPS_MAX) / 2.0

    search_region = cepstrum[q_min : q_max + 1]
    peak_idx = int(np.argmax(search_region))
    q_peak = q_min + peak_idx
    if q_peak == 0:
        q_peak = 1

    # Convert quefrency (samples) to RPS
    rps = SR / (q_peak * N_BLADES)
    return float(np.clip(rps, RPS_MIN, RPS_MAX))


def cepstral_tracker(audio: np.ndarray, sr: int = SR, n_rotors: int = N_ROTORS) -> np.ndarray:
    """Greedy multi-rotor cepstral RPS tracker.

    For each STFT frame the real cepstrum is computed; the strongest peak in
    the possible quefrency band gives the RPS of the first rotor. Its harmonic
    bins then become zero and the search continues for the other rotors.
    """
    specs, _ = _frame_spectra(audio)
    return _greedy_multi_rotor(specs, _cepstral_rps_estimate, n_rotors)


# ===========================================================================
# 3. Harmonic Product Spectrum (HPS)
# ===========================================================================


def _hps_spectrum(spec_frame: np.ndarray, max_downsample: int = 4) -> np.ndarray:
    """Compute the Harmonic Product Spectrum.

    The spectrum is downsampled again and again by the integer factors 2, 3,
    and so on, and the downsampled spectra are multiplied together. The result
    is a vector of the same length as the initial spectrum; the peak location
    shows the fundamental frequency.
    """
    hps = spec_frame.copy()
    for d in range(2, max_downsample + 1):
        down = spec_frame[::d]
        n = min(len(hps), len(down))
        hps = hps[:n] * down[:n]
    return hps


def _hps_rps_estimate(spec_frame: np.ndarray) -> float:
    """Estimate one RPS from a single frame with HPS peak picking."""
    hps = _hps_spectrum(spec_frame)
    # Search in the frequency band that maps to RPS_MIN ... RPS_MAX only
    f_min_bin = _freq_to_bin(N_BLADES * RPS_MIN)
    f_max_bin = _freq_to_bin(N_BLADES * RPS_MAX)
    f_max_bin = min(f_max_bin, len(hps) - 1)

    if f_max_bin <= f_min_bin:
        return (RPS_MIN + RPS_MAX) / 2.0

    search_region = hps[f_min_bin : f_max_bin + 1]
    peak_idx = int(np.argmax(search_region))
    f0_bin = f_min_bin + peak_idx
    f0 = f0_bin * SR / N_FFT
    rps = f0 / N_BLADES
    return float(np.clip(rps, RPS_MIN, RPS_MAX))


def hps_tracker(audio: np.ndarray, sr: int = SR, n_rotors: int = N_ROTORS) -> np.ndarray:
    """Greedy multi-rotor Harmonic Product Spectrum tracker."""
    specs, _ = _frame_spectra(audio)
    return _greedy_multi_rotor(specs, _hps_rps_estimate, n_rotors)


# ===========================================================================
# 4. Matched-filter bank (harmonic-comb templates)
# ===========================================================================

_TEMPLATE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _build_templates(
    n_freqs: int, rps_min: float = RPS_MIN, rps_max: float = RPS_MAX, step: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Build a bank of harmonic-comb templates.

    Returns
    -------
    rps_grid : ndarray, shape (n_templates,)
    templates : ndarray, shape (n_templates, n_freqs)
        Each template is a binary mask (0/1) with narrow windows around the
        harmonic frequencies of the related RPS. This is equivalent to the
        ``harmonic summation`` classical pitch detector.
    """
    cached = _TEMPLATE_CACHE.get(n_freqs)
    if cached is not None:
        return cached

    rps_grid = np.arange(rps_min, rps_max + step, step)
    templates = np.zeros((len(rps_grid), n_freqs), dtype=np.float32)
    for i, rps in enumerate(rps_grid):
        freqs = _rps_to_harmonic_freqs(float(rps))
        for f in freqs:
            cb = _freq_to_bin(f)
            if cb < 0 or cb >= n_freqs:
                continue
            hw = HARM_BW_BINS
            lo = max(0, cb - hw)
            hi = min(n_freqs, cb + hw + 1)
            templates[i, lo:hi] = 1.0
    _TEMPLATE_CACHE[n_freqs] = (rps_grid, templates)
    return rps_grid, templates


def _matched_filter_rps_estimate(spec_frame: np.ndarray) -> float:
    """Estimate one RPS by correlation of the spectrum with a bank of comb templates."""
    n_freqs = len(spec_frame)
    rps_grid, templates = _build_templates(n_freqs)
    # Normalized correlation (cosine similarity)
    spec_norm = np.linalg.norm(spec_frame)
    if spec_norm < 1e-12:
        return (RPS_MIN + RPS_MAX) / 2.0

    spec_u = spec_frame / spec_norm
    tmpl_norms = np.linalg.norm(templates, axis=1, keepdims=True)
    tmpl_norms = np.where(tmpl_norms < 1e-12, 1.0, tmpl_norms)
    scores = templates @ spec_u / tmpl_norms.squeeze()
    best_idx = int(np.argmax(scores))
    return float(rps_grid[best_idx])


def matched_filter_tracker(audio: np.ndarray, sr: int = SR, n_rotors: int = N_ROTORS) -> np.ndarray:
    """Greedy multi-rotor matched-filter-bank tracker.

    A dense grid of RPS candidates (0.5 rev/s resolution) is correlated with
    the magnitude spectrum of each STFT frame. The best-match template gives
    the first rotor; its harmonic bins are suppressed and the search repeats.
    """
    specs, _ = _frame_spectra(audio)
    return _greedy_multi_rotor(specs, _matched_filter_rps_estimate, n_rotors)


# ===========================================================================
# 5. Semi-supervised NMF with harmonic-comb dictionary
# ===========================================================================

_NMF_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _build_nmf_dictionary(
    n_freqs: int,
    sr: int = SR,
    rps_min: float = RPS_MIN,
    rps_max: float = RPS_MAX,
    step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed NMF dictionary of Gaussian harmonic-comb spectral templates.

    Returns
    -------
    rps_grid : ndarray, shape (n_atoms,)
    W : ndarray, shape (n_freqs, n_atoms)
        Each column is a unit-norm spectral template with Gaussian peaks
        centered at the harmonic frequencies of the related RPS.
    """
    cached = _NMF_CACHE.get(n_freqs)
    if cached is not None:
        return cached

    rps_grid = np.arange(rps_min, rps_max + step, step)
    n_atoms = len(rps_grid)
    freq_bins = np.arange(n_freqs) * sr / N_FFT

    W = np.zeros((n_freqs, n_atoms), dtype=np.float64)
    for i, rps in enumerate(rps_grid):
        freqs = _rps_to_harmonic_freqs(float(rps))
        for f in freqs:
            # Gaussian peak: sigma is approximately 2 Hz (about 4 FFT bins at
            # 16 kHz / 2048)
            sigma = 2.0 * N_FFT / sr
            W[:, i] += np.exp(-0.5 * ((freq_bins - f) / sigma) ** 2)

    # Normalize the columns to unit L2
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    W = W / np.where(col_norms > 1e-12, col_norms, 1.0)

    _NMF_CACHE[n_freqs] = (rps_grid, W)
    return rps_grid, W


def nmf_tracker(
    audio: np.ndarray,
    sr: int = SR,
    n_rotors: int = N_ROTORS,
    n_iter: int = 30,
) -> np.ndarray:
    """Semi-supervised NMF with a fixed harmonic-comb dictionary.

    The magnitude spectrogram ``V`` is factorized as ``V ~ W.H``, where ``W``
    is a bank of Gaussian harmonic-comb templates (one column for each
    candidate RPS) and ``H`` is learned with multiplicative updates. For each
    frame the ``n_rotors`` strongest activations become the rotor-speed
    estimates. This is a true multi-pitch method: all four rotors are
    estimated together, with no greedy suppression step.

    Parameters
    ----------
    audio : ndarray, shape (n_samples,)
    sr    : int, default 16000
    n_rotors : int, default 4
    n_iter : int, default 30
        Number of multiplicative-update iterations.

    Returns
    -------
    preds : ndarray, shape (n_rotors, n_frames)
    """
    V = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True))
    n_freqs, n_frames = V.shape

    rps_grid, W = _build_nmf_dictionary(n_freqs, sr)

    # Warm-start H with the non-negative least-squares solution
    H_init = np.linalg.lstsq(W, V.astype(np.float64), rcond=None)[0]
    H = np.maximum(H_init, 0.0)

    eps = 1e-9
    for _ in range(n_iter):
        WH = W @ H + eps
        H *= (W.T @ V.astype(np.float64)) / (W.T @ WH + eps)

    # Extract the top n_rotors per frame (sorted down by activation)
    preds = np.zeros((n_rotors, n_frames), dtype=np.float32)
    for t in range(n_frames):
        top_idx = np.argsort(H[:, t])[-n_rotors:][::-1]
        preds[:, t] = rps_grid[top_idx].astype(np.float32)

    return preds


# ===========================================================================
# The public surface
# ===========================================================================

#: The five classical predictors, each a callable of one waveform that returns
#: an ``(N_ROTORS, n_frames)`` array of rotor speeds in rev/s.
CLASSICAL_TRACKERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "pyin": pyin_single_f0,
    "cepstral": cepstral_tracker,
    "hps": hps_tracker,
    "matched_filter": matched_filter_tracker,
    "nmf": nmf_tracker,
}


# ===========================================================================
# Convenience: evaluate any classical predictor against ground truth
# ===========================================================================


def evaluate_predictions(preds: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute scalar regression metrics.

    This is the metric definition of the 2026-05-29 report. It does NOT do
    per-frame PIT matching — for the comparable protocol numbers use
    :mod:`experiments.classical_rps.valid_eval`.

    Parameters
    ----------
    preds : ndarray, shape (4, n_frames) or (n_frames,)
    target : ndarray, shape (4, n_frames)

    Returns
    -------
    dict with the keys ``mse``, ``mae``, ``r2``.
    """
    if preds.ndim == 1:
        preds = np.tile(preds, (target.shape[0], 1))
    # Make the lengths the same
    min_len = min(preds.shape[1], target.shape[1])
    preds = preds[:, :min_len]
    target = target[:, :min_len]

    mse = float(np.mean((preds - target) ** 2))
    mae = float(np.mean(np.abs(preds - target)))

    # Per-sample (clip) R2, macro-averaged - the same definition as the paper
    r2_per_sample = []
    for p_i, t_i in zip(preds, target):
        ss_res = ((p_i - t_i) ** 2).sum()
        ss_tot = ((t_i - t_i.mean()) ** 2).sum()
        if ss_tot > 1e-6:
            r2_per_sample.append(1.0 - ss_res / ss_tot)
    r2 = float(np.mean(r2_per_sample)) if r2_per_sample else float("nan")

    return {"mse": mse, "mae": mae, "r2": r2}
