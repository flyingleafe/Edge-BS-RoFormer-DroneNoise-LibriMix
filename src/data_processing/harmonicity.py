"""Harmonicity metrics for rotating-source / harmonic noise.

The project targets *harmonic* noise from rotating sources (C1). To judge how
useful an external noise recording is, we need a cheap, source-agnostic measure
of **how harmonic** it is — how prominent its harmonic-comb structure is above
the broadband floor. This module computes that from raw audio, using only
numpy/scipy (no torch), so it runs on the small box, on the CPU cluster, and
inside DataLoader workers.

The metric, per clip (multichannel input is reduced by averaging the scalar
metrics across channels):

1. Welch PSD on a capped analysis window.
2. Fundamental ``f0`` via the harmonic-product spectrum, restricted to the
   plausible rotating-fundamental range ``[fmin, fmax]``.
3. An integer harmonic comb ``k·f0`` (``k = 1..K`` up to Nyquist); per harmonic,
   the local peak and an interpolated broadband floor in a fractional band.

Reported (:class:`Harmonicity`):

- ``f0_hz`` — estimated fundamental (0.0 if no harmonic structure found).
- ``harmonic_energy_ratio`` ∈ [0, 1] — comb energy above floor / total energy.
- ``harmonic_to_noise_db`` — comb energy vs residual (non-comb) energy.
- ``n_prominent_harmonics`` — harmonics with a peak > floor + 6 dB.
- ``spectral_flatness`` ∈ (0, 1] — Wiener entropy (geo-mean / mean of the PSD);
  low = tonal, ≈1 = noise-like. A cheap complement independent of ``f0``.

These are descriptors, not calibrated physical quantities; they order recordings
by harmonic prominence and are cross-checkable against the project's
``multif0``/``salience`` f0 estimators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import welch

__all__ = ["Harmonicity", "measure_harmonicity"]

_EPS = 1e-12


@dataclass(frozen=True)
class Harmonicity:
    """Scalar harmonicity descriptors for one clip (see module docstring)."""

    f0_hz: float
    harmonic_energy_ratio: float
    harmonic_to_noise_db: float
    n_prominent_harmonics: int
    spectral_flatness: float

    def as_dict(self) -> dict[str, Any]:
        """Plain JSON-serializable dict (native Python scalars)."""
        return {
            k: (int(v) if k == "n_prominent_harmonics" else float(v))
            for k, v in asdict(self).items()
        }


def _analysis_signal(audio: np.ndarray, sample_rate: int, max_seconds: float) -> np.ndarray:
    """``(T,)`` or ``(C, T)`` → a list of mono channels, DC-removed, centre-cropped."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"audio must be (T,) or (C, T), got shape {x.shape}")
    n = x.shape[-1]
    cap = int(round(max_seconds * sample_rate))
    if cap > 0 and n > cap:
        start = (n - cap) // 2
        x = x[:, start : start + cap]
    return x - x.mean(axis=-1, keepdims=True)


def _welch_psd(x: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """One-channel Welch PSD, ~2 Hz resolution with heavy (75%) averaging.

    A short-enough ``nperseg`` (power of two ≈ ``sr/2``) at 75% overlap yields
    many averaged segments, so the broadband floor is smooth — essential for the
    prominence gate to separate real harmonics from noise-floor fluctuations.
    """
    n = x.shape[-1]
    target = int(2 ** np.round(np.log2(max(sample_rate / 2.0, 2.0))))  # ~2 Hz res
    nperseg = int(min(target, n))
    nperseg = max(nperseg, 256)
    noverlap = (nperseg * 3) // 4
    freqs, psd = welch(x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def _estimate_f0(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float, n_hps: int) -> float:
    """Harmonic-product-spectrum fundamental in ``[fmin, fmax]`` (0.0 if none).

    Sub-bin refined by a parabolic fit around the HPS peak.
    """
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    if df <= 0:
        return 0.0
    hps = psd.copy()
    for h in range(2, n_hps + 1):
        dec = psd[::h]
        hps[: len(dec)] *= dec
    lo = max(int(np.ceil(fmin / df)), 1)
    hi = min(int(np.floor(fmax / df)), len(hps) - 2)
    if hi <= lo:
        return 0.0
    k = lo + int(np.argmax(hps[lo : hi + 1]))
    a, b, c = hps[k - 1], hps[k], hps[k + 1]
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > _EPS else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return float((k + delta) * df)


def _comb_metrics(
    freqs: np.ndarray, psd: np.ndarray, f0: float, rel_band: float, prom_db: float
) -> tuple[float, float, int]:
    """(harmonic_energy_ratio, harmonic_to_noise_db, n_prominent) for comb ``k·f0``.

    A harmonic contributes energy only if its band peak clears the local floor by
    ``prom_db`` — so a smooth noise floor (no real peaks) contributes ≈0.
    """
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    nyq = float(freqs[-1])
    total = float(np.sum(psd)) + _EPS
    if f0 <= 0 or df <= 0:
        return 0.0, -np.inf, 0
    prom_lin = 10.0 ** (prom_db / 10.0)
    harmonic_energy = 0.0
    n_prominent = 0
    k = 1
    while k * f0 <= nyq:
        centre = k * f0
        half = max(rel_band * centre, df)  # at least one bin
        lo = max(int(np.floor((centre - half) / df)), 0)
        hi = min(int(np.ceil((centre + half) / df)), len(psd) - 1)
        if hi >= lo:
            band = psd[lo : hi + 1]
            peak = float(band.max())
            # Broadband floor: median PSD in a wider neighbourhood, excluding
            # the harmonic band itself.
            wlo = max(lo - 3 * (hi - lo + 1), 0)
            whi = min(hi + 3 * (hi - lo + 1), len(psd) - 1)
            neigh = np.concatenate([psd[wlo:lo], psd[hi + 1 : whi + 1]])
            floor = float(np.median(neigh)) if len(neigh) else 0.0
            if peak > 0 and floor > 0 and peak >= prom_lin * floor:
                harmonic_energy += float(np.sum(np.clip(band - floor, 0.0, None)))
                n_prominent += 1
        k += 1
    harmonic_energy = min(harmonic_energy, total)
    ratio = float(np.clip(harmonic_energy / total, 0.0, 1.0))
    residual = total - harmonic_energy + _EPS
    hnr_db = float(10.0 * np.log10((harmonic_energy + _EPS) / residual))
    return ratio, hnr_db, n_prominent


def _spectral_flatness(psd: np.ndarray) -> float:
    """Wiener entropy geo-mean / arith-mean of the (positive) PSD, ∈ (0, 1]."""
    p = psd[psd > 0]
    if len(p) == 0:
        return 1.0
    geo = float(np.exp(np.mean(np.log(p))))
    arith = float(np.mean(p))
    return float(np.clip(geo / (arith + _EPS), 0.0, 1.0))


def measure_harmonicity(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = 10.0,
    fmax: float = 1000.0,
    n_hps: int = 5,
    rel_band: float = 0.03,
    prom_db: float = 6.0,
    max_seconds: float = 10.0,
) -> Harmonicity:
    """Measure harmonic prominence of ``audio`` (``(T,)`` or ``(C, T)``).

    Metrics are computed per channel and averaged (``f0`` by median). ``fmin``/
    ``fmax`` bound the *fundamental* search (harmonics extend to Nyquist);
    ``n_hps`` is the harmonic-product-spectrum depth; ``rel_band`` the fractional
    half-width of each harmonic band; ``max_seconds`` centre-crops long clips.

    A silent/constant clip yields ``f0_hz=0``, ``harmonic_energy_ratio=0``,
    ``spectral_flatness=1`` (noise-like), ``harmonic_to_noise_db=-inf``.
    """
    channels = _analysis_signal(audio, sample_rate, max_seconds)
    f0s: list[float] = []
    ratios: list[float] = []
    hnrs: list[float] = []
    n_proms: list[int] = []
    flats: list[float] = []
    for ch in channels:
        if not np.any(ch):
            f0s.append(0.0)
            ratios.append(0.0)
            hnrs.append(-np.inf)
            n_proms.append(0)
            flats.append(1.0)
            continue
        freqs, psd = _welch_psd(ch, sample_rate)
        f0 = _estimate_f0(freqs, psd, fmin, fmax, n_hps)
        ratio, hnr, n_prom = _comb_metrics(freqs, psd, f0, rel_band, prom_db)
        f0s.append(f0)
        ratios.append(ratio)
        hnrs.append(hnr)
        n_proms.append(n_prom)
        flats.append(_spectral_flatness(psd))
    finite_hnr = [h for h in hnrs if np.isfinite(h)]
    return Harmonicity(
        f0_hz=float(np.median(f0s)),
        harmonic_energy_ratio=float(np.mean(ratios)),
        harmonic_to_noise_db=float(np.mean(finite_hnr)) if finite_hnr else float("-inf"),
        n_prominent_harmonics=int(round(float(np.mean(n_proms)))),
        spectral_flatness=float(np.mean(flats)),
    )
