"""Welch cross-spectral density of a multichannel signal.

Returns the per-segment spectra as well as the average, because the segment
axis is what the bootstrap in :mod:`.fit` resamples. Scaling matches
``scipy.signal.csd(..., scaling="density")``, so ``diag(R)`` is exactly
``scipy.signal.welch`` and the units are power per Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Csd", "welch_csd", "coherence", "band_edges", "band_average"]


@dataclass
class Csd:
    """Welch cross-spectra of an ``(M, T)`` signal."""

    freqs: np.ndarray  # (F,)
    segments: np.ndarray  # (L, M, F) complex — windowed, scaled segment spectra
    fs: float
    nperseg: int
    noverlap: int

    @property
    def n_seg(self) -> int:
        return int(self.segments.shape[0])

    def matrix(self, seg_index: np.ndarray | None = None, chunk: int = 64) -> np.ndarray:
        """``(F, M, M)`` Hermitian CSD averaged over the selected segments.

        Accumulated in segment chunks: a bootstrap draw fancy-indexes the
        segment axis, and materialising that copy (plus its conjugate) for a
        long recording is what blows the memory budget.
        """
        idx = np.arange(self.n_seg) if seg_index is None else np.asarray(seg_index)
        n_mic, n_f = self.segments.shape[1], self.segments.shape[2]
        out = np.zeros((n_f, n_mic, n_mic), dtype=np.complex128)
        for a in range(0, len(idx), chunk):
            s = self.segments[idx[a : a + chunk]]
            # R[f,m,n] += sum_l s[l,m,f] conj(s[l,n,f])
            out += np.einsum("lmf,lnf->fmn", s, s.conj())
        return out / len(idx)


def welch_csd(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 4096,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: bool = True,
) -> Csd:
    """Segment ``x`` ``(M, T)`` and return its :class:`Csd`."""
    from scipy.signal import get_window

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be (M, T), got {x.shape}")
    n_mic, n_t = x.shape
    noverlap = int(round(nperseg * overlap))
    step = nperseg - noverlap
    n_seg = 1 + (n_t - nperseg) // step
    if n_seg < 2:
        raise ValueError(f"need >=2 segments, got {n_seg} (T={n_t}, nperseg={nperseg})")

    win = np.asarray(get_window(window, nperseg), dtype=np.float64)
    scale = 1.0 / (fs * (win**2).sum())  # density scaling

    starts = np.arange(n_seg) * step
    # (L, M, nperseg) view -> spectra. 500 segments x 8 mics x 4096 float64 = 130 MB;
    # build the rfft segment by segment to keep the peak modest.
    n_f = nperseg // 2 + 1
    segs = np.empty((n_seg, n_mic, n_f), dtype=np.complex128)
    for li, a0 in enumerate(starts):
        blk = x[:, a0 : a0 + nperseg]
        if detrend:
            blk = blk - blk.mean(axis=-1, keepdims=True)
        segs[li] = np.fft.rfft(blk * win, axis=-1)
    # one-sided density: double every bin except DC and Nyquist
    segs *= np.sqrt(scale)
    dbl = np.full(n_f, np.sqrt(2.0))
    dbl[0] = 1.0
    if nperseg % 2 == 0:
        dbl[-1] = 1.0
    segs *= dbl

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return Csd(freqs=freqs, segments=segs, fs=fs, nperseg=nperseg, noverlap=noverlap)


def coherence(R: np.ndarray) -> np.ndarray:
    """Magnitude-squared coherence ``(F, M, M)`` from a CSD matrix."""
    d = np.real(np.einsum("fmm->fm", R))
    denom = np.sqrt(np.maximum(d[:, :, None] * d[:, None, :], 1e-300))
    return (np.abs(R) ** 2) / (denom**2)


def band_edges(f_lo: float, f_hi: float, n_bands: int, log: bool = True) -> np.ndarray:
    """``(n_bands+1,)`` band edges."""
    if log:
        return np.geomspace(max(f_lo, 1e-6), f_hi, n_bands + 1)
    return np.linspace(f_lo, f_hi, n_bands + 1)


def band_average(freqs: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Average ``values`` ``(F, ...)`` inside each band; NaN for empty bands."""
    out = np.full((len(edges) - 1, *values.shape[1:]), np.nan)
    for b in range(len(edges) - 1):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if m.any():
            out[b] = values[m].mean(axis=0)
    return out
