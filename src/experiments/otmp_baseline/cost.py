"""Frequency / pitch grids and the inverse-harmonic-clustering ground cost.

Reference: A. Björkman and F. Elvander, "Inverse Harmonic Clustering for
Multi-Pitch Estimation: An Optimal Transport Approach", IEEE TSP 2026
(arXiv:2508.02471). Equations quoted below are that paper's.

The one idea that lives here is eq (18):

    c(w, w0) = (1 / w0**2) * min_{k in Z+} (w - k * w0)**2
             = min_{k in Z+} (w / w0 - k)**2

i.e. the squared distance to the closest harmonic *measured in units of the
fundamental*. The unnormalized cost of eq (17) treats w0 and w0/2 alike on a
perturbed harmonic (both give the same absolute deviation), so an OT-based
estimator drifts to sub-octaves. Dividing by w0**2 makes the deviation of an
off-tooth partial cost four times more under w0/2 than under w0, so the
highest fundamental consistent with all partials wins.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ground_cost",
    "harmonic_cost_unnormalized",
    "linear_grid",
    "nearest_harmonic_order",
]


def linear_grid(lo: float, hi: float, step: float | None = None, n: int | None = None) -> NDArray:
    """Uniform grid on ``[lo, hi]``, given either a step or a point count.

    Exactly one of ``step`` / ``n`` must be given. With ``step``, ``hi`` is
    included only if it lands on the lattice (the last point is the largest
    ``lo + i*step <= hi``); with ``n``, both endpoints are included.
    """
    if (step is None) == (n is None):
        raise ValueError("give exactly one of step=, n=")
    if hi <= lo:
        raise ValueError(f"empty grid: lo={lo} hi={hi}")
    if n is not None:
        if n < 2:
            raise ValueError("n must be >= 2")
        return np.linspace(lo, hi, int(n), dtype=np.float64)
    assert step is not None
    if step <= 0:
        raise ValueError("step must be > 0")
    count = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(count, dtype=np.float64)


def nearest_harmonic_order(freqs: NDArray, pitches: NDArray) -> NDArray:
    """``k*[f, g] = argmin_{k in Z+} (freqs[f] / pitches[g] - k)**2``.

    ``Z+`` is the *positive* integers, so a frequency below half a candidate
    fundamental is still charged against ``k = 1`` (it cannot be explained
    away by ``k = 0``).
    """
    ratio = np.asarray(freqs, dtype=np.float64)[:, None] / np.asarray(pitches, dtype=np.float64)
    k = np.rint(ratio)
    np.clip(k, 1.0, None, out=k)
    return k


def ground_cost(freqs: NDArray, pitches: NDArray) -> NDArray:
    """The eq-(18) ground cost matrix ``C[f, g] = min_k (w_f / w0_g - k)**2``.

    ``freqs`` and ``pitches`` may be in any consistent unit (Hz or rad/sample)
    — the cost is a ratio and therefore unit-free. Shape ``(F, G)``.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    pitches = np.asarray(pitches, dtype=np.float64)
    ratio = freqs[:, None] / pitches
    k = np.clip(np.rint(ratio), 1.0, None)
    return (ratio - k) ** 2


def harmonic_cost_unnormalized(freqs: NDArray, pitches: NDArray) -> NDArray:
    """The eq-(17) cost ``min_k (w_f - k*w0_g)**2`` — kept for comparison only.

    This is what the proposed cost normalizes; it is octave-ambiguous by
    construction (``c_hat(w, w0/2) <= c_hat(w, w0)`` for every ``w``).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    pitches = np.asarray(pitches, dtype=np.float64)
    return ground_cost(freqs, pitches) * pitches**2
