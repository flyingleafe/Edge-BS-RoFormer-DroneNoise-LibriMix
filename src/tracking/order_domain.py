"""Order-domain comb analysis: does the acoustic comb sit on the telemetry?

Resample the audio uniformly in the telemetry rotor PHASE, then take the FFT.
The result is an ORDER spectrum. If the acoustic comb follows the telemetry
exactly, harmonic k lands on the integer order k. If the comb is displaced by a
constant factor s, harmonic k lands on order s*k — a fan that opens linearly
with k. Nothing in this module uses a peak-search window, a band gate, or a
collision gate, so no per-harmonic search can bias the answer.

A whole comb gets one score at a time:

    S(s) = mean over k in the band of  excess_dB(order = s * k)

If the comb sits on the telemetry, S peaks at s = 1. If the comb is displaced,
S peaks at that s. The score is a mean over MANY harmonics, so a spurious peak
needs all of them to conspire. As a result, the height of the peak over the
background of the score IS the significance.

Bands are scanned one at a time (low k, mid k, high k). Thus "does the comb
survive to k = 100" gets an answer band by band, and the fan of a scale error
cannot be confused with a fixed-frequency artefact.

The null is the same scan on a HALF-INTEGER comb, at orders s*(k + 0.5), where
no rotor line can exist. Both scans are identical searches, so the contest
between their two peaks is fair.

Short segments are necessary at high k. Spectral autocorrelation of a DREGON
cruise window shows the 172 Hz comb (the blade-passage rate, 2 x the 86 rev/s
shaft) alive at 5.5-6.5 kHz on 0.1 s segments and gone by 1 s. The high-k line
thus has a coherence time much shorter than a 16 s window, and a long window
averages it away. :func:`segment_comb_scan` splits the window into short
phase-resampled segments and averages their scores INCOHERENTLY. Both earlier
"no high-k line" readings were coherence-limited, and they are not evidence of
absence.

Pure array code: numpy and scipy only. Data loading, plots and CLIs stay
outside. The campaign driver is ``scripts/displacement/combscan.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

__all__ = [
    "DEFAULT_SPR",
    "OrderSpectrum",
    "comb_scan",
    "order_spectrum",
    "peak_orders",
    "scan_summary",
    "segment_comb_scan",
]

#: Samples per revolution of the phase-resampled signal. The order Nyquist is
#: half of this, so 1024 reaches order 512.
DEFAULT_SPR = 1024

#: Minimum width of the median floor filter, in bins: whole window / segment.
#: A short segment has a coarse order grid, so its floor needs fewer bins.
_FLOOR_MIN_WHOLE = 51
_FLOOR_MIN_SEGMENT = 11


@dataclass(frozen=True)
class OrderSpectrum:
    """One phase-resampled spectrum of a slice, with its telemetry rate."""

    orders: np.ndarray
    """Order axis: harmonics of the telemetry shaft rate."""

    db: np.ndarray
    """Power in dB over the in-band median."""

    excess_db: np.ndarray
    """``db`` minus a median-filtered floor, that is, the true excess in dB."""

    rate_mean: float
    """Mean telemetry rate over the slice, rev/s."""


def _as_2d(audio: np.ndarray) -> np.ndarray:
    """``(C, N)`` float64 view of the audio. A ``(N,)`` input is one channel."""
    a = np.asarray(audio, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def _slice_phase(
    t_tel: np.ndarray, rate: np.ndarray, sr: float, t0: float, t1: float
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Sample bounds, sample times, telemetry rate and cumulative phase.

    The phase is the number of revolutions elapsed since the start of the
    slice, integrated from the telemetry rate.
    """
    a0, a1 = int(t0 * sr), int(t1 * sr)
    t = np.arange(a0, a1) / sr
    r = np.interp(t, t_tel, rate)
    phi = np.cumsum(r) / sr
    phi -= phi[0]
    return a0, a1, t, r, phi


def _order_power(
    chunk: np.ndarray, t: np.ndarray, phi: np.ndarray, p0: float, n_out: int, spr: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(orders, power)`` of ``chunk`` resampled uniformly in rotor phase.

    The phase grid starts at ``p0`` and holds ``spr`` samples per revolution.
    ``np.interp`` inverts phi(t) to get the sample time of each phase point.
    The power is the mean over the channels of ``|rfft|^2``, that is, an
    INCOHERENT average: the microphones see the same rotor line with different
    phases, so a coherent sum can cancel it.
    """
    grid = p0 + np.arange(n_out) / spr
    t_at = np.interp(grid, phi, t)
    win = np.hanning(n_out)
    acc = None
    for c in range(chunk.shape[0]):
        y = np.interp(t_at, t, chunk[c])
        # Mean removal keeps the DC of the window out of the low orders.
        spec = np.abs(np.fft.rfft((y - y.mean()) * win)) ** 2
        acc = spec if acc is None else acc + spec
    orders = np.fft.rfftfreq(n_out, d=1.0 / spr)
    return orders, acc / chunk.shape[0]


def _floor(db: np.ndarray, orders: np.ndarray, floor_orders: float, min_size: int) -> np.ndarray:
    """Slowly varying dB floor: a median filter ``floor_orders`` wide.

    The ``| 1`` makes the width odd, so the filter has no half-bin bias.
    """
    n_sm = max(min_size, int(floor_orders / (orders[1] - orders[0])) | 1)
    return median_filter(db, size=n_sm, mode="nearest")


def order_spectrum(
    audio: np.ndarray,
    sr: float,
    t_tel: np.ndarray,
    rate: np.ndarray,
    t0: float,
    t1: float,
    *,
    spr: int = DEFAULT_SPR,
    floor_orders: float | None = 2.0,
) -> OrderSpectrum:
    """Order spectrum of ``audio`` over ``[t0, t1]``, resampled in rotor phase.

    ``t_tel`` and ``rate`` are the telemetry time base and the rate in rev/s of
    ONE rotor. ``floor_orders`` is the width of the median floor; ``None``
    keeps the raw dB, and then ``excess_db`` equals ``db``.

    Native 44.1 kHz audio reaches order 275 at 80 rev/s, so this also answers
    "how far up does the comb go".
    """
    chunk = _as_2d(audio)
    a0, a1, t, r, phi = _slice_phase(t_tel, rate, sr, t0, t1)
    chunk = chunk[:, a0:a1]
    n_out = int(float(phi[-1]) * spr)
    orders, p = _order_power(chunk, t, phi, 0.0, n_out, spr)
    db = 10.0 * np.log10(p / np.median(p) + 1e-30)
    if floor_orders is None:
        excess = db
    else:
        # The floor comes off the ABSOLUTE dB, not the median-normalized dB.
        # The two differ by one constant, and this order of the operations
        # keeps the excess bit-identical to the campaign's published numbers.
        db_abs = 10.0 * np.log10(p + 1e-30)
        excess = db_abs - _floor(db_abs, orders, floor_orders, _FLOOR_MIN_WHOLE)
    return OrderSpectrum(orders=orders, db=db, excess_db=excess, rate_mean=float(np.mean(r)))


def comb_scan(
    orders: np.ndarray,
    excess_db: np.ndarray,
    s_grid: np.ndarray,
    k_lo: int,
    k_hi: int,
    *,
    half: bool = False,
    f_limit: float | None = None,
    rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(score, count)`` of the comb k = k_lo..k_hi over the scale grid.

    ``half=True`` scans the half-integer comb k + 0.5 — the null, where no
    rotor line can exist. ``count`` is the number of harmonics that contribute
    to each scale point. A harmonic contributes if its order stays inside the
    axis and, when ``f_limit`` and ``rate`` are given, if ``k * rate`` stays
    below ``f_limit`` Hz.
    """
    s_grid = np.asarray(s_grid, dtype=np.float64)
    do = orders[1] - orders[0]
    o_max = orders[-1]
    ks = np.arange(k_lo, k_hi + 1, dtype=np.float64) + (0.5 if half else 0.0)
    score = np.zeros(len(s_grid))
    count = np.zeros(len(s_grid))
    for k in ks:
        o = s_grid * k
        ok = o < o_max
        if f_limit is not None and rate is not None:
            ok = ok & (k * rate < f_limit)
        # Nearest bin, clipped: the grid step is far finer than one bin, so the
        # scan reads a bin index, never an interpolated value.
        idx = np.clip(np.round(o / do).astype(int), 0, len(excess_db) - 1)
        score += np.where(ok, excess_db[idx], 0.0)
        count += ok
    return score / np.maximum(count, 1), count


def segment_comb_scan(
    audio: np.ndarray,
    sr: float,
    t_tel: np.ndarray,
    rate: np.ndarray,
    t0: float,
    t1: float,
    s_grid: np.ndarray,
    k_lo: int,
    k_hi: int,
    *,
    half: bool = False,
    seg_s: float = 0.25,
    spr: int = 512,
    floor_orders: float = 4.0,
    f_limit_frac: float = 0.45,
) -> tuple[np.ndarray, int]:
    """``(score, n_segments)``: the comb score averaged over SHORT segments.

    The window is cut into segments of ``seg_s`` seconds, each segment gets its
    own phase-resampled order spectrum, and the scores are averaged
    INCOHERENTLY. This is the coherence-time fix: the high-k comb decoheres in
    less than a second, so a 16 s spectrum averages it away.

    A 0.25 s segment holds about 20 revolutions, that is, an order resolution
    of 0.05. The high-k displacement is 0.0054 x 75 = 0.4 orders, so the effect
    stays measurable at this resolution.
    """
    chunk = _as_2d(audio)
    a0, a1, t, r, phi = _slice_phase(t_tel, rate, sr, t0, t1)
    chunk = chunk[:, a0:a1]
    n_seg = int(seg_s * sr)
    f_limit = f_limit_frac * sr
    acc = np.zeros(len(np.asarray(s_grid)))
    n_used = 0
    for s0 in range(0, (a1 - a0) - n_seg, n_seg):
        ph = phi[s0 : s0 + n_seg]
        n_out = int(float(ph[-1] - ph[0]) * spr)
        if n_out < 64:
            continue
        orders, p = _order_power(chunk, t, phi, float(ph[0]), n_out, spr)
        db = 10.0 * np.log10(p + 1e-30)
        excess = db - _floor(db, orders, floor_orders, _FLOOR_MIN_SEGMENT)
        r_bar = float(np.mean(r[s0 : s0 + n_seg]))
        score, count = comb_scan(
            orders, excess, s_grid, k_lo, k_hi, half=half, f_limit=f_limit, rate=r_bar
        )
        if count.max() < 1:
            continue
        acc += score
        n_used += 1
    return acc / max(n_used, 1), n_used


def scan_summary(
    s_grid: np.ndarray, score: np.ndarray, *, null: np.ndarray | None = None
) -> dict[str, float]:
    """Peak location and peak height of one comb scan.

    ``peak_over_bg_db`` is the peak over the MEDIAN of the score, and ``z``
    divides it by the 90-10 percentile spread of the score.

    ``peak_over_null_db`` is max(on) - max(null): a fair contest between two
    IDENTICAL searches. The earlier peak-minus-own-median statistic was wrong —
    the on-comb peak is broad (it spans most of the +-1.5 % grid), so its own
    median sits inside the peak and the excess collapses to nothing. Read
    ``peak_over_bg_db`` for a narrow peak and ``peak_over_null_db`` for a broad
    one.
    """
    s_grid = np.asarray(s_grid, dtype=np.float64)
    j = int(np.argmax(score))
    s_hat = float(s_grid[j])
    peak = float(score[j])
    bg = float(np.median(score))
    spread = float(np.percentile(score, 90) - np.percentile(score, 10))
    out = {
        "s_hat": round(s_hat, 6),
        "pct": round((s_hat - 1.0) * 100, 4),
        "peak_db": round(peak, 3),
        "background_db": round(bg, 3),
        "peak_over_bg_db": round(peak - bg, 3),
        "z": round((peak - bg) / max(spread, 1e-9), 2),
    }
    if null is not None:
        out["peak_over_null_db"] = round(peak - float(np.max(null)), 3)
        out["mean_over_null_db"] = round(float(np.mean(score)) - float(np.mean(null)), 3)
    return out


def peak_orders(
    orders: np.ndarray, db: np.ndarray, k_lo: int, k_hi: int, *, tol: float = 0.35
) -> list[tuple[int, float, float, float]]:
    """Per-harmonic peak fan: ``(k, peak order, prominence dB, order / k)``.

    For each integer k the peak is the maximum of ``db`` inside ``+-tol``
    orders of k, and the prominence is that maximum over the median of a
    ``+-0.5`` order neighbourhood. The last column is the displacement of that
    one harmonic: it stays at 1 when the comb sits on the telemetry, and it
    stays at a constant other value when the comb is displaced by a scale.
    """
    rows: list[tuple[int, float, float, float]] = []
    for k in range(k_lo, k_hi + 1):
        m = np.abs(orders - k) <= tol
        if m.sum() < 5:
            continue
        j = int(np.argmax(db[m]))
        sub_o, sub_d = orders[m], db[m]
        mf = np.abs(orders - k) <= 0.5
        floor = float(np.median(db[mf]))
        rows.append((k, float(sub_o[j]), float(sub_d[j] - floor), float(sub_o[j] / k)))
    return rows
