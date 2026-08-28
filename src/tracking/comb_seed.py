"""Blind rotor-speed seeding by a peeled Whittle comb scan.

WHY THIS EXISTS. Recovering rotor speeds from a static comb splits into a
search problem and a precision problem, and only the second one was solved. The
phase-increment refiner reaches 0.002 rev/s on a four-rotor static comb from a
good initialization, but its capture range is the demod band (0.35 rev/s at the
default), and the blind seed it was being handed carried about 2.7 rev/s of
error. Everything downstream was therefore working outside its capture range,
which is why iterating the refiner made the blind ladder WORSE rather than
better.

THE SCORE. Over a window short enough that a rotor's speed is nearly constant,
a comb at rate `r` puts energy in every bin `k r`. The natural score is the
Whittle log-likelihood of a line-plus-noise model summed over harmonics,

    S(r) = (1 / K) sum_k log(1 + Y(k r) / sigma^2)

with `Y` the periodogram and `sigma^2` its median. The logarithm is what makes
this work. A plain harmonic SUM is maximized by the half-rate `r / 2`, which
covers every true line and pockets the noise in the odd bins as a bonus;
measured on a four-rotor comb, `r / 2` outscored the truth and took rank 1. Under
the log, an empty bin contributes zero instead of noise, so the half-rate keeps
only its half of the harmonics and drops behind. That single change moved the
three true rotors from ranks 2, 4 and beyond to ranks 1, 2 and 3.

PEELING. The strongest comb masks the weaker ones, so after each pick its lines
are notched out of the periodogram and the scan repeats. Four peels take
coverage within 0.5 rev/s from 69% to 82%, and eight take it to 90%.

WHAT THIS IS NOT. The window scan assumes a constant rate, so it is a SEED, not
an answer. It exists to land inside the phase-increment refiner's capture range,
where the precision already measured takes over.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

_BIG = 1e9

__all__ = ["comb_score", "local_floor", "octave_correct", "peel_scan", "seed_tracks"]


def _periodogram(y: np.ndarray, sr: float, over: int = 8):
    n = len(y)
    nfft = 1 << int(np.ceil(np.log2(max(n * over, 16))))
    pw = np.abs(np.fft.rfft(y * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return pw, f, float(np.median(pw[f > 200.0])), sr / n


def local_floor(pw: np.ndarray, f: np.ndarray, span_hz: float = 120.0) -> np.ndarray:
    """Running-median floor of the periodogram, for a LOCAL line test.

    An absolute threshold cannot say whether a bin holds a line when four dense
    combs are present: most bins then clear it, harmonic presence saturates,
    and the octave test walks a pick down to a third of its true rate. Against
    a local median a real line still stands out and the bins between do not.
    """
    w = max(3, int(round(span_hz / max(f[1] - f[0], 1e-9))) | 1)
    return np.maximum(median_filter(pw, size=w, mode="nearest"), 1e-300)


def comb_score(
    pw: np.ndarray, f: np.ndarray, noise: float, grid: np.ndarray,
    k_max: int = 40, f_max: float = 7500.0,
) -> np.ndarray:
    """Whittle comb score over a rate grid: ``(len(grid),)``, larger is better."""
    sc = np.zeros(len(grid))
    cnt = np.zeros(len(grid))
    for k in range(1, k_max + 1):
        fk = k * grid
        m = fk < f_max
        if not m.any():
            break
        sc[m] += np.log1p(np.interp(fk[m], f, pw) / noise)
        cnt[m] += 1.0
    return sc / np.maximum(cnt, 1.0)


def _refine(grid: np.ndarray, sc: np.ndarray, p: int) -> float:
    """Parabolic sub-grid peak location."""
    if p <= 0 or p >= len(sc) - 1:
        return float(grid[p])
    a, b, c = sc[p - 1], sc[p], sc[p + 1]
    if not (np.isfinite(a) and np.isfinite(c)):
        return float(grid[p])   # a barred neighbour carries no curvature
    den = a - 2.0 * b + c
    d = 0.0 if abs(den) < 1e-30 else 0.5 * (a - c) / den
    return float(grid[p] + np.clip(d, -1.0, 1.0) * (grid[1] - grid[0]))


#: Small-integer rate ratios a comb can be confused with: the half rate covers
#: every line it has, and 2/3 or 3/2 of it covers two thirds of them.
_ALIAS_RATIOS = (0.5, 1 / 3, 2 / 3, 1.5, 2.0, 3.0, 0.25, 4.0, 3 / 4, 4 / 3)


def _presence(
    pw: np.ndarray, f: np.ndarray, floor: np.ndarray, r: float,
    k_max: int, f_max: float, thresh: float,
) -> float:
    """Fraction of a comb's harmonics that stand above the LOCAL floor."""
    ks = np.arange(1, k_max + 1, dtype=float)
    fk = ks * r
    m = fk < f_max
    if not m.any():
        return 0.0
    return float(np.mean(np.interp(fk[m], f, pw) > thresh * np.interp(fk[m], f, floor)))


def _odd_even(
    pw: np.ndarray, f: np.ndarray, floor: np.ndarray, r: float,
    k_max: int, f_max: float,
) -> float:
    """Ratio of the odd-harmonic level to the even-harmonic level of a comb.

    If `r` is really `r_true / 2`, its EVEN harmonics are `r_true`'s comb and
    its ODD ones fall between the true lines, so the ratio is small. If `r` is
    the true rate, both sets are its own lines and the ratio is near one. This
    is a ratio of two quantities measured the same way, so it needs no absolute
    threshold — which is what makes it usable. The absolute-presence test it
    replaces sat on a knife edge (0.70 against a 0.75 cut) and flipped between
    the right and the wrong answer with the peel order.
    """
    ks = np.arange(1, k_max + 1, dtype=float)
    fk = ks * r
    m = fk < f_max
    if m.sum() < 4:
        return 1.0
    lev = np.log1p(np.interp(fk[m], f, pw) / np.interp(fk[m], f, floor))
    odd = lev[::2]
    even = lev[1::2]
    if even.size == 0 or float(np.mean(even)) <= 1e-9:
        return 1.0
    return float(np.mean(odd) / np.mean(even))


def octave_correct(
    pw: np.ndarray, f: np.ndarray, floor: np.ndarray, r: float,
    k_max: int = 24, f_max: float = 7500.0, ratio: float = 0.6,
    max_mult: int = 3, r_hi: float = np.inf,
) -> float:
    """Raise a pick off a subharmonic, using the odd-to-even level ratio.

    A comb at `r` puts energy at every `k r`, so the SUBHARMONIC `r / m` covers
    all of the same lines and pockets whatever sits in the bins between. The
    Whittle score does not always reject it: it averages over `k_max`
    harmonics, and when a rotor's comb dies out before `k_max` the true
    fundamental's mean is diluted by empty high bins until the subharmonic ties.
    Measured on a four-rotor comb, the third peel was the half-rate of the third
    rotor in EVERY window, and that rotor was never found directly.

    The test doubles `r` while its odd harmonics are weak relative to its even
    ones — the signature of a half-rate — and stops as soon as they are
    comparable. A MULTIPLE of the true rate cannot be reached this way, because
    at the true rate the ratio is already near one.

    `r_hi` bounds the result by the caller's search range. Without it a doubling
    can overshoot to a rate no rotor could have (measured: 140 and 152 rev/s
    against a 30-100 search band), and those few windows dominate the mean
    error even when the median is fine.
    """
    cur = r
    for _ in range(max_mult):
        if _odd_even(pw, f, floor, cur, k_max, f_max) >= ratio:
            break
        if 2.0 * cur > r_hi:
            break  # a rate outside the search range is not a candidate answer
        cur = 2.0 * cur
    return cur


def peel_scan(
    y: np.ndarray, sr: float, grid: np.ndarray, n_src: int = 8,
    k_max: int = 40, f_max: float = 7500.0, notch: float = 1.5,
    exclude: float = 0.0, alias_exclude: float = 0.0, octave: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Peeled comb scan of one window: ``(rates, scores)``, strongest first.

    After each pick the comb's lines are notched out of the periodogram AND a
    neighbourhood of the picked rate is barred from the score. The second step
    is what makes the peels distinct: a spectral notch never removes a line
    completely, so the residual re-peaks a few tenths of a rev/s away and the
    next peel returns the SAME rotor. Measured without the rate exclusion, four
    peels of a four-rotor comb returned three rotors and one duplicate, and the
    missing rotor only appeared at peel five or later.

    ``exclude`` (rev/s) bars the neighbourhood of a picked rate;
    ``alias_exclude`` bars the small-integer multiples and fractions of it,
    which score well because they cover a subset of the same lines.

    BOTH DEFAULT TO ZERO, because on trajectories that cross they do more harm
    than good: two rotors pass within 0.6 rev/s of each other often, and an
    exclusion wide enough to suppress a duplicate also forbids the second rotor
    of a close pair. Measured with ``exclude=0.6`` the four-peel set error rose
    from 2.01 to 2.58 rev/s. Turn them on only when the rotors are known to
    stay apart.
    """
    pw, f, noise, df = _periodogram(y, sr)
    floor = local_floor(pw, f)
    pw = pw.copy()
    open_ = np.ones(len(grid), dtype=bool)
    rates, scores = [], []
    for _ in range(n_src):
        sc = comb_score(pw, f, noise, grid, k_max, f_max)
        sc = np.where(open_, sc, -np.inf)
        if not np.isfinite(sc).any():
            break
        p = int(np.argmax(sc))
        r = _refine(grid, sc, p)
        if octave:
            r = octave_correct(pw, f, floor, r, f_max=f_max, r_hi=float(grid[-1]))
        rates.append(r)
        scores.append(float(sc[p]))
        if exclude > 0:
            open_ &= np.abs(grid - r) >= exclude
        if alias_exclude > 0:
            for q in _ALIAS_RATIOS:
                open_ &= np.abs(grid - r * q) >= alias_exclude
        for k in range(1, k_max + 1):
            fc = k * r
            if fc >= f_max:
                break
            pw[np.abs(f - fc) < notch * df] = noise
    return np.asarray(rates), np.asarray(scores)


def seed_tracks(
    y: np.ndarray, sr: float, ft: np.ndarray, n_rot: int,
    r_lo: float = 30.0, r_hi: float = 100.0, d_grid: float = 0.02,
    win_s: float = 0.25, hop_s: float = 0.125, n_src: int = 8,
    k_max: int = 40, f_max: float = 7500.0, max_slew: float = 10.0,
    slew_cost: float = 3.0, gate_slew: float = 3.0,
) -> np.ndarray:
    """Blind rotor tracks on the frame grid ``ft``: ``(n_rot, len(ft))`` rev/s.

    Scans overlapping windows, then chains the per-window candidates into
    tracks by dynamic programming — one track at a time, strongest first, each
    consuming the candidates it uses. ``max_slew`` (rev/s per second) is the
    scale at which a rate change costs ``slew_cost`` units of score; it is a
    soft cost, not a gate, so a rotor may slew faster when the evidence pays
    for it.
    """
    grid = np.arange(r_lo, r_hi, d_grid)
    n = int(round(win_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    starts = list(range(0, max(1, len(y) - n + 1), hop))
    tc = np.array([(s + n / 2) / sr for s in starts])
    cand, csc = [], []
    for s in starts:
        r, sc = peel_scan(y[s : s + n], sr, grid, n_src, k_max, f_max)
        cand.append(r)
        csc.append(sc)

    # One-to-one assignment per window, not a greedy per-track search. The
    # candidates of a window are essentially the R true rotors plus junk, so
    # the structure of the problem is a matching. A greedy DP that fits one
    # track at a time lets a track ride whichever rotor is loudest, which is
    # smooth and high-scoring and completely wrong; measured, that cost 1.88
    # rev/s against the 0.28 this matching reaches on the same candidates.
    n_w = len(starts)
    tracks = np.empty((n_rot, n_w))
    order = np.argsort(-csc[0])[:n_rot]
    tracks[:, 0] = np.sort(cand[0][order])
    vel = np.zeros(n_rot)
    for w in range(1, n_w):
        dtw = max(tc[w] - tc[w - 1], 1e-9)
        pred = tracks[:, w - 1] + vel * dtw
        c = cand[w]
        # Cost: rate change against the predicted position, in slew units, less
        # the candidate's own score. A candidate the tracker cannot reach is
        # simply expensive, so a track with no good candidate holds its
        # prediction instead of jumping.
        cost = ((c[None, :] - pred[:, None]) / (max_slew * dtw)) ** 2 - csc[w][None, :] / slew_cost
        ri, ci = linear_sum_assignment(cost)
        new_r = pred.copy()
        for a_, b_ in zip(ri, ci):
            if abs(c[b_] - pred[a_]) <= gate_slew * max_slew * dtw:
                new_r[a_] = c[b_]
        vel = 0.5 * vel + 0.5 * (new_r - tracks[:, w - 1]) / dtw
        tracks[:, w] = new_r
    return np.stack([np.interp(ft, tc, t) for t in tracks])
