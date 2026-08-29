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

__all__ = [
    "comb_gram",
    "comb_score",
    "local_floor",
    "octave_correct",
    "peel_scan",
    "seed_from_gram",
    "seed_tracks",
]


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


# ---------------------------------------------------------------------------
# the comb-gram: track ridges in the score surface, do not threshold early


def comb_gram(
    y: np.ndarray, sr: float, grid: np.ndarray, win_s: float = 0.25,
    hop_s: float = 0.125, k_max: int = 40, f_max: float = 7500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Whittle comb score over (window, rate): ``(S, t_centres)``.

    The per-window peel scan throws away everything but its handful of peaks,
    and that is where it loses rotors: in 16% of windows one rotor's comb is too
    weak to reach the top four, a junk candidate takes its place, and those
    windows alone set the aggregate error. A rotor that is momentarily weak
    still leaves a CONTINUOUS ridge in the score surface, so tracking the
    surface recovers it from its own history instead of re-detecting it from
    scratch in every window.
    """
    n = int(round(win_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    starts = list(range(0, max(1, len(y) - n + 1), hop))
    rows = []
    for st in starts:
        pw, f, noise, _ = _periodogram(y[st : st + n], sr)
        rows.append(comb_score(pw, f, noise, grid, k_max, f_max))
    tc = np.array([(st + n / 2) / sr for st in starts])
    return np.stack(rows), tc


def _viterbi_ridge(
    S: np.ndarray, grid: np.ndarray, slew: float, dt: float, stiff: float = 1.0
) -> np.ndarray:
    """Best-scoring smooth path through a score surface: indices per window.

    The penalty must be large, and it must be a HINGE rather than a plain
    quadratic. With a penalty of order the score contrast, the optimal path does
    not follow a rotor at all: it HOPS between rotors and alias ridges
    collecting each window's maximum. Measured on a four-rotor clip after two
    peels, the wandering path scored 1.62 per window against 0.87 for the true
    rotor, so the Viterbi returned it — correctly, for the wrong objective.

    But a quadratic stiff enough to stop a hop also blocks the legitimate
    motion of a fast rotor: it improved two regimes (0.0374 -> 0.0326 and
    2.54 -> 1.75) and destroyed the third, where rotors genuinely slew hard
    (0.554 -> 5.27). Physical slew and a rotor hop differ by nearly an order of
    magnitude — about 4 rev/s^2 against 24 — so the cost is FREE up to `slew`
    and steep past it.
    """
    step_free = max(slew * dt / (grid[1] - grid[0]), 1e-9)
    span = max(1, int(round(4.0 * step_free)))
    offs = np.arange(-span, span + 1)
    excess = np.maximum(np.abs(offs.astype(float)) - step_free, 0.0)
    pen = stiff * (excess / step_free) ** 2
    n_w, n_g = S.shape
    best = S[0].copy()
    back = np.zeros((n_w, n_g), dtype=np.int32)
    for w in range(1, n_w):
        # For each destination, the best source within +-span.
        cand = np.full((len(offs), n_g), -np.inf)
        for a, o in enumerate(offs):
            src = np.roll(best, o)
            if o > 0:
                src[:o] = -np.inf
            elif o < 0:
                src[o:] = -np.inf
            cand[a] = src - pen[a]
        a_best = np.argmax(cand, axis=0)
        back[w] = np.arange(n_g) - offs[a_best]
        best = cand[a_best, np.arange(n_g)] + S[w]
    path = np.empty(n_w, dtype=np.int64)
    path[-1] = int(np.argmax(best))
    for w in range(n_w - 1, 0, -1):
        path[w - 1] = back[w][path[w]]
    return path


def seed_from_gram(
    y: np.ndarray, sr: float, ft: np.ndarray, n_rot: int,
    r_lo: float = 30.0, r_hi: float = 100.0, d_grid: float = 0.02,
    win_s: float = 0.25, hop_s: float = 0.125, k_max: int = 40,
    f_max: float = 7500.0, slew: float = 12.0, notch: float = 1.5,
    stiff: float = 40.0, n_refine: int = 0, n_restart: int = 1,
    octave: bool = True,
) -> np.ndarray:
    """Blind rotor tracks by peeling ridges out of the comb-gram.

    Each track is the best-scoring smooth path through the score surface. The
    track's comb is then notched OUT OF THE SPECTRUM of every window and the
    surface is rescored, so the next path cannot be the same rotor again.

    Suppressing the surface in rate space instead does not work: a strong
    rotor's comb score has sidelobes wider than any exclusion narrow enough to
    keep a close pair apart, so the second path lands beside the first.
    Measured that way, tracks one and three were the same rotor (64.80 and
    64.81 against true rates 64.81, 71.85, 78.23, 84.88).
    """
    grid = np.arange(r_lo, r_hi, d_grid)
    n = int(round(win_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    starts = list(range(0, max(1, len(y) - n + 1), hop))
    tc = np.array([(st + n / 2) / sr for st in starts])
    dt = float(tc[1] - tc[0]) if len(tc) > 1 else 1.0

    pws, fs, noises, dfs, floors = [], None, [], None, []
    for st in starts:
        pw, f, noise, df = _periodogram(y[st : st + n], sr)
        pws.append(pw.copy())
        floors.append(local_floor(pw, f))
        noises.append(noise)
        fs, dfs = f, df
    f, df = fs, dfs

    def notch_into(dest, r_path):
        for w in range(len(starts)):
            for k in range(1, k_max + 1):
                fc = k * r_path[w]
                if fc >= f_max:
                    break
                dest[w][np.abs(f - fc) < notch * df] = noises[w]

    def find(exclude, band=None):
        """Best path with every track in `exclude` notched out of the spectra.

        `band` restricts the path to a rate range, which is how a restart is
        forced to begin on a different rotor.
        """
        work = [pw.copy() for pw in pws]
        for r_path in exclude:
            notch_into(work, r_path)
        S = np.stack([
            comb_score(work[w], f, noises[w], grid, k_max, f_max) for w in range(len(starts))
        ])
        if band is not None:
            S = S.copy()
            S[:, (grid < band[0]) | (grid >= band[1])] = -1e9
        r_path = grid[_viterbi_ridge(S, grid, slew, dt, stiff)]
        return _octave_path(work, f, floors, r_path, k_max, f_max, r_hi) if octave else r_path

    def joint_score(tset) -> float:
        """Whittle evidence over the UNION of a solution's lines. Higher is better.

        The objective that ranks whole solutions, and the part two earlier
        attempts got wrong in opposite directions. Residual energy after notching
        rewards notching wherever energy happens to be rather than where combs
        are, so it prefers a solution parked on loud junk. Per-line MEAN evidence
        fixes that and breaks the other way: four tracks stacked on one loud
        rotor score perfectly, because every line they claim is real. Counting
        each spectral bin ONCE removes both incentives — a duplicate track adds
        no bins and so gains nothing, while covering a rotor nobody else covers
        adds all of its lines.

        It is the right objective of the three and it is still not sufficient.
        With 8 restarts it takes the fast-slew regime to 0/12 failures (0.065
        rev/s, worst clip 0.119, against 0.092 and 1/12 for the plain greedy
        sweep), but the regime where trajectories interleave does not improve
        under ANY of the three objectives or ANY restart count. That regime's
        failure is therefore not a search problem and not a scoring problem.
        """
        tot = 0.0
        for w in range(len(starts)):
            covered = np.zeros(len(f), dtype=bool)
            for r_path in tset:
                for k in range(1, k_max + 1):
                    fc = k * r_path[w]
                    if fc >= f_max:
                        break
                    covered |= np.abs(f - fc) < notch * df
            if covered.any():
                tot += float(np.log1p(pws[w][covered] / floors[w][covered]).sum())
        return tot

    def greedy(first_band) -> list[np.ndarray]:
        tset: list[np.ndarray] = []
        for i in range(n_rot):
            tset.append(find(tset, band=first_band if i == 0 else None))
        return tset

    # Restart the greedy sweep from each band of the rate range and keep the
    # solution that explains the most signal. The greedy sweep commits to its
    # first track before the others have a say, and where rotors cross that
    # commitment is what strands the rest; coordinate descent cannot undo it
    # (it converges inside the same basin), so the escape has to be a different
    # STARTING basin. Forcing the first track into each band in turn guarantees
    # a restart that begins on each rotor, ranked by `joint_score`.
    bands = [None] if n_restart <= 1 else [
        (r_lo + j * (r_hi - r_lo) / n_restart, r_lo + (j + 1) * (r_hi - r_lo) / n_restart)
        for j in range(n_restart)
    ]
    sols = [greedy(b) for b in bands]
    tracks = max(sols, key=joint_score)

    # Coordinate descent on the joint problem. The greedy sweep above commits to
    # track 1 before track 2 has had a say, and where two rotors CROSS the first
    # path may follow either branch; its notch then removes part of one rotor and
    # part of the other, leaving the rest as ridges that belong to nobody. The
    # measured symptoms were a rotor missed with a spurious track below the
    # ensemble, and two tracks splitting the difference between two crossing
    # rotors — in both cases with correct per-track standard deviation, so the
    # tracks were following motion and only their identity was wrong.
    #
    # IT DOES NOT WORK, and `n_refine` therefore defaults to 0. Re-solving each
    # track against the others converges after ONE pass (n_refine 1, 2 and 4 all
    # give 0.025 / 0.097 / 0.189) and leaves the total failure count where the
    # greedy sweep left it, 3 clips in 36 — it only moves a failure from one
    # regime to another, at three to five times the runtime. That is what
    # coordinate descent from a greedy initialization does: it finds the nearest
    # fixed point, which is inside the basin the greedy sweep already chose.
    # Escaping it needs a genuinely joint search (k-best paths per track, then
    # an assignment over the combinations), not a better local step.
    for _ in range(n_refine):
        for i in range(n_rot):
            tracks[i] = find([t for j, t in enumerate(tracks) if j != i])

    return np.stack([np.interp(ft, tc, t) for t in tracks])


def _octave_path(
    pws: list[np.ndarray], f: np.ndarray, floors: list[np.ndarray],
    r_path: np.ndarray, k_max: int, f_max: float, r_hi: float,
    ratio: float = 0.6, max_mult: int = 3,
) -> np.ndarray:
    """Octave-correct a whole track by AGGREGATING its odd-to-even evidence.

    Two things were wrong with deciding per window and voting. A rotor does not
    change its blade count mid-clip, so the decision belongs to the track, and a
    single window's odd-to-even ratio is noisy enough that the vote fails on
    about one rotor in twenty — measured, one track of one clip in five sat at
    35.96 rev/s against a true 71.86, and that single track produced the whole
    of a 1.82 rev/s average set error. Accumulating the harmonic levels over
    every window of the track first, and taking one ratio at the end, uses all
    the evidence the track has.

    The levels must also be read from the PEELED spectra — the ones the ridge
    was actually found in — not from the original signal.
    """
    cur = r_path.copy()
    for _ in range(max_mult):
        odd_t, even_t = [], []
        for w in range(len(pws)):
            ks = np.arange(1, k_max + 1, dtype=float)
            fk = ks * cur[w]
            m = fk < f_max
            if m.sum() < 4:
                continue
            lev = np.log1p(np.interp(fk[m], f, pws[w]) / np.interp(fk[m], f, floors[w]))
            odd_t.append(lev[::2].mean())
            even_t.append(lev[1::2].mean() if lev[1::2].size else 0.0)
        if not odd_t:
            break
        ev = float(np.mean(even_t))
        if ev <= 1e-9 or float(np.mean(odd_t)) / ev >= ratio:
            break
        if float(cur.max()) * 2.0 > r_hi:
            break
        cur = cur * 2.0
    return cur
