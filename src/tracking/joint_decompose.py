"""The JOINT decomposition (v3): one alternation of three linear-Gaussian blocks.

The v2 decomposition (:mod:`tracking.decompose`) splits a recording into
per-(rotor, harmonic, microphone) Vold-Kalman envelopes plus a per-microphone
residual, under two silent assumptions: that every timing deviation fits through
one fixed envelope bandwidth, and that the leftover is WHITE. Both are wrong.
A shaft wanders by about 0.6 rev/s, so the §identity ``k (phi + theta)`` widens
harmonic ``k`` to about ``0.6 k`` Hz — 24 Hz at ``k`` 40 against a 3 Hz band —
and the flanks of every line become "residual" by construction. Drone noise is
also strongly colored, so an unweighted least squares buys forgiveness for comb
structure exactly where the floor is loud.

The model this module estimates is::

    y_c(t) = sum_{r,k} Re[ g_{r,k,c}(t) e^{j(k phi_r(t) + k theta_r(t)
                                            + psi_{r,k}(t))} ] + n_c(t)

with ``phi_r`` the annotated shaft phase, ``theta_r`` a SLOW coherent shaft
correction (a rig-common part plus a small per-rotor part), ``psi_{r,k}`` a slow
per-track phase correction, ``g`` the residual envelope — which now only needs
AMPLITUDE bandwidth — and ``n_c`` colored noise with a smooth log spectrum
``S_c(f, t)``. There is no per-microphone arrival term: it was measured and it
is not there.

Three blocks, alternated (:func:`joint_solve_window`), which is block-coordinate
descent on one MAP objective — every block is linear-Gaussian given the others:

**Block A — the whitened VK solve** (:func:`whiten_weights`, then the existing
solver). The coherent phases fold into the carrier, which is an exact
reparametrization, and the whitening collapses to ONE scalar per (track, time)
because ``S`` is smooth and a line is narrow — so the banded structure of the
solver survives untouched. The three hooks in
:func:`tracking.vk_tracking.vk_envelopes` are the whole seam.

**Block B — the phase split** (:func:`split_phases`). The solved envelope phases
are unwrapped, combined over microphones, and regressed against ``k``:
``theta`` is the ``k``-weighted mean of ``arg g / k`` over the currently
trustable tracks, smoothed by a Whittaker-Henderson smoother
(:func:`wh_smooth`) — which IS a one-dimensional VK, so its weight comes from
the same Tuma relation. What is left per track is ``psi``, smoothed with a
wider band at higher ``k``.

**Block C — the masked smooth floor** (:func:`masked_smooth_psd`). Welch the
current residual with every predicted comb line masked, per short frame so the
mask follows a moving line, then fit a smooth log spectrum through the gaps.
Because ``S`` is smooth and the lines are sparse, ``S`` is identifiable from
BETWEEN the lines, which is what stops the floor estimate from swallowing the
comb it is supposed to expose.

**Annealing.** Iteration 1 estimates ``theta`` from low harmonics only, where
``|k theta| < pi`` and the unwrap is not ambiguous. Folding it in shrinks the
phase error everywhere, which brings higher harmonics under the ceiling; each
later iteration extends the trustable range and adds ``psi``.

**The acceptance instruments live here too**, because a verdict is only worth
what its instrument is worth:

- :func:`order_cell_profile` — THE probe. Power spectrogram, each frame's
  frequency axis re-expressed in ORDERS of one rotor, averaged, then every unit
  cell of a harmonic band folded into one profile. The modulation depth (cell
  peak over cell median, in decibels) is comb strength that a broadened or
  displaced comb cannot hide. Never use a narrow on-order/half-order slot
  contrast instead: that instrument reads about zero for a comb whose linewidth
  exceeds the slot or whose peak sits outside it, and it has already produced
  and then withdrawn one published verdict.
- :func:`whitened_flatness` — the spectral flatness of ``|N(f)|^2 / S(f)`` per
  microphone. A correct floor model leaves a flat whitened residual.

Purity: numpy, scipy, and the sibling tracking modules only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np

from tracking.decompose import (
    DEFAULT_BANDS,
    BandwidthSchedule,
    band_name,
    fade_weights,
    reconstruct,
    stitch_bank,
    track_bands,
    track_rho2_gain,
    welch_psd,
)
from tracking.fitness_vk import FVKConfig
from tracking.vk_tracking import (
    Envelopes,
    _track_table,
    _tuma_bw_min,
    _tuma_rho,
    edge_taper,
    env_stride,
    second_diff,
    vk_envelopes,
)

__all__ = [
    "DEFAULT_BANDS",
    "HPowers",
    "JointConfig",
    "JointResult",
    "JointState",
    "PhaseSplit",
    "SmoothPSD",
    "StochasticSplit",
    "bw_psi_hz",
    "cell_profile",
    "comb_lines",
    "d2_pseudo_logdet",
    "corrected_phase",
    "fit_floor_powers",
    "floor_at_tracks",
    "floor_block",
    "floor_lambda",
    "floor_penalty",
    "frame_starts",
    "global_rate_correction",
    "line_half_widths",
    "joint_objective",
    "joint_result",
    "joint_state",
    "map_objective",
    "masked_smooth_psd",
    "order_cell_bands",
    "order_cell_profile",
    "prior_logdet",
    "solve_block",
    "solve_report",
    "split_block",
    "split_phases",
    "stft_power",
    "stitch_windows",
    "stochastic_block",
    "stochastic_half_widths",
    "stochastic_split",
    "theta_rate",
    "upsample_env",
    "v4_ridge",
    "v4_rho2_gain",
    "wh_lambda",
    "wh_smooth",
    "whiten_weights",
    "whittle_floor_objective",
    "whitened_flatness",
    "window_extra_phase",
]

#: Measured shaft-wander law: harmonic ``k`` is about this many Hz wide per
#: harmonic index (``docs/experiments/vk-decomposition.md``, sigma_r ~ 0.6 rev/s).
LINEWIDTH_HZ_PER_K = 0.6

#: How many linewidths of SEARCH REGION regime 3 gives one line
#: (:func:`stochastic_half_widths`). Three of them is about +/- 3 FWHM, which
#: is 90 % of a Lorentzian's power. It is a search region and not a subtraction
#: width, because the gain inside it is per bin: a bin that sits at floor level
#: gets a gain of one whether or not a band claims it — so a wider region has
#: no over-subtraction cost, and the sweep on both rigs measured it: 3.0 takes
#: FLY124 k10-24 from 10.3 % to 6.9 % retained (tails outside +/- 2 FWHM) and
#: moves nothing anywhere else (4.0 adds nothing more).
STOCHASTIC_WIDTH_FACTOR = 3.0

#: Boxcar widths (in frames and in frequency bins) of the measured periodogram
#: the regime-3 gain is taken against. Both are FIXED, and the number that fixes
#: them is the chi-square variance of a periodogram bin: one bin is an
#: exponential deviate with 100 % relative standard deviation, so a gain built
#: on it is noise. A 5 x 3 boxcar averages 15 bins — fewer in effect, because a
#: Hann analysis at hop ``n_fft / 4`` correlates its neighbours — which brings
#: the power estimate to about 26 % and the amplitude gain, which is its square
#: root, to about 13 %. Wider would blur the line SHAPE the split exists to
#: read; narrower would put chi-square noise straight into the gain.
P_SMOOTH_FRAMES = 5
P_SMOOTH_BINS = 3


# ---------------------------------------------------------------------------
# the one-dimensional smoother (a 1-D Vold-Kalman)


def wh_lambda(bw_hz: float, fs: float) -> float:
    """Whittaker-Henderson weight for a given -3 dB bandwidth, at rate ``fs``.

    The smoother's transfer is ``1 / (1 + lam (2 sin(w/2))^4)``, which is the
    VK-2 transfer with ``lam = rho^2`` — so the bandwidth relation is the
    solver's own (:func:`tracking.vk_tracking._tuma_rho`) and there is one
    calibration in the package, not two. The band is held at or above the
    smallest numerically usable one.
    """
    lo = _tuma_bw_min(float(fs), 2)
    return float(_tuma_rho(max(float(bw_hz), lo), float(fs), 2) ** 2)


def wh_smooth(y: Any, lam: float, weight: Any | None = None) -> np.ndarray:
    """Whittaker-Henderson smooth of ``(..., N)`` rows: ``min ||w(y-x)||^2 + lam ||D2 x||^2``.

    One Hermitian banded solve with bandwidth 2 — the same normal equations the
    coupled solver assembles, in one dimension and with no coupling. Rows share
    the factorization only when ``weight`` is None or one dimensional; a short
    row (fewer than three samples) has no second difference and is returned
    unchanged.
    """
    from scipy.linalg import solveh_banded

    v = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n = int(v.shape[-1])
    if n < 3:
        return np.asarray(y, dtype=np.float64)
    d2 = second_diff(n)
    d2td2 = (d2.T @ d2).tocsr()
    d0 = np.asarray(d2td2.diagonal(0), dtype=np.float64)
    d1 = np.asarray(d2td2.diagonal(1), dtype=np.float64)
    dd2 = np.asarray(d2td2.diagonal(2), dtype=np.float64)
    w = np.ones(n) if weight is None else np.broadcast_to(np.asarray(weight, np.float64), (n,))
    ab = np.zeros((3, n), dtype=np.float64)
    ab[2] = float(lam) * d0 + w
    ab[1, 1:] = float(lam) * d1
    ab[0, 2:] = float(lam) * dd2
    out = np.asarray(solveh_banded(ab, (w[None, :] * v).T, lower=False)).T
    return out.reshape(np.shape(y))


def upsample_env(vals: Any, n_out: int, stride: int) -> np.ndarray:
    """``(R, J)`` envelope-grid rows -> ``(R, n_out)`` at audio rate, linearly.

    Knot ``j`` sits at audio sample ``j * stride``; the tail beyond the last
    knot is held, which is the rule every other upsample in the package uses.
    """
    v = np.atleast_2d(np.asarray(vals, dtype=np.float64))
    j = np.arange(int(v.shape[-1]), dtype=np.float64) * float(stride)
    q = np.arange(int(n_out), dtype=np.float64)
    return np.stack([np.interp(q, j, row) for row in v])


# ---------------------------------------------------------------------------
# the framed power spectrogram both readings of the residual are built on


def frame_starts(n_t: int, n_fft: int, hop: int) -> np.ndarray:
    """Start sample of every whole ``n_fft`` frame that fits in ``n_t``.

    A clip shorter than one frame still gives ONE start, which each caller
    then admits or refuses on its own terms.
    """
    return np.arange(0, max(1, int(n_t) - int(n_fft) + 1), max(1, int(hop)), dtype=np.int64)


def stft_power(audio: Any, starts: Any, n_fft: int, frames_per_chunk: int = 64) -> Any:
    """Yield ``(chunk starts, (C, frames, F) power)`` — Hann, one sided.

    THE spectrogram of this module, shared by the floor fit
    (:func:`masked_smooth_psd`) and the order-cell probe
    (:func:`order_cell_profile`). It is a generator because a whole recording's
    frames do not fit in memory at ``n_fft`` 8192: one chunk holds
    ``frames_per_chunk`` frames of every channel and the caller reduces it.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    st = np.asarray(starts, dtype=np.int64)
    win = np.hanning(int(n_fft))
    off = np.arange(int(n_fft))[None, :]
    for c0 in range(0, st.size, int(frames_per_chunk)):
        sub = st[c0 : c0 + int(frames_per_chunk)]
        seg = y[:, sub[:, None] + off] * win[None, None, :]
        yield sub, np.abs(np.fft.rfft(seg, axis=-1)) ** 2


# ---------------------------------------------------------------------------
# instrument 1: the order-cell profile (probe B)


def order_cell_bands(audio: Any, sr: float, r_audio: Any, **kwargs: Any) -> dict[str, Any]:
    """The band table of :func:`order_cell_profile`, without the plot arrays.

    What every REPORT carries: the per-band readings with the ``offsets`` and
    ``cell`` rows dropped, because a folded cell is 200 numbers a reader of a
    JSON report never looks at. One construction, so an iteration's table and a
    recording's table have the same keys.
    """
    return {
        nm: {kk: vv for kk, vv in d.items() if kk not in ("offsets", "cell")}
        for nm, d in order_cell_profile(audio, sr, r_audio, **kwargs)["bands"].items()
    }


def order_cell_profile(
    audio: Any,
    sr: float,
    r_audio: Any,
    *,
    rotors: Any | None = None,
    exclude_others: bool = False,
    n_fft: int = 8192,
    hop: int | None = None,
    order_step: float = 0.005,
    k_max: int = 80,
    bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS,
    f_min_hz: float = 20.0,
    mask_factor: float = 1.5,
    mask_min_hz: float = 3.0,
    fold: str = "mean",
    detrend_orders: float = 1.0,
    frames_per_chunk: int = 64,
) -> dict[str, Any]:
    """THE comb-removal verdict: modulation depth of the folded order cell.

    Take the power spectrogram (Hann, ``n_fft``, averaged over microphones),
    re-express each frame's frequency axis in ORDERS of one REFERENCE rotor
    (frequency over that rotor's instantaneous rate, so the comb stops drifting
    and its teeth sit on the integers), average over the recording onto a fixed
    order grid of ``order_step``, then fold every unit cell of a harmonic band
    into ONE profile (:func:`cell_profile`). A comb is a bump in that profile;
    its height over the cell median is the **modulation depth**, and the position
    of the bump is the systematic order OFFSET of the real lines against the
    predicted ones. Every rotor is used as the reference in turn and the band
    reading is the mean over them, with the per-rotor values kept beside it.

    Why this and not a slot contrast: a contrast between a narrow on-order slot
    and a narrow half-order slot raises both slots by nearly the same amount
    once the comb is broadened past the slot or displaced out of it, so it reads
    about zero whether or not the comb was removed. The full profile at ALL
    offsets cannot be fooled that way. Success is depth about 0 in every band.

    ``exclude_others`` is what makes the reading meaningful on a MULTI-ROTOR rig,
    and it is on by default. In one rotor's order frame the OTHER rotors' lines
    sit at non-integer orders that move with the rate ratio, so they raise the
    folded cell at scattered offsets and put a floor under the depth — measured
    on a synthetic four-rotor fixture, an almost perfect decomposition (0.24 % of
    the energy left) still read 1.08 dB at k10-24 with the foreign lines in. So
    every bin within ``max(mask_factor * LINEWIDTH_HZ_PER_K * k, mask_min_hz)``
    Hz of any OTHER rotor's line is dropped, per frame, before the order mapping.

    Returns ``{"order_step", "grid", "profile", "bands": {name: {...}}}``;
    ``profile`` is the ``(R, G)`` un-folded order profile, kept so a caller can
    plot it.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_rot = int(r.shape[0])
    refs = list(range(n_rot)) if rotors is None else [int(v) for v in np.atleast_1d(rotors)]
    n_t = int(y.shape[-1])
    step = int(n_fft // 4 if hop is None else hop)
    starts = frame_starts(n_t, n_fft, step)
    grid = np.arange(0.0, float(k_max) + 0.5 + 0.5 * order_step, float(order_step))
    acc = np.zeros((len(refs), grid.size), dtype=np.float64)
    cnt = np.zeros((len(refs), grid.size), dtype=np.float64)
    if starts.size == 0 or n_t < n_fft:
        return {
            "order_step": float(order_step),
            "grid": grid,
            "profile": acc,
            "n_frames": 0,
            "bands": {band_name(lo, hi): _empty_cell() for lo, hi in bands},
        }

    freq = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    keep = (freq >= float(f_min_hz)) & (freq <= 0.45 * float(sr))
    fk = freq[keep]
    ks = np.arange(1, int(k_max) + 1, dtype=np.float64)
    half = np.maximum(mask_factor * LINEWIDTH_HZ_PER_K * ks, float(mask_min_hz))
    for sub, chunk in stft_power(y, starts, n_fft, frames_per_chunk):
        # (mic, frame, F) -> power averaged over the microphones
        power = np.mean(chunk, axis=0)[:, keep]
        for i, s in enumerate(sub):
            rate = r[:, s : s + n_fft].mean(axis=-1)
            for a, ref in enumerate(refs):
                r_t = float(rate[ref])
                if not np.isfinite(r_t) or r_t <= 1e-6:
                    continue
                orders = fk / r_t
                prof = np.interp(grid, orders, power[i], left=np.nan, right=np.nan)
                if exclude_others and n_rot > 1:
                    other = [j for j in range(n_rot) if j != ref]
                    lines = (ks[None, :] * rate[other][:, None]).ravel()
                    mask = _line_mask(fk, lines, np.tile(half, len(other)))
                    # A grid point is kept only when BOTH bracketing bins are
                    # unmasked, so the interpolation cannot bridge across a line.
                    bridged = np.interp(grid, orders, mask.astype(np.float64), left=1.0, right=1.0)
                    prof = np.where(bridged > 0.0, np.nan, prof)
                ok = np.isfinite(prof)
                acc[a, ok] += prof[ok]
                cnt[a, ok] += 1.0
    profile = np.where(cnt > 0, acc / np.maximum(cnt, 1e-30), np.nan)
    trend = np.stack(
        [_order_trend(profile[a], order_step, detrend_orders) for a in range(len(refs))]
    )
    out_bands: dict[str, Any] = {}
    for lo, hi in bands:
        per = [
            cell_profile(profile[a], grid, lo, hi, order_step, fold, trend=trend[a])
            for a in range(len(refs))
        ]
        got = [d for d in per if d["depth_db"] is not None]
        out_bands[band_name(lo, hi)] = (
            _empty_cell()
            if not got
            else {
                "depth_db": round(float(np.mean([d["depth_db"] for d in got])), 4),
                "depth_db_max": round(float(np.max([d["depth_db"] for d in got])), 4),
                "excess_db": round(
                    float(
                        10.0
                        * np.log10(
                            max(
                                float(np.mean([10.0 ** (d["excess_db"] / 10.0) for d in got])),
                                1e-300,
                            )
                        )
                    ),
                    4,
                ),
                "peak_offset": round(float(np.mean([d["peak_offset"] for d in got])), 4),
                "n_cells": int(np.sum([d["n_cells"] for d in got])),
                "per_rotor": [
                    {
                        "rotor": int(refs[a]),
                        "depth_db": per[a]["depth_db"],
                        "peak_offset": per[a]["peak_offset"],
                    }
                    for a in range(len(refs))
                ],
                "offsets": got[0]["offsets"],
                "cell": got[0]["cell"],
            }
        )
    return {
        "order_step": float(order_step),
        "grid": grid,
        "profile": profile,
        "trend": trend,
        "n_frames": int(starts.size),
        "bands": out_bands,
    }


def _order_trend(profile: Any, order_step: float, detrend_orders: float) -> np.ndarray:
    """Running median of the order profile over ``detrend_orders`` of order.

    THE fix to the instrument's one measured bias. Each unit cell spans a whole
    order, which at a rotor rate of 80 rev/s is 80 Hz of frequency, and the
    broadband floor falls steeply across that span at low harmonics. Dividing a
    cell by its own SCALAR median leaves that slope in, so a folded cell of pure
    smooth floor is a monotone ramp — high at the low edge (offset -0.5), low at
    the high edge — and ``argmax`` then reports a peak at -0.5 that is not a line
    at all. A running median over one order tracks the slope and removes it,
    while a comb line, which occupies a small fraction of an order, passes
    through untouched.

    Measured on the v3 DREGON residual: the un-detrended fold read 1.43 dB at
    k1-9 peaking at -0.4962 orders, and the two ENDS of the cell — which are the
    SAME physical half-integer position — read +1.6 dB and -0.4 dB, which no
    line can do. The detrended fold reads the truth instead.
    """
    p = np.asarray(profile, dtype=np.float64)
    n = max(3, int(round(float(detrend_orders) / float(order_step))) | 1)
    ok = np.isfinite(p)
    if not ok.any():
        return np.ones_like(p)
    filled = np.interp(np.arange(p.size), np.flatnonzero(ok), p[ok])
    from scipy.ndimage import median_filter

    tr = median_filter(filled, size=n, mode="nearest")
    return np.where(tr > 0, tr, np.nan)


def _empty_cell() -> dict[str, Any]:
    return {
        "depth_db": None,
        "depth_db_max": None,
        "excess_db": None,
        "peak_offset": None,
        "n_cells": 0,
        "per_rotor": [],
        "offsets": [],
        "cell": [],
    }


def cell_profile(
    profile: Any,
    grid: Any,
    lo: int,
    hi: int,
    order_step: float,
    fold: str = "mean",
    *,
    trend: Any | None = None,
) -> dict[str, Any]:
    """Fold the unit cells ``[m - 0.5, m + 0.5)``, ``m`` in ``[lo, hi]``, into one.

    Each cell is first normalized by its OWN median, so the fold measures
    modulation and not the spectral tilt across the band — a band spans a decade
    of frequency, and its loudest cell would otherwise be the only cell the fold
    sees. ``fold`` then combines the cells: ``"mean"`` is the published v2
    reading, ``"median"`` is a robust variant that discards the other rotors'
    lines (they hit one or two cells each, while the reference rotor's line is in
    every cell) at the cost of sensitivity to the loudest cells.

    TWO readings come back, and a verdict needs both:

    ``depth_db``
        The folded peak over the folded median — RELATIVE comb strength. It is
        the published instrument, and its weakness is that it is a ratio: as a
        decomposition drives the residual down toward the broadband floor, the
        depth can hold or even rise while the absolute comb energy falls, and on
        a four-rotor rig the other rotors' lines put a floor under it (measured
        on the synthetic fixture: 1.40 dB for the ORIGINAL audio at k10-24
        against 1.08 dB for an almost perfect decomposition — no discrimination
        at all in that band).
    ``excess_db``
        Ten times the log of the summed ABSOLUTE excess ``peak - median`` over
        the band's cells, before the per-cell normalization. It is in power
        units of the input, so it is comparable ACROSS signals: the original
        audio's ``excess_db`` minus the residual's is how many decibels of comb
        the decomposition removed, and it does not move when the floor moves.

    ``peak_offset`` is where the folded peak sits, in orders, and 0 means the
    lines are where the labels predict.
    """
    p = np.asarray(profile, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    tr = None if trend is None else np.asarray(trend, dtype=np.float64)
    n_cell = int(round(1.0 / float(order_step)))
    offsets = -0.5 + np.arange(n_cell, dtype=np.float64) * float(order_step)
    cells: list[np.ndarray] = []
    excess = 0.0
    for m in range(int(lo), int(hi) + 1):
        j0 = int(round((float(m) - 0.5 - float(g[0])) / float(order_step)))
        if j0 < 0 or j0 + n_cell > p.size:
            continue
        seg = p[j0 : j0 + n_cell]
        if not np.all(np.isfinite(seg)):
            continue
        # The local trend, or the cell's own scalar median when no trend is
        # given (the un-detrended reading, kept for a direct caller).
        base = np.full(n_cell, float(np.median(seg))) if tr is None else tr[j0 : j0 + n_cell]
        if not np.all(np.isfinite(base)) or float(np.min(base)) <= 0.0:
            continue
        cells.append(seg / base)
        excess += max(float(np.max(seg - base)), 0.0)
    if not cells:
        return _empty_cell()
    stack = np.stack(cells)
    cell = np.median(stack, axis=0) if fold == "median" else np.mean(stack, axis=0)
    med = float(np.median(cell))
    peak = int(np.argmax(cell))
    return {
        "depth_db": round(float(10.0 * np.log10(max(cell[peak], 1e-30) / max(med, 1e-30))), 4),
        "excess_db": round(float(10.0 * np.log10(max(excess, 1e-300))), 4),
        "peak_offset": round(float(offsets[peak]), 4),
        "n_cells": len(cells),
        "offsets": [round(float(v), 4) for v in offsets],
        "cell": [round(float(v), 6) for v in cell],
    }


# ---------------------------------------------------------------------------
# block C: the masked smooth floor, and instrument 2


@dataclass
class SmoothPSD:
    """Smooth log power spectral density per microphone and per time block."""

    freq: np.ndarray  # (F,) Hz
    t_block: np.ndarray  # (B,) seconds — the block centers
    log_s: np.ndarray  # (C, B, F) natural log of the power spectral density
    n_masked_frac: float = 0.0  # share of time-frequency cells the comb mask took
    n_cep: int = 0

    def pooled(self) -> np.ndarray:
        """``(B, F)`` log spectrum pooled over microphones (the geometric mean).

        The whitening weight reads THIS and not the per-microphone surface, on
        purpose: a per-microphone weight makes the banded system channel
        dependent and costs one factorization per microphone, while the shape of
        the floor barely differs between seats — only its level does. The
        per-microphone surface is still estimated and still reported, because it
        is what a downstream noise model wants.
        """
        return np.asarray(self.log_s, dtype=np.float64).mean(axis=0)


def _line_mask(freq: np.ndarray, lines: np.ndarray, half_hz: np.ndarray) -> np.ndarray:
    """Boolean ``(F,)``: which bins sit inside ``+/- half_hz`` of any line.

    Built as a difference array over bin ranges, so one frame costs two
    ``searchsorted`` calls and a ``cumsum`` however many lines there are — a
    Python loop over the 320 lines of a four-rotor comb, per frame, is what this
    replaces.
    """
    n_f = int(freq.size)
    lo = np.searchsorted(freq, lines - half_hz, side="left")
    hi = np.searchsorted(freq, lines + half_hz, side="right")
    keep = lo < hi
    diff = np.zeros(n_f + 1, dtype=np.int32)
    np.add.at(diff, np.clip(lo[keep], 0, n_f), 1)
    np.add.at(diff, np.clip(hi[keep], 0, n_f), -1)
    return np.cumsum(diff)[:n_f] > 0


def masked_smooth_psd(
    audio: Any,
    sr: float,
    r_audio: Any,
    k_hi: int,
    *,
    n_fft: int = 4096,
    hop: int | None = None,
    n_blocks: int = 4,
    n_cep: int = 40,
    med_bins: int = 9,
    mask_factor: float = 3.0,
    mask_min_hz: float = 10.0,
    mask_frac_of_rate: float = 0.45,
    frames_per_chunk: int = 64,
    t_start_s: float = 0.0,
) -> SmoothPSD:
    """Smooth log spectrum of the floor, fitted BETWEEN the comb lines.

    Per short frame every predicted line ``k r_r(t)`` is masked out to
    ``+/- clip(mask_factor * LINEWIDTH_HZ_PER_K * k, mask_min_hz,
    mask_frac_of_rate * r_r(t))`` Hz — per frame, so a moving line is followed —
    and the unmasked power is pooled into ``n_blocks`` time blocks. The mask has
    to be several linewidths wide, not one: a line whose skirts are 25 dB above
    the floor still lifts the fit two linewidths out. Measured on the synthetic
    fixture, the log spectrum of the v3 residual against truth reads 3.5 dB rms
    at ``(1.5, 3 Hz)``, **0.6 dB** at ``(3, 10 Hz)``, and 6.5 dB again at
    ``(4, 30 Hz)`` — too wide is as bad as too narrow, because the fit then has
    to bridge gaps instead of seeing the floor. The ``mask_frac_of_rate`` cap is
    what keeps the rule usable at ``k`` 80, where ``3 * 0.6 * k`` alone is wider
    than the whole distance between a rotor's own neighbouring lines.

    Each block's spectrum is then made smooth in two
    steps: a moving median across frequency (which removes what narrow peaks
    survived the mask) and a cepstral lift to ``n_cep`` coefficients (which is
    the low-order curve the prior asks for).

    Masking is what makes the estimate honest. An unmasked floor fit rises under
    every line, and a floor that rises under the lines tells block A not to
    bother fitting them — the failure mode is self-sealing, which is why the
    mask is not optional.
    """
    from scipy.fft import dct, idct
    from scipy.ndimage import median_filter

    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_ch, n_t = int(y.shape[0]), int(y.shape[-1])
    step = int(n_fft // 2 if hop is None else hop)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    n_f = int(freq.size)
    starts = frame_starts(n_t, n_fft, step)
    n_bl = max(1, int(n_blocks))
    acc = np.zeros((n_ch, n_bl, n_f), dtype=np.float64)
    cnt = np.zeros((n_bl, n_f), dtype=np.float64)
    seen = np.zeros((n_bl, n_f), dtype=np.float64)
    # ``frame_starts`` hands back ONE start even for a clip shorter than a
    # frame, and admitting or refusing it is each caller's own business
    # (see its docstring). This caller refuses: a frame that runs off the end of
    # the clip is not a measurement, and indexing it is an error.
    if starts.size == 0 or n_t < n_fft:
        base = np.log(np.maximum(np.mean(y**2, axis=-1), 1e-30))
        return SmoothPSD(
            freq=freq,
            t_block=np.array([t_start_s]),
            log_s=np.repeat(base[:, None, None], n_f, axis=-1),
            n_cep=int(n_cep),
        )

    scale = 1.0 / (float(sr) * float((np.hanning(n_fft) ** 2).sum()))
    ks = np.arange(1, int(k_hi) + 1, dtype=np.float64)
    want = np.maximum(mask_factor * LINEWIDTH_HZ_PER_K * ks, float(mask_min_hz))
    block_of = np.minimum((starts * n_bl) // max(1, n_t), n_bl - 1)
    done = 0
    for sub, chunk in stft_power(y, starts, n_fft, frames_per_chunk):
        power = chunk * scale  # (C, frames, F) power spectral density
        for i, s in enumerate(sub):
            rate = r[:, s : s + n_fft].mean(axis=-1)
            lines = (ks[None, :] * rate[:, None]).ravel()
            half = np.minimum(want[None, :], float(mask_frac_of_rate) * rate[:, None]).ravel()
            mask = _line_mask(freq, lines, half)
            b = int(block_of[done + i])
            free = (~mask).astype(np.float64)
            acc[:, b] += power[:, i] * free[None, :]
            cnt[b] += free
            seen[b] += 1.0
        done += sub.size

    log_s = np.empty((n_ch, n_bl, n_f), dtype=np.float64)
    for c in range(n_ch):
        for b in range(n_bl):
            good = cnt[b] > 0
            if not good.any():
                log_s[c, b] = np.log(1e-30)
                continue
            vals = np.where(good, acc[c, b] / np.maximum(cnt[b], 1e-30), np.nan)
            # Gaps (bins masked in every frame of the block) are bridged across
            # frequency: a smooth spectrum is defined there, it is only unseen.
            xs = np.flatnonzero(good)
            vals = np.interp(np.arange(n_f), xs, vals[xs])
            vals = median_filter(vals, size=max(1, int(med_bins)), mode="nearest")
            lg = np.log(np.maximum(vals, 1e-30))
            cep = np.asarray(dct(lg, type=2, norm="ortho"), dtype=np.float64)
            cep[int(n_cep) :] = 0.0
            log_s[c, b] = np.asarray(idct(cep, type=2, norm="ortho"), dtype=np.float64)
    edges = np.linspace(0.0, n_t / float(sr), n_bl + 1)
    t_block = float(t_start_s) + 0.5 * (edges[:-1] + edges[1:])
    masked_frac = float(1.0 - cnt.sum() / max(seen.sum(), 1e-30))
    return SmoothPSD(
        freq=freq,
        t_block=t_block,
        log_s=log_s,
        n_masked_frac=round(masked_frac, 5),
        n_cep=int(n_cep),
    )


#: The floor's smoothness LENGTH SCALE in hertz — the ``B_f`` of the v4 floor
#: fit (:func:`fit_floor_powers`). It says "the broadband floor may not vary
#: faster than this", which is the one statement about ``S`` the model makes,
#: and it is a length in HERTZ so that it means the same thing at any ``n_fft``
#: and any sample rate.
#:
#: The v3 cepstral lift is the scale to beat: 40 kept cosines over 2049 bins
#: pass nothing shorter than ``2 x 2049 / 40`` bins, which at 32 kHz and
#: ``n_fft`` 4096 is about 800 Hz of period, so the smallest feature it carries
#: is about 400 Hz wide. CALIBRATED on the dense synthetic (a single comb whose
#: density ``gamma / Delta`` runs from 0.06 to 0.48, so the top of it is
#: blanketed by the v3 mask), the fitted floor lands at
#: 0.37 / 0.53 / 0.70 / 0.46 dB rms in the four bands at 600 Hz, against
#: 0.37 / 0.63 / 1.03 / 0.73 at 400 Hz and 0.33 / 0.61 / 0.49 / 0.31 at 800 Hz.
#: 600 Hz is where the gate is met with the smallest departure from the
#: smoothness v3 shipped with; below about 400 Hz the two starts also stop being
#: separable on the screening budget and the fit can be left in the wrong basin.
FLOOR_LENGTH_HZ = 600.0

#: Budget of ``(S, H)`` rounds per (microphone, block). Each round is one
#: non-negative least squares plus one damped Newton, and the loop stops as soon
#: as a round buys less than ``round_tol`` of the objective — so this is a CAP
#: and not a count. Measured on the dense fixture: three rounds leave the floor
#: 2.6 dB low inside a blanketed band and ten leave it inside 0.5 dB, because
#: the trade of power between the line field and the floor is what converges
#: slowly.
FLOOR_POWER_ROUNDS = 12

#: How many of those rounds each START gets before one of them is chosen. The
#: two basins part company immediately — a blanketed cell is 10 dB apart after
#: one round — so screening is what keeps the guard from doubling the cost.
FLOOR_SCREEN_ROUNDS = 2


def _lambda_from_df(df: float, b_f_hz: float, p: int = 2) -> float:
    """The floor penalty's weight from the BIN WIDTH and the length scale."""
    if b_f_hz <= 0.0:
        raise ValueError(f"the floor length scale must be positive, got {b_f_hz}")
    return float(_tuma_rho(min(2.0 * float(df) / float(b_f_hz), 0.999), 1.0, int(p)) ** 2)


def floor_penalty(psd: SmoothPSD, b_f_hz: float = FLOOR_LENGTH_HZ) -> float:
    """``lam_f sum_{c,b} ||D2_f log S||^2`` — the floor's own term of ``J_v4``.

    The v4 objective carries the floor's smoothness penalty EXPLICITLY, because
    ``S`` is a fitted parameter of the model and not a projection any more: a
    hypothesis whose floor has to bend to fit its own residual pays for the
    bending. It is read off the fitted surface, so a caller needs no knowledge
    of the grid the fit ran on beyond the surface itself.
    """
    freq = np.asarray(psd.freq, dtype=np.float64)
    if freq.size < 3:
        return 0.0
    lam = _lambda_from_df(float(freq[1] - freq[0]), float(b_f_hz))
    return lam * _curvature(np.asarray(psd.log_s, dtype=np.float64).reshape(-1, freq.size))


def floor_lambda(b_f_hz: float, sr: float, n_fft: int, p: int = 2) -> float:
    """Penalty weight of ``||D2_f log S||^2`` for a length scale of ``b_f_hz`` Hz.

    The floor penalty is a Whittaker-Henderson smoother along FREQUENCY, so its
    weight comes from the solver's own Tuma relation (:func:`wh_lambda`) with
    the frequency axis read as the time axis: one "sample" is one bin, so the
    "sampling rate" is 1 and a period of ``b_f_hz`` Hz is
    ``b_f_hz / df`` bins, that is ``w_c = 2 pi df / b_f_hz`` radians per bin.
    Since :func:`tracking.vk_tracking._tuma_rho` places its half-power point at
    ``w = pi bw / fs``, the band to ask for is ``2 df / b_f_hz`` at ``fs = 1``.

    There is therefore ONE bandwidth calibration in the package and not two, and
    ``b_f_hz`` is a physical length that means the same thing on every grid.
    """
    return _lambda_from_df(float(sr) / int(n_fft), float(b_f_hz), int(p))


@dataclass
class HPowers:
    """Per-line powers of the comb — the v4 model's own amplitude parameters.

    ``h[c, b, l]`` is the PEAK power spectral density of line ``l`` on
    microphone ``c`` during time block ``b``, in the units
    :func:`masked_smooth_psd` normalizes to, so it is directly comparable with
    ``exp(SmoothPSD.log_s)``. The line sits at ``lines[b, l]`` Hz with a
    Lorentzian half width at half maximum of ``half[b, l]`` Hz, and its total
    power is ``pi * half * h`` (a Lorentzian's integral).

    The line table is the SOLVER's track table — rotor major, ``k`` from 1 to
    ``k_hi`` (:func:`tracking.vk_tracking._track_table`) — so ``h`` indexes by
    track and the block-A ridge is a lookup and not a search. Every field is a
    plain array, so the whole record survives an ``npz`` or a JSON round trip.
    """

    rotor: np.ndarray  # (L,) int — rotor index of each line
    k: np.ndarray  # (L,) int — harmonic index
    t_block: np.ndarray  # (B,) seconds — the block centers, SmoothPSD's own
    lines: np.ndarray  # (B, L) Hz — where the line sat in each block
    half: np.ndarray  # (B, L) Hz — the Lorentzian half width at half maximum
    h: np.ndarray  # (C, B, L) >= 0 — the peak power spectral density
    diag: dict[str, Any] = field(default_factory=dict)

    def pooled(self) -> np.ndarray:
        """``(B, L)`` line power pooled over microphones — the ARITHMETIC mean.

        What the block-A ridge reads, because every microphone is a right-hand
        side against ONE system and the system therefore needs one amplitude
        prior per track. The mean is arithmetic and not geometric (the floor's
        pooling) for one reason: a line is silent on some microphones and a
        geometric mean of a set containing zero is zero, which would put an
        infinite ridge on a track the array as a whole hears perfectly well.
        """
        return np.asarray(self.h, dtype=np.float64).mean(axis=0)

    def block_of(self, t: Any) -> np.ndarray:
        """Which block each time in ``t`` (seconds, same reference) falls in.

        Nearest block center, which is the piecewise-constant reading the
        amplitude prior wants: ``H`` is a level and not a spectrum, so it is
        held over its block rather than blended across a boundary.
        """
        tb = np.asarray(self.t_block, dtype=np.float64)
        q = np.asarray(t, dtype=np.float64)
        if tb.size < 2:
            return np.zeros(q.shape, dtype=np.int64)
        return np.clip(np.searchsorted(0.5 * (tb[:-1] + tb[1:]), q), 0, tb.size - 1)


def _pool_periodogram(
    audio: np.ndarray,
    starts: np.ndarray,
    n_fft: int,
    scale: float,
    block_of: np.ndarray,
    n_blocks: int,
    frames_per_chunk: int,
) -> np.ndarray:
    """``(C, B, F)`` block-pooled power spectral density — the MEDIAN, debiased.

    One periodogram bin is an exponential deviate with 100 % relative standard
    deviation and a heavy right tail, so a block's MEAN is dragged around by its
    loudest frames; the median is not. What the median is not either is the
    level: for ``Exp(1)`` it reads ``ln 2`` of the mean, so it is divided back
    out and the result is an unbiased LEVEL that the Whittle likelihood below
    may treat as one.
    """
    n_ch = int(audio.shape[0])
    n_f = int(n_fft // 2 + 1)
    acc = np.empty((n_ch, int(starts.size), n_f), dtype=np.float32)
    done = 0
    for sub, chunk in stft_power(audio, starts, n_fft, frames_per_chunk):
        acc[:, done : done + sub.size] = (chunk * scale).astype(np.float32)
        done += sub.size
    out = np.empty((n_ch, int(n_blocks), n_f), dtype=np.float64)
    for b in range(int(n_blocks)):
        sel = block_of == b
        out[:, b] = (
            np.median(acc[:, sel], axis=1).astype(np.float64) / np.log(2.0) if sel.any() else np.nan
        )
    # A block with no frame of its own (a window shorter than the block grid)
    # holds the nearest block that has one, so the surface is defined everywhere.
    bad = ~np.isfinite(out[:, :, 0]).all(axis=0)
    if bad.any() and not bad.all():
        good = np.flatnonzero(~bad)
        for b in np.flatnonzero(bad):
            out[:, b] = out[:, good[int(np.argmin(np.abs(good - b)))]]
    return np.maximum(out, 1e-300)


#: Trust radius of one S-step, in nats of ``log S`` (about 8.7 dB). Where the
#: line powers dominate a bin the Fisher weight ``(S / M)^2`` goes to zero, and
#: the penalty alone is SEMI-definite (it cannot see a constant or a ramp), so
#: the Newton direction there is unbounded. A trust radius is the honest bound
#: on it: the direction is still the Newton one, its LENGTH is capped, and the
#: line search below decides how much of it to take.
FLOOR_TRUST_NATS = 2.0


def whittle_floor_objective(p: np.ndarray, hump: np.ndarray, g: np.ndarray, lam: float) -> float:
    """``sum_f [P~/M + log M] + lam ||D2 g||^2`` with ``M = e^g + H`` — the F1 cost.

    One (microphone, time block) of the penalized Whittle likelihood
    :func:`fit_floor_powers` minimizes. It is separate from the steps because it
    is what GUARDS them: a step is taken only if this decreases, and the start
    that ends lowest is the fit that is kept.
    """
    m = np.maximum(np.exp(g) + hump, 1e-300)
    pen = 0.0 if g.size < 3 else float(np.sum((g[:-2] - 2.0 * g[1:-1] + g[2:]) ** 2))
    return float(np.sum(p / m) + np.sum(np.log(m))) + float(lam) * pen


def _whittle_floor_step(
    p: np.ndarray,
    hump: np.ndarray,
    g0: np.ndarray,
    lam: float,
    d2: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    iters: int,
    tol: float,
) -> tuple[np.ndarray, float, int]:
    """Damped Newton on ``g = log S`` of one (channel, block) — the S-step.

    Minimizes :func:`whittle_floor_objective` over the whole frequency axis at
    once, at a fixed ``H``.

    The curvature used is the FISHER information ``(S / M)^2``, not the exact
    second derivative. The exact one goes negative wherever a bin's periodogram
    happens to sit far below the model, which on a 26 %-noisy estimate is a
    third of the band, and a Newton system that is not positive definite has no
    banded Cholesky. Fisher scoring is positive by construction, so the system
    is ``diag((S/M)^2) + 2 lam D2^T D2`` — pentadiagonal, one ``solveh_banded``
    per step — and it converges to the same stationary point.

    The direction is held inside :data:`FLOOR_TRUST_NATS` and then halved until
    the objective decreases, and the iterate that comes back is the best one
    SEEN: the start is a valid answer, so a Newton pass that cannot improve on
    it returns it unchanged.
    """
    from scipy.linalg import solveh_banded

    d0, d1, dd2 = d2
    n = int(g0.size)
    g = np.asarray(g0, dtype=np.float64).copy()
    best, best_obj = g.copy(), whittle_floor_objective(p, hump, g, lam)
    used = 0
    for _ in range(int(iters)):
        s = np.exp(g)
        m = np.maximum(s + hump, 1e-300)
        grad = s * (m - p) / m**2
        if n >= 3:
            grad = grad + 2.0 * float(lam) * (d0 * g + _band_mul(g, d1, dd2))
        w = (s / m) ** 2
        ab = np.zeros((3, n), dtype=np.float64)
        ab[2] = w + 2.0 * float(lam) * d0
        ab[1, 1:] = 2.0 * float(lam) * d1
        ab[0, 2:] = 2.0 * float(lam) * dd2
        try:
            step = np.asarray(solveh_banded(ab, -grad, lower=False), dtype=np.float64)
        except np.linalg.LinAlgError:  # pragma: no cover — Fisher curvature is PD
            break
        big = float(np.max(np.abs(step)))
        if not np.isfinite(big) or big <= 0.0:
            break
        if big > FLOOR_TRUST_NATS:
            step = step * (FLOOR_TRUST_NATS / big)
        moved = False
        damp = 1.0
        for _ in range(8):
            cand = g + damp * step
            val = whittle_floor_objective(p, hump, cand, lam)
            if val < best_obj:
                g, best, best_obj, moved = cand, cand.copy(), val, True
                break
            damp *= 0.5
        used += 1
        if not moved or min(big, FLOOR_TRUST_NATS) < float(tol):
            break
    return best, best_obj, used


def _lorentzian_powers(
    a_mat: np.ndarray, p: np.ndarray, s_lin: np.ndarray, hump: np.ndarray, *, iters: int = 2
) -> np.ndarray:
    """The H-step: non-negative line powers of one (channel, block), by IRLS.

    The conditional problem in ``H`` is the same penalized Whittle likelihood,
    whose Fisher weight per bin is ``1 / M^2``. So the Gauss-Newton step is one
    WEIGHTED non-negative least squares — rows divided by ``M`` — against the
    SIGNED excess ``P~ - S``, and it is iterated two or three times because
    ``M`` moves with the answer.

    The weight is the whole difference from a plain fit of the clipped excess.
    A periodogram bin's variance is its own level squared, so an unweighted fit
    is dominated by the loudest line in a block and reads the quiet ones off its
    residual; and clipping at zero hides the direction that says a line is
    OVER-explained, which is exactly the direction a floor that swallowed the
    comb has to be pushed back along.

    The design ``a_mat`` is the truncated Lorentzian basis on the block's own
    region bins (:func:`_lorentzian_design`), and ``p``, ``s_lin`` and ``hump``
    are that same slice of the pooled periodogram, the floor and the current
    line field.
    """
    from scipy.optimize import nnls

    if a_mat.size == 0:
        return np.zeros(a_mat.shape[-1], dtype=np.float64)
    m = np.maximum(s_lin + hump, 1e-300)
    amp = np.zeros(a_mat.shape[-1], dtype=np.float64)
    for _ in range(max(1, int(iters))):
        w = 1.0 / m
        amp, _ = nnls(a_mat * w[:, None], (p - s_lin) * w)
        m = np.maximum(s_lin + a_mat @ amp, 1e-300)
    return amp


def _band_mul(g: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Off-diagonal part of ``D2^T D2 @ g`` from its two super-diagonals.

    The matrix is symmetric pentadiagonal, so its product is the diagonal term
    (the caller's) plus this — one shifted multiply per band. Assembling the
    sparse operator to take one product per Newton step is the alternative, and
    it is the slower one.
    """
    out = np.zeros_like(g)
    out[:-1] += d1 * g[1:]
    out[1:] += d1 * g[:-1]
    out[:-2] += d2 * g[2:]
    out[2:] += d2 * g[:-2]
    return out


#: The starts of the ``(S, H)`` alternation, in decibels ON the warm floor. The
#: alternation is BISTABLE where the lines blanket a band, and both basins are
#: honest stationary points: from the masked fit the floor already sits on the
#: blanket, so the first H-step finds no excess and nothing ever moves; from a
#: floor 12 dB lower the lines claim the blanket and the floor comes back up
#: only as far as the pure-floor bins demand. Measured on the dense fixture the
#: two differ by 13 dB of fitted floor and the OBJECTIVE tells them apart, so
#: the fit runs both and keeps the lower one. It is not a tuning knob; it is
#: the admission that one start is not enough.
FLOOR_START_DB = (0.0, -12.0)


def _fit_cell(
    p: np.ndarray,
    g0: np.ndarray,
    a_mat: np.ndarray,
    kept: np.ndarray,
    bins: np.ndarray,
    lam: float,
    diags: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    rounds: int,
    newton_iters: int,
    tol: float,
    round_tol: float,
    n_f: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """One (microphone, block) fit from one start: ``(log S, H, objective, steps)``.

    The alternation itself — H-step, S-step, repeat — on the one cell every
    other loop of :func:`fit_floor_powers` is a loop over. The objective comes
    back with it because the caller CHOOSES by it, and the loop stops as soon as
    a round buys less than ``tol`` of it: on a real window most cells are done
    in two or three rounds and only the blanketed ones need the full budget.
    """
    g = np.asarray(g0, dtype=np.float64).copy()
    amp = np.zeros(int(kept.size), dtype=np.float64)
    hump = np.zeros(int(n_f), dtype=np.float64)
    obj = whittle_floor_objective(p, hump, g, lam)
    used = 0
    for i in range(int(rounds)):
        prev = obj
        if kept.size:
            # The first round has no line field yet, so its weights are the
            # floor's alone and one more IRLS pass is worth taking; afterwards
            # the weights move by percents and one is enough.
            amp = _lorentzian_powers(
                a_mat, p[bins], np.exp(g[bins]), hump[bins], iters=2 if i == 0 else 1
            )
            hump = np.zeros(int(n_f), dtype=np.float64)
            hump[bins] = a_mat @ amp
        g, obj, it = _whittle_floor_step(
            p, hump, g, lam, diags, iters=int(newton_iters), tol=float(tol)
        )
        used += it
        if i and prev - obj < float(round_tol) * max(abs(obj), 1.0):
            break
    return g, amp, obj, used


def fit_floor_powers(
    audio: Any,
    sr: float,
    r_audio: Any,
    k_hi: int,
    *,
    n_fft: int = 4096,
    hop: int | None = None,
    n_blocks: int = 4,
    b_f_hz: float = FLOOR_LENGTH_HZ,
    slope_hz_per_k: float = LINEWIDTH_HZ_PER_K,
    rounds: int = FLOOR_POWER_ROUNDS,
    screen_rounds: int = FLOOR_SCREEN_ROUNDS,
    warm: SmoothPSD | None = None,
    start_db: tuple[float, ...] = FLOOR_START_DB,
    newton_iters: int = 30,
    tol: float = 1e-8,
    round_tol: float = 1e-7,
    frames_per_chunk: int = 64,
    t_start_s: float = 0.0,
) -> tuple[SmoothPSD, HPowers]:
    """F1: the floor and the line powers, fitted JOINTLY and with NO mask.

    The v3 floor (:func:`masked_smooth_psd`) is a projection: mask every
    predicted line, pool what is left, and smooth it. That works while the lines
    are sparse and fails exactly where they are not — above about ``k`` 10 four
    interleaved combs whose lines are ``0.6 k`` Hz wide leave no unmasked bin at
    all, so the fit has to BRIDGE a dense band instead of reading it, and the
    bridge is wherever the smoother's tension puts it. The v4 fit removes the
    mask and puts the lines in the MODEL instead::

        minimize over (g = log S, H >= 0):
          sum_f [ P~_f / M_f + log M_f ] + lam_f ||D2_f g||^2 ,
          M_f = e^{g_f} + sum_l H_l / (1 + ((f - f_l) / gamma_l)^2)

    per (microphone, time block), alternating two steps that are each solvable:

    - the **H-step** is a non-negative least squares of the truncated
      Lorentzian design (:func:`_lorentzian_design`, the same construction the
      H-aware measure uses) against the clipped excess ``max(0, P~ - S)``, with
      the lines at the block's own ``k r_r`` and the half width the measured
      law ``max(0.6 k, one bin)``;
    - the **S-step** is damped Fisher scoring on ``g`` under the pentadiagonal
      penalty (:func:`_whittle_floor_step`), warm started from ``warm`` — the
      current masked fit — and objective guarded, so it can only improve on it.

    ``P~`` is the block's periodogram pooled by the MEDIAN over its frames and
    debiased by ``ln 2`` (:func:`_pool_periodogram`): a chi-square-noisy mean is
    dragged by the loudest frame of a block, and the fit is a level.

    The fit runs on the ORIGINAL signal and never on a residual. The model
    explains the lines through ``H``, so there is nothing to subtract first, and
    subtracting first is what makes a hard-EM alternation biased: the coherent
    reconstruction is itself an estimate, and its error would be read as floor.

    ``b_f_hz`` is the floor's smoothness LENGTH SCALE in hertz
    (:func:`floor_lambda`), which is the one hyperparameter the fit adds.

    Returns the same :class:`SmoothPSD` every v3 consumer already reads — same
    grid, same block centers, same units, ``n_cep`` 0 because there is no
    cepstral lift any more — beside the :class:`HPowers` that are the v4 model's
    amplitude parameters and the generator's targets.
    """

    tic = perf_counter()
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_ch, n_t = int(y.shape[0]), int(y.shape[-1])
    n_fft = int(n_fft)
    step = int(n_fft // 2 if hop is None else hop)
    n_bl = max(1, int(n_blocks))
    freq = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    df = float(sr) / n_fft
    starts = frame_starts(n_t, n_fft, step)

    base = (
        masked_smooth_psd(
            y,
            float(sr),
            r,
            int(k_hi),
            n_fft=n_fft,
            hop=step,
            n_blocks=n_bl,
            t_start_s=float(t_start_s),
        )
        if warm is None
        else warm
    )
    rotor_l, k_l = _track_table(int(r.shape[0]), 1, int(k_hi))
    if starts.size == 0 or n_t < n_fft:
        # Nothing to fit on. The masked fit's own degenerate answer stands, and
        # every line power is zero — which makes the v4 arm the v3 arm here.
        return base, HPowers(
            rotor=rotor_l,
            k=k_l,
            t_block=np.asarray(base.t_block, dtype=np.float64),
            lines=np.zeros((len(base.t_block), len(k_l))),
            half=np.zeros((len(base.t_block), len(k_l))),
            h=np.zeros((n_ch, len(base.t_block), len(k_l))),
            diag={"n_frames": 0, "rounds": 0},
        )

    scale = 1.0 / (float(sr) * float((np.hanning(n_fft) ** 2).sum()))
    block_of = np.minimum((starts * n_bl) // max(1, n_t), n_bl - 1)
    p_block = _pool_periodogram(y, starts, n_fft, scale, block_of, n_bl, frames_per_chunk)

    # The line table, per block: the block's own mean rate, so a line sits where
    # it sat on average while that block's periodogram was measured.
    lines = np.empty((n_bl, len(k_l)), dtype=np.float64)
    half = np.empty((n_bl, len(k_l)), dtype=np.float64)
    for b in range(n_bl):
        sel = starts[block_of == b]
        span = (
            r[:, int(sel[0]) : int(sel[-1]) + n_fft].mean(axis=-1) if sel.size else r.mean(axis=-1)
        )
        lines[b], kk = comb_lines(span, int(k_hi))
        half[b] = np.maximum(float(slope_hz_per_k) * kk, df)

    lam = floor_lambda(float(b_f_hz), float(sr), n_fft)
    log_s = np.asarray(_floor_on_grid(base, float(sr), n_fft), dtype=np.float64).copy()
    if log_s.shape[0] != n_ch:
        raise ValueError(f"the warm floor has {log_s.shape[0]} microphones, the audio has {n_ch}")
    h = np.zeros((n_ch, n_bl, len(k_l)), dtype=np.float64)
    d2 = second_diff(int(freq.size))
    d2td2 = (d2.T @ d2).tocsr()
    diags = (
        np.asarray(d2td2.diagonal(0), dtype=np.float64),
        np.asarray(d2td2.diagonal(1), dtype=np.float64),
        np.asarray(d2td2.diagonal(2), dtype=np.float64),
    )
    # The bins each block's fit may look at: the union of the line supports,
    # which is where the design has a non-zero column at all.
    region = [
        np.flatnonzero(_line_mask(freq, lines[b], LORENTZ_SUPPORT_HWHM * half[b]))
        for b in range(n_bl)
    ]
    design = [
        _lorentzian_design(freq, region[b], lines[b], half[b])
        if region[b].size
        else (np.zeros((0, 0)), np.zeros(0, dtype=np.int64))
        for b in range(n_bl)
    ]

    n_pos = 0.0
    n_amp = 0.0
    newton = 0
    n_low = 0
    for b in range(n_bl):
        kept = design[b][1]
        for c in range(n_ch):

            def cell(g0: np.ndarray, n: int, _b: int = b, _c: int = c) -> Any:
                return _fit_cell(
                    p_block[_c, _b],
                    g0,
                    design[_b][0],
                    design[_b][1],
                    region[_b],
                    lam,
                    diags,
                    rounds=max(1, int(n)),
                    newton_iters=int(newton_iters),
                    tol=float(tol),
                    round_tol=float(round_tol),
                    n_f=int(freq.size),
                )

            # SCREEN the starts on a short budget, then REFINE only the one that
            # is ahead. The two basins part company in the first round or two —
            # the blanketed cells go 10 dB apart immediately — so the screen
            # buys the guard at a fraction of the price of running both to
            # convergence, which on a full window is most of the cost.
            screen = [
                cell(log_s[c, b] + float(off) * np.log(10.0) / 10.0, screen_rounds)
                for off in start_db
            ]
            pick = int(np.argmin([v[2] for v in screen]))
            newton += int(sum(v[3] for v in screen))
            n_low += int(pick > 0)
            g_out, amp, _, it = (
                cell(screen[pick][0], int(rounds) - int(screen_rounds))
                if int(rounds) > int(screen_rounds)
                else screen[pick]
            )
            log_s[c, b] = g_out
            h[c, b, kept] = amp
            newton += it
            n_pos += float(np.count_nonzero(amp > 0.0))
            n_amp += float(amp.size)

    psd = SmoothPSD(
        freq=freq,
        t_block=np.asarray(base.t_block, dtype=np.float64),
        log_s=log_s,
        n_masked_frac=0.0,
        n_cep=0,
    )
    return psd, HPowers(
        rotor=rotor_l,
        k=k_l,
        t_block=np.asarray(base.t_block, dtype=np.float64),
        lines=lines,
        half=half,
        h=h,
        diag={
            "n_frames": int(starts.size),
            "n_blocks": n_bl,
            "n_lines": int(len(k_l)),
            "rounds": max(1, int(rounds)),
            "b_f_hz": float(b_f_hz),
            "lambda_f": lam,
            "newton_steps": int(newton),
            # How many of the offered lines took a positive amplitude, and how
            # much of the band the line supports cover. The second is the number
            # that says whether a masked fit had anything left to read.
            "active_line_frac": round(n_pos / max(n_amp, 1.0), 4),
            "region_bin_frac": round(
                float(np.mean([r_b.size for r_b in region]) / max(freq.size, 1)), 4
            ),
            # How often the LOWERED start beat the warm one. Near zero says the
            # masked fit was already in the right basin; near one says the comb
            # blankets the band and the v3 floor was sitting on top of it.
            "low_start_frac": round(n_low / max(n_ch * n_bl, 1), 4),
            "start_db": [float(v) for v in start_db],
            "h_total_power": float(np.pi * np.sum(half[None, :, :] * h)),
            "wall_s": round(perf_counter() - tic, 2),
        },
    )


def floor_at_tracks(psd: SmoothPSD, k: Any, rotor: Any, r_env: Any, t_env: Any) -> np.ndarray:
    """``(M, J)`` log floor at every track's own line, ``log S(k r(t), t)``.

    THE lookup both users of the floor make: the whitening weight
    (:func:`whiten_weights`) reads it as ``-0.5 log S`` and the v4 amplitude
    prior reads it as the level the line power is compared against. It is the
    microphone-POOLED surface (:meth:`SmoothPSD.pooled`) because a track is one
    column of a system every microphone is a right-hand side against, and it is
    interpolated linearly in time between the block centers.
    """
    ks = np.asarray(k, dtype=np.float64)
    rot = np.asarray(rotor, dtype=int)
    rr = np.atleast_2d(np.asarray(r_env, dtype=np.float64))
    tt = np.asarray(t_env, dtype=np.float64)
    f_mj = ks[:, None] * rr[rot]  # (M, J)
    pooled = psd.pooled()  # (B, F)
    per_block = np.stack(
        [
            np.interp(f_mj.ravel(), psd.freq, pooled[b]).reshape(f_mj.shape)
            for b in range(len(pooled))
        ]
    )  # (B, M, J)
    if per_block.shape[0] == 1:
        return per_block[0]
    tb = np.asarray(psd.t_block, dtype=np.float64)
    pos = np.clip(np.interp(tt, tb, np.arange(len(tb), dtype=np.float64)), 0, len(tb) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(tb) - 1)
    frac = (pos - lo)[None, :]
    return per_block[lo, :, np.arange(len(tt))].T * (1.0 - frac) + (
        per_block[hi, :, np.arange(len(tt))].T * frac
    )


def whiten_weights(
    psd: SmoothPSD,
    k: Any,
    rotor: Any,
    r_env: Any,
    t_env: Any,
    *,
    clamp_db: float = 15.0,
) -> np.ndarray:
    """``(M, J)`` amplitude weight ``1 / sqrt(S)`` per track and envelope frame.

    The Whittle likelihood of colored noise weights each frequency by ``1/S``.
    Because ``S`` is smooth and a track occupies a few Hz, that whole weighting
    collapses, per track, to the one scalar ``1 / S(k r(t), t)`` — which is why
    whitening costs the banded solver nothing at all.

    The weight is normalized to geometric mean 1 over the finite cells, so it
    changes only the RELATIVE trust between tracks and never the balance against
    the smoothness prior, and it is clamped to ``+/- clamp_db`` so a single
    quiet band cannot dominate a solve.
    """
    log_u = -0.5 * floor_at_tracks(psd, k, rotor, r_env, t_env)
    lim = float(clamp_db) * np.log(10.0) / 20.0
    ok = np.isfinite(log_u)
    log_u = np.clip(log_u - float(np.mean(log_u[ok])), -lim, lim)
    # Centered again AFTER the clamp, so the geometric mean is exactly 1 and the
    # whitening cannot move the balance between the data term and the prior. The
    # clamp then bounds the SPREAD at 2 * clamp_db, not each value.
    return np.exp(log_u - float(np.mean(log_u[ok])))


#: Base of the v4 envelope bandwidth law, in hertz. It is a FLOOR under
#: ``0.6 k``, so the fundamental keeps a usable band and every harmonic above
#: about ``k`` 2 gets the measured linewidth instead.
V4_BW0_HZ = 1.0

#: The one calibrated constant of the v4 amplitude prior: the precision a track
#: is given is ``beta = c0 * S / H`` in the solver's own data-weight units.
#:
#: It is 1 for a reason and not by fitting. The envelope band is
#: ``b_A(k) = 0.6 k`` Hz and the line's own half width is the same ``0.6 k``, so
#: the line's spectrum is nearly flat across the band the envelope admits; the
#: noise is flat there too, so the ratio of the two POWERS the band admits is
#: exactly the ratio of the two DENSITIES, ``S / H``. The optimal scalar
#: shrinkage of that observation is ``H / (H + S)``, which is what a diagonal of
#: ``S / H`` beside a data weight of 1 produces.
#:
#: Measured (``tests/tracking/test_v4_ridge.py``, three seeds): the ridged
#: solve's reconstructed power over the Wiener target is 0.97 / 1.02 / 1.01 at
#: ``k`` 5 / 20 / 60 for a line 10 dB over the floor, and 1.00 / 1.09 / 1.07 for
#: one 4.8 dB over it — inside the +/-20 % the design asks for, where 0.8 and
#: 1.5 are both outside it at the second signal-to-noise ratio.
V4_RIDGE_C0 = 1.0

#: Floor under the v4 amplitude prior, as a fraction of the coupling group's own
#: mean data curvature (:func:`tracking.vk_tracking._floor_beta`). A proper prior
#: never has infinite variance, and ``beta = c0 S / H`` goes to zero for a strong
#: line — which is exactly the track whose near-degenerate pairs then have
#: NOTHING holding their difference direction positive, because a wide band also
#: means a small ``rho^2``. Measured on a full-scale DREGON spin-up window
#: (32 kHz, ``k_hi`` 83, 332 tracks): with no floor the banded Cholesky reported
#: the system non definite, and the run then DIED because the splu fallback tried
#: to factorize it. That second half is fixed by refusing the fallback; this
#: constant is the first half.
#:
#: 0.03 is measured from both ends, and both ends are tight:
#:
#: - **From below.** The deficiency is not rounding, so the floor has to be
#:   large. On a four-rotor spin-up fixture (120 tracks in one group, rotors
#:   fanning by 2 rev/s and crossing) nothing under 0.03 factorizes at all,
#:   because the near-degenerate directions are killed by the DECIMATED cross
#:   term's own approximation error and not by float rounding — the error is
#:   percent-relative, not epsilon-relative.
#: - **From above.** At 0.03 the Wiener-target calibration is unmoved to three
#:   decimals (0.973 / 1.022 / 1.011 at ``k`` 5 / 20 / 60), because the floor
#:   sits below the ``c0 S / H`` every line in it asks for anyway; the only thing
#:   that moves is a very strong line, which keeps 0.945 of its unridged power
#:   instead of 0.980. At 0.3 the calibration breaks (0.72 to 0.76, outside the
#:   +/-20 % bar) and a strong line keeps 0.61.
#:
#: What it does NOT do is rescue the tightest groups: four rotors within about
#: 1 rev/s of each other need 0.1, which costs a strong line 17 %. Those windows
#: are unidentifiable at these bands — four combs 0.3 Hz apart at ``k`` 1 are not
#: four combs to a 3-second window — and they now FAIL, loudly, instead of
#: taking the pool down. The remedy for them is a narrower ``k_hi`` or the v3
#: spacing cap, not a bigger ridge.
RIDGE_FLOOR_FRAC = 0.03


def v4_rho2_gain(
    r_audio: Any,
    k_hi: int,
    cfg: Any,
    *,
    b0_hz: float = V4_BW0_HZ,
    slope_hz_per_k: float = LINEWIDTH_HZ_PER_K,
    rho_scale: float = 1.0,
) -> np.ndarray:
    """``(M,)`` gain on ``rho^2`` for the v4 band law ``max(b0, 0.6 k)`` Hz.

    The v3 arm caps every band at a fraction of the local LINE SPACING
    (:class:`tracking.decompose.BandwidthSchedule`), and that cap is not a
    modelling statement — it is what an IMPROPER prior needs to stay
    identifiable, because two overlapping passbands with nothing bounding their
    levels have a cancelling mode. Under the v4 amplitude prior
    (:func:`v4_ridge`) nothing is improper any more, so the band is the measured
    linewidth law and nothing else.

    The gain is taken against :func:`tracking.decompose.base_bandwidths`, which
    is the band the solver would have used by itself, exactly as the v2 schedule
    does — so one construction converts a wanted bandwidth into the currency the
    solver takes, and :attr:`tracking.Envelopes.bw_track` still records what the
    solve really used.
    """
    from tracking.decompose import base_bandwidths

    vk = cfg.vk_config(int(k_hi))
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    _, k = _track_table(int(r.shape[0]), vk.k_min, int(k_hi))
    _, fs_env = env_stride(vk)
    base = np.asarray(base_bandwidths(r, int(k_hi), cfg), dtype=np.float64)
    want = np.maximum(float(b0_hz), float(slope_hz_per_k) * k.astype(np.float64))
    want = np.clip(want, _tuma_bw_min(fs_env, vk.p), 0.9 * fs_env)
    return (
        np.array(
            [
                (_tuma_rho(float(b), fs_env, vk.p) / _tuma_rho(float(b0), fs_env, vk.p)) ** 2
                for b, b0 in zip(want, base, strict=True)
            ]
        )
        * float(rho_scale) ** 2
    )


def v4_ridge(
    psd: SmoothPSD,
    hp: HPowers,
    k: Any,
    rotor: Any,
    r_env: Any,
    t_env: Any,
    weight: Any | None = None,
    *,
    c0: float = V4_RIDGE_C0,
) -> np.ndarray:
    """``(M, J)`` amplitude-prior precision ``beta = c0 S / H`` for block A.

    The envelope of track ``m`` is given a Gaussian prior whose variance is that
    track's own fitted line power, so its posterior is the Wiener one: a line
    far above the floor keeps its amplitude, a track sitting ON the floor is
    shrunk toward zero, and the shrinkage is a RATIO and not a threshold. That
    ratio is what lets the bands open to ``0.6 k`` Hz with no spacing cap — the
    v3 cap protected an improper prior, and this prior is proper.

    Three things decide the arithmetic:

    - ``H`` is POOLED over microphones (:meth:`HPowers.pooled`). Every channel
      is a right-hand side against ONE banded system, so the system carries one
      prior per track and not one per (track, microphone), and the pooled line
      power is what that one prior can be.
    - ``H`` is read at the track's own block (:meth:`HPowers.block_of`) —
      piecewise constant in time, because it is a level and not a spectrum.
    - the precision is expressed in the DATA TERM's own units. The whitened
      solve carries ``u^2 = S_geo / S`` on its diagonal, so a ridge that is to
      mean "noise over signal" is ``c0 u^2 S / H``, which is ``c0 S_geo / H`` up
      to the whitening clamp. With no whitening the data term is the bare
      validity mask and the same expression at ``u = 1`` is the ratio itself.

    A track whose line power came back at zero — the fit found no line there —
    gets a very large ridge rather than an infinite one, so the solve stays
    finite and that envelope is simply pulled to zero.
    """
    log_s = floor_at_tracks(psd, k, rotor, r_env, t_env)  # (M, J)
    u2 = np.ones_like(log_s) if weight is None else np.asarray(weight, dtype=np.float64) ** 2
    idx = np.asarray(hp.block_of(np.asarray(t_env, dtype=np.float64)), dtype=np.int64)
    pooled = hp.pooled()  # (B, L)
    ks = np.asarray(k, dtype=np.int64)
    rot = np.asarray(rotor, dtype=np.int64)
    lookup = {(int(rr), int(kk)): i for i, (rr, kk) in enumerate(zip(hp.rotor, hp.k, strict=True))}
    take = np.array([lookup.get((int(rot[m]), int(ks[m])), -1) for m in range(len(ks))])
    h_mj = np.where(take[:, None] >= 0, pooled[idx[None, :], np.maximum(take, 0)[:, None]], 0.0)
    s_lin = np.exp(log_s)
    return float(c0) * u2 * s_lin / np.maximum(h_mj, 1e-12 * np.maximum(s_lin, 1e-300))


def whitened_flatness(
    residual: Any,
    sr: float,
    psd: SmoothPSD,
    *,
    nperseg: int = 4096,
    f_lo: float = 50.0,
) -> dict[str, Any]:
    """Spectral flatness of ``|N(f)|^2 / S(f)`` per microphone, and of ``|N|^2``.

    Flatness is the geometric mean over the arithmetic mean of the power
    spectrum: 1 for white noise, small for a tilted or peaky one. If the floor
    model is right the whitened residual is flat, so the pair (raw, whitened) is
    the reading — the whitened value should be near 1 and clearly above the raw
    one.
    """
    y = np.atleast_2d(np.asarray(residual, dtype=np.float64))
    f, p = welch_psd(y, sr, int(nperseg))
    band = (np.asarray(f) >= float(f_lo)) & (np.asarray(f) <= 0.45 * float(sr))
    s_mean = np.exp(np.asarray(psd.log_s, dtype=np.float64).mean(axis=1))  # (C, F)
    raw, whit = [], []
    for c in range(int(y.shape[0])):
        pc = np.asarray(p)[min(c, p.shape[0] - 1)][band]
        sc = np.interp(np.asarray(f)[band], psd.freq, s_mean[min(c, s_mean.shape[0] - 1)])
        raw.append(_flatness(pc))
        whit.append(_flatness(pc / np.maximum(sc, 1e-30)))
    return {
        "flatness_raw": [round(v, 5) for v in raw],
        "flatness_whitened": [round(v, 5) for v in whit],
        "flatness_raw_mean": round(float(np.mean(raw)), 5),
        "flatness_whitened_mean": round(float(np.mean(whit)), 5),
    }


def _flatness(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(p))) / max(float(np.mean(p)), 1e-30))


# ---------------------------------------------------------------------------
# block B: the phase split


def bw_psi_hz(
    k: Any, slope: float = LINEWIDTH_HZ_PER_K, cap: float = 8.0, floor: float = 1.5
) -> np.ndarray:
    """Per-track phase-correction bandwidth, ``clip(slope k, floor, cap)`` in Hz.

    The linewidth law says harmonic ``k`` wanders ``slope * k`` Hz, so its own
    phase correction needs that much room, and the cap keeps a high-``k``
    correction from turning into a second envelope that absorbs the floor.

    The FLOOR is the third guard, and it is there because the law under-serves
    the LOW harmonics: at ``k`` 1 to 3 it allows only 0.6 to 1.8 Hz, which is
    narrower than the incoherent linewidth a strong low-``k`` line really has,
    so the line keeps a skirt that the model cannot follow. The strongest
    measured leftover of the v3 production run is exactly there — 2.24 dB of
    integer-order residual at k1-9 on one FLY124 rotor.
    """
    return np.clip(
        float(slope) * np.asarray(k, dtype=np.float64), float(floor), max(float(cap), float(floor))
    )


@dataclass
class PhaseSplit:
    """One increment of the coherent phase split (block B)."""

    theta_rig: np.ndarray  # (J,) rig-common shaft correction, radians
    theta: np.ndarray  # (R, J) total per-rotor shaft correction (rig + per rotor)
    psi: np.ndarray  # (M, J) per-track phase correction
    diag: dict[str, Any] = field(default_factory=dict)


def _combine_channels(x: np.ndarray) -> np.ndarray:
    """``(C, M, J)`` -> ``(M, J)`` complex, microphones aligned and SNR weighted.

    Every microphone sees the same phase deviation up to a CONSTANT of its own
    (its propagation delay), so each channel is first rotated onto the loudest
    one by the constant ``sum_j x_c conj(x_ref)`` and then added. The sum is
    implicitly amplitude weighted, which is the weighting a phase estimate
    wants: a quiet channel contributes a short vector.
    """
    xa = np.asarray(x)
    n_ch = int(xa.shape[0])
    if n_ch == 1:
        return xa[0].copy()
    power = np.sum(np.abs(xa) ** 2, axis=-1)  # (C, M)
    ref = np.argmax(power, axis=0)  # (M,)
    out = np.zeros(xa.shape[1:], dtype=np.complex128)
    for m in range(int(xa.shape[1])):
        xr = xa[int(ref[m]), m]
        for c in range(n_ch):
            s = np.vdot(xr, xa[c, m])  # sum_j conj(xr) x_c
            mag = abs(s)
            out[m] += xa[c, m] * (np.conj(s) / mag if mag > 0 else 1.0)
    return out


def split_phases(
    x: Any,
    k: Any,
    rotor: Any,
    valid: Any,
    fs_env: float,
    *,
    k_trust: int,
    conc_min: float = 0.5,
    bw_theta_hz: float = 1.5,
    bw_psi_slope: float = LINEWIDTH_HZ_PER_K,
    bw_psi_max: float = 8.0,
    bw_psi_min: float = 1.5,
    per_rotor: bool = True,
    with_psi: bool = True,
) -> PhaseSplit:
    """Split the solved envelope phases into shaft, per-rotor and per-track parts.

    ``x`` is the ``(C, M, J)`` envelope bank of the CURRENT carrier, so its angle
    is the phase error that is left. By the model that angle is
    ``k theta_r + psi_{r,k}`` plus noise, and every harmonic measures ``theta``
    with precision ``k^2`` times its own — so the shaft estimate is the
    ``k``-weighted mean of ``arg x / k``, and it is far better determined than
    any one harmonic or the telemetry.

    Trustable tracks are the ones whose unwrap is not ambiguous. Two gates: the
    annealing cap ``k_trust`` (low harmonics see the shaft error only ``k``-fold,
    so they are unambiguous first), and a phase-increment CONCENTRATION
    ``|mean exp(j d arg x)|`` above ``conc_min`` — a scale-free signal-to-noise
    proxy that reads about 0 for a noise-dominated envelope and near 1 for a
    locked one. A track whose increment ever reaches pi is dropped whatever its
    concentration, because its unwrap is a guess.

    ``theta`` is hierarchical: a rig-common part from every rotor's tracks
    first, then a small per-rotor part from that rotor's own tracks. This is the
    measured structure — there is no per-microphone arrival term, which was
    measured and is not there.
    """
    xa = np.asarray(x)
    ks = np.asarray(k, dtype=np.float64)
    rot = np.asarray(rotor, dtype=int)
    n_tracks, n_env = int(xa.shape[1]), int(xa.shape[-1])
    n_rot = int(rot.max()) + 1 if n_tracks else 1
    theta = np.zeros((n_rot, n_env))
    psi = np.zeros((n_tracks, n_env))
    if n_tracks == 0 or n_env < 3:
        return PhaseSplit(np.zeros(n_env), theta, psi, {"n_trust": 0})

    v = _combine_channels(xa)
    ang = np.unwrap(np.angle(v), axis=-1)
    ang = ang - ang.mean(axis=-1, keepdims=True)  # the per-track constant is a gauge
    inc = np.diff(np.angle(v), axis=-1)
    inc = np.angle(np.exp(1j * inc))  # wrapped increments, for the concentration
    conc = np.abs(np.mean(np.exp(1j * inc), axis=-1))
    step_max = np.max(np.abs(np.diff(ang, axis=-1)), axis=-1)
    val = np.asarray(valid, dtype=bool)
    ok = val.all(axis=-1) if val.ndim == 2 else np.ones(n_tracks, dtype=bool)
    strong = ok & (conc >= float(conc_min)) & (step_max < np.pi)
    trust = strong & (ks <= float(k_trust))

    lam_th = wh_lambda(bw_theta_hz, fs_env)
    weight = (ks**2) * (conc**2)
    # The solver FADES its data term at both window ends, so the envelopes there
    # are the prior's extrapolation and not a measurement. Carrying the same
    # taper into the smoother's data weight makes the shaft estimate extrapolate
    # over that span too, instead of fitting the transient. Without it the RATE
    # at the very first and last frames is a fabrication, and the stitch, which
    # is built on the rate, then sees two neighbouring windows disagree by half
    # a rev/s at their seam (measured 46 Hz of track rotation at k 80 on the
    # DREGON production run, against 0.003 Hz on a single-window smoke).
    tap = edge_taper(n_env)
    theta_rig = np.zeros(n_env)
    if trust.any():
        w = weight[trust][:, None]
        theta_rig = wh_smooth(
            np.sum(w * (ang[trust] / ks[trust][:, None]), axis=0) / w.sum(), lam_th, tap
        )
    for r in range(n_rot):
        theta[r] = theta_rig
        sel = trust & (rot == r)
        if per_rotor and sel.any():
            w = weight[sel][:, None]
            rest = ang[sel] - ks[sel][:, None] * theta_rig[None, :]
            theta[r] = theta_rig + wh_smooth(
                np.sum(w * (rest / ks[sel][:, None]), axis=0) / w.sum(), lam_th, tap
            )

    if with_psi:
        bw = bw_psi_hz(ks, bw_psi_slope, bw_psi_max, bw_psi_min)
        for m in np.flatnonzero(strong):
            rest = ang[m] - ks[m] * theta[rot[m]]
            psi[m] = wh_smooth(rest, wh_lambda(float(bw[m]), fs_env), tap)

    bands = track_bands(np.asarray(k))
    return PhaseSplit(
        theta_rig=theta_rig,
        theta=theta,
        psi=psi,
        diag={
            "n_tracks": int(n_tracks),
            "n_trust": int(trust.sum()),
            "n_strong": int(strong.sum()),
            "k_trust": int(k_trust),
            "with_psi": bool(with_psi),
            "conc_by_band": {
                nm: (round(float(conc[s].mean()), 4) if s.any() else None)
                for nm, s in bands.items()
            },
            "theta_rig_rms_rad": round(float(np.sqrt(np.mean(theta_rig**2))), 5),
            "theta_rms_rad": [round(float(np.sqrt(np.mean(t**2))), 5) for t in theta],
            "theta_rate_rms_rev_s": [
                round(float(np.sqrt(np.mean(theta_rate(t, fs_env) ** 2))), 5) for t in theta
            ],
            "psi_rms_rad_by_band": {
                nm: (round(float(np.sqrt(np.mean(psi[s] ** 2))), 5) if s.any() else None)
                for nm, s in bands.items()
            },
            "max_abs_step_rad": round(float(step_max.max()), 4),
        },
    )


def theta_rate(theta: Any, fs_env: float) -> np.ndarray:
    """``d theta / dt / (2 pi)`` in rev/s — the shaft correction as a RATE.

    The rate is the gauge-free form of the shaft correction (the additive
    constant of a phase differentiates away), which is what makes it the thing
    to carry across windows and to add to the labels.
    """
    t = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    dr = np.gradient(t, axis=-1) * float(fs_env) / (2.0 * np.pi)
    # ``np.gradient`` is ONE-SIDED at the two ends, so the first and last frame
    # carry a different estimator from every other frame. Hold the nearest
    # interior value instead: the stitch is built on this array, and a single
    # wrong frame at a window edge becomes a seam.
    if dr.shape[-1] > 2:
        dr[:, 0] = dr[:, 1]
        dr[:, -1] = dr[:, -2]
    return dr.reshape(np.shape(theta))


# ---------------------------------------------------------------------------
# the alternation


@dataclass(frozen=True)
class JointConfig:
    """Knobs of the joint alternation. The defaults are the shipped v3 arm."""

    iters: int = 3
    #: Harmonic cap of the trustable set, per iteration (the annealing ladder).
    #: It starts at 3 and not at 10, and the reason is the ENVELOPE BAND, not
    #: the unwrap. Harmonic ``k`` of a shaft wandering by ``sigma_r`` rev/s is a
    #: frequency modulation of bandwidth about ``k sigma_r`` Hz, and a band of
    #: ``B`` Hz distorts its phase badly once ``k sigma_r > B / 2`` — at
    #: ``sigma_r`` 0.6 and ``B`` 3 that is ``k`` 2.5. Measured on the synthetic
    #: fixture: a ladder starting at 6 recovers 43 % of the true shaft phase in
    #: three iterations, a ladder starting at 3 recovers 100 % (correlation
    #: 0.999). Each fold shrinks the residual wander, so the next rung can be
    #: far higher.
    k_trust: tuple[int, ...] = (3, 12, 80)
    #: From which iteration (1 based) the per-track ``psi`` is estimated.
    psi_from_iter: int = 2
    bw_theta_hz: float = 1.5
    bw_psi_slope: float = LINEWIDTH_HZ_PER_K
    bw_psi_max: float = 8.0
    #: Floor of the per-track phase-correction band — see :func:`bw_psi_hz`.
    bw_psi_min: float = 1.5
    conc_min: float = 0.5
    per_rotor_theta: bool = True
    whiten: bool = True
    whiten_clamp_db: float = 15.0
    #: Keep each track's ACHIEVED bandwidth at the tuned v2 value under
    #: whitening, by carrying the track's mean weight into ``rho^2`` as well.
    #: Without it a track whose floor is 15 dB loud has its data term scaled
    #: down but its curvature prior unchanged, so its effective band narrows by
    #: the same factor and the envelope is over-smoothed — measured on the
    #: fixture, that alone put the k1-9 residual comb at 12.6 dB against 4.3 dB
    #: unwhitened. Whitening then does what it is for (the relative trust
    #: between coupled tracks and across time) and does not silently retune the
    #: bandwidth law it inherited.
    bandwidth_neutral: bool = True
    psd_n_fft: int = 4096
    psd_blocks: int = 4
    psd_n_cep: int = 40
    profile_n_fft: int = 8192
    profile_order_step: float = 0.005
    #: Compute the order-cell profile of every iteration's residual, not the
    #: last one only. It is the acceptance instrument, so it is on by default.
    profile_every_iter: bool = True
    #: STFT length of the stochastic channel (:func:`stochastic_split`). It is
    #: the floor block's own length by default, so the two read one grid.
    stochastic_n_fft: int = 4096
    #: Read the stochastic split's floor per FRAME, by interpolating ``log S``
    #: in time between the block centers, instead of per block
    #: (:func:`stochastic_split`). The block floor is one spectrum per ~4 s,
    #: which is a modelling statement that the wash is stationary over that
    #: span; the per-frame ``P / S`` quantiles say it is not. Its visible cost is
    #: a SEAM — much of the comb band sits within ~1 dB of the floor, so the clip
    #: at ``a = 1`` toggles whole bands on and off across a block boundary and
    #: the demonstration spectrograms carry rectangular patches with vertical
    #: edges exactly on the block grid. Off by default, because the shipped
    #: numbers were all produced on the block floor.
    stochastic_floor_interp: bool = False
    #: Add the exact Gaussian envelope MARGINALIZATION to the objective readout
    #: (:func:`map_objective`, ``total_marginal``). Profiling the envelopes pays
    #: no rent for their freedom, so absorption is free; the marginal term is
    #: what charges for it. It is a pure reading — it moves no product — and it
    #: is off by default because the pseudo-determinant costs one banded
    #: factorization per envelope length.
    marginal: bool = False
    #: Add the H-AWARE data term to the objective readout
    #: (:func:`map_objective`, ``data_h`` / ``total_h``). The coherent envelopes
    #: cannot carry the line FLANKS, so the profiled data term charges every
    #: hypothesis for the same flank energy and the true trajectory gains
    #: nothing by sitting on it. The H-aware term gives the noise model a
    #: comb-shaped nuisance inside the hypothesis's OWN search regions, so a
    #: trajectory that explains the humps stops paying for them and a coverage
    #: fan that opens regions on empty floor gains nothing. Also a pure reading
    #: — it moves no product — and off by default because it holds the whole
    #: measured spectrogram and smooths it once.
    h_aware: bool = False
    #: Give the floor a per-FRAME scale before the readout
    #: (:func:`map_objective`, ``floor_gamma``). ``S`` is fitted per 4-second
    #: block, so a recording with non-stationary rotor wash — DREGON's gusts —
    #: makes a whole span of frames pay a Whittle misfit that no comb hypothesis
    #: caused, and the block floor plus its rent then move with however each
    #: hypothesis's solve happened to distribute that energy. A pure reading,
    #: like the two above, and off by default.
    adaptive_floor: bool = False
    #: Constrain the H-aware nuisance to LORENTZIANS pinned at the hypothesis's
    #: own line positions (:func:`_h_aware_data`). Only meaningful together with
    #: ``h_aware``, and off by default because it costs one non-negative least
    #: squares per (channel, frame).
    h_lorentzian: bool = False
    #: THE v4 master switch — one model instead of a stack of bolt-ons. It
    #: changes four things at once, and they only make sense together:
    #:
    #: - the floor block fits ``S`` and the line powers ``H`` JOINTLY with no
    #:   mask, on the ORIGINAL audio (:func:`fit_floor_powers`);
    #: - block A opens its bands to the physical law ``max(b0, 0.6 k)`` Hz with
    #:   NO spacing cap (:func:`v4_rho2_gain`) and carries a proper amplitude
    #:   prior ``c0 S / H`` that makes the overlapping system definite again
    #:   (:func:`v4_ridge`);
    #: - the objective becomes the MARGINAL Whittle likelihood ``J_v4``
    #:   (:func:`map_objective`), which has no envelope term and no separate
    #:   rent;
    #: - the stochastic WOLA split is not run — the comb channel already carries
    #:   the line flanks, so the decomposition is two channels and a
    #:   subtraction.
    #:
    #: Off by default, and off is the v3 arm call for call. The four measure
    #: bolt-ons above (``marginal``, ``h_aware``, ``adaptive_floor``,
    #: ``h_lorentzian``) are what v4 replaces; they stay for the comparison runs
    #: and are not developed further.
    v4: bool = False
    #: Smoothness length scale of the v4 floor, in hertz (:func:`floor_lambda`).
    v4_b_f_hz: float = FLOOR_LENGTH_HZ
    #: Round budget of the v4 ``(S, H)`` fit, per (microphone, block).
    v4_rounds: int = FLOOR_POWER_ROUNDS
    #: Strength of the v4 amplitude prior (:data:`V4_RIDGE_C0`).
    v4_ridge_c0: float = V4_RIDGE_C0
    #: Floor under that prior, relative to the group's own data curvature
    #: (:data:`RIDGE_FLOOR_FRAC`). It is what keeps a group of overlapping wide
    #: bands positive definite, so it is not optional in practice — 0 is
    #: available only so a test can measure what happens without it.
    v4_ridge_floor_frac: float = RIDGE_FLOOR_FRAC
    #: Floor of the v4 envelope bandwidth law, in hertz (:data:`V4_BW0_HZ`).
    v4_b0_hz: float = V4_BW0_HZ
    #: Use the v4 BAND LAW (:func:`v4_rho2_gain`, the physical linewidth with no
    #: spacing cap). Turning it off keeps every other part of v4 — the joint
    #: ``(S, H)`` fit, the amplitude prior, ``J_v4`` — and takes the envelope
    #: bands from whatever schedule the state was seeded with instead.
    #:
    #: It is a real modelling position and not only an escape hatch. The
    #: amplitude targets are ``(S, H)``, and those come from the F1 fit, which
    #: never looks at a band; the uncapped bands are a refinement of the
    #: WAVEFORM channel, and a waveform channel is only identifiable where the
    #: rotor spreads allow it. On a twin rig whose pairs sit 0.43 and 0.81 rev/s
    #: apart the uncapped system is genuinely singular at ``k_hi`` 83 — four
    #: combs that close cannot be told apart by a 12-second window — so those
    #: windows legitimately use the capped law and lose nothing the model is
    #: actually estimating.
    v4_band_law: bool = True

    def k_cap(self, it: int) -> int:
        """Trustable harmonic cap of iteration ``it`` (1 based)."""
        if not self.k_trust:
            return 10
        return int(self.k_trust[min(max(it, 1) - 1, len(self.k_trust) - 1)])


@dataclass
class JointResult:
    """One window's joint decomposition."""

    env: Envelopes  # .x = g e^{j psi} against .phase = phi_hat + theta
    theta_env: np.ndarray  # (R, J) accumulated shaft correction, radians
    psi: np.ndarray  # (M, J) accumulated per-track correction, radians
    psd: SmoothPSD  # the final floor model
    residual: np.ndarray  # (C, T)
    track_energy: np.ndarray  # (M,)
    #: v4 only: the per-line powers fitted beside the floor. They are the
    #: model's own amplitude parameters, so they are a first-class PRODUCT and
    #: not a diagnostic — the generator's targets come from here.
    h_powers: HPowers | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    #: Regime 3, when :func:`stochastic_block` ran: the comb-locked part of
    #: ``residual`` that no coherent envelope can carry. ``None`` is the shipped
    #: default, and then the residual IS the broadband channel.
    stochastic: np.ndarray | None = None


@dataclass(frozen=True)
class JointState:
    """Everything one window's alternation accumulates — the ``meta["joint"]`` seam.

    Three blocks read it and each returns a NEW state (frames are immutable, and
    so is this). What it holds is the model's own vocabulary: the carrier the
    corrections are measured against, the two coherent corrections ``theta`` and
    ``psi``, the floor model, and the last solve's products.

    ``carrier`` is the audio-rate rate array, and it is held rather than
    re-derived. The alternation is CONDITIONED on one carrier for the whole
    window — ``theta`` is a correction on top of it — so a block that
    re-interpolated the frame's trajectory would be solving against a slightly
    different measurement every round.
    """

    cfg: FVKConfig
    jcfg: JointConfig
    k_hi: int
    carrier: np.ndarray  # (R, T) rev/s at audio rate
    n_t: int
    rho2_gain: np.ndarray | None = None
    t_start_s: float = 0.0
    theta: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    psi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    psd: SmoothPSD | None = None
    #: v4 only: the line powers the floor block fitted beside ``psd``. Block A
    #: reads them as its amplitude prior, so the two always travel together.
    h_powers: HPowers | None = None
    env: Envelopes | None = None
    x_eff: np.ndarray | None = None
    residual: np.ndarray | None = None
    track_energy: np.ndarray | None = None
    #: Regime 3 (:func:`stochastic_block`): the comb-locked part of the residual.
    #: ``residual`` is NEVER rewritten by that block, so the broadband channel is
    #: ``residual - stochastic`` and the three-way identity is exact by
    #: subtraction, exactly as the residual itself is exact by subtraction.
    stochastic: np.ndarray | None = None
    #: How many block-A solves have run. It is the annealing ladder's index and
    #: the ``psi_from_iter`` gate, so the blocks need no loop counter of their own.
    n_solves: int = 0

    @property
    def vk(self) -> Any:
        return self.cfg.vk_config(int(self.k_hi))

    @property
    def grid(self) -> tuple[int, float]:
        """``(stride, fs_env)`` of the envelope grid."""
        return env_stride(self.vk)

    @property
    def tracks(self) -> tuple[np.ndarray, np.ndarray]:
        """``(rotor, k)`` of every track — the identical set in every window."""
        return _track_table(int(self.carrier.shape[0]), self.vk.k_min, int(self.k_hi))

    @property
    def n_env(self) -> int:
        return len(range(0, int(self.n_t), self.grid[0]))

    def k_trust(self) -> int:
        """The trustable harmonic cap of the NEXT phase split (the ladder)."""
        return min(self.jcfg.k_cap(self.n_solves), int(self.k_hi))


def joint_state(
    r_audio: Any,
    cfg: FVKConfig,
    *,
    k_hi: int,
    n_t: int,
    jcfg: JointConfig | None = None,
    bw_schedule: BandwidthSchedule | None = None,
    rho_scale: float = 1.0,
    t_start_s: float = 0.0,
) -> JointState:
    """THE seed of an alternation: zero corrections, no floor, the tuned gain.

    ``k_hi`` is the harmonic cap of the whole RECORDING (never of the window),
    so every window holds the identical track set and the banks can be stitched
    track by track.
    """
    jc = JointConfig() if jcfg is None else jcfg
    if int(jc.iters) < 1:
        raise ValueError(f"JointConfig.iters must be at least 1, got {jc.iters}")
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    state = JointState(
        cfg=cfg,
        jcfg=jc,
        k_hi=int(k_hi),
        carrier=r,
        n_t=int(n_t),
        rho2_gain=track_rho2_gain(r, int(k_hi), cfg, bw_schedule, float(rho_scale)),
        t_start_s=float(t_start_s),
    )
    _, k = state.tracks
    return replace(
        state,
        theta=np.zeros((int(r.shape[0]), state.n_env)),
        psi=np.zeros((len(k), state.n_env)),
    )


def solve_block(state: JointState, audio: Any) -> JointState:
    """Block A — one whitened VK solve at the current carrier, floor and ``psi``.

    The coherent phases fold into the CARRIER (exact) and the smooth floor
    collapses to one scalar per track and frame, so the banded structure of the
    solver survives untouched: the three optional arguments of
    :func:`tracking.vk_envelopes` are the whole seam. What comes back is the
    EFFECTIVE envelope ``g e^{j psi}`` against ``env.phase``, which is what every
    v2 consumer already knows how to read.
    """
    jc = state.jcfg
    y = np.asarray(audio, dtype=np.float64)
    stride, _ = state.grid
    rotor, k = state.tracks
    r_env = state.carrier[:, ::stride][:, : state.n_env]
    t_env = np.arange(state.n_env, dtype=np.float64) * stride / float(state.cfg.sr)
    weight = (
        whiten_weights(state.psd, k, rotor, r_env, t_env, clamp_db=jc.whiten_clamp_db)
        if (jc.whiten and state.psd is not None)
        else None
    )
    # The v4 band law REPLACES the v2 schedule the state was seeded with: the
    # spacing cap it carries exists to keep an improper prior identifiable, and
    # the amplitude prior below is proper.
    gain = (
        v4_rho2_gain(
            state.carrier,
            int(state.k_hi),
            state.cfg,
            b0_hz=jc.v4_b0_hz,
            slope_hz_per_k=jc.bw_psi_slope,
        )
        if (jc.v4 and jc.v4_band_law)
        else state.rho2_gain
    )
    if weight is not None and jc.bandwidth_neutral:
        mean_u2 = np.mean(weight**2, axis=-1)
        gain = mean_u2 if gain is None else gain * mean_u2
    ridge = (
        v4_ridge(state.psd, state.h_powers, k, rotor, r_env, t_env, weight, c0=jc.v4_ridge_c0)
        if (jc.v4 and state.psd is not None and state.h_powers is not None)
        else None
    )
    # The three joint hooks, passed as a mapping: they are what turns the v2
    # solver into block A (see vk_envelopes' docstring). The v4 arm adds a
    # fourth, the amplitude prior.
    hooks: dict[str, Any] = {
        "phase_offset": upsample_env(state.theta, int(y.shape[-1]), stride),
        "env_rotation": state.psi,
        "data_weight": weight,
        "ridge": ridge,
        "ridge_floor_frac": float(jc.v4_ridge_floor_frac) if ridge is not None else 0.0,
    }
    env: Envelopes = vk_envelopes(
        y, state.carrier, state.vk, k_hi=int(state.k_hi), rho2_gain=gain, **hooks
    )
    x_eff = env.x * np.exp(1j * state.psi)[None, :, :]
    recon, track_e = reconstruct(x_eff, k, rotor, env.phase, stride)
    return replace(
        state,
        env=env,
        x_eff=x_eff,
        residual=y - recon,
        track_energy=track_e,
        n_solves=state.n_solves + 1,
    )


def split_block(state: JointState) -> tuple[JointState, PhaseSplit]:
    """Block B — the phase split, folded into the accumulated corrections.

    The increments and not the totals come back in the :class:`PhaseSplit`, so a
    caller reads what THIS round learned; the state carries the sum.
    """
    jc = state.jcfg
    if state.env is None:
        raise ValueError("split_block: no envelope bank yet — run solve_block first")
    rotor, k = state.tracks
    split = split_phases(
        state.env.x,
        k,
        rotor,
        state.env.valid,
        state.grid[1],
        k_trust=state.k_trust(),
        conc_min=jc.conc_min,
        bw_theta_hz=jc.bw_theta_hz,
        bw_psi_slope=jc.bw_psi_slope,
        bw_psi_max=jc.bw_psi_max,
        bw_psi_min=jc.bw_psi_min,
        per_rotor=jc.per_rotor_theta,
        with_psi=state.n_solves >= int(jc.psi_from_iter),
    )
    return replace(state, theta=state.theta + split.theta, psi=state.psi + split.psi), split


def floor_block(state: JointState, audio: Any) -> JointState:
    """Block C — the masked smooth floor of what the model has not explained.

    Before the first solve there is no residual, so the floor is fitted on the
    AUDIO; after one, on the residual. Both are the same statement — the floor
    is read between the lines of whatever is left.

    Under ``JointConfig.v4`` neither sentence holds any more. The fit is the
    JOINT one (:func:`fit_floor_powers`), so there is no mask and no "between
    the lines"; and it runs on the ORIGINAL audio every round, because the model
    explains the lines through ``H`` and subtracting an estimate first is what
    makes a hard-EM alternation biased. What DOES change between rounds is the
    carrier: the lines are placed at the shaft correction block B has learned so
    far, which is the only reason a refit is worth making at all. The previous
    round's floor is the warm start, so the later refits are cheap.
    """
    jc = state.jcfg
    if jc.v4:
        stride, fs_env = state.grid
        dr = theta_rate(state.theta, fs_env) if state.theta.size else state.theta
        carrier = state.carrier
        if dr.size:
            carrier = carrier + upsample_env(dr, int(state.n_t), stride)
        psd, hp = fit_floor_powers(
            np.asarray(audio, dtype=np.float64),
            float(state.cfg.sr),
            carrier,
            int(state.k_hi),
            n_fft=jc.psd_n_fft,
            n_blocks=jc.psd_blocks,
            b_f_hz=float(jc.v4_b_f_hz),
            slope_hz_per_k=float(jc.bw_psi_slope),
            rounds=int(jc.v4_rounds),
            warm=state.psd,
            t_start_s=float(state.t_start_s),
        )
        return replace(state, psd=psd, h_powers=hp)
    src = np.asarray(audio if state.residual is None else state.residual, dtype=np.float64)
    return replace(
        state,
        psd=masked_smooth_psd(
            src,
            float(state.cfg.sr),
            state.carrier,
            int(state.k_hi),
            n_fft=jc.psd_n_fft,
            n_blocks=jc.psd_blocks,
            n_cep=jc.psd_n_cep,
            t_start_s=float(state.t_start_s),
        ),
    )


# ---------------------------------------------------------------------------
# regime 3: the STOCHASTIC comb channel


@dataclass
class StochasticSplit:
    """The residual, split once more into a comb-locked part and a floor."""

    stochastic: np.ndarray  # (C, T) the comb-locked part
    broadband: np.ndarray  # (C, T) what is left — residual - stochastic
    psd: SmoothPSD  # the floor model the gain was taken against
    diag: dict[str, Any] = field(default_factory=dict)


def comb_lines(rate: Any, k_hi: int) -> tuple[np.ndarray, np.ndarray]:
    """``(line frequencies Hz, harmonic index)`` of one frame's whole comb.

    Row major over rotors, so the harmonic index simply tiles — the same
    ``(k r_r)`` construction :func:`masked_smooth_psd` masks with, handed back
    with its ``k`` beside it because the band law is written in ``k``.
    """
    r = np.asarray(rate, dtype=np.float64).ravel()
    ks = np.arange(1, int(k_hi) + 1, dtype=np.float64)
    return (ks[None, :] * r[:, None]).ravel(), np.tile(ks, r.size)


def line_half_widths(
    lines: Any, k: Any, *, slope_hz_per_k: float = LINEWIDTH_HZ_PER_K, min_half_hz: float = 0.0
) -> np.ndarray:
    """``min(slope k, local line spacing)`` Hz per line, floored at ``min_half_hz``.

    The band a REGIME-3 line needs is its own incoherent linewidth, ``0.6 k`` Hz
    by the measured shaft-wander law. The spacing cap is what keeps one line's
    band from reaching over its neighbour: a coherent envelope is identifiable
    only out to about 0.4 of the local spacing, and a STOCHASTIC band that
    crosses a neighbour would take that neighbour's energy twice. The floor is
    the readout's own resolution — a band narrower than one bin selects nothing.

    ``lines`` and ``k`` are one flat array each (:func:`comb_lines`); the
    spacing is read on the sorted line set, so it is the distance to the nearest
    OTHER line whichever rotor that line belongs to.
    """
    f = np.asarray(lines, dtype=np.float64)
    ks = np.asarray(k, dtype=np.float64)
    n = int(f.size)
    if n < 2:
        sep = np.full(n, np.inf)
    else:
        order = np.argsort(f)
        d = np.diff(f[order])
        near = np.empty(n, dtype=np.float64)
        near[0], near[-1] = d[0], d[-1]
        if n > 2:
            near[1:-1] = np.minimum(d[:-1], d[1:])
        sep = np.empty(n, dtype=np.float64)
        sep[order] = near
    return np.maximum(np.minimum(float(slope_hz_per_k) * ks, sep), float(min_half_hz))


def stochastic_half_widths(
    k: Any,
    *,
    slope_hz_per_k: float = LINEWIDTH_HZ_PER_K,
    width_factor: float = STOCHASTIC_WIDTH_FACTOR,
    min_half_hz: float = 0.0,
) -> np.ndarray:
    """``max(width_factor * slope * k, min_half_hz)`` Hz — regime 3's OWN law.

    Deliberately NOT :func:`line_half_widths`. That law caps a line's band at
    the local line spacing, and the cap is there to protect COHERENT
    identifiability: two envelopes whose passbands overlap cannot be told apart,
    and a band that reaches over its neighbour would take the neighbour's energy
    twice. Neither reason applies to a per-bin POWER split. Its bands are
    unioned before any gain is taken, so nothing is counted twice, and its gain
    is per bin, so a bin the region covers but no line occupies reads
    ``P ~ S`` and passes through at a gain of one.

    The cap COSTS, though, and the FLY124 v3c report is the measurement: at
    high ``k`` the ``0.6 k`` Hz flanks of four interleaved combs merge into one
    continuous rotor-locked field BETWEEN the nominal lines, the spacing-capped
    bands never reach it, and the capped bands then read a ``band_floor_share``
    of 1.02 — they saw floor-level edges and nothing else. Two linewidths
    (``+/- 3`` FWHM, 90 % of a Lorentzian's power) reaches the field.
    """
    ks = np.asarray(k, dtype=np.float64)
    return np.maximum(float(width_factor) * float(slope_hz_per_k) * ks, float(min_half_hz))


def _smooth_power(power: np.ndarray, n_frames: int, n_bins: int) -> np.ndarray:
    """Boxcar the ``(C, frames, F)`` periodogram in frequency, then in time.

    ``mode="nearest"`` at both edges, which holds the end frame and the end bin
    rather than pulling the estimate toward zero — a gain built on a
    zero-padded average would be a spurious subtraction at the two ends.
    """
    from scipy.ndimage import uniform_filter1d

    out = power
    if int(n_bins) > 1:
        out = uniform_filter1d(out, int(n_bins), axis=-1, mode="nearest")
    if int(n_frames) > 1:
        out = uniform_filter1d(out, int(n_frames), axis=-2, mode="nearest")
    return out


def _wola_plan(n_t: int, n_fft: int, hop: int) -> tuple[int, int, np.ndarray]:
    """``(pad, padded length, frame starts)`` of an EXACTLY invertible WOLA.

    The signal is padded by a whole frame on each side and the frame grid is
    extended to cover the padding, so every original sample lies under the FULL
    window-square sum and the weighted overlap-add divides that sum out exactly.
    That is what makes the round trip an identity at any window and any hop,
    without asking a COLA constant to be true.
    """
    pad = int(n_fft)
    want = pad + int(n_t) + pad
    n_frames = 1 + max(0, -(-(want - int(n_fft)) // int(hop)))
    starts = np.arange(n_frames, dtype=np.int64) * int(hop)
    return pad, int(starts[-1]) + int(n_fft), starts


def stochastic_split(
    residual: Any,
    sr: float,
    r_audio: Any,
    k_hi: int,
    *,
    psd: SmoothPSD | None = None,
    n_fft: int = 4096,
    hop: int | None = None,
    slope_hz_per_k: float = LINEWIDTH_HZ_PER_K,
    width_factor: float = STOCHASTIC_WIDTH_FACTOR,
    psd_blocks: int = 4,
    psd_n_cep: int = 40,
    t_start_s: float = 0.0,
    frames_per_chunk: int = 64,
    floor_time_interp: bool = False,
) -> StochasticSplit:
    """Split the residual into a comb-locked STOCHASTIC channel and a floor.

    Regime 3 of the decomposition. Regimes 1 and 2 — the coherent envelope at
    the annotated carrier and the coherent envelope at the CORRECTED carrier —
    both need a band narrow against the local line spacing, because two coherent
    envelopes whose passbands overlap are not identifiable (a cluster run at a
    cap of 1.5 times the line separation went singular). Above about ``k`` 10 the
    measured linewidth ``0.6 k`` Hz is wider than that cap, so the flanks of
    every line are energy that IS comb locked and that no coherent envelope can
    carry. This is the channel that carries it, and it is a POWER split rather
    than a waveform fit: nothing here has a phase model.

    The split is a PER-BIN AMPLITUDE gain — spectral subtraction to the floor —
    on the floor block's own short-time grid. Per frame every predicted line
    ``k r_r(t)`` gets the search region
    ``+/- max(2 * 0.6 k, one bin)`` Hz (:func:`stochastic_half_widths`), the
    regions are UNIONED, and inside the union every bin gets

        a(f, frame) = clip(sqrt(S(f) / P~(f, frame)), 0, 1)

    with ``a = 1`` outside. The broadband channel is ``a Y`` and the stochastic
    channel is ``(1 - a) Y``. ``S`` is the fitted floor and ``P~`` the MEASURED
    per-bin periodogram, boxcar smoothed over ``P_SMOOTH_FRAMES`` frames and
    ``P_SMOOTH_BINS`` bins; both are power spectral DENSITIES on the
    :func:`masked_smooth_psd` normalization, so the ratio is scale free.

    Two things about that gain, and both are the fix for a MEASURED failure of
    the flat per-band Wiener gain it replaces (full-scale DREGON,
    ``results/vk_decompose_v3c``):

    - **Amplitude, not power.** The conditional mean ``E[floor | Y] =
      (S / P) Y`` is the right answer to a different question: it is the
      MINIMUM-ERROR estimate, not a typical realization of the floor, and its
      power ``S^2 / P`` sits BELOW the floor at every strong line. That is a
      notch, and the acceptance gate — the order-cell excess of the BROADBAND
      channel — reads a notch and a line alike (k1-9 excess went UP, 4.3 % ->
      7.8 % retained, with the profile peak moving to +/- 0.5 orders, which is
      the signature of a dent between the lines and not of comb energy). The
      amplitude gain ``sqrt(S / P~)`` leaves the broadband channel with power
      ``S`` in expectation, which is what the gate and the downstream
      floor-training targets both want.
    - **Per bin, not per band.** One gain over a union band that spans many
      lines scales the whole region uniformly, so the comb PATTERN survives at
      reduced amplitude (order-cell depth 0.386 -> 0.380 dB at k10-24). The
      smoothed periodogram ``P~`` carries the line SHAPE, so a per-bin gain
      concentrates the removal at the line cores by itself — no line model, no
      Lorentzian fit, and no knob beyond the two fixed smoothing widths.

    The gain is applied to the residual's own short-time transform and
    overlap-added back (:func:`_wola_plan` — the round trip is an identity), and
    the broadband channel is the SUBTRACTION ``residual - stochastic``, so

        coherent + stochastic + broadband = original

    holds to float roundoff by construction, exactly as the residual itself
    does. Per-bin gains are linear through the same transform, so the identity
    is untouched by any of this.

    ``psd`` is the floor to score against; ``None`` fits one on the residual
    with the same block C the alternation uses, which is what a caller with a
    WHOLE recording wants — a window's floor does not describe a minute.

    ``floor_time_interp`` reads ``S`` at each FRAME instead of at its block, by
    linear interpolation of ``log S`` in time between the block centers. The
    floor is fitted per block — about four seconds of recording each — and
    taking it as a step function is a modelling statement that the rotor wash is
    stationary over one, which the per-frame ``P / S`` quantiles say is false.
    The visible cost is a SEAM: because much of the comb band sits within about
    1 dB of the floor, the clip at ``a = 1`` makes whole bands toggle between
    "nothing taken" and "something taken" across one block boundary, and the
    demonstration spectrograms then carry rectangular patches whose vertical
    edges land exactly on the boundaries of the block grid. Interpolating moves
    the floor continuously instead, so a band fades in rather than switching on.
    It is OFF by default: the block-constant floor is what every published
    number was produced with, and the two paths must not be confused.
    """
    y = np.atleast_2d(np.asarray(residual, dtype=np.float64))
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_ch, n_t = int(y.shape[0]), int(y.shape[-1])
    n_fft = int(n_fft)
    step = int(n_fft // 4 if hop is None else hop)
    if step < 1 or step > n_fft:
        raise ValueError(f"hop {step} must be in 1..{n_fft}")
    use = (
        masked_smooth_psd(
            y,
            float(sr),
            r,
            int(k_hi),
            n_fft=n_fft,
            n_blocks=int(psd_blocks),
            n_cep=int(psd_n_cep),
            t_start_s=float(t_start_s),
        )
        if psd is None
        else psd
    )
    log_s = _floor_on_grid(use, float(sr), n_fft)
    if log_s.shape[0] != n_ch:
        raise ValueError(f"the floor has {log_s.shape[0]} microphones, the residual has {n_ch}")
    n_bl = int(log_s.shape[1])
    s_lin = np.exp(log_s)  # (C, B, F)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    df = float(sr) / n_fft

    pad, n_pad, starts = _wola_plan(n_t, n_fft, step)
    # The per-frame floor lookup, when it is asked for: the two bracketing blocks
    # and one weight per frame, so the inner loop stays two array reads. The
    # block CENTERS come from the floor's own ``t_block`` (which is
    # ``t_start_s`` plus the center of each block, so the window-relative form is
    # the difference); a floor that does not carry one center per block — an
    # older or hand-built ``SmoothPSD`` — falls back to the centers of the even
    # block grid ``masked_smooth_psd`` builds.
    interp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if floor_time_interp and n_bl > 1:
        tb = np.asarray(use.t_block, dtype=np.float64) - float(t_start_s)
        if tb.size != n_bl or not np.all(np.isfinite(tb)) or np.any(np.diff(tb) <= 0):
            tb = (np.arange(n_bl, dtype=np.float64) + 0.5) * (n_t / float(n_bl)) / float(sr)
        # The frame's own time is its CENTER, not its start: the periodogram it
        # is compared against is the whole windowed frame.
        t_frame = (starts.astype(np.float64) - pad + 0.5 * n_fft) / float(sr)
        pos = np.interp(np.clip(t_frame, tb[0], tb[-1]), tb, np.arange(n_bl, dtype=np.float64))
        lo_b = np.floor(pos).astype(int)
        interp = (lo_b, np.minimum(lo_b + 1, n_bl - 1), pos - lo_b)
    win = np.hanning(n_fft)
    w2 = win**2
    scale = 1.0 / (float(sr) * float(w2.sum()))
    yp = np.zeros((n_ch, n_pad), dtype=np.float64)
    yp[:, pad : pad + n_t] = y
    num = np.zeros((n_ch, n_pad), dtype=np.float64)
    den = np.zeros(n_pad, dtype=np.float64)
    off = np.arange(n_fft)
    chunk = max(1, int(frames_per_chunk))

    # Pass 1: the MEASURED periodogram of every frame, then the boxcar. The
    # whole ``(C, frames, F)`` surface is built because the smoother reaches
    # across chunk boundaries and a halo is a second way to get the same array
    # wrong; float32 keeps it to ~130 MB on a minute of eight-channel audio,
    # which is far more precision than a 26 %-noisy power estimate carries.
    p_meas = np.empty((n_ch, int(starts.size), int(freq.size)), dtype=np.float32)
    for c0 in range(0, starts.size, chunk):
        sub = starts[c0 : c0 + chunk]
        seg = yp[:, sub[:, None] + off] * win[None, None, :]
        p_meas[:, c0 : c0 + sub.size] = (np.abs(np.fft.rfft(seg, axis=-1)) ** 2 * scale).astype(
            np.float32
        )
    p_smooth = _smooth_power(p_meas, P_SMOOTH_FRAMES, P_SMOOTH_BINS)
    del p_meas

    n_bins = 0
    n_bands = 0
    band_power = 0.0
    band_floor = 0.0
    gain_sum = 0.0
    for c0 in range(0, starts.size, chunk):
        sub = starts[c0 : c0 + chunk]
        seg = yp[:, sub[:, None] + off] * win[None, None, :]
        spec = np.fft.rfft(seg, axis=-1)  # (C, frames, F)
        # The STOCHASTIC gain, ``1 - a``: zero everywhere the union bands do not
        # reach, so the broadband channel is the input untouched there.
        gain = np.zeros(spec.shape, dtype=np.float64)
        for i, s in enumerate(sub):
            s0 = int(s) - pad
            a0 = int(np.clip(s0, 0, max(n_t - 1, 0)))
            b0 = int(np.clip(s0 + n_fft, 1, max(n_t, 1)))
            rate = r[:, a0:b0].mean(axis=-1)
            lines, ks = comb_lines(rate, int(k_hi))
            half = stochastic_half_widths(
                ks, slope_hz_per_k=slope_hz_per_k, width_factor=width_factor, min_half_hz=df
            )
            idx = np.flatnonzero(_line_mask(freq, lines, half))
            if idx.size == 0:
                continue
            p_bin = np.maximum(p_smooth[:, c0 + i, :][:, idx].astype(np.float64), 1e-300)
            if interp is None:
                # The block-constant floor: one spectrum for the whole block the
                # frame starts in. Bit for bit what every shipped run used.
                bl = min((max(s0, 0) * n_bl) // max(n_t, 1), n_bl - 1)
                s_bin = s_lin[:, bl, :][:, idx]
            else:
                lo_b, hi_b, frac = interp
                j = c0 + i
                s_bin = np.exp(
                    log_s[:, lo_b[j], :][:, idx] * (1.0 - frac[j])
                    + log_s[:, hi_b[j], :][:, idx] * frac[j]
                )
            amp = np.clip(np.sqrt(s_bin / p_bin), 0.0, 1.0)
            gain[:, i, idx] = 1.0 - amp
            # Band accounting. The union bands no longer carry a gain, only the
            # SEARCH REGION they delimit, so they are counted and not applied.
            brk = np.flatnonzero(np.diff(idx) > 1)
            n_bins += int(idx.size)
            n_bands += int(brk.size) + 1
            band_power += float(p_bin.sum())
            band_floor += float(s_bin.sum())
            gain_sum += float((1.0 - amp).sum())
        out = np.fft.irfft(spec * gain, n=n_fft, axis=-1) * win[None, None, :]
        for i, s in enumerate(sub):
            num[:, int(s) : int(s) + n_fft] += out[:, i]
            den[int(s) : int(s) + n_fft] += w2

    stoch = (num / np.maximum(den, 1e-300)[None, :])[:, pad : pad + n_t]
    broadband = y - stoch
    e_res = float((y**2).sum())
    e_st = float((stoch**2).sum())
    cells = max(n_bins * n_ch, 1)
    return StochasticSplit(
        stochastic=stoch,
        broadband=broadband,
        psd=use,
        diag={
            "n_fft": n_fft,
            "hop": step,
            "n_frames": int(starts.size),
            "k_hi": int(k_hi),
            "slope_hz_per_k": float(slope_hz_per_k),
            "width_factor": float(width_factor),
            "min_half_hz": round(df, 4),
            "p_smooth_frames": int(P_SMOOTH_FRAMES),
            "p_smooth_bins": int(P_SMOOTH_BINS),
            # Which floor the gain was taken against in TIME: one spectrum per
            # block, or the block centers interpolated per frame. It is a
            # property of the run and a report that does not name it cannot be
            # reproduced.
            "floor_time_interp": bool(interp is not None),
            "n_bands_per_frame": round(n_bands / max(starts.size, 1), 2),
            "band_bin_fraction": round(n_bins / max(starts.size * freq.size, 1), 5),
            # Mean STOCHASTIC amplitude gain ``1 - a`` over the band bins: 0 is
            # "the region was all floor and nothing was taken", 1 is "the region
            # was all line". It is an AMPLITUDE now, not a power gain.
            "mean_gain": round(gain_sum / cells, 5),
            # Floor over measured power, summed over the band bins. Above 1 it
            # says the search region saw no excess at all to remove.
            "band_floor_share": round(band_floor / max(band_power, 1e-300), 5),
            "residual_energy": e_res,
            "stochastic_energy": e_st,
            "broadband_energy": float((broadband**2).sum()),
            "stochastic_fraction": round(e_st / max(e_res, 1e-30), 6),
            # The overlap-add weight the division undoes. It is positive
            # everywhere by construction (:func:`_wola_plan`), and a caller that
            # sees a zero here is looking at a broken round trip, not at a gain.
            "wola_min_weight": round(float(den[pad : pad + n_t].min()), 8),
        },
    )


def stochastic_block(state: JointState) -> tuple[JointState, StochasticSplit]:
    """Regime 3 as a BLOCK: the last solve's residual, split once more.

    It runs AFTER the alternation, on the state the last block-A solve left, and
    it reads the floor that same alternation fitted. Nothing it does feeds back:
    the residual on the state is not rewritten, only ``stochastic`` is added
    beside it, so every earlier reading of the window is untouched.
    """
    if state.residual is None or state.psd is None:
        raise ValueError("stochastic_block: nothing solved yet — run solve_block first")
    split = stochastic_split(
        state.residual,
        float(state.cfg.sr),
        state.carrier,
        int(state.k_hi),
        psd=state.psd,
        n_fft=int(state.jcfg.stochastic_n_fft),
        slope_hz_per_k=float(state.jcfg.bw_psi_slope),
        t_start_s=float(state.t_start_s),
        floor_time_interp=bool(state.jcfg.stochastic_floor_interp),
    )
    return replace(state, stochastic=split.stochastic), split


def solve_report(state: JointState, audio: Any, *, profile: bool) -> dict[str, Any]:
    """The READING of one block-A solve — the numbers a report carries.

    Separate from :func:`solve_block` because it is measurement and not
    estimation: the energy shares, the whitened flatness against the floor the
    solve used, and (when asked) the order-cell band table of the residual.
    """
    jc = state.jcfg
    if state.residual is None or state.track_energy is None or state.psd is None:
        raise ValueError("solve_report: nothing solved yet")
    y = np.asarray(audio, dtype=np.float64)
    total = max(float((y**2).sum()), 1e-30)
    out: dict[str, Any] = {
        "iter": int(state.n_solves),
        "k_trust": state.k_trust(),
        "residual_fraction": round(float((state.residual**2).sum() / total), 6),
        "track_fraction": round(float(state.track_energy.sum() / total), 6),
        "psd_masked_frac": state.psd.n_masked_frac,
        "whitened": bool(jc.whiten),
        "flatness": whitened_flatness(state.residual, float(state.cfg.sr), state.psd),
    }
    if profile:
        out["order_cell"] = order_cell_bands(
            state.residual,
            float(state.cfg.sr),
            state.carrier,
            n_fft=jc.profile_n_fft,
            order_step=jc.profile_order_step,
            k_max=int(state.k_hi),
        )
    return out


#: How many half widths of a Lorentzian each of its design columns is carried
#: out to (:func:`_lorentzian_design`). A Lorentzian at 8 HWHM is down by a
#: factor of 65, and the column is a SHAPE fitted on top of a floor of 1 in the
#: same units, so the truncated tail is far below what the data resolves — the
#: measured move in ``data_h`` on a 16 s, 8-microphone, ``k_hi`` 40 window is
#: 0.04 %. It is NOT a speed measure: the untruncated design is if anything
#: faster (5.9 s against 8.9 s on that window, because a dense column set costs
#: the active-set solve fewer iterations). What it buys is a LOCAL basis, and
#: with it the rule that a line whose support reaches no region bin has no
#: column at all — otherwise a far-off line contributes an almost flat column
#: and the fit gains a pedestal it can raise anywhere.
LORENTZ_SUPPORT_HWHM = 8.0

#: Clip floor of the per-frame floor gain (:func:`_adaptive_floor_gain`). A gain
#: this small means the frame carries a thousandth of the block's power, which
#: is a silent frame and not a measurement; without the clip its rent is
#: unbounded below.
FLOOR_GAIN_MIN = 1e-3


def _adaptive_floor_gain(
    p_meas: np.ndarray, log_s: np.ndarray, block_of: np.ndarray, frames_per_chunk: int = 64
) -> np.ndarray:
    """``(C, frames)`` robust per-frame gain of the block floor — a profiled scale.

    The floor block fits ONE spectrum per time block (four blocks over a
    16-second window), so a non-stationary broadband source — rotor wash under a
    gust, which is what DREGON's low band carries — leaves a whole span of frames
    an order of magnitude above the block's fitted level. Those frames then pay a
    Whittle misfit of ``P / S`` per cell that no comb hypothesis caused and that
    no comb hypothesis can remove, and, worse, the block floor and its rent both
    move with however a given hypothesis's solve happened to distribute that
    energy — the term becomes a lottery on a quantity the measure is not about.

    The fix is one profiled parameter per (channel, frame): the noise model
    becomes ``gamma(c, t) S_c(f, block(t))``. Under the Whittle model each
    ``P / S`` cell is an Exp(1) deviate, whose MEDIAN is ``ln 2`` and not 1, so
    the median of the ratio over frequency divided by ``ln 2`` is an unbiased
    scale on a floor that is already correct — and, being a median, it reads the
    BULK of the band rather than the comb lines sitting in the top few percent of
    it. The frame axis is then smoothed with the readout's own boxcar
    (``P_SMOOTH_FRAMES``), because a per-frame median is still a noisy statistic
    and the gusts it is built for are many frames long.

    The rent pays for it: ``rent`` becomes ``sum log(gamma S)``, so invoking a
    loud frame costs ``n_freq log gamma`` — the natural Occam charge, and the
    reason the scale cannot simply be minimized away.
    """
    from scipy.ndimage import uniform_filter1d

    n_ch, n_fr, _ = (int(v) for v in p_meas.shape)
    s_lin = np.exp(np.asarray(log_s, dtype=np.float64))
    gain = np.empty((n_ch, n_fr), dtype=np.float64)
    chunk = max(1, int(frames_per_chunk))
    for c0 in range(0, n_fr, chunk):
        sl = slice(c0, min(c0 + chunk, n_fr))
        gain[:, sl] = np.median(p_meas[:, sl] / s_lin[:, block_of[sl], :], axis=-1)
    gain /= np.log(2.0)
    gain = uniform_filter1d(gain, int(P_SMOOTH_FRAMES), axis=-1, mode="nearest")
    return np.maximum(gain, FLOOR_GAIN_MIN)


def _lorentzian_design(
    freq: np.ndarray, bins: np.ndarray, lines: np.ndarray, half: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(A, kept)`` — the truncated Lorentzian design on one frame's region bins.

    Column ``l`` is ``1 / (1 + ((f - f_l) / gamma_l)^2)`` sampled on ``bins``,
    zero beyond ``LORENTZ_SUPPORT_HWHM`` half widths, and a line whose support
    reaches no region bin at all is dropped — it has no column to fit and
    carrying it would only give the solver a free zero.
    """
    f = np.asarray(freq, dtype=np.float64)[np.asarray(bins, dtype=np.int64)]
    lo = np.asarray(lines, dtype=np.float64)[None, :]
    hw = np.asarray(half, dtype=np.float64)[None, :]
    d = (f[:, None] - lo) / hw
    a = np.where(np.abs(d) <= LORENTZ_SUPPORT_HWHM, 1.0 / (1.0 + d * d), 0.0)
    kept = np.flatnonzero(a.any(axis=0))
    return np.ascontiguousarray(a[:, kept]), kept


def _h_aware_data(
    p_meas: np.ndarray,
    log_s: np.ndarray,
    block_of: np.ndarray,
    starts: np.ndarray,
    r_audio: Any,
    k: np.ndarray,
    *,
    sr: float,
    n_fft: int,
    slope_hz_per_k: float = LINEWIDTH_HZ_PER_K,
    width_factor: float = STOCHASTIC_WIDTH_FACTOR,
    frames_per_chunk: int = 64,
    floor_gain: np.ndarray | None = None,
    lorentzian: bool = False,
) -> dict[str, Any]:
    """The H-AWARE data term of :func:`map_objective` — the stochastic comb in it.

    ``p_meas`` is the ``(C, frames, F)`` MEASURED power spectral density the data
    term itself summed, ``log_s`` the ``(C, B, F)`` fitted floor and ``block_of``
    the frame-to-block map, so the two halves of the readout see one grid by
    construction. What this adds is the comb-shaped nuisance ``H``:

    - the SEARCH REGIONS are the hypothesis's own — per frame, ``k r_r(t)`` for
      every ``k`` the track set names, half widths from
      :func:`stochastic_half_widths` (floored at one bin), unioned by
      :func:`_line_mask`, exactly as regime 3 delimits them
    - ``H = max(0, P~ - S)`` inside them and ``0`` outside, with ``P~`` the
      regime-3 boxcar of ``p_meas`` (:func:`_smooth_power`) — the same bounded
      estimator the split's gain is built on, so the nuisance can never claim
      more than the measured excess

    and returns ``data_h = sum [ P/(S+H) + log(S+H) - log S ]``, which is the
    Whittle pair's own change: the ``log S`` half is subtracted back out because
    ``map_objective`` already carries it as ``rent``. At ``H = 0`` every cell
    reduces to ``P / S`` and ``data_h`` IS ``data``.

    ``floor_gain`` is the per-frame floor scale of :func:`_adaptive_floor_gain`.
    When it is given, ``S`` above means ``gamma(c, t) S_c(f, block(t))``
    everywhere in this function — in the hump's own baseline and in the ``log S``
    that is taken back out — because that is the floor ``rent`` then carries.

    ``lorentzian`` constrains the SHAPE of ``H``. The shape-free hump above is
    ``max(0, P~ - S)``, which explains ANY excess a region happens to cover; on a
    rig whose four dense combs make every hypothesis's regions blanket the band
    above ``k`` 10 — DREGON free flight — the term is then hypothesis
    independent and discriminates nothing. Under this flag ``H`` is instead a
    NON-NEGATIVE mixture of Lorentzians pinned at the hypothesis's OWN line
    positions, with the measured half width ``0.6 k`` Hz (the linewidth law
    itself, never the region's 3.0 multiplier — that multiplier delimits where
    the fit may look, not how wide a line is):

        H(f, frame) = sum_l a_l(c, frame) / (1 + ((f - k_l r(t)) / (0.6 k_l))^2)

    with ``a >= 0`` fitted per (channel, frame) by non-negative least squares
    against the same clipped excess the shape-free hump used. A hypothesis can
    then only claim energy that sits ON its comb with the physical line shape: a
    wrong comb's bumps fall BETWEEN its lines, no non-negative amplitude puts a
    Lorentzian peak there, and the amplitudes come back near zero. The design is
    shared by the channels of one frame, so it is built once per frame.
    """
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_ch, n_fr, n_f = (int(v) for v in p_meas.shape)
    n_fft = int(n_fft)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    df = float(sr) / n_fft
    # The harmonic set is the PINNED one the tracks already name, so two
    # hypotheses on one window open regions over the identical ``k`` range.
    k_hi = int(np.max(k)) if k.size else 0
    log_gain = None if floor_gain is None else np.log(np.asarray(floor_gain, dtype=np.float64))

    # The smoother reaches across chunk boundaries, so the whole surface is
    # smoothed at once and only the ARITHMETIC is chunked — a halo is a second
    # way to get the same array wrong (the same call regime 3 makes).
    p_smooth = _smooth_power(p_meas, P_SMOOTH_FRAMES, P_SMOOTH_BINS)
    region = np.zeros((n_fr, n_f), dtype=bool)
    rates = np.empty((n_fr, int(r.shape[0])), dtype=np.float64)
    for i in range(n_fr):
        s0 = int(starts[i])
        rates[i] = r[:, s0 : s0 + n_fft].mean(axis=-1)
        lines, kk = comb_lines(rates[i], k_hi)
        half = stochastic_half_widths(
            kk, slope_hz_per_k=slope_hz_per_k, width_factor=width_factor, min_half_hz=df
        )
        region[i] = _line_mask(freq, lines, half)

    fit: dict[str, Any] = {}
    if lorentzian:
        hump_all, fit = _lorentzian_hump(
            p_smooth, log_s, log_gain, block_of, region, freq, rates, k_hi, slope_hz_per_k, df
        )

    data_h = 0.0
    h_energy = 0.0
    chunk = max(1, int(frames_per_chunk))
    for c0 in range(0, n_fr, chunk):
        sl = slice(c0, min(c0 + chunk, n_fr))
        ls = log_s[:, block_of[sl], :]  # (C, frames, F)
        if log_gain is not None:
            ls = ls + log_gain[:, sl, None]
        s_lin = np.exp(ls)
        hump = (
            hump_all[:, sl]
            if lorentzian
            else np.where(region[sl][None, :, :], np.maximum(p_smooth[:, sl] - s_lin, 0.0), 0.0)
        )
        sh = np.maximum(s_lin + hump, 1e-300)
        data_h += float(np.sum(p_meas[:, sl] / sh)) + float(np.sum(np.log(sh) - ls))
        h_energy += float(np.sum(hump))
    return {
        "data_h": data_h,
        # Cells the regions cover, on the same (channel, frame, bin) counting as
        # ``n_cells`` — the region itself is the same for every microphone.
        "h_cells": int(region.sum()) * n_ch,
        "h_energy": h_energy,
        **({"h_fit": fit} if fit else {}),
    }


def _lorentzian_hump(
    p_smooth: np.ndarray,
    log_s: np.ndarray,
    log_gain: np.ndarray | None,
    block_of: np.ndarray,
    region: np.ndarray,
    freq: np.ndarray,
    rates: np.ndarray,
    k_hi: int,
    slope_hz_per_k: float,
    df: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``(H, diagnostics)`` — the SHAPE-CONSTRAINED nuisance of :func:`_h_aware_data`.

    One non-negative least squares per (channel, frame) against the clipped
    excess ``max(0, P~ - S)`` on that frame's region bins, with the design
    :func:`_lorentzian_design` builds from the frame's OWN line positions. The
    design is built once per frame and reused by every channel, which is where
    most of the saving is on an eight-microphone window.

    The diagnostics are the three numbers that say whether the constraint did
    anything: how many of the offered lines took a positive amplitude, how much
    of the excess the mixture failed to explain, and what it cost.
    """
    from scipy.optimize import nnls

    tic = perf_counter()
    n_ch, n_fr, _ = (int(v) for v in p_smooth.shape)
    hump = np.zeros_like(p_smooth)
    s_all = np.exp(log_s)
    n_pos = 0.0
    n_amp = 0.0
    res_sq = 0.0
    tgt_sq = 0.0
    for i in range(n_fr):
        bins = np.flatnonzero(region[i])
        if bins.size == 0:
            continue
        lines, kk = comb_lines(rates[i], k_hi)
        # The LINEWIDTH law itself: the region's 3.0 multiplier says where the
        # fit may look, this says how wide one line is.
        half = np.maximum(float(slope_hz_per_k) * kk, df)
        design, kept = _lorentzian_design(freq, bins, lines, half)
        if kept.size == 0:
            continue
        base = s_all[:, int(block_of[i]), :][:, bins]
        if log_gain is not None:
            base = base * np.exp(log_gain[:, i])[:, None]
        for c in range(n_ch):
            y = np.maximum(p_smooth[c, i, bins] - base[c], 0.0)
            amp, _ = nnls(design, y)
            hump[c, i, bins] = design @ amp
            n_pos += float(np.count_nonzero(amp > 0.0))
            n_amp += float(amp.size)
            res_sq += float(np.sum((y - hump[c, i, bins]) ** 2))
            tgt_sq += float(np.sum(y**2))
    return hump, {
        "active_line_frac": round(n_pos / max(n_amp, 1.0), 4),
        "fit_residual_share": round(res_sq / max(tgt_sq, 1e-300), 4),
        "wall_s": round(perf_counter() - tic, 2),
    }


def _v4_data(
    p_meas: np.ndarray,
    log_s: np.ndarray,
    block_of: np.ndarray,
    starts: np.ndarray,
    r_audio: Any,
    hp: HPowers,
    *,
    sr: float,
    n_fft: int,
    slope_hz_per_k: float = LINEWIDTH_HZ_PER_K,
) -> dict[str, Any]:
    """The MARGINAL Whittle pair of the v4 model — the whole data term of ``J_v4``.

    ``M = S + sum_l H_l L_l`` with the lines at the carrier's own positions in
    each FRAME and the fitted powers of that frame's block, and the readout is
    ``sum [P / M + log M]`` over every cell. It is one term and not two: the
    v3 split into ``data`` and ``rent`` existed because ``S`` was the whole noise
    model and its logarithm was the only Occam charge there was. Here the comb is
    IN the noise model, so a hypothesis that puts a line where there is none pays
    for it through the same ``log M`` that a floor pays through, and separating
    the two halves would only invite reading one without the other.

    The line positions move from frame to frame while the powers are held over
    their block. That asymmetry is the model's: ``H`` is a level, which four
    seconds of recording can measure, and a position is a trajectory, which they
    cannot.
    """
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_ch, n_fr, n_f = (int(v) for v in p_meas.shape)
    n_fft = int(n_fft)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    df = float(sr) / n_fft
    k_hi = int(np.max(hp.k)) if hp.k.size else 0
    total = 0.0
    h_energy = 0.0
    for i in range(n_fr):
        s0 = int(starts[i])
        b = int(block_of[i])
        rate = r[:, s0 : s0 + n_fft].mean(axis=-1)
        lines, kk = comb_lines(rate, k_hi)
        half = np.maximum(float(slope_hz_per_k) * kk, df)
        s_lin = np.exp(log_s[:, b, :])  # (C, F)
        m = s_lin.copy()
        bins = np.flatnonzero(_line_mask(freq, lines, LORENTZ_SUPPORT_HWHM * half))
        if bins.size and lines.size == hp.h.shape[-1]:
            design, kept = _lorentzian_design(freq, bins, lines, half)
            if kept.size:
                # (bins, kept) @ (kept, C) — one matmul for every microphone.
                m[:, bins] += (design @ hp.h[:, b, :][:, kept].T).T
                h_energy += float(np.sum(m[:, bins] - s_lin[:, bins]))
        m = np.maximum(m, 1e-300)
        total += float(np.sum(p_meas[:, i] / m)) + float(np.sum(np.log(m)))
    return {
        "data_v4": total,
        "h_energy_v4": h_energy,
        "n_lines_v4": int(hp.k.size),
    }


def map_objective(
    residual: Any,
    sr: float,
    psd: SmoothPSD,
    *,
    x: Any,
    k: Any,
    bw_track: Any,
    theta: Any,
    psi: Any,
    fs_env: float,
    n_fft: int = 4096,
    hop: int | None = None,
    p: int = 2,
    bw_theta_hz: float = 1.5,
    bw_psi_slope: float = LINEWIDTH_HZ_PER_K,
    bw_psi_max: float = 8.0,
    bw_psi_min: float = 1.5,
    frames_per_chunk: int = 64,
    logdet_posterior: float | None = None,
    h_carrier: Any | None = None,
    h_slope_hz_per_k: float = LINEWIDTH_HZ_PER_K,
    h_width_factor: float = STOCHASTIC_WIDTH_FACTOR,
    adaptive_floor: bool = False,
    h_lorentzian: bool = False,
    v4_powers: HPowers | None = None,
    v4_carrier: Any | None = None,
    v4_b_f_hz: float = FLOOR_LENGTH_HZ,
) -> dict[str, Any]:
    """THE converged MAP objective of the joint model, term by term::

        J = sum_{c,f,t} [ P_c(f,t) / S_c(f,t) + log S_c(f,t) ]
          + lam_theta   sum_i     ||D2 theta_i||^2
          + sum_{i,k} lam_psi(k)  ||D2 psi_{i,k}||^2
          + sum_{c,i,k}  rho_k^2  ||D2 A_{c,i,k}||^2

    It is the number the three blocks of the alternation each decrease in their
    own coordinate, so it is the one reading that compares two runs on ONE
    window whatever they did differently — a different trajectory hypothesis
    above all, which is what makes the decomposition a MEASURE of a trajectory
    and not only a product of one.

    The four terms, and where each weight comes from:

    ``data`` and ``rent``
        The Gaussian negative log likelihood of colored noise, on the SAME
        short-time grid the floor block reads (:func:`masked_smooth_psd`):
        ``P`` is the framed Hann power spectral density of the residual, ``S``
        is ``exp(log_s)`` of the fitted floor, and a frame takes the floor of
        the time block it falls in. ``rent`` is what the floor pays for being
        loud, and it is what stops the pair from being minimized by a floor
        that swallows everything. NOTE that the two are summed over EVERY cell,
        including the cells the floor fit masked out: the mask decides what the
        floor is FITTED on, never what it is scored on.
    ``phase_priors``
        The Whittaker-Henderson curvature priors of block B, at the weights
        block B smoothed with — :func:`wh_lambda` of ``bw_theta_hz`` for the
        shaft and of :func:`bw_psi_hz` per track for ``psi``.
    ``envelope_prior``
        The VK curvature prior of block A, at the selectivity the solver
        ACTUALLY used: ``rho_m^2`` is read back from ``bw_track`` through the
        solver's own Tuma relation, so every clamp, schedule and whitening gain
        that moved a track's band is already in it. The envelope is the SOLVED
        variable ``g`` (``Envelopes.x``), not the effective ``g e^{j psi}``.

    ``logdet_posterior`` turns the PROFILED objective above into the MARGINAL
    one. Profiling substitutes the envelopes' best value back, which is free —
    it pays no rent for the envelopes' own freedom — and that is exactly why a
    hypothesis with more usable envelopes can win by absorption. Integrating the
    envelopes out instead of profiling them adds the Gaussian correction

        J_marg = J + 0.5 (log det M - log det' R)

    with ``M`` the whitened banded posterior precision block A factorizes
    (``Envelopes.logdet``, one channel's worth, taken here ``n_channels`` times
    because every channel is a right-hand side against that same system) and
    ``R`` the improper envelope prior ``blkdiag(rho_m^2 D2^T D2)``, read through
    the pseudo-determinant :func:`prior_logdet`. It comes back as
    ``marginal_correction`` and ``total_marginal``; ``total`` is untouched.

    ONE scaling is not in that formula and has to be: ``data`` and ``rent`` are
    summed over frames that overlap, so each sample is under ``n_fft / hop`` of
    them and the pair is that many times ONE likelihood, while the correction is
    one likelihood's worth. ``marginal_correction`` therefore carries
    ``marginal_redundancy = n_fft / hop`` and reports it, so a caller who wants
    the bare formula divides it back out. It is not a tuning knob: without it
    the correction is out-scaled two to one on the shipped grid and a spurious
    track pays less than it gains (measured — see
    ``tests/tracking/test_marginal_objective.py``).

    Four hypothesis-INDEPENDENT constants are dropped, and the readout is
    therefore valid ONLY for comparing hypotheses on the same window, the same
    audio and the same cells — which is what ``scripts/joint_rescore.py`` pins:

    1. The Gaussian volume factors of the marginalization (``pi^N`` or
       ``(2 pi)^(N/2)``, by convention), with ``N`` the envelope count.
    2. The null-space volume of the improper prior — two directions per track,
       so it is fixed by the track count, which the pinned ``k_hi`` fixes.
    3. The real-parameterization factor. A circular COMPLEX Gaussian would carry
       ``1`` rather than ``0.5`` in front of the pair; that scales the whole
       correction and cannot change its sign, so the Occam ordering is the same.
    4. The solver's own scaling convention (the data term enters the normal
       equations as ``w`` with a right-hand side of ``2 w z``), and the ``1e-8``
       ridge plus any ``diag_scale`` PD repair that are inside ``M``.

    ``h_carrier`` switches on the H-AWARE data term, which is the STOCHASTIC
    COMB written into the likelihood. The coherent envelopes cannot carry the
    ``0.6 k`` Hz flanks of a line (regime 3, :func:`stochastic_split`), so the
    profiled ``J`` charges EVERY hypothesis for that flank energy and the true
    trajectory gains nothing by sitting on it — measured on five frozen windows,
    which is why adversarial coverage fans win the data term on three of them.
    The H-aware term gives the noise model a comb-shaped nuisance ``H`` on top
    of the smooth floor:

        H(f, frame) = max(0, P~(f, frame) - S(f))   inside the hypothesis's own
        H(f, frame) = 0                             comb SEARCH REGIONS

    with ``P~`` the measured power on this readout's own grid, smoothed by the
    regime-3 boxcar (``P_SMOOTH_FRAMES`` x ``P_SMOOTH_BINS``, edge mode
    nearest), and the regions the union of ``k r_r(t) +/- 3 x 0.6 k`` Hz per
    frame (:func:`stochastic_half_widths`) over the SAME harmonic set the tracks
    name. The hypothesis controls only WHERE the regions are; ``H`` inside them
    is the profiled nuisance, bounded by the estimator the split already uses.
    The data term is then the Whittle pair with ``S + H`` in place of ``S``, and
    ``data_h`` is reported so that

        total_h = total - data + data_h ,

    i.e. ``data_h = sum [ P / (S + H) + log(S + H) - log S ]`` — the pair's own
    change, with the ``log S`` half that ``rent`` already carries taken back out,
    so no logarithm is counted twice and ``rent`` stays what it was. At ``H = 0``
    it is exactly ``data``, cell by cell.

    The asymmetry is honest and it is the whole mechanism. ``H`` is fitted from
    the same data it explains, so INSIDE a hypothesis's regions the term is
    nearly hypothesis independent: on floor-level bins ``H`` is zero up to the
    small positive bias of clipping a noisy ``P~ - S`` at zero, and on a real
    hump it absorbs the hump whoever asked for it. The discrimination is where a
    real hump exists and a hypothesis's regions MISS it — those cells pay
    ``P / S + log S`` in full, exactly as before. A fan that opens regions on
    empty floor buys nothing, and a trajectory whose regions sit on the humps
    stops paying for them.

    ``adaptive_floor`` gives ``S`` one profiled scale per (channel, frame),
    ``S_eff = gamma(c, t) S_c(f, block(t))``, and every term of the readout then
    reads ``S_eff``: the data term, the rent, and the ``H`` above. It is the
    collapsed form of a heavier noise model — a Student-t floor, or one variance
    per cell — profiled down to the single degree of freedom the data actually
    supports, and :func:`_adaptive_floor_gain` estimates it as a median ratio
    over frequency. The reason it exists is that a block floor is CONSTANT over
    four seconds while the rotor wash under a gust is not, so on DREGON a whole
    span of frames pays a Whittle misfit no comb hypothesis caused, and the block
    floor plus its rent then move with however a hypothesis's own solve happened
    to distribute that energy. Under the scale the gusts stop dominating the data
    term, and ``rent`` grows by ``n_freq log gamma`` per frame — the natural
    Occam charge for invoking a loud frame, which is what keeps the scale from
    simply being minimized away. The marginal correction is untouched by it. The
    gain's distribution comes back as ``floor_gamma``.

    ``h_lorentzian`` constrains the SHAPE of ``H`` (see :func:`_h_aware_data`).
    The shape-free hump explains any excess a region covers, which is exactly
    hypothesis independent on a rig whose combs make every hypothesis's regions
    blanket the band; a non-negative Lorentzian mixture pinned at the
    hypothesis's OWN lines, at the measured ``0.6 k`` Hz half width, can only
    claim energy that sits on that comb with the physical line shape. It needs
    ``h_carrier`` and does nothing without it.

    Returns the total, the four components, the two sub-terms of the phase
    prior, and ``n_cells`` — the caller normalizes, because a window's cell
    count depends on its length and its microphone count. Under ``h_carrier`` it
    also returns ``data_h`` / ``total_h`` beside them (never instead of them),
    plus ``h_cells`` (region cells) and ``h_energy`` (the summed ``H``).

    ``v4_powers`` with ``v4_carrier`` switches on ``J_v4``, THE objective of the
    v4 model — and unlike everything above it, it is not a correction on top of
    the profiled readout but a different objective::

        J_v4 = sum_{c,f,t} [ P / M + log M ]        M = S + sum_l H_l L_l
             + lam_theta ||D2 theta||^2 + sum lam_psi(k) ||D2 psi||^2
             + lam_f sum ||D2 log S||^2

    Three differences from ``total``, and each is a deletion:

    - there is no ENVELOPE term. The line processes are integrated out, not
      profiled, so there is nothing to charge curvature on and nothing to
      correct for with a pseudo-determinant. What the ``marginal`` bolt-on
      approximated, this does not need.
    - there is no separate ``rent``. The comb is in the noise model, so
      ``log M`` is the one Occam charge and it covers both the floor and the
      lines. A hypothesis that opens a line on empty floor pays through it.
    - the floor's smoothness penalty is EXPLICIT (:func:`floor_penalty`),
      because ``S`` is a fitted parameter here and not a projection.

    It comes back as ``data_v4`` / ``floor_penalty`` / ``total_v4`` beside every
    v3 column, never instead of them, so one run compares the two.

    It is a pure OBSERVER: it reads finished arrays and touches nothing the
    solver will read again, so switching it on cannot move a single product.
    """
    y = np.atleast_2d(np.asarray(residual, dtype=np.float64))
    n_ch, n_t = int(y.shape[0]), int(y.shape[-1])
    step = int(n_fft // 2 if hop is None else hop)
    starts = frame_starts(n_t, int(n_fft), step)
    log_s = _floor_on_grid(psd, float(sr), int(n_fft))
    if log_s.shape[0] != n_ch:
        raise ValueError(f"the floor has {log_s.shape[0]} microphones, the residual has {n_ch}")
    n_bl = int(log_s.shape[1])
    n_f = int(log_s.shape[-1])

    data = 0.0
    rent = 0.0
    n_frames = 0
    h_aware: dict[str, Any] = {}
    gamma_read: dict[str, Any] = {}
    v4_read: dict[str, Any] = {}
    if starts.size and n_t >= int(n_fft):
        block_of = np.minimum((starts * n_bl) // max(1, n_t), n_bl - 1)
        s_lin = np.exp(log_s)
        scale = 1.0 / (float(sr) * float((np.hanning(int(n_fft)) ** 2).sum()))
        # The measured surface is KEPT only for the readouts that need every
        # frame at once — the H-aware one (its smoother reaches across chunk
        # boundaries) and the adaptive floor (its gain is smoothed over frames).
        want_all = h_carrier is not None or adaptive_floor or v4_powers is not None
        p_all = np.empty((n_ch, int(starts.size), n_f), dtype=np.float64) if want_all else None
        done = 0
        for sub, chunk in stft_power(y, starts, int(n_fft), frames_per_chunk):
            sel = block_of[done : done + sub.size]
            power = chunk * scale
            if not adaptive_floor:
                data += float(np.sum(power / s_lin[:, sel, :]))
            if p_all is not None:
                p_all[:, done : done + sub.size] = power
            done += sub.size
        # The rent is the same value for every frame of a block, so it is
        # counted per block instead of per frame — exactly the same sum.
        counts = np.bincount(block_of, minlength=n_bl).astype(np.float64)
        rent = float(np.sum(counts[None, :, None] * log_s))
        n_frames = int(starts.size)
        gain = None
        if adaptive_floor and p_all is not None:
            gain = _adaptive_floor_gain(p_all, log_s, block_of, frames_per_chunk)
            chunk_n = max(1, int(frames_per_chunk))
            for c0 in range(0, n_frames, chunk_n):
                sl = slice(c0, min(c0 + chunk_n, n_frames))
                data += float(
                    np.sum(p_all[:, sl] / (s_lin[:, block_of[sl], :] * gain[:, sl, None]))
                )
            # ``sum log S_eff`` in its cheap form: the block-counted rent above
            # plus one log per (channel, frame) taken across the whole band.
            rent += float(n_f) * float(np.sum(np.log(gain)))
            gamma_read = {
                "floor_gamma": {
                    "mean": round(float(np.mean(gain)), 4),
                    "p05": round(float(np.percentile(gain, 5.0)), 4),
                    "p50": round(float(np.percentile(gain, 50.0)), 4),
                    "p95": round(float(np.percentile(gain, 95.0)), 4),
                    "max": round(float(np.max(gain)), 4),
                }
            }
        if v4_powers is not None and v4_carrier is not None and p_all is not None:
            v4_read = _v4_data(
                p_all,
                log_s,
                block_of,
                starts,
                v4_carrier,
                v4_powers,
                sr=float(sr),
                n_fft=int(n_fft),
                slope_hz_per_k=float(h_slope_hz_per_k),
            )
        if h_carrier is not None and p_all is not None:
            h_aware = _h_aware_data(
                p_all,
                log_s,
                block_of,
                starts,
                h_carrier,
                np.asarray(k, dtype=np.float64),
                sr=float(sr),
                n_fft=int(n_fft),
                slope_hz_per_k=float(h_slope_hz_per_k),
                width_factor=float(h_width_factor),
                frames_per_chunk=frames_per_chunk,
                floor_gain=gain,
                lorentzian=bool(h_lorentzian),
            )

    ks = np.asarray(k, dtype=np.float64)
    lam_theta = wh_lambda(float(bw_theta_hz), float(fs_env))
    theta_prior = lam_theta * _curvature(theta)
    lam_psi = np.array(
        [
            wh_lambda(float(b), float(fs_env))
            for b in bw_psi_hz(ks, bw_psi_slope, bw_psi_max, bw_psi_min)
        ]
    )
    psi_prior = _curvature(psi, weight=lam_psi)
    rho2 = np.array([_tuma_rho(float(b), float(fs_env), int(p)) for b in np.asarray(bw_track)]) ** 2
    env_prior = _curvature(np.asarray(x, dtype=np.complex128), weight=rho2)

    phase_priors = theta_prior + psi_prior
    total = data + rent + phase_priors + env_prior
    xa = np.asarray(x)
    n_env = int(xa.shape[-1])
    # Channels are right-hand sides against ONE system, so the full posterior
    # and the full prior are both that block repeated once per microphone.
    n_ch_env = int(xa.shape[0]) if xa.ndim == 3 else 1
    marginal: dict[str, Any] = {}
    if logdet_posterior is not None:
        ld_prior = prior_logdet(bw_track, n_env, float(fs_env), int(p))
        # The data term is a sum over frames that OVERLAP, so every sample is
        # under n_fft / hop of them and `data` + `rent` are that many times ONE
        # likelihood. The correction is one likelihood's worth, so it carries
        # the same factor or the two halves of the readout are not commensurate
        # — measured on a spurious-track fixture, without it the correction is
        # out-scaled two to one and the Occam property fails.
        redundancy = float(n_fft) / float(step)
        # Extensive sums of order 1e5 to 1e6 — never rounded, because two
        # hypotheses are compared by a DIFFERENCE a rounding would quantize.
        correction = 0.5 * n_ch_env * redundancy * (float(logdet_posterior) - ld_prior)
        marginal = {
            "marginal_correction": correction,
            "total_marginal": total + correction,
            "logdet_posterior": float(logdet_posterior) * n_ch_env,
            "logdet_prior": ld_prior * n_ch_env,
            "marginal_redundancy": redundancy,
            "n_env": n_env,
        }
    # NOT rounded, unlike every other reading in this module. The four terms
    # span nine orders of magnitude on a real window (the rent alone is tens of
    # thousands of cells times a log spectral density), and two hypotheses are
    # compared by their DIFFERENCE, which a rounded total would quantize away.
    if h_aware:
        # ``data_h`` already carries the pair's whole change, so the H-aware
        # total swaps it for ``data`` and leaves every other term alone.
        h_aware["total_h"] = total - data + float(h_aware["data_h"])
    if v4_read:
        # J_v4 is built from its OWN terms and not from ``total``: it has no
        # envelope term and no separate rent, so there is nothing of the
        # profiled total to reuse beyond the two phase priors.
        pen = floor_penalty(psd, float(v4_b_f_hz))
        v4_read["floor_penalty"] = pen
        v4_read["total_v4"] = float(v4_read["data_v4"]) + phase_priors + pen
    return {
        "total": total,
        "data": data,
        "rent": rent,
        "phase_priors": phase_priors,
        "envelope_prior": env_prior,
        "theta_prior": theta_prior,
        "psi_prior": psi_prior,
        **marginal,
        **h_aware,
        **gamma_read,
        **v4_read,
        "n_cells": int(n_ch * n_frames * n_f),
        "n_frames": n_frames,
        "n_freq": n_f,
        "n_channels": n_ch,
        "n_fft": int(n_fft),
    }


def d2_pseudo_logdet(n_env: int) -> float:
    """``log det'`` of ``D2^T D2`` at length ``n_env`` — the non-zero part only.

    ``D2`` is the ``(n-2, n)`` second difference, so ``D2^T D2`` is singular in
    exactly two directions (a constant and a ramp) and its pseudo-determinant is
    the product of the ``n - 2`` remaining eigenvalues. Those are the eigenvalues
    of ``D2 D2^T``, which is the ``(n-2, n-2)`` pentadiagonal ``[1, -4, 6, -4, 1]``
    and is positive definite — so this is one banded Cholesky and not an
    eigendecomposition, and it costs ``O(n)`` instead of ``O(n^3)``.

    Cached, because it depends on the envelope LENGTH and nothing else: every
    hypothesis on one window asks for the identical number.
    """
    return _d2_pseudo_logdet_cached(int(n_env))


@lru_cache(maxsize=64)
def _d2_pseudo_logdet_cached(n_env: int) -> float:
    from scipy.linalg import cholesky_banded

    m = int(n_env) - 2
    if m < 1:
        return 0.0
    ab = np.zeros((3, m), dtype=np.float64)
    ab[2] = 6.0
    ab[1, 1:] = -4.0
    ab[0, 2:] = 1.0
    return 2.0 * float(np.sum(np.log(np.asarray(cholesky_banded(ab, lower=False)[2]))))


def prior_logdet(bw_track: Any, n_env: int, fs_env: float, p: int = 2) -> float:
    """``log det'`` of the envelope prior ``blkdiag(rho_m^2 D2^T D2)``.

    The prior is IMPROPER — ``D2`` kills a constant and a ramp, so each track's
    block has two null directions and no determinant at all. What a marginal
    likelihood needs and what this returns is the PSEUDO-determinant,

        sum_m [ (T - 2) log(rho_m^2) ] + M log det'(D2^T D2) ,

    which drops the null space's (infinite) volume. That volume is the same for
    every hypothesis scored on one window — same track count, same envelope
    length — so it cancels in a comparison, which is the only use the marginal
    readout has. ``rho_m^2`` comes back out of the ACHIEVED bandwidth through
    the solver's own Tuma relation, exactly as the envelope prior's does.
    """
    bw = np.asarray(bw_track, dtype=np.float64)
    n = int(n_env)
    if bw.size == 0 or n < 3:
        return 0.0
    rho2 = np.array([_tuma_rho(float(b), float(fs_env), int(p)) for b in bw]) ** 2
    return float((n - 2) * np.sum(np.log(rho2)) + bw.size * d2_pseudo_logdet(n))


def _floor_on_grid(psd: SmoothPSD, sr: float, n_fft: int) -> np.ndarray:
    """``(C, B, F)`` log floor on the readout's own ``rfftfreq`` grid.

    The floor is normally fitted at the same ``n_fft`` the readout uses, and
    then this is the identity. A caller that asks for a different resolution
    gets the floor interpolated instead of an error, because ``S`` is smooth by
    construction — that is the whole premise of block C.
    """
    log_s = np.asarray(psd.log_s, dtype=np.float64)
    freq = np.asarray(psd.freq, dtype=np.float64)
    want = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))
    if log_s.shape[-1] == want.size and np.allclose(freq, want):
        return log_s
    return np.stack([np.stack([np.interp(want, freq, row) for row in chan]) for chan in log_s])


def _curvature(v: Any, weight: Any | None = None) -> float:
    """``sum |D2 v|^2`` over the rows of ``(..., N)``, optionally row weighted.

    ``D2`` is the same second difference the priors are written with
    (:func:`tracking.vk_tracking.second_diff`), applied here as the three-term
    stencil because only its squared norm is wanted. A row shorter than three
    samples has no second difference and contributes nothing.
    """
    a = np.atleast_2d(np.asarray(v))
    if a.shape[-1] < 3:
        return 0.0
    d2 = a[..., :-2] - 2.0 * a[..., 1:-1] + a[..., 2:]
    sq = np.abs(d2) ** 2
    if weight is None:
        return float(np.sum(sq))
    w = np.asarray(weight, dtype=np.float64)
    return float(np.sum(w * np.sum(sq, axis=-1)))


def joint_objective(state: JointState, audio: Any | None = None) -> dict[str, Any]:
    """:func:`map_objective` of a :class:`JointState` — the state's own weights.

    Every argument is read off the state, so the objective is evaluated at the
    configuration the alternation actually ran with: the floor block's grid
    (``JointConfig.psd_n_fft``), the phase bands of block B, and the per-track
    selectivity the solver reported in ``Envelopes.bw_track``.

    It reads the LAST solve's residual against the LAST floor, which on the
    shipped recipe (``floor -> (solve, split, floor) x (iters-1) -> solve``) is
    the floor fitted on the previous round's residual. That is the block
    coordinate the alternation stopped at, and it is the value every window
    reports, so the windows are comparable.

    Under ``JointConfig.v4`` the SIGNAL changes and ``audio`` becomes required.
    The v4 objective is the MARGINAL likelihood — the line processes are
    integrated out rather than conditioned on — so the thing it scores is the
    ORIGINAL signal against ``S + sum H L``, not the residual against ``S``.
    Scoring the residual there would be counting the comb twice: once by
    subtracting it and once by modelling it. Every term of the readout is then
    about the original signal, including the v3 columns (``data`` and ``rent``
    become what the floor ALONE would cost on it), which is what makes them
    still comparable with ``total_v4`` beside them.
    """
    if state.env is None or state.residual is None or state.psd is None:
        raise ValueError("joint_objective: nothing solved yet")
    jc = state.jcfg
    if jc.v4 and audio is None:
        raise ValueError(
            "joint_objective: the v4 objective is the MARGINAL likelihood of the ORIGINAL "
            "signal, so it needs the audio — the residual has the comb subtracted out of it"
        )
    return map_objective(
        state.residual if not jc.v4 else np.asarray(audio, dtype=np.float64),
        float(state.cfg.sr),
        state.psd,
        x=state.env.x,
        k=state.env.k,
        bw_track=state.env.bw_track,
        theta=state.theta,
        psi=state.psi,
        fs_env=float(state.env.fs_env),
        n_fft=int(jc.psd_n_fft),
        p=int(state.vk.p),
        bw_theta_hz=jc.bw_theta_hz,
        bw_psi_slope=jc.bw_psi_slope,
        bw_psi_max=jc.bw_psi_max,
        bw_psi_min=jc.bw_psi_min,
        # The determinant of the system block A actually factorized, so the
        # marginal correction pays for the freedom the profiled objective
        # silently took. Off unless JointConfig.marginal asks for it.
        logdet_posterior=float(state.env.logdet) if jc.marginal else None,
        # The carrier the whole alternation is conditioned on — the hypothesis's
        # own trajectory, so its comb regions are its own. Off unless
        # JointConfig.h_aware asks for it.
        h_carrier=state.carrier if jc.h_aware else None,
        h_slope_hz_per_k=jc.bw_psi_slope,
        # One profiled floor scale per (channel, frame), so a gust stops paying
        # a misfit no comb hypothesis caused. Off unless
        # JointConfig.adaptive_floor asks for it.
        adaptive_floor=bool(jc.adaptive_floor),
        # The nuisance constrained to the hypothesis's OWN line shape. Off
        # unless JointConfig.h_lorentzian asks for it, and inert without
        # h_aware.
        h_lorentzian=bool(jc.h_lorentzian),
        # J_v4: the fitted line powers ARE the noise model's comb, so the
        # objective needs no nuisance to profile and no correction to add.
        v4_powers=state.h_powers if jc.v4 else None,
        v4_carrier=state.carrier if jc.v4 else None,
        v4_b_f_hz=float(jc.v4_b_f_hz),
    )


def joint_result(state: JointState, iterations: list[dict[str, Any]]) -> JointResult:
    """The alternation's state, read out as the record a driver writes to disk."""
    if state.env is None or state.x_eff is None or state.psd is None:
        raise ValueError("joint_result: the alternation ran no solve")
    return JointResult(
        env=replace(state.env, x=state.x_eff),
        theta_env=state.theta,
        psi=state.psi,
        psd=state.psd,
        residual=np.asarray(state.residual),
        track_energy=np.asarray(state.track_energy),
        h_powers=state.h_powers,
        iterations=iterations,
        stochastic=None if state.stochastic is None else np.asarray(state.stochastic),
    )


# ---------------------------------------------------------------------------
# stitching windows that each carry their own shaft correction


def global_rate_correction(
    windows: list[dict[str, Any]], stride: int, a_min: int, a_max: int, ramp: int
) -> np.ndarray:
    """Cross-fade per-window shaft RATE corrections onto one global envelope grid.

    ``windows`` is a list of ``{"a0": start sample, "dr": (R, J) rev/s}``. The
    rate and not the phase is what crosses a window boundary, because a phase
    carries an arbitrary additive constant per window and two overlapping
    windows would then hold one physical correction at two origins — exactly the
    failure the envelope stitch already guards against.
    """
    e0 = int(a_min) // int(stride)
    n_env = int(a_max) // int(stride) - e0
    n_rot = int(np.asarray(windows[0]["dr"]).shape[0])
    num = np.zeros((n_rot, n_env), dtype=np.float64)
    den = np.zeros(n_env, dtype=np.float64)
    for w in windows:
        dr = np.atleast_2d(np.asarray(w["dr"], dtype=np.float64))
        j0 = int(w["a0"]) // int(stride) - e0
        n_w = int(dr.shape[-1])
        fade = fade_weights(n_w, min(int(ramp), n_w // 2))
        num[:, j0 : j0 + n_w] += dr * fade[None, :]
        den[j0 : j0 + n_w] += fade
    return num / np.maximum(den, 1e-12)[None, :]


def corrected_phase(
    r_audio: Any, dr_env: Any, sr: float, stride: int, a_min: int, a_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(corrected rate, corrected shaft phase)`` of a WHOLE recording.

    The stitched rate correction is upsampled onto the audio grid inside the
    covered span (zero outside it) and added to the labels; the phase is its
    running integral, which is the one carrier every stitched envelope is
    referenced to.
    """
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    n_t = int(r.shape[-1])
    corr = np.zeros_like(r)
    span = min(int(a_max), n_t) - int(a_min)
    if span > 0:
        corr[:, int(a_min) : int(a_min) + span] = upsample_env(dr_env, span, int(stride))
    r_corr = r + corr
    return r_corr, 2.0 * np.pi * np.cumsum(r_corr, axis=-1) / float(sr)


def window_extra_phase(
    theta_w: Any, phi_hat: Any, phi_tilde: Any, a0: int, stride: int, n_env_w: int
) -> np.ndarray:
    """``(R, J)`` extra rotor phase that moves one window onto the global carrier.

    A window's envelopes sit against ``phi_hat(t) - phi_hat(a0-1) + theta_w(t)``;
    the stitch wants them against ``phi_tilde(t) - phi_tilde(a0-1)``. The
    difference is what this returns, and a caller multiplies track ``m`` by
    ``exp(j k_m e[rotor_m])`` before handing the window to
    :func:`tracking.decompose.stitch_bank` with ``phi_tilde``. It is slow by
    construction — the two carriers are the same trajectory up to the blend
    between neighbouring windows — but it is REPORTED (``max_rate_hz`` in the
    driver) rather than assumed, because at high ``k`` a large disagreement
    would alias on the 100 Hz envelope grid.
    """
    th = np.atleast_2d(np.asarray(theta_w, dtype=np.float64))
    ph = np.atleast_2d(np.asarray(phi_hat, dtype=np.float64))
    pt = np.atleast_2d(np.asarray(phi_tilde, dtype=np.float64))
    idx = int(a0) + np.arange(int(n_env_w)) * int(stride)
    idx = np.clip(idx, 0, ph.shape[-1] - 1)
    base = np.zeros((ph.shape[0], 1)) if a0 == 0 else None
    ph0 = base if base is not None else ph[:, int(a0) - 1 : int(a0)]
    pt0 = base if base is not None else pt[:, int(a0) - 1 : int(a0)]
    return (ph[:, idx] - ph0 + th[:, : len(idx)]) - (pt[:, idx] - pt0)


def stitch_windows(
    windows: list[dict[str, Any]],
    phi: Any,
    stride: int,
    ramp: int,
    *,
    r_audio: Any = None,
    sr: float = 0.0,
) -> dict[str, Any]:
    """Cross-fade per-window envelope banks onto ONE carrier — v2 or v3.

    ``windows`` is a list of ``{"a0", "x", "valid", "rotor", "k"}`` dicts, and a
    JOINT window carries ``"dr"`` and ``"theta"`` beside them. Arrays only: a
    driver that read them from files and a caller that just solved them in
    memory hand over the same list, which is why this is the one stitch.

    A v2 window set is :func:`tracking.decompose.stitch_bank` and nothing else.
    A JOINT window set carries its own shaft correction, so the windows are
    first brought onto ONE carrier, in the only gauge-free currency there is —
    the RATE. Each window's ``dr`` is cross-faded into one global rate
    correction, the corrected phase is its integral, and each bank is rotated by
    the difference between its own carrier and that global one
    (:func:`window_extra_phase`). The rotation is slow by construction, and
    ``theta_stitch_max_rate_hz`` REPORTS how fast the fastest track's rotation
    really is, so a caller can see whether it stayed inside the envelope grid.
    Read the fade-weighted number: a rotation at a window EDGE is applied where
    that window contributes almost nothing, so the raw maximum overstates what
    reaches the bank.
    """

    if not windows:
        raise ValueError("stitch_windows: no window to stitch")
    if "dr" not in windows[0]:
        return {**stitch_bank(windows, phi, stride, ramp), "phi": np.asarray(phi), "joint": False}

    k = np.asarray(windows[0]["k"], dtype=np.int64)
    rot = np.asarray(windows[0]["rotor"], dtype=np.int64)
    a_min = min(int(w["a0"]) for w in windows)
    a_max = max(int(w["a0"]) + int(np.asarray(w["x"]).shape[-1]) * int(stride) for w in windows)
    dr_g = global_rate_correction(windows, int(stride), a_min, a_max, int(ramp))
    r_corr, phi_t = corrected_phase(r_audio, dr_g, float(sr), int(stride), a_min, a_max)

    turned: list[dict[str, Any]] = []
    max_rate = 0.0
    max_rate_raw = 0.0
    for w in windows:
        n_w = int(np.asarray(w["x"]).shape[-1])
        e_w = window_extra_phase(w["theta"], phi, phi_t, int(w["a0"]), int(stride), n_w)
        x = np.asarray(w["x"], dtype=np.complex64) * np.exp(
            1j * k[None, :, None] * e_w[rot][None, :, :]
        ).astype(np.complex64)
        turned.append({**w, "x": x})
        if e_w.shape[-1] > 1:
            step = np.abs(np.diff(e_w, axis=-1)) * float(sr) / int(stride) / (2.0 * np.pi)
            fade = fade_weights(n_w, min(int(ramp), n_w // 2))
            pair = np.minimum(fade[:-1], fade[1:])[None, :]
            max_rate = max(max_rate, float((step * pair).max()) * float(k.max()))
            max_rate_raw = max(max_rate_raw, float(step.max()) * float(k.max()))
    return {
        **stitch_bank(turned, phi_t, int(stride), int(ramp)),
        "dr_global": dr_g,
        "r_corrected": r_corr,
        "phi": phi_t,
        "theta_stitch_max_rate_hz": round(max_rate, 3),
        "theta_stitch_max_rate_hz_raw": round(max_rate_raw, 3),
        "joint": True,
    }
