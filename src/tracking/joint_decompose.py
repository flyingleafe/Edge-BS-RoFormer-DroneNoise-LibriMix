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
    "JointConfig",
    "JointResult",
    "JointState",
    "PhaseSplit",
    "SmoothPSD",
    "bw_psi_hz",
    "cell_profile",
    "corrected_phase",
    "floor_block",
    "frame_starts",
    "global_rate_correction",
    "joint_result",
    "joint_state",
    "masked_smooth_psd",
    "order_cell_bands",
    "order_cell_profile",
    "solve_block",
    "solve_report",
    "split_block",
    "split_phases",
    "stft_power",
    "stitch_windows",
    "theta_rate",
    "upsample_env",
    "wh_lambda",
    "wh_smooth",
    "whiten_weights",
    "whitened_flatness",
    "window_extra_phase",
]

#: Measured shaft-wander law: harmonic ``k`` is about this many Hz wide per
#: harmonic index (``docs/experiments/vk-decomposition.md``, sigma_r ~ 0.6 rev/s).
LINEWIDTH_HZ_PER_K = 0.6


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
    if starts.size == 0:
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
        log_s = per_block[0]
    else:
        tb = np.asarray(psd.t_block, dtype=np.float64)
        pos = np.clip(np.interp(tt, tb, np.arange(len(tb), dtype=np.float64)), 0, len(tb) - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(tb) - 1)
        frac = (pos - lo)[None, :]
        log_s = per_block[lo, :, np.arange(len(tt))].T * (1.0 - frac) + (
            per_block[hi, :, np.arange(len(tt))].T * frac
        )
    log_u = -0.5 * log_s
    lim = float(clamp_db) * np.log(10.0) / 20.0
    ok = np.isfinite(log_u)
    log_u = np.clip(log_u - float(np.mean(log_u[ok])), -lim, lim)
    # Centered again AFTER the clamp, so the geometric mean is exactly 1 and the
    # whitening cannot move the balance between the data term and the prior. The
    # clamp then bounds the SPREAD at 2 * clamp_db, not each value.
    return np.exp(log_u - float(np.mean(log_u[ok])))


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
    iterations: list[dict[str, Any]] = field(default_factory=list)


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
    env: Envelopes | None = None
    x_eff: np.ndarray | None = None
    residual: np.ndarray | None = None
    track_energy: np.ndarray | None = None
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
    weight = (
        whiten_weights(
            state.psd,
            k,
            rotor,
            state.carrier[:, ::stride][:, : state.n_env],
            np.arange(state.n_env, dtype=np.float64) * stride / float(state.cfg.sr),
            clamp_db=jc.whiten_clamp_db,
        )
        if (jc.whiten and state.psd is not None)
        else None
    )
    gain = state.rho2_gain
    if weight is not None and jc.bandwidth_neutral:
        mean_u2 = np.mean(weight**2, axis=-1)
        gain = mean_u2 if gain is None else gain * mean_u2
    # The three joint hooks, passed as a mapping: they are what turns the v2
    # solver into block A (see vk_envelopes' docstring).
    hooks: dict[str, Any] = {
        "phase_offset": upsample_env(state.theta, int(y.shape[-1]), stride),
        "env_rotation": state.psi,
        "data_weight": weight,
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
    """
    jc = state.jcfg
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
        iterations=iterations,
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
