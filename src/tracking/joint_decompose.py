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
    base_bandwidths,
    line_separations,
    reconstruct,
    schedule_rho2_gain,
    track_bands,
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
    "PhaseSplit",
    "SmoothPSD",
    "bw_psi_hz",
    "cell_profile",
    "corrected_phase",
    "global_rate_correction",
    "joint_solve_window",
    "masked_smooth_psd",
    "order_cell_profile",
    "split_phases",
    "theta_rate",
    "track_rho2_gain",
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
# instrument 1: the order-cell profile (probe B)


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
    starts = np.arange(0, max(1, n_t - n_fft + 1), max(1, step), dtype=np.int64)
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

    win = np.hanning(n_fft)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    keep = (freq >= float(f_min_hz)) & (freq <= 0.45 * float(sr))
    fk = freq[keep]
    ks = np.arange(1, int(k_max) + 1, dtype=np.float64)
    half = np.maximum(mask_factor * LINEWIDTH_HZ_PER_K * ks, float(mask_min_hz))
    for c0 in range(0, starts.size, int(frames_per_chunk)):
        sub = starts[c0 : c0 + int(frames_per_chunk)]
        idx = sub[:, None] + np.arange(n_fft)[None, :]
        # (mic, frame, n_fft) -> power averaged over microphones
        seg = y[:, idx] * win[None, None, :]
        power = np.mean(np.abs(np.fft.rfft(seg, axis=-1)) ** 2, axis=0)[:, keep]
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
    trend = np.stack([_order_trend(profile[a], order_step, detrend_orders) for a in range(len(refs))])
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
    starts = np.arange(0, max(1, n_t - n_fft + 1), max(1, step), dtype=np.int64)
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

    win = np.hanning(n_fft)
    scale = 1.0 / (float(sr) * float((win**2).sum()))
    ks = np.arange(1, int(k_hi) + 1, dtype=np.float64)
    want = np.maximum(mask_factor * LINEWIDTH_HZ_PER_K * ks, float(mask_min_hz))
    block_of = np.minimum((starts * n_bl) // max(1, n_t), n_bl - 1)
    for c0 in range(0, starts.size, int(frames_per_chunk)):
        sub = starts[c0 : c0 + int(frames_per_chunk)]
        idx = sub[:, None] + np.arange(n_fft)[None, :]
        seg = y[:, idx] * win[None, None, :]
        power = (np.abs(np.fft.rfft(seg, axis=-1)) ** 2) * scale  # (C, frames, F)
        for i, s in enumerate(sub):
            rate = r[:, s : s + n_fft].mean(axis=-1)
            lines = (ks[None, :] * rate[:, None]).ravel()
            half = np.minimum(want[None, :], float(mask_frac_of_rate) * rate[:, None]).ravel()
            mask = _line_mask(freq, lines, half)
            b = int(block_of[c0 + i])
            free = (~mask).astype(np.float64)
            acc[:, b] += power[:, i] * free[None, :]
            cnt[b] += free
            seen[b] += 1.0

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
    from scipy.signal import welch

    y = np.atleast_2d(np.asarray(residual, dtype=np.float64))
    f, p = welch(y, fs=float(sr), nperseg=int(nperseg), axis=-1)
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


def track_rho2_gain(
    r_audio: Any, k_hi: int, cfg: FVKConfig, sched: BandwidthSchedule | None, rho_scale: float
) -> np.ndarray | None:
    """The v2 per-track selectivity gain, so v3 keeps the tuned bandwidth law.

    One construction shared with :func:`tracking.decompose.solve_window`; the
    joint solve changes the CARRIER and the WEIGHT, never the bandwidth law it
    inherited.
    """
    if sched is None and rho_scale == 1.0:
        return None
    vk = cfg.vk_config(int(k_hi))
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    rotor, k = _track_table(int(r.shape[0]), vk.k_min, int(k_hi))
    if sched is None:
        return np.full(len(k), float(rho_scale) ** 2)
    _, fs_env = env_stride(vk)
    return (
        schedule_rho2_gain(
            k,
            line_separations(r, rotor, k),
            sched,
            base_bandwidths(r, int(k_hi), cfg),
            fs_env,
            vk.p,
        )
        * float(rho_scale) ** 2
    )


def joint_solve_window(
    audio: Any,
    r_audio: Any,
    cfg: FVKConfig,
    *,
    k_hi: int,
    mics: int | None = None,
    jcfg: JointConfig | None = None,
    bw_schedule: BandwidthSchedule | None = None,
    rho_scale: float = 1.0,
    t_start_s: float = 0.0,
) -> JointResult:
    """The v3 alternation on one window: A (whitened solve), B (phases), C (floor).

    One iteration is one v2-sized banded solve plus two negligible blocks, so
    ``jcfg.iters`` iterations cost about that many v2 solves. The returned
    envelope is the EFFECTIVE one, ``g e^{j psi}``, against the corrected
    carrier in ``env.phase`` — so every v2 consumer (``reconstruct``,
    ``stitch_bank``, the ledger) reads it unchanged, and only the carrier moved.
    """
    jc = JointConfig() if jcfg is None else jcfg
    n_mic = int(cfg.max_channels if mics is None else mics)
    y = np.ascontiguousarray(np.asarray(audio, dtype=np.float64)[:n_mic])
    r = np.atleast_2d(np.asarray(r_audio, dtype=np.float64))
    sr = float(cfg.sr)
    vk = cfg.vk_config(int(k_hi))
    stride, fs_env = env_stride(vk)
    n_t = int(y.shape[-1])
    n_env = len(range(0, n_t, stride))
    rotor, k = _track_table(int(r.shape[0]), vk.k_min, int(k_hi))
    r_env = r[:, ::stride][:, :n_env]
    t_env = np.arange(n_env, dtype=np.float64) * stride / sr
    gain = track_rho2_gain(r, int(k_hi), cfg, bw_schedule, float(rho_scale))

    theta_env = np.zeros((int(r.shape[0]), n_env))
    psi = np.zeros((len(k), n_env))
    psd = masked_smooth_psd(
        y,
        sr,
        r,
        int(k_hi),
        n_fft=jc.psd_n_fft,
        n_blocks=jc.psd_blocks,
        n_cep=jc.psd_n_cep,
        t_start_s=float(t_start_s),
    )

    env: Envelopes | None = None
    x_eff = np.zeros((0, 0, 0), dtype=np.complex128)
    resid = np.zeros_like(y)
    track_e = np.zeros(len(k))
    iters: list[dict[str, Any]] = []
    for it in range(1, int(jc.iters) + 1):
        weight = (
            whiten_weights(psd, k, rotor, r_env, t_env, clamp_db=jc.whiten_clamp_db)
            if jc.whiten
            else None
        )
        gain_it = gain
        if weight is not None and jc.bandwidth_neutral:
            mean_u2 = np.mean(weight**2, axis=-1)
            gain_it = mean_u2 if gain is None else gain * mean_u2
        # The three joint hooks, passed as a mapping: they are what turns the v2
        # solver into block A (see vk_envelopes' docstring).
        hooks: dict[str, Any] = {
            "phase_offset": upsample_env(theta_env, n_t, stride),
            "env_rotation": psi,
            "data_weight": weight,
        }
        e: Envelopes = vk_envelopes(y, r, vk, k_hi=int(k_hi), rho2_gain=gain_it, **hooks)
        env = e
        x_eff = e.x * np.exp(1j * psi)[None, :, :]
        recon, track_e = reconstruct(x_eff, k, rotor, e.phase, stride)
        resid = y - recon
        last = it == int(jc.iters)
        step: dict[str, Any] = {
            "iter": it,
            "k_trust": min(jc.k_cap(it), int(k_hi)),
            "residual_fraction": round(
                float((resid**2).sum() / max(float((y**2).sum()), 1e-30)), 6
            ),
            "track_fraction": round(float(track_e.sum() / max(float((y**2).sum()), 1e-30)), 6),
            "psd_masked_frac": psd.n_masked_frac,
            "whitened": bool(jc.whiten),
            "flatness": whitened_flatness(resid, sr, psd),
        }
        if jc.profile_every_iter or last:
            step["order_cell"] = {
                nm: {kk: vv for kk, vv in d.items() if kk not in ("offsets", "cell")}
                for nm, d in order_cell_profile(
                    resid,
                    sr,
                    r,
                    n_fft=jc.profile_n_fft,
                    order_step=jc.profile_order_step,
                    k_max=int(k_hi),
                )["bands"].items()
            }
        if not last:
            split = split_phases(
                e.x,
                k,
                rotor,
                e.valid,
                fs_env,
                k_trust=min(jc.k_cap(it), int(k_hi)),
                conc_min=jc.conc_min,
                bw_theta_hz=jc.bw_theta_hz,
                bw_psi_slope=jc.bw_psi_slope,
                bw_psi_max=jc.bw_psi_max,
                bw_psi_min=jc.bw_psi_min,
                per_rotor=jc.per_rotor_theta,
                with_psi=it >= int(jc.psi_from_iter),
            )
            theta_env = theta_env + split.theta
            psi = psi + split.psi
            step["phase_split"] = split.diag
            psd = masked_smooth_psd(
                resid,
                sr,
                r,
                int(k_hi),
                n_fft=jc.psd_n_fft,
                n_blocks=jc.psd_blocks,
                n_cep=jc.psd_n_cep,
                t_start_s=float(t_start_s),
            )
        iters.append(step)

    if env is None:
        raise ValueError(f"JointConfig.iters must be at least 1, got {jc.iters}")
    return JointResult(
        env=replace(env, x=x_eff),
        theta_env=theta_env,
        psi=psi,
        psd=psd,
        residual=resid,
        track_energy=track_e,
        iterations=iters,
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
    from tracking.decompose import fade_weights

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
