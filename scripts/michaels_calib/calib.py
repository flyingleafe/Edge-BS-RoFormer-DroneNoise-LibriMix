#!/usr/bin/env python3
"""Referee + calibration stages for Michael's telemetry (FLY124 + FLY125).

**Referee** (label-free, absolute): the VK reconstruction residual ratio
``||x - x_hat|| / ||x||`` under ``RECON_CFG`` (k 1..30) — how well the harmonic
model built ALONG a given rotor-speed trajectory explains the recorded audio.
Every hypothesis below is a shift/scale of the TELEMETRY ONLY, scored on the
SAME audio; our own blind RPS estimate is never consulted, so nothing here can
be "fitted to the audio via ours".

**Stages** (one window each, JSON out — the driver runs them in parallel):

  ``lag``   coarse + fine scan of a pure time shift of the telemetry.  A
            constant error shows up as a constant lag, a DILATION error as a
            lag that drifts linearly with window time (fitted in `fit.py`).
  ``val``   at the window's best lag: additive ``gt + b`` vs multiplicative
            ``gt * (1 + g)`` on MATCHED grids (``g = b / 80``, so the families
            are compared at equal correction size at the mean cruise rps).
  ``prot``  blind PER-ROTOR offset scan (rotor r only, the other three
            untouched) — the additive-vs-multiplicative discriminator with the
            most leverage: additive ⇒ the per-rotor optimum is INDEPENDENT of
            that rotor's mean rps, multiplicative ⇒ PROPORTIONAL to it.  At
            cruise the four rotors span 74–91 rev/s (a 23 % spread), so the two
            hypotheses predict measurably different per-rotor optima.
  ``post``  residual lag + residual global offset on CORRECTED labels (the
            window itself is rebuilt with the candidate constants) — the
            after-the-fact validation that the correction closed the gap.
"""

from __future__ import annotations

import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "scripts"), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vk_blind_annotation import MIDBAND_CFGS  # noqa: E402
from windows import SR, Window  # noqa: E402

from tracking.vk_tracking import vk_envelopes, vk_reconstruct  # noqa: E402

#: Identical to ``rps_refine_lab.RECON_CFG`` (same one-line definition) — the
#: established referee configuration of the precision campaign.
RECON_CFG = dc_replace(MIDBAND_CFGS[0], k_min=1, k_max=30)

#: rev/s at which the additive and multiplicative correction families are
#: matched (the mean cruise rotor speed of both recordings).
MATCH_RPS = 80.0


# ────────────────────────────────────────────────────────── referee
def recon_ratio(win: Window, traj: np.ndarray) -> float:
    """``||x - VK_reconstruct(VK_envelopes(x, traj))|| / ||x||``, all 4 rotors."""
    n_t = win.audio.shape[-1]
    t_aud = np.arange(n_t) / SR
    r_aud = np.stack([np.interp(t_aud, win.ft, row) for row in traj])
    env = vk_envelopes(win.audio, r_aud, RECON_CFG)
    recon = vk_reconstruct(env, n_samples=n_t)
    num = float(np.sqrt(np.mean((win.audio - recon) ** 2)))
    den = float(np.sqrt(np.mean(win.audio**2)))
    return num / den


def shift(x: np.ndarray, ft: np.ndarray, n_frames: float) -> np.ndarray:
    """``gt(t - n·dt)``: telemetry pushed LATER by ``n`` frames (edge-clamped)."""
    dt = float(ft[1] - ft[0])
    return np.stack([np.interp(ft, ft + n_frames * dt, row) for row in x])


class Ev:
    """Referee evaluator that records every hypothesis it scores."""

    def __init__(self, win: Window, tag: str = "") -> None:
        self.win = win
        self.tag = tag or win.name
        self.out: dict[str, float] = {}

    def __call__(self, key: str, traj: np.ndarray) -> float:
        v = recon_ratio(self.win, traj)
        self.out[key] = round(v, 6)
        print(f"  {self.tag} {key:16s} {v:.6f}", flush=True)
        return v


def parab_min(xs: Any, ys: Any) -> tuple[float, float, bool]:
    """Sub-grid minimiser via the parabola through the 3 points around the min.

    Returns ``(x*, y*, at_edge)``; ``at_edge`` flags a minimum on the grid
    boundary (or a non-convex triple), where the estimate is unreliable.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    i = int(np.argmin(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), float(ys[i]), True
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    den = y0 - 2 * y1 + y2
    if den <= 0:
        return float(xs[i]), float(y1), True
    h = float(xs[i] - xs[i - 1])
    return (
        float(xs[i] - 0.5 * h * (y2 - y0) / den),
        float(y1 - 0.125 * (y2 - y0) ** 2 / den),
        False,
    )


def grid(lo: float, hi: float, step: float) -> np.ndarray:
    return np.round(np.arange(lo, hi + 1e-9, step), 6)


def scan(
    ev: Ev, prefix: str, make: Any, xs: np.ndarray, fmt: str = "{:+.4f}"
) -> tuple[float, float, bool, list[float]]:
    ys = [ev(prefix + fmt.format(x), make(float(x))) for x in xs]
    xm, ym, edge = parab_min(xs, ys)
    return xm, ym, edge, ys


def refine_scan(
    ev: Ev, prefix: str, make: Any, xs: np.ndarray, step: float, div: int = 4
) -> dict[str, Any]:
    """Coarse scan on ``xs``, then a fine pass of ``2·div+1`` points around it."""
    xm, ym, edge, _ = scan(ev, prefix, make, xs)
    out: dict[str, Any] = {
        "coarse": {"lo": float(xs[0]), "hi": float(xs[-1]), "step": step},
        "coarse_best": round(xm, 5),
        "coarse_edge": edge,
    }
    fstep = step / div
    fxs = grid(xm - step, xm + step, fstep)
    xf, yf, edgef, _ = scan(ev, prefix, make, fxs)
    out |= {
        "fine_step": fstep,
        "best": round(xf, 5),
        "best_recon": round(yf, 6),
        "edge": bool(edge or edgef),
    }
    return out


# ────────────────────────────────────────────────────────── stages
def base_result(win: Window, stage: str) -> dict[str, Any]:
    return {
        "name": win.name,
        "rid": win.rid,
        "widx": win.widx,
        "stage": stage,
        "regime": win.regime,
        "window": [win.start_s, win.end_s],
        "t_centre": win.t_centre,
        "gt_mean": np.round(win.r_meas.mean(1), 4).tolist(),
        "gt_std": np.round(win.r_meas.std(1), 4).tolist(),
    }


def stage_lag(win: Window, lo: float, hi: float, step: float) -> dict[str, Any]:
    """Coarse+fine scan of a pure telemetry time shift."""
    ev = Ev(win)
    gt, ft = win.r_meas, win.ft
    res = base_result(win, "lag")
    sc = refine_scan(ev, "lag", lambda n: shift(gt, ft, n), grid(lo, hi, step), step)
    res |= {
        "best_lag_frames": sc["best"],
        "best_lag_ms": round(sc["best"] * 1000.0 * (ft[1] - ft[0]), 3),
        "best_recon": sc["best_recon"],
        "edge": sc["edge"],
        "raw_recon": ev.out.get("lag+0.0000"),
        "scan": sc,
        "recon": ev.out,
    }
    return res


def stage_val(win: Window, best_lag: float, lo: float, hi: float, step: float) -> dict[str, Any]:
    """Additive ``gt + b`` vs multiplicative ``gt·(1+g)``, matched at MATCH_RPS."""
    ev = Ev(win)
    base = shift(win.r_meas, win.ft, best_lag)
    res = base_result(win, "val")
    res["at_lag_frames"] = best_lag
    bs = grid(lo, hi, step)
    xb, yb, eb, ysb = scan(ev, "b", lambda b: base + b, bs, "{:+.4f}")
    gs = np.round(bs / MATCH_RPS, 10)
    xg, yg, eg, ysg = scan(ev, "g", lambda g: base * (1.0 + g), gs, "{:+.8f}")
    res |= {
        "match_rps": MATCH_RPS,
        "b_grid": bs.tolist(),
        "b_recon": [round(v, 6) for v in ysb],
        "g_recon": [round(v, 6) for v in ysg],
        "best_b": round(xb, 5),
        "best_b_recon": round(yb, 6),
        "edge_b": eb,
        "best_g": float(f"{xg:.8g}"),
        "best_g_recon": round(yg, 6),
        "edge_g": eg,
        "best_g_at_match": round(xg * MATCH_RPS, 5),
        # >0 => the multiplicative family reaches a LOWER residual (wins)
        "delta_add_minus_mul": round(yb - yg, 6),
        "recon": ev.out,
    }
    return res


def stage_prot(
    win: Window, best_lag: float, rotor: int, lo: float, hi: float, step: float
) -> dict[str, Any]:
    """Per-rotor offset scan: rotor ``rotor`` gets ``+b``, the others untouched."""
    ev = Ev(win, f"{win.name}:r{rotor}")
    base = shift(win.r_meas, win.ft, best_lag)
    res = base_result(win, f"prot_r{rotor}")
    res["at_lag_frames"] = best_lag
    res["rotor"] = rotor
    res["mean_rps"] = round(float(base[rotor].mean()), 4)

    def make(b: float) -> np.ndarray:
        t = base.copy()
        t[rotor] = t[rotor] + b
        return t

    sc = refine_scan(ev, f"p{rotor}b", make, grid(lo, hi, step), step)
    res |= {
        "best_b": sc["best"],
        "best_recon": sc["best_recon"],
        "edge": sc["edge"],
        "scan": sc,
        "recon": ev.out,
    }
    return res


def stage_post(
    win: Window, lo: float, hi: float, step: float, blo: float, bhi: float, bstep: float
) -> dict[str, Any]:
    """Residual lag + residual global offset on already-CORRECTED labels.

    ``win`` must have been built with the candidate constants; success is
    ``best_lag ≈ 0`` and ``best_b ≈ 0``.
    """
    ev = Ev(win, f"{win.name}:post")
    gt, ft = win.r_meas, win.ft
    res = base_result(win, "post")
    sc = refine_scan(ev, "lag", lambda n: shift(gt, ft, n), grid(lo, hi, step), step)
    base = shift(gt, ft, sc["best"])
    xb, yb, eb, _ = scan(ev, "b", lambda b: base + b, grid(blo, bhi, bstep), "{:+.4f}")
    res |= {
        "resid_lag_frames": sc["best"],
        "resid_lag_ms": round(sc["best"] * 1000.0 * (ft[1] - ft[0]), 3),
        "resid_lag_recon": sc["best_recon"],
        "edge": sc["edge"],
        "resid_b": round(xb, 5),
        "resid_b_recon": round(yb, 6),
        "edge_b": eb,
        "scan": sc,
        "recon": ev.out,
    }
    return res
