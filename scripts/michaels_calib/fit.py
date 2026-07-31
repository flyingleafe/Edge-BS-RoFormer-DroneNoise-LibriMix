#!/usr/bin/env python3
"""Turn the per-window scans into constants and verdicts.

**Lag → constants.**  With the CURRENT constants the audio prefers the
telemetry shifted by ``lag_s(t) = a + b·t`` seconds (``t`` = window-centre time,
seconds from the re-anchored audio start).  Folding that into the loader's
parameterisation ``W = d·tau + (ts_raw[0] − time_offset)``::

    time_dilation_new = time_dilation_old / (1 - b)
    time_offset_new   = time_offset_old - a / (1 - b)

A pure OFFSET error is the special case ``b = 0`` (constant lag); a residual
DILATION error shows up as ``b != 0`` (linear trend).  The comparison that
decides between them is the residual RMS of the linear fit vs the constant-lag
model.

**val / prot → additive vs multiplicative.**  ``val`` compares the two families
at matched size; the discriminator with real leverage is ``prot``: regress the
per-rotor optimum ``b_r`` on that rotor's mean rps.  Additive ⇒ ``b_r = const``
(slope 0); multiplicative ⇒ ``b_r = g·mean_r`` (intercept 0).  Both constrained
models are scored by residual RMS against the free line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

FRAME_S = 0.032


def ols(t: Any, y: Any) -> dict[str, Any]:
    """Free line ``y = b·t + a`` plus the constant-only null model."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    a_mat = np.stack([t, np.ones_like(t)], 1)
    (b, a), *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    pred = a_mat @ np.array([b, a])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "slope": float(b),
        "intercept": float(a),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rms": float(np.sqrt(ss_res / len(y))),
        "rms_const_model": float(np.sqrt(np.mean((y - y.mean()) ** 2))),
        "pred": pred.tolist(),
    }


def fit_lag(rows: list[dict[str, Any]], off_old: float, dil_old: float) -> dict[str, Any]:
    """Regress best lag (seconds) on window-centre time; derive new constants."""
    rows = sorted(rows, key=lambda r: r["t_centre"])
    t = [float(r["t_centre"]) for r in rows]
    lag_s = [float(r["best_lag_frames"]) * FRAME_S for r in rows]
    if len(rows) < 3:
        return {
            "error": f"only {len(rows)} usable cruise windows — not enough to regress a line",
            "windows": [
                {"name": r["name"], "t_centre": r["t_centre"], "lag_ms": round(ll * 1e3, 3)}
                for r, ll in zip(rows, lag_s, strict=True)
            ],
        }
    f = ols(t, lag_s)
    b, a = f["slope"], f["intercept"]
    dil_new = dil_old / (1.0 - b)
    off_new = off_old - a / (1.0 - b)
    return {
        "windows": [
            {
                "name": r["name"],
                "t_centre": r["t_centre"],
                "lag_ms": round(ll * 1e3, 3),
                "fit_ms": round(pp * 1e3, 3),
                "resid_ms": round((ll - pp) * 1e3, 3),
                "best_recon": r.get("best_recon"),
                "raw_recon": r.get("raw_recon"),
            }
            for r, ll, pp in zip(rows, lag_s, f["pred"], strict=True)
        ],
        "slope_ms_per_s": round(b * 1e3, 5),
        "intercept_ms": round(a * 1e3, 4),
        "r2": round(f["r2"], 5),
        "resid_rms_ms": round(f["rms"] * 1e3, 4),
        "resid_rms_ms_constant_lag_model": round(f["rms_const_model"] * 1e3, 4),
        "dilation_verdict": (
            "dilation (linear trend beats constant lag)"
            if f["rms"] < 0.5 * f["rms_const_model"]
            else "constant lag (no significant trend)"
        ),
        "shipped": {"time_offset": off_old, "time_dilation": dil_old},
        "proposed": {
            "time_offset": round(off_new, 6),
            "time_dilation": round(dil_new, 9),
            "dilation_multiplier": round(1.0 / (1.0 - b), 9),
            "offset_delta_ms": round(-a / (1.0 - b) * 1e3, 3),
        },
        "offset_only_alternative": {
            "mean_lag_ms": round(float(np.mean(lag_s)) * 1e3, 3),
            "time_offset": round(off_old - float(np.mean(lag_s)), 6),
            "time_dilation": dil_old,
        },
    }


def fit_prot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-rotor optima vs rotor mean rps → additive or multiplicative?"""
    rows = [r for r in rows if not r.get("edge")]
    if len(rows) < 4:
        return {"error": f"only {len(rows)} non-edge per-rotor points"}
    m = np.array([float(r["mean_rps"]) for r in rows])
    b = np.array([float(r["best_b"]) for r in rows])
    f = ols(m, b)
    # constrained models
    add_b = float(b.mean())  # b_r = const
    add_rms = float(np.sqrt(np.mean((b - add_b) ** 2)))
    mul_g = float(np.sum(m * b) / np.sum(m * m))  # b_r = g·mean_r
    mul_rms = float(np.sqrt(np.mean((b - mul_g * m) ** 2)))
    verdict = "additive" if add_rms < mul_rms else "multiplicative"
    return {
        "points": [
            {
                "name": r["name"],
                "rotor": r["rotor"],
                "mean_rps": r["mean_rps"],
                "best_b": r["best_b"],
                "best_recon": r.get("best_recon"),
            }
            for r in rows
        ],
        "free_line": {
            "slope_per_revps": round(f["slope"], 6),
            "intercept_revps": round(f["intercept"], 5),
            "r2": round(f["r2"], 5),
            "rms": round(f["rms"], 5),
        },
        "additive_model": {"b": round(add_b, 5), "rms": round(add_rms, 5)},
        "multiplicative_model": {
            "g": float(f"{mul_g:.6g}"),
            "b_at_80": round(mul_g * 80.0, 5),
            "rms": round(mul_rms, 5),
        },
        "verdict": verdict,
        "margin_rms": round(abs(add_rms - mul_rms), 5),
        # multiplicative predicts b_r spread = g·(max-min mean rps); additive 0
        "predicted_spread_if_multiplicative": round(mul_g * (m.max() - m.min()), 5),
        "observed_spread": round(float(b.max() - b.min()), 5),
    }


def fit_val(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the matched additive-vs-multiplicative comparison."""
    if not rows:
        return {"error": "no val windows"}
    d = [float(r["delta_add_minus_mul"]) for r in rows]
    return {
        "windows": [
            {
                "name": r["name"],
                "best_b": r["best_b"],
                "best_b_recon": r["best_b_recon"],
                "best_g_at_match": r["best_g_at_match"],
                "best_g_recon": r["best_g_recon"],
                "delta_add_minus_mul": r["delta_add_minus_mul"],
                "edge_b": r.get("edge_b"),
                "edge_g": r.get("edge_g"),
            }
            for r in rows
        ],
        "mean_delta_add_minus_mul": round(float(np.mean(d)), 6),
        "n_windows_multiplicative_wins": int(sum(x > 0 for x in d)),
        "note": (
            "the families are matched at MATCH_RPS, so this comparison is weak by "
            "construction; prot carries the leverage"
        ),
    }


def _load(raw: Path, pattern: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(raw.glob(pattern))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/michaels_calib")
    ap.add_argument("--rid", default="FLY125")
    ap.add_argument("--time-offset", type=float, required=True)
    ap.add_argument("--time-dilation", type=float, required=True)
    args = ap.parse_args()
    raw = Path(args.results) / "raw"
    tag = args.rid.lower()
    lag = [
        r for r in _load(raw, f"{tag}_w*__lag.json") if r["regime"] == "cruise" and not r["edge"]
    ]
    print(json.dumps(fit_lag(lag, args.time_offset, args.time_dilation), indent=1))
    print(json.dumps(fit_val(_load(raw, f"{tag}_w*__val.json")), indent=1))
    print(json.dumps(fit_prot(_load(raw, f"{tag}_w*__prot_r*.json")), indent=1))


if __name__ == "__main__":
    main()
