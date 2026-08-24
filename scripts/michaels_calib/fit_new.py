#!/usr/bin/env python3
"""Fine telemetry calibration of the held-out Michael's recordings (FLY103/FLY108).

The WP13/WP14 procedure of ``docs/experiments/rps-refine-precision.md``, run
against the recordings that had no constants at all. The referee is the same
LABEL-FREE Vold-Kalman reconstruction residual ``||x - x_hat|| / ||x||``
(``calib.RECON_CFG``, k 1..30) on the same 16 s window protocol
(``windows.py``); only the TELEMETRY is shifted/scaled.

Three stages (each is a restartable ``utils.gridrun`` unit):

  ``lag``   coarse+fine scan of a pure telemetry time shift, one unit per
            usable cruise window. The optima are regressed on window-centre
            time -> ``time_offset`` / ``time_dilation``.
  ``val``   at each window's own best lag, the additive ``gt + b`` and
            multiplicative ``gt * (1 + g)`` families on matched grids. The
            shipped rev/s constant is the mean of the per-window ``g``: ONE
            global multiplicative correction per recording, the WP14 model.
            Both families move all four rotors together, so neither can drive
            a rotor pair into degeneracy.
  ``prot``  OPTIONAL per-rotor additive offset (the WP14 non-twin fit).
            **Fragile here** and off by default: the referee has one channel
            instead of eight, and FLY103's rotor pairs sit 2.0/2.5 rev/s
            apart, so a per-rotor offset of the expected size walks one rotor
            into its neighbour and the near-degenerate VK solve stalls or
            blows up. WP14 already settled what this stage decides — the
            correction is global and multiplicative, per-rotor constants are
            unidentifiable, and the global gain is degenerate with a
            sample-clock error — so nothing here re-opens it.

The seed comes from ``coarse_align.py`` and is already in
``sources.michaels.MICHAELS_TEST_FILES``; the window cache is built with it,
so the lag optima measured here are RESIDUALS on that seed and the folding
below is the usual composition.

**Folding (the anchored parameterisation).** These recordings align as
``t_log = time_offset + t_audio / time_dilation``. If the audio prefers the
telemetry pushed later by ``lag(t) = a + b*t`` seconds, then::

    time_dilation_new = time_dilation_old / (1 - b)
    time_offset_new   = time_offset_old  - a / time_dilation_old

(FLY124/FLY125 fold as ``time_offset - a / (1 - b)`` instead, because their
legacy path puts the offset on the other side of the dilation.)

Cost: one referee evaluation is 3.6 CPU-s per 16 s MONO window at k_max 30
(measured, BLAS pinned to one thread), and the default grid is 64 evaluations
per window over 9 windows — about 35 CPU-minutes. Submit it::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 1h \\
        --yes -- python scripts/michaels_calib/fit_new.py --jobs 16

Locally, ``--smoke`` runs one window per recording on tiny grids (~2 minutes)
to prove the wiring before you submit.

Outputs under ``--out`` (default ``results/michaels_fit_new``):
``raw/<uid>.json`` per unit, ``summary.json`` (the fits + the proposed
constants) and ``constants.json`` (just the three constants per recording,
ready to paste into ``MICHAELS_TEST_FILES`` / ``MICHAELS_RPS_SCALE``).
"""

from __future__ import annotations

import os

# Pinned BEFORE numpy is imported anywhere in this process: the parallelism is
# across PROCESSES, and forked workers inherit the parent's already-initialized
# BLAS pool. Capping it later (as ``gridrun`` does) is too late to help them,
# and every worker then runs multi-threaded on a shared machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

DEFAULT_OUT = REPO / "results" / "michaels_fit_new"
DEFAULT_CACHE = REPO / ".cache" / "michaels_fit_new"
RIDS = ("FLY103", "FLY108")
ROTOR_NAMES = ("RFront", "LFront", "LBack", "RBack")


# ────────────────────────────────────────────────────────── unit worker
def worker(unit: Unit) -> dict[str, Any]:
    """One (window, stage[, rotor]) unit — heavy imports stay in here."""
    from dataclasses import replace as dc_replace

    import calib as C
    import windows as W

    p = dict(unit.params)
    C.RECON_CFG = dc_replace(C.RECON_CFG, k_min=1, k_max=int(p["k_max"]))
    win = W.load_cached(Path(p["cache_dir"]), str(p["name"]))
    stage = str(p["stage"])
    if stage == "lag":
        res = C.stage_lag(win, p["lo"], p["hi"], p["step"])
    elif stage == "prot":
        res = C.stage_prot(win, p["best_lag"], int(p["rotor"]), p["lo"], p["hi"], p["step"])
    elif stage == "val":
        res = C.stage_val(win, p["best_lag"], p["lo"], p["hi"], p["step"])
    else:
        raise ValueError(f"unknown stage {stage!r}")
    res["uid"] = unit.uid
    res["k_max"] = int(p["k_max"])
    return res


# ────────────────────────────────────────────────────────── fits
def ols(t: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Free line ``y = b*t + a`` plus the constant-only null model."""
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
    """Regress the best lag (seconds) on window-centre time -> new constants."""
    rows = sorted(rows, key=lambda r: r["t_centre"])
    lag_s = np.array([float(r["best_lag_frames"]) * 0.032 for r in rows])
    tc = np.array([float(r["t_centre"]) for r in rows])
    if len(rows) < 3:
        return {
            "error": f"only {len(rows)} usable cruise windows — cannot regress a line",
            "windows": [
                {"name": r["name"], "t_centre": r["t_centre"], "lag_ms": round(v * 1e3, 3)}
                for r, v in zip(rows, lag_s, strict=True)
            ],
        }
    f = ols(tc, lag_s)
    b, a = f["slope"], f["intercept"]
    return {
        "windows": [
            {
                "name": r["name"],
                "t_centre": r["t_centre"],
                "lag_ms": round(v * 1e3, 3),
                "fit_ms": round(p * 1e3, 3),
                "resid_ms": round((v - p) * 1e3, 3),
                "best_recon": r.get("best_recon"),
                "raw_recon": r.get("raw_recon"),
                "edge": r.get("edge"),
            }
            for r, v, p in zip(rows, lag_s, f["pred"], strict=True)
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
        "seed": {"time_offset": off_old, "time_dilation": dil_old},
        # anchored parameterisation: t_log = offset + t_audio / dilation
        "proposed": {
            "time_offset": round(off_old - a / dil_old, 6),
            "time_dilation": round(dil_old / (1.0 - b), 9),
            "offset_delta_ms": round(-a / dil_old * 1e3, 3),
            "dilation_multiplier": round(1.0 / (1.0 - b), 9),
        },
        "offset_only_alternative": {
            "mean_lag_ms": round(float(lag_s.mean()) * 1e3, 3),
            "time_offset": round(off_old - float(lag_s.mean()) / dil_old, 6),
            "time_dilation": dil_old,
        },
    }


def twin_rotors(rows: list[dict[str, Any]], sep: float) -> dict[str, Any]:
    """Which rotor pair the audio cannot resolve (mean speeds closer than ``sep``).

    WP14 fits the rev/s magnitude on the NON-twin rotors only. FLY124/FLY125
    had a fixed twin pair (LFront/RBack); here the pair is measured, because
    these are different flights.
    """
    means = np.array([np.mean([r["mean_rps"] for r in rows if r["rotor"] == i]) for i in range(4)])
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    gaps = {f"{i}-{j}": round(float(abs(means[i] - means[j])), 4) for i, j in pairs}
    closest = min(pairs, key=lambda ij: abs(means[ij[0]] - means[ij[1]]))
    gap = float(abs(means[closest[0]] - means[closest[1]]))
    twins = set(closest) if gap < sep else set()
    return {
        "mean_rps": [round(float(v), 4) for v in means],
        "pair_gaps_revps": gaps,
        "closest_pair": [ROTOR_NAMES[closest[0]], ROTOR_NAMES[closest[1]]],
        "closest_gap_revps": round(gap, 4),
        "twin_threshold_revps": sep,
        "twins": sorted(twins),
        "twin_names": [ROTOR_NAMES[i] for i in sorted(twins)],
    }


def fit_global_scale(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """ONE global multiplicative rev/s correction from the ``val`` windows.

    Each window contributes its own multiplicative optimum ``g``; the shipped
    constant is their mean, with the standard error of that mean as the
    uncertainty. ``delta_add_minus_mul`` is carried for the record only — the
    two families are matched at ``calib.MATCH_RPS`` by construction, so the
    comparison has no power (WP14).
    """
    used = [r for r in rows if not r.get("edge_g")]
    if not used:
        return {"error": "no usable val windows"}
    g = np.array([float(r["best_g"]) for r in used])
    delta = np.array([float(r["delta_add_minus_mul"]) for r in used])
    se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else None
    return {
        "n_windows": len(used),
        "n_dropped_edge": sum(1 for r in rows if r.get("edge_g")),
        "g_percent": round(float(g.mean()) * 100.0, 4),
        "g_stderr_percent": None if se is None else round(se * 100.0, 4),
        "rps_scale": round(1.0 + float(g.mean()), 6),
        "rps_scale_stderr": None if se is None else round(se, 6),
        "b_at_80_revps": round(float(g.mean()) * 80.0, 4),
        "per_window_g_percent": [round(float(v) * 100.0, 4) for v in g],
        "mean_delta_add_minus_mul": round(float(delta.mean()), 6),
        "n_windows_multiplicative_wins": int((delta > 0).sum()),
        "note": (
            "the additive and multiplicative families are matched at "
            "MATCH_RPS, so delta_add_minus_mul is weak by construction; the "
            "multiplicative form is the WP14 verdict, not a finding here"
        ),
    }


def fit_scale(rows: list[dict[str, Any]], twins: set[int]) -> dict[str, Any]:
    """ONE global multiplicative rev/s correction, non-twin rotors only (WP14).

    ``b_r = g * mean_rps_r`` fitted through the origin over every (window,
    rotor) point, plus the additive null model for the record.
    """
    used = [r for r in rows if r["rotor"] not in twins and not r.get("edge")]
    if len(used) < 3:
        return {"error": f"only {len(used)} usable non-twin per-rotor points"}
    x = np.array([float(r["mean_rps"]) for r in used])
    y = np.array([float(r["best_b"]) for r in used])
    g = float(np.sum(x * y) / np.sum(x * x))
    resid = y - g * x
    # standard error of a through-the-origin slope
    se = float(np.sqrt(np.sum(resid**2) / max(len(used) - 1, 1)) / np.sqrt(np.sum(x * x)))
    add_b = float(y.mean())
    return {
        "n_points": len(used),
        "n_dropped_twin": sum(1 for r in rows if r["rotor"] in twins),
        "n_dropped_edge": sum(1 for r in rows if r.get("edge") and r["rotor"] not in twins),
        "g_percent": round(g * 100.0, 4),
        "g_stderr_percent": round(se * 100.0, 4),
        "rps_scale": round(1.0 + g, 6),
        "rps_scale_stderr": round(se, 6),
        "b_at_80_revps": round(g * 80.0, 4),
        "rms_multiplicative": round(float(np.sqrt(np.mean(resid**2))), 5),
        "rms_additive": round(float(np.sqrt(np.mean((y - add_b) ** 2))), 5),
        "additive_b_revps": round(add_b, 5),
        "points": [
            {
                "name": r["name"],
                "rotor": ROTOR_NAMES[int(r["rotor"])],
                "mean_rps": r["mean_rps"],
                "best_b": r["best_b"],
                "best_recon": r.get("best_recon"),
            }
            for r in used
        ],
    }


# ────────────────────────────────────────────────────────── driver
def usable_windows(manifest: dict[str, Any], rid: str, max_std: float) -> list[dict[str, Any]]:
    """Cruise windows steady enough to carry an offset scan.

    ``regime == "cruise"`` alone still admits the takeoff ramp (its mean is
    high while the speeds sweep); the per-rotor standard deviation separates
    the two. A ramp window smears every per-rotor optimum and, on MONO audio,
    can drive the near-degenerate VK solve into a blow-up.
    """
    out = []
    for w in manifest["recordings"][rid]["windows"]:
        if w["regime"] != "cruise":
            continue
        std = float(np.mean(w["gt_std"]))
        if std > max_std:
            print(f"  drop {w['name']}: per-rotor std {std:.2f} > {max_std} rev/s (ramp)")
            continue
        out.append(w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rid", action="append", choices=list(RIDS), help="default: both")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--stages", default="lag,val", help="comma list of lag,val,prot")
    ap.add_argument("--k-max", type=int, default=30, help="referee harmonic count")
    ap.add_argument("--max-window-std", type=float, default=3.0, help="rev/s, ramp rejection")
    ap.add_argument("--twin-sep", type=float, default=3.0, help="rev/s, twin-pair threshold")
    ap.add_argument("--lag-lo", type=float, default=-3.0, help="frames of 0.032 s")
    ap.add_argument("--lag-hi", type=float, default=3.0)
    ap.add_argument("--lag-step", type=float, default=0.25)
    ap.add_argument("--prot-lo", type=float, default=-1.5, help="rev/s")
    ap.add_argument("--prot-hi", type=float, default=2.5)
    ap.add_argument("--prot-step", type=float, default=0.25)
    ap.add_argument("--val-lo", type=float, default=-1.8)
    ap.add_argument("--val-hi", type=float, default=2.4)
    ap.add_argument("--val-step", type=float, default=0.3)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one window per recording on tiny grids — a wiring check, not a calibration",
    )
    add_gridrun_args(ap, jobs=4)
    args = ap.parse_args()

    import windows as W

    rids = tuple(args.rid or RIDS)
    out_dir = Path(args.out)
    cache_dir = Path(args.cache)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.smoke:
        args.lag_lo, args.lag_hi, args.lag_step = -0.5, 0.5, 0.5
        args.val_lo, args.val_hi, args.val_step = 0.0, 1.2, 0.6
        args.prot_lo, args.prot_hi, args.prot_step = 0.0, 1.0, 0.5
        args.k_max = min(args.k_max, 12)

    root, how = W.test_data_root()
    print(f"raw root: {root}  ({how})", flush=True)
    manifest = W.build_cache(cache_dir, rids=rids)
    manifest["argv"] = sys.argv[1:]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    selection: dict[str, list[dict[str, Any]]] = {}
    for rid in rids:
        rec = manifest["recordings"][rid]
        wins = usable_windows(manifest, rid, args.max_window_std)
        selection[rid] = wins[:1] if args.smoke else wins
        print(
            f"{rid}: span {rec['eval_span']}, {len(rec['windows'])} windows, "
            f"{len(selection[rid])} used {[w['name'] for w in selection[rid]]}  "
            f"seed offset {rec['time_offset']} dilation {rec['time_dilation']}",
            flush=True,
        )

    common = {"cache_dir": str(cache_dir), "k_max": int(args.k_max)}

    # ── phase 1: lag ────────────────────────────────────────────────────────
    if "lag" in stages:
        units = [
            Unit(
                uid=f"{w['name']}__lag",
                params={
                    **common,
                    "stage": "lag",
                    "name": w["name"],
                    "lo": args.lag_lo,
                    "hi": args.lag_hi,
                    "step": args.lag_step,
                },
            )
            for rid in rids
            for w in selection[rid]
        ]
        res = gridrun_from_args(
            args, units, worker, out_dir, summarize=lambda rows: {"phase": "lag", "n": len(rows)}
        )
        if res.n_failed:
            raise SystemExit(res.exit_code)

    def unit_rows(pattern: str) -> list[dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted((out_dir / "raw").glob(pattern))]

    lag_rows = {rid: [r for r in unit_rows("*__lag.json") if r["rid"] == rid] for rid in rids}
    lag_fits = {
        rid: fit_lag(
            [r for r in lag_rows[rid] if not r.get("edge")],
            float(manifest["recordings"][rid]["time_offset"]),
            float(manifest["recordings"][rid]["time_dilation"]),
        )
        for rid in rids
    }
    for rid, fit in lag_fits.items():
        print(f"\n[{rid}] lag fit\n{json.dumps(fit.get('proposed', fit), indent=1)}", flush=True)

    best_lag = {r["name"]: float(r["best_lag_frames"]) for r in unit_rows("*__lag.json")}

    # ── phase 2: per-rotor offsets (and optionally val) at each best lag ─────
    units = []
    for rid in rids:
        for w in selection[rid]:
            lag = best_lag.get(w["name"])
            if lag is None:
                print(f"  {w['name']}: no lag scan -> prot/val skipped", flush=True)
                continue
            if "prot" in stages:
                units += [
                    Unit(
                        uid=f"{w['name']}__prot_r{rot}",
                        params={
                            **common,
                            "stage": "prot",
                            "name": w["name"],
                            "best_lag": lag,
                            "rotor": rot,
                            "lo": args.prot_lo,
                            "hi": args.prot_hi,
                            "step": args.prot_step,
                        },
                    )
                    for rot in range(4)
                ]
            if "val" in stages:
                units.append(
                    Unit(
                        uid=f"{w['name']}__val",
                        params={
                            **common,
                            "stage": "val",
                            "name": w["name"],
                            "best_lag": lag,
                            "lo": args.val_lo,
                            "hi": args.val_hi,
                            "step": args.val_step,
                        },
                    )
                )
    if units:
        gridrun_from_args(
            args, units, worker, out_dir, summarize=lambda rows: {"phase": "scale", "n": len(rows)}
        )

    # ── phase 3: constants ──────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "data_root": how,
        "argv": sys.argv[1:],
        "referee": {"k_min": 1, "k_max": int(args.k_max), "window_s": W.WINDOW_S, "mono": True},
        "smoke": bool(args.smoke),
        "recordings": {},
    }
    constants: dict[str, Any] = {}
    for rid in rids:
        prot = [r for r in unit_rows("*__prot_r*.json") if r["rid"] == rid]
        entry: dict[str, Any] = {
            "seed": {
                "time_offset": manifest["recordings"][rid]["time_offset"],
                "time_dilation": manifest["recordings"][rid]["time_dilation"],
            },
            "n_windows": len(selection[rid]),
            "windows": [w["name"] for w in selection[rid]],
            "lag": lag_fits[rid],
        }
        val = [r for r in unit_rows("*__val.json") if r["rid"] == rid]
        if val:
            entry["scale"] = fit_global_scale(val)
            entry["val_windows"] = [
                {
                    "name": r["name"],
                    "best_b": r["best_b"],
                    "best_g_at_match": r["best_g_at_match"],
                    "delta_add_minus_mul": r["delta_add_minus_mul"],
                    "edge_b": r.get("edge_b"),
                    "edge_g": r.get("edge_g"),
                }
                for r in val
            ]
        if prot:  # opt-in cross-check, never the shipped number
            twin = twin_rotors(prot, args.twin_sep)
            entry["twins"] = twin
            entry["scale_perrotor_crosscheck"] = fit_scale(prot, set(twin["twins"]))
        summary["recordings"][rid] = entry
        proposed = entry["lag"].get("proposed", {})
        scale = entry.get("scale") or {}
        constants[rid] = {
            "time_offset": proposed.get("time_offset"),
            "time_dilation": proposed.get("time_dilation"),
            "rps_scale": scale.get("rps_scale"),
            "rps_scale_stderr": scale.get("rps_scale_stderr"),
            "n_lag_windows": len(entry["lag"].get("windows", [])),
            "lag_resid_rms_ms": entry["lag"].get("resid_rms_ms"),
            "lag_resid_rms_ms_constant_lag_model": entry["lag"].get(
                "resid_rms_ms_constant_lag_model"
            ),
            "lag_r2": entry["lag"].get("r2"),
            "n_scale_windows": scale.get("n_windows"),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "constants.json").write_text(json.dumps(constants, indent=1))
    print(f"\nCONSTANTS\n{json.dumps(constants, indent=1)}")
    print(f"\nwrote {out_dir}/summary.json + constants.json")
    if args.smoke:
        print("\n!! --smoke: tiny grids, one window — these constants are NOT a calibration")


if __name__ == "__main__":
    main()
