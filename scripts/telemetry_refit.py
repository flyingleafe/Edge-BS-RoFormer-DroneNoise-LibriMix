#!/usr/bin/env python3
"""Fit the rotors to the harmonics on protocol windows (GitHub issue 17, phase 6b).

A thin ``utils.gridrun`` driver over ``tracking.telemetry_refit``. One unit is
one (window, arm); an arm is a named :class:`tracking.telemetry_refit.RefitConfig`
override, so the six-step procedure itself lives in the library and this file
only chooses windows and knobs.

Every unit writes TWO things:

  ``<out>/raw/<uid>.json``     the report — k ladder, per-iteration deltas,
                               convergence, the scale readings and the 6a
                               ``residual_decompose`` block against the RAW
                               telemetry.
  ``<out>/traj/<arm>/<key>.npz`` the trajectories: ``ft``, ``r_raw``,
                               ``r_init`` (pre-smoothed), ``r_fit``.

The ``.npz`` is the CANDIDATE FILE FORMAT shared with ``scripts/telemetry_fitness.py``
(phase 6a), whose candidate language already has the hook: ``file:PATH:KEY``
loads ``np.load(PATH)[KEY]`` and uses it on the window's own frame grid. The two
scripts agree on one further convention so a whole directory can be scored in
one command — ``{key}`` in a candidate spec is replaced by the window key:

    python scripts/telemetry_refit.py --arms main --jobs 4
    python scripts/telemetry_fitness.py --dataset all \\
        --candidates 'telemetry,file:results/telemetry_refit/traj/main/{key}.npz:r_fit'

Examples:
  python scripts/telemetry_refit.py --smoke --jobs 2
  python scripts/telemetry_refit.py --windows FLY124__w02 --arms main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]  # this checkout (code)
sys.path.insert(0, str(ROOT / "src"))

from telemetry_fitness import ALL_WINDOWS, DREGON_WINDOWS, FLY124_WINDOWS  # noqa: E402

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402
from utils.paths import get_data_root  # noqa: E402

PREP = get_data_root() / (
    "omnirun-outputs/bandadm-ladder-7fb2e4/results/beatvk_bandadm/vk_arms/prep_cache"
)
OUT_DEFAULT = "results/telemetry_refit"

#: Named arms — each is a ``RefitConfig`` field override. ``main`` is the
#: procedure exactly as issue 17 states it. The others isolate ONE step, so the
#: campaign can say what each step bought rather than shipping a bundle.
ARMS: dict[str, dict[str, Any]] = {
    "main": {},
    "nosmooth": {"smooth_cut_hz": 0.0},  # step 1 off
    "nopeel": {"peel": False},  # step 3 off
    "flatk": {"k_start_max": 96, "e0_rev_s": 0.08},  # step 2 off: the old k_caps=(80,80,80)
    "gate": {"pair_mode": "gate"},  # step 6 off
    "b0_3": {"band_b0": 3.0},  # the campaign's other identity-preserving arm
}

#: The frozen DREGON window of the displacement campaign, and the first FLY124
#: cruise window — the same smoke pair phase 6a uses.
SMOKE_WINDOWS = ("free-flight_nosource_room1__w01", "FLY124__w02")

PROTOCOL = {
    "dataset": "beatvk-valid-raw@54849c13ed3a",
    "procedure": "issue 17 steps 1-6: pre-smoothed telemetry init (5 Hz), "
    "coarse-to-fine k from the phase-wrap capture rule, LS-projected peel "
    "alternating with pi_kalman_refine, convergence stop on max |dr|, "
    "pair_mode=joint for the twins",
    "band": "k-scaled, b0 rev/s of capture at every harmonic",
    "reported": "the k ladder, every iteration's delta, the scale two ways "
    "(per-rotor mean shift and one joint global LS scale) and the 6a "
    "residual_decompose block against the RAW telemetry",
    "not_reported": "any goodness of fit — that is scripts/telemetry_fitness.py, "
    "at fixed degrees of freedom with its four controls",
}


def _load(key: str) -> dict[str, Any]:
    """The frozen prep-cache window: audio, frame grid, telemetry, regime."""
    import numpy as np

    with np.load(PREP / f"{key}.npz") as z:
        return {
            "audio": np.asarray(z["audio"], np.float64),
            "ft": np.asarray(z["ft"], np.float64),
            "r": np.asarray(z["r_meas"], np.float64),
            "regime": str(z["regime"]),
        }


def worker(unit: Unit) -> dict[str, Any]:
    """One (window, arm) unit: the refit, the report, and the candidate ``.npz``."""
    import numpy as np

    from tracking.telemetry_refit import RefitConfig, refit_window

    p = unit.params
    key, arm = str(p["key"]), str(p["arm"])
    over = dict(ARMS[arm])
    for name in ("max_iters", "tol_rev_s", "k_top", "band_b0"):
        if p.get(name) is not None:
            over[name] = p[name]
    cfg = RefitConfig(**over)
    win = _load(key)
    res = refit_window(win["audio"], win["r"], win["ft"], 16000, cfg=cfg, verbose=True)

    traj_dir = Path(str(p["out"])) / "traj" / arm
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / f"{key}.npz"
    np.savez(
        path, ft=res.ft, r_raw=res.r_raw, r_init=res.r_init, r_fit=res.r_fit.astype(np.float64)
    )
    return {
        **res.as_dict(),
        "key": key,
        "recording": key.split("__")[0],
        "regime": win["regime"],
        "arm": arm,
        "candidate_spec": f"file:{path}:r_fit",
    }


# ---------------------------------------------------------------------------
# report


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled scale / convergence / identity per (dataset, arm)."""
    pooled: dict[str, Any] = {}
    for name, is_fly in (("dregon", False), ("fly124", True)):
        sel = [r for r in rows if r["key"].startswith("FLY124") == is_fly and "scale" in r]
        if not sel:
            continue
        block: dict[str, Any] = {}
        for arm in sorted({r["arm"] for r in sel}):
            got = [r for r in sel if r["arm"] == arm]
            block[arm] = {
                "n_windows": len(got),
                "global_pct": _mean([r["scale"]["global_pct"] for r in got]),
                "global_pct_spread": _spread([r["scale"]["global_pct"] for r in got]),
                "d_rms": _mean([v for r in got for v in r["scale"]["d_rms"]]),
                "resid_rms": _mean([r["residual"]["pooled"].get("resid_rms") for r in got]),
                "tach_bound_frac": _mean(
                    [r["residual"]["pooled"].get("tach_bound_frac") for r in got]
                ),
                "n_iters": _mean([r["n_iters"] for r in got]),
                "converged": sum(bool(r["converged"]) for r in got),
                "stop_reason": sorted({r.get("stop_reason", "?") for r in got}),
                "design_cond": _mean([r["residual"]["pooled"].get("design_cond") for r in got]),
                "k_ladder_last": [r["k_ladder"] for r in got][0],
                "order_kept": sum(bool(r["identity"]["order_kept"]) for r in got),
                "gap_ratio": _mean([v for r in got for v in r["identity"]["gap_ratio"] or []]),
            }
        pooled[name] = block
    return {"protocol": PROTOCOL, "arms": {k: v for k, v in ARMS.items()}, "pooled": pooled}


def _mean(vals: list[Any]) -> float | None:
    import numpy as np

    v = np.asarray([x for x in vals if isinstance(x, (int, float))], dtype=np.float64)
    v = v[np.isfinite(v)]
    return round(float(v.mean()), 5) if v.size else None


def _spread(vals: list[Any]) -> float | None:
    import numpy as np

    v = np.asarray([x for x in vals if isinstance(x, (int, float))], dtype=np.float64)
    v = v[np.isfinite(v)]
    return round(float(v.max() - v.min()), 5) if v.size > 1 else None


def print_table(summary: dict[str, Any]) -> None:
    """Scale beside its identity check — a scale with a collapsed gap is not a scale."""
    for ds, block in summary.get("pooled", {}).items():
        print(f"\n=== {ds} ===")
        print(
            f"{'arm':10s}{'n':>3s}{'global %':>10s}{'spread':>9s}{'d_rms':>8s}"
            f"{'resid':>8s}{'iters':>7s}{'conv':>6s}{'order':>7s}{'gap x':>8s}"
        )
        for arm in sorted(block):
            b = block[arm]

            def f(key: str, w: int, nd: int = 4, b: dict[str, Any] = b) -> str:
                v = b.get(key)
                return f"{v:{w}.{nd}f}" if isinstance(v, (int, float)) else f"{'—':>{w}s}"

            print(
                f"{arm:10s}{b['n_windows']:3d}{f('global_pct', 10)}{f('global_pct_spread', 9)}"
                f"{f('d_rms', 8, 3)}{f('resid_rms', 8, 3)}{f('n_iters', 7, 1)}"
                f"{b['converged']:6d}{b['order_kept']:7d}{f('gap_ratio', 8, 3)}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="all", choices=("dregon", "fly124", "all"))
    ap.add_argument("--windows", default="", help="comma-separated keys (overrides --dataset)")
    ap.add_argument("--arms", default="main", help=f"comma-separated; known: {','.join(ARMS)}")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--tol", type=float, default=None, help="convergence tolerance, rev/s")
    ap.add_argument("--k-top", type=int, default=None, help="ceiling of the k ladder")
    ap.add_argument("--b0", type=float, default=None, help="k-scaled band, rev/s of capture")
    ap.add_argument("--smoke", action="store_true", help="the two smoke windows only")
    ap.add_argument("--out", default=OUT_DEFAULT)
    add_gridrun_args(ap, jobs=2)
    args = ap.parse_args()

    if args.smoke:
        keys = list(SMOKE_WINDOWS)
    elif args.windows:
        keys = [k.strip() for k in args.windows.split(",") if k.strip()]
    else:
        keys = list(
            {"dregon": DREGON_WINDOWS, "fly124": FLY124_WINDOWS, "all": ALL_WINDOWS}[args.dataset]
        )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for bad in [k for k in keys if k not in ALL_WINDOWS]:
        ap.error(f"unknown window {bad!r}; known: {', '.join(ALL_WINDOWS)}")
    for bad in [a for a in arms if a not in ARMS]:
        ap.error(f"unknown arm {bad!r}; known: {', '.join(ARMS)}")

    common = {
        "out": str(args.out),
        "max_iters": args.max_iters,
        "tol_rev_s": args.tol,
        "k_top": args.k_top,
        "band_b0": args.b0,
    }
    units = [Unit(f"{k}__{a}", {"key": k, "arm": a, **common}) for k in sorted(keys) for a in arms]
    print(f"[telemetry_refit] {len(units)} units", flush=True)
    res = gridrun_from_args(args, units, worker, args.out, summarize=summarize)
    print_table(res.summary)
    print(
        "\nscore these with:\n  python scripts/telemetry_fitness.py --dataset all \\\n"
        f"    --candidates 'telemetry,file:{args.out}/traj/{arms[0]}/{{key}}.npz:r_fit'"
    )
    raise SystemExit(res.exit_code)


if __name__ == "__main__":
    main()
