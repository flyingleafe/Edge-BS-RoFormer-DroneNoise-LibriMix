#!/usr/bin/env python3
"""joint_beam (WP17) — arm sweep and full-protocol run, one job, many cores.

The lab harness (``scripts/rps_refine_lab.py``) runs its windows serially in one
process; this fans out one process per (arm, window) so a 28-window protocol
plus an 8-arm sweep fits in a single CPU job.

Two modes:

``--mode sweep``
    A handful of ``joint_beam`` configurations on a small, cheap window set, to
    pick the emission/prior balance.  The levers are the ones WP17 could not
    settle on synthetic data alone: the k-scaled analysis bandwidth
    (``b0_rps``), the harmonic weighting (``k_weight``), the emission-vs-prior
    balance (``lambda_e`` / ``s_scale``), whether the differential trim is
    frozen at frame 0 or tracked, and whether mean-reverting the COMMON mode
    (which the WP16 measurement says would flatten ramps) actually hurts.

``--mode init``
    **Init quality only** (design §6 item 1) — the honest test of the idea, and
    the cheap one: run ONLY the init stage and score its trajectory against the
    telemetry.  No VK capture, no M1, no M2, and — for ``joint_beam`` — no
    ``blind_seed``, because the joint search takes its candidates from the score
    surface rather than from seed bases.  A window costs seconds instead of two
    minutes, so the whole 15-window protocol runs anywhere.  Reports PIT-MAE
    plus the two structural numbers that say whether the shared-shape defect is
    fixed: per-rotor ``std_ratio`` (should approach 1) and the SPREAD of the
    four shape correlations (the coarse stage makes them nearly identical by
    construction, so a larger spread is the point).  ``fullrange_init`` is
    scored alongside on any window whose blind seed is already cached.

``--mode protocol``
    One or more named arms plus a ``refine_v2`` control on the full 15-window
    real protocol, the synthetic battery, the band-limited battery and the
    trace window.  The control is re-run rather than quoted so both columns come
    from the same prep build and the same machine.

Both write one JSON per unit and are restartable (an existing unit is skipped),
so a re-submit after a timeout resumes.

Typical cluster invocation::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 6h -- \\
        python scripts/jb_sweep.py --mode sweep --jobs 16
    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 8h -- \\
        python scripts/jb_sweep.py --mode protocol --jobs 16
"""

from __future__ import annotations

import os

# Pinned BEFORE numpy/torch: parallelism is across processes, so each worker
# stays single-threaded or the node thrashes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402

#: The 15-window frozen real protocol (`beatvk-valid-raw@54849c13ed3a`).
REAL_WINDOWS: tuple[str, ...] = tuple(
    f"real:{rid}:{w}"
    for rid in (
        "free-flight_nosource_room1",
        "free-flight_speech-low_room1",
        "free-flight_whitenoise-low_room1",
    )
    for w in range(3)
) + tuple(f"real:FLY124:{w}" for w in range(6))

#: The cheap window set the arm sweep runs on: both lab reference windows (one
#: DREGON ramp, one FLY124 cruise) plus one steady DREGON and one more FLY124
#: cruise, so an arm cannot win by suiting a single regime.
SWEEP_WINDOWS = (
    "real:free-flight_nosource_room1:0",  # dregon ramp
    "real:free-flight_nosource_room1:1",  # dregon steady
    "real:FLY124:3",  # fly124 cruise (the comb-invisible-rotor window)
    "real:FLY124:4",  # fly124 cruise
    "synth_trace",
)

PROTOCOL_WINDOWS = (
    *REAL_WINDOWS,
    "synth_trace",
    *(f"synth{i:02d}" for i in range(6)),
    *(f"synthbl{i:02d}" for i in range(6)),
)

#: Arms.  ``ou`` / ``emis`` / ``beam`` are kwargs of the corresponding
#: ``joint_beam_tracker`` dataclasses; ``{}`` means "the shipped default".
ARMS: dict[str, dict[str, Any]] = {
    # --- the default, and the two levers the synthetic could not settle
    "jb_default": {},
    "jb_b0_025": {"emis": {"b0_rps": 0.25}},
    "jb_b0_100": {"emis": {"b0_rps": 1.0}},
    "jb_kw_uniform": {"emis": {"k_weight": "uniform"}},
    # --- emission-vs-prior balance (only the ratio matters)
    "jb_stiff": {"ou": {"s_scale": 1.5}},
    "jb_loose": {"ou": {"s_scale": 6.0}},
    # --- the differential trim reference
    "jb_mu_running": {"beam": {"mu_mode": "running"}},
    # --- WP16 says a mean-reverting COMMON mode flattens ramps.  Measure it.
    "jb_ou_common": {"ou": {"tau_common": 1.28}},
    # --- the overlap correction's width, and turning it off entirely
    "jb_overlap_wide": {"beam": {"overlap_sigma_rps": 1.5}},
    "jb_overlap_off": {"beam": {"overlap_gain": 0.0}},
}

#: Arms carried into `--mode protocol` by default (plus the control).
PROTOCOL_ARMS = ("jb_default",)


def init_unit(task: tuple[str, str, Path]) -> tuple[str, str, str]:
    """One (arm, window) INIT-ONLY pair: run the init stage, score it, stop."""
    arm, window, results = task
    out = unit_path(results, arm, window)
    if out.exists():
        return arm, window, "skip"
    tic = time.perf_counter()
    try:
        import rps_refine_lab as lab
        from beatvk_vk_arms import _coarse_spec, fullrange_init

        from data_processing.joint_beam_tracker import (
            BeamCfg,
            EmissionCfg,
            OUPrior,
            joint_beam_track,
        )

        _, rid, widx = window.split(":")
        prep, _weights, meta = lab.real_window(rid, int(widx))
        rec: dict[str, Any] = {"arm": arm, "window": window, "regime": meta.get("regime")}
        if arm == "fullrange_init":
            # The control DOES need `blind_seed` (~1 min/window, cached).  That
            # asymmetry is itself a result: the joint search needs no seed.
            seed = lab.get_seed(f"{rid}_w{int(widx):02d}", prep, True)
            r0, _, diag = fullrange_init(prep, seed)
            rec["coarse_mode"] = diag["coarse_mode"]
        else:
            cfg = ARMS.get(arm, {})
            lm, bin_hz, st, _ = _coarse_spec(prep.audio)
            r0, diag = joint_beam_track(
                lm,
                bin_hz,
                st,
                prep.ft,
                ou=OUPrior(**cfg.get("ou", {})),
                emis=EmissionCfg(**cfg.get("emis", {})),
                beam=BeamCfg(**cfg.get("beam", {})),
            )
            rec["jb"] = diag
        rec.update(lab.stage_metrics(r0, prep))
        per = rec["per_rotor"]
        corrs = [q["shape_corr"] for q in per if q["shape_corr"] is not None]
        rec["std_ratio_mean"] = float(np.mean([q["std_ratio"] for q in per]))
        rec["shape_corr_mean"] = float(np.mean(corrs)) if corrs else None
        rec["shape_corr_spread"] = float(np.ptp(corrs)) if len(corrs) > 1 else None
        rec["wall_s"] = round(time.perf_counter() - tic, 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(lab.r3(rec), indent=1))
        os.replace(tmp, out)
        return arm, window, "ok"
    except Exception:  # noqa: BLE001
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(traceback.format_exc())
        return arm, window, "ERROR"


def unit_path(results: Path, arm: str, window: str) -> Path:
    return results / "raw" / f"{arm}__{window.replace(':', '_')}.json"


def run_unit(task: tuple[str, str, Path, int]) -> tuple[str, str, str]:
    """One (arm, window) pair.  Returns ``(arm, window, status)``."""
    arm, window, results, v2_rounds = task
    out = unit_path(results, arm, window)
    if out.exists():
        return arm, window, "skip"
    tic = time.perf_counter()
    try:
        import rps_refine_lab as lab

        cfg = ARMS.get(arm, {})
        lab.JB_OU = cfg.get("ou", {})
        lab.JB_EMIS = cfg.get("emis", {})
        lab.JB_BEAM = cfg.get("beam", {})
        chain = "refine_v2" if arm == "refine_v2" else "joint_beam"

        if window.startswith("real:"):
            _, rid, widx = window.split(":")
            name = f"{rid}_w{int(widx):02d}"
            prep, weights, meta = lab.real_window(rid, int(widx))
        elif window == "synth_trace":
            name = window
            prep, weights, meta = lab.synth_window(99, 1.0, mode_means=lab.TRACE_MODES)
        elif window.startswith("synthbl"):
            i = int(window[7:])
            name = window
            prep, weights, meta = lab.synth_window(
                100 + i, lab.AGGR_CYCLE[i % 3], fc_hz=8.0, snr_db=0.0
            )
        else:
            i = int(window[5:])
            name = window
            prep, weights, meta = lab.synth_window(100 + i, lab.AGGR_CYCLE[i % 3])

        res = lab.run_chain(
            name,
            prep,
            weights,
            meta,
            chain,
            dict(lab.DEFAULT_PK),
            1,
            False,
            True,
            True,
            v2_rounds,
        )
        res["arm"] = arm
        res["window"] = window
        res["wall_s"] = round(time.perf_counter() - tic, 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(lab.r3(res), indent=1))
        os.replace(tmp, out)
        return arm, window, "ok"
    except Exception:  # noqa: BLE001 - one bad unit must not kill the sweep
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(traceback.format_exc())
        return arm, window, "ERROR"


# --------------------------------------------------------------------------
# summarisation


def _pool_of(window: str, regime: str | None) -> str:
    if window.startswith("real:FLY124"):
        w = int(window.rsplit(":", 1)[1])
        return "fly124_warmup" if w < 2 else "fly124_cruise"
    if window.startswith("real:"):
        return "dregon_cruise"
    if window.startswith("synthbl"):
        return "synthbl"
    if window == "synth_trace":
        return "synth_trace"
    return "synth"


#: Windows whose regime is a takeoff ramp rather than steady flight.  The
#: protocol tags them `cruise` (window-mean rps 59.8-67.6 >= 45) but each is its
#: recording's takeoff window, and design §6 asks for them separately because
#: the stage being replaced has ramps as its one genuine strength.
RAMP_WINDOWS = frozenset(
    {
        "real:free-flight_nosource_room1:0",
        "real:free-flight_speech-low_room1:0",
        "real:free-flight_whitenoise-low_room1:0",
        "real:FLY124:0",
        "real:FLY124:1",
    }
)


def summarise_init(results: Path) -> dict[str, Any]:
    rows = [json.loads(f.read_text()) for f in sorted((results / "raw").glob("*.json"))]
    arms: dict[str, Any] = {}
    for arm in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == arm]
        agg: dict[str, Any] = {}
        for pool in sorted({_pool_of(r["window"], None) for r in sel}):
            grp = [r for r in sel if _pool_of(r["window"], None) == pool]
            agg[pool] = {
                "n": len(grp),
                "mae": round(float(np.mean([r["pooled_mae"] for r in grp])), 3),
                "std_ratio": round(float(np.mean([r["std_ratio_mean"] for r in grp])), 3),
                "shape_corr": round(float(np.mean([r["shape_corr_mean"] for r in grp])), 3),
                "corr_spread": round(float(np.mean([r["shape_corr_spread"] for r in grp])), 3),
            }
        for tag, want in (("ramp", True), ("steady", False)):
            grp = [r for r in sel if (r["window"] in RAMP_WINDOWS) is want]
            if grp:
                agg[tag] = {
                    "n": len(grp),
                    "mae": round(float(np.mean([r["pooled_mae"] for r in grp])), 3),
                    "std_ratio": round(float(np.mean([r["std_ratio_mean"] for r in grp])), 3),
                    "corr_spread": round(float(np.mean([r["shape_corr_spread"] for r in grp])), 3),
                }
        agg["wall_s"] = round(float(np.mean([r["wall_s"] for r in sel])), 1)
        arms[arm] = agg
    return {"rows": rows, "arms": arms}


def summarise(results: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for f in sorted((results / "raw").glob("*.json")):
        d = json.loads(f.read_text())
        stages = {s["stage"]: s for s in d["stages"]}
        init = stages.get("joint_beam") or stages.get("coarse_init") or {}
        # The stage the two chains have in common right after their inits.
        cap = stages.get("capture", {})
        per = init.get("per_rotor") or []
        corrs = [q["shape_corr"] for q in per if q.get("shape_corr") is not None]
        rows.append(
            {
                "arm": d["arm"],
                "window": d["window"],
                "pool": _pool_of(d["window"], d["meta"].get("regime")),
                "ramp": d["window"] in RAMP_WINDOWS,
                "init_mae": init.get("pooled_mae"),
                "init_std_ratio": (float(np.mean([q["std_ratio"] for q in per])) if per else None),
                "init_shape_corr": float(np.mean(corrs)) if corrs else None,
                # spread of the four shape correlations: the shared-shape defect
                # makes them nearly identical, so a LARGER spread is the point
                "init_corr_spread": float(np.ptp(corrs)) if len(corrs) > 1 else None,
                "capture_mae": cap.get("pooled_mae"),
                "final_mae": d["final_pooled_mae"],
                "oracle": (d.get("oracle_floor") or {}).get("pooled_mae"),
                "wall_s": d.get("wall_s"),
                "jb": (d["meta"].get("alt") or {}).get("joint_beam"),
            }
        )
    pools: dict[str, dict[str, Any]] = {}
    for arm in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == arm]
        agg: dict[str, Any] = {}
        for pool in sorted({r["pool"] for r in sel}):
            grp = [r for r in sel if r["pool"] == pool]
            agg[pool] = {
                "n": len(grp),
                "final": round(float(np.mean([r["final_mae"] for r in grp])), 3),
                "init": round(float(np.mean([r["init_mae"] for r in grp])), 3),
            }
        for tag, want in (("ramp", True), ("steady", False)):
            grp = [r for r in sel if r["ramp"] is want and r["window"].startswith("real:")]
            if grp:
                agg[tag] = {
                    "n": len(grp),
                    "final": round(float(np.mean([r["final_mae"] for r in grp])), 3),
                    "init": round(float(np.mean([r["init_mae"] for r in grp])), 3),
                }
        pools[arm] = agg
    return {"rows": rows, "pools": pools}


def main() -> None:
    ap = argparse.ArgumentParser(description="joint_beam sweep / protocol driver")
    ap.add_argument(
        "--mode", choices=("sweep", "protocol", "init", "summarise"), default="sweep"
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--results", default=None)
    ap.add_argument("--arms", default=None, help="comma list overriding the mode's arms")
    ap.add_argument("--windows", default=None, help="comma list overriding the mode's windows")
    ap.add_argument("--v2-rounds", type=int, default=1, help="M1 rounds (WP6 real default: 1)")
    args = ap.parse_args()

    results = Path(args.results or f"results/jb_{args.mode}")
    if args.mode == "summarise":
        s = summarise(results)
        (results / "summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s["pools"], indent=1))
        return

    if args.mode == "init":
        arms = ["jb_default", "jb_b0_100", "fullrange_init"]
        windows = list(REAL_WINDOWS)
    elif args.mode == "sweep":
        arms = list(ARMS)
        windows = list(SWEEP_WINDOWS)
    else:
        arms = [*PROTOCOL_ARMS, "refine_v2"]
        windows = list(PROTOCOL_WINDOWS)
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.windows:
        windows = [w.strip() for w in args.windows.split(",") if w.strip()]

    if args.mode == "init":
        itasks = [(a, w, results) for a in arms for w in windows]
        print(f"{len(itasks)} init units, {args.jobs} jobs")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for fut in as_completed({pool.submit(init_unit, t): t for t in itasks}):
                arm, window, status = fut.result()
                print(f"[{time.perf_counter() - t0:6.0f}s] {status:7s} {arm} {window}", flush=True)
        s = summarise_init(results)
        (results / "summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s["arms"], indent=1))
        return

    tasks = [(a, w, results, args.v2_rounds) for a in arms for w in windows]
    print(f"{len(tasks)} units ({len(arms)} arms x {len(windows)} windows), {args.jobs} jobs")
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_unit, t): t for t in tasks}
        for fut in as_completed(futs):
            arm, window, status = fut.result()
            done += 1
            print(
                f"[{done}/{len(tasks)} {time.perf_counter() - t0:6.0f}s] {status:5s} "
                f"{arm} {window}",
                flush=True,
            )
    s = summarise(results)
    (results / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s["pools"], indent=1))
    print(f"wrote {results / 'summary.json'}")


if __name__ == "__main__":
    main()
