#!/usr/bin/env python3
"""Michael's telemetry calibration — the FULL sweep, one job, many cores.

Answers three questions about the SHIPPED alignment constants in
``src/data_processing/michaels.py`` (which this script never touches):

1. **FLY125 has never been measured.**  Scan the audio-optimal telemetry lag on
   every cruise window and regress it on window time — a constant lag means the
   ``time_offset`` is wrong, a linear trend means ``time_dilation`` is.
2. **FLY124 confirmation** on two windows only — a cheap cross-check of the
   already-established numbers (WP12 of ``docs/experiments/rps-refine-precision.md``:
   w02 −61.83, w03 −49.00, w04 −34.61, w05 −31.77 ms; OLS +0.654 ms/s·t − 86.1 ms,
   R² 0.94), NOT a re-derivation.
3. **Is the rev/s error additive or multiplicative**, for both recordings —
   ``val`` (matched families) plus the per-rotor ``prot`` discriminator, which
   has the leverage: additive ⇒ the per-rotor optimum is independent of that
   rotor's mean rps, multiplicative ⇒ proportional to it.

Layout::

    results/michaels_calib/manifest.json     window manifest + data-root provenance
    results/michaels_calib/raw/<name>__<stage>.json    one file per unit of work
    results/michaels_calib/summary.json      the fits + verdicts
    .cache/michaels_calib/<name>.npz         16 s prep cache (NOT an output)

**Restartable**: any unit whose JSON already exists is skipped, so a re-submit
after a timeout resumes.  **Parallel**: one process per (window, stage[, rotor]),
BLAS pinned to one thread each; the 44.1 kHz load + resample happens once, in
the parent, so workers only ever touch ~8 MB NPZs.

Typical cluster invocation (CPU-only, no GPU)::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 6h -- \\
        python scripts/michaels_calib/run_sweep.py --jobs 16

**After** the constants have been applied to ``michaels.py``, ``--post-shipped``
re-runs only the ``post`` stage on every cruise window of both recordings, on
windows built with whatever the loader now ships (offset, dilation AND the rev/s
scale, which lives inside ``_load_michaels_data_raw``). Success = residual lag
and residual offset both ≈ 0::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 4h -- \\
        python scripts/michaels_calib/run_sweep.py --jobs 16 --post-shipped \\
            --results results/michaels_calib_post --cache .cache/michaels_calib_post
"""

from __future__ import annotations

import os

# Pinned BEFORE numpy is imported: parallelism is across processes, so each
# worker must stay single-threaded or the node thrashes.
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

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "scripts"), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)

import calib as C  # noqa: E402  (imported in the PARENT so forked workers inherit it)
import windows as W  # noqa: E402
from fit import fit_lag, fit_prot, fit_val  # noqa: E402

#: FLY124 lags already established in WP12 of
#: ``docs/experiments/rps-refine-precision.md`` — the confirmation target.
FLY124_REFERENCE_MS = {
    "fly124_w02": -61.83,
    "fly124_w03": -49.00,
    "fly124_w04": -34.61,
    "fly124_w05": -31.77,
}

DEFAULT_RESULTS = REPO / "results" / "michaels_calib"
DEFAULT_CACHE = REPO / ".cache" / "michaels_calib"

#: per-recording lag-scan grid (frames of 0.032 s): (lo, hi, coarse step).
#: FLY124's optimum is known to sit near −2..−1 frames, FLY125's near −5; both
#: grids are wide enough to contain the whole recording's drift with margin.
LAG_GRID: dict[str, tuple[float, float, float]] = {
    "FLY124": (-4.0, 1.0, 0.25),
    "FLY125": (-10.0, 0.5, 0.5),
}


# ──────────────────────────────────────────────────────── unit of work
def unit_path(raw_dir: Path, name: str, stage: str) -> Path:
    return raw_dir / f"{name}__{stage}.json"


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def run_unit(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one (window, stage[, rotor]) unit in a worker process."""

    cache_dir = Path(task["cache_dir"])
    raw_dir = Path(task["raw_dir"])
    name, stage = task["name"], task["stage"]
    out = unit_path(raw_dir, name, stage)
    t0 = time.time()
    try:
        win = W.load_cached(cache_dir, name)
        kind = task["kind"]
        if kind == "lag":
            res = C.stage_lag(win, task["lo"], task["hi"], task["step"])
        elif kind == "val":
            res = C.stage_val(win, task["best_lag"], task["lo"], task["hi"], task["step"])
        elif kind == "prot":
            res = C.stage_prot(
                win, task["best_lag"], task["rotor"], task["lo"], task["hi"], task["step"]
            )
        elif kind == "post":
            res = C.stage_post(
                win, task["lo"], task["hi"], task["step"], task["blo"], task["bhi"], task["bstep"]
            )
        else:
            raise ValueError(f"unknown kind {kind!r}")
        res["elapsed_s"] = round(time.time() - t0, 1)
        write_json(out, res)
        return {"ok": True, "name": name, "stage": stage, "elapsed_s": res["elapsed_s"]}
    except Exception as exc:  # one bad unit must not sink the sweep
        return {
            "ok": False,
            "name": name,
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.time() - t0, 1),
        }


def dispatch(tasks: list[dict[str, Any]], jobs: int, label: str) -> list[dict[str, Any]]:
    if not tasks:
        print(f"[{label}] nothing to do", flush=True)
        return []
    print(f"[{label}] {len(tasks)} units on {jobs} workers", flush=True)
    t0 = time.time()
    done: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_unit, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done.append(r)
            flag = "ok " if r["ok"] else "FAIL"
            print(
                f"[{label}] {i}/{len(tasks)} {flag} {r['name']}__{r['stage']} "
                f"({r['elapsed_s']}s, {time.time() - t0:.0f}s elapsed)",
                flush=True,
            )
            if not r["ok"]:
                print(r["traceback"], flush=True)
    return done


# ──────────────────────────────────────────────────────── selection
def cruise_windows(manifest: dict[str, Any], rid: str) -> list[dict[str, Any]]:
    return [w for w in manifest["recordings"][rid]["windows"] if w["regime"] == "cruise"]


def stablest(wins: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The ``n`` windows with the steadiest telemetry (lowest mean per-rotor std).

    Deterministic given the manifest; the ``val``/``prot`` offset scans want
    near-constant speeds so the per-rotor optimum is not smeared by transients.
    """
    ranked = sorted(wins, key=lambda w: (float(sum(w["gt_std"])) / len(w["gt_std"]), w["index"]))
    return sorted(ranked[:n], key=lambda w: w["index"])


def best_lag_of(raw_dir: Path, name: str) -> float | None:
    p = unit_path(raw_dir, name, "lag")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if d.get("edge") else float(d["best_lag_frames"])


# ──────────────────────────────────────────────────────── driver
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--jobs", type=int, default=0, help="0 = all visible cores")
    ap.add_argument(
        "--stages",
        default="lag,val,prot",
        help="comma list of lag,val,prot (post is opt-in via --with-post)",
    )
    ap.add_argument("--fly124-windows", default="3,4", help="window indices for the FLY124 check")
    ap.add_argument("--val-windows", type=int, default=4, help="FLY125 windows for val/prot")
    ap.add_argument("--val-lo", type=float, default=-1.8)
    ap.add_argument("--val-hi", type=float, default=2.4)
    ap.add_argument("--val-step", type=float, default=0.3)
    ap.add_argument("--prot-lo", type=float, default=-1.5)
    ap.add_argument("--prot-hi", type=float, default=2.5)
    ap.add_argument("--prot-step", type=float, default=0.5)
    ap.add_argument(
        "--with-post",
        action="store_true",
        help="after fitting, rebuild both recordings with the PROPOSED constants and "
        "re-scan for residual lag/offset (adds a second load+resample pass)",
    )
    ap.add_argument(
        "--post-shipped",
        action="store_true",
        help="ONLY validate the constants currently shipped in "
        "src/data_processing/michaels.py (offset, dilation AND the rev/s scale): "
        "run the `post` stage on every cruise window of both recordings and stop. "
        "Success = residual lag and residual offset both ~0.",
    )
    ap.add_argument("--force", action="store_true", help="recompute units that already have JSON")
    ap.add_argument("--selftest", action="store_true", help="resolve data, build cache, then stop")
    args = ap.parse_args()

    results = Path(args.results)
    raw_dir = results / "raw"
    cache_dir = Path(args.cache)
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or len(os.sched_getaffinity(0))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    t_start = time.time()

    # ── phase 0: data + prep cache (serial, memory-bounded, cached) ──────────
    root, how = W.data_root()
    print(f"data root: {root}  ({how})", flush=True)
    manifest = W.build_cache(cache_dir, force=args.force)
    manifest["argv"] = sys.argv[1:]
    write_json(results / "manifest.json", manifest)
    for rid, rec in manifest["recordings"].items():
        n_c = len(cruise_windows(manifest, rid))
        print(
            f"{rid}: span {rec['eval_span']}  {len(rec['windows'])} windows "
            f"({n_c} cruise)  shipped offset {rec['time_offset']} dilation {rec['time_dilation']}",
            flush=True,
        )
    W.selfcheck()
    if args.selftest:
        print("selftest done", flush=True)
        return

    f124_idx = [int(x) for x in args.fly124_windows.split(",") if x.strip()]
    f124_wins = [w for w in manifest["recordings"]["FLY124"]["windows"] if w["index"] in f124_idx]
    f125_lag_wins = cruise_windows(manifest, "FLY125")
    f125_val_wins = stablest(f125_lag_wins, args.val_windows)
    scan_sets = {
        "FLY124": {"lag": f124_wins, "val": f124_wins},
        "FLY125": {"lag": f125_lag_wins, "val": f125_val_wins},
    }
    for rid, sel in scan_sets.items():
        print(
            f"{rid}: lag on {[w['name'] for w in sel['lag']]}; "
            f"val/prot on {[w['name'] for w in sel['val']]}",
            flush=True,
        )

    common = {"cache_dir": str(cache_dir), "raw_dir": str(raw_dir)}

    def pending(name: str, stage: str) -> bool:
        return args.force or not unit_path(raw_dir, name, stage).exists()

    # ── post-only: validate whatever michaels.py currently ships ─────────────
    # The prep cache above was built with the SHIPPED constants, and the rev/s
    # calibration lives inside `_load_michaels_data_raw`, so these windows carry
    # the fully corrected labels: a residual scan on them IS the validation.
    if args.post_shipped:
        summary = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_root": how,
            "mode": "post_shipped",
            "shipped": {
                rid: {
                    "time_offset": rec["time_offset"],
                    "time_dilation": rec["time_dilation"],
                    "rps_scale": W.shipped_rps_scale(rid),
                }
                for rid, rec in manifest["recordings"].items()
            },
        }
        tasks = []
        for rid in manifest["recordings"]:
            lo, hi, step = LAG_GRID[rid]
            half = (hi - lo) / 4.0
            tasks += [
                {
                    **common,
                    "kind": "post",
                    "name": w["name"],
                    "stage": "post",
                    "lo": -half,
                    "hi": half,
                    "step": step,
                    "blo": args.val_lo,
                    "bhi": args.val_hi,
                    "bstep": args.val_step,
                }
                for w in cruise_windows(manifest, rid)
                if pending(w["name"], "post")
            ]
        dispatch(tasks, jobs, "post-shipped")
        rows = [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*__post.json"))]
        summary["post"] = {r["name"]: r for r in rows}
        for rid in manifest["recordings"]:
            sel = [r for r in rows if r["rid"] == rid]
            if not sel:
                continue
            lags = [float(r["resid_lag_ms"]) for r in sel]
            bs = [float(r["resid_b"]) for r in sel]
            summary.setdefault("post_aggregate", {})[rid] = {
                "n_windows": len(sel),
                "resid_lag_ms": {
                    "mean": round(float(np.mean(lags)), 3),
                    "rms": round(float(np.sqrt(np.mean(np.square(lags)))), 3),
                    "max_abs": round(float(np.max(np.abs(lags))), 3),
                },
                "resid_b_revps": {
                    "mean": round(float(np.mean(bs)), 4),
                    "max_abs": round(float(np.max(np.abs(bs))), 4),
                },
                "n_edge": int(sum(bool(r["edge"]) or bool(r["edge_b"]) for r in sel)),
            }
        summary["wall_s"] = round(time.time() - t_start, 1)
        write_json(results / "summary.json", summary)
        print(json.dumps(summary.get("post_aggregate", {}), indent=1), flush=True)
        print(f"\nwrote {results / 'summary.json'} ({summary['wall_s']}s wall)", flush=True)
        return

    # ── phase 1: lag scans ──────────────────────────────────────────────────
    if "lag" in stages:
        tasks = []
        for rid, sel in scan_sets.items():
            lo, hi, step = LAG_GRID[rid]
            tasks += [
                {
                    **common,
                    "kind": "lag",
                    "name": w["name"],
                    "stage": "lag",
                    "lo": lo,
                    "hi": hi,
                    "step": step,
                }
                for w in sel["lag"]
                if pending(w["name"], "lag")
            ]
        dispatch(tasks, jobs, "lag")

    # ── phase 2: lag fit (needed by val/prot for the per-window best lag) ────
    summary: dict[str, Any] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "data_root": how}
    for rid, rec in manifest["recordings"].items():
        rows = []
        for w in scan_sets[rid]["lag"]:
            p = unit_path(raw_dir, w["name"], "lag")
            if p.exists():
                d = json.loads(p.read_text())
                if d["regime"] == "cruise" and not d.get("edge"):
                    rows.append(d)
                else:
                    print(
                        f"  skip {d['name']}: regime={d['regime']} edge={d.get('edge')}", flush=True
                    )
        summary.setdefault("lag", {})[rid] = fit_lag(rows, rec["time_offset"], rec["time_dilation"])
    # FLY124 is a CONFIRMATION, not a re-derivation: check the two rescanned
    # windows against WP12's published lags rather than refitting the line.
    summary["fly124_confirmation"] = [
        {
            "name": r["name"],
            "measured_ms": r["best_lag_ms"],
            "reference_ms": FLY124_REFERENCE_MS.get(r["name"]),
            "delta_ms": (
                None
                if r["name"] not in FLY124_REFERENCE_MS
                else round(r["best_lag_ms"] - FLY124_REFERENCE_MS[r["name"]], 3)
            ),
            "best_recon": r["best_recon"],
            "raw_recon": r["raw_recon"],
        }
        for w in scan_sets["FLY124"]["lag"]
        if (p := unit_path(raw_dir, w["name"], "lag")).exists()
        for r in [json.loads(p.read_text())]
    ]
    write_json(results / "summary.json", summary)
    print(json.dumps(summary["lag"], indent=1)[:4000], flush=True)

    # ── phase 3: val + prot at each window's best lag ────────────────────────
    tasks = []
    for sel in scan_sets.values():
        for w in sel["val"]:
            lag = best_lag_of(raw_dir, w["name"])
            if lag is None:
                print(f"  {w['name']}: no usable lag scan -> val/prot skipped", flush=True)
                continue
            if "val" in stages and pending(w["name"], "val"):
                tasks.append(
                    {
                        **common,
                        "kind": "val",
                        "name": w["name"],
                        "stage": "val",
                        "best_lag": lag,
                        "lo": args.val_lo,
                        "hi": args.val_hi,
                        "step": args.val_step,
                    }
                )
            if "prot" in stages:
                tasks += [
                    {
                        **common,
                        "kind": "prot",
                        "name": w["name"],
                        "stage": f"prot_r{r}",
                        "best_lag": lag,
                        "rotor": r,
                        "lo": args.prot_lo,
                        "hi": args.prot_hi,
                        "step": args.prot_step,
                    }
                    for r in range(W.N_ROTORS)
                    if pending(w["name"], f"prot_r{r}")
                ]
    dispatch(tasks, jobs, "val+prot")

    # ── phase 4: verdicts ───────────────────────────────────────────────────
    for rid in scan_sets:
        tag = rid.lower()
        summary.setdefault("val", {})[rid] = fit_val(
            [json.loads(p.read_text()) for p in sorted(raw_dir.glob(f"{tag}_w*__val.json"))]
        )
        summary.setdefault("prot", {})[rid] = fit_prot(
            [json.loads(p.read_text()) for p in sorted(raw_dir.glob(f"{tag}_w*__prot_r*.json"))]
        )
    write_json(results / "summary.json", summary)

    # ── phase 5 (opt-in): validate the proposed constants ────────────────────
    if args.with_post:
        try:
            post_cache = cache_dir.parent / f"{cache_dir.name}_post"
            offs = {r: summary["lag"][r]["proposed"]["time_offset"] for r in scan_sets}
            dils = {r: summary["lag"][r]["proposed"]["time_dilation"] for r in scan_sets}
            print(f"post: rebuilding with offsets {offs} dilations {dils}", flush=True)
            post_man = W.build_cache(post_cache, time_offsets=offs, time_dilations=dils)
            write_json(results / "manifest_post.json", post_man)
            tasks = []
            for rid, sel in scan_sets.items():
                lo, hi, step = LAG_GRID[rid]
                half = (hi - lo) / 4.0
                tasks += [
                    {
                        "cache_dir": str(post_cache),
                        "raw_dir": str(raw_dir),
                        "kind": "post",
                        "name": w["name"],
                        "stage": "post",
                        "lo": -half,
                        "hi": half,
                        "step": step,
                        "blo": args.val_lo,
                        "bhi": args.val_hi,
                        "bstep": args.val_step,
                    }
                    for w in sel["val"]
                    if pending(w["name"], "post")
                ]
            dispatch(tasks, jobs, "post")
            summary["post"] = {
                p.stem.split("__")[0]: json.loads(p.read_text())
                for p in sorted(raw_dir.glob("*__post.json"))
            }
        except Exception as exc:
            summary["post"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(traceback.format_exc(), flush=True)

    summary["wall_s"] = round(time.time() - t_start, 1)
    write_json(results / "summary.json", summary)
    print(f"\nwrote {results / 'summary.json'} ({summary['wall_s']}s wall)", flush=True)


if __name__ == "__main__":
    main()
