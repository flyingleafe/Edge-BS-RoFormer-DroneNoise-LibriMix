#!/usr/bin/env python3
"""Per-rotor telemetry error on Michael's recordings: offset vs LAG, many windows.

The 2026-07-31 calibration shipped ONE global multiplicative rev/s scale per
recording, fitted from 2-4 windows.  This script tests the two objections to
that model, on **every cruise window** of both recordings:

1. **Is the error per-rotor rather than global?**  Stage ``off`` scans a blind
   additive offset on rotor ``r`` alone (the other three untouched), on the
   SAME label-free referee as ``run_sweep.py`` (VK reconstruction residual).
   With every cruise window scanned we can finally compare a rotor's
   *within-recording scatter* against the *between-rotor spread*.
2. **Is it a LAG rather than an offset?**  A per-rotor time lag ``tau_r`` makes
   the telemetry error proportional to ``d(rps)/dt``, not constant.  Three
   probes:
   - stage ``lag``: scan a shift of rotor ``r`` alone -> do per-rotor optimal
     lags differ?
   - the ``off`` scan stores **per-frame** residuals, so the apparent offset
     can be re-minimised on any frame subset -> split by ``|d rps/dt|`` and
     see whether the offset grows with acceleration (lag) or stays flat
     (calibration).
   - stage ``joint``: a 2D ``(tau_r, b_r)`` grid -> which parameter carries the
     explanatory power.

**Baseline telemetry is RAW rev/s** (``rps_scale=1.0``) with the shipped
``time_offset``/``time_dilation`` applied, so a measured ``b_r`` is directly
"what rotor r's raw telemetry is short by" and is comparable with the shipped
global correction (+0.671 rev/s @ 80 for FLY124, +0.552 for FLY125).  The
dilation fix means the per-window residual global lag is already ~0 (validated
at 3 ms RMS), so tau = 0 is the right origin for the lag scans.

Cluster invocation (CPU-only)::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 4h -- \\
        python scripts/michaels_calib/run_perrotor.py --jobs 16

Layout mirrors ``run_sweep.py``: ``results/michaels_perrotor/{manifest,summary}.json``
plus one restartable ``raw/<window>__<stage>_r<n>.json`` per unit of work.
"""

from __future__ import annotations

import os

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

import calib as C  # noqa: E402
import windows as W  # noqa: E402

from data_processing.vk_tracking import vk_envelopes, vk_reconstruct  # noqa: E402

DEFAULT_RESULTS = REPO / "results" / "michaels_perrotor"
DEFAULT_CACHE = REPO / ".cache" / "michaels_perrotor"

#: frames of 0.032 s; the audio-rate block size the per-frame residual uses.
BLOCK = int(round(W.FRAME_S * W.SR))


# ────────────────────────────────────────────────── per-frame referee
def frame_residual(win: W.Window, traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(num2, den2)`` per 0.032 s frame block: residual and signal energy.

    ``sqrt(num2.sum() / den2.sum())`` reproduces :func:`calib.recon_ratio`
    exactly (up to the trailing partial block, which both drop), so any frame
    subset can be re-minimised after the fact.
    """
    n_t = win.audio.shape[-1]
    t_aud = np.arange(n_t) / W.SR
    r_aud = np.stack([np.interp(t_aud, win.ft, row) for row in traj])
    env = vk_envelopes(win.audio, r_aud, C.RECON_CFG)
    recon = vk_reconstruct(env, n_samples=n_t)
    n_b = n_t // BLOCK
    d = (win.audio - recon)[:, : n_b * BLOCK].reshape(win.audio.shape[0], n_b, BLOCK)
    x = win.audio[:, : n_b * BLOCK].reshape(win.audio.shape[0], n_b, BLOCK)
    return (d**2).sum(axis=(0, 2)), (x**2).sum(axis=(0, 2))


def pooled(num2: np.ndarray, den2: np.ndarray) -> float:
    return float(np.sqrt(num2.sum() / den2.sum()))


def parab(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, bool]:
    return C.parab_min(xs, ys)


# ────────────────────────────────────────────────────────── stages
def base(win: W.Window, stage: str, rotor: int) -> dict[str, Any]:
    return {
        "name": win.name,
        "rid": win.rid,
        "widx": win.widx,
        "stage": stage,
        "rotor": rotor,
        "regime": win.regime,
        "window": [win.start_s, win.end_s],
        "t_centre": win.t_centre,
        "gt_mean": np.round(win.r_meas.mean(1), 4).tolist(),
        "gt_std": np.round(win.r_meas.std(1), 4).tolist(),
        "mean_rps": round(float(win.r_meas[rotor].mean()), 4),
    }


def drdt(win: W.Window, rotor: int) -> np.ndarray:
    """d(rps)/dt of rotor ``rotor``, one value per 0.032 s frame block."""
    n_b = win.audio.shape[-1] // BLOCK
    g = np.gradient(win.r_meas[rotor], float(win.ft[1] - win.ft[0]))
    return g[:n_b]


def stage_off(win: W.Window, rotor: int, lo: float, hi: float, step: float) -> dict[str, Any]:
    """Per-rotor additive-offset scan with PER-FRAME residuals retained."""
    bs = C.grid(lo, hi, step)
    num: list[np.ndarray] = []
    den = np.zeros(1)
    for b in bs:
        t = win.r_meas.copy()
        t[rotor] = t[rotor] + float(b)
        n2, den = frame_residual(win, t)
        num.append(n2)
    num_a = np.stack(num)  # (n_b_grid, n_frames)
    ys = np.array([pooled(n, den) for n in num_a])
    xm, ym, edge = parab(bs, ys)
    res = base(win, "off", rotor)
    res |= {
        "b_grid": bs.tolist(),
        "recon": [round(float(v), 6) for v in ys],
        "best_b": round(float(xm), 5),
        "best_recon": round(float(ym), 6),
        "edge": bool(edge),
        "num2": np.round(num_a, 8).tolist(),
        "den2": np.round(den, 8).tolist(),
        "drdt": np.round(drdt(win, rotor), 5).tolist(),
        "rps_frames": np.round(win.r_meas[rotor][: num_a.shape[1]], 5).tolist(),
    }
    return res


def stage_lag(win: W.Window, rotor: int, lo: float, hi: float, step: float) -> dict[str, Any]:
    """Per-rotor time-shift scan (rotor ``rotor`` only) with per-frame residuals."""
    taus = C.grid(lo, hi, step)
    num: list[np.ndarray] = []
    den = np.zeros(1)
    for tau in taus:
        t = win.r_meas.copy()
        t[rotor] = C.shift(win.r_meas[rotor][None], win.ft, float(tau))[0]
        n2, den = frame_residual(win, t)
        num.append(n2)
    num_a = np.stack(num)
    ys = np.array([pooled(n, den) for n in num_a])
    xm, ym, edge = parab(taus, ys)
    dt = float(win.ft[1] - win.ft[0])
    res = base(win, "lag", rotor)
    res |= {
        "tau_grid": taus.tolist(),
        "recon": [round(float(v), 6) for v in ys],
        "best_tau_frames": round(float(xm), 5),
        "best_tau_ms": round(float(xm) * dt * 1000.0, 3),
        "best_recon": round(float(ym), 6),
        "edge": bool(edge),
        "recon_at_zero": round(float(ys[int(np.argmin(np.abs(taus)))]), 6),
    }
    return res


def stage_joint(
    win: W.Window,
    rotor: int,
    tlo: float,
    thi: float,
    tstep: float,
    blo: float,
    bhi: float,
    bstep: float,
) -> dict[str, Any]:
    """2D ``(tau_r, b_r)`` grid — which parameter carries the explanatory power."""
    taus = C.grid(tlo, thi, tstep)
    bs = C.grid(blo, bhi, bstep)
    grid = np.zeros((len(taus), len(bs)))
    for i, tau in enumerate(taus):
        shifted = C.shift(win.r_meas[rotor][None], win.ft, float(tau))[0]
        for j, b in enumerate(bs):
            t = win.r_meas.copy()
            t[rotor] = shifted + float(b)
            n2, d2 = frame_residual(win, t)
            grid[i, j] = pooled(n2, d2)
    i0, j0 = np.unravel_index(int(np.argmin(grid)), grid.shape)
    dt = float(win.ft[1] - win.ft[0])
    # profile likelihood: best over the other axis
    res = base(win, "joint", rotor)
    res |= {
        "tau_grid": taus.tolist(),
        "b_grid": bs.tolist(),
        "recon": np.round(grid, 6).tolist(),
        "argmin_tau": float(taus[i0]),
        "argmin_tau_ms": round(float(taus[i0]) * dt * 1000.0, 3),
        "argmin_b": float(bs[j0]),
        "min_recon": round(float(grid[i0, j0]), 6),
        # how much each axis buys ALONE, from the (tau=0, b=0) corner
        "recon_at_origin": round(float(grid[np.argmin(np.abs(taus)), np.argmin(np.abs(bs))]), 6),
        "best_b_at_tau0": float(bs[int(np.argmin(grid[np.argmin(np.abs(taus))]))]),
        "best_tau_at_b0": float(taus[int(np.argmin(grid[:, int(np.argmin(np.abs(bs)))]))]),
        "profile_over_tau": np.round(grid.min(axis=1), 6).tolist(),  # per tau, best b
        "profile_over_b": np.round(grid.min(axis=0), 6).tolist(),  # per b, best tau
    }
    return res


# ────────────────────────────────────────────────────────── driver
def unit_path(raw: Path, name: str, stage: str, rotor: int) -> Path:
    return raw / f"{name}__{stage}_r{rotor}.json"


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def run_unit(task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    raw = Path(task["raw_dir"])
    out = unit_path(raw, task["name"], task["stage"], task["rotor"])
    try:
        win = W.load_cached(Path(task["cache_dir"]), task["name"])
        k = task["stage"]
        if k == "off":
            r = stage_off(win, task["rotor"], task["lo"], task["hi"], task["step"])
        elif k == "lag":
            r = stage_lag(win, task["rotor"], task["lo"], task["hi"], task["step"])
        elif k == "joint":
            r = stage_joint(
                win,
                task["rotor"],
                task["tlo"],
                task["thi"],
                task["tstep"],
                task["blo"],
                task["bhi"],
                task["bstep"],
            )
        else:
            raise ValueError(k)
        r["elapsed_s"] = round(time.time() - t0, 1)
        write_json(out, r)
        return {"ok": True, "u": f"{task['name']}__{k}_r{task['rotor']}", "s": r["elapsed_s"]}
    except Exception as exc:
        return {
            "ok": False,
            "u": f"{task['name']}__{task['stage']}_r{task['rotor']}",
            "s": round(time.time() - t0, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def dispatch(tasks: list[dict[str, Any]], jobs: int) -> None:
    if not tasks:
        print("nothing to do", flush=True)
        return
    print(f"{len(tasks)} units on {jobs} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(run_unit, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            print(
                f"{i}/{len(tasks)} {'ok  ' if r['ok'] else 'FAIL'} {r['u']} "
                f"({r['s']}s, {time.time() - t0:.0f}s elapsed)",
                flush=True,
            )
            if not r["ok"]:
                print(r["traceback"], flush=True)


def build_raw_cache(cache_dir: Path, force: bool = False) -> dict[str, Any]:
    """Every window of both recordings at RAW rev/s + shipped timing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    man: dict[str, Any] = {"data_root": W.data_root()[1], "rps_scale": 1.0, "recordings": {}}
    for rid in ("FLY124", "FLY125"):
        rec = W.load_recording(rid, rps_scale=1.0)
        for w in rec["windows"]:
            p = W.cache_path(cache_dir, str(w["name"]))
            if p.exists() and not force:
                continue
            win = W.cut_window(rec, int(w["index"]))
            np.savez(
                p,
                audio=win.audio.astype(np.float32),
                ft=win.ft,
                r_meas=win.r_meas,
                meta=json.dumps(
                    {
                        "name": win.name,
                        "rid": win.rid,
                        "widx": win.widx,
                        "regime": win.regime,
                        "start_s": win.start_s,
                        "end_s": win.end_s,
                    }
                ),
            )
        man["recordings"][rid] = {
            "time_offset": rec["time_offset"],
            "time_dilation": rec["time_dilation"],
            "shipped_rps_scale": W.shipped_rps_scale(rid),
            "eval_span": rec["eval_span"],
            "windows": rec["windows"],
        }
        del rec
    return man


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--stages", default="off,lag,joint")
    ap.add_argument("--off-lo", type=float, default=-1.5)
    ap.add_argument("--off-hi", type=float, default=2.5)
    ap.add_argument("--off-step", type=float, default=0.25)
    ap.add_argument("--lag-lo", type=float, default=-3.0)
    ap.add_argument("--lag-hi", type=float, default=3.0)
    ap.add_argument("--lag-step", type=float, default=0.5)
    ap.add_argument("--joint-tau", default="-2,2,1.0", help="lo,hi,step in frames")
    ap.add_argument("--joint-b", default="-0.5,1.5,0.5", help="lo,hi,step in rev/s")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    results = Path(args.results)
    raw = results / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    jobs = args.jobs or len(os.sched_getaffinity(0))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    t_start = time.time()

    root, how = W.data_root()
    print(f"data root: {root}  ({how})", flush=True)
    man = build_raw_cache(cache, force=args.force)
    man["argv"] = sys.argv[1:]
    write_json(results / "manifest.json", man)
    cruise = {
        rid: [w for w in rec["windows"] if w["regime"] == "cruise"]
        for rid, rec in man["recordings"].items()
    }
    for rid, ws in cruise.items():
        print(
            f"{rid}: {len(man['recordings'][rid]['windows'])} windows, "
            f"{len(ws)} cruise {[w['index'] for w in ws]}  "
            f"(shipped rps_scale {man['recordings'][rid]['shipped_rps_scale']}, "
            f"baseline here = RAW)",
            flush=True,
        )
    if args.selftest:
        return

    tlo, thi, tstep = (float(x) for x in args.joint_tau.split(","))
    blo, bhi, bstep = (float(x) for x in args.joint_b.split(","))
    common = {"cache_dir": str(cache), "raw_dir": str(raw)}
    tasks: list[dict[str, Any]] = []
    for ws in cruise.values():
        for w in ws:
            for rot in range(W.N_ROTORS):
                for st in stages:
                    if not args.force and unit_path(raw, str(w["name"]), st, rot).exists():
                        continue
                    t = {**common, "name": str(w["name"]), "stage": st, "rotor": rot}
                    if st == "off":
                        t |= {"lo": args.off_lo, "hi": args.off_hi, "step": args.off_step}
                    elif st == "lag":
                        t |= {"lo": args.lag_lo, "hi": args.lag_hi, "step": args.lag_step}
                    else:
                        t |= {
                            "tlo": tlo,
                            "thi": thi,
                            "tstep": tstep,
                            "blo": blo,
                            "bhi": bhi,
                            "bstep": bstep,
                        }
                    tasks.append(t)
    # cheap stages first so partial results are still useful on a timeout
    order = {"off": 0, "lag": 1, "joint": 2}
    tasks.sort(key=lambda t: order[str(t["stage"])])
    dispatch(tasks, jobs)

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": how,
        "baseline": "RAW rev/s (rps_scale=1.0) + shipped time_offset/time_dilation",
        "shipped_rps_scale": {
            rid: man["recordings"][rid]["shipped_rps_scale"] for rid in man["recordings"]
        },
        "n_units": len(list(raw.glob("*.json"))),
        "wall_s": round(time.time() - t_start, 1),
    }
    write_json(results / "summary.json", summary)
    print(f"\nwrote {results / 'summary.json'} ({summary['wall_s']}s wall)", flush=True)


if __name__ == "__main__":
    main()
