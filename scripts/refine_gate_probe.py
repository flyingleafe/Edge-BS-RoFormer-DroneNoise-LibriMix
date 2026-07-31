#!/usr/bin/env python3
"""Two diagnostics for the post-recalibration beat-VK protocol — CLUSTER driver.

Both questions this answers were raised by `docs/experiments/beat-vk.md`
§ "Protocol recalibrated and re-scored":

1. **The FLY124 w03 blind-seed flip.**  The 86.188 ms audio realignment makes
   arm R's residual re-scan accept a spurious 54.45 rev/s base instead of the
   comb-invisible 4th rotor near 82.7, costing +1.571 rev/s on the FLY124
   cruise pool.  Stage ``seed`` dumps, per window and per protocol build, the
   FULL residual scan (grid + scores + every peak with its robust z,
   completeness and admissibility flags) so any candidate acceptance rule can
   be evaluated offline, exactly, without re-running the 50 s scan per variant.

2. **The M2 steady-window regression.**  `refine_v2` loses 0.29-0.55 rev/s on
   every steady DREGON cruise window, and the per-stage table says it is all
   M2 (M1 is a no-op there).  Stage ``m2`` runs the production chain with
   ``--m2-dump``, so every per-rotor M2 proposal — its move, its sibling-
   residual ratio, its comb confidence before/after, and the replacement track
   itself — lands in an NPZ next to the truth.  Any accept/reject rule is then
   scorable offline at zero further compute.

Run (the grid is ~45 units of 50-190 s each)::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 3h \\
        -- python scripts/refine_gate_probe.py --jobs 8

Restartable: a unit whose output exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

DEFAULT_OUT = Path("results/refine_gate_probe")
DEFAULT_PREP_ROOT = Path(".cache/beatvk_rescore_prep")
BUILDS = ("new", "old")


def _windows(prep_out: Path) -> dict[str, list[int]]:
    manifest = json.loads((prep_out / "manifest.json").read_text())
    return {
        rid: [int(w["index"]) for w in rec["windows"]]
        for rid, rec in manifest["recordings"].items()
    }


# ---------------------------------------------------------------------------
# stage `seed` — one window's blind seeding, fully instrumented


def run_seed_unit(prep_out: Path, rid: str, widx: int, out: Path) -> None:
    """`blind_seed` on one window + every quantity a residual-acceptance rule
    could use, written to ``out`` (NPZ) and ``out.with_suffix('.json')``."""
    import beatvk_vk_arms as arms
    from vk_blind_sweep import SEED_CFG

    from data_processing.vk_blind_seeding import (
        blind_seed,
        completeness,
        scan_peaks,
        whitened_logmag,
    )

    prep, regime = arms.load_prep(prep_out, rid, widx, channels=8)
    tic = time.perf_counter()
    seed = blind_seed(prep.audio, float(arms.SR), 4, SEED_CFG, arms=frozenset({"K", "R"}))
    d = seed.diagnostics
    grid = np.asarray(d["grid"], dtype=np.float64)
    scores_flat = np.asarray(d["scores_flat"], dtype=np.float64)
    scores_res = np.asarray(d.get("residual_scores", np.zeros_like(grid)), dtype=np.float64)
    used = np.asarray(d.get("residual_used", []), dtype=np.float64)

    # Presence spectrum for the completeness of a residual candidate (arm C's
    # own statistic, recomputed here because blind_seed does not export it).
    white, bin_hz, _ = whitened_logmag(prep.audio, float(arms.SR), SEED_CFG)
    qvec = np.quantile(white, SEED_CFG.tooth_quantile, axis=1)

    med = float(np.median(scores_res))
    mad = 1.4826 * float(np.median(np.abs(scores_res - med)))
    pk = scan_peaks(grid, scores_res, SEED_CFG)
    peaks: list[dict[str, Any]] = []
    for i in pk:
        b = float(grid[i])
        z = (float(scores_res[i]) - med) / max(mad, 1e-12)
        frac, n_pres, n_teeth = completeness(qvec, bin_hz, b, SEED_CFG)
        far = float(np.min(np.abs(used - b))) if len(used) else np.inf
        gi = int(np.argmin(np.abs(grid - b)))
        peaks.append(
            {
                "base": round(b, 3),
                "score_res": round(float(scores_res[i]), 5),
                "z_res": round(float(z), 3),
                "score_flat": round(float(scores_flat[gi]), 5),
                "completeness": round(float(frac), 3),
                "teeth_present": int(n_pres),
                "teeth_total": int(n_teeth),
                "min_dist_to_used": round(float(far), 3),
                "ratio_to_max_used": round(b / float(used.max()), 4) if len(used) else None,
                "ratio_to_min_used": round(b / float(used.min()), 4) if len(used) else None,
            }
        )
    doc = {
        "recording_id": rid,
        "window": widx,
        "regime": regime,
        "prep": str(prep_out),
        "bases": [round(float(v), 3) for v in seed.bases],
        "primary": round(float(d["primary"]), 3),
        "update_gate": None if seed.update_gate is None else round(float(seed.update_gate), 4),
        "bw_hz": None if seed.bw_hz is None else round(float(seed.bw_hz), 4),
        "residual_used": [round(float(v), 3) for v in used],
        "residual_new": [round(float(v), 3) for v in d.get("residual_new", [])],
        "residual_new_z": [round(float(v), 3) for v in d.get("residual_new_z", [])],
        "residual_peaks": peaks,
        "candidates": [
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in c.items()
                if k in ("base", "score", "score_flat", "completeness", "accepted", "reason")
            }
            for c in seed.candidates
        ],
        "wall_s": round(time.perf_counter() - tic, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        grid=grid,
        scores_flat=scores_flat,
        scores_res=scores_res,
        used=used,
        bases=np.asarray(seed.bases, dtype=np.float64),
        qvec=qvec,
        bin_hz=np.float64(bin_hz),
    )
    out.with_suffix(".json").write_text(json.dumps(doc, indent=1))
    # Populate `rps_refine_lab.get_seed`'s cache too (same build, same window,
    # same cfg -> the identical 50 s call), so the `m2` stage and any later
    # chain run on this prep build start from the ladder instead.
    sc = prep_out / "seed_cache" / f"{rid}_w{widx:02d}.npz"
    sc.parent.mkdir(parents=True, exist_ok=True)
    tmp = sc.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(
        tmp,
        bases=np.asarray(seed.bases, dtype=np.float64),
        update_gate=np.float64(np.nan if seed.update_gate is None else seed.update_gate),
        bw_hz=np.float64(np.nan if seed.bw_hz is None else seed.bw_hz),
    )
    os.replace(tmp, sc)
    print(f"[seed] {rid} w{widx:02d} {prep_out.name}: {doc['bases']} ({doc['wall_s']:.0f}s)")


# ---------------------------------------------------------------------------
# stage `m2` — the production chain with the proposal dump on


def run_m2_unit(prep_out: Path, rid: str, widx: int, out: Path, threads: int) -> bool:
    name = out.stem
    jpath = out.with_suffix(".json")
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "rps_refine_lab.py"),
        "--chain",
        "refine_v2",
        "--v2-rounds",
        "1",
        "--windows",
        f"real:{rid}:{widx}",
        "--beatvk-out",
        str(prep_out),
        "--m2-dump",
        str(out),
        "--out",
        str(jpath),
    ]
    env = {**os.environ}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(threads)
    log = out.parent / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False)
    ok = proc.returncode == 0 and out.exists()
    print(
        f"[m2] {name}: {'ok' if ok else f'FAILED rc={proc.returncode}'} "
        f"({time.perf_counter() - tic:.0f}s)",
        flush=True,
    )
    return ok


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--prep-root", default=str(DEFAULT_PREP_ROOT))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stages", default="seed,m2")
    ap.add_argument("--unit", default=None, help="internal: run ONE seed unit in-process")
    args = ap.parse_args()

    os.chdir(REPO)
    out = Path(args.out)
    prep_root = Path(args.prep_root)

    if args.unit:  # internal worker: seed:<build>:<rid>:<widx>
        _, build, rid, widx = args.unit.split("|")
        run_seed_unit(
            prep_root / f"prep_{build}",
            rid,
            int(widx),
            out / "seed" / f"{build}__{rid}__w{int(widx):02d}.npz",
        )
        return

    stages = [s for s in args.stages.split(",") if s]
    # The prep caches live outside `results/`, so on a cluster worktree
    # (`$PROJECT_ROOT/.trees/<sha12>`, fresh per commit) they do not exist yet —
    # rebuild them with the SAME production code path `beatvk_rescore` uses.
    if not (prep_root / "prep_new" / "manifest.json").exists():
        import beatvk_rescore as rescore

        dregon_dir = rescore.resolve_dregon_dir()
        print(f"[prep] building caches under {prep_root} (DREGON geom {dregon_dir})", flush=True)
        rescore.build_prep_cache(prep_root / "prep_new", None, dregon_dir)
        rescore.build_prep_cache(prep_root / "prep_old", rescore.OLD_VERSION, dregon_dir)
    jobs_windows = _windows(prep_root / "prep_new")
    failed: list[str] = []

    if "seed" in stages:
        units = []
        for build in BUILDS:
            po = prep_root / f"prep_{build}"
            if not (po / "manifest.json").exists():
                continue
            for rid, widxs in _windows(po).items():
                for widx in widxs:
                    dst = out / "seed" / f"{build}__{rid}__w{widx:02d}.npz"
                    if dst.exists() and dst.with_suffix(".json").exists():
                        continue
                    units.append((build, rid, widx, dst))
        print(f"[grid] {len(units)} seed units on {args.jobs} workers", flush=True)

        def _seed(u: tuple) -> bool:
            build, rid, widx, dst = u
            env = {**os.environ}
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                env[k] = str(args.threads)
            log = out / "seed" / "logs" / f"{build}__{rid}__w{widx:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as fh:
                p = subprocess.run(
                    [
                        sys.executable,
                        str(REPO / "scripts" / "refine_gate_probe.py"),
                        "--out",
                        str(out),
                        "--prep-root",
                        str(prep_root),
                        "--unit",
                        f"seed|{build}|{rid}|{widx}",
                    ],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
            ok = p.returncode == 0 and dst.exists()
            print(f"[seed] {build} {rid} w{widx:02d}: {'ok' if ok else 'FAILED'}", flush=True)
            if not ok:
                failed.append(f"seed/{build}/{rid}/w{widx:02d}")
            return ok

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(_seed, units))

    if "m2" in stages:
        po = prep_root / "prep_new"
        units2 = [
            (po, rid, widx, out / "m2" / f"{rid}__w{widx:02d}.npz")
            for rid, widxs in jobs_windows.items()
            for widx in widxs
            if not (out / "m2" / f"{rid}__w{widx:02d}.npz").exists()
        ]
        print(f"[grid] {len(units2)} m2 units on {args.jobs} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            oks = list(
                pool.map(lambda u: run_m2_unit(u[0], u[1], u[2], u[3], args.threads), units2)
            )
        failed += [f"m2/{u[1]}/w{u[2]:02d}" for u, ok in zip(units2, oks) if not ok]

    print(f"\nfailed units: {failed or 'none'}")


if __name__ == "__main__":
    main()
