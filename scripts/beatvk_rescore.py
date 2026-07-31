#!/usr/bin/env python3
"""Re-score the refinement chains on the frozen beat-VK protocol — CLUSTER driver.

Why this exists: Michael's telemetry was recalibrated (clock dilation + a
~0.7 % rev/s scale — `docs/experiments/rps-refine-precision.md` §§ WP13/WP14)
and `beatvk-valid-raw` was republished on top of the corrected
`michaels-frames`.  Every FLY124 number on the beat-VK scoreboard was scored
against the OLD labels and had to be re-measured.  This driver does that in
ONE CPU job:

1. **Prep caches.** Builds a per-window prep cache (16 kHz audio slice +
   telemetry on the window frame grid) for the CURRENT `beatvk-valid-raw` pin
   and — for the label-vs-estimator attribution — a second one for the
   pre-recalibration version (``--old-version``).  Both use
   ``beatvk_vk_arms.build_preps``, i.e. the production code path.
   DREGON windows are asserted bit-identical between the two builds: DREGON
   telemetry did NOT change, so any difference there is a red flag.

   MEASURED, and not obvious: FLY124's window *boundaries* (0/16/.../96 s) and
   window *count* (6) are unchanged, but its window **audio is not** — the
   recalibrated ``time_offset`` (−20.84 → −20.753813) changes how much of the
   WAV `michaels._load_michaels_data_raw` trims off the head, so every FLY124
   window now cuts audio **86.19 ms earlier** in the recording (measured by
   cross-correlation, r = 0.9997, a pure shift).  The re-score therefore
   changes the *predictions* too, not only the score they are graded against.
2. **The grid.**  ``scripts/rps_refine_lab.py`` (the production CLI, one
   subprocess per unit, restartable) for every chain × every protocol window.
   `refine_v2` / `refine_v3` run at ``--v2-rounds 1`` (the WP6/WP12 real-data
   default).  On the OLD prep cache only FLY124 is run (DREGON is invariant).
3. **The cross-window M3 arm.**  `refine_v3` is re-run on the FLY124 CRUISE
   windows with ``--m3-pool`` / ``--m3-ref`` derived from the OTHER cruise
   windows' phase-2 final track means (`rps_refine_lab.cross_window_pool`) —
   the WP12 arm that recovered w05 (3.380 -> 1.147).  Reported SEPARATELY:
   these options are off by default and stay off for the headline numbers.
4. **Summary.**  `summary.json` + a printed table: per window, per chain,
   pooled by (recording-group x regime), old vs new labels.

Run (see the module docstring of `beatvk_vk_arms.py` for the protocol)::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 6h \\
        -- python scripts/beatvk_rescore.py --jobs 8

Restartable: a unit whose ``raw/<unit>.json`` exists is skipped, so a
re-submission of the same command resumes.
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

import beatvk_vk_arms as arms  # noqa: E402
from rps_refine_lab import cross_window_pool  # noqa: E402

DEFAULT_OUT = Path("results/beatvk_rescore")
#: Prep caches live OUTSIDE results/ on purpose: they are ~240 MB of
#: regenerable intermediates and `omnirun pull` copies all of results/**.
DEFAULT_PREP_ROOT = Path(".cache/beatvk_rescore_prep")
CHAINS = ("baseline", "refine_v2", "refine_v3")
FLY124 = arms.FLY124_REC
#: pre-recalibration `beatvk-valid-raw` (old FLY124 labels), for attribution.
OLD_VERSION = "268c766052cb045ac1a6483ea41fa051a776ea5a5b0c07440afa412d7a3665f7"


def resolve_dregon_dir() -> str:
    """Geometry source for the DREGON mic weights, cluster-safe.

    ``$DATA_ROOT`` -> the repo's own ``data/`` -> the shared checkout's
    ``data/`` two levels up (omnirun runs cluster jobs in
    ``$PROJECT_ROOT/.trees/<sha12>``) -> the dload dataset (2.8 GiB; last
    resort, we only need ``coordinates.mat``).
    """
    cands = [
        os.environ.get("DATA_ROOT"),
        REPO / "data",
        REPO.parent.parent / "data",  # $PROJECT_ROOT/data from .trees/<sha12>
    ]
    for c in cands:
        if c and (Path(c) / "DREGON" / "coordinates.mat").exists():
            return str(Path(c) / "DREGON")
    return "dload:DREGON"


# ---------------------------------------------------------------------------
# prep caches


def build_prep_cache(out: Path, version: str | None, dregon_dir: str) -> dict[str, list[int]]:
    """Materialize `out/prep_cache` for every manifest window; return {rid: [widx]}."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = arms.load_manifest(out, None, version)
    jobs_windows = {
        rid: [int(w["index"]) for w in rec["windows"]]
        for rid, rec in manifest["recordings"].items()
    }
    arms.build_preps(out, jobs_windows, version, dregon_dir)
    print(
        f"[prep] {out}: {arms.DATASET}@{manifest['dataset_version'][:12]} — "
        f"{sum(len(v) for v in jobs_windows.values())} windows",
        flush=True,
    )
    return jobs_windows


def audio_shift_ms(new: np.ndarray, old: np.ndarray, max_lag_ms: float = 500.0) -> dict[str, float]:
    """Best integer-sample lag aligning `old` onto `new` (channel 0), + its r.

    A single number that says whether two builds of the same window hold the
    same acoustic content merely displaced (a trim change) or genuinely
    different audio.  Positive = the new window starts EARLIER in the
    recording, i.e. the content is delayed inside the window.
    """
    a = np.asarray(new, dtype=np.float64)
    b = np.asarray(old, dtype=np.float64)
    n = min(len(a), len(b))
    max_lag = int(max_lag_ms * 1e-3 * arms.SR)
    lo, hi = n // 3, min(n // 3 + 40000, n - max_lag)
    x = a[lo:hi] - a[lo:hi].mean()
    xn = float(np.sqrt((x * x).sum()))
    best_lag, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        y = b[lo - lag : hi - lag]
        y = y - y.mean()
        den = xn * float(np.sqrt((y * y).sum()))
        r = float((x * y).sum() / den) if den > 0 else 0.0
        if r > best_r:
            best_lag, best_r = lag, r
    return {"shift_ms": round(1e3 * best_lag / arms.SR, 3), "corr_at_shift": round(best_r, 4)}


def compare_preps(new_out: Path, old_out: Path, jobs_windows: dict[str, list[int]]) -> list[dict]:
    """Per-window old-vs-new label diff.  DREGON must be bit-identical."""
    rows: list[dict[str, Any]] = []
    for rid, widxs in jobs_windows.items():
        for widx in widxs:
            npz_new = arms.prep_path(new_out, rid, widx)
            npz_old = arms.prep_path(old_out, rid, widx)
            if not npz_old.exists():
                continue
            with np.load(npz_new) as a, np.load(npz_old) as b:
                same_audio = bool(np.array_equal(a["audio"], b["audio"]))
                shift: dict[str, float] = (
                    {} if same_audio else audio_shift_ms(a["audio"][0], b["audio"][0])
                )
                same_win = float(a["start_s"]) == float(b["start_s"]) and float(
                    a["end_s"]
                ) == float(b["end_s"])
                d = a["r_meas"] - b["r_meas"]
                rows.append(
                    {
                        "recording_id": rid,
                        "window": widx,
                        "start_s": float(a["start_s"]),
                        "end_s": float(a["end_s"]),
                        "window_boundaries_identical": same_win,
                        "audio_identical": same_audio,
                        **shift,
                        "labels_identical": bool(np.array_equal(a["r_meas"], b["r_meas"])),
                        "mean_rps_old": round(float(b["r_meas"].mean()), 4),
                        "mean_rps_new": round(float(a["r_meas"].mean()), 4),
                        "label_delta_mean": round(float(d.mean()), 4),
                        "label_delta_max_abs": round(float(np.abs(d).max()), 4),
                    }
                )
    for r in rows:
        if r["recording_id"] != FLY124 and not r["labels_identical"]:
            raise SystemExit(
                f"RED FLAG: DREGON labels changed for {r['recording_id']} w{r['window']:02d} "
                f"(mean delta {r['label_delta_mean']}) — DREGON telemetry was not recalibrated"
            )
    return rows


# ---------------------------------------------------------------------------
# the grid


def unit_name(tag: str, chain: str, rid: str, widx: int, arm: str = "") -> str:
    return f"{tag}__{chain}{arm}__{rid}__w{widx:02d}"


def run_unit(
    out: Path,
    prep_out: Path,
    tag: str,
    chain: str,
    rid: str,
    widx: int,
    v2_rounds: int,
    threads: int,
    extra: list[str] | None = None,
    arm: str = "",
) -> tuple[str, bool]:
    """One `rps_refine_lab.py` subprocess.  Skips a unit already on disk."""
    name = unit_name(tag, chain, rid, widx, arm)
    jpath = out / "raw" / f"{name}.json"
    if jpath.exists():
        return name, True
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "rps_refine_lab.py"),
        "--chain",
        chain,
        "--windows",
        f"real:{rid}:{widx}",
        "--beatvk-out",
        str(prep_out),
        "--out",
        str(jpath),
    ]
    if chain in ("refine_v2", "refine_v3"):
        cmd += ["--v2-rounds", str(v2_rounds)]
    cmd += extra or []
    env = {**os.environ}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(threads)
    log = out / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False)
    ok = proc.returncode == 0 and jpath.exists()
    print(
        f"[unit] {name}: {'ok' if ok else f'FAILED rc={proc.returncode}'} "
        f"({time.perf_counter() - tic:.0f}s)",
        flush=True,
    )
    return name, ok


def read_unit(out: Path, name: str) -> dict[str, Any] | None:
    jpath = out / "raw" / f"{name}.json"
    if not jpath.exists():
        return None
    doc = json.loads(jpath.read_text())
    ((wname, res),) = doc["windows"].items()
    return {
        "window_key": wname,
        "final": res["final_pooled_mae"],
        "final_tgrid": res["final_pooled_mae_tgrid"],
        "per_rotor_mae": [round(q["mae"], 3) for q in res["stages"][-1]["per_rotor"]],
        "final_means": res["meta"]["final_means"],
        "seed_bases": res["meta"]["seed_bases"],
        "regime": res["meta"].get("regime"),
        "wall_s": res["meta"]["wall_s"],
        "m3_pool": res["meta"].get("m3_pool") or [],
    }


# ---------------------------------------------------------------------------


def pooled(rows: list[dict[str, Any]]) -> float | None:
    vals = [r["final"] for r in rows if r is not None]
    return round(float(np.mean(vals)), 3) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--prep-root",
        default=str(DEFAULT_PREP_ROOT),
        help="where the (regenerable, ~240 MB) prep caches live; kept out of results/",
    )
    ap.add_argument("--jobs", type=int, default=8, help="parallel rps_refine_lab processes")
    ap.add_argument("--threads", type=int, default=2, help="BLAS threads per process")
    ap.add_argument("--chains", default=",".join(CHAINS))
    ap.add_argument("--v2-rounds", type=int, default=1, help="M1 corridor rounds (WP6 default)")
    ap.add_argument(
        "--old-version",
        default=OLD_VERSION,
        help="pre-recalibration beatvk-valid-raw version for the FLY124 "
        "label-vs-estimator attribution ('' disables that half)",
    )
    ap.add_argument(
        "--no-m3-pool-arm",
        action="store_true",
        help="skip the cross-window-seeded refine_v3 arm on FLY124 cruise",
    )
    ap.add_argument(
        "--prep-only",
        action="store_true",
        help="build both prep caches, print the old-vs-new window/label diff, exit "
        "(the cheap local dry run; the grid itself is cluster work)",
    )
    args = ap.parse_args()

    os.chdir(REPO)  # rps_refine_lab chdirs too; keep results/* repo-relative
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chains = [c for c in args.chains.split(",") if c]
    dregon_dir = resolve_dregon_dir()
    print(f"[beatvk_rescore] DREGON geometry from {dregon_dir}", flush=True)

    # -- 1. prep caches (new = current dload.lock pin; old = the frozen pin)
    prep_root = Path(args.prep_root)
    new_out = prep_root / "prep_new"
    jobs_windows = build_prep_cache(new_out, None, dregon_dir)
    new_version = json.loads((new_out / "manifest.json").read_text())["dataset_version"]
    old_out = prep_root / "prep_old"
    old_version = ""
    prep_diff: list[dict[str, Any]] = []
    if args.old_version:
        build_prep_cache(old_out, args.old_version, dregon_dir)
        old_version = json.loads((old_out / "manifest.json").read_text())["dataset_version"]
        prep_diff = compare_preps(new_out, old_out, jobs_windows)
        print("[prep] old-vs-new diff (DREGON must be identical):", flush=True)
        for r in prep_diff:
            shift_txt = (
                "audio identical"
                if r["audio_identical"]
                else f"audio shift {r['shift_ms']:+.3f} ms (r={r['corr_at_shift']:.4f})"
            )
            print(
                f"  {r['recording_id']:<32s} w{r['window']:02d} "
                f"[{r['start_s']:8.3f},{r['end_s']:8.3f}] "
                f"win_same={r['window_boundaries_identical']} {shift_txt}; "
                f"labels_same={r['labels_identical']} "
                f"mean {r['mean_rps_old']:7.3f} -> {r['mean_rps_new']:7.3f} "
                f"(d {r['label_delta_mean']:+.3f}, max|d| {r['label_delta_max_abs']:.3f})",
                flush=True,
            )

    if args.prep_only:
        (out / "prep_diff.json").write_text(
            json.dumps(
                {
                    "new_version": new_version,
                    "old_version": old_version,
                    "windows": prep_diff,
                },
                indent=1,
            )
        )
        print(f"\n[prep-only] wrote {out}/prep_diff.json")
        return

    # -- 2. the grid: (labels x chain x window)
    units: list[tuple] = []
    for chain in chains:
        for rid, widxs in jobs_windows.items():
            for widx in widxs:
                units.append((new_out, "new", chain, rid, widx, [], ""))
        if args.old_version:  # only FLY124 — DREGON labels are invariant
            for widx in jobs_windows[FLY124]:
                units.append((old_out, "old", chain, FLY124, widx, [], ""))
    print(f"[grid] {len(units)} units on {args.jobs} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [
            pool.submit(
                run_unit,
                out,
                po,
                tag,
                chain,
                rid,
                widx,
                args.v2_rounds,
                args.threads,
                extra,
                arm,
            )
            for po, tag, chain, rid, widx, extra, arm in units
        ]
        results = [f.result() for f in futs]
    failed = [n for n, ok in results if not ok]

    # -- 3. cross-window-seeded refine_v3 on FLY124 cruise (reported separately)
    pool_rows: dict[str, Any] = {}
    if not args.no_m3_pool_arm and "refine_v3" in chains:
        manifest = json.loads((new_out / "manifest.json").read_text())["recordings"][FLY124]
        cruise = [int(w["index"]) for w in manifest["windows"] if w["regime"] == "cruise"]
        means = {}
        for widx in cruise:
            u = read_unit(out, unit_name("new", "refine_v3", FLY124, widx))
            if u:
                means[f"w{widx:02d}"] = u["final_means"]
        jobs2 = []
        for widx in cruise:
            key = f"w{widx:02d}"
            if key not in means:
                continue
            pool_bases = cross_window_pool(means, exclude=key)
            ref = ";".join(",".join(f"{v:.3f}" for v in ms) for k, ms in means.items() if k != key)
            extra = ["--m3-pool", ",".join(f"{b:.3f}" for b in pool_bases), "--m3-ref", ref]
            pool_rows[key] = {"pool": pool_bases, "ref_windows": [k for k in means if k != key]}
            jobs2.append((new_out, "new", "refine_v3", FLY124, widx, extra, "_pool"))
        print(f"[grid] {len(jobs2)} cross-window-seeded units", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool_ex:
            futs = [
                pool_ex.submit(
                    run_unit, out, po, tag, ch, rid, w, args.v2_rounds, args.threads, ex, arm
                )
                for po, tag, ch, rid, w, ex, arm in jobs2
            ]
            failed += [n for n, ok in (f.result() for f in futs) if not ok]

    # -- 4. summary
    table: dict[str, Any] = {}
    for tag in ("new", "old"):
        for chain in chains:
            for arm in ("", "_pool"):
                for rid, widxs in jobs_windows.items():
                    for widx in widxs:
                        name = unit_name(tag, chain, rid, widx, arm)
                        u = read_unit(out, name)
                        if u:
                            table[name] = {
                                "labels": tag,
                                "chain": chain + arm,
                                "recording_id": rid,
                                "window": widx,
                                **u,
                            }

    def sel(tag: str, chain: str, rid_pred, regime: str | None) -> list[dict[str, Any]]:
        return [
            v
            for v in table.values()
            if v["labels"] == tag
            and v["chain"] == chain
            and rid_pred(v["recording_id"])
            and (regime is None or v["regime"] == regime)
        ]

    groups = {
        "dregon_cruise": (lambda r: r != FLY124, "cruise"),
        "fly124_cruise": (lambda r: r == FLY124, "cruise"),
        "fly124_warmup": (lambda r: r == FLY124, "warmup"),
        "fly124_all": (lambda r: r == FLY124, None),
    }
    agg: dict[str, Any] = {}
    for tag in ("new", "old"):
        for chain in [c for c in chains] + ["refine_v3_pool"]:
            for gname, (pred, regime) in groups.items():
                rows = sel(tag, chain, pred, regime)
                if rows:
                    agg[f"{tag}/{chain}/{gname}"] = {"pooled": pooled(rows), "n": len(rows)}

    summary = {
        "dataset": {"new_version": new_version, "old_version": old_version},
        "v2_rounds": args.v2_rounds,
        "chains": chains,
        "prep_diff": prep_diff,
        "m3_pool_arm": pool_rows,
        "windows": table,
        "pooled": agg,
        "failed_units": failed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\n===== per-window final PIT-MAE (v2_rounds={args.v2_rounds}) =====")
    hdr = f"{'recording':<32s} {'w':>3s} {'regime':<7s} {'labels':<6s}"
    for chain in chains:
        hdr += f" {chain:>10s}"
    hdr += f" {'v3_pool':>10s}"
    print(hdr)
    for rid, widxs in jobs_windows.items():
        for widx in widxs:
            for tag in ("new", "old"):
                cells, regime = [], "?"
                for chain in list(chains) + ["refine_v3_pool"]:
                    key = unit_name(
                        tag,
                        chain.replace("_pool", ""),
                        rid,
                        widx,
                        "_pool" if chain.endswith("_pool") else "",
                    )
                    v = table.get(key)
                    cells.append(f"{v['final']:10.3f}" if v else f"{'--':>10s}")
                    if v:
                        regime = v["regime"] or "?"
                if all(c.strip() == "--" for c in cells):
                    continue
                print(f"{rid:<32s} {widx:3d} {regime:<7s} {tag:<6s}" + "".join(cells))
    print("\n===== pooled =====")
    for k, v in sorted(agg.items()):
        print(f"  {k:<45s} {v['pooled']:7.3f}  (n={v['n']})")
    if failed:
        print(f"\nFAILED units ({len(failed)}): {failed}")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
