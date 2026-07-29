#!/usr/bin/env python3
"""VK-tracker arms on the frozen beat-VK protocol (``beatvk-valid-raw``).

Runs the validated vit2dsp VK ladder (``vk_blind_annotation.vit2dsp_pipeline``,
exactly as ``scripts/vk_blind_sweep.py`` composes it: geometry 1/d² mic
weights, PIT phys_map, ``stage_guard=True``, seed auto-knobs spliced when
present) on EVERY manifest 16 s window of every recording of the frozen
dataset published by ``scripts/publish_beatvk_valid.py``, one independent run
per (recording × window × arm), and assembles per-arm NPZ trajectory files in
the exact format ``scripts/beatvk_eval.py --pred npz:<dir>`` consumes.

Arms (``--arms``), differing ONLY in the seed/ladder-init:

* ``blind_baseline`` / ``blind_R`` / ``blind_KR`` — ``vk_blind_seeding
  .blind_seed`` with arm sets {} / {"R"} / {"K","R"}; ladder init = constant
  seed bases (the blind protocol's r0 shape). ``blind_KR`` splices the seed's
  auto ``update_gate`` into the ladder's midband + refine configs (the sweep's
  arm-K behaviour on the vit2dsp ladder).
* ``neural_traj`` / ``neural_bases`` — the ``--neural-model`` checkpoint's
  stitched chmean prediction on the window (``rps_predictor_vk_eval``
  conventions: sliding 251-frame windows, 32-frame hop, all mics
  permutation-aligned per window, overlap-aligned stitch, per-frame mean —
  the same forward as ``vk_blind_sweep.get_neural_traj``); init = the full
  trajectory (``traj``) or the constant window-median bases (``bases``).
* ``telem_init`` — raw telemetry linearly interpolated onto the window frame
  grid as the ladder init (the oracle-seed upper bound; init ONLY, the full
  ladder still runs on top).

Audio per window: the recording's native 44.1 kHz 8ch audio soxr-resampled to
the VK pipeline's 16 kHz (``frames.resample_audio_series`` — same resampler
as the scorer's ``model:`` path), then sliced to [start_s, end_s).

Outputs (``--out``, default ``results/beatvk_vk_arms/``):

* ``runs/<rid>__wNN__<arm>[__<model>].npz`` — one per job (resumable cache):
  window-relative ft, all ladder stage snapshots, final trajectory, seed
  bases/knobs, guard-revert log, wall times.
* ``<arm>/<recording_id>.npz`` — ``ft`` (absolute seconds from recording
  start) + ``rps`` (4, N): the windows' FINAL-stage (post-guard) trajectories
  concatenated in window order. ft is written ONLY where a window was run
  (NaN-free); the scorer edge-clamps outside ft and linearly interpolates
  across any interior gap, so score partial-coverage outputs only on their
  covered windows (full runs tile the eval span contiguously — no gaps).
* ``summary.json`` — per arm × recording × window: wall times (seed +
  ladder), seed bases/knobs, guard reverts, informational window PIT-MAE vs
  raw telemetry.
* ``manifest.json`` / ``prep_cache/`` / ``neural_cache/`` — dataset window
  manifest + per-window prep/neural caches (delete the out dir after a
  dataset re-pin).

Run::

    .venv/bin/python scripts/beatvk_vk_arms.py --list-windows
    .venv/bin/python scripts/beatvk_vk_arms.py \
        --arms blind_KR,telem_init --recordings FLY124 --windows 3 --jobs 2
    .venv/bin/python scripts/beatvk_vk_arms.py --jobs 8          # full run
    .venv/bin/python scripts/beatvk_eval.py \
        --pred npz:results/beatvk_vk_arms/blind_KR --tag vk_blind_KR
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (process-level parallelism instead) —
# same convention as vk_blind_sweep.py / vk_validation.py.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Pin THIS repo's src/ ahead of site-packages (the .venv's absolute-path
# editable install can otherwise shadow a worktree's src/ — see the
# vk_blind_sweep.py round-2 post-mortem). Module-level so spawned workers
# re-execute it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beatvk_eval import (  # noqa: E402
    DATASET,
    FRAME_S,
    HOP,
    SR,
    STITCH_SLIDE_FRAMES,
    STITCH_WIN_FRAMES,
    load_recordings,
)
from vk_blind_annotation import (  # noqa: E402
    MIDBAND_CFGS,
    REFINE_CFG,
    pit_perm,
    vit2dsp_pipeline,
)
from vk_blind_sweep import SEED_CFG  # noqa: E402  (identical seed config)
from vk_validation import Prepared, smooth_frames  # noqa: E402

from data_processing.vk_blind_seeding import SeedResult, blind_seed  # noqa: E402

DEFAULT_OUT = Path("results/beatvk_vk_arms")
DEFAULT_NEURAL_MODEL = "ckla_phaseonly_best"
N_ROTORS = 4
FLY124_REC = "FLY124"

#: blind arms -> blind_seed arm sets (subset of the sweep's ARM_SETS).
BLIND_ARM_SETS: dict[str, frozenset[str]] = {
    "blind_baseline": frozenset(),
    "blind_R": frozenset({"R"}),
    "blind_KR": frozenset({"K", "R"}),
}
NEURAL_ARMS = ("neural_traj", "neural_bases")
ALL_ARMS = (*BLIND_ARM_SETS, *NEURAL_ARMS, "telem_init")


# ---------------------------------------------------------------------------
# paths


def prep_dir(out: Path) -> Path:
    return out / "prep_cache"


def prep_path(out: Path, rid: str, widx: int) -> Path:
    return prep_dir(out) / f"{rid}__w{widx:02d}.npz"


def weights_path(out: Path, rid: str) -> Path:
    return prep_dir(out) / f"{rid}__weights.npz"


def neural_path(out: Path, rid: str, widx: int, model: str) -> Path:
    return out / "neural_cache" / f"{rid}__w{widx:02d}__{model}.npz"


def run_path(out: Path, rid: str, widx: int, arm: str, model: str) -> Path:
    tag = f"{rid}__w{widx:02d}__{arm}"
    if arm in NEURAL_ARMS:
        tag += f"__{model}"
    return out / "runs" / f"{tag}.npz"


# ---------------------------------------------------------------------------
# dataset manifest + per-window prep (main process; cached)


def load_manifest(out: Path, wanted: set[str] | None, version: str | None) -> dict[str, Any]:
    """Cached window manifest {rid: {windows, dataset_version}} (out/manifest.json)."""
    mpath = out / "manifest.json"
    if mpath.exists():
        cached = json.loads(mpath.read_text())
        rids = set(cached["recordings"])
        ver_ok = version is None or cached["dataset_version"].startswith(version)
        if ver_ok and (wanted is None or wanted <= rids) and wanted is not None:
            return cached
        if ver_ok and wanted is None and len(rids) >= 4:
            return cached
    recs = load_recordings(version, wanted, keep_audio=False)
    cached = json.loads(mpath.read_text()) if mpath.exists() else {"recordings": {}}
    cached["dataset_version"] = recs[0]["dataset_version"]
    for r in recs:
        cached["recordings"][r["recording_id"]] = {"windows": r["windows"]}
    out.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(cached, indent=2))
    return cached


def rotor_mic_weights(rid: str, dregon_dir: str) -> np.ndarray:
    """``(n_mics, 4)`` per-rotor mic weights ∝ 1/d², normalized per rotor —
    exactly ``vk_blind_sweep.rotor_mic_weights`` keyed by the frozen-protocol
    recording ids (FLY124 -> Michael's geometry, else DREGON)."""
    if rid == FLY124_REC:
        from data_processing.michaels import get_geometry

        mic, rot = get_geometry()
    else:
        from data_processing.dregon import get_geometry
        from data_processing.streams import resolve_source

        mic, rot = get_geometry(Path(resolve_source(dregon_dir)))
    d = np.linalg.norm(mic[:, None, :] - rot[None, :, :], axis=2)
    w = 1.0 / d**2
    return w / w.sum(axis=0, keepdims=True)


def build_preps(
    out: Path,
    jobs_windows: dict[str, list[int]],
    version: str | None,
    dregon_dir: str,
) -> None:
    """Materialize missing per-window prep NPZs (audio sliced + resampled to
    16 kHz, telemetry on the window frame grid). Streams the dataset only if
    something is missing."""
    missing = {
        rid: [w for w in ws if not prep_path(out, rid, w).exists()]
        for rid, ws in jobs_windows.items()
    }
    missing = {rid: ws for rid, ws in missing.items() if ws}
    need_weights = [rid for rid in jobs_windows if not weights_path(out, rid).exists()]
    prep_dir(out).mkdir(parents=True, exist_ok=True)
    for rid in need_weights:
        np.savez(weights_path(out, rid), weights=rotor_mic_weights(rid, dregon_dir))
    if not missing:
        return

    from data_processing.frames import resample_audio_series

    recs = load_recordings(version, set(missing), keep_audio=True)
    for rec in recs:
        rid = rec["recording_id"]
        widxs = set(missing[rid])
        windows = {int(w["index"]): w for w in rec["windows"]}
        tic = time.perf_counter()
        # The protocol resample (native 44.1 kHz -> the VK pipeline's 16 kHz,
        # librosa soxr_hq — same as beatvk_eval's model: path), once per
        # recording, then per-window slicing by sample index.
        audio16 = np.atleast_2d(
            np.asarray(resample_audio_series(rec["audio"], SR).data, dtype=np.float32)
        )
        ts, vals = rec["ts"], rec["vals"]
        for widx in sorted(widxs):
            w = windows[widx]
            start, end = float(w["start_s"]), float(w["end_s"])
            a0, a1 = int(round(start * SR)), int(round(end * SR))
            if not (0 <= a0 < a1 <= audio16.shape[-1]):
                raise ValueError(f"{rid} w{widx}: window [{start}, {end}] outside audio")
            seg = audio16[:, a0:a1]
            ft = np.arange(0.0, (a1 - a0) / SR - FRAME_S / 2, FRAME_S)
            r_meas = np.stack([np.interp(ft + start, ts, vals[i]) for i in range(N_ROTORS)])
            edge = (ft > 0.5) & (ft < ft[-1] - 0.5)
            np.savez(
                prep_path(out, rid, widx),
                allow_pickle=False,
                start_s=np.float64(start),
                end_s=np.float64(end),
                regime=np.str_(w["regime"]),
                audio=seg,
                ft=ft,
                r_meas=r_meas,
                r_meas_sm=smooth_frames(r_meas),
                edge=edge,
            )
        print(
            f"[prep] {rid}: {len(widxs)} windows resampled+cached "
            f"({time.perf_counter() - tic:.0f}s)",
            flush=True,
        )
        rec["audio"] = None


def load_prep(out: Path, rid: str, widx: int, channels: int) -> tuple[Prepared, str]:
    """Window prep NPZ -> ``Prepared`` (audio truncated to ``channels``) + regime."""
    with np.load(prep_path(out, rid, widx)) as z:
        start, end = float(z["start_s"]), float(z["end_s"])
        prep = Prepared(
            rid=f"{rid}__w{widx:02d}",
            tau=0.0,
            seg_lo=start,
            seg_hi=end,
            audio=z["audio"][:channels],
            ft=z["ft"],
            r_init=z["r_meas"].copy(),
            r_meas=z["r_meas"],
            r_meas_sm=z["r_meas_sm"],
            edge=z["edge"].astype(bool),
        )
        regime = str(z["regime"])
    return prep, regime


# ---------------------------------------------------------------------------
# neural seeds (main process, serial, model loaded once; cached NPZ so
# spawned workers never touch torch — the vk_blind_sweep pattern)


def compute_neural_seeds(
    out: Path, needed: list[tuple[str, int]], model_key: str, device: str | None, batch: int
) -> None:
    todo = [(rid, w) for rid, w in needed if not neural_path(out, rid, w, model_key).exists()]
    if not todo:
        return
    import rps_predictor_vk_eval as vkev
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    experiment, ckpt_uri, _ = vkev.MODELS[model_key]
    tic = time.perf_counter()
    model = vkev.load_model(experiment, ckpt_uri, dev)
    print(f"[neural] loaded {model_key} in {time.perf_counter() - tic:.0f}s ({dev})", flush=True)
    (out / "neural_cache").mkdir(parents=True, exist_ok=True)
    for rid, widx in todo:
        prep, _ = load_prep(out, rid, widx, channels=8)  # chmean uses ALL mics
        audio32 = np.ascontiguousarray(np.asarray(prep.audio, dtype=np.float32))
        f_total = audio32.shape[-1] // HOP + 1
        if f_total < STITCH_WIN_FRAMES:
            raise ValueError(f"{rid} w{widx}: {f_total} frames < the 8 s model window")
        tic = time.perf_counter()
        starts = vkev.window_starts(f_total, STITCH_WIN_FRAMES, STITCH_SLIDE_FRAMES)
        preds = vkev.predict_windows(
            model, audio32, starts, "chmean", dev, batch, STITCH_WIN_FRAMES
        )
        stack = vkev.stitch_stack(preds, starts, f_total, STITCH_WIN_FRAMES)
        traj_f = np.nanmean(stack, axis=0)  # (4, f_total) on the model frame grid
        times = np.arange(f_total) * FRAME_S
        traj = np.stack([np.interp(prep.ft, times, traj_f[i]) for i in range(N_ROTORS)])
        wall = time.perf_counter() - tic
        np.savez(neural_path(out, rid, widx, model_key), traj=traj, wall_s=np.float64(wall))
        print(
            f"[neural | {rid} w{widx:02d}] medians {np.round(np.median(traj, axis=1), 2)} "
            f"({wall:.0f}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# one (recording, window, arm) job — the vit2dsp ladder exactly as
# vk_blind_sweep.run_pipeline composes it (phys_map, gate splice, stage_guard)


def run_job(rid: str, widx: int, arm: str, cfg: dict[str, Any]) -> str:
    out = Path(cfg["out"])
    path = run_path(out, rid, widx, arm, cfg["neural_model"])
    if path.exists():
        return str(path)
    prep, regime = load_prep(out, rid, widx, cfg["channels"])
    with np.load(weights_path(out, rid)) as z:
        weights = z["weights"][: cfg["channels"]]

    arms = BLIND_ARM_SETS.get(arm, frozenset())
    if arm in BLIND_ARM_SETS:
        tic = time.perf_counter()
        seed = blind_seed(prep.audio, float(SR), N_ROTORS, SEED_CFG, arms=arms)
        wall_seed = time.perf_counter() - tic
        r0 = np.repeat(seed.bases[:, None], len(prep.ft), axis=1)
    else:
        if arm == "telem_init":
            traj, wall_seed = prep.r_meas.copy(), 0.0
        else:  # neural_traj / neural_bases
            with np.load(neural_path(out, rid, widx, cfg["neural_model"])) as z:
                traj, wall_seed = z["traj"], float(z["wall_s"])
        med = np.median(traj, axis=1)
        seed = SeedResult(
            bases=np.sort(med), candidates=[], template=None, update_gate=None, bw_hz=None
        )
        r0 = (
            traj.copy()
            if arm in ("neural_traj", "telem_init")
            else np.repeat(med[:, None], len(prep.ft), axis=1)
        )

    gate = seed.update_gate if ("K" in arms and seed.update_gate is not None) else None
    # Track -> physical rotor map: PIT vs measured truth (experiment-level,
    # exactly the validated run_vit2dsp / vk_blind_sweep methodology; per the
    # corrected-geometry rerun the assignment only provides surface diversity).
    p = pit_perm(r0, prep.r_meas, prep.edge)
    phys_map = np.empty(N_ROTORS, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        phys_map[track_row] = truth_row
    mid_cfg = MIDBAND_CFGS[0] if gate is None else replace(MIDBAND_CFGS[0], update_gate=gate)
    ref_cfg = REFINE_CFG if gate is None else replace(REFINE_CFG, update_gate=gate)
    stages, _, extras, wall_scan, wall_vk = vit2dsp_pipeline(
        prep, r0, weights, phys_map, midband_cfg=mid_cfg, refine_cfg=ref_cfg, stage_guard=True
    )
    stages = stages[1:]  # drop the duplicate "init" stage (sweep convention)
    final = stages[-1][1]
    guard = {
        k[len("guard_reverted_") :]: [int(v) for v in np.asarray(arr).ravel()]
        for k, arr in extras.items()
        if k.startswith("guard_reverted_")
    }
    print(
        f"[{rid} w{widx:02d} | {arm}] seeds {np.round(seed.bases, 2)} gate={seed.update_gate} "
        f"seed {wall_seed:.0f}s scan {wall_scan:.0f}s vk {wall_vk:.0f}s "
        f"guard={{{', '.join(f'{k}:{v}' for k, v in guard.items() if v)}}}",
        flush=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        allow_pickle=False,
        start_s=np.float64(prep.seg_lo),
        end_s=np.float64(prep.seg_hi),
        regime=np.str_(regime),
        ft=prep.ft,
        traj=final,
        stage_labels=np.array([lb for lb, _ in stages]),
        stage_snaps=np.stack([tr for _, tr in stages]),
        init=r0,
        seed_bases=seed.bases,
        seed_update_gate=np.float64(np.nan if seed.update_gate is None else seed.update_gate),
        seed_bw_hz=np.float64(np.nan if seed.bw_hz is None else seed.bw_hz),
        guard_reverts=np.str_(json.dumps(guard)),
        wall_seed_s=np.float64(wall_seed),
        wall_scan_s=np.float64(wall_scan),
        wall_vk_s=np.float64(wall_vk),
    )
    return str(path)


# ---------------------------------------------------------------------------
# assembly: per-arm scorer NPZs + summary.json


def window_pit_mae(traj: np.ndarray, r_meas: np.ndarray) -> float:
    """Informational window PIT-MAE vs raw telemetry (full window, no edge
    trim — closest to what the scorer computes, modulo its 0.032 s regrid)."""
    full = np.ones(traj.shape[-1], dtype=bool)
    a = traj[list(pit_perm(traj, r_meas, full))]
    return float(np.mean(np.abs(a - r_meas)))


def assemble(
    out: Path,
    arm_names: list[str],
    jobs_windows: dict[str, list[int]],
    model_key: str,
    dataset_version: str,
) -> None:
    summary: dict[str, Any] = {
        "dataset": {"name": DATASET, "version": dataset_version},
        "protocol": (
            "per manifest 16 s window: seed/init per arm -> vit2dsp ladder "
            "(vk_blind_annotation.vit2dsp_pipeline, geometry 1/d^2 mic weights, "
            "stage_guard=True, K-gate spliced when present) -> final post-guard "
            "stage; per-arm NPZ = windows concatenated on the recording timeline"
        ),
        "neural_model": model_key if any(a in NEURAL_ARMS for a in arm_names) else None,
        "arms": {},
    }
    for arm in arm_names:
        arm_dir = out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        summary["arms"][arm] = {}
        for rid, widxs in jobs_windows.items():
            fts, trajs, rows = [], [], {}
            for widx in sorted(widxs):
                rp = run_path(out, rid, widx, arm, model_key)
                if not rp.exists():
                    continue
                with np.load(rp) as z:
                    start = float(z["start_s"])
                    ft, traj = z["ft"], z["traj"]
                    rows[str(widx)] = {
                        "start_s": start,
                        "end_s": float(z["end_s"]),
                        "regime": str(z["regime"]),
                        "wall_seed_s": round(float(z["wall_seed_s"]), 1),
                        "wall_ladder_s": round(float(z["wall_scan_s"] + z["wall_vk_s"]), 1),
                        "wall_scan_s": round(float(z["wall_scan_s"]), 1),
                        "wall_vk_s": round(float(z["wall_vk_s"]), 1),
                        "seed_bases": [round(float(v), 2) for v in z["seed_bases"]],
                        "seed_update_gate": None
                        if np.isnan(float(z["seed_update_gate"]))
                        else round(float(z["seed_update_gate"]), 2),
                        "guard_reverts": json.loads(str(z["guard_reverts"])),
                        "pit_mae_raw_info": round(
                            window_pit_mae(traj, load_prep(out, rid, widx, 8)[0].r_meas), 3
                        ),
                    }
                fts.append(start + ft)
                trajs.append(traj)
            if not fts:
                continue
            ft_all = np.concatenate(fts)
            rps_all = np.concatenate(trajs, axis=1)
            if not np.all(np.diff(ft_all) > 0):
                raise RuntimeError(f"{arm}/{rid}: non-monotonic assembled ft")
            if not np.all(np.isfinite(rps_all)):
                raise RuntimeError(f"{arm}/{rid}: non-finite trajectory values")
            np.savez(arm_dir / f"{rid}.npz", ft=ft_all, rps=rps_all)
            summary["arms"][arm][rid] = {
                "n_windows": len(fts),
                "coverage_s": [float(ft_all[0]), float(ft_all[-1])],
                "windows": rows,
            }
            print(f"[assemble] {arm}/{rid}.npz: {len(fts)} windows, {ft_all.size} frames")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[assemble] wrote {out}/summary.json", flush=True)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--arms", default=",".join(ALL_ARMS), help=f"comma subset of {ALL_ARMS}")
    ap.add_argument(
        "--recordings",
        default="",
        help="comma subset of the frozen recordings (default: all 4)",
    )
    ap.add_argument(
        "--windows",
        default="",
        help="comma list of window indices to run (applied to every selected "
        "recording; default: all manifest windows)",
    )
    ap.add_argument("--neural-model", default=DEFAULT_NEURAL_MODEL)
    ap.add_argument("--jobs", type=int, default=4, help="parallel worker processes")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--dregon-dir", default="data/DREGON", help="path or dload:DREGON (geometry)")
    ap.add_argument("--channels", type=int, default=8, help="audio channels used (<=8)")
    ap.add_argument("--device", default=None, help="cuda|cpu for neural forwards (default: auto)")
    ap.add_argument("--batch", type=int, default=16, help="neural inference batch")
    ap.add_argument(
        "--list-windows", action="store_true", help="print the window manifest and exit"
    )
    opts = ap.parse_args()

    arm_names = [a for a in opts.arms.split(",") if a]
    unknown = [a for a in arm_names if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; valid: {list(ALL_ARMS)}")
    if opts.neural_model:
        import rps_predictor_vk_eval as vkev

        if opts.neural_model not in vkev.MODELS:
            raise SystemExit(f"unknown model {opts.neural_model!r}; known: {sorted(vkev.MODELS)}")
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {r for r in opts.recordings.split(",") if r} or None

    manifest = load_manifest(out, wanted, opts.dataset_version)
    version = manifest["dataset_version"]
    print(f"[beatvk_vk_arms] {DATASET}@{version[:12]}", flush=True)
    rec_windows = {
        rid: rec["windows"]
        for rid, rec in manifest["recordings"].items()
        if wanted is None or rid in wanted
    }
    if wanted:
        missing = wanted - set(rec_windows)
        if missing:
            raise SystemExit(f"unknown recordings {sorted(missing)}")
    if opts.list_windows:
        for rid, windows in rec_windows.items():
            print(f"\n{rid}:")
            for w in windows:
                print(
                    f"  w{int(w['index']):02d}  [{w['start_s']:8.2f}, {w['end_s']:8.2f}) "
                    f" {w['regime']:<7} mean_rps {w['mean_rps']:.1f}"
                )
        return

    widx_filter = {int(v) for v in opts.windows.split(",") if v} or None
    jobs_windows: dict[str, list[int]] = {}
    for rid, windows in rec_windows.items():
        idxs = [int(w["index"]) for w in windows]
        if widx_filter is not None:
            idxs = [i for i in idxs if i in widx_filter]
        if idxs:
            jobs_windows[rid] = idxs
    if not jobs_windows:
        raise SystemExit("no (recording, window) pairs selected")

    build_preps(out, jobs_windows, opts.dataset_version, opts.dregon_dir)
    if any(a in NEURAL_ARMS for a in arm_names):
        needed = [(rid, w) for rid, ws in jobs_windows.items() for w in ws]
        compute_neural_seeds(out, needed, opts.neural_model, opts.device, opts.batch)

    cfg = {
        "out": str(out),
        "channels": opts.channels,
        "neural_model": opts.neural_model,
    }
    jobs = [
        (rid, widx, arm)
        for rid, ws in jobs_windows.items()
        for widx in ws
        for arm in arm_names
        if not run_path(out, rid, widx, arm, opts.neural_model).exists()
    ]
    if jobs:
        print(f"running {len(jobs)} jobs on {opts.jobs} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=opts.jobs, mp_context=ctx) as pool:
            futs = [pool.submit(run_job, rid, widx, arm, cfg) for rid, widx, arm in jobs]
            for f in futs:
                f.result()

    assemble(out, arm_names, jobs_windows, opts.neural_model, version)
    print(
        "\nscore with e.g.:\n  .venv/bin/python scripts/beatvk_eval.py "
        f"--pred npz:{out}/<arm> --tag vk_<arm>",
        flush=True,
    )


if __name__ == "__main__":
    main()
