#!/usr/bin/env python3
"""VK-tracker arms on the frozen beat-VK protocol (``beatvk-valid-raw``).

Runs the validated vit2dsp VK ladder (``vk_blind_annotation.vit2dsp_pipeline``,
exactly as ``scripts/vk_blind_sweep.py`` composes it: geometry 1/d² mic
weights, PIT phys_map, ``stage_guard=True``, seed auto-knobs spliced when
present) on EVERY manifest 16 s window of every recording of the frozen
dataset published by ``data_processing.derivations (beatvk_valid generator)``, one independent run
per (recording × window × arm), and assembles per-arm NPZ trajectory files in
the exact format ``scripts/beatvk_eval.py --pred npz:<dir>`` consumes.

Arms (``--arms``), differing ONLY in the seed/ladder-init:

* ``blind_baseline`` / ``blind_R`` / ``blind_KR`` — ``vk_blind_seeding
  .blind_seed`` with arm sets {} / {"R"} / {"K","R"}; ladder init = constant
  seed bases (the blind protocol's r0 shape). ``blind_KR`` splices the seed's
  auto ``update_gate`` into the ladder's midband + refine configs (the sweep's
  arm-K behaviour on the vit2dsp ladder).
* ``blind_fullrange`` — blind_KR with a prepended BPF octave check on the
  seed bases (FLY124-warmup fix) and a COARSE FULL-RANGE Viterbi pass
  (12-120 rev/s, frame-rate, slope-tolerant, energy-timed takeoff bridge)
  that re-centres the seed bases onto a time-varying coarse c(t), so
  takeoff/warmup ramps inside a window are reachable by the ladder. The pass
  is ``tracking.coarse_init`` — read the block comment above it in
  ``src/tracking/pipelines.py`` for the mechanism and the measured numbers.
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
import hashlib  # noqa: E402
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
    FLY124_REC,
    FRAME_S,
    HOP,
    N_ROTORS,
    SR,
    STITCH_SLIDE_FRAMES,
    STITCH_WIN_FRAMES,
    load_recordings,
)
from vk_validation import Prepared, smooth_frames  # noqa: E402

from tracking.pipelines import (  # noqa: E402
    MIDBAND_CFGS,
    REFINE_CFG,
    SEED_CFG,
    CoarseConfig,
    coarse_init,
    vit2dsp_pipeline,
)
from tracking.protocols import BEATVK, iter_windows, pit_align, slice_window  # noqa: E402
from tracking.vk_blind_seeding import (  # noqa: E402
    SeedResult,
    blind_seed,
)

DEFAULT_OUT = Path("results/beatvk_vk_arms")
DEFAULT_NEURAL_MODEL = "ckla_phaseonly_best"

#: blind arms -> blind_seed arm sets (subset of the sweep's ARM_SETS).
BLIND_ARM_SETS: dict[str, frozenset[str]] = {
    "blind_baseline": frozenset(),
    "blind_R": frozenset({"R"}),
    "blind_KR": frozenset({"K", "R"}),
}
NEURAL_ARMS = ("neural_traj", "neural_bases")
FULLRANGE_ARM = "blind_fullrange"
#: blind_fullrange with a 2x longer coarse-DP STFT window — the mechanism and
#: the halved transition penalty are documented on :class:`tracking.CoarseConfig`.
FULLRANGE_2X_ARM = "blind_fullrange_2xwin"
COARSE_2X_CFG = CoarseConfig(nfft=4096, hop=1024, gamma=CoarseConfig().gamma / 2.0)
FULLRANGE_ARMS = (FULLRANGE_ARM, FULLRANGE_2X_ARM)
ALL_ARMS = (*BLIND_ARM_SETS, *FULLRANGE_ARMS, *NEURAL_ARMS, "telem_init")

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


def run_path(out: Path, rid: str, widx: int, arm: str, model: str, chan: str = "") -> Path:
    """Run-cache path. ``chan`` is the :func:`chan_tag` mic-subset suffix --
    a non-default subset changes the result, so it must not share a cache
    entry with the full-array run."""
    tag = f"{rid}__w{widx:02d}__{arm}"
    if arm in NEURAL_ARMS:
        tag += f"__{model}"
    return out / "runs" / f"{tag}{chan}.npz"


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
        from data_processing.sources.michaels import get_geometry

        mic, rot = get_geometry()
    else:
        from data_processing.sources.dregon import get_geometry
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
        specs = {s.index: s for s in iter_windows(BEATVK, {rid: {"windows": rec["windows"]}})}
        tic = time.perf_counter()
        # The protocol resample (native 44.1 kHz -> the VK pipeline's 16 kHz,
        # librosa soxr_hq — same as beatvk_eval's model: path), once per
        # recording; the per-window slicing is the protocol's own
        # (tracking.protocols.slice_window).
        audio16 = np.atleast_2d(
            np.asarray(resample_audio_series(rec["audio"], SR).data, dtype=np.float32)
        )
        ts, vals = rec["ts"], rec["vals"]
        for widx in sorted(widxs):
            spec = specs[widx]
            seg, ft, r_meas, edge = slice_window(audio16, SR, spec, ts, vals)
            assert r_meas is not None
            np.savez(
                prep_path(out, rid, widx),
                allow_pickle=False,
                start_s=np.float64(spec.start_s),
                end_s=np.float64(spec.end_s),
                regime=np.str_(spec.regime),
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


def channel_subset(
    rid: str, widx: int, n_total: int, channels: int, seed: int | None
) -> np.ndarray:
    """Channel indices for a window: first-``channels`` when ``seed`` is None,
    else a per-(seed, rid, widx) random subset (the mic-count ablation).

    The subset must be identical in every worker process, so the stream is
    seeded with a stable digest -- NOT ``hash()``, which is salted per process
    for strings.
    """
    if seed is None or channels >= n_total:
        return np.arange(min(channels, n_total))
    digest = hashlib.sha256(f"{seed}|{rid}|{widx}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    return np.sort(rng.permutation(n_total)[:channels])


def chan_tag(channels: int, seed: int | None) -> str:
    """Cache-key suffix for a non-default mic subset ('' for the full array)."""
    if seed is None and channels >= 8:
        return ""
    return f"__c{channels}" + (f"s{seed}" if seed is not None else "")


def load_prep(
    out: Path, rid: str, widx: int, channels: int, channel_seed: int | None = None
) -> tuple[Prepared, str]:
    """Window prep NPZ -> ``Prepared`` (audio truncated to ``channels``) + regime."""
    with np.load(prep_path(out, rid, widx)) as z:
        start, end = float(z["start_s"]), float(z["end_s"])
        sub = channel_subset(rid, widx, z["audio"].shape[0], channels, channel_seed)
        prep = Prepared(
            rid=f"{rid}__w{widx:02d}",
            tau=0.0,
            seg_lo=start,
            seg_hi=end,
            audio=z["audio"][sub],
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


def fullrange_seed(
    prep: Prepared, arm: str = FULLRANGE_ARM
) -> tuple[np.ndarray, SeedResult, dict[str, Any], float]:
    """The blind_fullrange init: blind_KR seed + the coarse full-range pass.

    Returns ``(r0, effective seed, coarse diagnostics, wall seconds)``. Both
    fullrange arms come through here — ``FULLRANGE_2X_ARM`` only changes the
    coarse-DP STFT resolution and its transition penalty
    (:class:`tracking.CoarseConfig`). The pass itself is
    :func:`tracking.coarse_init`; a halved seed drops the auto ``update_gate``,
    because the K calibration ran on the rejected 2x bases.
    """
    if arm not in FULLRANGE_ARMS:
        raise ValueError(f"{arm!r} is not a fullrange arm; valid: {list(FULLRANGE_ARMS)}")
    tic = time.perf_counter()
    seed = blind_seed(prep.audio, float(SR), N_ROTORS, SEED_CFG, arms=frozenset({"K", "R"}))
    cfg = COARSE_2X_CFG if arm == FULLRANGE_2X_ARM else CoarseConfig()
    r0, bases, halved, diag = coarse_init(prep, seed.bases, cfg=cfg, sr=float(SR))
    if halved:
        seed = SeedResult(
            bases=bases.copy(),
            candidates=seed.candidates,
            template=seed.template,
            update_gate=None,
            bw_hz=seed.bw_hz,
        )
    return r0, seed, diag, time.perf_counter() - tic


def run_ladder(
    prep: Prepared,
    r0: np.ndarray,
    weights: np.ndarray,
    gate: float | None = None,
    *,
    stage_guard: bool = True,
) -> tuple[list[tuple[str, np.ndarray]], Any, dict[str, Any], float, float]:
    """The vit2dsp ladder on one prepared window, as the sweep composes it.

    Track -> physical rotor map: PIT vs measured truth (experiment-level,
    exactly the validated run_vit2dsp / vk_blind_sweep methodology; per the
    corrected-geometry rerun the assignment only provides surface diversity).
    ``gate`` (the seed's auto ``update_gate``, arm K) is spliced into the
    midband + refine configs.
    """
    _, p = pit_align(r0, prep.r_meas, cost="mae", edge_mask=prep.edge)
    phys_map = np.empty(N_ROTORS, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        phys_map[track_row] = truth_row
    mid_cfg = MIDBAND_CFGS[0] if gate is None else replace(MIDBAND_CFGS[0], update_gate=gate)
    ref_cfg = REFINE_CFG if gate is None else replace(REFINE_CFG, update_gate=gate)
    return vit2dsp_pipeline(
        prep,
        r0,
        weights,
        phys_map,
        midband_cfg=mid_cfg,
        refine_cfg=ref_cfg,
        stage_guard=stage_guard,
    )


def run_job(rid: str, widx: int, arm: str, cfg: dict[str, Any]) -> str:
    out = Path(cfg["out"])
    path = run_path(
        out, rid, widx, arm, cfg["neural_model"], chan_tag(cfg["channels"], cfg.get("channel_seed"))
    )
    if path.exists():
        return str(path)
    prep, regime = load_prep(out, rid, widx, cfg["channels"], cfg.get("channel_seed"))
    with np.load(weights_path(out, rid)) as z:
        w_all = z["weights"]
        sub = channel_subset(rid, widx, w_all.shape[0], cfg["channels"], cfg.get("channel_seed"))
        weights = w_all[sub]

    coarse_diag: dict[str, Any] = {}
    if arm in FULLRANGE_ARMS:
        arms = frozenset({"K", "R"})
        r0, seed, coarse_diag, wall_seed = fullrange_seed(prep, arm)
    elif arm in BLIND_ARM_SETS:
        arms = BLIND_ARM_SETS[arm]
        tic = time.perf_counter()
        seed = blind_seed(prep.audio, float(SR), N_ROTORS, SEED_CFG, arms=arms)
        r0 = np.repeat(seed.bases[:, None], len(prep.ft), axis=1)
        wall_seed = time.perf_counter() - tic
    else:
        arms = frozenset()
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
    stages, _, extras, wall_scan, wall_vk = run_ladder(prep, r0, weights, gate)
    stages = stages[1:]  # drop the duplicate "init" stage (sweep convention)
    final = stages[-1][1]
    guard = {
        k[len("guard_reverted_") :]: [int(v) for v in np.asarray(arr).ravel()]
        for k, arr in extras.items()
        if k.startswith("guard_reverted_")
    }
    coarse_msg = ""
    if coarse_diag:
        cspan = np.round(np.percentile(coarse_diag["coarse_c"], [0, 50, 100]), 1)
        coarse_msg = (
            f" coarse[{coarse_diag['coarse_mode']} halved={coarse_diag['coarse_halved']} "
            f"bpf_ratio={coarse_diag['coarse_bpf_ratio']:.2f} "
            f"span={coarse_diag['coarse_span']:.1f} shift={coarse_diag['coarse_shift']:.1f} "
            f"{coarse_diag['coarse_bridge']} c min/med/max {cspan}]"
        )
    print(
        f"[{rid} w{widx:02d} | {arm}] seeds {np.round(seed.bases, 2)} gate={seed.update_gate} "
        f"seed {wall_seed:.0f}s scan {wall_scan:.0f}s vk {wall_vk:.0f}s "
        f"guard={{{', '.join(f'{k}:{v}' for k, v in guard.items() if v)}}}{coarse_msg}",
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
        **(
            {
                "coarse_c": coarse_diag["coarse_c"],
                "coarse_bpf_ratio": np.float64(coarse_diag["coarse_bpf_ratio"]),
                "coarse_halved": np.bool_(coarse_diag["coarse_halved"]),
                "coarse_grid": np.asarray(coarse_diag["coarse_grid"]),
                "coarse_bridge": np.str_(coarse_diag["coarse_bridge"]),
                "coarse_mode": np.str_(coarse_diag["coarse_mode"]),
                "coarse_span": np.float64(coarse_diag["coarse_span"]),
                "coarse_shift": np.float64(coarse_diag["coarse_shift"]),
            }
            if coarse_diag
            else {}
        ),
    )
    return str(path)


# ---------------------------------------------------------------------------
# assembly: per-arm scorer NPZs + summary.json


def window_pit_mae(traj: np.ndarray, r_meas: np.ndarray) -> float:
    """Informational window PIT-MAE vs raw telemetry (full window, no edge
    trim — closest to what the scorer computes, modulo its 0.032 s regrid)."""
    full = np.ones(traj.shape[-1], dtype=bool)
    a, _ = pit_align(traj, r_meas, cost="mae", edge_mask=full)
    return float(np.mean(np.abs(a - r_meas)))


def assemble(
    out: Path,
    arm_names: list[str],
    jobs_windows: dict[str, list[int]],
    model_key: str,
    dataset_version: str,
    chan: str = "",
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
        arm_dir = out / (arm + chan)
        arm_dir.mkdir(parents=True, exist_ok=True)
        summary["arms"][arm] = {}
        for rid, widxs in jobs_windows.items():
            fts, trajs, rows = [], [], {}
            for widx in sorted(widxs):
                rp = run_path(out, rid, widx, arm, model_key, chan)
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
    spath = out / f"summary{chan}.json"
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[assemble] wrote {spath}", flush=True)


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
    # Window specs come from the declarative protocol (tracking.protocols):
    # the manifest supplies the frozen per-window bounds, iter_windows the
    # canonical order + recording validation.
    try:
        specs = list(iter_windows(BEATVK, manifest["recordings"], recordings=wanted))
    except KeyError as exc:
        raise SystemExit(str(exc)) from None
    if opts.list_windows:
        for spec in specs:
            if spec.index == 0:
                print(f"\n{spec.recording_id}:")
            start = spec.start_s if spec.start_s is not None else float("nan")
            end = spec.end_s if spec.end_s is not None else float("nan")
            mean = spec.mean_rps if spec.mean_rps is not None else float("nan")
            print(
                f"  w{spec.index:02d}  [{start:8.2f}, {end:8.2f}) "
                f" {spec.regime or '?':<7} mean_rps {mean:.1f}"
            )
        return

    widx_filter = {int(v) for v in opts.windows.split(",") if v} or None
    jobs_windows: dict[str, list[int]] = {}
    for spec in specs:
        if widx_filter is None or spec.index in widx_filter:
            jobs_windows.setdefault(spec.recording_id, []).append(spec.index)
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
