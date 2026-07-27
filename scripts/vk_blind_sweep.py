"""Blind-seeding-v2 arm sweep (design §7 sweep protocol).

Arms {baseline, T, C, N, K, T+C, T+C+N, T+C+N+K} of
``data_processing.vk_blind_seeding.blind_seed`` × base ladders
{plain, vit2dsp} × recordings {DREGON free-flight nosource/speech-low/
whitenoise-low room1, FLY124 cruise window}, blind init only.

Ladders (what runs on top of the seeds):
  * ``plain``   — CAPTURE (``vk_blind_annotation.CAPTURE_CFG``, annealed grow
    schedule) → REFINE (``REFINE_CFG``), the validated blind-condition ladder.
  * ``vit2dsp`` — the spatial-DP ladder (``vk_blind_annotation
    .vit2dsp_pipeline``: Viterbi pair-mean c(t) → spatial joint 2-rotor
    Viterbi with per-rotor 1/d² mic mixes → midband bw 6 → refine) — the
    prior best blind arm (DREGON pooled err_sm 0.688), §7's required
    composition target. Mic/rotor geometry: DREGON ``get_geometry``;
    FLY124 ``michaels.get_geometry``. Track→physical-rotor map is PIT vs
    measured truth (experiment-level, as in the validated ``run_vit2dsp``;
    per the corrected-geometry rerun the assignment only provides surface
    diversity, not accuracy).
Arm K swaps its auto ``update_gate`` into every VK stage of the chosen
ladder and its auto ``bw_hz`` into the plain ladder's capture phase.

Metrics vs telemetry truth (DREGON ``motors_measured``; FLY124 29 Hz ``rps``),
PIT-aligned: pooled + per-rotor err / err_sm, twin-resolution flag, capture
rate (fraction of rotors with err_sm < 2 rev/s — §7's FLY124 success bar),
plus per-stage err_sm and a stop-at-best stage (the prior blindvit2dsp
methodology reported stop-at-best; comparisons against its 0.688 bar must use
``best``, the honest blind number is ``refined``).
Reference bars: blindvit2dsp DREGON pooled err_sm 0.688; FLY124 from-peaks
blind err_sm 4.0 (3/4 rotors ~1 rev/s, twin lost). NB the first sweep's
whitenoise-low failure (22.06) exactly reproduced the old blind ladder's
22.063 — that number was the small-base scan bias now fixed by
``SeedConfig.scan_f_max``, not the white-noise harmonic-SNR floor; if the
capped scan still fails there, THEN it is the documented ~0 dB floor.

Data access (works locally and on remote CPU boxes):
  * DREGON — ``--dregon-dir`` (default ``data/DREGON``; pass ``dload:DREGON``
    to stream from R2 via ``data_processing.streams.resolve_source``).
  * FLY124 — ``--michaels-root`` (default ``data``; pass
    ``dload:recording_with_motor_speed`` — the loader accepts either the
    parent data root or the dataset directory itself).

Run:
  nice -n 10 python scripts/vk_blind_sweep.py                      # full sweep
  ... --quick                       # one recording (nosource), 10 s segment
  ... --synthetic-selftest          # 5 s synthetic 4-rotor mixture, no data
  ... --arms baseline,T+C+N+K --ladders plain    # subsets
  ... --recordings free-flight_whitenoise-low_room1,FLY124-cruise  # partial rerun
Startup prints ``[vk_blind_sweep] seeding module: <file> | scan_f_max=...`` —
grep it in job logs to verify the band-capped seeding module is the one
actually imported (the .venv's absolute-path editable install can otherwise
shadow a worktree's src/; guarded by the sys.path pin below).
Remote (CPU): omnirun submit --backend uni-cpu --gpus 0 --time 48h --yes -- \
  python scripts/vk_blind_sweep.py --dregon-dir dload:DREGON \
  --michaels-root dload:recording_with_motor_speed --jobs 8

Artifacts (``results/vk_blind_sweep/``): per-run ``<rid>__<ladder>__<arm>.npz``
(resumable), ``prep_cache/*.npz``, ``sweep_report.json`` + ``sweep_report.csv``.
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (process-level parallelism instead).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Pin THIS repo's src/ ahead of site-packages: the project is installed
# editable with an ABSOLUTE path into the checkout that owns .venv, so
# omnirun worktree jobs (which reuse the login checkout's .venv) would
# otherwise import data_processing from the login checkout at whatever
# commit it happens to have — exactly how sweep round 2 (python-e972ab) ran
# the new script with the OLD seeding module (stale scan, junk
# FLY124/whitenoise seeds). Module-level so spawned workers re-execute it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vk_blind_annotation import (  # noqa: E402
    CAPTURE_CFG,
    MIDBAND_CFGS,
    REFINE_CFG,
    min_pair_sep,
    pit_perm,
    vit2dsp_pipeline,
)
from vk_validation import (  # noqa: E402
    FRAME_HOP_S,
    Prepared,
    prepare_recording,
    smooth_frames,
)

from data_processing.vk_blind_seeding import SeedConfig, SeedResult, blind_seed  # noqa: E402
from data_processing.vk_tracking import vk_track  # noqa: E402

SR = 16000
OUT_DIR = Path("results/vk_blind_sweep")
PREP_CACHE_DIR = OUT_DIR / "prep_cache"
SEED_CFG = SeedConfig()
CAPTURE_RATE_THRESH = 2.0  # rev/s (§7: "FLY124: all 4 rotors < 2 rev/s")

ARM_SETS: dict[str, frozenset[str]] = {
    "baseline": frozenset(),
    "T": frozenset("T"),
    "C": frozenset("C"),
    "N": frozenset("N"),
    "K": frozenset("K"),
    "T+C": frozenset("TC"),
    "T+C+N": frozenset("TCN"),
    "T+C+N+K": frozenset("TCNK"),
    "R": frozenset("R"),
    "K+R": frozenset("KR"),
    # B = IAVKF-style bandwidth adaptation (VKConfig.bw_adapt) on the
    # midband/refine stages — orthogonality fixed-point per-track gain.
    "R+B": frozenset("RB"),
    "K+R+B": frozenset("KRB"),
}
LADDERS = ("plain", "vit2dsp")
DREGON_RIDS = [
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
]
FLY124_RID = "FLY124-cruise"
FLY124_WINDOW = (76.0, 92.0)  # seconds on the audio clock: mid-cruise, <=20 s
# (FLY124 timeline: idle ~31 rev/s t≈9-30, throttle-up, cruise ~74-92 rev/s
# t≈36-108 — blind annotation MUST be evaluated on cruise, not idle; the
# near-identical rotor pair makes one huge VK coupling group, so keep the
# window short. See project memory "blind-reannotation-dregon-vs-fly124".)

_PREP_FIELDS = ("tau", "seg_lo", "seg_hi", "audio", "ft", "r_init", "r_meas", "r_meas_sm", "edge")


# ---------------------------------------------------------------------------
# recording preparation (cached; DREGON mirrors vk_blind_annotation exactly)


def prepare_fly124(michaels_root: str, window: tuple[float, float] = FLY124_WINDOW) -> Prepared:
    """FLY124 cruise segment as a ``Prepared`` (truth = 29 Hz telemetry ``rps``).

    Loads via ``michaels.load_michaels_timeframe`` (manual audio↔telemetry
    alignment constants baked in, so ``tau = 0``). ``michaels_root`` may be
    the project data root (containing ``recording_with_motor_speed/``) or the
    dataset directory itself (e.g. a dload-materialized tree).
    """
    from data_processing.michaels import MICHAELS_FILES, load_michaels_timeframe
    from data_processing.streams import resolve_source

    root = resolve_source(michaels_root)
    wav_rel, csv_rel, t_off, t_dil = MICHAELS_FILES[0]  # recording_1 = FLY124
    wav, csv_p = root / wav_rel, root / csv_rel
    if not wav.exists():  # root points AT recording_with_motor_speed
        wav = root / Path(wav_rel).relative_to("recording_with_motor_speed")
        csv_p = root / Path(csv_rel).relative_to("recording_with_motor_speed")
    frame = load_michaels_timeframe(
        wav, csv_p, time_offset=t_off, time_dilation=t_dil, sr=SR, recording_id="FLY124"
    )
    audio = np.asarray(frame["audio"].data)  # (8, T)
    t0 = float(frame["audio"].tindex.t_start)
    rps = np.asarray(frame["rps"].data)  # (4, M)
    mt = np.asarray(frame["rps"].tindex.abs_stamps)

    lo, hi = window
    a0 = int(round((lo - t0) * SR))
    a1 = int(round((hi - t0) * SR))
    if not (0 <= a0 < a1 <= audio.shape[-1]):
        raise ValueError(
            f"FLY124 window {window} outside audio [{t0}, {t0 + audio.shape[-1] / SR}]"
        )
    seg = audio[:, a0:a1]
    ft = np.arange(0.0, (a1 - a0) / SR - FRAME_HOP_S / 2, FRAME_HOP_S)
    r_meas = np.stack([np.interp(ft + lo, mt, rps[i]) for i in range(4)])
    edge = (ft > 0.5) & (ft < ft[-1] - 0.5)
    return Prepared(
        rid=FLY124_RID,
        tau=0.0,
        seg_lo=lo,
        seg_hi=hi,
        audio=seg,
        ft=ft,
        r_init=r_meas.copy(),
        r_meas=r_meas,
        r_meas_sm=smooth_frames(r_meas),
        edge=edge,
    )


def get_prep(rid: str, opts: argparse.Namespace) -> Prepared:
    """Cached recording prep (NPZ keyed by rid + segment length)."""
    tag = f"{rid}@{opts.seg_len:g}s" if rid != FLY124_RID else rid
    path = PREP_CACHE_DIR / f"{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            arrs = {k: z[k] for k in _PREP_FIELDS}
        return Prepared(
            rid=rid,
            tau=float(arrs["tau"]),
            seg_lo=float(arrs["seg_lo"]),
            seg_hi=float(arrs["seg_hi"]),
            audio=arrs["audio"],
            ft=arrs["ft"],
            r_init=arrs["r_init"],
            r_meas=arrs["r_meas"],
            r_meas_sm=arrs["r_meas_sm"],
            edge=arrs["edge"].astype(bool),
        )
    if rid == FLY124_RID:
        prep = prepare_fly124(opts.michaels_root)
    else:
        prep = prepare_recording(rid, seg_len=opts.seg_len, dregon_dir=opts.dregon_dir)
    PREP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, allow_pickle=False, **{k: np.asarray(getattr(prep, k)) for k in _PREP_FIELDS})
    return prep


# ---------------------------------------------------------------------------
# pipeline + metrics


def rotor_mic_weights(rid: str, opts: argparse.Namespace) -> np.ndarray:
    """``(n_mics, 4)`` per-rotor mic weights ∝ 1/d², normalized per rotor."""
    if rid == FLY124_RID:
        from data_processing.michaels import get_geometry

        mic, rot = get_geometry()
    else:
        from data_processing.dregon import get_geometry
        from data_processing.streams import resolve_source

        mic, rot = get_geometry(Path(resolve_source(opts.dregon_dir)))
    d = np.linalg.norm(mic[:, None, :] - rot[None, :, :], axis=2)
    w = 1.0 / d**2
    return w / w.sum(axis=0, keepdims=True)


def run_pipeline(
    prep: Prepared,
    arms: frozenset[str],
    channels: int,
    ladder: str,
    weights: np.ndarray | None = None,
) -> tuple[SeedResult, np.ndarray, list[tuple[str, np.ndarray]], dict[str, Any]]:
    """seed → ladder; returns (seed, r0, [(stage label, traj)], timings)."""
    audio = prep.audio[:channels]
    tic = time.perf_counter()
    # "B" (bandwidth adaptation) only toggles the VK refine/midband configs below;
    # blind_seed validates arms against T/C/N/K/R and must not see it.
    seed = blind_seed(audio, float(SR), 4, SEED_CFG, arms=arms - {"B"})
    wall_seed = time.perf_counter() - tic

    r0 = np.repeat(seed.bases[:, None], len(prep.ft), axis=1)
    gate = seed.update_gate if ("K" in arms and seed.update_gate is not None) else None
    prep_run = replace(prep, audio=audio)

    if ladder == "vit2dsp":
        assert weights is not None, "vit2dsp ladder needs per-rotor mic weights"
        # Track -> physical rotor map: PIT vs measured truth (experiment-level,
        # exactly the validated run_vit2dsp methodology; the assignment only
        # provides surface diversity — see the corrected-geometry rerun).
        p = pit_perm(r0, prep.r_meas, prep.edge)
        phys_map = np.empty(4, dtype=int)
        for truth_row, track_row in enumerate(list(p)):
            phys_map[track_row] = truth_row
        mid_cfg = MIDBAND_CFGS[0] if gate is None else replace(MIDBAND_CFGS[0], update_gate=gate)
        ref_cfg = REFINE_CFG if gate is None else replace(REFINE_CFG, update_gate=gate)
        if "B" in arms:
            # Arm B: bandwidth adaptation on the tracking stages; capture
            # keeps its deliberately wide annealed band untouched.
            mid_cfg = replace(mid_cfg, bw_adapt=True)
            ref_cfg = replace(ref_cfg, bw_adapt=True)
        tic = time.perf_counter()
        stages, _, _, wall_scan, wall_vk = vit2dsp_pipeline(
            prep_run,
            r0,
            weights[:channels],
            phys_map,
            midband_cfg=mid_cfg,
            refine_cfg=ref_cfg,
            stage_guard=True,  # blind per-track revert of stage damage (the
            # r4 FLY124 finding: viterbi_c tracked all 4 rotors at pooled
            # 1.03, then the joint-DP re-captured the weak track onto the
            # 91 comb — a stage-robustness failure, not an information one)
        )
        timings = {"wall_seed_s": wall_seed, "wall_capture_s": wall_scan, "wall_refine_s": wall_vk}
        return seed, r0, stages[1:], timings  # drop the duplicate "init" stage

    if ladder != "plain":
        raise ValueError(f"unknown ladder {ladder!r} (expected one of {LADDERS})")
    cap_cfg, ref_cfg = CAPTURE_CFG, REFINE_CFG
    if gate is not None:
        # §7.4: auto update_gate everywhere; auto bw only for the CAPTURE
        # phase (its final annealed band must admit k * wander) — refine
        # keeps its narrow de-biasing band.
        cap_cfg = replace(cap_cfg, update_gate=gate, bw_hz=float(seed.bw_hz or 1.5))
        ref_cfg = replace(ref_cfg, update_gate=gate)
    if "B" in arms:
        ref_cfg = replace(ref_cfg, bw_adapt=True)
    tic = time.perf_counter()
    cap = vk_track(audio, r0, prep.ft, cap_cfg)
    wall_cap = time.perf_counter() - tic
    tic = time.perf_counter()
    ref = vk_track(audio, cap.r_refined, prep.ft, ref_cfg)
    wall_ref = time.perf_counter() - tic
    timings = {"wall_seed_s": wall_seed, "wall_capture_s": wall_cap, "wall_refine_s": wall_ref}
    return seed, r0, [("captured", cap.r_refined), ("refine", ref.r_refined)], timings


def traj_metrics(traj: np.ndarray, prep: Prepared) -> dict[str, Any]:
    """PIT-aligned metrics vs telemetry truth (§7: err_sm, twins, capture)."""
    p = pit_perm(traj, prep.r_meas, prep.edge)
    a = traj[list(p)]
    d = (a - prep.r_meas)[:, prep.edge]
    d_sm = (a - prep.r_meas_sm)[:, prep.edge]
    err_rotor = np.mean(np.abs(d), axis=1)
    err_rotor_sm = np.mean(np.abs(d_sm), axis=1)
    truth_sep = min_pair_sep(prep.r_meas, prep.edge)
    track_sep = min_pair_sep(a, prep.edge)
    return {
        "perm": [int(v) for v in p],
        "err": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "err_sm": float(np.mean(np.abs(d_sm))),
        "bias_sm": float(np.mean(d_sm)),
        "err_rotor": [float(v) for v in err_rotor],
        "err_rotor_sm": [float(v) for v in err_rotor_sm],
        "min_pair_sep_tracks": track_sep,
        "min_pair_sep_truth": truth_sep,
        "twins_resolved": bool(np.all(err_rotor < truth_sep) and track_sep > 0.5 * truth_sep),
        "capture_rate": float(np.mean(err_rotor_sm < CAPTURE_RATE_THRESH)),
        "_d": d,
        "_d_sm": d_sm,
    }


def job_tag(rid: str, ladder: str, arm_name: str, opts: argparse.Namespace) -> str:
    base = f"{rid}@{opts.seg_len:g}s" if rid != FLY124_RID else rid
    return f"{base}__{ladder}__{arm_name}"


def run_job(rid: str, ladder: str, arm_name: str, opts: argparse.Namespace) -> str:
    """Worker: one (recording, ladder, arm) run; NPZ-cached (resumable)."""
    path = OUT_DIR / f"{job_tag(rid, ladder, arm_name, opts)}.npz"
    if path.exists():
        return str(path)
    prep = get_prep(rid, opts)
    weights = rotor_mic_weights(rid, opts) if ladder == "vit2dsp" else None
    seed, r0, stages, timings = run_pipeline(
        prep, ARM_SETS[arm_name], opts.channels, ladder, weights
    )
    print(
        f"[{rid} | {ladder} | {arm_name}] seeds {np.round(seed.bases, 2)} "
        f"gate={seed.update_gate} bw={seed.bw_hz} "
        f"seed {timings['wall_seed_s']:.0f}s capture {timings['wall_capture_s']:.0f}s "
        f"refine {timings['wall_refine_s']:.0f}s",
        flush=True,
    )
    np.savez(
        path,
        allow_pickle=False,
        ft=prep.ft,
        edge=prep.edge,
        init=r0,
        measured=prep.r_meas,
        measured_sm=prep.r_meas_sm,
        stage_labels=np.array([lb for lb, _ in stages]),
        stage_snaps=np.stack([tr for _, tr in stages]),
        seed_bases=seed.bases,
        seed_update_gate=np.float64(np.nan if seed.update_gate is None else seed.update_gate),
        seed_bw_hz=np.float64(np.nan if seed.bw_hz is None else seed.bw_hz),
        seed_template=(
            np.zeros(0) if seed.template is None else np.asarray(seed.template, dtype=np.float64)
        ),
        seed_candidates=json.dumps(seed.candidates),
        tau=prep.tau,
        seg_bounds=np.array([prep.seg_lo, prep.seg_hi]),
        **{k: np.float64(v) for k, v in timings.items()},
    )
    return str(path)


def load_row(rid: str, ladder: str, arm_name: str, opts: argparse.Namespace) -> dict[str, Any]:
    """Recompute metrics for one finished run (main process, resumable)."""
    with np.load(OUT_DIR / f"{job_tag(rid, ladder, arm_name, opts)}.npz") as z:
        arrs = {k: z[k] for k in z.files}
    prep = get_prep(rid, opts)
    labels = [str(v) for v in arrs["stage_labels"]]
    snaps = arrs["stage_snaps"]
    stage_metrics = {lb: traj_metrics(snaps[i], prep) for i, lb in enumerate(labels)}
    best_stage = min(stage_metrics, key=lambda lb: stage_metrics[lb]["err_sm"])
    captured_lb = "vit2dsp" if "vit2dsp" in stage_metrics else "captured"
    return {
        "recording": rid,
        "ladder": ladder,
        "arm": arm_name,
        "stages_err_sm": {lb: round(m["err_sm"], 4) for lb, m in stage_metrics.items()},
        "best_stage": best_stage,
        "best": stage_metrics[best_stage],
        "captured": stage_metrics[captured_lb],
        "refined": stage_metrics[labels[-1]],
        "init": traj_metrics(arrs["init"], prep),
        "seed_bases": [round(float(v), 2) for v in arrs["seed_bases"]],
        "seed_update_gate": None
        if np.isnan(float(arrs["seed_update_gate"]))
        else round(float(arrs["seed_update_gate"]), 2),
        "seed_bw_hz": None
        if np.isnan(float(arrs["seed_bw_hz"]))
        else round(float(arrs["seed_bw_hz"]), 2),
        "seed_candidates": json.loads(str(arrs["seed_candidates"])),
        "wall_seed_s": round(float(arrs["wall_seed_s"]), 1),
        "wall_capture_s": round(float(arrs["wall_capture_s"]), 1),
        "wall_refine_s": round(float(arrs["wall_refine_s"]), 1),
    }


def pooled(rows: list[dict[str, Any]], stage: str = "refined") -> dict[str, float]:
    d = np.concatenate([r[stage]["_d"] for r in rows], axis=1)
    d_sm = np.concatenate([r[stage]["_d_sm"] for r in rows], axis=1)
    return {
        "err": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "err_sm": float(np.mean(np.abs(d_sm))),
        "bias_sm": float(np.mean(d_sm)),
        "capture_rate": float(np.mean([r[stage]["capture_rate"] for r in rows])),
        "n_twins_resolved": int(sum(r[stage]["twins_resolved"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# reporting


def _strip(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def write_report(
    rows: list[dict[str, Any]],
    ladder_names: list[str],
    arm_names: list[str],
    opts: argparse.Namespace,
    out_dir: Path,
) -> None:
    dregon_rows = [r for r in rows if r["recording"] != FLY124_RID]
    summary: dict[str, Any] = {
        "config": {
            "ladders": ladder_names,
            "arms": arm_names,
            "recordings": sorted({r["recording"] for r in rows}),
            "seg_len_s": opts.seg_len,
            "channels": opts.channels,
            "capture_rate_thresh": CAPTURE_RATE_THRESH,
            "seed_config": asdict(SEED_CFG),
            "capture_config": asdict(CAPTURE_CFG),
            "refine_config": asdict(REFINE_CFG),
            "reference": {
                "blindvit2dsp_dregon_err_sm": 0.688,  # stop-at-best -> compare "best"
                "fly124_blind_err_sm": 4.0,
                "note": (
                    "the first sweep's whitenoise-low 22.06 reproduced the OLD blind "
                    "ladder's 22.063 exactly — that was the small-base scan bias fixed "
                    "by scan_f_max, not the white-noise ~0 dB harmonic-SNR floor; if the "
                    "capped scan still fails there, it IS the floor"
                ),
            },
        },
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined", "best")},
                "init": _strip(r["init"]),
                "captured": _strip(r["captured"]),
                "refined": _strip(r["refined"]),
                "best": _strip(r["best"]),
            }
            for r in rows
        ],
        "pooled_dregon": {
            ladder: {
                arm: {
                    stage: pooled(sub, stage)
                    for stage in ("refined", "best")
                    if (
                        sub := [r for r in dregon_rows if r["arm"] == arm and r["ladder"] == ladder]
                    )
                }
                for arm in arm_names
                if any(r["arm"] == arm and r["ladder"] == ladder for r in dregon_rows)
            }
            for ladder in ladder_names
        },
        "fly124": {
            f"{r['ladder']}|{r['arm']}": {
                "refined": _strip(r["refined"]),
                "best": _strip(r["best"]),
            }
            for r in rows
            if r["recording"] == FLY124_RID
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "sweep_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "recording",
                "ladder",
                "arm",
                "err",
                "bias",
                "err_sm",
                "bias_sm",
                "err_rotor_sm_0",
                "err_rotor_sm_1",
                "err_rotor_sm_2",
                "err_rotor_sm_3",
                "twins_resolved",
                "capture_rate",
                "best_stage",
                "best_err_sm",
                "seed_bases",
                "seed_update_gate",
                "seed_bw_hz",
                "wall_total_s",
            ]
        )
        for r in rows:
            m = r["refined"]
            w.writerow(
                [
                    r["recording"],
                    r["ladder"],
                    r["arm"],
                    round(m["err"], 4),
                    round(m["bias"], 4),
                    round(m["err_sm"], 4),
                    round(m["bias_sm"], 4),
                    *[round(v, 4) for v in m["err_rotor_sm"]],
                    int(m["twins_resolved"]),
                    round(m["capture_rate"], 3),
                    r["best_stage"],
                    round(r["best"]["err_sm"], 4),
                    " ".join(f"{v:.2f}" for v in r["seed_bases"]),
                    r["seed_update_gate"],
                    r["seed_bw_hz"],
                    round(r["wall_seed_s"] + r["wall_capture_s"] + r["wall_refine_s"], 1),
                ]
            )
    print(f"report: {out_dir}/sweep_report.json + sweep_report.csv", flush=True)


def print_table(rows: list[dict[str, Any]]) -> None:
    hdr = (
        f"{'recording':<34} {'ladder':<8} {'arm':<9} {'err_sm':>7} {'bias_sm':>8} "
        f"{'twins':>5} {'capt':>5} {'best':>7}"
    )
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for r in rows:
        m = r["refined"]
        print(
            f"{r['recording']:<34} {r['ladder']:<8} {r['arm']:<9} {m['err_sm']:>7.3f} "
            f"{m['bias_sm']:>+8.3f} {str(m['twins_resolved']):>5} {m['capture_rate']:>5.2f} "
            f"{r['best']['err_sm']:>7.3f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# synthetic self-test (no real data; exercises the full pipeline + report)


def synthetic_selftest(opts: argparse.Namespace) -> None:
    """5 s synthetic 4-rotor mixture (twin pair + 2 rotors, common-mode
    wander) through seed → both ladders → metrics → report writing."""
    rng = np.random.default_rng(0)
    dur, fs = 5.0, float(SR)
    t = np.arange(int(dur * fs)) / fs
    wander = 0.4 * np.sin(2 * np.pi * 0.3 * t)
    bases = (74.0, 74.65, 83.0, 97.0)
    sig = np.zeros_like(t)
    r_true = []
    for b in bases:
        r = b + wander
        r_true.append(r)
        phase = 2 * np.pi * np.cumsum(r) / fs
        for k in range(1, 31):
            sig += (1.0 / np.sqrt(k)) * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
    noise = rng.standard_normal(len(t))
    noise *= np.sqrt(np.mean(sig**2) / np.mean(noise**2))  # 0 dB SNR
    sig2 = sig + 0.5 * rng.standard_normal(len(t)) * float(np.std(noise))
    audio = np.stack([sig + noise, sig2])  # 2 "mics" (vit2dsp needs C >= 1 each)

    ft = np.arange(0.0, dur - FRAME_HOP_S / 2, FRAME_HOP_S)
    r_meas = np.stack([np.interp(ft, t, r) for r in r_true])
    edge = (ft > 0.5) & (ft < ft[-1] - 0.5)
    prep = Prepared(
        rid="synthetic-selftest",
        tau=0.0,
        seg_lo=0.0,
        seg_hi=dur,
        audio=audio,
        ft=ft,
        r_init=r_meas.copy(),
        r_meas=r_meas,
        r_meas_sm=smooth_frames(r_meas),
        edge=edge,
    )
    arm_name = "T+C+N+K"
    weights = np.full((2, 4), 0.5)  # uniform mic mix: mechanics-only check
    rows = []
    tic = time.perf_counter()
    for ladder, n_ch in (("plain", 1), ("vit2dsp", 2)):
        seed, r0, stages, timings = run_pipeline(prep, ARM_SETS[arm_name], n_ch, ladder, weights)
        labels = [lb for lb, _ in stages]
        stage_metrics = {lb: traj_metrics(tr, prep) for lb, tr in stages}
        best_stage = min(stage_metrics, key=lambda lb: stage_metrics[lb]["err_sm"])
        rows.append(
            {
                "recording": prep.rid,
                "ladder": ladder,
                "arm": arm_name,
                "seed_bases": [round(float(v), 2) for v in seed.bases],
                "seed_update_gate": seed.update_gate,
                "seed_bw_hz": seed.bw_hz,
                "seed_candidates": seed.candidates,
                "stages_err_sm": {lb: round(m["err_sm"], 4) for lb, m in stage_metrics.items()},
                "best_stage": best_stage,
                "best": stage_metrics[best_stage],
                **{k: round(v, 1) for k, v in timings.items()},
                "init": traj_metrics(r0, prep),
                "captured": stage_metrics.get("vit2dsp", stage_metrics[labels[0]]),
                "refined": stage_metrics[labels[-1]],
            }
        )
    print_table(rows)
    write_report(rows, ["plain", "vit2dsp"], [arm_name], opts, OUT_DIR / "selftest")
    print(f"selftest wall {time.perf_counter() - tic:.0f}s", flush=True)
    for row in rows:
        m = row["refined"]
        print(
            f"selftest[{row['ladder']}]: seeds {row['seed_bases']} "
            f"gate {row['seed_update_gate']} bw {row['seed_bw_hz']} | refined err_sm "
            f"{m['err_sm']:.3f} capture_rate {m['capture_rate']:.2f} best {row['best_stage']}",
            flush=True,
        )
        assert len(row["seed_bases"]) == 4, "selftest: expected 4 seeds"
        assert np.isfinite(m["err_sm"]), "selftest: non-finite metrics"
        if row["ladder"] == "plain":
            assert m["capture_rate"] >= 0.75, f"selftest: capture rate {m['capture_rate']} < 0.75"
        else:  # uniform weights carry no spatial signature: mechanics-only bar
            assert row["best"]["capture_rate"] >= 0.5, (
                f"selftest[vit2dsp]: best capture rate {row['best']['capture_rate']} < 0.5"
            )
    print("selftest OK", flush=True)


# ---------------------------------------------------------------------------


def main() -> None:
    # Provenance banner (grep "seeding module:"): which vk_blind_seeding file
    # was ACTUALLY imported and its effective scan band cap — the round-2
    # sweep silently ran a stale module via the .venv's absolute-path
    # editable install; this line makes that regression impossible to miss.
    import data_processing.vk_blind_seeding as _seeding

    scan_f_max = getattr(SEED_CFG, "scan_f_max", None)
    print(
        f"[vk_blind_sweep] seeding module: {_seeding.__file__} | scan_f_max={scan_f_max}",
        flush=True,
    )
    if scan_f_max is None:
        raise SystemExit(
            "stale data_processing.vk_blind_seeding imported (no/None scan_f_max) — "
            "the band-capped scan is not in effect; check the sys.path pin above"
        )

    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--arms", default=",".join(ARM_SETS), help="comma-separated arm subset")
    ap.add_argument(
        "--ladders", default=",".join(LADDERS), help="comma-separated base-ladder subset"
    )
    ap.add_argument(
        "--recordings",
        default="",
        help="comma-separated recording subset (of DREGON rids + FLY124-cruise); "
        "default: all — use for partial reruns, e.g. "
        "--recordings free-flight_whitenoise-low_room1,FLY124-cruise",
    )
    ap.add_argument("--dregon-dir", default="data/DREGON", help="path or dload:DREGON")
    ap.add_argument(
        "--michaels-root", default="data", help="path or dload:recording_with_motor_speed"
    )
    ap.add_argument("--seg-len", type=float, default=25.0, help="DREGON segment length (s)")
    ap.add_argument("--channels", type=int, default=8, help="audio channels used (<=8)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel worker processes")
    ap.add_argument("--quick", action="store_true", help="one recording, 10 s segment")
    ap.add_argument(
        "--synthetic-selftest",
        action="store_true",
        help="run the whole pipeline on a 5 s synthetic 4-rotor mixture (no real data)",
    )
    opts = ap.parse_args()

    if opts.synthetic_selftest:
        synthetic_selftest(opts)
        return

    arm_names = [a for a in opts.arms.split(",") if a]
    unknown = [a for a in arm_names if a not in ARM_SETS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; valid: {list(ARM_SETS)}")
    ladder_names = [ld for ld in opts.ladders.split(",") if ld]
    bad = [ld for ld in ladder_names if ld not in LADDERS]
    if bad:
        raise SystemExit(f"unknown ladders {bad}; valid: {list(LADDERS)}")
    rids = [DREGON_RIDS[0]] if opts.quick else DREGON_RIDS + [FLY124_RID]
    if opts.recordings:
        wanted = [r for r in opts.recordings.split(",") if r]
        bad_r = [r for r in wanted if r not in DREGON_RIDS + [FLY124_RID]]
        if bad_r:
            raise SystemExit(f"unknown recordings {bad_r}; valid: {DREGON_RIDS + [FLY124_RID]}")
        rids = wanted
    if opts.quick:
        opts.seg_len = 10.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rid in rids:  # prep serially first (heavy load + resample, cached)
        get_prep(rid, opts)

    jobs = [
        (rid, ladder, arm)
        for rid in rids
        for ladder in ladder_names
        for arm in arm_names
        if not (OUT_DIR / f"{job_tag(rid, ladder, arm, opts)}.npz").exists()
    ]
    if jobs:
        print(f"running {len(jobs)} jobs on {opts.jobs} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=opts.jobs, mp_context=ctx) as pool:
            futs = [pool.submit(run_job, rid, ladder, arm, opts) for rid, ladder, arm in jobs]
            for f in futs:
                f.result()

    rows = [
        load_row(rid, ladder, arm, opts)
        for rid in rids
        for ladder in ladder_names
        for arm in arm_names
    ]
    print_table(rows)
    write_report(rows, ladder_names, arm_names, opts, OUT_DIR)


if __name__ == "__main__":
    main()
