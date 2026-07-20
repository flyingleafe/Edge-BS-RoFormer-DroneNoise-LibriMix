"""Blind-seeding-v2 arm sweep (design §7 sweep protocol).

Arms {baseline, T, C, N, K, T+C, T+C+N, T+C+N+K} of
``data_processing.vk_blind_seeding.blind_seed`` × recordings
{DREGON free-flight nosource/speech-low/whitenoise-low room1, FLY124 cruise
window}, blind init only. Per (recording, arm): seed → CAPTURE
(``vk_blind_annotation.CAPTURE_CFG``, annealed grow schedule) → REFINE
(``REFINE_CFG``), exactly the validated blind-condition ladder; arm K swaps
in its auto-derived ``update_gate`` (capture + refine) and capture ``bw_hz``.

Metrics vs telemetry truth (DREGON ``motors_measured``; FLY124 29 Hz ``rps``),
PIT-aligned: pooled + per-rotor err / err_sm, twin-resolution flag, capture
rate (fraction of rotors with err_sm < 2 rev/s — §7's FLY124 success bar).
Reference bars: blindvit2dsp DREGON pooled err_sm 0.688; FLY124 blind pooled
err_sm 4.0 with 3/4 rotors < ~1.3 (the unresolved twin dominates).

Data access (works locally and on remote CPU boxes):
  * DREGON — ``--dregon-dir`` (default ``data/DREGON``; pass ``dload:DREGON``
    to stream from R2 via ``data_processing.streams.resolve_source``).
  * FLY124 — ``--michaels-root`` (default ``data``; pass
    ``dload:recording_with_motor_speed`` — the loader accepts either the
    parent data root or the dataset directory itself).

Run:
  nice -n 10 .venv/bin/python scripts/vk_blind_sweep.py            # full sweep
  ... --quick                       # one recording (nosource), 10 s segment
  ... --synthetic-selftest          # 5 s synthetic 4-rotor mixture, no data
  ... --arms baseline,T+C+N+K       # subset of arms
Remote (CPU): omnirun submit --backend uni-cpu --gpus 0 --time 24h --yes -- \
  .venv/bin/python scripts/vk_blind_sweep.py --dregon-dir dload:DREGON \
  --michaels-root dload:recording_with_motor_speed --jobs 8

Artifacts (``results/vk_blind_sweep/``): per-run ``<rid>__<arm>.npz``
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

from vk_blind_annotation import (  # noqa: E402
    CAPTURE_CFG,
    REFINE_CFG,
    min_pair_sep,
    pit_perm,
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
}
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


def run_pipeline(
    prep: Prepared, arms: frozenset[str], channels: int
) -> tuple[SeedResult, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """seed → capture → refine; returns (seed, r0, captured, refined, timings)."""
    audio = prep.audio[:channels]
    tic = time.perf_counter()
    seed = blind_seed(audio, float(SR), 4, SEED_CFG, arms=arms)
    wall_seed = time.perf_counter() - tic

    r0 = np.repeat(seed.bases[:, None], len(prep.ft), axis=1)
    cap_cfg, ref_cfg = CAPTURE_CFG, REFINE_CFG
    if "K" in arms and seed.update_gate is not None:
        # §7.4: auto update_gate everywhere; auto bw only for the CAPTURE
        # phase (its final annealed band must admit k * wander) — refine
        # keeps its narrow de-biasing band.
        cap_cfg = replace(cap_cfg, update_gate=seed.update_gate, bw_hz=float(seed.bw_hz or 1.5))
        ref_cfg = replace(ref_cfg, update_gate=seed.update_gate)

    tic = time.perf_counter()
    cap = vk_track(audio, r0, prep.ft, cap_cfg)
    wall_cap = time.perf_counter() - tic
    tic = time.perf_counter()
    ref = vk_track(audio, cap.r_refined, prep.ft, ref_cfg)
    wall_ref = time.perf_counter() - tic
    timings = {"wall_seed_s": wall_seed, "wall_capture_s": wall_cap, "wall_refine_s": wall_ref}
    return seed, r0, cap.r_refined, ref.r_refined, timings


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


def run_job(rid: str, arm_name: str, opts: argparse.Namespace) -> str:
    """Worker: one (recording, arm) run; NPZ-cached (resumable)."""
    tag = f"{rid}@{opts.seg_len:g}s" if rid != FLY124_RID else rid
    path = OUT_DIR / f"{tag}__{arm_name}.npz"
    if path.exists():
        return str(path)
    prep = get_prep(rid, opts)
    seed, r0, captured, refined, timings = run_pipeline(prep, ARM_SETS[arm_name], opts.channels)
    print(
        f"[{rid} | {arm_name}] seeds {np.round(seed.bases, 2)} "
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
        captured=captured,
        refined=refined,
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


def load_row(rid: str, arm_name: str, opts: argparse.Namespace) -> dict[str, Any]:
    """Recompute metrics for one finished run (main process, resumable)."""
    tag = f"{rid}@{opts.seg_len:g}s" if rid != FLY124_RID else rid
    with np.load(OUT_DIR / f"{tag}__{arm_name}.npz") as z:
        arrs = {k: z[k] for k in z.files}
    prep = get_prep(rid, opts)
    return {
        "recording": rid,
        "arm": arm_name,
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
        "init": traj_metrics(arrs["init"], prep),
        "captured": traj_metrics(arrs["captured"], prep),
        "refined": traj_metrics(arrs["refined"], prep),
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
    rows: list[dict[str, Any]], arm_names: list[str], opts: argparse.Namespace, out_dir: Path
) -> None:
    dregon_rows = [r for r in rows if r["recording"] != FLY124_RID]
    summary: dict[str, Any] = {
        "config": {
            "arms": arm_names,
            "recordings": sorted({r["recording"] for r in rows}),
            "seg_len_s": opts.seg_len,
            "channels": opts.channels,
            "capture_rate_thresh": CAPTURE_RATE_THRESH,
            "seed_config": asdict(SEED_CFG),
            "capture_config": asdict(CAPTURE_CFG),
            "refine_config": asdict(REFINE_CFG),
            "reference": {"blindvit2dsp_dregon_err_sm": 0.688, "fly124_blind_err_sm": 4.0},
        },
        "rows": [
            {
                **{k: v for k, v in r.items() if k not in ("init", "captured", "refined")},
                "init": _strip(r["init"]),
                "captured": _strip(r["captured"]),
                "refined": _strip(r["refined"]),
            }
            for r in rows
        ],
        "pooled_dregon": {
            arm: pooled([r for r in dregon_rows if r["arm"] == arm])
            for arm in arm_names
            if any(r["arm"] == arm for r in dregon_rows)
        },
        "fly124": {r["arm"]: _strip(r["refined"]) for r in rows if r["recording"] == FLY124_RID},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "sweep_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "recording",
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
                    r["arm"],
                    round(m["err"], 4),
                    round(m["bias"], 4),
                    round(m["err_sm"], 4),
                    round(m["bias_sm"], 4),
                    *[round(v, 4) for v in m["err_rotor_sm"]],
                    int(m["twins_resolved"]),
                    round(m["capture_rate"], 3),
                    " ".join(f"{v:.2f}" for v in r["seed_bases"]),
                    r["seed_update_gate"],
                    r["seed_bw_hz"],
                    round(r["wall_seed_s"] + r["wall_capture_s"] + r["wall_refine_s"], 1),
                ]
            )
    print(f"report: {out_dir}/sweep_report.json + sweep_report.csv", flush=True)


def print_table(rows: list[dict[str, Any]]) -> None:
    hdr = f"{'recording':<34} {'arm':<9} {'err_sm':>7} {'bias_sm':>8} {'twins':>5} {'capt':>5}"
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for r in rows:
        m = r["refined"]
        print(
            f"{r['recording']:<34} {r['arm']:<9} {m['err_sm']:>7.3f} {m['bias_sm']:>+8.3f} "
            f"{str(m['twins_resolved']):>5} {m['capture_rate']:>5.2f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# synthetic self-test (no real data; exercises the full pipeline + report)


def synthetic_selftest(opts: argparse.Namespace) -> None:
    """5 s synthetic 4-rotor mixture (twin pair + 2 rotors, common-mode
    wander) through seed → capture → refine → metrics → report writing."""
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
    audio = sig + noise

    ft = np.arange(0.0, dur - FRAME_HOP_S / 2, FRAME_HOP_S)
    r_meas = np.stack([np.interp(ft, t, r) for r in r_true])
    edge = (ft > 0.5) & (ft < ft[-1] - 0.5)
    prep = Prepared(
        rid="synthetic-selftest",
        tau=0.0,
        seg_lo=0.0,
        seg_hi=dur,
        audio=audio[None, :],
        ft=ft,
        r_init=r_meas.copy(),
        r_meas=r_meas,
        r_meas_sm=smooth_frames(r_meas),
        edge=edge,
    )
    arm_name = "T+C+N+K"
    tic = time.perf_counter()
    seed, r0, captured, refined, timings = run_pipeline(prep, ARM_SETS[arm_name], channels=1)
    row = {
        "recording": prep.rid,
        "arm": arm_name,
        "seed_bases": [round(float(v), 2) for v in seed.bases],
        "seed_update_gate": seed.update_gate,
        "seed_bw_hz": seed.bw_hz,
        "seed_candidates": seed.candidates,
        **{k: round(v, 1) for k, v in timings.items()},
        "init": traj_metrics(r0, prep),
        "captured": traj_metrics(captured, prep),
        "refined": traj_metrics(refined, prep),
    }
    print_table([row])
    write_report([row], [arm_name], opts, OUT_DIR / "selftest")
    m = row["refined"]
    print(
        f"selftest: seeds {row['seed_bases']} gate {seed.update_gate} bw {seed.bw_hz} | "
        f"refined err_sm {m['err_sm']:.3f} capture_rate {m['capture_rate']:.2f} "
        f"({time.perf_counter() - tic:.0f}s)",
        flush=True,
    )
    assert len(seed.bases) == 4, "selftest: expected 4 seeds"
    assert np.isfinite(m["err_sm"]), "selftest: non-finite metrics"
    assert m["capture_rate"] >= 0.75, f"selftest: capture rate {m['capture_rate']} < 0.75"
    print("selftest OK", flush=True)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--arms", default=",".join(ARM_SETS), help="comma-separated arm subset")
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
    rids = [DREGON_RIDS[0]] if opts.quick else DREGON_RIDS + [FLY124_RID]
    if opts.quick:
        opts.seg_len = 10.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rid in rids:  # prep serially first (heavy load + resample, cached)
        get_prep(rid, opts)

    jobs = [
        (rid, arm)
        for rid in rids
        for arm in arm_names
        if not (
            OUT_DIR / f"{rid if rid == FLY124_RID else f'{rid}@{opts.seg_len:g}s'}__{arm}.npz"
        ).exists()
    ]
    if jobs:
        print(f"running {len(jobs)} jobs on {opts.jobs} workers", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=opts.jobs, mp_context=ctx) as pool:
            futs = [pool.submit(run_job, rid, arm, opts) for rid, arm in jobs]
            for f in futs:
                f.result()

    rows = [load_row(rid, arm, opts) for rid in rids for arm in arm_names]
    print_table(rows)
    write_report(rows, arm_names, opts, OUT_DIR)


if __name__ == "__main__":
    main()
