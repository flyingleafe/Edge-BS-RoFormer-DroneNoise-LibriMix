"""Validate RPS trajectory refinement against annotated ground truth.

The refiner (``data_processing.rps_refinement``) nudges telemetry-derived rotor
trajectories toward the audio's harmonic comb. This script is the natural
experiment that tests whether that nudge moves *toward the truth*:

* **DREGON** ``free-flight_*_room1`` (5 recordings) carry BOTH
  ``motors_command`` (what training uses) and ``motors_measured`` (actual rotor
  speeds = ground truth). We refine the command and score command-vs-measured
  and refined-vs-measured error, plus the coherent harmonic LSQ residual at the
  command / refined / measured trajectories.
* **Michael's** FLY124/FLY125 have no independent ground truth. We refine the
  ``rps`` track and report only audio-fit (LSQ residual init vs refined),
  refinement delta magnitude, and comb confidence.

Everything runs CPU-only and per ~30 s segment to bound memory/runtime.

Artifacts (``results/rps_refinement/validation/``):
  * ``<dataset>_<recording>.npz`` — frame_times, command/init, measured (DREGON),
    refined, confidence (+centers), tau, per-rotor error rows.
  * ``summary.csv`` — one row per recording (DREGON) / per segment (Michael's),
    plus a pooled DREGON row.
  * ``preview_dregon.png`` / ``preview_michaels.png`` — trajectory overlay,
    error-vs-time, and spectrogram with refined harmonic tracks.

Run: ``uv run python scripts/rps_refinement_validation.py``
"""

from __future__ import annotations

import os

# Cap BLAS/torch threads BEFORE numpy/torch import. The work is parallelised
# at the process level (one worker per recording, see main), so each worker
# keeps a small thread pool; this also avoids OpenBLAS oversubscription when
# the box is shared with other heavy jobs (the first serial run of this script
# blew a 40 min budget purely on lstsq thrash).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import multiprocessing  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from itertools import permutations  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_processing.dregon import (  # noqa: E402
    clean_command_spikes,
    discover_recordings,
    get_geometry,
    load_timeframe,
)
from data_processing.michaels import load_michaels_timeframes  # noqa: E402
from data_processing.rps_refinement import (  # noqa: E402
    RefineConfig,
    compute_logmag,
    estimate_clock_offset,
    harmonic_lsq_residual,
    refine_trajectories,
)

SR = 16000
SEG_LEN_S = 30.0
DREGON_MIN_RPS = 30.0  # in-flight threshold (both motor tracks median > this)
MICHAELS_MIN_RPS = 45.0  # skip near-idle segments (project memory: FLY124)
LSQ_KMAX = 40
# LSQ residual is computed on this much audio (channel 0). Keep short:
# harmonic_lsq_residual materialises framed phasors (R, K, frames, window) —
# ~0.4 GB per second of audio at k_max=40 across its temporaries (measured) —
# and the residual ratio only needs a same-window comparison of trajectories.
LSQ_WIN_S = 5.0
OUT_DIR = Path("results/rps_refinement/validation")

# The 5 DREGON recordings that carry motors_measured (ground truth).
DREGON_TARGETS = [
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_speech-high_room1",
    "free-flight_whitenoise-low_room1",
    "free-flight_whitenoise-high_room1",
]
DREGON_PREVIEW = "free-flight_nosource_room1"  # cleanest harmonics for figures


@dataclass
class SegmentResult:
    """Refinement outputs for one ~30 s segment, on the global frame grid."""

    seg_lo: float
    frame_times: np.ndarray  # (N,) global seconds
    r_init: np.ndarray  # (R, N) tau-shifted init (command / rps)
    r_refined: np.ndarray  # (R, N)
    r_measured: np.ndarray | None  # (R, N) or None (Michael's)
    confidence: np.ndarray  # (R, W)
    conf_centers: np.ndarray  # (W,) global seconds


def segment_bounds(
    t_lo: float, t_hi: float, seg_len: float = SEG_LEN_S
) -> list[tuple[float, float]]:
    """Tile ``[t_lo, t_hi]`` into <=seg_len chunks; merge a tiny tail (<5 s)."""
    bounds: list[tuple[float, float]] = []
    s = t_lo
    while s < t_hi - 1e-3:
        e = min(s + seg_len, t_hi)
        bounds.append((s, e))
        s = e
    if len(bounds) >= 2 and (bounds[-1][1] - bounds[-1][0]) < 5.0:
        a = bounds[-2][0]
        b = bounds[-1][1]
        bounds = bounds[:-2] + [(a, b)]
    return bounds


def refine_segment(
    audio: np.ndarray,
    seg_lo: float,
    seg_hi: float,
    motor_times: np.ndarray,
    init_values: np.ndarray,
    cfg: RefineConfig,
    tau: float,
    measured_values: np.ndarray | None,
) -> SegmentResult:
    """Refine one segment; all telemetry evaluated at ``ft + tau`` (audio clock)."""
    a0 = int(round(seg_lo * SR))
    a1 = int(round(seg_hi * SR))
    seg = audio[:, a0:a1]
    spec = compute_logmag(seg, cfg)
    ft = spec.frame_times  # relative to segment start
    mt_rel = motor_times - seg_lo
    n_rotor = init_values.shape[0]
    r_init = np.stack([np.interp(ft + tau, mt_rel, init_values[i]) for i in range(n_rotor)])
    res = refine_trajectories(spec, r_init, cfg)
    r_meas = None
    if measured_values is not None:
        r_meas = np.stack([np.interp(ft + tau, mt_rel, measured_values[i]) for i in range(n_rotor)])
    return SegmentResult(
        seg_lo=seg_lo,
        frame_times=seg_lo + ft,
        r_init=r_init,
        r_refined=res.r_refined,
        r_measured=r_meas,
        confidence=res.confidence,
        conf_centers=seg_lo + res.conf_centers,
    )


def concat_segments(segs: list[SegmentResult]) -> dict[str, np.ndarray]:
    """Concatenate per-segment arrays; sort/dedupe frame times for monotonicity."""
    ft = np.concatenate([s.frame_times for s in segs])
    r_init = np.concatenate([s.r_init for s in segs], axis=1)
    r_ref = np.concatenate([s.r_refined for s in segs], axis=1)
    order = np.argsort(ft)
    ft = ft[order]
    r_init = r_init[:, order]
    r_ref = r_ref[:, order]
    keep = np.concatenate([[True], np.diff(ft) > 1e-9])
    out: dict[str, np.ndarray] = {
        "frame_times": ft[keep],
        "r_init": r_init[:, keep],
        "r_refined": r_ref[:, keep],
        "confidence": np.concatenate([s.confidence for s in segs], axis=1),
        "conf_centers": np.concatenate([s.conf_centers for s in segs]),
    }
    if segs[0].r_measured is not None:
        r_meas = np.concatenate([s.r_measured for s in segs], axis=1)  # type: ignore[misc]
        out["r_measured"] = r_meas[:, order][:, keep]
    return out


def pit_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Best-permutation mean |error| over rotors (separates swaps from error)."""
    r = pred.shape[0]
    cost = np.array([[np.mean(np.abs(pred[i] - gt[j])) for j in range(r)] for i in range(r)])
    best = min(sum(cost[i, p[i]] for i in range(r)) for p in permutations(range(r)))
    return float(best / r)


def dregon_recording_task(rid: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Refine one measured-carrying DREGON recording (worker process).

    Saves the per-recording NPZ and returns ``(summary_row, |cmd-meas|,
    |ref-meas|)`` with the error arrays (R, N) for pooling.
    """
    cfg = RefineConfig()
    dregon_dir = Path("data") / "DREGON"
    geom = get_geometry(dregon_dir)
    by_id = {s["recording_id"]: s for s in discover_recordings(dregon_dir)}

    print(f"[DREGON] {rid}", flush=True)
    frame = load_timeframe(by_id[rid], geometry=geom, target_sr=SR)
    audio = np.asarray(frame["audio"].data)
    t0 = float(frame["audio"].tindex.t_start)
    command = np.asarray(frame["motors_command"].data)
    measured = np.asarray(frame["motors_measured"].data)
    mt = np.asarray(frame["motors_command"].tindex.abs_stamps) - t0
    command_clean = clean_command_spikes(command)

    mask = (np.median(command, axis=0) > DREGON_MIN_RPS) & (
        np.median(measured, axis=0) > DREGON_MIN_RPS
    )
    idx = np.where(mask)[0]
    t_lo = float(mt[idx[0]]) + 0.2
    t_hi = float(mt[idx[-1]]) - 0.2

    # Stage A: clock offset from the (cleaned) command on the first segment.
    # Channel 0 only: tau is a global scalar, and the 201-candidate tau scan
    # materialises (C, K, taus, R, N) score tensors — 8 channels cost ~0.5 GB
    # in temporaries per worker for no accuracy gain.
    first_hi = min(t_lo + SEG_LEN_S, t_hi)
    spec0 = compute_logmag(audio[:1, int(t_lo * SR) : int(first_hi * SR)], cfg)
    tau, _, _ = estimate_clock_offset(spec0, mt - t_lo, command_clean, cfg)

    segs = []
    for s_lo, s_hi in segment_bounds(t_lo, t_hi):
        segs.append(refine_segment(audio, s_lo, s_hi, mt, command_clean, cfg, tau, measured))
        print(f"  [{rid}] refined [{s_lo:.1f}, {s_hi:.1f}) s", flush=True)
    cat = concat_segments(segs)
    r_cmd, r_ref, r_meas = cat["r_init"], cat["r_refined"], cat["r_measured"]

    err_cmd = np.abs(r_cmd - r_meas)  # (R, N)
    err_ref = np.abs(r_ref - r_meas)

    # LSQ residual on a representative window (first LSQ_WIN_S of inflight,
    # channel 0) — the full-inflight fit costs minutes per trajectory and
    # the ratio is stable across windows.
    lsq_hi = min(t_lo + LSQ_WIN_S, t_hi)
    lsq_audio = audio[:1, int(t_lo * SR) : int(lsq_hi * SR)]
    in_win = cat["frame_times"] <= lsq_hi
    ft_rel = cat["frame_times"][in_win] - t_lo
    resid = {
        k: harmonic_lsq_residual(lsq_audio, traj[:, in_win], ft_rel, cfg, k_max=LSQ_KMAX)[
            "residual_ratio"
        ]
        for k, traj in (("cmd", r_cmd), ("ref", r_ref), ("meas", r_meas))
    }

    row: dict[str, Any] = {
        "dataset": "dregon",
        "recording": rid,
        "tau": round(tau, 4),
        "inflight_s": round(t_hi - t_lo, 1),
        "n_frames": int(r_ref.shape[1]),
        "err_command": float(err_cmd.mean()),
        "err_refined": float(err_ref.mean()),
        "err_refined_pit": pit_mae(r_ref, r_meas),
        "err_command_p50": float(np.percentile(err_cmd, 50)),
        "err_command_p90": float(np.percentile(err_cmd, 90)),
        "err_refined_p50": float(np.percentile(err_ref, 50)),
        "err_refined_p90": float(np.percentile(err_ref, 90)),
        "resid_command": resid["cmd"],
        "resid_refined": resid["ref"],
        "resid_measured": resid["meas"],
        "mean_confidence": float(cat["confidence"].mean()),
        "improved": bool(err_ref.mean() < err_cmd.mean()),
    }
    for i in range(err_cmd.shape[0]):
        row[f"err_command_r{i}"] = float(err_cmd[i].mean())
        row[f"err_refined_r{i}"] = float(err_ref[i].mean())

    np.savez(
        OUT_DIR / f"dregon_{rid}.npz",
        frame_times=cat["frame_times"],
        command=r_cmd,
        measured=r_meas,
        refined=r_ref,
        confidence=cat["confidence"],
        conf_centers=cat["conf_centers"],
        tau=tau,
        err_command_per_rotor=err_cmd.mean(axis=1),
        err_refined_per_rotor=err_ref.mean(axis=1),
    )
    return row, err_cmd, err_ref


def michaels_recording_task(rec_index: int) -> list[dict[str, Any]]:
    """Refine up to 4 non-idle 30 s segments of one Michael's recording (worker).

    No independent ground truth: reports refinement delta magnitude, LSQ
    residual at init vs refined, and comb confidence. Saves the NPZ.
    """
    cfg = RefineConfig()
    rows: list[dict[str, Any]] = []
    frame = load_michaels_timeframes("data")[rec_index]
    rid = str(frame["meta"]["recording_id"])
    print(f"[Michael's] {rid}", flush=True)
    audio = np.asarray(frame["audio"].data)
    rps = np.asarray(frame["rps"].data)
    t0 = float(frame["audio"].tindex.t_start)
    rt = np.asarray(frame["rps"].tindex.abs_stamps) - t0
    dur = audio.shape[1] / SR

    picked: list[tuple[float, float]] = []
    for s_lo, s_hi in segment_bounds(0.0, dur):
        m = (rt >= s_lo) & (rt < s_hi)
        if m.any() and float(np.median(rps[:, m])) > MICHAELS_MIN_RPS:
            picked.append((s_lo, s_hi))
        if len(picked) >= 4:
            break

    seg_ft: list[np.ndarray] = []
    seg_init: list[np.ndarray] = []
    seg_ref: list[np.ndarray] = []
    seg_conf: list[np.ndarray] = []
    for k, (s_lo, s_hi) in enumerate(picked):
        # Channel 0 only for the tau scan (memory; see dregon_recording_task).
        spec0 = compute_logmag(audio[:1, int(s_lo * SR) : int(s_hi * SR)], cfg)
        tau, _, _ = estimate_clock_offset(spec0, rt - s_lo, rps, cfg)
        seg = refine_segment(audio, s_lo, s_hi, rt, rps, cfg, tau, None)
        print(f"  [{rid}] refined [{s_lo:.1f}, {s_hi:.1f}) s", flush=True)
        delta = np.abs(seg.r_refined - seg.r_init)
        # Residual on the first LSQ_WIN_S of the segment, channel 0 (memory).
        lsq_hi = min(s_lo + LSQ_WIN_S, s_hi)
        in_win = seg.frame_times <= lsq_hi
        ft_rel = seg.frame_times[in_win] - s_lo
        ch0 = audio[:1, int(s_lo * SR) : int(lsq_hi * SR)]  # (1, T) channel 0
        res_init = harmonic_lsq_residual(ch0, seg.r_init[:, in_win], ft_rel, cfg, k_max=LSQ_KMAX)[
            "residual_ratio"
        ]
        res_ref = harmonic_lsq_residual(ch0, seg.r_refined[:, in_win], ft_rel, cfg, k_max=LSQ_KMAX)[
            "residual_ratio"
        ]
        rows.append(
            {
                "dataset": "michaels",
                "recording": f"{rid}_seg{k}",
                "tau": round(tau, 4),
                "inflight_s": round(s_hi - s_lo, 1),
                "n_frames": int(seg.r_refined.shape[1]),
                "delta_mean": float(delta.mean()),
                "delta_p50": float(np.percentile(delta, 50)),
                "delta_p90": float(np.percentile(delta, 90)),
                "resid_command": res_init,
                "resid_refined": res_ref,
                "mean_confidence": float(seg.confidence.mean()),
                "improved": bool(res_ref < res_init),
            }
        )
        seg_ft.append(seg.frame_times)
        seg_init.append(seg.r_init)
        seg_ref.append(seg.r_refined)
        seg_conf.append(seg.confidence)

    if seg_ft:
        np.savez(
            OUT_DIR / f"michaels_{rid}.npz",
            frame_times=np.concatenate(seg_ft),
            init=np.concatenate(seg_init, axis=1),
            refined=np.concatenate(seg_ref, axis=1),
            confidence=np.concatenate(seg_conf, axis=1),
            seg_starts=np.array([s for s, _ in picked]),
        )
    return rows


# ---------------------------------------------------------------------------
# Preview figures
# ---------------------------------------------------------------------------


def _overlay_harmonics(ax: Any, spec: Any, r_ref_win: np.ndarray, ft_win: np.ndarray) -> None:
    """Spectrogram (0-2 kHz, ch0) + refined harmonic tracks for k in {1,2,4,8,16}."""
    logmag = spec.logmag[0].cpu().numpy()  # (F, N)
    f_hi_bin = int(2000 / spec.bin_hz)
    ax.imshow(
        logmag[:f_hi_bin],
        origin="lower",
        aspect="auto",
        extent=(float(ft_win[0]), float(ft_win[-1]), 0.0, f_hi_bin * spec.bin_hz),
        cmap="magma",
    )
    for k in (1, 2, 4, 8, 16):
        track = k * r_ref_win[0]
        vis = track <= 2000
        ax.plot(ft_win[vis], track[vis], lw=1.0, label=f"k={k}")
    ax.set_ylim(0, 2000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    ax.set_title("refined harmonic tracks (rotor 0)")
    ax.legend(loc="upper right", fontsize=7, ncol=5)


def make_dregon_preview(cfg: RefineConfig) -> None:
    npz = np.load(OUT_DIR / f"dregon_{DREGON_PREVIEW}.npz")
    ft = npz["frame_times"]
    cmd, meas, ref = npz["command"], npz["measured"], npz["refined"]

    z0 = float(ft[0])
    zoom = (ft >= z0) & (ft <= z0 + 10.0)
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))

    ax[0].plot(ft[zoom], meas[0][zoom], "k-", lw=1.6, label="measured (GT)")
    ax[0].plot(ft[zoom], cmd[0][zoom], "--", color="tab:gray", lw=1.3, label="command")
    ax[0].plot(ft[zoom], ref[0][zoom], color="tab:red", lw=1.3, label="refined")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("rev/s")
    ax[0].set_title(f"{DREGON_PREVIEW}: rotor 0 trajectory (10 s zoom)")
    ax[0].legend()

    err_cmd = np.abs(cmd - meas).mean(axis=0)
    err_ref = np.abs(ref - meas).mean(axis=0)
    ax[1].plot(ft, err_cmd, color="tab:gray", lw=1.0, label="|command - measured|")
    ax[1].plot(ft, err_ref, color="tab:red", lw=1.0, label="|refined - measured|")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("mean |error| (rev/s)")
    ax[1].set_title("rotor-averaged error vs time")
    ax[1].legend()

    # Panel (c): recompute a spectrogram on the 10 s zoom window.
    dregon_dir = Path("data") / "DREGON"
    by_id = {s["recording_id"]: s for s in discover_recordings(dregon_dir)}
    frame = load_timeframe(by_id[DREGON_PREVIEW], geometry=get_geometry(dregon_dir), target_sr=SR)
    audio = np.asarray(frame["audio"].data)
    win = audio[:, int(z0 * SR) : int((z0 + 10.0) * SR)]
    spec = compute_logmag(win, cfg)
    ref_win = ref[:, zoom]
    _overlay_harmonics(ax[2], spec, ref_win, spec.frame_times + z0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview_dregon.png", dpi=150)
    plt.close(fig)


def make_michaels_preview(cfg: RefineConfig) -> None:
    frame = load_michaels_timeframes("data")[0]
    rid = str(frame["meta"]["recording_id"])
    npz = np.load(OUT_DIR / f"michaels_{rid}.npz")
    ft, init, ref = npz["frame_times"], npz["init"], npz["refined"]
    starts = npz["seg_starts"]

    z0 = float(starts[0])
    zoom = (ft >= z0) & (ft <= z0 + 10.0)
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))

    ax[0].plot(ft[zoom], init[0][zoom], "--", color="tab:gray", lw=1.3, label="init (rps track)")
    ax[0].plot(ft[zoom], ref[0][zoom], color="tab:red", lw=1.3, label="refined")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("rev/s")
    ax[0].set_title(f"Michael's {rid}: rotor 0 trajectory (10 s zoom; no GT)")
    ax[0].legend()

    delta = np.abs(ref - init).mean(axis=0)
    ax[1].plot(ft[zoom], delta[zoom], color="tab:purple", lw=1.0, label="|refined - init|")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("mean |delta| (rev/s)")
    ax[1].set_title("rotor-averaged refinement delta vs time")
    ax[1].legend()

    audio = np.asarray(frame["audio"].data)
    win = audio[:, int(z0 * SR) : int((z0 + 10.0) * SR)]
    spec = compute_logmag(win, cfg)
    _overlay_harmonics(ax[2], spec, ref[:, zoom], spec.frame_times + z0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview_michaels.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------


def print_summary(dregon_rows: list[dict[str, Any]], pooled: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 96)
    print("DREGON: refinement vs measured ground truth (rev/s error; residual = LSQ audio-fit)")
    print("=" * 96)
    hdr = (
        f"{'recording':<34}{'tau':>7}{'err_cmd':>9}{'err_ref':>9}"
        f"{'err_pit':>9}{'r_cmd':>8}{'r_ref':>8}{'r_meas':>8}"
    )
    print(hdr)
    print("-" * 96)
    for r in dregon_rows:
        flag = "" if r["improved"] else "  <-- WORSE"
        print(
            f"{r['recording']:<34}{r['tau']:>7.3f}{r['err_command']:>9.3f}"
            f"{r['err_refined']:>9.3f}{r['err_refined_pit']:>9.3f}"
            f"{r['resid_command']:>8.3f}{r['resid_refined']:>8.3f}{r['resid_measured']:>8.3f}{flag}"
        )
    pc = float(pooled["cmd"].mean())
    pr = float(pooled["ref"].mean())
    print("-" * 96)
    print(f"{'POOLED (all 5)':<34}{'':>7}{pc:>9.3f}{pr:>9.3f}")
    n_worse = sum(not r["improved"] for r in dregon_rows)
    print(
        f"\nPooled mean |error|: command {pc:.3f}  ->  refined {pr:.3f} rev/s  "
        f"({'IMPROVED' if pr < pc else 'WORSE'})"
    )
    if n_worse:
        print(
            f"\n!! ANOMALY: refinement moved AWAY from measured GT in {n_worse}/"
            f"{len(dregon_rows)} DREGON recordings (err_ref > err_cmd). "
            "See err_pit to gauge how much is rotor-swap vs genuine drift."
        )
        for r in dregon_rows:
            if not r["improved"]:
                print(
                    f"   - {r['recording']}: err {r['err_command']:.3f} -> {r['err_refined']:.3f} "
                    f"(pit {r['err_refined_pit']:.3f}); residual {r['resid_command']:.3f} -> "
                    f"{r['resid_refined']:.3f} (measured {r['resid_measured']:.3f})"
                )
    print("=" * 96)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = RefineConfig()

    # One worker process per recording (7 tasks). Michael's recordings are the
    # longest (4 segments each), so submit them first.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
        michaels_futs = [pool.submit(michaels_recording_task, i) for i in range(2)]
        dregon_futs = {rid: pool.submit(dregon_recording_task, rid) for rid in DREGON_TARGETS}
        dregon_rows = []
        pool_cmd = []
        pool_ref = []
        for rid in DREGON_TARGETS:  # keep DREGON_TARGETS order in outputs
            row, err_cmd, err_ref = dregon_futs[rid].result()
            dregon_rows.append(row)
            pool_cmd.append(err_cmd)
            pool_ref.append(err_ref)
        michaels_rows = [row for fut in michaels_futs for row in fut.result()]
    pooled = {"cmd": np.concatenate(pool_cmd, axis=1), "ref": np.concatenate(pool_ref, axis=1)}

    # Pooled DREGON row for the CSV.
    pooled_row: dict[str, Any] = {
        "dataset": "dregon",
        "recording": "POOLED",
        "err_command": float(pooled["cmd"].mean()),
        "err_refined": float(pooled["ref"].mean()),
        "improved": bool(pooled["ref"].mean() < pooled["cmd"].mean()),
    }
    for i in range(pooled["cmd"].shape[0]):
        pooled_row[f"err_command_r{i}"] = float(pooled["cmd"][i].mean())
        pooled_row[f"err_refined_r{i}"] = float(pooled["ref"][i].mean())

    pd.DataFrame([*dregon_rows, pooled_row, *michaels_rows]).to_csv(
        OUT_DIR / "summary.csv", index=False
    )

    make_dregon_preview(cfg)
    make_michaels_preview(cfg)
    print_summary(dregon_rows, pooled)
    print(f"\nArtifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
