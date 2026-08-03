"""Validate RPS trajectory refinement against annotated ground truth.

The refiner (``data_processing.rps_refinement``) nudges telemetry-derived rotor
trajectories toward the audio's harmonic comb. This script is the natural
experiment that tests whether that nudge moves *toward the truth*:

* **DREGON** ``free-flight_*_room1`` (5 recordings) carry BOTH
  ``motors_command`` (what training uses) and ``motors_measured`` (actual rotor
  speeds = ground truth). Four-way comparison per recording: command
  (tau-shifted), stage B+C ``refine_trajectories`` (magnitude comb), stage D
  ``refine_coherent`` (phase slope), measured. Metrics: unsigned error and
  SIGNED bias vs measured and vs 0.25 s-smoothed measured (separates fast
  sensor jitter from systematic drift), plus the harmonic LSQ residual for all
  four trajectories. Context from the first pass: the magnitude comb is biased
  here because the four rotors form two tight pairs (~0.65 rev/s apart), while
  command labels are nearly unbiased (their unsigned error is zero-mean fast
  jitter) — stage D exists to fix exactly that.
* **Michael's** FLY124/FLY125 have no independent ground truth. Both refiners
  run from the telemetry ``rps`` init; we report audio-fit (LSQ residual at
  init / stage B+C / stage D), refinement delta magnitudes, and comb
  confidence.

Everything runs CPU-only and per ~30 s segment to bound memory/runtime.

Artifacts (``results/rps_refinement/validation/``):
  * ``<dataset>_<recording>.npz`` — frame_times, command/init, measured (DREGON),
    refined (stage B+C), refined_coherent (stage D), confidence (+centers), tau.
  * ``summary.csv`` — one row per recording (DREGON) / per segment (Michael's),
    plus a pooled DREGON row; four-way err/bias/residual columns.
  * ``preview_dregon.png`` / ``preview_michaels.png`` — four-way trajectory
    overlay, error-vs-time, and spectrogram with refined harmonic tracks.

Run: ``.venv/bin/python scripts/rps_refinement_validation.py``
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

from data_processing.rps_refinement import (  # noqa: E402
    RefineConfig,
    compute_logmag,
    estimate_clock_offset,
    harmonic_lsq_residual,
    refine_coherent,
    refine_trajectories,
)
from data_processing.sources.dregon import (  # noqa: E402
    clean_command_spikes,
    discover_recordings,
    get_geometry,
    load_timeframe,
)
from data_processing.sources.michaels import load_michaels_timeframes  # noqa: E402

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
    r_refined: np.ndarray  # (R, N) stage B+C (magnitude comb)
    r_coherent: np.ndarray  # (R, N) stage D (phase slope), from the same init
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
    res = refine_trajectories(spec, r_init, cfg)  # stage B+C (magnitude comb)
    r_coh = refine_coherent(seg, r_init, ft, cfg)  # stage D (phase slope), same init
    r_meas = None
    if measured_values is not None:
        r_meas = np.stack([np.interp(ft + tau, mt_rel, measured_values[i]) for i in range(n_rotor)])
    return SegmentResult(
        seg_lo=seg_lo,
        frame_times=seg_lo + ft,
        r_init=r_init,
        r_refined=res.r_refined,
        r_coherent=r_coh,
        r_measured=r_meas,
        confidence=res.confidence,
        conf_centers=seg_lo + res.conf_centers,
    )


def concat_segments(segs: list[SegmentResult]) -> dict[str, np.ndarray]:
    """Concatenate per-segment arrays; sort/dedupe frame times for monotonicity."""
    ft = np.concatenate([s.frame_times for s in segs])
    order = np.argsort(ft)
    keep = np.concatenate([[True], np.diff(ft[order]) > 1e-9])

    def cat(tracks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(tracks, axis=1)[:, order][:, keep]

    out: dict[str, np.ndarray] = {
        "frame_times": ft[order][keep],
        "r_init": cat([s.r_init for s in segs]),
        "r_refined": cat([s.r_refined for s in segs]),
        "r_coherent": cat([s.r_coherent for s in segs]),
        "confidence": np.concatenate([s.confidence for s in segs], axis=1),
        "conf_centers": np.concatenate([s.conf_centers for s in segs]),
    }
    if segs[0].r_measured is not None:
        out["r_measured"] = cat([s.r_measured for s in segs])  # type: ignore[misc]
    return out


def pit_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Best-permutation mean |error| over rotors (separates swaps from error)."""
    r = pred.shape[0]
    cost = np.array([[np.mean(np.abs(pred[i] - gt[j])) for j in range(r)] for i in range(r)])
    best = min(sum(cost[i, p[i]] for i in range(r)) for p in permutations(range(r)))
    return float(best / r)


# 0.25 s on the 32 ms frame grid — separates measured's fast sensor jitter
# (which no smooth refiner can track) from systematic drift/bias.
SMOOTH_FRAMES = 8


def smooth_frames(x: np.ndarray, win: int = SMOOTH_FRAMES) -> np.ndarray:
    """Per-rotor moving average along the frame axis."""
    ker = np.ones(win) / win
    return np.stack([np.convolve(row, ker, mode="same") for row in x])


def err_bias_stats(
    traj: np.ndarray, meas: np.ndarray, meas_sm: np.ndarray, name: str
) -> dict[str, float]:
    """Unsigned error + signed bias vs measured and vs smoothed measured."""
    d = traj - meas
    d_sm = traj - meas_sm
    return {
        f"err_{name}": float(np.mean(np.abs(d))),
        f"bias_{name}": float(np.mean(d)),
        f"err_sm_{name}": float(np.mean(np.abs(d_sm))),
        f"bias_sm_{name}": float(np.mean(d_sm)),
    }


def dregon_recording_task(rid: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Refine one measured-carrying DREGON recording (worker process).

    Saves the per-recording NPZ and returns ``(summary_row, diffs)`` where
    ``diffs`` holds the signed (R, N) trajectory-minus-measured arrays (raw
    and vs smoothed measured) for command / stage B+C / stage D, for pooling.
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
    r_cmd, r_bc, r_d = cat["r_init"], cat["r_refined"], cat["r_coherent"]
    r_meas = cat["r_measured"]
    r_meas_sm = smooth_frames(r_meas)
    trajs = {"command": r_cmd, "stagebc": r_bc, "staged": r_d}

    # LSQ residual on a representative window (first LSQ_WIN_S of inflight,
    # channel 0) — the full-inflight fit costs minutes per trajectory and
    # the ratio is stable across windows.
    lsq_hi = min(t_lo + LSQ_WIN_S, t_hi)
    lsq_audio = audio[:1, int(t_lo * SR) : int(lsq_hi * SR)]
    in_win = cat["frame_times"] <= lsq_hi
    ft_rel = cat["frame_times"][in_win] - t_lo
    resid = {
        name: harmonic_lsq_residual(lsq_audio, traj[:, in_win], ft_rel, cfg, k_max=LSQ_KMAX)[
            "residual_ratio"
        ]
        for name, traj in {**trajs, "measured": r_meas}.items()
    }

    row: dict[str, Any] = {
        "dataset": "dregon",
        "recording": rid,
        "tau": round(tau, 4),
        "inflight_s": round(t_hi - t_lo, 1),
        "n_frames": int(r_bc.shape[1]),
        "mean_confidence": float(cat["confidence"].mean()),
        "err_pit_stagebc": pit_mae(r_bc, r_meas),
        "err_pit_staged": pit_mae(r_d, r_meas),
    }
    for name, traj in trajs.items():
        row.update(err_bias_stats(traj, r_meas, r_meas_sm, name))
        row[f"resid_{name}"] = resid[name]
    row["resid_measured"] = resid["measured"]

    diffs = {name: traj - r_meas for name, traj in trajs.items()}
    diffs |= {f"{name}_sm": traj - r_meas_sm for name, traj in trajs.items()}

    np.savez(
        OUT_DIR / f"dregon_{rid}.npz",
        frame_times=cat["frame_times"],
        command=r_cmd,
        measured=r_meas,
        refined=r_bc,
        refined_coherent=r_d,
        confidence=cat["confidence"],
        conf_centers=cat["conf_centers"],
        tau=tau,
    )
    return row, diffs


def michaels_recording_task(rec_index: int) -> list[dict[str, Any]]:
    """Refine up to 4 non-idle 30 s segments of one Michael's recording (worker).

    No independent ground truth: both refiners run from the telemetry init;
    reports delta magnitudes, LSQ residual at init / stage B+C / stage D, and
    comb confidence. Saves the NPZ.
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
    seg_bc: list[np.ndarray] = []
    seg_d: list[np.ndarray] = []
    seg_conf: list[np.ndarray] = []
    for k, (s_lo, s_hi) in enumerate(picked):
        # Channel 0 only for the tau scan (memory; see dregon_recording_task).
        spec0 = compute_logmag(audio[:1, int(s_lo * SR) : int(s_hi * SR)], cfg)
        tau, _, _ = estimate_clock_offset(spec0, rt - s_lo, rps, cfg)
        seg = refine_segment(audio, s_lo, s_hi, rt, rps, cfg, tau, None)
        print(f"  [{rid}] refined [{s_lo:.1f}, {s_hi:.1f}) s", flush=True)
        delta_bc = np.abs(seg.r_refined - seg.r_init)
        delta_d = np.abs(seg.r_coherent - seg.r_init)
        # Residual on the first LSQ_WIN_S of the segment, channel 0 (memory).
        lsq_hi = min(s_lo + LSQ_WIN_S, s_hi)
        in_win = seg.frame_times <= lsq_hi
        ft_rel = seg.frame_times[in_win] - s_lo
        ch0 = audio[:1, int(s_lo * SR) : int(lsq_hi * SR)]  # (1, T) channel 0
        resid = {
            name: harmonic_lsq_residual(ch0, traj[:, in_win], ft_rel, cfg, k_max=LSQ_KMAX)[
                "residual_ratio"
            ]
            for name, traj in (
                ("init", seg.r_init),
                ("stagebc", seg.r_refined),
                ("staged", seg.r_coherent),
            )
        }
        rows.append(
            {
                "dataset": "michaels",
                "recording": f"{rid}_seg{k}",
                "tau": round(tau, 4),
                "inflight_s": round(s_hi - s_lo, 1),
                "n_frames": int(seg.r_refined.shape[1]),
                "delta_bc_mean": float(delta_bc.mean()),
                "delta_bc_p50": float(np.percentile(delta_bc, 50)),
                "delta_bc_p90": float(np.percentile(delta_bc, 90)),
                "delta_d_mean": float(delta_d.mean()),
                "delta_d_p50": float(np.percentile(delta_d, 50)),
                "delta_d_p90": float(np.percentile(delta_d, 90)),
                "resid_init": resid["init"],
                "resid_stagebc": resid["stagebc"],
                "resid_staged": resid["staged"],
                "mean_confidence": float(seg.confidence.mean()),
            }
        )
        seg_ft.append(seg.frame_times)
        seg_init.append(seg.r_init)
        seg_bc.append(seg.r_refined)
        seg_d.append(seg.r_coherent)
        seg_conf.append(seg.confidence)

    if seg_ft:
        np.savez(
            OUT_DIR / f"michaels_{rid}.npz",
            frame_times=np.concatenate(seg_ft),
            init=np.concatenate(seg_init, axis=1),
            refined=np.concatenate(seg_bc, axis=1),
            refined_coherent=np.concatenate(seg_d, axis=1),
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
    ax.set_title("stage D refined harmonic tracks (rotor 0)")
    ax.legend(loc="upper right", fontsize=7, ncol=5)


def make_dregon_preview(cfg: RefineConfig) -> None:
    npz = np.load(OUT_DIR / f"dregon_{DREGON_PREVIEW}.npz")
    ft = npz["frame_times"]
    cmd, meas = npz["command"], npz["measured"]
    bc, coh = npz["refined"], npz["refined_coherent"]

    z0 = float(ft[0])
    zoom = (ft >= z0) & (ft <= z0 + 10.0)
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))

    ax[0].plot(ft[zoom], meas[0][zoom], "k-", lw=1.6, label="measured (GT)")
    ax[0].plot(ft[zoom], cmd[0][zoom], "--", color="tab:gray", lw=1.3, label="command")
    ax[0].plot(ft[zoom], bc[0][zoom], color="tab:red", lw=1.2, label="stage B+C (comb)")
    ax[0].plot(ft[zoom], coh[0][zoom], color="tab:blue", lw=1.2, label="stage D (coherent)")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("rev/s")
    ax[0].set_title(f"{DREGON_PREVIEW}: rotor 0 trajectory (10 s zoom)")
    ax[0].legend()

    for traj, color, name in (
        (cmd, "tab:gray", "command"),
        (bc, "tab:red", "stage B+C"),
        (coh, "tab:blue", "stage D"),
    ):
        err = np.abs(traj - meas).mean(axis=0)
        ax[1].plot(ft, err, color=color, lw=0.9, alpha=0.85, label=f"|{name} - measured|")
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
    _overlay_harmonics(ax[2], spec, coh[:, zoom], spec.frame_times + z0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview_dregon.png", dpi=150)
    plt.close(fig)


def make_michaels_preview(cfg: RefineConfig) -> None:
    frame = load_michaels_timeframes("data")[0]
    rid = str(frame["meta"]["recording_id"])
    npz = np.load(OUT_DIR / f"michaels_{rid}.npz")
    ft, init = npz["frame_times"], npz["init"]
    bc, coh = npz["refined"], npz["refined_coherent"]
    starts = npz["seg_starts"]

    z0 = float(starts[0])
    zoom = (ft >= z0) & (ft <= z0 + 10.0)
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))

    ax[0].plot(ft[zoom], init[0][zoom], "--", color="tab:gray", lw=1.3, label="init (rps track)")
    ax[0].plot(ft[zoom], bc[0][zoom], color="tab:red", lw=1.2, label="stage B+C (comb)")
    ax[0].plot(ft[zoom], coh[0][zoom], color="tab:blue", lw=1.2, label="stage D (coherent)")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("rev/s")
    ax[0].set_title(f"Michael's {rid}: rotor 0 trajectory (10 s zoom; no GT)")
    ax[0].legend()

    for traj, color, name in ((bc, "tab:red", "stage B+C"), (coh, "tab:blue", "stage D")):
        delta = np.abs(traj - init).mean(axis=0)
        ax[1].plot(ft[zoom], delta[zoom], color=color, lw=1.0, label=f"|{name} - init|")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("mean |delta| (rev/s)")
    ax[1].set_title("rotor-averaged refinement delta vs time")
    ax[1].legend()

    audio = np.asarray(frame["audio"].data)
    win = audio[:, int(z0 * SR) : int((z0 + 10.0) * SR)]
    spec = compute_logmag(win, cfg)
    _overlay_harmonics(ax[2], spec, coh[:, zoom], spec.frame_times + z0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview_michaels.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------


def print_summary(dregon_rows: list[dict[str, Any]], pooled_row: dict[str, Any]) -> None:
    width = 128
    print("\n" + "=" * width)
    print(
        "DREGON four-way: command / stage B+C (comb) / stage D (coherent) vs measured GT "
        "(rev/s; resid = LSQ audio-fit)"
    )
    print("=" * width)
    hdr = (
        f"{'recording':<34}{'tau':>7}"
        f"{'err_cmd':>9}{'err_BC':>8}{'err_D':>8}"
        f"{'bias_cmd':>9}{'bias_BC':>9}{'bias_D':>8}"
        f"{'r_cmd':>7}{'r_BC':>7}{'r_D':>7}{'r_meas':>8}"
    )
    print(hdr)
    print("-" * width)
    for r in [*dregon_rows, pooled_row]:
        tau = f"{r['tau']:>7.3f}" if "tau" in r else f"{'':>7}"
        resid = (
            f"{r['resid_command']:>7.3f}{r['resid_stagebc']:>7.3f}"
            f"{r['resid_staged']:>7.3f}{r['resid_measured']:>8.3f}"
            if "resid_command" in r
            else f"{'':>7}{'':>7}{'':>7}{'':>8}"
        )
        print(
            f"{r['recording']:<34}{tau}"
            f"{r['err_command']:>9.3f}{r['err_stagebc']:>8.3f}{r['err_staged']:>8.3f}"
            f"{r['bias_command']:>9.3f}{r['bias_stagebc']:>9.3f}{r['bias_staged']:>8.3f}"
            f"{resid}"
        )
    print("-" * width)
    print(
        "vs 0.25s-smoothed measured (pooled): "
        + "  ".join(
            f"{name}: err {pooled_row[f'err_sm_{name}']:.3f} bias {pooled_row[f'bias_sm_{name}']:+.3f}"
            for name in ("command", "stagebc", "staged")
        )
    )

    # --- expectation checks (flag loudly, do not smooth over surprises) -----
    # Expected: stage B+C worse than command (pair-capture bias); stage D ~
    # command in unsigned err with near-zero bias and resid <= command's;
    # measured best LSQ everywhere.
    surprises: list[str] = []
    if pooled_row["err_stagebc"] <= pooled_row["err_command"]:
        surprises.append(
            f"stage B+C pooled err {pooled_row['err_stagebc']:.3f} <= command "
            f"{pooled_row['err_command']:.3f} — comb bias expected to hurt, but it did not"
        )
    if abs(pooled_row["bias_staged"]) > 0.15:
        surprises.append(
            f"stage D pooled bias {pooled_row['bias_staged']:+.3f} rev/s — expected near-zero "
            "(<0.08 on room1 per the module diagnostic)"
        )
    if pooled_row["err_staged"] > 1.15 * pooled_row["err_command"]:
        surprises.append(
            f"stage D pooled err {pooled_row['err_staged']:.3f} >> command "
            f"{pooled_row['err_command']:.3f} — expected roughly equal"
        )
    for r in dregon_rows:
        if r["resid_staged"] > r["resid_command"] + 1e-3:
            surprises.append(
                f"{r['recording']}: stage D residual {r['resid_staged']:.3f} worse than "
                f"command {r['resid_command']:.3f} — expected slightly better"
            )
        if r["resid_measured"] > min(r["resid_command"], r["resid_stagebc"], r["resid_staged"]):
            surprises.append(
                f"{r['recording']}: measured is NOT the best LSQ fit "
                f"(meas {r['resid_measured']:.3f}) — check tau/protocol"
            )
    if surprises:
        print("\n!! SURPRISES (deviations from the expected result shape):")
        for s in surprises:
            print(f"   - {s}")
    else:
        print(
            "\nAll expectation checks passed: stage B+C degrades err (comb pair-capture bias), "
            "stage D ~ command err with near-zero bias, measured best LSQ."
        )
    print("=" * width)


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
        pool_diffs: dict[str, list[np.ndarray]] = {}
        for rid in DREGON_TARGETS:  # keep DREGON_TARGETS order in outputs
            row, diffs = dregon_futs[rid].result()
            dregon_rows.append(row)
            for name, d in diffs.items():
                pool_diffs.setdefault(name, []).append(d)
        michaels_rows = [row for fut in michaels_futs for row in fut.result()]
    pooled = {name: np.concatenate(ds, axis=1) for name, ds in pool_diffs.items()}

    # Pooled DREGON row for the CSV (signed diffs pooled over all recordings).
    pooled_row: dict[str, Any] = {"dataset": "dregon", "recording": "POOLED"}
    for name in ("command", "stagebc", "staged"):
        pooled_row[f"err_{name}"] = float(np.mean(np.abs(pooled[name])))
        pooled_row[f"bias_{name}"] = float(np.mean(pooled[name]))
        pooled_row[f"err_sm_{name}"] = float(np.mean(np.abs(pooled[f"{name}_sm"])))
        pooled_row[f"bias_sm_{name}"] = float(np.mean(pooled[f"{name}_sm"]))

    pd.DataFrame([*dregon_rows, pooled_row, *michaels_rows]).to_csv(
        OUT_DIR / "summary.csv", index=False
    )

    make_dregon_preview(cfg)
    make_michaels_preview(cfg)
    print_summary(dregon_rows, pooled_row)
    print(f"\nArtifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
