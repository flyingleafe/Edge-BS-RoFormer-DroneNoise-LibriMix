#!/usr/bin/env python3
"""Generate figures and tables for the synthetic-RPS-trajectory report.

Loads the real DREGON ``in_flight_noise`` + Michael's rotor-speed telemetry,
calibrates the control-mode synthesizer
(:mod:`data_processing.rps_synthesis`), and produces:

  assets/calibration.csv        per-recording mode statistics + fitted defaults
  assets/intermittency.csv      per-recording maneuver-activity statistics
  assets/rc_sticks.png          raw DJI pilot RC stick inputs (bursty evidence)
  assets/model_comparison.png   real vs continuous-OU vs intermittent trajectory
  assets/intermittent_agg.png   intermittent trajectories: gentle / normal / aggressive
  assets/drone_profile_sweep.png  DREGON -> in-between -> Michael's airframe dynamics
  assets/traj_examples.png      OU trajectories: gentle / normal / aggressive
  assets/real_vs_synth.png      a real DREGON flight vs a calibrated OU trajectory
  assets/mode_decomposition.png the four control modes, real vs OU
  assets/distributions.png      per-rotor RPS histograms + rotor-correlation matrices
  assets/aggressiveness_sweep.png  maneuver spread vs aggressiveness, against real flights

Real data is found under the first of ``$DATA_ROOT`` / ``<repo>/data`` /
the main checkout that actually contains a ``DREGON/`` directory.  The raw DJI
RC-stick columns are read directly from ``FLY12{4,5}.csv`` against their own
logger clock (``Clock:offsetTime``); aligning them to audio would apply the same
whole-table ``time_offset`` / ``time_dilation`` correction used in
``data_processing/michaels.py`` (not needed here, where we only show structure).
"""

from __future__ import annotations

import csv
import os
import pathlib
import subprocess
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tdseries as td

from data_processing.dregon import (
    MOTOR_SAMPLE_RATE,
    clean_command_spikes,
    load_dregon_timeframes,
)
from data_processing.frames import get_meta
from data_processing.michaels import load_michaels_timeframes
from data_processing.rps_synthesis import (
    DEFAULT_CONFIG,
    MODE_NAMES,
    NUM_ROTORS,
    RPSSynthConfig,
    fit_config,
    generate,
    modes_from_rps,
)

ASSETS = pathlib.Path("assets")
ROTOR_LABELS = ("RFront", "LFront", "LBack", "RBack")
ROTOR_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")
SYNTH_FS = 100.0  # Hz: synthetic trajectory sample rate used throughout
AGG_LEVELS = {"gentle": 0.4, "normal": 1.0, "aggressive": 2.5}


# ---------------------------------------------------------------------------
# Real-data loading
# ---------------------------------------------------------------------------
def _find_data_root() -> pathlib.Path | None:
    candidates = []
    if os.environ.get("DATA_ROOT"):
        candidates.append(pathlib.Path(os.environ["DATA_ROOT"]))
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
        candidates.append(pathlib.Path(root) / "data")
        # worktrees keep data only in the main checkout
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        candidates.append(pathlib.Path(common).parent / "data")
    except subprocess.CalledProcessError:
        pass
    for c in candidates:
        if (c / "DREGON").is_dir():
            return c
    return None


def load_real_traces() -> tuple[list[tuple[str, np.ndarray, float]], pathlib.Path | None]:
    """Return ``[(label, (4, M) rps, dt), ...]`` for DREGON + Michael's flights."""
    root = _find_data_root()
    out: list[tuple[str, np.ndarray, float]] = []
    if root is None:
        print("WARNING: no real data found; comparison figures use DEFAULT_CONFIG only.")
        return out, None
    for tf in load_dregon_timeframes(root, splits=["in_flight_noise"], download=False):
        cmd = cast(td.Series, tf["motors_command"])
        w = clean_command_spikes(np.asarray(cmd.data, dtype=np.float64))
        out.append((f"DREGON/{get_meta(tf, 'recording_id')}", w, 1.0 / MOTOR_SAMPLE_RATE))
    for tf in load_michaels_timeframes(data_root=root):
        es = cast(td.Series, tf["rps"])
        w = np.asarray(es.data, dtype=np.float64)
        ts = np.asarray(cast(td.StampIndex, es.tindex).abs_stamps, dtype=np.float64)
        out.append((f"michaels/{get_meta(tf, 'recording_id')}", w, float(np.median(np.diff(ts)))))
    return out, root


def inflight(w: np.ndarray, thr: float = 30.0) -> np.ndarray:
    return w[:, w.mean(axis=0) > thr]


def cruise(w: np.ndarray, dt: float, thr: float = 55.0, pad_s: float = 0.9) -> np.ndarray:
    """Trim to the steady cruise region: drop takeoff ramp and landing.

    Takes the airborne span (rotor-mean above ``thr``) and skips ``pad_s`` seconds
    after the first crossing so the takeoff transient does not dominate plots.
    """
    mean = w.mean(axis=0)
    idx = np.where(mean > thr)[0]
    if idx.size == 0:
        return w
    lo = idx[0] + int(pad_s / dt)
    hi = idx[-1]
    return w[:, lo:hi] if hi > lo else w


def maneuver_activity(w: np.ndarray) -> float:
    """Mean temporal std of the differential (roll/pitch/yaw) control modes.

    This isolates *maneuvering intensity* — how much the differential modes move
    over time — from the static trim biases and the common (altitude) mode.
    """
    m = modes_from_rps(w)
    return float(np.mean([m[k].std() for k in (1, 2, 3)]))


def intermittency_stats(w: np.ndarray, dt: float) -> dict:
    """Maneuver-activity intermittency of the differential modes (see report §6).

    Returns the active fraction (frames whose 0.5 s rolling differential-mode std
    exceeds median + 3·MAD), the maneuver onset rate, and the mean cruise dwell.
    """
    from scipy.ndimage import uniform_filter1d

    w = inflight(w)
    m = modes_from_rps(w)
    win = max(3, int(0.5 / dt))
    diff = m[1:]
    lm = uniform_filter1d(diff, win, axis=1, mode="nearest")
    lv = uniform_filter1d((diff - lm) ** 2, win, axis=1, mode="nearest")
    activity = np.sqrt(np.clip(lv, 0.0, None)).mean(axis=0)
    med = float(np.median(activity))
    mad = float(np.median(np.abs(activity - med))) + 1e-9
    active = activity > med + 3.0 * 1.4826 * mad
    onsets = int(np.sum((~active[:-1]) & active[1:]))
    dur_s = w.shape[1] * dt
    return {
        "dur_s": round(dur_s, 1),
        "active_pct": round(100 * float(active.mean()), 1),
        "maneuver_rate_hz": round(onsets / dur_s, 3),
        "mean_hold_s": round((1.0 - active.mean()) * dur_s / max(onsets, 1), 1),
    }


# ---------------------------------------------------------------------------
# Raw DJI RC stick inputs (bursty-pilot evidence)
# ---------------------------------------------------------------------------
def load_rc_sticks(root: pathlib.Path) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Load raw RC stick inputs from the two DJI logs, keyed by recording id.

    Returns ``{id: (t, {axis: values})}`` against the logger clock
    (``Clock:offsetTime``).  No audio alignment — this is to show *structure*.
    """
    import pandas as pd

    from data_processing.michaels import MICHAELS_FILES

    axes = {
        "roll": "RC_Info:input_cur_roll:D",
        "pitch": "RC_Info:input_cur_pitch:D",
        "yaw": "RC_Info:input_cur_yaw:D",
        "throttle": "RC_Info:input_cur_throttle:D",
    }
    out: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for _wav_rel, csv_rel, _off, _dil in MICHAELS_FILES:
        path = root / csv_rel
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        t = np.asarray(pd.to_numeric(df["Clock:offsetTime"], errors="coerce"), dtype=np.float64)
        sticks = {
            name: np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=np.float64)
            for name, col in axes.items()
        }
        out[pathlib.Path(csv_rel).stem] = (t, sticks)
    return out


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------
def write_calibration_csv(real: list[tuple[str, np.ndarray, float]]) -> RPSSynthConfig:
    rows = []
    for label, w, dt in real:
        m = modes_from_rps(inflight(w))
        rec = {"recording": label, "dt_ms": round(dt * 1e3, 3)}
        for k, name in enumerate(MODE_NAMES):
            x = m[k] - m[k].mean()
            v = float(np.var(x))
            rho1 = float(np.mean(x[:-1] * x[1:]) / v) if v > 0 else 0.0
            rho1 = min(max(rho1, 1e-4), 1 - 1e-4)
            rec[f"{name}_mean"] = round(float(m[k].mean()), 3)
            rec[f"{name}_std"] = round(float(np.sqrt(v)), 3)
            rec[f"{name}_tau"] = round(dt / -np.log(rho1), 3)
        rows.append(rec)
    if rows:
        with open(ASSETS / "calibration.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    # Fit from DREGON only (high sample-rate, clean tau).
    dregon = [(w, dt) for label, w, dt in real if label.startswith("DREGON")]
    if dregon:
        return fit_config([w for w, _ in dregon], [dt for _, dt in dregon])
    return DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_rps(ax, t, w, title, ylim=None):
    for k in range(NUM_ROTORS):
        ax.plot(t, w[k], color=ROTOR_COLORS[k], lw=0.8, label=ROTOR_LABELS[k])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("RPS")
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)


def fig_traj_examples(cfg: RPSSynthConfig):
    dur = 10.0
    t = np.arange(int(dur * SYNTH_FS)) / SYNTH_FS
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, (name, agg) in zip(axes, AGG_LEVELS.items(), strict=True):
        w = generate(dur, SYNTH_FS, config=cfg, aggressiveness=agg, rng=2024)
        _plot_rps(ax, t, w, f"{name} (aggressiveness = {agg})", ylim=(50, 108))
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Synthetic RPS trajectories at three aggressiveness levels", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "traj_examples.png", dpi=150)
    plt.close(fig)


def fig_real_vs_synth(cfg: RPSSynthConfig, real):
    dur = 10.0
    real_pick = next((r for r in real if "nosource_room2" in r[0] and "free" in r[0]), None)
    if real_pick is None and real:
        real_pick = real[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    if real_pick is not None:
        label, w, dt = real_pick
        w = cruise(w, dt)
        n = min(int(dur / dt), w.shape[1])
        _plot_rps(axes[0], np.arange(n) * dt, w[:, :n], f"real — {label}", ylim=(60, 95))
    else:
        axes[0].set_title("real — (unavailable)")
    t = np.arange(int(dur * SYNTH_FS)) / SYNTH_FS
    ws = generate(dur, SYNTH_FS, config=cfg, aggressiveness=1.0, rng=7)
    _plot_rps(axes[1], t, ws, "synthetic — aggressiveness 1.0", ylim=(60, 95))
    axes[1].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.set_xlabel("time (s)")
    fig.suptitle("Real flight vs. calibrated synthetic trajectory", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "real_vs_synth.png", dpi=150)
    plt.close(fig)


def fig_mode_decomposition(cfg: RPSSynthConfig, real):
    dur = 12.0
    real_pick = next((r for r in real if "rectangle" in r[0]), real[0] if real else None)
    fig, axes = plt.subplots(NUM_ROTORS, 2, figsize=(11, 7), sharex="col")
    if real_pick is not None:
        label, w, dt = real_pick
        w = cruise(w, dt)
        n = min(int(dur / dt), w.shape[1])
        m = modes_from_rps(w[:, :n])
        tr = np.arange(n) * dt
    else:
        label = "(unavailable)"
        m, tr = None, None
    ws = generate(dur, SYNTH_FS, config=cfg, aggressiveness=1.0, rng=11)
    ms = modes_from_rps(ws)
    ts = np.arange(ws.shape[1]) / SYNTH_FS
    for k, name in enumerate(MODE_NAMES):
        if m is not None:
            axes[k, 0].plot(tr, m[k], color="#333", lw=0.8)
        axes[k, 0].set_ylabel(name)
        axes[k, 0].grid(alpha=0.3)
        axes[k, 1].plot(ts, ms[k], color="#c44", lw=0.8)
        axes[k, 1].grid(alpha=0.3)
    axes[0, 0].set_title(f"real modes — {label}", fontsize=10)
    axes[0, 1].set_title("synthetic modes", fontsize=10)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Control-mode decomposition (common / roll / pitch / yaw)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "mode_decomposition.png", dpi=150)
    plt.close(fig)


def fig_distributions(cfg: RPSSynthConfig, real):
    # Pool real DREGON rotor speeds (centred per recording to remove hover-level
    # offsets) against a long synthetic flight, plus correlation matrices.
    real_dregon = [inflight(w) for label, w, _ in real if label.startswith("DREGON")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    # (a) centred per-rotor RPS distribution
    if real_dregon:
        rc = np.concatenate([w - w.mean() for w in real_dregon], axis=1).ravel()
        axes[0].hist(rc, bins=80, density=True, alpha=0.55, label="real (DREGON)", color="#444")
    ws = generate(300.0, SYNTH_FS, config=cfg, aggressiveness=1.0, rng=3)
    axes[0].hist(
        (ws - ws.mean()).ravel(), bins=80, density=True, alpha=0.55, label="synthetic", color="#c44"
    )
    axes[0].set_title("centred rotor-speed distribution")
    axes[0].set_xlabel("RPS − mean")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    # (b)(c) correlation matrices
    for ax, mat, title in (
        (
            axes[1],
            np.corrcoef(np.concatenate(real_dregon, axis=1)) if real_dregon else None,
            "real rotor corr",
        ),
        (axes[2], np.corrcoef(ws), "synthetic rotor corr"),
    ):
        if mat is None:
            ax.set_title(f"{title} (unavailable)")
            continue
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(NUM_ROTORS), ROTOR_LABELS, rotation=45, fontsize=7)
        ax.set_yticks(range(NUM_ROTORS), ROTOR_LABELS, fontsize=7)
        for i in range(NUM_ROTORS):
            for j in range(NUM_ROTORS):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Marginal and joint rotor-speed statistics: real vs synthetic", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "distributions.png", dpi=150)
    plt.close(fig)


def fig_aggressiveness_sweep(cfg: RPSSynthConfig, real):
    aggs = np.linspace(0.0, 3.5, 15)
    activity = []
    for a in aggs:
        w = generate(120.0, SYNTH_FS, config=cfg, aggressiveness=float(a), rng=99)
        activity.append(maneuver_activity(w))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(aggs, activity, "-o", color="#c44", label="synthetic")
    # overlay each real flight's maneuver activity as a labelled horizontal line
    for label, w, dt in real:
        s = maneuver_activity(cruise(w, dt))
        short = label.split("/")[-1].replace("_nosource_room2", "").replace("_nosource_room1", "")
        ax.axhline(s, ls="--", lw=0.8, alpha=0.6)
        ax.text(3.52, s, short, fontsize=6, va="center")
    ax.set_xlabel("aggressiveness")
    ax.set_ylabel("maneuver activity\n(mean roll/pitch/yaw temporal std)")
    ax.set_title("Aggressiveness knob vs. real flight types")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "aggressiveness_sweep.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Intermittent-model figures + intermittency table
# ---------------------------------------------------------------------------
def write_intermittency_csv(real: list[tuple[str, np.ndarray, float]]) -> None:
    rows = []
    for label, w, dt in real:
        rec = {"recording": label, **intermittency_stats(w, dt)}
        rows.append(rec)
    if rows:
        with open(ASSETS / "intermittency.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)


def fig_rc_sticks(root: pathlib.Path | None) -> None:
    if root is None:
        return
    sticks = load_rc_sticks(root)
    if not sticks:
        return
    fig, axes = plt.subplots(len(sticks), 1, figsize=(11, 2.6 * len(sticks)), squeeze=False)
    for ax, (rid, (t, axesvals)) in zip(axes[:, 0], sticks.items(), strict=True):
        for name, v in axesvals.items():
            ax.plot(t, v, lw=0.6, label=name)
        ax.set_title(f"{rid}: raw RC stick inputs (normalised pilot commands)", fontsize=10)
        ax.set_ylabel("stick")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("logger time (s)")
    fig.suptitle(
        "Pilot commands are intermittent: mostly centred, with brief deflections", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "rc_sticks.png", dpi=140)
    plt.close(fig)


def fig_model_comparison(real) -> None:
    from data_processing.rps_synthesis import DREGON_PROFILE, generate_intermittent

    dur = 20.0
    pick = next((r for r in real if "rectangle" in r[0]), real[0] if real else None)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    if pick is not None:
        label, w, dt = pick
        w = cruise(w, dt)
        n = min(int(dur / dt), w.shape[1])
        _plot_rps(
            axes[0], np.arange(n) * dt, w[:, :n], f"real — {label} (cruise window)", ylim=(60, 95)
        )
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    t = np.arange(int(dur * SYNTH_FS)) / SYNTH_FS
    _plot_rps(
        axes[1],
        t,
        generate(dur, SYNTH_FS, aggressiveness=1.0, rng=4),
        "continuous OU — wanders at all times",
        ylim=(60, 95),
    )
    _plot_rps(
        axes[2],
        t,
        generate_intermittent(dur, SYNTH_FS, profile=DREGON_PROFILE, rng=4),
        "intermittent (pilot + airframe) — steady, with brief maneuvers",
        ylim=(60, 95),
    )
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Real vs. continuous-OU vs. intermittent RPS trajectories", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_comparison.png", dpi=140)
    plt.close(fig)


def fig_intermittent_aggressiveness() -> None:
    from data_processing.rps_synthesis import DREGON_PROFILE, generate_intermittent

    dur = 25.0
    t = np.arange(int(dur * SYNTH_FS)) / SYNTH_FS
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, (name, agg) in zip(axes, AGG_LEVELS.items(), strict=True):
        w = generate_intermittent(
            dur, SYNTH_FS, profile=DREGON_PROFILE, aggressiveness=agg, rng=2024
        )
        _plot_rps(ax, t, w, f"{name} (aggressiveness = {agg})", ylim=(50, 105))
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Intermittent trajectories at three aggressiveness levels", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "intermittent_agg.png", dpi=140)
    plt.close(fig)


def fig_drone_profile_sweep() -> None:
    from data_processing.rps_synthesis import blend_profiles, generate_intermittent

    dur = 25.0
    t = np.arange(int(dur * SYNTH_FS)) / SYNTH_FS
    levels = [
        (0.0, "DREGON-like (small, fast — short motor_tau)"),
        (0.5, "in-between airframe"),
        (1.0, "Michael's-like (DJI M100 — slow, long motor_tau)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, (t_blend, title) in zip(axes, levels, strict=True):
        prof = blend_profiles(t=t_blend)
        w = generate_intermittent(dur, SYNTH_FS, profile=prof, rng=2024)
        _plot_rps(ax, t, w, f"drone_profile = {t_blend} — {title}", ylim=(55, 95))
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Drone-dynamics knob: DREGON → in-between → Michael's", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "drone_profile_sweep.png", dpi=140)
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    real, root = load_real_traces()
    print(f"Loaded {len(real)} real recordings from {root}")
    cfg = write_calibration_csv(real)
    write_intermittency_csv(real)
    print("Intermittency (real):")
    for label, w, dt in real:
        print(f"  {label:42s} {intermittency_stats(w, dt)}")
    print("Fitted OU config:")
    for name, p in zip(MODE_NAMES, cfg.modes, strict=True):
        print(f"  {name:7s} mean={p.mean:7.3f} std={p.std:6.3f} tau={p.tau:5.3f}")
    # Intermittent model (the recommended generator) + drone-dynamics knob
    fig_rc_sticks(root)
    fig_model_comparison(real)
    fig_intermittent_aggressiveness()
    fig_drone_profile_sweep()
    # OU model (continuous baseline) figures
    fig_traj_examples(cfg)
    fig_real_vs_synth(cfg, real)
    fig_mode_decomposition(cfg, real)
    fig_distributions(cfg, real)
    fig_aggressiveness_sweep(cfg, real)
    print("Figures written to", ASSETS)


if __name__ == "__main__":
    main()
