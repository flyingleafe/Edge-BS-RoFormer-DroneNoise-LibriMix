"""Validate the coupled Vold-Kalman tracker on the DREGON natural experiment.

Mirrors ``scripts/rps_refinement_validation.py`` (the predecessor evaluation of
``rps_refinement`` stages B+C / D) for the new tracker
(``data_processing.vk_tracking``, design ``docs/vk-order-tracking-design.md``
§5.1). The 5 DREGON ``free-flight_*_room1`` recordings carry BOTH
``motors_command`` (the init training uses) and ``motors_measured`` (actual
rotor speeds = ground truth):

* **Main run** — init = ``clean_command_spikes``-cleaned command interpolated
  to a 32 ms frame grid (tau-shifted onto the audio clock, stage-A
  ``estimate_clock_offset`` exactly as the predecessor); ``vk_track`` on all
  8 channels at 16 kHz; truth = measured, raw and 0.25 s-smoothed (the
  predecessor's ``SMOOTH_FRAMES`` boxcar). Metrics: unsigned error + signed
  bias, pooled / per recording / per rotor, for both the *unrefined* command
  init (sanity: predecessor pooled 0.633 / -0.057 on full in-flight windows)
  and the VK-refined trajectories. Twin-pair diagnostics are explicit:
  DREGON's rotors fly as 2 tight pairs (~0.35-0.9 rev/s apart) and the
  predecessor's B+C failure was -0.44 pooled bias from pair-mean capture —
  per-rotor bias and pair-separation change reveal any recurrence.
* **Trajectory-bandwidth sweep** — ``traj_lambda`` over 5 log-spaced values on
  2 recordings; expect a U-curve (small lambda tracks noise, large lambda
  cannot follow), report where it bottoms.
* **Capture basin** — one recording, init perturbed by constant offsets
  0 / +-0.5 / +-1 / +-2 / +-3 rev/s (all rotors), annealed ``grow`` schedule;
  report final error per offset and the basin edge.

Runtime note: ``vk_track`` costs ~10-20 s per second of 8-channel audio per
(n_outer x k_max/40), so each recording is evaluated on ONE 25 s
**mid-recording** segment of its in-flight window (takeoff/landing trimmed via
the same command+measured median>30 mask as the predecessor) rather than the
predecessor's full ~50-60 s tiling — the command-init numbers reported here
are recomputed on the same grid so the comparison stays internally consistent.
Metrics exclude 0.5 s at each segment edge (zero-phase filter + D2-prior
transients, same convention as ``tests/test_vk_tracking.py``).

Success gates (design §5.1): pooled |bias| <= 0.1 AND pooled unsigned err
<= command's (refinement must not damage the labels).

Artifacts (``results/vk_tracking/validation/``):
  * ``dregon_<recording>.npz`` — frame grid, command init, measured (raw +
    smoothed), VK-refined trajectories, confidence, residual ratios, tau.
  * ``summary.json`` — all numbers: per-recording / pooled / per-rotor tables,
    twin-pair diagnostics, lambda sweep, capture basin, wall-clocks, gates.
  * ``preview_vk.png`` (refined vs command vs measured + twin-pair zoom +
    error vs time), ``lambda_ucurve.png``, ``capture_basin.png``.

Run: ``.venv/bin/python scripts/vk_validation.py``
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import: the work is parallelised at the
# process level (one worker per vk_track call), same rationale as the
# predecessor script.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data_processing.dregon import (  # noqa: E402
    clean_command_spikes,
    discover_recordings,
    get_geometry,
    load_timeframe,
)
from data_processing.rps_refinement import (  # noqa: E402
    RefineConfig,
    compute_logmag,
    estimate_clock_offset,
)
from data_processing.vk_tracking import VKConfig, vk_track  # noqa: E402

SR = 16000
SEG_LEN_S = 25.0  # ONE mid-recording segment per recording (runtime bound)
DREGON_MIN_RPS = 30.0  # in-flight mask threshold (as predecessor)
FRAME_HOP_S = 0.032  # evaluation grid (predecessor's STFT hop)
EDGE_TRIM_S = 0.5  # metric exclusion at segment edges (filter transients)
SMOOTH_FRAMES = 8  # 0.25 s boxcar on the frame grid (as predecessor)
OUT_DIR = Path("results/vk_tracking/validation")

# The 5 DREGON recordings that carry motors_measured (ground truth).
DREGON_TARGETS = [
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_speech-high_room1",
    "free-flight_whitenoise-low_room1",
    "free-flight_whitenoise-high_room1",
]
PREVIEW_RID = "free-flight_nosource_room1"  # cleanest harmonics for figures

# Main-run tracker config: REFINE mode, not the annealed capture schedule.
# Pre-run diagnosis on a 10 s nosource segment showed the "grow" schedule's
# wide-bandwidth early rounds drive a systematic downward drift on real DREGON
# audio (pooled bias -0.43 at n_outer=12, all four rotors together — the same
# structural trap as the predecessor's stage B+C), and k_min=1 keeps the
# twin-merged low harmonics in the Fisher fusion. Stage D (the only unbiased
# predecessor) used narrow fixed bands and k_min=6. Hence: k_schedule="fixed"
# (which also disables the bandwidth annealing), narrow bw, k_min raised,
# max_step tightened. couple_hz=20 keeps same-rotor adjacent harmonics
# (~75-90 Hz apart) uncoupled while twin-pair harmonics couple (test-suite
# convention). Annealing remains in use ONLY for the capture-basin experiment
# (BASIN_CFG), which is what it exists for.
MAIN_CFG = VKConfig(
    fs=float(SR),
    couple_hz=20.0,
    n_outer=5,
    k_min=6,
    k_max=30,
    k_schedule="fixed",
    bw_hz=1.5,
    max_step=0.3,
)
BASIN_CFG = replace(MAIN_CFG, k_schedule="grow", n_outer=12)
MAIN_CHANNELS = 8  # headline numbers use all 8 mics
AUX_CHANNELS = 4  # lambda sweep + capture basin: 4 mics (runtime bound)

# Sweep / basin protocol (design §5.1).
LAMBDA_GRID = [1e2, 1e3, 1e4, 1e5, 1e6]
SWEEP_TARGETS = ["free-flight_nosource_room1", "free-flight_speech-low_room1"]
BASIN_RID = "free-flight_nosource_room1"
BASIN_OFFSETS = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
N_WORKERS = 5

# Predecessor pooled numbers (full in-flight windows) for the report table.
PREDECESSOR = {
    "command": {"err": 0.633, "bias": -0.057},
    "smoothed_measured_ref": {"err": 0.484, "bias": 0.017},
    "stage_bc": {"err": 0.848, "bias": -0.440},
    "stage_d": {"err": 0.638, "bias": -0.078},
}


@dataclass
class Prepared:
    """One recording's 25 s mid-flight evaluation segment on the frame grid."""

    rid: str
    tau: float
    seg_lo: float
    seg_hi: float
    audio: np.ndarray  # (8, T) segment audio at SR
    ft: np.ndarray  # (N,) seconds, segment-relative
    r_init: np.ndarray  # (4, N) cleaned command at ft + tau
    r_meas: np.ndarray  # (4, N) measured at ft + tau
    r_meas_sm: np.ndarray  # (4, N) 0.25 s-smoothed measured
    edge: np.ndarray  # (N,) bool metric mask (edges trimmed)


def smooth_frames(x: np.ndarray, win: int = SMOOTH_FRAMES) -> np.ndarray:
    """Per-rotor moving average along the frame axis (predecessor convention)."""
    ker = np.ones(win) / win
    return np.stack([np.convolve(row, ker, mode="same") for row in x])


def prepare_recording(
    rid: str, seg_len: float = SEG_LEN_S, dregon_dir: str | Path = Path("data") / "DREGON"
) -> Prepared:
    """Load a recording, estimate tau, cut the mid-flight evaluation segment.

    ``dregon_dir`` accepts a plain path or a ``dload:DREGON`` URI (resolved via
    ``data_processing.streams.resolve_source``) so remote jobs without a
    ``data/`` checkout can stream the dataset.
    """
    from data_processing.streams import resolve_source

    dregon_dir = resolve_source(dregon_dir)
    by_id = {s["recording_id"]: s for s in discover_recordings(dregon_dir)}
    frame = load_timeframe(by_id[rid], geometry=get_geometry(dregon_dir), target_sr=SR)
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

    # Stage A clock offset from the cleaned command, channel 0, first 30 s
    # of the in-flight window — identical to the predecessor.
    cfg_r = RefineConfig()
    spec0 = compute_logmag(audio[:1, int(t_lo * SR) : int(min(t_lo + 30.0, t_hi) * SR)], cfg_r)
    tau, _, _ = estimate_clock_offset(spec0, mt - t_lo, command_clean, cfg_r)

    mid = 0.5 * (t_lo + t_hi)
    seg_lo = max(t_lo, mid - seg_len / 2.0)
    seg_hi = min(t_hi, seg_lo + seg_len)
    a0, a1 = int(round(seg_lo * SR)), int(round(seg_hi * SR))
    seg = audio[:, a0:a1]
    ft = np.arange(0.0, (a1 - a0) / SR - FRAME_HOP_S / 2, FRAME_HOP_S)
    mt_rel = mt - seg_lo
    r_init = np.stack([np.interp(ft + tau, mt_rel, command_clean[i]) for i in range(4)])
    r_meas = np.stack([np.interp(ft + tau, mt_rel, measured[i]) for i in range(4)])
    edge = (ft > EDGE_TRIM_S) & (ft < ft[-1] - EDGE_TRIM_S)
    return Prepared(
        rid=rid,
        tau=float(tau),
        seg_lo=seg_lo,
        seg_hi=seg_hi,
        audio=seg,
        ft=ft,
        r_init=r_init,
        r_meas=r_meas,
        r_meas_sm=smooth_frames(r_meas),
        edge=edge,
    )


def traj_stats(traj: np.ndarray, prep: Prepared) -> dict[str, Any]:
    """Unsigned err + signed bias vs raw and smoothed measured, per rotor + pooled."""
    e = prep.edge
    d = (traj - prep.r_meas)[:, e]
    d_sm = (traj - prep.r_meas_sm)[:, e]
    return {
        "err": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "err_sm": float(np.mean(np.abs(d_sm))),
        "bias_sm": float(np.mean(d_sm)),
        "err_rotor": [float(v) for v in np.mean(np.abs(d), axis=1)],
        "bias_rotor": [float(v) for v in np.mean(d, axis=1)],
    }


def pair_diagnostics(traj: np.ndarray, prep: Prepared) -> list[dict[str, Any]]:
    """Twin-pair capture diagnostics: pair rotors by mean measured speed.

    For each pair report the measured separation, the refined separation, and
    the per-rotor signed bias — pair-mean capture (the B+C failure) shows up
    as the refined separation collapsing toward 0 (upper rotor biased down,
    lower rotor biased up).
    """
    e = prep.edge
    means = prep.r_meas[:, e].mean(axis=1)
    order = np.argsort(means)[::-1]  # descending speed: pair = (0,1) and (2,3)
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]
    out = []
    for hi, lo in pairs:
        sep_meas = float(np.mean(prep.r_meas[hi, e] - prep.r_meas[lo, e]))
        sep_traj = float(np.mean(traj[hi, e] - traj[lo, e]))
        out.append(
            {
                "rotors": [hi, lo],
                "sep_measured": sep_meas,
                "sep_refined": sep_traj,
                "bias_hi": float(np.mean((traj - prep.r_meas)[hi, e])),
                "bias_lo": float(np.mean((traj - prep.r_meas)[lo, e])),
            }
        )
    return out


def run_recording(rid: str, cfg: VKConfig) -> dict[str, Any]:
    """Worker: prepare one recording, run vk_track, save NPZ, return stats+diffs."""
    prep = prepare_recording(rid)
    tic = time.perf_counter()
    res = vk_track(prep.audio, prep.r_init, prep.ft, cfg)
    wall = time.perf_counter() - tic
    print(
        f"[{rid}] vk_track {wall:.0f}s  resid {res.residual_ratios[0]:.3f}"
        f"->{res.residual_ratios[-1]:.3f}  max_delta_last {res.max_deltas[-1]:.3f}",
        flush=True,
    )
    np.savez(
        OUT_DIR / f"dregon_{rid}.npz",
        frame_times=prep.seg_lo + prep.ft,
        command=prep.r_init,
        measured=prep.r_meas,
        measured_smoothed=prep.r_meas_sm,
        refined=res.r_refined,
        confidence=res.confidence,
        conf_times=prep.seg_lo + res.conf_times,
        residual_ratios=np.array(res.residual_ratios),
        max_deltas=np.array(res.max_deltas),
        tau=prep.tau,
        seg_bounds=np.array([prep.seg_lo, prep.seg_hi]),
        edge_mask=prep.edge,
    )
    e = prep.edge
    return {
        "recording": rid,
        "tau": round(prep.tau, 4),
        "segment": [round(prep.seg_lo, 2), round(prep.seg_hi, 2)],
        "wall_s": round(wall, 1),
        "residual_ratios": [round(r, 4) for r in res.residual_ratios],
        "max_deltas": [round(d, 4) for d in res.max_deltas],
        "mean_confidence": float(res.confidence.mean()),
        "command": traj_stats(prep.r_init, prep),
        "vk": traj_stats(res.r_refined, prep),
        "smoothed_measured_ref": traj_stats(prep.r_meas_sm, prep),
        "pairs_command": pair_diagnostics(prep.r_init, prep),
        "pairs_vk": pair_diagnostics(res.r_refined, prep),
        # signed diff arrays for exact pooling (not JSON-serialised)
        "_diff_cmd": (prep.r_init - prep.r_meas)[:, e],
        "_diff_cmd_sm": (prep.r_init - prep.r_meas_sm)[:, e],
        "_diff_vk": (res.r_refined - prep.r_meas)[:, e],
        "_diff_vk_sm": (res.r_refined - prep.r_meas_sm)[:, e],
        "_diff_ref_sm": (prep.r_meas_sm - prep.r_meas)[:, e],
    }


def run_lambda_point(rid: str, lam: float, cfg: VKConfig) -> dict[str, Any]:
    """Worker: one traj_lambda sweep point on one recording."""
    prep = prepare_recording(rid)
    tic = time.perf_counter()
    res = vk_track(prep.audio[:AUX_CHANNELS], prep.r_init, prep.ft, replace(cfg, traj_lambda=lam))
    wall = time.perf_counter() - tic
    st = traj_stats(res.r_refined, prep)
    print(
        f"[sweep {rid} lambda={lam:g}] err {st['err']:.3f} bias {st['bias']:+.3f} ({wall:.0f}s)",
        flush=True,
    )
    return {"recording": rid, "traj_lambda": lam, "wall_s": round(wall, 1), **st}


def run_basin_point(rid: str, offset: float, cfg: VKConfig) -> dict[str, Any]:
    """Worker: one capture-basin point (constant init offset on all rotors)."""
    prep = prepare_recording(rid)
    tic = time.perf_counter()
    res = vk_track(prep.audio[:AUX_CHANNELS], prep.r_init + offset, prep.ft, cfg)
    wall = time.perf_counter() - tic
    st = traj_stats(res.r_refined, prep)
    init_st = traj_stats(prep.r_init + offset, prep)
    print(
        f"[basin {offset:+.1f}] err {st['err']:.3f} bias {st['bias']:+.3f} ({wall:.0f}s)",
        flush=True,
    )
    return {
        "offset": offset,
        "wall_s": round(wall, 1),
        "init_err": init_st["err"],
        **{k: st[k] for k in ("err", "bias", "err_sm", "bias_sm", "bias_rotor")},
    }


def pooled_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the signed diff arrays over recordings (exact, frame-weighted)."""
    out: dict[str, Any] = {}
    for name, raw_key, sm_key in (
        ("command", "_diff_cmd", "_diff_cmd_sm"),
        ("vk", "_diff_vk", "_diff_vk_sm"),
        ("smoothed_measured_ref", "_diff_ref_sm", None),
    ):
        d = np.concatenate([r[raw_key] for r in rows], axis=1)
        entry = {
            "err": float(np.mean(np.abs(d))),
            "bias": float(np.mean(d)),
            "err_rotor": [float(v) for v in np.mean(np.abs(d), axis=1)],
            "bias_rotor": [float(v) for v in np.mean(d, axis=1)],
        }
        if sm_key is not None:
            d_sm = np.concatenate([r[sm_key] for r in rows], axis=1)
            entry["err_sm"] = float(np.mean(np.abs(d_sm)))
            entry["bias_sm"] = float(np.mean(d_sm))
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Figures


def make_preview(rid: str) -> None:
    npz = np.load(OUT_DIR / f"dregon_{rid}.npz")
    ft = npz["frame_times"]
    cmd, meas, meas_sm, vk = (
        npz["command"],
        npz["measured"],
        npz["measured_smoothed"],
        npz["refined"],
    )

    z0 = float(ft[0]) + 5.0
    zoom = (ft >= z0) & (ft <= z0 + 10.0)
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))

    ax[0].plot(ft[zoom], meas[0][zoom], "k-", lw=1.4, label="measured (GT)")
    ax[0].plot(
        ft[zoom],
        meas_sm[0][zoom],
        "-",
        color="tab:green",
        lw=1.0,
        label="measured (0.25 s smoothed)",
    )
    ax[0].plot(ft[zoom], cmd[0][zoom], "--", color="tab:gray", lw=1.2, label="command (init)")
    ax[0].plot(ft[zoom], vk[0][zoom], color="tab:red", lw=1.4, label="VK refined")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("rev/s")
    ax[0].set_title(f"{rid}: rotor 0 trajectory (10 s zoom)")
    ax[0].legend()

    # Twin-pair zoom: the two fastest rotors (a tight pair) — pair-mean
    # capture would show both VK tracks collapsing onto one line.
    order = np.argsort(meas.mean(axis=1))[::-1]
    hi, lo = int(order[0]), int(order[1])
    for r, ls in ((hi, "-"), (lo, "--")):
        ax[1].plot(ft[zoom], meas[r][zoom], "k" + ls, lw=1.0, label=f"measured r{r}")
        ax[1].plot(ft[zoom], vk[r][zoom], color="tab:red", ls=ls, lw=1.4, label=f"VK r{r}")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("rev/s")
    ax[1].set_title(f"twin pair (rotors {hi}/{lo}): VK vs measured")
    ax[1].legend(ncol=2)

    for traj, color, name in ((cmd, "tab:gray", "command"), (vk, "tab:red", "VK")):
        err = np.abs(traj - meas).mean(axis=0)
        ax[2].plot(ft, err, color=color, lw=0.9, alpha=0.85, label=f"|{name} - measured|")
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("mean |error| (rev/s)")
    ax[2].set_title("rotor-averaged error vs time")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview_vk.png", dpi=150)
    plt.close(fig)


def make_ucurve(sweep_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for rid in SWEEP_TARGETS:
        pts = sorted(
            (r for r in sweep_rows if r["recording"] == rid), key=lambda r: r["traj_lambda"]
        )
        lams = [p["traj_lambda"] for p in pts]
        ax.plot(lams, [p["err"] for p in pts], "o-", label=f"{rid} (vs measured)")
        ax.plot(lams, [p["err_sm"] for p in pts], "s--", alpha=0.6, label=f"{rid} (vs smoothed)")
    ax.set_xscale("log")
    ax.set_xlabel("traj_lambda")
    ax.set_ylabel("pooled unsigned error (rev/s)")
    ax.set_title("trajectory-smoothness U-curve")
    ax.axvline(MAIN_CFG.traj_lambda, color="k", ls=":", lw=1, label="default")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lambda_ucurve.png", dpi=150)
    plt.close(fig)


def make_basin_plot(basin_rows: list[dict[str, Any]]) -> None:
    pts = sorted(basin_rows, key=lambda r: r["offset"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([p["offset"] for p in pts], [p["err"] for p in pts], "o-", label="VK final err")
    ax.plot(
        [p["offset"] for p in pts],
        [p["init_err"] for p in pts],
        "s--",
        color="tab:gray",
        label="perturbed init err",
    )
    ax.set_xlabel("init offset (rev/s, all rotors)")
    ax.set_ylabel("pooled unsigned error vs measured (rev/s)")
    ax.set_title(f"capture basin ({BASIN_RID})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capture_basin.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------


def print_table(rows: list[dict[str, Any]], pooled: dict[str, Any]) -> None:
    width = 110
    print("\n" + "=" * width)
    print("DREGON VK validation: command init vs VK-refined vs measured GT (rev/s)")
    print("=" * width)
    print(
        f"{'recording':<36}{'tau':>7}{'err_cmd':>9}{'err_vk':>8}"
        f"{'bias_cmd':>10}{'bias_vk':>9}{'err_sm_vk':>10}{'bias_sm_vk':>11}{'wall_s':>8}"
    )
    print("-" * width)
    for r in rows:
        print(
            f"{r['recording']:<36}{r['tau']:>7.3f}"
            f"{r['command']['err']:>9.3f}{r['vk']['err']:>8.3f}"
            f"{r['command']['bias']:>10.3f}{r['vk']['bias']:>9.3f}"
            f"{r['vk']['err_sm']:>10.3f}{r['vk']['bias_sm']:>11.3f}{r['wall_s']:>8.0f}"
        )
    print("-" * width)
    p = pooled
    print(
        f"{'POOLED':<36}{'':>7}{p['command']['err']:>9.3f}{p['vk']['err']:>8.3f}"
        f"{p['command']['bias']:>10.3f}{p['vk']['bias']:>9.3f}"
        f"{p['vk']['err_sm']:>10.3f}{p['vk']['bias_sm']:>11.3f}"
    )
    print(f"pooled per-rotor VK bias: {[round(b, 3) for b in p['vk']['bias_rotor']]}")
    print(f"pooled per-rotor cmd bias: {[round(b, 3) for b in p['command']['bias_rotor']]}")
    print(
        "predecessor (full windows): command 0.633/-0.057, smoothed-meas ref 0.484/+0.017, "
        "B+C 0.848/-0.440, stage D 0.638/-0.078"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
        main_futs = {rid: pool.submit(run_recording, rid, MAIN_CFG) for rid in DREGON_TARGETS}
        sweep_futs = [
            pool.submit(run_lambda_point, rid, lam, MAIN_CFG)
            for rid in SWEEP_TARGETS
            for lam in LAMBDA_GRID
        ]
        basin_futs = [
            pool.submit(run_basin_point, BASIN_RID, off, BASIN_CFG) for off in BASIN_OFFSETS
        ]
        rows = [main_futs[rid].result() for rid in DREGON_TARGETS]
        sweep_rows = [f.result() for f in sweep_futs]
        basin_rows = [f.result() for f in basin_futs]

    pooled = pooled_stats(rows)

    # Gates (design §5.1): must not damage the labels.
    gate_bias = abs(pooled["vk"]["bias"]) <= 0.1
    gate_err = pooled["vk"]["err"] <= pooled["command"]["err"]
    print_table(rows, pooled)
    print(
        f"\nGATES: |pooled bias| <= 0.1: {'PASS' if gate_bias else 'FAIL'} "
        f"({pooled['vk']['bias']:+.3f}); "
        f"pooled err <= command ({pooled['command']['err']:.3f}): "
        f"{'PASS' if gate_err else 'FAIL'} ({pooled['vk']['err']:.3f})"
    )

    # U-curve bottom per sweep recording.
    ucurve_bottom = {}
    for rid in SWEEP_TARGETS:
        pts = [r for r in sweep_rows if r["recording"] == rid]
        best = min(pts, key=lambda r: r["err"])
        ucurve_bottom[rid] = {"traj_lambda": best["traj_lambda"], "err": best["err"]}
        print(f"U-curve bottom [{rid}]: lambda={best['traj_lambda']:g} err={best['err']:.3f}")

    # Basin edge: largest |offset| whose final err stays within 1.2x of the
    # unperturbed (offset 0) run's err.
    base_err = next(r["err"] for r in basin_rows if r["offset"] == 0.0)
    recovered = [r for r in basin_rows if r["err"] <= 1.2 * base_err]
    basin_edge = max(abs(r["offset"]) for r in recovered)
    print(f"capture basin edge: +-{basin_edge:g} rev/s (recovered within 1.2x of unperturbed err)")

    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}  # noqa: E731
    summary = {
        "config": asdict(MAIN_CFG),
        "basin_config": asdict(BASIN_CFG),
        "protocol": {
            "segment_len_s": SEG_LEN_S,
            "segment_placement": "mid in-flight window",
            "frame_hop_s": FRAME_HOP_S,
            "edge_trim_s": EDGE_TRIM_S,
            "smooth_frames": SMOOTH_FRAMES,
            "channels_main": MAIN_CHANNELS,
            "channels_sweep_basin": AUX_CHANNELS,
            "note": "25 s mid-recording segments, NOT the predecessor's full-window tiling; "
            "refine-mode config chosen after the grow-schedule downward-drift diagnosis "
            "(see MAIN_CFG comment)",
        },
        "predecessor_pooled_full_window": PREDECESSOR,
        "recordings": [strip(r) for r in rows],
        "pooled": pooled,
        "gates": {
            "abs_bias_le_0.1": gate_bias,
            "err_le_command": gate_err,
            "pooled_vk_bias": pooled["vk"]["bias"],
            "pooled_vk_err": pooled["vk"]["err"],
            "pooled_command_err": pooled["command"]["err"],
        },
        "lambda_sweep": sweep_rows,
        "ucurve_bottom": ucurve_bottom,
        "capture_basin": basin_rows,
        "basin_edge_revs": basin_edge,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_preview(PREVIEW_RID)
    make_ucurve(sweep_rows)
    make_basin_plot(basin_rows)
    print(f"\nArtifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
