#!/usr/bin/env python3
"""Operating-envelope measurement for the RPS refinement stack.

Measures three properties of ``data_processing.rps_refinement`` on a controlled
synthetic 4-rotor comb testbed:

1. **Basin of attraction** — how large a constant init offset the coarse+spline
   refinement can still pull back to the truth, and where the cliff sits
   relative to ``RefineConfig.delta_max``.
2. **Non-harmonic robustness** — refined error and comb-confidence as harmonic
   SNR drops under white / pink / speech-like noise; plus a confidence-vs-error
   scatter (across *both* sweeps) that validates ``comb_confidence`` as an
   acceptance gate.
3. **Efficiency** — wall-clock per stage (``compute_logmag`` / ``coarse_delta``
   / ``refine_trajectories`` / ``harmonic_lsq_residual``) as seconds per
   audio-second, over duration x channel grids.

CPU-only. Artifacts (CSVs + PNGs) land in ``results/rps_refinement/robustness/``.
Run: ``python scripts/rps_refinement_robustness.py``.
"""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import butter, filtfilt  # noqa: E402

from data_processing.rps_refinement import (  # noqa: E402
    RefineConfig,
    coarse_delta,
    compute_logmag,
    harmonic_lsq_residual,
    refine_trajectories,
)

# --------------------------------------------------------------------------- #
# Testbed constants
# --------------------------------------------------------------------------- #
SR = 16000
N_ROTORS = 4
BASE_RPS = 75.0
TRIMS = np.array([1.3, -0.8, 2.5, 0.0])  # realistic per-rotor offsets (rev/s)
WOBBLE_HZ = 0.3  # slow common-mode RPS oscillation
WOBBLE_AMP = 1.5  # rev/s
COMB_K = 40  # synthesised harmonics
SUCCESS_TOL = 0.15  # rev/s; |refined - truth| threshold for "locked"

RESULTS_DIR = Path("results/rps_refinement/robustness")


# --------------------------------------------------------------------------- #
# Synthetic fixture
# --------------------------------------------------------------------------- #
def rotor_trajectories(times: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Ground-truth ``(R, len(times))`` rev/s at the given time stamps."""
    out = np.empty((N_ROTORS, times.size))
    for i in range(N_ROTORS):
        out[i] = (
            BASE_RPS + TRIMS[i] + WOBBLE_AMP * np.sin(2 * np.pi * WOBBLE_HZ * times + phases[i])
        )
    return out


def synth_comb(r_samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sum-of-harmonics comb audio from per-sample rev/s ``(R, T)``.

    Component ``k`` of rotor ``i`` has instantaneous frequency ``k * r_i(t)``,
    amplitude ``0.5 / k``, and a random phase.
    """
    n = r_samples.shape[1]
    x = np.zeros(n, dtype=np.float64)
    for i in range(N_ROTORS):
        phase = 2 * np.pi * np.cumsum(r_samples[i]) / SR
        for k in range(1, COMB_K + 1):
            x += (0.5 / k) * np.sin(k * phase + rng.uniform(0.0, 2 * np.pi))
    return x


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise by FFT magnitude shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    spec = spec / np.sqrt(f)
    return np.fft.irfft(spec, n=n)


def speech_like_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """White noise band-passed to 300-3000 Hz with 4 Hz amplitude modulation."""
    white = rng.standard_normal(n)
    b, a = cast(
        "tuple[np.ndarray, np.ndarray]",
        butter(4, [300.0 / (SR / 2), 3000.0 / (SR / 2)], btype="band"),
    )
    filt = filtfilt(b, a, white)
    t = np.arange(n) / SR
    am = 1.0 + 0.8 * np.sin(2 * np.pi * 4.0 * t)
    return filt * am


NOISE_FNS = {
    "white": white_noise,
    "pink": pink_noise,
    "speech": speech_like_noise,
}


def build_fixture(
    duration_s: float,
    snr_db: float,
    seed: int,
    *,
    noise_type: str = "white",
    n_channels: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(audio, r_truth_frames, frame_times)``.

    ``audio`` is ``(T,)`` for mono or ``(C, T)`` for multichannel (independent
    noise per channel; shared comb). ``r_truth_frames`` is ``(R, N)`` on the
    STFT frame grid, ``frame_times`` is ``(N,)``.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2 * np.pi, size=N_ROTORS)
    n = int(round(duration_s * SR))
    t = np.arange(n) / SR
    r_samples = rotor_trajectories(t, phases)
    comb = synth_comb(r_samples, rng)
    comb_energy = float(np.sum(comb**2))
    scale = np.sqrt(comb_energy / (10.0 ** (snr_db / 10.0)))

    def one_channel() -> np.ndarray:
        noise = NOISE_FNS[noise_type](n, rng)
        noise = noise / np.sqrt(max(np.sum(noise**2), 1e-12)) * scale
        return comb + noise

    if n_channels == 1:
        audio = one_channel()
    else:
        audio = np.stack([one_channel() for _ in range(n_channels)])

    cfg = RefineConfig()
    frame_times = np.arange(0, n, cfg.hop_length)[: 1 + (n - 1) // cfg.hop_length]
    # frame grid must match compute_logmag: use it directly for exactness.
    spec = compute_logmag(audio, cfg)
    frame_times = spec.frame_times
    r_truth_frames = rotor_trajectories(frame_times, phases)
    return audio, r_truth_frames, frame_times


# --------------------------------------------------------------------------- #
# Trial driver
# --------------------------------------------------------------------------- #
def run_trial(
    audio: np.ndarray,
    r_truth: np.ndarray,
    r_init: np.ndarray,
    cfg: RefineConfig,
) -> tuple[float, float]:
    """Refine ``r_init`` and return ``(mean_abs_error, mean_confidence)``."""
    spec = compute_logmag(audio, cfg)
    result = refine_trajectories(spec, r_init, cfg)
    err = float(np.mean(np.abs(result.r_refined - r_truth)))
    conf = float(np.mean(result.confidence))
    return err, conf


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Sweep 1 — basin of attraction
# --------------------------------------------------------------------------- #
OFFSETS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
SIGN_MODES = {
    "same": np.array([1.0, 1.0, 1.0, 1.0]),
    "opposite": np.array([1.0, -1.0, 1.0, -1.0]),
}
BASIN_CONFIGS = {
    "default_dmax3": RefineConfig(iters=200, delta_max=3.0, delta_step=0.05),
    "dmax6": RefineConfig(iters=200, delta_max=6.0, delta_step=0.05),
}
BASIN_SNR = 10.0
BASIN_DURATION = 8.0
N_SEEDS = 3


def sweep_basin() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("\n=== Sweep 1: basin of attraction ===")
    rows: list[dict[str, Any]] = []
    scatter: list[dict[str, Any]] = []
    for seed in range(N_SEEDS):
        audio, r_truth, _ = build_fixture(BASIN_DURATION, BASIN_SNR, seed)
        for cfg_name, cfg in BASIN_CONFIGS.items():
            for sign_name, sign in SIGN_MODES.items():
                for offset in OFFSETS:
                    r_init = r_truth + offset * sign[:, None]
                    err, conf = run_trial(audio, r_truth, r_init, cfg)
                    success = err < SUCCESS_TOL
                    rows.append(
                        {
                            "config": cfg_name,
                            "sign_mode": sign_name,
                            "offset": offset,
                            "seed": seed,
                            "mean_error": err,
                            "success": int(success),
                            "confidence": conf,
                        }
                    )
                    scatter.append(
                        {
                            "sweep": "basin",
                            "error": err,
                            "confidence": conf,
                            "success": int(success),
                        }
                    )
            print(f"  seed={seed} {cfg_name}: done")
    return rows, scatter


def summarise_basin(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Aggregate over seeds; find the cliff offset per (config, sign_mode).

    Semantics: ``success_rate`` is the fraction of *trials* (seeds) whose
    rotor-mean error beats ``SUCCESS_TOL``; ``mean_error`` averages over ALL
    seeds including failed ones, so it can sit below the tolerance while
    success_rate < 1. Concretely, seed 1's fixture draws near-identical
    wobble phases for rotors 1 and 3 (0.78 rev/s apart in lockstep), and the
    coarse stage snaps rotor 3 onto rotor 1's comb regardless of init offset
    — a rotor-identity capture (per-rotor MAE ~[0.01, 0.13, 0.01, 0.69],
    mean 0.209) that caps success_rate at 2/3 for offsets <= 2.0. Identical
    mean_error across small offsets is the coarse stage snapping every init
    in the basin to the same solution.
    """
    summary: list[dict[str, Any]] = []
    cliffs: dict[str, float] = {}
    for cfg_name in BASIN_CONFIGS:
        for sign_name in SIGN_MODES:
            last_success = 0.0
            for offset in OFFSETS:
                cell = [
                    r
                    for r in rows
                    if r["config"] == cfg_name
                    and r["sign_mode"] == sign_name
                    and r["offset"] == offset
                ]
                srate = float(np.mean([r["success"] for r in cell]))
                merr = float(np.mean([r["mean_error"] for r in cell]))
                summary.append(
                    {
                        "config": cfg_name,
                        "sign_mode": sign_name,
                        "offset": offset,
                        "success_rate": srate,
                        "mean_error": merr,
                    }
                )
                if srate >= 0.5:
                    last_success = offset
            cliffs[f"{cfg_name}/{sign_name}"] = last_success
    return summary, cliffs


def plot_basin(summary: list[dict[str, Any]]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    for cfg_name in BASIN_CONFIGS:
        for sign_name in SIGN_MODES:
            cell = [s for s in summary if s["config"] == cfg_name and s["sign_mode"] == sign_name]
            cell.sort(key=lambda s: s["offset"])
            xs = [s["offset"] for s in cell]
            label = f"{cfg_name} / {sign_name}"
            ax0.plot(xs, [s["success_rate"] for s in cell], marker="o", label=label)
            ax1.plot(xs, [s["mean_error"] for s in cell], marker="o", label=label)
    for cfg in BASIN_CONFIGS.values():
        ax0.axvline(cfg.delta_max, ls="--", color="gray", alpha=0.5)
        ax1.axvline(cfg.delta_max, ls="--", color="gray", alpha=0.5)
    ax0.set_xlabel("init offset (rev/s)")
    ax0.set_ylabel("success rate")
    ax0.set_title("Basin of attraction: success rate vs offset")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)
    ax1.axhline(SUCCESS_TOL, ls=":", color="red", alpha=0.7, label=f"tol={SUCCESS_TOL}")
    ax1.set_xlabel("init offset (rev/s)")
    ax1.set_ylabel("mean |refined - truth| (rev/s)")
    ax1.set_title("Final error vs offset (dashed = delta_max)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "basin.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sweep 2 — non-harmonic robustness
# --------------------------------------------------------------------------- #
ROBUST_SNRS = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0]
ROBUST_OFFSET = 1.0
ROBUST_CFG = RefineConfig(iters=200)
ROBUST_DURATION = 8.0


def sweep_robustness() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("\n=== Sweep 2: non-harmonic robustness ===")
    rows: list[dict[str, Any]] = []
    scatter: list[dict[str, Any]] = []
    sign = SIGN_MODES["same"]
    for noise_type in NOISE_FNS:
        for snr in ROBUST_SNRS:
            for seed in range(N_SEEDS):
                audio, r_truth, _ = build_fixture(ROBUST_DURATION, snr, seed, noise_type=noise_type)
                r_init = r_truth + ROBUST_OFFSET * sign[:, None]
                err, conf = run_trial(audio, r_truth, r_init, ROBUST_CFG)
                success = err < SUCCESS_TOL
                rows.append(
                    {
                        "noise_type": noise_type,
                        "snr_db": snr,
                        "seed": seed,
                        "mean_error": err,
                        "confidence": conf,
                        "success": int(success),
                    }
                )
                scatter.append(
                    {"sweep": "robust", "error": err, "confidence": conf, "success": int(success)}
                )
            print(f"  {noise_type} snr={snr:+.0f}dB: done")
    return rows, scatter


def summarise_robustness(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    summary: list[dict[str, Any]] = []
    min_usable: dict[str, float] = {}
    for noise_type in NOISE_FNS:
        usable = np.inf
        for snr in ROBUST_SNRS:
            cell = [r for r in rows if r["noise_type"] == noise_type and r["snr_db"] == snr]
            merr = float(np.mean([r["mean_error"] for r in cell]))
            mconf = float(np.mean([r["confidence"] for r in cell]))
            summary.append(
                {
                    "noise_type": noise_type,
                    "snr_db": snr,
                    "mean_error": merr,
                    "mean_confidence": mconf,
                }
            )
            if merr < SUCCESS_TOL:
                usable = min(usable, snr)
        min_usable[noise_type] = usable
    return summary, min_usable


def plot_robustness(summary: list[dict[str, Any]]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    for noise_type in NOISE_FNS:
        cell = [s for s in summary if s["noise_type"] == noise_type]
        cell.sort(key=lambda s: -s["snr_db"])
        xs = [s["snr_db"] for s in cell]
        ax0.plot(xs, [s["mean_error"] for s in cell], marker="o", label=noise_type)
        ax1.plot(xs, [s["mean_confidence"] for s in cell], marker="o", label=noise_type)
    ax0.axhline(SUCCESS_TOL, ls=":", color="red", alpha=0.7, label=f"tol={SUCCESS_TOL}")
    ax0.set_xlabel("harmonic SNR (dB)")
    ax0.set_ylabel("mean |refined - truth| (rev/s)")
    ax0.set_title("Refined error vs SNR")
    ax0.set_yscale("log")
    ax0.invert_xaxis()
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)
    ax1.set_xlabel("harmonic SNR (dB)")
    ax1.set_ylabel("mean comb confidence")
    ax1.set_title("Comb confidence vs SNR")
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "robustness.png", dpi=150)
    plt.close(fig)


def _gate_stats(conf: np.ndarray, success: np.ndarray, thr: float) -> tuple[float, float]:
    """Precision and recall of the accept rule ``confidence > thr``."""
    kept = conf > thr
    precision = float(np.mean(success[kept])) if kept.any() else 1.0
    recall = float(np.sum(kept & success) / max(np.sum(success), 1))
    return precision, recall


def analyse_confidence_gate(scatter: list[dict[str, Any]]) -> dict[str, float]:
    """Validate ``comb_confidence`` as an acceptance gate.

    Reports the confidence/error correlation, a *practical* threshold
    (maximising Youden's J = TPR - FPR over accept = ``conf > thr``), and the
    *strict* precision-1.0 threshold (max confidence among failures) — the
    latter is brittle when a few biased-but-confident locks slip through.
    """
    conf = np.array([s["confidence"] for s in scatter])
    err = np.array([s["error"] for s in scatter])
    success = np.array([s["success"] for s in scatter], dtype=bool)
    corr = float(np.corrcoef(conf, err)[0, 1])

    # Practical threshold: scan candidates, maximise TPR - FPR.
    n_pos = max(int(success.sum()), 1)
    n_neg = max(int((~success).sum()), 1)
    candidates = np.unique(conf)
    best_j, best_thr = -1.0, float(candidates[0])
    for thr in candidates:
        kept = conf > thr
        tpr = float(np.sum(kept & success)) / n_pos
        fpr = float(np.sum(kept & ~success)) / n_neg
        if tpr - fpr > best_j:
            best_j, best_thr = tpr - fpr, float(thr)
    prec, rec = _gate_stats(conf, success, best_thr)

    # Strict precision-1.0 threshold: above the most-confident failure.
    fail_conf = conf[~success]
    strict = float(np.max(fail_conf)) if fail_conf.size else float(np.min(conf) - 1e-6)
    strict_prec, strict_rec = _gate_stats(conf, success, strict)
    return {
        "corr_conf_err": corr,
        "threshold": best_thr,
        "precision_above": prec,
        "recall_above": rec,
        "youden_j": best_j,
        "strict_threshold": strict,
        "strict_precision": strict_prec,
        "strict_recall": strict_rec,
        "n_trials": float(len(scatter)),
        "n_success": float(np.sum(success)),
    }


def plot_confidence_scatter(scatter: list[dict[str, Any]], gate: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for sweep, color in (("basin", "tab:blue"), ("robust", "tab:orange")):
        pts = [s for s in scatter if s["sweep"] == sweep]
        ax.scatter(
            [p["confidence"] for p in pts],
            [p["error"] for p in pts],
            s=22,
            alpha=0.6,
            color=color,
            label=sweep,
        )
    ax.axhline(SUCCESS_TOL, ls=":", color="red", alpha=0.7, label=f"err tol={SUCCESS_TOL}")
    ax.axvline(
        gate["threshold"],
        ls="--",
        color="green",
        alpha=0.8,
        label=f"Youden thr={gate['threshold']:.3f}",
    )
    ax.axvline(
        gate["strict_threshold"],
        ls="--",
        color="purple",
        alpha=0.6,
        label=f"strict thr={gate['strict_threshold']:.3f}",
    )
    ax.set_xlabel("mean comb confidence")
    ax.set_ylabel("mean |refined - truth| (rev/s)")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_title(
        f"Confidence gate (corr={gate['corr_conf_err']:.2f}, "
        f"precision above thr={gate['precision_above']:.2f})"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confidence_scatter.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sweep 3 — efficiency
# --------------------------------------------------------------------------- #
EFF_DURATIONS = [10.0, 30.0, 60.0]
EFF_CHANNELS = [1, 8]
EFF_CFG = RefineConfig(iters=300)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def sweep_efficiency() -> list[dict[str, Any]]:
    """Per-stage cost, single-threaded, measured in CPU time.

    ``torch.set_num_threads(1)`` + ``time.process_time`` give the single-core
    CPU cost per audio-second — reproducible even on a loaded machine, where
    wall-clock is dominated by scheduler contention (min-of-N wall-clock was
    still inflated up to 60x during shared-box runs). Wall-clock is recorded
    alongside for reference; on an idle box the two agree.
    """
    print("\n=== Sweep 3: efficiency (single-threaded, CPU time) ===")
    n_threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    try:
        for duration in EFF_DURATIONS:
            for n_ch in EFF_CHANNELS:
                audio, r_truth, frame_times = build_fixture(
                    duration, 10.0, 0, noise_type="white", n_channels=n_ch
                )

                c0, t0 = time.process_time(), time.perf_counter()
                spec = compute_logmag(audio, EFF_CFG)
                c_logmag, t_logmag = time.process_time() - c0, time.perf_counter() - t0

                c0, t0 = time.process_time(), time.perf_counter()
                coarse_delta(spec, r_truth, EFF_CFG)
                c_coarse, t_coarse = time.process_time() - c0, time.perf_counter() - t0

                c0, t0 = time.process_time(), time.perf_counter()
                refine_trajectories(spec, r_truth, EFF_CFG)
                c_refine, t_refine = time.process_time() - c0, time.perf_counter() - t0

                c0, t0 = time.process_time(), time.perf_counter()
                harmonic_lsq_residual(audio, r_truth, frame_times, EFF_CFG, k_max=40)
                c_lsq, t_lsq = time.process_time() - c0, time.perf_counter() - t0

                for stage, cpu_s, wall_s in (
                    ("compute_logmag", c_logmag, t_logmag),
                    ("coarse_delta", c_coarse, t_coarse),
                    ("refine_trajectories@300", c_refine, t_refine),
                    ("harmonic_lsq_residual@k40", c_lsq, t_lsq),
                ):
                    rows.append(
                        {
                            "duration_s": duration,
                            "channels": n_ch,
                            "stage": stage,
                            "cpu_seconds": cpu_s,
                            "wall_seconds": wall_s,
                            "cpu_sec_per_audio_sec": cpu_s / duration,
                        }
                    )
                print(f"  dur={duration:.0f}s ch={n_ch}: done")
    finally:
        torch.set_num_threads(n_threads_before)
    return rows


def print_efficiency_table(rows: list[dict[str, Any]]) -> None:
    stages = [
        "compute_logmag",
        "coarse_delta",
        "refine_trajectories@300",
        "harmonic_lsq_residual@k40",
    ]
    print(f"\n  CPU: {_cpu_model()}")
    print("  single-thread CPU seconds per audio-second:")
    header = f"  {'dur/ch':>10} | " + " | ".join(f"{s[:20]:>20}" for s in stages)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for duration in EFF_DURATIONS:
        for n_ch in EFF_CHANNELS:
            cells = []
            for stage in stages:
                match = next(
                    r
                    for r in rows
                    if r["duration_s"] == duration and r["channels"] == n_ch and r["stage"] == stage
                )
                cells.append(f"{match['cpu_sec_per_audio_sec']:>20.4f}")
            print(f"  {f'{duration:.0f}s/{n_ch}ch':>10} | " + " | ".join(cells))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    basin_rows, basin_scatter = sweep_basin()
    basin_summary, cliffs = summarise_basin(basin_rows)
    write_csv(
        RESULTS_DIR / "basin.csv",
        basin_rows,
        ["config", "sign_mode", "offset", "seed", "mean_error", "success", "confidence"],
    )
    write_csv(
        RESULTS_DIR / "basin_summary.csv",
        basin_summary,
        ["config", "sign_mode", "offset", "success_rate", "mean_error"],
    )
    plot_basin(basin_summary)

    robust_rows, robust_scatter = sweep_robustness()
    robust_summary, min_usable = summarise_robustness(robust_rows)
    write_csv(
        RESULTS_DIR / "robustness.csv",
        robust_rows,
        ["noise_type", "snr_db", "seed", "mean_error", "confidence", "success"],
    )
    write_csv(
        RESULTS_DIR / "robustness_summary.csv",
        robust_summary,
        ["noise_type", "snr_db", "mean_error", "mean_confidence"],
    )
    plot_robustness(robust_summary)

    all_scatter = basin_scatter + robust_scatter
    gate = analyse_confidence_gate(all_scatter)
    write_csv(
        RESULTS_DIR / "confidence_scatter.csv",
        all_scatter,
        ["sweep", "error", "confidence", "success"],
    )
    plot_confidence_scatter(all_scatter, gate)

    eff_rows = sweep_efficiency()
    write_csv(
        RESULTS_DIR / "efficiency.csv",
        eff_rows,
        ["duration_s", "channels", "stage", "cpu_seconds", "wall_seconds", "cpu_sec_per_audio_sec"],
    )

    # ------------------------------------------------------------------- #
    # Compact summary
    # ------------------------------------------------------------------- #
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBasin cliff (largest offset with success rate >= 0.5, rev/s):")
    for name, cfg in BASIN_CONFIGS.items():
        for sign_name in SIGN_MODES:
            key = f"{name}/{sign_name}"
            print(f"  {key:<24} cliff={cliffs[key]:>4.2f}  (delta_max={cfg.delta_max})")
    print("\nMinimum usable harmonic SNR (mean err < 0.15 rev/s), per noise type:")
    for noise_type, snr in min_usable.items():
        val = "none" if not np.isfinite(snr) else f"{snr:+.0f} dB"
        print(f"  {noise_type:<10} {val}")
    print("\nConfidence acceptance gate:")
    print(f"  corr(confidence, error)   = {gate['corr_conf_err']:.3f}")
    print(f"  suggested threshold       = {gate['threshold']:.4f}  (Youden-optimal)")
    print(
        f"    accept conf>thr: precision={gate['precision_above']:.3f} "
        f"recall={gate['recall_above']:.3f}"
    )
    print(f"  strict precision-1.0 thr  = {gate['strict_threshold']:.4f}")
    print(
        f"    accept conf>thr: precision={gate['strict_precision']:.3f} "
        f"recall={gate['strict_recall']:.3f}  (brittle: confident biased locks)"
    )
    print(f"  {int(gate['n_success'])}/{int(gate['n_trials'])} trials succeed overall")
    print_efficiency_table(eff_rows)
    print(f"\nTotal wall-clock: {elapsed / 60:.1f} min")
    print(f"Artifacts written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
