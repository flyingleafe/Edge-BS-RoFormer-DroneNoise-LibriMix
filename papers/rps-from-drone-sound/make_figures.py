#!/usr/bin/env python3
"""
Regenerate all paper figures using the unified RPS API.

Replaces the old 470-line make_figures.py (~4 LOC → 1 CLI / API call per figure).

Figures:
  fig_training_curves.pdf      → make-plot --type=rps_prediction.training_curves
  fig_qualitative_combined.pdf → make-plot --type=rps_prediction.sample_comparison (×3, combined)
  fig_highsnr_per_sample.pdf   → evaluate-rps on high-SNR clips + custom bar chart
  fig_highsnr_outlier.pdf      → make-plot --type=rps_prediction.sample_comparison
  fig_full_sequence.pdf        → make-plot --type=rps_prediction.full_sequence

Usage from project root:
    python papers/rps-from-drone-sound/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from tasks.rps_prediction import load_predictor
from utils.plots.rps_prediction import PLOT_TYPES

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]

# Paper style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

from utils.paths import get_data_path, get_datasets_path, get_results_path

CHECKPOINT = str(get_results_path("rps_exp_simple_conv/best_simple_conv.pt"))
DREGON_VALID = get_datasets_path("DREGON-LM/valid")
DREGON_DIR = get_data_path("DREGON")
RECORDING = "DREGON_free-flight_speech-high_room1"
TARGET_SR = 16000
N_FFT = 2048
HOP = 512


# ── Fig 1: Training curves ──────────────────────────────────────────────


def fig_training_curves():
    csv_path = get_results_path("rps_predictor/rps_predictor/training_log.csv")
    if not csv_path.is_file():
        csv_path = get_results_path("rps_exp_simple_conv/training_log.csv")
    fig = PLOT_TYPES["rps_prediction.training_curves"](log_paths=[str(csv_path)])
    out = FIG_DIR / "fig_training_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ── Fig 2: Qualitative examples (3-sample composite) ────────────────────


def fig_qualitative_examples():
    from tasks.rps_prediction import load_input_set

    sample_ids = ("sample_00000", "sample_00149", "sample_00599")
    all_samples = {s.tags.get("id", ""): s for s in load_input_set(str(DREGON_VALID))}
    pred = load_predictor(f"simple_conv@{CHECKPOINT}")

    n = len(sample_ids)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(7.1, 3.8),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40, "wspace": 0.30},
    )
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, sid in enumerate(sample_ids):
        sample = all_samples[sid]
        audio = np.asarray(sample["audio"].samples, dtype=np.float32)
        sr = sample["audio"].sr
        dur = len(audio) / sr

        # Spectrogram
        ax = axes[0, col]
        spec = np.abs(
            np.fft.rfft(
                np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP] * np.hanning(N_FFT),
                axis=-1,
            )
        )
        log_mag = np.log1p(spec.T)
        ax.imshow(
            log_mag,
            origin="lower",
            aspect="auto",
            extent=[0, dur, 0, sr / 2 / 1000],
            cmap="hot",
            vmin=np.percentile(log_mag, 2),
            vmax=np.percentile(log_mag, 99),
        )
        ax.set_ylim(0, 4)
        ax.set_ylabel("freq [kHz]" if col == 0 else "")
        ax.set_title(f"sample {sid.split('_')[1]}")
        ax.set_xticklabels([])
        ax.grid(False)

        # RPS overlays
        ax = axes[1, col]
        n_frames = len(audio) // HOP + 1
        frame_times = np.arange(n_frames) * HOP / sr
        gt = sample["rps"].interpolate(frame_times).T  # (4, F)
        pred_arr = pred.predict(audio, sr=sr)
        T = min(pred_arr.shape[1], n_frames)
        for r in range(4):
            ax.plot(
                frame_times[:T],
                gt[r, :T],
                ":",
                color=ROTOR_COLORS[r],
                lw=0.5,
                alpha=0.55,
                label=f"GT R{r + 1}" if col == 0 else None,
            )
            ax.plot(
                frame_times[:T],
                pred_arr[r, :T],
                "-",
                color=ROTOR_COLORS[r],
                lw=0.5,
                alpha=0.75,
                label=f"pred R{r + 1}" if col == 0 else None,
            )
        ax.set_xlabel("time [s]")
        if col == 0:
            ax.set_ylabel("rotor speed [rev/s]")
        ax.set_xlim(0, dur)

    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=0.5, ls=":", alpha=0.55, label="ground truth"),
        plt.Line2D([0], [0], color="black", lw=0.5, ls="-", alpha=0.75, label="predicted"),
    ] + [
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r]) for r in range(4)
    ]
    fig.subplots_adjust(bottom=0.20)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=8,
    )

    out = FIG_DIR / "fig_qualitative_combined.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out)


# ── Fig 3: High-SNR per-sample MSE ──────────────────────────────────────


def fig_highsnr_per_sample():
    data_path = get_results_path("rps_high_snr_analysis.json")
    if not data_path.is_file():
        print("SKIP: results/rps_high_snr_analysis.json not found")
        return
    data = json.load(open(data_path))
    samples = data["results"]
    times = [s["rel_time"] for s in samples]
    sc_mse = [s["simple_conv"]["mse"] for s in samples]

    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    x = np.arange(len(samples))
    ax.bar(x, sc_mse, 0.4, label="SimpleConv", color="#1f77b4")
    ax.axhline(5.15, ls="--", lw=1.0, color="#444", label="held-out synthetic, MSE = 5.15")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}s" for t in times], fontsize=7)
    ax.set_xlabel("clip start time within DREGON free-flight speech-high room1")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)

    outlier_idx = int(np.argmax(sc_mse))
    ax.annotate(
        "drone-landing regime\n(RPS→0)",
        xy=(outlier_idx, sc_mse[outlier_idx]),
        xytext=(outlier_idx - 2.8, sc_mse[outlier_idx] * 0.6),
        arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
        fontsize=7,
        color="#555",
    )

    out = FIG_DIR / "fig_highsnr_per_sample.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ── Fig 4: High-SNR outlier ─────────────────────────────────────────────


def fig_highsnr_outlier():
    rec_dir = DREGON_DIR / RECORDING
    audio_full, sr = torchaudio.load(str(rec_dir / f"{RECORDING}.wav"))
    audio_ts = scipy.io.loadmat(rec_dir / f"{RECORDING}_audiots.mat")["audio_timestamps"].flatten()
    motor_mat = scipy.io.loadmat(rec_dir / f"{RECORDING}_motors.mat")
    motor_data = motor_mat["motor"][0, 0]
    measured = motor_data["measured"]
    motor_ts = motor_data["timestamps"].flatten()
    motor_sr = len(motor_ts) / (motor_ts[-1] - motor_ts[0])

    SAMPLE_DUR = 8.224
    recording_start = motor_ts[0]
    recording_dur = motor_ts[-1] - motor_ts[0]
    start_offset = 5.0
    usable_dur = recording_dur - start_offset - SAMPLE_DUR
    step = usable_dur / 9
    rel_time = start_offset + 9 * step  # sample 9

    unix_time = recording_start + rel_time
    audio_start = int((unix_time - audio_ts[0]) * sr)
    audio_end = min(audio_start + int(SAMPLE_DUR * sr), audio_full.shape[1])
    audio_start = min(audio_start, audio_full.shape[1] - int(SAMPLE_DUR * sr))
    audio_chunk = audio_full[:, audio_start:audio_end]

    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
    audio_mono = audio_chunk.mean(dim=0).unsqueeze(0)
    audio_16k = resampler(audio_mono)
    peak = audio_16k.abs().max()
    if peak > 0:
        audio_16k = audio_16k / peak * 0.9

    motor_start = np.searchsorted(motor_ts, recording_start + rel_time) - 1
    motor_end = np.searchsorted(motor_ts, recording_start + rel_time + SAMPLE_DUR) + 1
    rps_chunk = measured[motor_start:motor_end].T.astype(np.float32)

    pred = load_predictor(f"simple_conv@{CHECKPOINT}")
    pred_arr = pred.predict(audio_16k.squeeze().numpy(), sr=TARGET_SR)

    n_frames = audio_16k.shape[-1] // HOP + 1
    rps_stft = np.array(
        [
            np.interp(
                np.arange(n_frames) * HOP / TARGET_SR,
                np.arange(rps_chunk.shape[1]) / motor_sr,
                rps_chunk[r],
            )
            for r in range(4)
        ]
    )

    audio_np = audio_16k.numpy()[0]
    dur = len(audio_np) / TARGET_SR
    fig, axes = plt.subplots(
        2, 1, figsize=(7.1, 3.8), gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40}
    )

    ax = axes[0]
    spec = np.abs(
        np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(audio_np, N_FFT)[::HOP] * np.hanning(N_FFT),
            axis=-1,
        )
    )
    log_mag = np.log1p(spec.T)
    ax.imshow(
        log_mag,
        origin="lower",
        aspect="auto",
        extent=[0, dur, 0, TARGET_SR / 2 / 1000],
        cmap="hot",
        vmin=np.percentile(log_mag, 2),
        vmax=np.percentile(log_mag, 99),
    )
    ax.set_ylim(0, 4)
    ax.set_ylabel("freq [kHz]")
    ax.set_title("outlier clip ($t \\approx 38.6$ s) — drone landing")
    ax.set_xticklabels([])
    ax.grid(False)

    ax = axes[1]
    for r in range(4):
        ax.plot(
            np.linspace(0, dur, rps_stft.shape[1]),
            rps_stft[r],
            ":",
            color=ROTOR_COLORS[r],
            lw=0.5,
            alpha=0.55,
        )
        ax.plot(
            np.linspace(0, dur, pred_arr.shape[1]),
            pred_arr[r],
            "-",
            color=ROTOR_COLORS[r],
            lw=0.5,
            alpha=0.75,
        )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rotor speed [rev/s]")
    ax.set_xlim(0, dur)

    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=0.5, ls=":", alpha=0.55, label="ground truth"),
        plt.Line2D([0], [0], color="black", lw=0.5, ls="-", alpha=0.75, label="predicted"),
    ] + [
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r]) for r in range(4)
    ]
    fig.subplots_adjust(bottom=0.20)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=8,
    )

    out = FIG_DIR / "fig_highsnr_outlier.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out)


# ── Fig 5: Full sequence ─────────────────────────────────────────────────


def fig_full_sequence():
    rec_dir = DREGON_DIR / RECORDING
    audio_full, file_sr = torchaudio.load(str(rec_dir / f"{RECORDING}.wav"))
    audio_ts = scipy.io.loadmat(rec_dir / f"{RECORDING}_audiots.mat")["audio_timestamps"].flatten()
    motor_mat = scipy.io.loadmat(rec_dir / f"{RECORDING}_motors.mat")
    motor_data = motor_mat["motor"][0, 0]
    measured = motor_data["measured"]
    motor_ts = motor_data["timestamps"].flatten()
    motor_sr_val = len(motor_ts) / (motor_ts[-1] - motor_ts[0])

    t0, t1 = motor_ts[0], motor_ts[-1]
    audio_crop = audio_full[
        :, int((t0 - audio_ts[0]) * file_sr) : int((t1 - audio_ts[0]) * file_sr)
    ]
    audio_mono = audio_crop.mean(dim=0, keepdim=True)
    if file_sr != TARGET_SR:
        audio_mono = torchaudio.transforms.Resample(file_sr, TARGET_SR)(audio_mono)
    peak = audio_mono.abs().max()
    if peak > 0:
        audio_mono = audio_mono / peak * 0.9

    rps_motor = measured.T.astype(np.float32)
    n_frames = audio_mono.shape[-1] // HOP + 1
    stft_times = np.arange(n_frames) * HOP / TARGET_SR
    motor_times = np.arange(rps_motor.shape[1]) / motor_sr_val
    gt = np.array([np.interp(stft_times, motor_times, rps_motor[r]) for r in range(4)])

    pred = load_predictor(f"simple_conv@{CHECKPOINT}")
    pred_arr = pred.predict(audio_mono.squeeze().numpy(), sr=TARGET_SR)

    fig = PLOT_TYPES["rps_prediction.full_sequence"](
        audio=audio_mono.squeeze().numpy(),
        rps_gt=gt,
        rps_pred=pred_arr,
    )
    out = FIG_DIR / "fig_full_sequence.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="*", default=[], help="Which figures to generate (default: all)")
    args = ap.parse_args()

    all_figs = {
        "training_curves",
        "qualitative",
        "highsnr_per_sample",
        "highsnr_outlier",
        "full_sequence",
    }
    to_run = set(args.fig) if args.fig else all_figs

    for name in sorted(to_run):
        print(f"\n=== {name} ===")
        try:
            {
                "training_curves": fig_training_curves,
                "qualitative": fig_qualitative_examples,
                "highsnr_per_sample": fig_highsnr_per_sample,
                "highsnr_outlier": fig_highsnr_outlier,
                "full_sequence": fig_full_sequence,
            }[name]()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
    print("\nDone.")
