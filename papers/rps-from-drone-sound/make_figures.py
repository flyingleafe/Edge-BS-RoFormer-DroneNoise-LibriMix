#!/usr/bin/env python3
"""
Generate all figures for the paper from local results.

Outputs go to papers/rps-from-drone-sound/figures/.

Figures produced:
- fig_training_curves.pdf      — train/val MSE + R² vs epoch (SimpleConv)
- fig_qualitative_<id>.pdf     — spectrogram + ground-truth + predicted RPS for selected samples
- fig_qualitative_combined.pdf — three-panel composite of qualitative samples
- fig_full_sequence.pdf        — full-sequence spectrogram + RPS + MSE on real high-SNR recording

Run from project root inside the nix dev shell:
    python papers/rps-from-drone-sound/make_figures.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F

# Add project root to path so we can import the model
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_rps_predictor import SimpleConv

# --- Plot style: clean, paper-friendly ----------------------------------------
plt.rcParams.update({
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
})

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]


# -----------------------------------------------------------------------------
# Fig. 1 — Training curves
# -----------------------------------------------------------------------------
def fig_training_curves() -> None:
    csv_path = ROOT / "results/rps_predictor/rps_predictor/training_log.csv"
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 2.4))

    # MSE — log scale, clip top so the catastrophic epoch 1 doesn't dominate
    ax.plot(df["epoch"], df["train_mse"], "-", lw=1.2, label="Train", color="#1f77b4")
    ax.plot(df["epoch"], df["val_mse"], "-", lw=1.2, label="Validation", color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.legend(frameon=False, loc="upper right")

    out = FIG_DIR / "fig_training_curves.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# -----------------------------------------------------------------------------
# Fig. 2 — Qualitative examples: spectrogram + GT/predicted RPS
# -----------------------------------------------------------------------------
def _load_sample(sample_id: str):
    base = ROOT / "results/rps_eval_specific_samples" / sample_id
    audio, sr = sf.read(base / "mixture.wav")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    gt_raw = np.load(base / "ground_truth_rps.npy")   # (4, T_gt)  native ~950 Hz
    pred   = np.load(base / "simple_conv_rps.npy")    # (4, T_pred) STFT grid
    # Linearly interpolate GT onto the prediction time grid so that
    # ground-truth and predicted traces are directly comparable.
    gt = np.zeros_like(pred)
    x_old = np.linspace(0, 1, gt_raw.shape[1])
    x_new = np.linspace(0, 1, pred.shape[1])
    for r in range(4):
        gt[r] = np.interp(x_new, x_old, gt_raw[r])
    return audio.astype(np.float32), sr, gt, pred


def fig_qualitative_examples(sample_ids=("sample_00000", "sample_00149", "sample_00599")) -> None:
    """Three-row composite: spectrogram on top, GT vs predicted RPS overlaid below."""
    n = len(sample_ids)
    fig, axes = plt.subplots(2, n, figsize=(7.1, 3.8),
                             gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40, "wspace": 0.30})
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, sid in enumerate(sample_ids):
        audio, sr, gt, pred = _load_sample(sid)
        duration = len(audio) / sr

        # Spectrogram
        ax = axes[0, col]
        n_fft, hop = 2048, 512
        spec = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] *
            np.hanning(n_fft), axis=-1))
        log_mag = np.log1p(spec.T)
        vmin = np.percentile(log_mag, 2)
        vmax = np.percentile(log_mag, 99)
        ax.imshow(
            log_mag, origin="lower", aspect="auto",
            extent=[0, duration, 0, sr / 2 / 1000],
            cmap="hot", vmin=vmin, vmax=vmax,
        )
        ax.set_ylim(0, 4)  # focus on lowest 4 kHz where harmonics live
        ax.set_ylabel("freq [kHz]" if col == 0 else "")
        ax.set_title(f"sample {sid.split('_')[1]}")
        ax.set_xticklabels([])
        ax.grid(False)

        # RPS overlays
        ax = axes[1, col]
        t_gt = np.linspace(0, duration, gt.shape[1])
        t_pred = np.linspace(0, duration, pred.shape[1])
        for r in range(4):
            ax.plot(t_gt,   gt[r],   ":",  color=ROTOR_COLORS[r], lw=0.5, alpha=0.55,
                    label=f"GT R{r+1}" if col == 0 else None)
            ax.plot(t_pred, pred[r], "-",  color=ROTOR_COLORS[r], lw=0.5, alpha=0.75,
                    label=f"pred R{r+1}" if col == 0 else None)
        ax.set_xlabel("time [s]")
        if col == 0:
            ax.set_ylabel("rotor speed [rev/s]")
        ax.set_xlim(0, duration)

    # Compact legend below; reserve space so it does not overlap x-labels
    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=0.5, ls=":",  alpha=0.55, label="ground truth"),
        plt.Line2D([0], [0], color="black", lw=0.5, ls="-",  alpha=0.75, label="predicted"),
    ] + [
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r]) for r in range(4)
    ]
    fig.subplots_adjust(bottom=0.20)
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8)

    out = FIG_DIR / "fig_qualitative_combined.pdf"
    fig.savefig(out, bbox_inches="tight")
    # Also save PNG for quick visual inspection
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    print("wrote", out, "and", png_out)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Fig. 3 — High-SNR generalization: per-sample MSE
# -----------------------------------------------------------------------------
def fig_highsnr_per_sample() -> None:
    """Per-sample MSE on 10 out-of-distribution high-SNR free-flight clips.

    Reference horizontal line: held-out synthetic mixtures MSE = 5.15.
    The last sample (drone landing) is annotated as an outlier with
    very different acoustic conditions.
    """
    data = json.load(open(ROOT / "results/rps_high_snr_analysis.json"))
    samples = data["results"]
    times = [s["rel_time"] for s in samples]
    sc_mse = [s["simple_conv"]["mse"] for s in samples]

    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    x = np.arange(len(samples))
    w = 0.4

    ax.bar(x, sc_mse, w, label="SimpleConv", color="#1f77b4")

    # Reference line: synthetic-mixture MSE = 5.15
    ax.axhline(5.15, ls="--", lw=1.0, color="#444",
               label="held-out synthetic, MSE = 5.15")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}s" for t in times], rotation=0, fontsize=7)
    ax.set_xlabel("clip start time within DREGON free-flight $\\mathit{speech\\,high}$ room1")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)

    # Annotate outlier
    outlier_idx = int(np.argmax(sc_mse))
    ax.annotate(
        "drone-landing regime\n(RPS$\\to$0)",
        xy=(outlier_idx, sc_mse[outlier_idx]),
        xytext=(outlier_idx - 2.8, sc_mse[outlier_idx] * 0.6),
        arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
        fontsize=7, color="#555",
    )

    out = FIG_DIR / "fig_highsnr_per_sample.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# -----------------------------------------------------------------------------
# Fig. 4 — High-SNR outlier: spectrogram + GT/predicted RPS (drone landing)
# -----------------------------------------------------------------------------
def fig_highsnr_outlier() -> None:
    """The outlier clip from the high-SNR evaluation (sample 9, t ~ 38.6 s),
    plotted in the same style as the qualitative validation examples.
    All four rotor speeds drop to zero during landing; the model has never
    seen this regime and continues to predict non-zero speeds.
    """
    DREGON_DIR = ROOT / "data/DREGON"
    RECORDING = "DREGON_free-flight_speech-high_room1"
    TARGET_SR = 16000
    N_FFT = 2048
    HOP_LENGTH = 512
    SAMPLE_DURATION = 8.224

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleConv(n_fft=N_FFT, hop_length=HOP_LENGTH).to(device)
    ckpt_path = ROOT / "results/rps_predictor/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load recording
    rec_dir = DREGON_DIR / RECORDING
    audio_path = rec_dir / f"{RECORDING}.wav"
    audio_full, sr = torchaudio.load(str(audio_path))

    audio_ts_mat = scipy.io.loadmat(rec_dir / f"{RECORDING}_audiots.mat")
    audio_ts = audio_ts_mat["audio_timestamps"].flatten()

    motor_mat = scipy.io.loadmat(rec_dir / f"{RECORDING}_motors.mat")
    motor_data = motor_mat["motor"][0, 0]
    measured = motor_data["measured"]
    motor_ts = motor_data["timestamps"].flatten()
    motor_sr = len(motor_ts) / (motor_ts[-1] - motor_ts[0])

    recording_start = motor_ts[0]
    recording_duration = motor_ts[-1] - motor_ts[0]

    # Compute sample 9 parameters (same logic as analyze_rps_high_snr.py)
    start_offset = 5.0
    usable_duration = recording_duration - start_offset - SAMPLE_DURATION
    step = usable_duration / 9  # num_samples - 1
    rel_time = start_offset + 9 * step

    # Extract audio chunk
    unix_time = recording_start + rel_time
    audio_start_idx = int((unix_time - audio_ts[0]) * sr)
    end_sample = min(audio_start_idx + int(SAMPLE_DURATION * sr), audio_full.shape[1])
    audio_start_idx = min(audio_start_idx, audio_full.shape[1] - int(SAMPLE_DURATION * sr))
    audio_chunk = audio_full[:, audio_start_idx:end_sample]

    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
    audio_mono = audio_chunk.mean(dim=0)
    audio_16k = resampler(audio_mono.unsqueeze(0))
    audio_max = audio_16k.abs().max()
    if audio_max > 0:
        audio_16k = audio_16k / audio_max * 0.9

    # Get motor data
    motor_rel_start = rel_time
    motor_start_idx = np.searchsorted(motor_ts, recording_start + motor_rel_start) - 1
    motor_end_idx = np.searchsorted(motor_ts, recording_start + motor_rel_start + SAMPLE_DURATION) + 1
    rps_chunk = measured[motor_start_idx:motor_end_idx].T.astype(np.float32)

    # Run model
    audio_input = audio_16k.to(device)
    with torch.no_grad():
        rps_pred = model(audio_input)
    rps_pred_np = rps_pred.cpu().numpy()[0]  # (4, T)

    # Compute GT at STFT rate
    n_frames = audio_input.shape[-1] // HOP_LENGTH + 1
    rps_stft = np.zeros((4, n_frames))
    for rotor in range(4):
        motor_times = np.arange(rps_chunk.shape[1]) / motor_sr
        stft_times = np.arange(n_frames) * (HOP_LENGTH / TARGET_SR)
        rps_stft[rotor] = np.interp(stft_times, motor_times, rps_chunk[rotor])

    # Plot (same style as fig_qualitative_examples, single column)
    audio_np = audio_16k.numpy()[0]
    duration = len(audio_np) / TARGET_SR

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 3.8),
                             gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40})

    # Spectrogram
    ax = axes[0]
    n_fft, hop = 2048, 512
    spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(audio_np, n_fft)[::hop] *
        np.hanning(n_fft), axis=-1))
    log_mag = np.log1p(spec.T)
    vmin = np.percentile(log_mag, 2)
    vmax = np.percentile(log_mag, 99)
    ax.imshow(
        log_mag, origin="lower", aspect="auto",
        extent=[0, duration, 0, TARGET_SR / 2 / 1000],
        cmap="hot", vmin=vmin, vmax=vmax,
    )
    ax.set_ylim(0, 4)
    ax.set_ylabel("freq [kHz]")
    ax.set_title("outlier clip ($t \\approx 38.6$ s) --- drone landing")
    ax.set_xticklabels([])
    ax.grid(False)

    # RPS overlays
    ax = axes[1]
    t_gt = np.linspace(0, duration, rps_stft.shape[1])
    t_pred = np.linspace(0, duration, rps_pred_np.shape[1])
    for r in range(4):
        ax.plot(t_gt,   rps_stft[r],   ":",  color=ROTOR_COLORS[r], lw=0.5, alpha=0.55,
                label=f"GT R{r+1}")
        ax.plot(t_pred, rps_pred_np[r], "-",  color=ROTOR_COLORS[r], lw=0.5, alpha=0.75,
                label=f"pred R{r+1}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rotor speed [rev/s]")
    ax.set_xlim(0, duration)

    # Compact legend below
    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=0.5, ls=":",  alpha=0.55, label="ground truth"),
        plt.Line2D([0], [0], color="black", lw=0.5, ls="-",  alpha=0.75, label="predicted"),
    ] + [
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r]) for r in range(4)
    ]
    fig.subplots_adjust(bottom=0.20)
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8)

    out = FIG_DIR / "fig_highsnr_outlier.pdf"
    fig.savefig(out, bbox_inches="tight")
    # Also save PNG for quick visual inspection
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    print("wrote", out, "and", png_out)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Fig. 5 — Full-sequence high-SNR recording: spectrogram + RPS + MSE over time
# -----------------------------------------------------------------------------
def fig_full_sequence() -> None:
    """Load pre-computed full-sequence predictions and render the 3-panel figure.

    Requires `python analyze_rps_full_sequence.py` to have been run first.
    """
    out_dir = ROOT / "results/rps_full_sequence"
    if not (out_dir / "rps_pred.npy").exists():
        print("WARNING: results/rps_full_sequence/ not found — skipping fig_full_sequence")
        return

    audio_np = np.load(out_dir / "audio_16k.npy")
    rps_pred = np.load(out_dir / "rps_pred.npy")
    rps_gt = np.load(out_dir / "rps_gt_stft.npy")
    mse_per_frame = np.load(out_dir / "mse_per_frame.npy")

    duration = len(audio_np) / 16000
    t_stft = np.linspace(0, duration, rps_gt.shape[1])

    # Smooth MSE trace (1-second moving average)
    hop = 512
    sr = 16000
    frame_dur = hop / sr
    w = max(1, int(1.0 / frame_dur))
    kernel = np.ones(w) / w
    mse_smooth = np.convolve(mse_per_frame, kernel, mode="same")

    # Identify low-RPS regions for highlighting
    low_rps = np.all(rps_gt < 50, axis=0)
    transitions = np.diff(low_rps.astype(int))
    low_starts = np.where(transitions == 1)[0] + 1
    low_ends = np.where(transitions == -1)[0]
    if low_rps[0]:
        low_starts = np.r_[0, low_starts]
    if low_rps[-1]:
        low_ends = np.r_[low_ends, len(low_rps) - 1]

    fig, axes = plt.subplots(
        3, 1, figsize=(7.1, 6.0),
        gridspec_kw={"height_ratios": [1.2, 1.0, 0.8], "hspace": 0.35}
    )

    # ── Panel 1: spectrogram ──
    ax = axes[0]
    n_fft = 2048
    spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(audio_np, n_fft)[::hop] *
        np.hanning(n_fft), axis=-1))
    log_mag = np.log1p(spec.T)
    vmin = np.percentile(log_mag, 2)
    vmax = np.percentile(log_mag, 99)
    ax.imshow(
        log_mag, origin="lower", aspect="auto",
        extent=[0, duration, 0, 16000 / 2 / 1000],
        cmap="hot", vmin=vmin, vmax=vmax,
    )
    ax.set_ylim(0, 4)
    ax.set_ylabel("freq [kHz]")
    ax.set_title("DREGON free-flight speech-high (full sequence)")
    ax.set_xticklabels([])
    ax.grid(False)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)

    # ── Panel 2: rotor speeds ──
    ax = axes[1]
    for r in range(4):
        ax.plot(t_stft, rps_gt[r], ":", color=ROTOR_COLORS[r], lw=0.5, alpha=0.55)
        ax.plot(t_stft, rps_pred[r], "-", color=ROTOR_COLORS[r], lw=0.5, alpha=0.75)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)
    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=0.5, ls=":", alpha=0.55, label="ground truth"),
        plt.Line2D([0], [0], color="black", lw=0.5, ls="-", alpha=0.75, label="predicted"),
    ] + [
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r]) for r in range(4)
    ]
    ax.legend(handles=legend_handles, loc="lower center", frameon=False,
              fontsize=7, ncol=3, columnspacing=0.8,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_ylabel("rotor speed [rev/s]")
    ax.set_xlim(0, duration)
    ax.set_xticklabels([])

    # ── Panel 3: MSE over time ──
    ax = axes[2]
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)
    ax.plot(t_stft, mse_smooth, "-", color="#d62728", lw=0.8)
    ax.fill_between(t_stft, mse_smooth, alpha=0.15, color="#d62728")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.set_xlim(0, duration)
    if mse_smooth.max() / (mse_smooth[mse_smooth > 0].min() + 1e-8) > 10:
        ax.set_yscale("log")
    ax.axhline(5.15, ls="--", lw=0.8, color="#444", alpha=0.7,
               label="held-out synthetic, MSE = 5.15")
    ax.legend(frameon=False, loc="upper center", fontsize=7,
              bbox_to_anchor=(0.5, 1.02))

    out = FIG_DIR / "fig_full_sequence.pdf"
    fig.savefig(out, bbox_inches="tight")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    print("wrote", out, "and", png_out)
    plt.close(fig)


if __name__ == "__main__":
    fig_training_curves()
    fig_qualitative_examples()
    fig_full_sequence()
