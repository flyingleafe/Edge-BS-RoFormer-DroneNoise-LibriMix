#!/usr/bin/env python3
"""
Generate 3-panel full-sequence plots for in-flight recordings.

Top:    Log-magnitude spectrogram of the audio (cropped to motor range).
Middle: 4-rotor RPS traces — dotted = ground-truth (commanded), solid = predicted.
Bottom: Per-frame MSE with mean reference line.

Audio is cropped to the motor-telemetry timestamp range and resampled to 16 kHz mono.
Ground-truth RPS is the *commanded* motor speed (what V2 models were trained on),
interpolated to the STFT frame rate of the cropped audio.

Usage:
    cd papers/dregon_lm_v2_rps_report
    python generate_fullseq_figures.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import scipy.io
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_rps_predictor import get_model
import itertools

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DREGON_DIR = PROJECT_ROOT / "data" / "DREGON"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ─────────────────────────────────────────────────────────────────
TARGET_SR = 16000
N_FFT = 2048
HOP_LENGTH = 512

# Model registry: (label, architecture, checkpoint path)
MODELS = {
    "old_simpleconv": (
        "OLD SimpleConv",
        "simple_conv",
        PROJECT_ROOT / "results" / "rps_exp_simple_conv" / "best_simple_conv.pt",
    ),
    "old_bigru-v2": (
        "OLD BiGRU-v2",
        "simple_conv_bigru_v2",
        PROJECT_ROOT / "results" / "rps_exp_simple_conv_bigru_v2" / "best_simple_conv_bigru_v2.pt",
    ),
    "v3_simpleconv": (
        "V3 SimpleConv",
        "simple_conv",
        PROJECT_ROOT / "results" / "rps_predictor_v3" / "simple_conv" / "best_simple_conv.pt",
    ),
    "v3_bigru-v2": (
        "V3 BiGRU-v2",
        "simple_conv_bigru_v2",
        PROJECT_ROOT / "results" / "rps_predictor_v3" / "simple_conv_bigru_v2" / "best_simple_conv_bigru_v2.pt",
    ),
}

RECORDINGS = [
    "DREGON_free-flight_speech-high_room1",
    "DREGON_free-flight_whitenoise-high_room1",
]

# ─── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["R1", "R2", "R3", "R4"]
_PERMS = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)


def load_aligned_data(recording_id: str):
    """Load audio and motor data, crop audio to motor range, resample to 16 kHz mono."""
    rec_dir = DREGON_DIR / recording_id

    audio_path = rec_dir / f"{recording_id}.wav"
    audio_full, sr = torchaudio.load(str(audio_path))

    audio_ts_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_audiots.mat")
    audio_ts = audio_ts_mat["audio_timestamps"].flatten()

    motor_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_motors.mat")
    motor_data = motor_mat["motor"][0, 0]
    command = motor_data["command"]
    motor_ts = motor_data["timestamps"].flatten()
    motor_sr = len(motor_ts) / (motor_ts[-1] - motor_ts[0])

    # Crop audio to motor timestamp range
    t0 = motor_ts[0]
    t1 = motor_ts[-1]
    audio_start_idx = int((t0 - audio_ts[0]) * sr)
    audio_end_idx = int((t1 - audio_ts[0]) * sr)
    audio_start_idx = max(0, audio_start_idx)
    audio_end_idx = min(audio_full.shape[1], audio_end_idx)
    audio_crop = audio_full[:, audio_start_idx:audio_end_idx]

    # Mono + resample + peak normalise (matches training / eval_rps_full_sequence.py)
    audio_mono = audio_crop.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio_16k = resampler(audio_mono)
    else:
        audio_16k = audio_mono

    peak = audio_16k.abs().max()
    if peak > 0:
        audio_16k = audio_16k / peak * 0.9

    # Clean command spikes (same logic as eval_cross.py)
    command_cleaned = command.copy()
    command_cleaned[command_cleaned < 5] = 0
    rps_motor = command_cleaned.T.astype(np.float32)

    return audio_16k, rps_motor, motor_sr, float(t0)


def run_inference(audio: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    model.eval()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    with torch.no_grad():
        pred = model(audio)
    return pred.cpu().numpy()[0]  # (4, T)


def interpolate_rps_to_stft(rps_motor: np.ndarray, motor_sr: float, n_frames: int) -> np.ndarray:
    motor_times = np.arange(rps_motor.shape[1]) / motor_sr
    stft_times = np.arange(n_frames) * (HOP_LENGTH / TARGET_SR)
    rps_stft = np.zeros((4, n_frames), dtype=np.float32)
    for r in range(4):
        rps_stft[r] = np.interp(stft_times, motor_times, rps_motor[r])
    return rps_stft


def compute_pit_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return per-frame MSE after solving permutation per frame.
    pred, gt: (4, T)
    """
    T = pred.shape[1]
    mse_per_frame = np.zeros(T, dtype=np.float32)
    # Per-frame pairwise MSE: (4, 4, T)
    diff = pred[:, None, :] - gt[None, :, :]  # (4, 4, T)
    costs_all = diff ** 2  # (4, 4, T)
    for t in range(T):
        costs = costs_all[:, :, t]  # (4, 4)
        best_cost = float("inf")
        for perm in _PERMS:
            cost = sum(costs[r, perm[r]].item() for r in range(4))
            if cost < best_cost:
                best_cost = cost
        mse_per_frame[t] = best_cost / 4.0
    return mse_per_frame


def plot_full_sequence(
    audio: torch.Tensor,
    pred_rps: np.ndarray,
    gt_rps: np.ndarray,
    mse_per_frame: np.ndarray,
    title: str,
    out_path_pdf: Path,
    out_path_png: Path,
):
    # ─── Spectrogram ────────────────────────────────────────────────────────
    audio_norm = audio / (audio.abs().max() + 1e-8)
    stft = torch.stft(
        audio_norm.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=torch.hann_window(N_FFT),
        return_complex=True,
    )
    spec_db = 20 * torch.log10(stft.abs().clamp_min(1e-10))
    freqs = np.arange(N_FFT // 2 + 1) * (TARGET_SR / N_FFT)
    times = np.arange(spec_db.shape[1]) * (HOP_LENGTH / TARGET_SR)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.4, 1]})

    # Top: spectrogram (use imshow + rasterized to keep PDF small)
    ax0 = axes[0]
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    im = ax0.imshow(
        spec_db.numpy(),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        rasterized=True,
    )
    ax0.set_ylim([0, 4000])
    ax0.set_ylabel("Frequency [Hz]")
    ax0.set_title(title)
    plt.colorbar(im, ax=ax0, fraction=0.02, pad=0.01, label="dB")

    # Middle: RPS traces
    ax1 = axes[1]
    for r in range(4):
        ax1.plot(times, gt_rps[r], ":", color=ROTOR_COLORS[r], linewidth=1.5, label=f"GT {ROTOR_LABELS[r]}")
    for r in range(4):
        ax1.plot(times, pred_rps[r], "-", color=ROTOR_COLORS[r], linewidth=1.2, alpha=0.9, label=f"Pred {ROTOR_LABELS[r]}")

    ax1.set_ylabel("RPS [rev/s]")
    ax1.set_ylim([0, 150])
    # 4-column legend: GT on left, Pred on right
    handles, labels = ax1.get_legend_handles_labels()
    # Reorder: GT R1-4 then Pred R1-4
    gt_handles = handles[:4]
    pred_handles = handles[4:]
    gt_labels = labels[:4]
    pred_labels = labels[4:]
    # Two-column legend
    ax1.legend(
        gt_handles + pred_handles,
        gt_labels + pred_labels,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handlelength=1.5,
    )

    # Bottom: per-frame MSE
    ax2 = axes[2]
    ax2.fill_between(times, mse_per_frame, alpha=0.3)
    ax2.plot(times, mse_per_frame, linewidth=0.8)
    mean_mse = float(mse_per_frame.mean())
    ax2.axhline(mean_mse, color="red", linestyle="--", linewidth=1)
    ax2.text(times[-1] * 0.98, mean_mse * 1.1, f"mean = {mean_mse:.1f}",
             color="red", fontsize=9, ha="right", va="bottom")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("MSE [(rev/s)$^2$]")
    # Cap y-axis at 99th percentile to avoid outlier distortion
    cap = float(np.percentile(mse_per_frame, 99))
    if cap > 0:
        ax2.set_ylim([0, cap * 1.15])

    plt.tight_layout()
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png, dpi=300)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-load all models
    loaded_models = {}
    for key, (label, arch, ckpt_path) in MODELS.items():
        print(f"Loading {label} ({arch}) ...")
        model = get_model(arch)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        loaded_models[key] = (label, model)

    for rec_id in RECORDINGS:
        rid = rec_id.replace("DREGON_", "")
        print(f"\n=== Processing {rid} ===")
        audio_16k, rps_motor, motor_sr, t0 = load_aligned_data(rec_id)
        audio_16k = audio_16k.to(device)

        # Compute spectrogram dimensions
        n_fft_frames = (audio_16k.shape[1] - 1) // HOP_LENGTH + 1
        gt_rps = interpolate_rps_to_stft(rps_motor, motor_sr, n_fft_frames)
        # Ensure time axis consistency
        times = np.arange(n_fft_frames) * (HOP_LENGTH / TARGET_SR)

        for mkey, (mlabel, model) in loaded_models.items():
            print(f"  Running {mlabel} ...")
            pred = run_inference(audio_16k, model)

            # Truncate to common length
            T = min(pred.shape[1], gt_rps.shape[1])
            pred = pred[:, :T]
            gt_rps_trunc = gt_rps[:, :T]
            times_trunc = times[:T]

            mse_per_frame = compute_pit_mse(pred, gt_rps_trunc)

            title = f"{rid} — {mlabel}"
            out_pdf = OUT_DIR / f"fig_fullseq_{rid}_{mkey}.pdf"
            out_png = OUT_DIR / f"fig_fullseq_{rid}_{mkey}.png"

            plot_full_sequence(
                audio_16k.cpu(),
                pred,
                gt_rps_trunc,
                mse_per_frame,
                title,
                out_pdf,
                out_png,
            )
            print(f"    Saved: {out_pdf.name}, {out_png.name}")

            # Save metrics
            metrics = {
                "recording": rid,
                "model": mkey,
                "mean_mse": float(mse_per_frame.mean()),
                "median_mse": float(np.median(mse_per_frame)),
                "max_mse": float(mse_per_frame.max()),
            }
            with open(OUT_DIR / f"metrics_fullseq_{rid}_{mkey}.json", "w") as f:
                json.dump(metrics, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
