#!/usr/bin/env python3
"""
Evaluate any saved RPS predictor on the full DREGON free-flight speech-high room1
recording (~47 s with motor telemetry).

Usage:
    python scripts/eval_rps_full_sequence.py \
        --model simple_conv_bigru \
        --checkpoint results/rps_exp_simple_conv_bigru/best_simple_conv_bigru.pt \
        --out_dir results/rps_eval_full_sequence/simple_conv_bigru
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_rps_predictor import get_model

# ─── Config ─────────────────────────────────────────────────────────────────
DREGON_DIR = Path("data/DREGON")
RECORDING = "DREGON_free-flight_speech-high_room1"
TARGET_SR = 16000
N_FFT = 2048
HOP_LENGTH = 512

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

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]


def load_aligned_data(recording_id: str):
    """Load audio and motor data, crop audio to motor timestamp range, resample to 16 kHz mono."""
    rec_dir = DREGON_DIR / recording_id

    audio_path = rec_dir / f"{recording_id}.wav"
    audio_full, sr = torchaudio.load(str(audio_path))

    audio_ts_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_audiots.mat")
    audio_ts = audio_ts_mat["audio_timestamps"].flatten()

    motor_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_motors.mat")
    motor_data = motor_mat["motor"][0, 0]
    measured = motor_data["measured"]
    motor_ts = motor_data["timestamps"].flatten()
    motor_sr = len(motor_ts) / (motor_ts[-1] - motor_ts[0])

    t0 = motor_ts[0]
    t1 = motor_ts[-1]
    audio_start_idx = int((t0 - audio_ts[0]) * sr)
    audio_end_idx = int((t1 - audio_ts[0]) * sr)
    audio_start_idx = max(0, audio_start_idx)
    audio_end_idx = min(audio_full.shape[1], audio_end_idx)
    audio_crop = audio_full[:, audio_start_idx:audio_end_idx]

    audio_mono = audio_crop.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio_16k = resampler(audio_mono)
    else:
        audio_16k = audio_mono

    peak = audio_16k.abs().max()
    if peak > 0:
        audio_16k = audio_16k / peak * 0.9

    rps_motor = measured.T.astype(np.float32)
    return audio_16k, rps_motor, motor_sr, float(t0)


def run_inference(audio: torch.Tensor, model: torch.nn.Module, device: str) -> np.ndarray:
    model.eval()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.to(device)
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


def smooth(x: np.ndarray, window_sec: float = 1.0) -> np.ndarray:
    frame_dur = HOP_LENGTH / TARGET_SR
    w = max(1, int(window_sec / frame_dur))
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def compute_metrics(rps_pred: np.ndarray, rps_gt: np.ndarray) -> dict:
    se_per_rotor = (rps_pred - rps_gt) ** 2
    mse_per_frame = se_per_rotor.mean(axis=0)
    mse_global = float(mse_per_frame.mean())
    mae_global = float(np.abs(rps_pred - rps_gt).mean())
    ss_res = float(np.sum((rps_gt - rps_pred) ** 2))
    ss_tot = float(np.sum((rps_gt - rps_gt.mean()) ** 2))
    r2_global = 1.0 - ss_res / (ss_tot + 1e-8)

    in_flight = np.all(rps_gt > 50, axis=0)
    mse_inflight = float(mse_per_frame[in_flight].mean()) if in_flight.sum() > 0 else None
    mae_inflight = float(np.abs(rps_pred - rps_gt)[:, in_flight].mean()) if in_flight.sum() > 0 else None

    return {
        "mse": mse_global,
        "mae": mae_global,
        "r2": r2_global,
        "mse_inflight": mse_inflight,
        "mae_inflight": mae_inflight,
        "inflight_frames": int(in_flight.sum()),
        "total_frames": int(len(mse_per_frame)),
    }


def generate_figure(audio_np, rps_gt, rps_pred, mse_trace, out_path, title_suffix=""):
    duration = len(audio_np) / TARGET_SR
    t_stft = np.linspace(0, duration, rps_gt.shape[1])

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

    ax = axes[0]
    n_fft, hop = N_FFT, HOP_LENGTH
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
    ax.set_title(f"{title_suffix}".strip())
    ax.set_xticklabels([])
    ax.grid(False)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)

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
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r])
        for r in range(4)
    ]
    ax.legend(handles=legend_handles, loc="lower center", frameon=False,
              fontsize=7, ncol=3, columnspacing=0.8,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_ylabel("rotor speed [rev/s]")
    ax.set_xlim(0, duration)
    ax.set_xticklabels([])

    ax = axes[2]
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)
    ax.plot(t_stft, mse_trace, "-", color="#d62728", lw=0.8)
    ax.fill_between(t_stft, mse_trace, alpha=0.15, color="#d62728")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.set_xlim(0, duration)
    if mse_trace.max() / (mse_trace[mse_trace > 0].min() + 1e-8) > 10:
        ax.set_yscale("log")
    ax.axhline(5.15, ls="--", lw=0.8, color="#444", alpha=0.7,
               label="held-out synthetic, MSE = 5.15")
    ax.legend(frameon=False, loc="upper center", fontsize=7,
              bbox_to_anchor=(0.5, 1.02))

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out_path, "and", out_path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recording", default=RECORDING)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smooth_window", type=float, default=1.0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--plot", action="store_true", help="Generate figure")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Loading {args.recording} ===")
    audio_16k, rps_motor, motor_sr, t0 = load_aligned_data(args.recording)
    audio_np = audio_16k.numpy()[0]
    duration = len(audio_np) / TARGET_SR
    print(f"Audio (16 kHz): {audio_16k.shape[1]} samples, {duration:.1f} s")
    print(f"Motor data: {rps_motor.shape[1]} frames, SR={motor_sr:.1f} Hz")

    print(f"\n=== Loading model ({args.model}) ===")
    model = get_model(args.model, n_fft=N_FFT, hop_length=HOP_LENGTH, num_rotors=4)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    print("\n=== Running inference on full sequence ===")
    rps_pred = run_inference(audio_16k, model, device)
    print(f"Predicted RPS shape: {rps_pred.shape}")

    rps_gt = interpolate_rps_to_stft(rps_motor, motor_sr, rps_pred.shape[1])
    print(f"Ground-truth RPS shape: {rps_gt.shape}")

    se_per_rotor = (rps_pred - rps_gt) ** 2
    mse_per_frame = se_per_rotor.mean(axis=0)
    mse_smooth = smooth(mse_per_frame, window_sec=args.smooth_window)

    metrics = compute_metrics(rps_pred, rps_gt)
    print(f"\n=== Global metrics ===")
    print(f"  MSE : {metrics['mse']:.3f}")
    print(f"  MAE : {metrics['mae']:.3f}")
    print(f"  R²  : {metrics['r2']:.4f}")
    if metrics['mse_inflight'] is not None:
        print(f"\n=== In-flight metrics (RPS > 50 rev/s) ===")
        print(f"  MSE : {metrics['mse_inflight']:.3f}")
        print(f"  MAE : {metrics['mae_inflight']:.3f}")
        print(f"  frames: {metrics['inflight_frames']} / {metrics['total_frames']}")

    np.save(out_dir / "rps_pred.npy", rps_pred)
    np.save(out_dir / "rps_gt_stft.npy", rps_gt)
    np.save(out_dir / "mse_per_frame.npy", mse_per_frame)

    full_metrics = {
        "recording": args.recording,
        "duration_sec": duration,
        "n_frames": int(rps_pred.shape[1]),
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        **metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Saved metrics to {out_dir / 'metrics.json'}")

    if args.plot:
        print("\n=== Generating figure ===")
        out_fig = out_dir / "fig_full_sequence.pdf"
        generate_figure(
            audio_np, rps_gt, rps_pred, mse_smooth,
            out_path=out_fig,
            title_suffix=f"{args.model} — {args.recording}",
        )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
