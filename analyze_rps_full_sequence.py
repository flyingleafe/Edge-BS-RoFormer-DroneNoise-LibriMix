#!/usr/bin/env python3
"""
Run inference on the FULL high-SNR free-flight recording and produce a
3-panel figure:
  1. spectrogram of the whole sequence
  2. predicted vs ground-truth rotor speeds over time
  3. per-frame MSE variation with time

The recording is DREGON_free-flight_speech-high_room1, which contains
takeoff → flight → landing in a single continuous ~47 s sequence
(where motor data is available; audio is ~53 s).

Usage:
    python analyze_rps_full_sequence.py
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_rps_predictor import SimpleConv

# ─── Config ─────────────────────────────────────────────────────────────────
DREGON_DIR = Path("data/DREGON")
RECORDING = "DREGON_free-flight_speech-high_room1"
TARGET_SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paper figure style
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
    """
    Load audio and motor data, crop audio to the motor timestamp range,
    and resample audio to 16 kHz mono.

    Returns
    -------
    audio_16k : torch.Tensor, shape (1, T_audio_16k)
    rps_motor : np.ndarray, shape (4, N_motor)
    motor_sr  : float
    t0        : float — absolute Unix timestamp of the first motor sample
    """
    rec_dir = DREGON_DIR / recording_id

    # ── Audio ──
    audio_path = rec_dir / f"{recording_id}.wav"
    audio_full, sr = torchaudio.load(str(audio_path))  # (C, S)

    audio_ts_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_audiots.mat")
    audio_ts = audio_ts_mat["audio_timestamps"].flatten()

    # ── Motor ──
    motor_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_motors.mat")
    motor_data = motor_mat["motor"][0, 0]
    measured = motor_data["measured"]           # (N_motor, 4)
    motor_ts = motor_data["timestamps"].flatten()
    motor_sr = len(motor_ts) / (motor_ts[-1] - motor_ts[0])

    # Crop audio to the motor timestamp range (where ground truth exists)
    t0 = motor_ts[0]
    t1 = motor_ts[-1]
    audio_start_idx = int((t0 - audio_ts[0]) * sr)
    audio_end_idx   = int((t1 - audio_ts[0]) * sr)
    audio_start_idx = max(0, audio_start_idx)
    audio_end_idx   = min(audio_full.shape[1], audio_end_idx)
    audio_crop = audio_full[:, audio_start_idx:audio_end_idx]

    # Mix to mono and resample to 16 kHz
    audio_mono = audio_crop.mean(dim=0, keepdim=True)  # (1, S)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio_16k = resampler(audio_mono)
    else:
        audio_16k = audio_mono

    # Normalise to ~0.9 peak (same as training / eval scripts)
    peak = audio_16k.abs().max()
    if peak > 0:
        audio_16k = audio_16k / peak * 0.9

    # Motor RPS as (4, N_motor)
    rps_motor = measured.T.astype(np.float32)

    return audio_16k, rps_motor, motor_sr, float(t0)


def compute_stft_frames(audio_length: int, hop_length: int) -> int:
    """Number of STFT frames with center=True."""
    return audio_length // hop_length + 1


def run_inference(audio: torch.Tensor, model: SimpleConv, device: str) -> np.ndarray:
    """
    Run model on the full audio tensor.

    Parameters
    ----------
    audio : (1, S) or (S,)

    Returns
    -------
    pred : np.ndarray, shape (4, T_stft)
    """
    model.eval()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.to(device)
    with torch.no_grad():
        pred = model(audio)
    return pred.cpu().numpy()[0]  # (4, T_stft)


def interpolate_rps_to_stft(
    rps_motor: np.ndarray, motor_sr: float, n_frames: int
) -> np.ndarray:
    """
    Interpolate motor RPS from motor sampling grid to STFT frame grid.

    Parameters
    ----------
    rps_motor : (4, N_motor)
    motor_sr  : motor sample rate in Hz
    n_frames  : number of STFT frames

    Returns
    -------
    rps_stft : (4, n_frames)
    """
    motor_times = np.arange(rps_motor.shape[1]) / motor_sr
    stft_times = np.arange(n_frames) * (HOP_LENGTH / TARGET_SR)
    rps_stft = np.zeros((4, n_frames), dtype=np.float32)
    for r in range(4):
        rps_stft[r] = np.interp(stft_times, motor_times, rps_motor[r])
    return rps_stft


def smooth(x: np.ndarray, window_sec: float = 1.0) -> np.ndarray:
    """Moving-average smoothing.  window_sec in seconds."""
    frame_dur = HOP_LENGTH / TARGET_SR
    w = max(1, int(window_sec / frame_dur))
    if w <= 1:
        return x
    # Convolution with uniform kernel, same length output
    kernel = np.ones(w) / w
    smoothed = np.convolve(x, kernel, mode="same")
    return smoothed


def generate_figure(
    audio_np: np.ndarray,
    rps_gt: np.ndarray,
    rps_pred: np.ndarray,
    mse_trace: np.ndarray,
    out_path: Path,
    title_suffix: str = "",
):
    """
    3-panel figure.

    Parameters
    ----------
    audio_np   : (S,)  mono audio at TARGET_SR
    rps_gt     : (4, T_stft)
    rps_pred   : (4, T_stft)
    mse_trace  : (T_stft,)  per-frame MSE (already smoothed)
    out_path   : where to save
    title_suffix : optional string appended to spectrogram title
    """
    duration = len(audio_np) / TARGET_SR
    t_audio = np.linspace(0, duration, len(audio_np))
    t_stft = np.linspace(0, duration, rps_gt.shape[1])

    # Identify low-RPS regions (takeoff / landing) for highlighting
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
    ax.set_title(f"DREGON free-flight speech-high {title_suffix}".strip())
    ax.set_xticklabels([])
    ax.grid(False)
    # Annotate takeoff / landing
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)

    # ── Panel 2: rotor speeds (GT + predicted) ──
    ax = axes[1]
    for r in range(4):
        ax.plot(t_stft, rps_gt[r], ":", color=ROTOR_COLORS[r], lw=0.5, alpha=0.55)
        ax.plot(t_stft, rps_pred[r], "-", color=ROTOR_COLORS[r], lw=0.5, alpha=0.75)

    # Highlight takeoff / landing regions
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)

    # Legend: GT vs predicted + rotor colours
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

    # ── Panel 3: per-frame MSE ──
    ax = axes[2]
    # Highlight takeoff / landing
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)
    ax.plot(t_stft, mse_trace, "-", color="#d62728", lw=0.8)
    ax.fill_between(t_stft, mse_trace, alpha=0.15, color="#d62728")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.set_xlim(0, duration)
    # Log scale if dynamic range is large
    if mse_trace.max() / (mse_trace[mse_trace > 0].min() + 1e-8) > 10:
        ax.set_yscale("log")
    # Reference line for average synthetic-val MSE
    ax.axhline(5.15, ls="--", lw=0.8, color="#444", alpha=0.7,
               label="held-out synthetic, MSE = 5.15")
    ax.legend(frameon=False, loc="upper center", fontsize=7,
              bbox_to_anchor=(0.5, 1.02))

    fig.savefig(out_path, bbox_inches="tight")
    # Also save PNG for quick inspection
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out_path, "and", out_path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description="Full-sequence RPS inference + plot")
    parser.add_argument("--recording", default=RECORDING)
    parser.add_argument("--checkpoint", default="results/rps_predictor/best.pt")
    parser.add_argument("--model_type", default="simple_conv", choices=["simple_conv", "dcunet", "dccrn"])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--smooth_window", type=float, default=1.0,
                        help="Moving-average window for MSE trace (seconds)")
    parser.add_argument("--out_dir", default="results/rps_full_sequence")
    parser.add_argument("--fig_dir", default="papers/rps-from-drone-sound/figures")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"\n=== Loading {args.recording} ===")
    audio_16k, rps_motor, motor_sr, t0 = load_aligned_data(args.recording)
    audio_np = audio_16k.numpy()[0]  # (S,)
    duration = len(audio_np) / TARGET_SR
    print(f"Audio (16 kHz): {audio_16k.shape[1]} samples, {duration:.1f} s")
    print(f"Motor data: {rps_motor.shape[1]} frames, SR={motor_sr:.1f} Hz")

    # ── Load model ──
    print(f"\n=== Loading model ({args.model_type}) ===")
    if args.model_type == "simple_conv":
        model = SimpleConv(n_fft=N_FFT, hop_length=HOP_LENGTH)
    else:
        # Fallback imports if needed later
        from train_rps_predictor import DCUNetEncRPS, DCCRNEncRPS
        if args.model_type == "dcunet":
            model = DCUNetEncRPS(n_fft=N_FFT, hop_length=HOP_LENGTH)
        else:
            model = DCCRNEncRPS(n_fft=N_FFT, hop_length=HOP_LENGTH)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # ── Inference ──
    print("\n=== Running inference on full sequence ===")
    rps_pred = run_inference(audio_16k, model, device)
    n_frames = compute_stft_frames(audio_16k.shape[1], HOP_LENGTH)
    print(f"Predicted RPS shape: {rps_pred.shape}, expected ~{n_frames} frames")

    # ── Ground truth at STFT rate ──
    rps_gt = interpolate_rps_to_stft(rps_motor, motor_sr, rps_pred.shape[1])
    print(f"Ground-truth RPS shape: {rps_gt.shape}")

    # ── Per-frame metrics ──
    # Squared error per rotor, then average
    se_per_rotor = (rps_pred - rps_gt) ** 2          # (4, T)
    mse_per_frame = se_per_rotor.mean(axis=0)         # (T,)
    mse_smooth = smooth(mse_per_frame, window_sec=args.smooth_window)

    # Global metrics
    mse_global = float(mse_per_frame.mean())
    mae_global = float(np.abs(rps_pred - rps_gt).mean())
    ss_res = float(np.sum((rps_gt - rps_pred) ** 2))
    ss_tot = float(np.sum((rps_gt - rps_gt.mean()) ** 2))
    r2_global = 1.0 - ss_res / (ss_tot + 1e-8)

    # In-flight metrics (rotors spinning > 50 rev/s — excludes takeoff / landing)
    in_flight = np.all(rps_gt > 50, axis=0)
    mse_inflight = float(mse_per_frame[in_flight].mean()) if in_flight.sum() > 0 else None
    mae_inflight = float(np.abs(rps_pred - rps_gt)[:, in_flight].mean()) if in_flight.sum() > 0 else None

    print(f"\n=== Global metrics ===")
    print(f"  MSE : {mse_global:.3f}")
    print(f"  MAE : {mae_global:.3f}")
    print(f"  R²  : {r2_global:.4f}")
    if mse_inflight is not None:
        print(f"\n=== In-flight metrics (RPS > 50 rev/s) ===")
        print(f"  MSE : {mse_inflight:.3f}")
        print(f"  MAE : {mae_inflight:.3f}")
        print(f"  frames: {in_flight.sum()} / {len(in_flight)}")

    # ── Save raw data ──
    np.save(out_dir / "audio_16k.npy", audio_np)
    np.save(out_dir / "rps_pred.npy", rps_pred)
    np.save(out_dir / "rps_gt_stft.npy", rps_gt)
    np.save(out_dir / "mse_per_frame.npy", mse_per_frame)

    metrics = {
        "recording": args.recording,
        "duration_sec": duration,
        "n_frames": int(rps_pred.shape[1]),
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint),
        "mse": mse_global,
        "mae": mae_global,
        "r2": r2_global,
        "mse_inflight": mse_inflight,
        "mae_inflight": mae_inflight,
        "inflight_frames": int(in_flight.sum()),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved raw data to {out_dir}")

    # ── Plot ──
    print("\n=== Generating figure ===")
    out_fig = fig_dir / "fig_full_sequence.pdf"
    generate_figure(
        audio_np, rps_gt, rps_pred, mse_smooth,
        out_path=out_fig,
        title_suffix="(full sequence)",
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
