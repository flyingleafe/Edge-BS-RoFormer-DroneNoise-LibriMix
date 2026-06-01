#!/usr/bin/env python3
"""
Evaluate the best RPS predictor on:
  1) 5 evenly-spaced DREGON-LM validation samples  (noise+speech mixtures)
  2) External drone recordings with DJI flight logs  (noise-only)

Saves per-sample .npz triples (log_mag, rps_target, rps_pred) and a
summary plot to  results/rps_predictor/.
"""

import os
import glob

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from train_rps_predictor import RPSPredictor
from data_processing.external_recordings import load_all_external_recordings

# ── constants ──────────────────────────────────────────────────────────────
CHECKPOINT = "results/rps_predictor/best.pt"
DREGON_LM_VALID = "datasets/DREGON-LM/valid"
EXT_DATA_ROOT = "data/recording_with_motor_speed"
OUT_DIR = "results/rps_predictor"
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")

N_FFT = 2048
HOP = 512
MODEL_SR = 16000          # model was trained at 16 kHz mono
CHUNK_SAMPLES = 131584    # 8.224 s – same as DREGON-LM samples
DEVICE = "cpu"

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]


# ── helpers ────────────────────────────────────────────────────────────────

def load_model():
    model = RPSPredictor(n_fft=N_FFT, hop_length=HOP)
    model.load_state_dict(
        torch.load(CHECKPOINT, weights_only=True, map_location=DEVICE)
    )
    model.eval()
    return model


def predict(model, audio_16k_mono: np.ndarray) -> np.ndarray:
    """Run model on a 16 kHz mono waveform, return (4, T) RPS prediction."""
    audio_t = torch.from_numpy(audio_16k_mono).float().unsqueeze(0)
    with torch.no_grad():
        pred = model(audio_t).squeeze(0).numpy()  # (4, T)
    return pred


def compute_spectrogram(audio_16k_mono: np.ndarray) -> np.ndarray:
    """Compute log-magnitude spectrogram matching model's internal STFT."""
    audio_t = torch.from_numpy(audio_16k_mono).float().unsqueeze(0)
    X = torch.stft(
        audio_t, n_fft=N_FFT, hop_length=HOP,
        window=torch.hann_window(N_FFT), return_complex=True, normalized=True,
    )
    return np.log1p(X.abs().squeeze(0).numpy())  # (F, T)


def resample_rps_to_frames(rps: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly resample (4, rps_T) to (4, n_frames)."""
    rps_t = torch.from_numpy(rps).float().unsqueeze(0)
    return F.interpolate(
        rps_t, size=n_frames, mode="linear", align_corners=False
    ).squeeze(0).numpy()


def save_sample(
    out_path: str,
    audio: np.ndarray,
    log_mag: np.ndarray,
    rps_target: np.ndarray,
    rps_pred: np.ndarray,
    sr: int = MODEL_SR,
):
    np.savez(
        out_path,
        audio=audio,
        log_mag=log_mag,
        rps_target=rps_target,
        rps_pred=rps_pred,
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP,
    )


def print_metrics(tag: str, rps_pred: np.ndarray, rps_target: np.ndarray):
    mae = np.abs(rps_pred - rps_target).mean()
    mae_per_rotor = np.abs(rps_pred - rps_target).mean(axis=1)
    mse = ((rps_pred - rps_target) ** 2).mean()
    ss_res = ((rps_pred - rps_target) ** 2).sum()
    ss_tot = ((rps_target - rps_target.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {tag}: MAE={mae:.2f} RPS  RMSE={mse**0.5:.2f}  R²={r2:.4f}  "
          f"per-rotor=[{', '.join(f'{v:.2f}' for v in mae_per_rotor)}]")
    return {"mae": mae, "rmse": mse ** 0.5, "r2": r2}


# ── 1. DREGON-LM validation samples ───────────────────────────────────────

def evaluate_dregon_lm(model, n_samples=5):
    """Evaluate on n_samples from DREGON-LM validation set."""
    sample_dirs = sorted(
        d for d in glob.glob(os.path.join(DREGON_LM_VALID, "sample_*"))
        if os.path.isfile(os.path.join(d, "mixture.wav"))
        and os.path.isfile(os.path.join(d, "rps.npy"))
    )
    indices = np.linspace(0, len(sample_dirs) - 1, n_samples, dtype=int)
    selected = [sample_dirs[i] for i in indices]

    print(f"\n{'='*70}")
    print(f"DREGON-LM validation  ({n_samples} samples)")
    print(f"{'='*70}")

    results = []
    for idx, d in enumerate(selected):
        name = os.path.basename(d)

        audio, sr = torchaudio.load(os.path.join(d, "mixture.wav"))
        audio_np = audio[0].numpy()  # mono

        rps_raw = np.load(os.path.join(d, "rps.npy"))  # (4, rps_T)
        n_frames = audio_np.shape[0] // HOP + 1
        rps_target = resample_rps_to_frames(rps_raw, n_frames)

        rps_pred = predict(model, audio_np)
        log_mag = compute_spectrogram(audio_np)

        out_path = os.path.join(SAMPLES_DIR, f"dregon_lm_{name}.npz")
        save_sample(out_path, audio_np, log_mag, rps_target, rps_pred)

        m = print_metrics(f"[{idx+1}/{n_samples}] {name}", rps_pred, rps_target)
        results.append((f"dregon_lm_{name}", m))

    return results


# ── 2. External drone recordings ──────────────────────────────────────────

def evaluate_external(model, n_chunks_per_recording=5):
    """
    Evaluate on external drone recordings.

    Segments each recording into non-overlapping chunks (same length as
    DREGON-LM), picks evenly-spaced ones, and evaluates.
    """
    print(f"\n{'='*70}")
    print(f"External recordings  ({n_chunks_per_recording} chunks each)")
    print(f"{'='*70}")

    records = load_all_external_recordings(EXT_DATA_ROOT)
    results = []

    for rec in records:
        print(f"\n--- {rec.recording_id} ---")

        # Valid in-flight region: where all 4 motors > 5 RPS
        in_flight = rec.motors.measured.min(axis=1) > 5
        if in_flight.sum() == 0:
            print("  No in-flight data, skipping.")
            continue
        flight_times = rec.motors.timestamps[in_flight]
        flight_start = float(flight_times[0])
        flight_end = float(flight_times[-1])
        print(f"  In-flight region: {flight_start:.1f}–{flight_end:.1f} s")

        # Compute chunk duration in original sample rate
        chunk_dur = CHUNK_SAMPLES / MODEL_SR  # 8.224 s
        n_possible = int((flight_end - flight_start) / chunk_dur)
        if n_possible < 1:
            print("  In-flight region too short for a full chunk, skipping.")
            continue

        # Pick evenly-spaced chunks
        n_chunks = min(n_chunks_per_recording, n_possible)
        chunk_starts_sec = np.linspace(
            flight_start,
            flight_end - chunk_dur,
            n_chunks,
        )

        for ci, t0 in enumerate(chunk_starts_sec):
            t1 = t0 + chunk_dur

            # Slice audio (still at original SR, 8-channel)
            sliced = rec.slice_by_time(t0, t1)
            audio_orig = sliced.audio[:, 0]  # channel 0, mono

            # Resample to 16 kHz
            audio_16k = librosa.resample(
                audio_orig.astype(np.float32),
                orig_sr=rec.sample_rate,
                target_sr=MODEL_SR,
            )
            # Ensure exact length
            if len(audio_16k) > CHUNK_SAMPLES:
                audio_16k = audio_16k[:CHUNK_SAMPLES]
            elif len(audio_16k) < CHUNK_SAMPLES:
                audio_16k = np.pad(
                    audio_16k, (0, CHUNK_SAMPLES - len(audio_16k))
                )

            # Ground-truth RPS from CSV (at native ~30 Hz)
            rps_native = sliced.motors.measured.T  # (4, n_motor_samples)
            n_frames = CHUNK_SAMPLES // HOP + 1
            rps_target = resample_rps_to_frames(rps_native, n_frames)

            # Predict
            rps_pred = predict(model, audio_16k)
            log_mag = compute_spectrogram(audio_16k)

            tag = f"{rec.recording_id}_chunk{ci:02d}"
            out_path = os.path.join(SAMPLES_DIR, f"{tag}.npz")
            save_sample(out_path, audio_16k, log_mag, rps_target, rps_pred)

            m = print_metrics(
                f"[{ci+1}/{n_chunks}] t={t0:.1f}–{t1:.1f}s", rps_pred, rps_target
            )
            results.append((tag, m))

    return results


# ── 3. Combined plot ──────────────────────────────────────────────────────

def plot_all_samples(dregon_results, ext_results):
    """
    Plot spectrogram + target + predicted for every saved sample.
    """
    npz_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.npz")))
    n = len(npz_files)
    if n == 0:
        print("No samples to plot.")
        return

    fig = plt.figure(figsize=(14, 4.2 * n))
    gs = GridSpec(n * 3, 1, hspace=0.45)

    for i, path in enumerate(npz_files):
        data = np.load(path)
        log_mag = data["log_mag"]
        rps_tgt = data["rps_target"]
        rps_prd = data["rps_pred"]
        hop = int(data["hop_length"])
        sr = int(data["sample_rate"])
        name = os.path.splitext(os.path.basename(path))[0]

        T = rps_tgt.shape[1]
        time_sec = np.arange(T) * hop / sr

        # --- spectrogram ---
        ax = fig.add_subplot(gs[i * 3])
        im = ax.imshow(
            log_mag, aspect="auto", origin="lower",
            extent=[0, time_sec[-1], 0, sr / 2],
            cmap="magma",
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(f"{name} — Input Spectrogram", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 4000)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02, label="log(1+|X|)")

        # --- target RPS ---
        ax = fig.add_subplot(gs[i * 3 + 1])
        for r in range(4):
            ax.plot(time_sec, rps_tgt[r], color=ROTOR_COLORS[r],
                    label=ROTOR_LABELS[r], lw=1)
        ax.set_ylabel("RPS")
        ax.set_title("Target Motor Speeds", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)

        # --- predicted RPS ---
        ax = fig.add_subplot(gs[i * 3 + 2])
        for r in range(4):
            ax.plot(time_sec, rps_tgt[r], color=ROTOR_COLORS[r],
                    alpha=0.25, lw=1, ls="--")
            ax.plot(time_sec, rps_prd[r], color=ROTOR_COLORS[r],
                    label=ROTOR_LABELS[r], lw=1)
        mae = np.abs(rps_prd - rps_tgt).mean()
        ax.set_ylabel("RPS")
        ax.set_title(
            f"Predicted (MAE={mae:.2f} RPS, dashed=target)", fontsize=9
        )
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "RPS Predictor — Evaluation Samples\n"
        "(DREGON-LM validation + External recordings)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    out_path = os.path.join(OUT_DIR, "sample_predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # Clear old samples
    for f in glob.glob(os.path.join(SAMPLES_DIR, "*.npz")):
        os.remove(f)

    model = load_model()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    dregon_results = evaluate_dregon_lm(model)
    ext_results = evaluate_external(model)

    # ── summary table ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sample':<45} {'MAE':>6} {'RMSE':>7} {'R²':>8}")
    print("-" * 70)

    all_results = dregon_results + ext_results
    for name, m in all_results:
        print(f"{name:<45} {m['mae']:6.2f} {m['rmse']:7.2f} {m['r2']:8.4f}")

    # Aggregate by dataset
    if dregon_results:
        avg_mae = np.mean([m["mae"] for _, m in dregon_results])
        avg_rmse = np.mean([m["rmse"] for _, m in dregon_results])
        print(f"\n  DREGON-LM avg:  MAE={avg_mae:.2f}  RMSE={avg_rmse:.2f}")
    if ext_results:
        avg_mae = np.mean([m["mae"] for _, m in ext_results])
        avg_rmse = np.mean([m["rmse"] for _, m in ext_results])
        print(f"  External avg:   MAE={avg_mae:.2f}  RMSE={avg_rmse:.2f}")

    # Plot
    plot_all_samples(dregon_results, ext_results)


if __name__ == "__main__":
    main()
