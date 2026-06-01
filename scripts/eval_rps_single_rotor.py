#!/usr/bin/env python3
"""
Evaluate any saved RPS predictor on clean DREGON individual motor recordings
and the allMotors_70 synchronized recording.

Usage:
    python scripts/eval_rps_single_rotor.py \
        --model simple_conv_bigru \
        --checkpoint results/rps_exp_simple_conv_bigru/best_simple_conv_bigru.pt \
        --out_dir results/rps_eval_single_rotor/simple_conv_bigru
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_rps_predictor import get_model

# ─── Config ─────────────────────────────────────────────────────────────────
SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
DATA_DIR = Path("data/DREGON/DREGON_individual_motors_recordings")


def parse_filename(fname: str):
    m = re.match(r"(?:Motor(\d)|allMotors)_(\d+)\.wav", fname)
    if not m:
        return None, None
    motor_id = m.group(1) if m.group(1) is not None else "all"
    return motor_id, int(m.group(2))


def load_and_trim(path: Path, target_sr: int = SR, trim_ratio: float = 0.3):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),
            orig_freq=sr, new_freq=target_sr,
        ).numpy()[0]
    n_samples = len(audio)
    start = int(n_samples * trim_ratio)
    end = int(n_samples * (1 - trim_ratio))
    return audio[start:end]


def evaluate_single_rotor(pred, target_rps):
    pred = np.asarray(pred)
    target = np.full_like(pred, fill_value=target_rps, dtype=np.float32)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-6 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2, "mean_pred": float(pred.mean()), "std_pred": float(pred.std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Loading model ({args.model}) ===")
    model = get_model(args.model, n_fft=N_FFT, hop_length=HOP_LENGTH, num_rotors=4)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    files = sorted(DATA_DIR.glob("*.wav"))
    results = []

    for fpath in files:
        fname = fpath.name
        motor_id, rps = parse_filename(fname)
        if motor_id is None:
            continue

        print(f"\n--- {fname} (motor={motor_id}, target RPS={rps}) ---")
        audio = load_and_trim(fpath)
        duration = len(audio) / SR
        print(f"  Trimmed duration: {duration:.2f}s")

        x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]  # (4, T)

        sample_result = {
            "file": fname,
            "motor_id": motor_id,
            "target_rps": rps,
        }

        # Evaluate each of the 4 outputs
        sc_metrics_per_rotor = []
        for r in range(4):
            m = evaluate_single_rotor(pred[r], rps)
            sc_metrics_per_rotor.append(m)
        avg_mse = float(np.mean([m["mse"] for m in sc_metrics_per_rotor]))
        avg_mae = float(np.mean([m["mae"] for m in sc_metrics_per_rotor]))
        best_idx = int(np.argmin([m["mse"] for m in sc_metrics_per_rotor]))

        sample_result["avg"] = {"mse": avg_mse, "mae": avg_mae}
        sample_result["best_rotor"] = {
            **sc_metrics_per_rotor[best_idx],
            "rotor_idx": best_idx,
        }
        for r in range(4):
            sample_result[f"rotor_{r}"] = sc_metrics_per_rotor[r]

        print(f"  Best rotor (R{best_idx+1}): MSE={sc_metrics_per_rotor[best_idx]['mse']:.2f}, MAE={sc_metrics_per_rotor[best_idx]['mae']:.2f}, mean={sc_metrics_per_rotor[best_idx]['mean_pred']:.1f}")
        print(f"  Avg across 4: MSE={avg_mse:.2f}, MAE={avg_mae:.2f}")

        results.append(sample_result)

    # Aggregate
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for metric in ["best_rotor", "avg"]:
        mses = [r[metric]["mse"] for r in results if metric in r]
        maes = [r[metric]["mae"] for r in results if metric in r]
        label = "Best rotor" if metric == "best_rotor" else "Avg over 4"
        print(f"{label:20s}  MSE={np.mean(mses):7.2f}±{np.std(mses):5.2f}   MAE={np.mean(maes):5.2f}±{np.std(maes):4.2f}")

    # Save JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"model": args.model, "checkpoint": str(args.checkpoint), "results": results}, f, indent=2)
    print(f"\nSaved results to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
