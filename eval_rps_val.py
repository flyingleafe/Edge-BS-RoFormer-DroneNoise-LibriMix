#!/usr/bin/env python3
"""
Run inference with a trained RPS predictor on the full DREGON-LM validation set.

Saves per-sample predicted RPS numpy arrays and an overall metrics summary.

Usage:
    python eval_rps_val.py \
        --checkpoint results/rps_predictor_comparison/best_simple_conv.pt \
        --data_dir   datasets/DREGON-LM/valid \
        --output_dir results/rps_predictor_comparison/val_inference \
        --model      simple_conv

Output structure:
    results/rps_predictor_comparison/val_inference/
    ├── metrics.json                  ← overall MSE, MAE, R², etc.
    └── sample_XXXXX/
        └── predicted_rps.npy         ← (4, T_stft) float32
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from train_rps_predictor import DREGONRPSDataset, get_model


def run_inference(args):
    device = torch.device(args.device)

    # Load dataset (no DataLoader — per-sample to preserve names)
    ds = DREGONRPSDataset(args.data_dir, n_fft=args.n_fft, hop_length=args.hop_length)
    print(f"Validation samples: {len(ds)}")

    # Load model
    model = get_model(args.model, n_fft=args.n_fft, hop_length=args.hop_length).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_mse, all_mae_frame, all_mae_clip = [], [], []
    ss_res_total, ss_tot_total = 0.0, 0.0
    n_frames_total = 0

    t0 = time.time()
    with torch.no_grad():
        for i, sample_dir in enumerate(ds.samples):
            audio, rps_target = ds[i]
            audio = audio.unsqueeze(0).to(device)       # (1, samples)
            rps_target_t = rps_target.to(device)         # (4, T)

            with torch.amp.autocast("cuda"):
                rps_pred = model(audio).squeeze(0)       # (4, T_pred)

            # Align lengths (model output T may differ by 1 from target)
            T = min(rps_pred.shape[-1], rps_target_t.shape[-1])
            rps_pred = rps_pred[..., :T]
            rps_target_t = rps_target_t[..., :T]

            # Per-sample metrics
            mse = F.mse_loss(rps_pred, rps_target_t).item()
            mae_frame = (rps_pred - rps_target_t).abs().mean().item()
            mae_clip = ((rps_pred - rps_target_t).mean(dim=-1).abs()).mean().item()

            all_mse.append(mse)
            all_mae_frame.append(mae_frame)
            all_mae_clip.append(mae_clip)

            # R² accumulators
            ss_res_total += ((rps_pred - rps_target_t) ** 2).sum().item()
            ss_tot_total += ((rps_target_t - rps_target_t.mean()) ** 2).sum().item()
            n_frames_total += T

            # Save prediction
            sample_name = os.path.basename(sample_dir)
            out_dir = os.path.join(args.output_dir, sample_name)
            os.makedirs(out_dir, exist_ok=True)
            np.save(
                os.path.join(out_dir, "predicted_rps.npy"),
                rps_pred.cpu().float().numpy(),
            )

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(ds)}  running MAE/clip={np.mean(all_mae_clip):.3f}")

    elapsed = time.time() - t0

    mean_mse = float(np.mean(all_mse))
    mean_mae_frame = float(np.mean(all_mae_frame))
    mean_mae_clip = float(np.mean(all_mae_clip))
    r2 = float(1 - ss_res_total / ss_tot_total)

    metrics = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "n_samples": len(ds),
        "mse": mean_mse,
        "rmse": mean_mse ** 0.5,
        "mae_frame": mean_mae_frame,
        "mae_clip": mean_mae_clip,
        "r2": r2,
        "elapsed_s": round(elapsed, 1),
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"MSE={mean_mse:.4f}  RMSE={mean_mse**0.5:.3f}  MAE/clip={mean_mae_clip:.3f}  R²={r2:.4f}")
    print(f"Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="results/rps_predictor_comparison/best_simple_conv.pt")
    parser.add_argument("--model", default="simple_conv")
    parser.add_argument("--data_dir", default="datasets/DREGON-LM/valid")
    parser.add_argument("--output_dir",
                        default="results/rps_predictor_comparison/val_inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
