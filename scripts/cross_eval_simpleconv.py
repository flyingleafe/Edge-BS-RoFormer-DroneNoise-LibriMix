#!/usr/bin/env python3
"""
Cross-evaluate SimpleConv checkpoints on DREGON-LM and DREGON-LM-V3 validation sets.

Reports per-channel metrics for V3 (uses metadata.json to assign channels).
Saves results as JSON.

Usage:
    python scripts/cross_eval_simpleconv.py \
        --ckpt results/rps_exp_simple_conv/best_simple_conv.pt \
        --lm_valid datasets/DREGON-LM/valid \
        --v3_valid datasets/DREGON-LM-V3/valid \
        --v3_meta datasets/DREGON-LM-V3/metadata.json \
        --output results/cross_eval_old.json

    python scripts/cross_eval_simpleconv.py \
        --ckpt results/rps_predictor_comparison/best_simple_conv.pt \
        --lm_valid datasets/DREGON-LM/valid \
        --v3_valid datasets/DREGON-LM-V3/valid \
        --v3_meta datasets/DREGON-LM-V3/metadata.json \
        --output results/cross_eval_new.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the dataset and model loading from train_rps_predictor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_rps_predictor import DREGONRPSDataset, get_model, pit_mse_loss


def load_model(ckpt_path, device):
    """Load a SimpleConv checkpoint."""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Determine model name and params from the path or defaults
    ckpt_name = Path(ckpt_path).stem  # "best_simple_conv"
    if "simple_conv" in ckpt_name:
        model_name = ckpt_name.replace("best_", "")
    else:
        model_name = "simple_conv"

    n_fft = 2048
    hop = 512

    model = get_model(model_name, n_fft=n_fft, hop_length=hop).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, state_dict, n_fft, hop, model_name


def evaluate(model, loader, device, dataset_len):
    """Run evaluation, returning metrics dict + all predictions/targets."""
    model.eval()
    total_pit_loss = 0.0
    total_std_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for audio, rps_target in loader:
            audio, rps_target = audio.to(device), rps_target.to(device)
            with torch.amp.autocast("cuda"):
                rps_pred = model(audio)
                pit_loss = pit_mse_loss(rps_pred, rps_target)
                std_loss = F.mse_loss(rps_pred, rps_target)
            total_pit_loss += pit_loss.item() * audio.size(0)
            total_std_loss += std_loss.item() * audio.size(0)
            all_preds.append(rps_pred.cpu())
            all_targets.append(rps_target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    pit_mse_val = total_pit_loss / dataset_len
    std_mse_val = total_std_loss / dataset_len
    mae_frame = (all_preds - all_targets).abs().mean().item()
    mae_clip = (all_preds.mean(dim=2) - all_targets.mean(dim=2)).abs().mean().item()

    return {
        "pit_mse": pit_mse_val,
        "std_mse": std_mse_val,
        "rmse_pit": pit_mse_val ** 0.5,
        "rmse_std": std_mse_val ** 0.5,
        "mae_frame": mae_frame,
        "mae_clip": mae_clip,
    }, all_preds, all_targets


def evaluate_per_channel(model, data_dir, meta_path, device, batch_size=128, n_fft=2048, hop_length=512):
    """
    Evaluate on V3 validation, grouping samples by DREGON channel.
    Returns per-channel metrics + aggregate.
    """
    # Build dataset and index by channel
    ds = DREGONRPSDataset(data_dir, n_fft=n_fft, hop_length=hop_length)

    # Load metadata to map sample ID -> channel
    with open(meta_path) as f:
        meta = json.load(f)

    # Build channel map from noise_source field
    # Format: "free-flight_nosource_room1_ch3" -> channel "ch3"
    sample_to_channel = {}
    for entry in meta.get("valid", meta.get("train", [])):
        noise_src = entry["noise_source"]  # e.g. "free-flight_nosource_room1_ch3"
        channel = noise_src.split("_")[-1]  # "ch3"
        sample_to_channel[entry["id"]] = channel

    # Group sample indices by channel
    channel_indices = {}
    for idx, sample_path in enumerate(ds.samples):
        sample_id = os.path.basename(sample_path)  # "sample_00042"
        channel = sample_to_channel.get(sample_id, "unknown")
        channel_indices.setdefault(channel, []).append(idx)

    print(f"\n  Channels found: {sorted(channel_indices.keys())}")
    for ch in sorted(channel_indices.keys()):
        print(f"    {ch}: {len(channel_indices[ch])} samples")

    # Evaluate per channel
    per_channel = {}
    all_preds_list, all_targets_list = [], []

    for channel in sorted(channel_indices.keys()):
        indices = channel_indices[channel]
        # Create a subset DataLoader
        subset = torch.utils.data.Subset(ds, indices)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        metrics, preds, targets = evaluate(model, loader, device, len(indices))
        per_channel[channel] = metrics
        all_preds_list.append(preds)
        all_targets_list.append(targets)

    # Aggregate over all channels
    all_preds = torch.cat(all_preds_list)
    all_targets = torch.cat(all_targets_list)

    aggregate = {
        "pit_mse": float(F.mse_loss(all_preds, all_targets).item()),  # NOT PIT here — order matches
        "std_mse": float(F.mse_loss(all_preds, all_targets).item()),
        "rmse": float(F.mse_loss(all_preds, all_targets).item() ** 0.5),
        "mae_frame": float((all_preds - all_targets).abs().mean().item()),
        "mae_clip": float((all_preds.mean(dim=2) - all_targets.mean(dim=2)).abs().mean().item()),
    }

    # Prefix with _ch for readability
    per_channel_out = {f"_{ch}": m for ch, m in per_channel.items()}

    return aggregate, per_channel_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--lm_valid", default="datasets/DREGON-LM/valid")
    parser.add_argument("--v3_valid", default="datasets/DREGON-LM-V3/valid")
    parser.add_argument("--v3_meta", default="datasets/DREGON-LM-V3/metadata.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, state, n_fft, hop, model_name = load_model(args.ckpt, device)
    print(f"\nCheckpoint: {args.ckpt}")
    print(f"  Model: {model_name}")
    print(f"  Epoch: {state.get('epoch', 'N/A')}")
    print(f"  Val MSE (saved): {state.get('val_mse', 'N/A')}")

    results = {
        "checkpoint": args.ckpt,
        "model_name": model_name,
    }

    # ── Evaluate on DREGON-LM (original) ──
    if os.path.isdir(args.lm_valid):
        print(f"\n{'='*60}")
        print(f"Evaluating on DREGON-LM: {args.lm_valid}")
        ds_lm = DREGONRPSDataset(args.lm_valid, n_fft=n_fft, hop_length=hop)
        loader_lm = DataLoader(
            ds_lm, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        metrics_lm, _, _ = evaluate(model, loader_lm, device, len(ds_lm))
        results["dregon_lm"] = {
            "n_samples": len(ds_lm),
            **{k: float(v) if isinstance(v, (np.floating,)) else v for k, v in metrics_lm.items()},
        }
        print(f"  PIT MSE: {metrics_lm['pit_mse']:.4f}  RMSE: {metrics_lm['rmse_pit']:.2f}  MAE/clip: {metrics_lm['mae_clip']:.2f}")
    else:
        print(f"\nDREGON-LM valid not found: {args.lm_valid} — skipping")

    # ── Evaluate on DREGON-LM-V3 (per channel) ──
    print(f"\n{'='*60}")
    print(f"Evaluating on DREGON-LM-V3 (per-channel): {args.v3_valid}")
    aggr, per_ch = evaluate_per_channel(
        model, args.v3_valid, args.v3_meta, device,
        batch_size=args.batch_size, n_fft=n_fft, hop_length=hop,
    )
    results["dregon_lm_v3"] = {
        "aggregate": {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in aggr.items()},
        "per_channel": {ch: {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in m.items()}
                       for ch, m in per_ch.items()},
    }

    print(f"\n  Aggregate: MSE={aggr['std_mse']:.4f}  RMSE={aggr['rmse']:.2f}  MAE/clip={aggr['mae_clip']:.2f}")
    print(f"  Per-channel:")
    for ch in sorted(per_ch.keys()):
        m = per_ch[ch]
        print(f"    {ch}: MSE={m['std_mse']:7.2f}  RMSE={m['rmse_std']:6.2f}  MAE/clip={m['mae_clip']:6.2f}")

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
