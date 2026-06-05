#!/usr/bin/env python3
"""
Evaluate RPS predictor checkpoints on DREGON-LM-V3 validation set per channel.

Filters:
  - Zero-RPS samples (motors fully off)
  - Spool-up/down samples (RPS mean < threshold, model never trained on these)

Reports per-channel metrics for the in-flight regime the model was trained for.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_rps_predictor import DREGONRPSDataset, get_model, evaluate, pit_mse_loss

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True,
    choices=["simple_conv_v2", "simple_conv_bigru_v2", "simple_conv", "simple_conv_bigru",
             "simple_conv_tcn", "simple_conv_wide", "simple_conv_multiscale",
             "simple_conv_magphase_bigru", "simple_conv_attn_pool", "simple_conv_se_next"])
parser.add_argument("--data_root", default="datasets/DREGON-LM-V3")
parser.add_argument("--device", default="cpu")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--rps_threshold", type=float, default=40.0,
    help="Minimum mean RPS to consider a sample 'in-flight'")
args = parser.parse_args()

device = torch.device(args.device)

# ─── Load metadata and classify samples ─────────────────────────────────────

meta_path = Path(args.data_root) / "metadata.json"
with open(meta_path) as f:
    meta = json.load(f)

channel_samples = defaultdict(list)       # in-flight samples per channel
zero_rps_count = defaultdict(int)
spool_count = defaultdict(int)
all_status = {}  # gidx -> status

for split in ["valid"]:
    for sample in meta[split]:
        sample_id = sample["id"]
        sample_idx = int(sample_id.split("_")[1])

        ns = sample["noise_source"]
        if "_ch" in ns:
            ch = int(ns.rsplit("_ch", 1)[1])
        else:
            ch = 0

        # Load RPS and classify
        rps_path = Path(args.data_root) / split / sample_id / "rps.npy"
        rps = np.load(rps_path)  # (4, T_rps)
        rps_mean = rps.mean()

        if (rps > 0).sum() == 0:
            all_status[sample_idx] = ("zero", ch)
            zero_rps_count[ch] += 1
        elif rps_mean < args.rps_threshold:
            all_status[sample_idx] = ("spool", ch)
            spool_count[ch] += 1
        else:
            all_status[sample_idx] = ("flight", ch)
            channel_samples[ch].append(sample_idx)

total_zero = sum(zero_rps_count.values())
total_spool = sum(spool_count.values())
total_flight = sum(len(v) for v in channel_samples.values())
total_all = total_zero + total_spool + total_flight

print(f"V3 validation set: {total_all} total")
print(f"  Zero RPS (motors off):  {total_zero}")
print(f"  Spooling (RPS < {args.rps_threshold}):    {total_spool}")
print(f"  In-flight (RPS >= {args.rps_threshold}):  {total_flight}")
print()
print("Per-channel breakdown:")
header = f"  {'Ch':>4} {'In-flight':>10} {'Spool':>7} {'Zero':>6} {'Total':>6}"
print(header)
print("  " + "-" * (len(header) - 2))
for ch in sorted(set(list(channel_samples.keys()) + list(spool_count.keys()) + list(zero_rps_count.keys()))):
    f = len(channel_samples.get(ch, []))
    s = spool_count.get(ch, 0)
    z = zero_rps_count.get(ch, 0)
    print(f"  ch{ch:<2} {f:>10} {s:>7} {z:>6} {f+s+z:>6}")

# ─── Load dataset and build index lookup ────────────────────────────────────

valid_ds = DREGONRPSDataset(Path(args.data_root) / "valid")
print(f"\nLoaded dataset: {len(valid_ds)} samples")

idx_to_global = {}
for i in range(len(valid_ds)):
    dname = Path(valid_ds.samples[i]).name
    gidx = int(dname.split("_")[1])
    idx_to_global[gidx] = i

# ─── Evaluate each model ────────────────────────────────────────────────────

CHECKPOINTS = {
    "simple_conv_v2": "results/rps_exp_v2/best_simple_conv_v2.pt",
    "simple_conv_bigru_v2": "results/rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt",
    "simple_conv": "results/rps_exp_simple_conv/best_simple_conv.pt",
    "simple_conv_bigru": "results/rps_exp_simple_conv_bigru/best_simple_conv_bigru.pt",
    "simple_conv_tcn": "results/rps_exp_tcn/best_simple_conv_tcn.pt",
    "simple_conv_wide": "results/rps_exp_wide/best_simple_conv_wide.pt",
    "simple_conv_multiscale": "results/rps_exp_multiscale/best_simple_conv_multiscale.pt",
    "simple_conv_magphase_bigru": "results/rps_exp_magphase_bigru/best_simple_conv_magphase_bigru.pt",
    "simple_conv_attn_pool": "results/rps_exp_attn_pool/best_simple_conv_attn_pool.pt",
    "simple_conv_se_next": "results/rps_exp_se_next/best_simple_conv_se_next.pt",
}

all_results = {}

for model_name in args.models:
    ckpt_path = CHECKPOINTS.get(model_name)
    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"\nSKIP {model_name}: checkpoint not found at {ckpt_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")

    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    channel_metrics = {}

    for ch in sorted(channel_samples.keys()):
        indices = [idx_to_global[idx] for idx in channel_samples[ch] if idx in idx_to_global]
        if not indices:
            continue
        subset = Subset(valid_ds, indices)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
        m = evaluate(model, loader, device, len(subset))
        channel_metrics[ch] = m
        print(f"  ch{ch}: n={len(indices):3d}  PIT_MSE={m['mse']:.2f}  MAE/f={m['mae_frame']:.2f}  "
              f"R²_median={m['r2_median']:.4f}")

    all_results[model_name] = channel_metrics

# ─── Print summary tables ────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"PER-CHANNEL RESULTS — In-flight regime only (RPS mean >= {args.rps_threshold})")
print(f"{'='*90}")

channels = sorted(set(ch for m in all_results.values() for ch in m.keys()))

for metric_name, metric_label, fmt in [
    ("mse", "PIT MSE ↓", "{:.2f}"),
    ("mae_frame", "MAE/frame ↓", "{:.2f}"),
    ("r2_median", "R² median ↑", "{:.4f}"),
]:
    print(f"\n{'─'*90}")
    print(f"Metric: {metric_label}")
    print(f"{'─'*90}")

    header = f"{'Model':<30}"
    for ch in channels:
        header += f" {'ch'+str(ch):>10}"
    header += f" {'mean':>10} {'min':>10} {'max':>10} {'ch0→max Δ':>12}"
    print(header)
    print("-" * len(header))

    for model_name in args.models:
        if model_name not in all_results:
            continue
        cm = all_results[model_name]
        values = {}
        for ch in channels:
            if ch in cm:
                values[ch] = cm[ch][metric_name]

        valid_vals = list(values.values())
        mean_val = sum(valid_vals) / len(valid_vals)
        min_val = min(valid_vals)
        max_val = max(valid_vals)
        ch0_val = values.get(0, float('nan'))

        # delta: worst degradation from ch0
        if metric_name in ("mse", "mae_frame"):
            worst_delta = max_val - ch0_val if ch0_val == ch0_val else float('nan')
        else:
            worst_delta = ch0_val - min_val if ch0_val == ch0_val else float('nan')

        row = f"{model_name:<30}"
        for ch in channels:
            v = values.get(ch, float('nan'))
            row += f" {fmt.format(v):>10}" if v == v else f" {'N/A':>10}"
        row += f" {fmt.format(mean_val):>10}"
        row += f" {fmt.format(min_val):>10}"
        row += f" {fmt.format(max_val):>10}"
        row += f" {worst_delta:+.2f}" if worst_delta == worst_delta else f" {'N/A':>12}"
        print(row)

# ─── Per-channel delta table ─────────────────────────────────────────────────

print(f"\n{'='*90}")
print("CROSS-CHANNEL DEGRADATION vs ch0 (Δ = how much worse)")
print(f"{'='*90}")

for metric_name, metric_label, delta_sign in [
    ("mse", "PIT MSE (higher=worse)", 1),
    ("r2_median", "R² median (lower=worse)", -1),
]:
    print(f"\n{metric_label}:")
    print(f"{'Model':<30}", end="")
    for ch in channels:
        label = "ch0" if ch == 0 else f"Δ ch{ch}"
        print(f" {label:>10}", end="")
    print()

    for model_name in args.models:
        if model_name not in all_results:
            continue
        cm = all_results[model_name]
        ch0 = cm.get(0, {}).get(metric_name, float('nan'))
        print(f"{model_name:<30}", end="")
        for ch in channels:
            v = cm.get(ch, {}).get(metric_name, float('nan'))
            if ch == 0:
                if metric_name == "mse":
                    print(f" {v:>10.2f}", end="")
                else:
                    print(f" {v:>10.4f}", end="")
            else:
                if v == v and ch0 == ch0:
                    delta = delta_sign * (v - ch0)
                    if metric_name == "mse":
                        print(f" {delta:+10.2f}", end="")
                    else:
                        print(f" {delta:+10.4f}", end="")
                else:
                    print(f" {'N/A':>10}", end="")
        print()

# ─── Save results ────────────────────────────────────────────────────────────

out_path = Path("results/channel_eval_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

serializable = {}
for model_name, cm in all_results.items():
    serializable[model_name] = {}
    for ch, m in cm.items():
        serializable[model_name][str(ch)] = {k: v for k, v in m.items()}

# Also save metadata
serializable["_meta"] = {
    "data_root": args.data_root,
    "rps_threshold": args.rps_threshold,
    "total_samples": total_all,
    "zero_rps": total_zero,
    "spooling": total_spool,
    "in_flight": total_flight,
    "per_channel_zero": {str(k): v for k, v in zero_rps_count.items()},
    "per_channel_spool": {str(k): v for k, v in spool_count.items()},
    "per_channel_flight": {str(k): len(v) for k, v in channel_samples.items()},
}

with open(out_path, "w") as f:
    json.dump(serializable, f, indent=2)

print(f"\nResults saved to {out_path}")
