#!/usr/bin/env python3
"""
5-fold cross-validation for the RPS predictor (simple_conv) over the training set.

Trains each fold for exactly --epochs epochs (no early stopping) so that we can
later plot validation metrics vs epoch across all folds and diagnose overfitting.

Output:
    results/rps_cv/
        cv_results.json     ← {fold_0: [{epoch, train_mse, val_mse, val_mae_clip,
                                          val_r2_mean, val_r2_median, lr}, …], …}
        fold_<i>_best.pt    ← checkpoint at best val_mse per fold

Usage:
    python train_rps_cv.py --data_dir datasets/DREGON-LM/train --epochs 100
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_rps_predictor import DREGONRPSDataset, SimpleConv, evaluate
from utils.paths import get_datasets_path, get_results_path

# ─── CV split ────────────────────────────────────────────────────────────────


def make_folds(n_samples: int, n_folds: int = 5, seed: int = 42):
    """Return list of (train_indices, val_indices) tuples, deterministic."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    fold_size = n_samples // n_folds
    folds = []
    for k in range(n_folds):
        val_start = k * fold_size
        val_end = val_start + fold_size if k < n_folds - 1 else n_samples
        val_idx = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds


# ─── Single-fold training ─────────────────────────────────────────────────────


def train_fold(fold_idx, train_indices, val_indices, dataset, args, device):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold_idx}  |  train={len(train_indices)}  val={len(val_indices)}")
    print(f"{'=' * 60}")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = SimpleConv(n_fft=args.n_fft, hop_length=args.hop_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val_mse = float("inf")
    best_path = os.path.join(args.output_dir, f"fold_{fold_idx}_best.pt")
    epoch_log = []

    t0 = time.time()
    print(
        f"{'Epoch':>5}  {'TrainMSE':>10}  {'ValMSE':>10}  "
        f"{'MAE/clip':>9}  {'R²mean':>8}  {'R²med':>8}  {'LR':>9}"
    )
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        for audio, rps_target in train_loader:
            audio, rps_target = audio.to(device), rps_target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                rps_pred = model(audio)
                loss = F.mse_loss(rps_pred, rps_target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * audio.size(0)

        train_mse = train_loss_sum / len(train_indices)

        # ── validate ───────────────────────────────────────────────────
        m = evaluate(model, val_loader, device, len(val_indices))
        scheduler.step(m["mse"])
        lr = optimizer.param_groups[0]["lr"]

        if m["mse"] < best_val_mse:
            best_val_mse = m["mse"]
            torch.save(model.state_dict(), best_path)

        epoch_log.append(
            {
                "epoch": epoch,
                "train_mse": round(train_mse, 6),
                "val_mse": round(m["mse"], 6),
                "val_mae_frame": round(m["mae_frame"], 6),
                "val_mae_clip": round(m["mae_clip"], 6),
                "val_r2_mean": round(m["r2"], 6),
                "val_r2_median": round(m.get("r2_median", float("nan")), 6),
                "lr": lr,
            }
        )

        print(
            f"{epoch:5d}  {train_mse:10.4f}  {m['mse']:10.4f}  "
            f"{m['mae_clip']:9.3f}  {m['r2']:8.4f}  {m.get('r2_median', float('nan')):8.4f}  {lr:9.1e}"
        )

    elapsed = time.time() - t0
    print(f"\nFold {fold_idx} done in {elapsed / 60:.1f} min  | best val_mse={best_val_mse:.4f}")
    return epoch_log


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="5-fold CV for RPS predictor — records per-epoch metrics"
    )
    parser.add_argument("--data_dir", default=str(get_datasets_path("DREGON-LM/train")))
    parser.add_argument("--output_dir", default=str(get_results_path("rps_cv")))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load full training dataset once (files are memory-mapped by torchaudio,
    # so the Subset just indexes into it cheaply).
    print(f"Loading dataset from {args.data_dir} ...")
    dataset = DREGONRPSDataset(args.data_dir, n_fft=args.n_fft, hop_length=args.hop_length)
    print(f"  {len(dataset)} samples")

    folds = make_folds(len(dataset), n_folds=args.n_folds, seed=args.seed)

    cv_results = {}
    t_total = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        epoch_log = train_fold(fold_idx, train_idx, val_idx, dataset, args, device)
        cv_results[f"fold_{fold_idx}"] = epoch_log

        # Checkpoint after each fold in case the job is interrupted
        tmp_path = os.path.join(args.output_dir, "cv_results.json")
        with open(tmp_path, "w") as f:
            json.dump(cv_results, f)
        print(f"  → cv_results.json updated ({fold_idx + 1}/{args.n_folds} folds done)")

    total_min = (time.time() - t_total) / 60
    print(f"\nAll {args.n_folds} folds done in {total_min:.1f} min")
    print(f"Results saved to {os.path.join(args.output_dir, 'cv_results.json')}")


if __name__ == "__main__":
    main()
