#!/usr/bin/env python3
"""
Train a small DCUNet-encoder-based model to predict 4 motor RPS per STFT frame
from noisy drone+speech audio mixtures.

Usage:
    python train_rps_predictor.py [--device cuda:1] [--epochs 200] [--patience 10]
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


# ─── Model ────────────────────────────────────────────────────────────────────


class RPSPredictor(nn.Module):
    """
    Predict 4 motor RPS per STFT frame from noisy audio.

    Architecture mirrors DCUNet encoder (same channel sizes, kernel sizes,
    LeakyReLU + BatchNorm) but uses real-valued convolutions on log-magnitude
    spectrograms and strides only in frequency, preserving the time axis for
    dense per-frame prediction.

    Encoder → AdaptiveAvgPool over freq → 1-D conv head → (B, 4, T)
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.register_buffer("window", torch.hann_window(n_fft))

        # --- encoder (DCUNet-like, real-valued, freq-only strides) ----------
        # Input: (B, 1, 1025, T) log-mag spectrogram
        self.encoder = nn.ModuleList()
        enc_spec = [
            # (in_ch, out_ch, kernel, stride_f, pad)
            (1, 45, (7, 5), (2, 1), (3, 2)),  # → (B,45, 513, T)
            (45, 90, (7, 5), (2, 1), (3, 2)),  # → (B,90, 257, T)
            (90, 90, (5, 3), (2, 1), (2, 1)),  # → (B,90, 129, T)
            (90, 90, (5, 3), (2, 1), (2, 1)),  # → (B,90,  65, T)
            (90, 90, (5, 3), (2, 1), (2, 1)),  # → (B,90,  33, T)
        ]
        for ic, oc, k, s, p in enc_spec:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ic, oc, k, stride=s, padding=p),
                    nn.BatchNorm2d(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        # --- prediction head ------------------------------------------------
        # Pool over frequency → (B, 90, T), then 1-D convs to (B, 4, T)
        self.head = nn.Sequential(
            nn.Conv1d(90, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, num_rotors, kernel_size=1),
        )

    def forward(self, audio):
        """
        audio: (B, samples) raw mono waveform at 16 kHz.
        Returns: (B, 4, T) predicted RPS per STFT frame.
        """
        # Handle 3D input (B, 1, samples) from demix batching
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        # STFT → log-magnitude
        X = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=True,
        )
        mag = X.abs()  # (B, F, T)
        mag = torch.log1p(mag)  # log(1+|X|) for compression
        mag = mag.unsqueeze(1)  # (B, 1, F, T)

        # Encoder
        h = mag
        for block in self.encoder:
            h = block(h)  # (B, 90, F', T)

        # Pool frequency axis
        h = h.mean(dim=2)  # (B, 90, T)

        # Predict RPS
        return self.head(h)  # (B, 4, T)


# ─── Dataset ──────────────────────────────────────────────────────────────────


class DREGONRPSDataset(Dataset):
    """Load mixture.wav + rps.npy from DREGON-LM, resample RPS to STFT frames."""

    def __init__(self, data_dir, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.samples = sorted(
            d
            for d in glob.glob(os.path.join(data_dir, "sample_*"))
            if os.path.isfile(os.path.join(d, "mixture.wav"))
            and os.path.isfile(os.path.join(d, "rps.npy"))
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        audio, _sr = torchaudio.load(os.path.join(d, "mixture.wav"))
        audio = audio[0]  # mono  (samples,)

        rps = torch.from_numpy(
            np.load(os.path.join(d, "rps.npy"))
        ).float()  # (4, rps_T)

        # Number of STFT frames (center=True default)
        n_frames = audio.shape[0] // self.hop_length + 1

        # Resample RPS to STFT time grid via linear interpolation
        rps = F.interpolate(
            rps.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
        ).squeeze(0)  # (4, n_frames)

        return audio, rps


# ─── Helpers ──────────────────────────────────────────────────────────────────


def evaluate(model, loader, device, dataset_len):
    """Run model on a dataloader, return per-frame and per-clip metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for audio, rps_target in loader:
            audio, rps_target = audio.to(device), rps_target.to(device)
            with torch.amp.autocast("cuda"):
                rps_pred = model(audio)
                loss = F.mse_loss(rps_pred, rps_target)
            total_loss += loss.item() * audio.size(0)
            all_preds.append(rps_pred.cpu())
            all_targets.append(rps_target.cpu())

    all_preds = torch.cat(all_preds)  # (N, 4, T)
    all_targets = torch.cat(all_targets)

    mse = total_loss / dataset_len
    mae_frame = (all_preds - all_targets).abs().mean().item()
    mae_per_rotor = (all_preds - all_targets).abs().mean(dim=(0, 2))  # (4,)

    # Per-clip (time-averaged)
    cp = all_preds.mean(dim=2)
    ct = all_targets.mean(dim=2)
    mae_clip = (cp - ct).abs().mean().item()

    # R²
    ss_res = ((all_preds - all_targets) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    return {
        "mse": mse,
        "mae_frame": mae_frame,
        "mae_per_rotor": mae_per_rotor.tolist(),
        "mae_clip": mae_clip,
        "r2": r2,
        "preds": all_preds,
        "targets": all_targets,
    }


# ─── Training loop ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_root", default="datasets/DREGON-LM")
    parser.add_argument("--save_path", default="results/rps_predictor")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(args.device)

    # --- Data ---
    n_fft, hop = 2048, 512
    train_ds = DREGONRPSDataset(os.path.join(args.data_root, "train"), n_fft, hop)
    valid_ds = DREGONRPSDataset(os.path.join(args.data_root, "valid"), n_fft, hop)
    print(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model ---
    model = RPSPredictor(n_fft=n_fft, hop_length=hop).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Optimizer / scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    scaler = torch.amp.GradScaler("cuda")

    # --- Naive baseline ---
    # Predict training-set mean RPS for every frame
    print("Computing naive baseline (predict train-set mean)...")
    rps_sum = torch.zeros(4)
    rps_count = 0
    for _, rps in train_loader:
        rps_sum += rps.sum(dim=(0, 2))
        rps_count += rps.shape[0] * rps.shape[2]
    rps_mean = rps_sum / rps_count  # (4,)
    print(f"  Train-set mean RPS per rotor: {rps_mean.tolist()}")

    # Evaluate naive baseline on validation set
    naive_mse = 0.0
    for _, rps in valid_loader:
        diff = rps - rps_mean.view(1, 4, 1)
        naive_mse += (diff**2).sum().item()
    naive_mse /= len(valid_ds) * valid_ds[0][1].shape[1]  # per element
    print(
        f"  Naive baseline val MSE: {naive_mse:.4f} (RMSE: {naive_mse**0.5:.2f} RPS)\n"
    )

    # --- Training ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_path = os.path.join(args.save_path, "best.pt")

    print(
        f"{'Epoch':>5} {'Train MSE':>10} {'Val MSE':>10} {'MAE/frame':>10} "
        f"{'MAE/clip':>10} {'R²':>8} {'LR':>10}"
    )
    print("-" * 72)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
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
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * audio.size(0)

        train_mse = train_loss_sum / len(train_ds)

        # ---- validate ----
        metrics = evaluate(model, valid_loader, device, len(valid_ds))
        val_mse = metrics["mse"]
        scheduler.step(val_mse)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:5d} {train_mse:10.4f} {val_mse:10.4f} "
            f"{metrics['mae_frame']:10.2f} {metrics['mae_clip']:10.2f} "
            f"{metrics['r2']:8.4f} {lr:10.1e}"
        )

        # ---- early stopping ----
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed / 60:.1f} min")

    # ─── Final evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL EVALUATION — best model on validation set")
    print("=" * 72)
    model.load_state_dict(torch.load(best_path, weights_only=True))
    m = evaluate(model, valid_loader, device, len(valid_ds))

    print(f"\nPer-frame metrics:")
    print(f"  MSE:  {m['mse']:.4f}  (RMSE: {m['mse'] ** 0.5:.2f} RPS)")
    print(f"  MAE:  {m['mae_frame']:.2f} RPS")
    print(f"  R²:   {m['r2']:.4f}")
    print(f"  MAE per rotor: [{', '.join(f'{v:.2f}' for v in m['mae_per_rotor'])}]")

    print(f"\nPer-clip (time-averaged) metrics:")
    print(f"  MAE:  {m['mae_clip']:.2f} RPS")

    tgt = m["targets"]
    print(
        f"\nTarget RPS stats: mean={tgt.mean():.1f}, std={tgt.std():.1f}, "
        f"min={tgt.min():.1f}, max={tgt.max():.1f}"
    )
    rng = tgt.max() - tgt.min()
    print(f"MAE as % of range: {m['mae_frame'] / rng * 100:.1f}%")
    print(f"MAE as % of mean:  {m['mae_frame'] / tgt.mean() * 100:.1f}%")

    print(
        f"\nNaive baseline MSE (predict mean): {naive_mse:.4f} "
        f"(RMSE: {naive_mse**0.5:.2f} RPS)"
    )
    improvement = (1 - m["mse"] / naive_mse) * 100
    print(f"Model improvement over naive: {improvement:.1f}%")


if __name__ == "__main__":
    main()
