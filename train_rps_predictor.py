#!/usr/bin/env python3
"""
Train RPS prediction models on DREGON-LM dataset.

Supports three model architectures for fair comparison:
1. SimpleConv: Lightweight CNN on log-magnitude spectrograms (baseline)
2. DCUNetEncRPS: DCUNet complex conv encoder + RPSPredictionHead (faithful copy)
3. DCCRNEncRPS: DCCRN complex conv encoder + RPSPredictionHead (faithful copy)

Usage:
    python train_rps_predictor.py --model simple_conv      # Default
    python train_rps_predictor.py --model dcunet_enc_rps   # DCUNet encoder + RPS head
    python train_rps_predictor.py --model dccrn_enc_rps   # DCCRN encoder + RPS head
    python train_rps_predictor.py --model dccrn_lite_rps   # DCCRNLite encoder + RPS head
    python train_rps_predictor.py --train_all              # Train all and compare

All models predict 4 rotor speeds per STFT frame from noisy audio.
Input:  (B, samples) raw mono waveform at 16 kHz
Output: (B, 4, T_stft) predicted RPS per STFT frame
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


# ─── Model: SimpleConv (baseline) ─────────────────────────────────────────────


class SimpleConv(nn.Module):
    """
    Lightweight CNN on log-magnitude spectrograms for RPS prediction.
    Architecture mirrors DCUNet encoder but uses real-valued convolutions.
    """

    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.register_buffer("window", torch.hann_window(n_fft))

        # Real-valued encoder (mirrors DCUNet channel sizes)
        self.encoder = nn.ModuleList()
        enc_spec = [
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

        # Prediction head: pool freq → (B, 4, T)
        self.head = nn.Sequential(
            nn.Conv1d(90, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, num_rotors, kernel_size=1),
        )

    def forward(self, audio):
        """
        audio: (B, samples) raw mono waveform at 16 kHz.
        Returns: (B, 4, T_stft) predicted RPS per STFT frame.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        X = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True, normalized=True,
        )
        mag = X.abs()
        mag = torch.log1p(mag)
        mag = mag.unsqueeze(1)  # (B, 1, F, T)

        h = mag
        for block in self.encoder:
            h = block(h)

        h = h.mean(dim=2)  # pool frequency (B, 90, T)
        return self.head(h)


# ─── Model: DCUNet Enc + RPS Head ────────────────────────────────────────────


def stft_time_frames(audio_length: int, hop_length: int, n_fft: int) -> int:
    """Number of STFT time frames for a given audio length (with center padding)."""
    # torch.stft with center=True pads by n_fft//2 on each side
    # effective signal length = audio_length + n_fft//2*2, then (eff - n_fft)//hop + 1
    # simplified: audio_length // hop_length + 1
    return audio_length // hop_length + 1


class RPSPredictionHead(nn.Module):
    """
    FPN-style auxiliary head that predicts per-STFT-frame RPS from all encoder levels.
    Faithfully replicated from models/dcunet.py.
    """
    def __init__(self, encoder_channels: list[int], target_t: int, common_dim: int = 64, num_rotors: int = 4):
        super().__init__()
        self.target_t = target_t
        self.level_projs = nn.ModuleList()
        for ch in encoder_channels:
            self.level_projs.append(nn.Conv1d(ch * 2, common_dim, 1))  # *2 for real+imag

        self.head = nn.Sequential(
            nn.Conv1d(common_dim, common_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(common_dim, num_rotors, kernel_size=1),
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """
        encoder_features: list of (B, C_i, F_i, T_i, 2) from each encoder level,
                          ordered finest-to-coarsest (level 0 = finest).
        Returns: (B, num_rotors, target_t)
        """
        level_feats = []
        for feat, proj in zip(encoder_features, self.level_projs):
            B, C, F_i, T_i, _ = feat.shape
            pooled = feat.mean(dim=2)       # (B, C, T_i, 2)
            pooled = pooled.reshape(B, C * 2, T_i)
            level_feats.append(proj(pooled))  # (B, common_dim, T_i)

        # Bottom-up merge: coarsest → finest
        merged = level_feats[-1]
        for i in range(len(level_feats) - 2, -1, -1):
            finer = level_feats[i]
            if merged.shape[-1] != finer.shape[-1]:
                merged = F.interpolate(merged, size=finer.shape[-1], mode="linear", align_corners=False)
            merged = merged + finer

        # Upsample to target STFT frame rate
        if merged.shape[-1] != self.target_t:
            merged = F.interpolate(merged, size=self.target_t, mode="linear", align_corners=False)

        return self.head(merged)  # (B, num_rotors, target_t)


class DCUNetEncRPS(nn.Module):
    """
    DCUNet encoder (complex conv) + RPSPredictionHead for standalone RPS prediction.
    Faithfully replicates the encoder architecture from models/dcunet.py.
    """
    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, num_layers=5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.register_buffer("window", torch.hann_window(n_fft))

        # Import complex conv components from dcunet
        from models.dcunet import CConv2d as DCUNetCConv, CBatchNorm2d as DCUNetCBN

        # DCUNet encoder spec — faithful copy from models/dcunet.py
        enc_spec = [
            (1, 45, (7, 5), (2, 2), (3, 2)),
            (45, 90, (7, 5), (2, 2), (3, 2)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 2), (2, 1)),
            (90, 90, (5, 3), (2, 1), (2, 1)),
        ]
        if num_layers == 6:
            enc_spec.append((90, 90, (5, 3), (2, 1), (2, 1)))

        self.encoders = nn.ModuleList()
        for ic, oc, k, s, p in enc_spec:
            self.encoders.append(
                nn.Sequential(
                    DCUNetCConv(ic, oc, k, s, p),
                    DCUNetCBN(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        enc_channels = [oc for _, oc, _, _, _ in enc_spec]
        target_t = stft_time_frames(131584, hop_length, n_fft)  # chunk_size=131584
        self.head = RPSPredictionHead(enc_channels, target_t, num_rotors=num_rotors)

    def forward(self, audio):
        """
        audio: (B, samples) raw mono waveform.
        Returns: (B, 4, T_stft) predicted RPS per STFT frame.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # STFT → complex tensor (B, F, T, 2)
        X = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True, normalized=True,
        )
        X = torch.view_as_real(X)  # (B, F, T, 2)
        X = X.unsqueeze(1)  # (B, 1, F, T, 2)

        # Forward through encoder, collect features
        encoder_features = []
        h = X
        for encoder in self.encoders:
            h = encoder(h)
            encoder_features.append(h)

        return self.head(encoder_features)


# ─── Model: DCCRN Enc + RPS Head ──────────────────────────────────────────────


class DCCRNEncRPS(nn.Module):
    """
    DCCRN encoder (complex conv) + RPSPredictionHead for standalone RPS prediction.
    Faithfully replicates the encoder architecture from models/dccrn.py.
    Supports lite variant with fewer layers and channels.
    """
    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, lite=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.lite = lite
        self.register_buffer("window", torch.hann_window(n_fft))

        # Import complex conv components from dccrn
        from models.dccrn import CConv2d as DCCRN_CConv, CBatchNorm2d as DCCRN_CBN

        if lite:
            encoder_channels = [16, 32, 64, 128]
        else:
            encoder_channels = [32, 64, 128, 256, 256, 512]

        # DCCRN encoder: kernel (5,2), stride (2,1), padding (2,0) for all layers
        enc_kernel = (5, 2)
        enc_stride = (2, 1)
        enc_padding = (2, 0)

        in_channels = [1] + encoder_channels[:-1]

        self.encoders = nn.ModuleList()
        for ic, oc in zip(in_channels, encoder_channels):
            self.encoders.append(
                nn.Sequential(
                    DCCRN_CConv(ic, oc, enc_kernel, enc_stride, enc_padding),
                    DCCRN_CBN(oc),
                    nn.LeakyReLU(0.2),
                )
            )

        target_t = stft_time_frames(131584, hop_length, n_fft)
        self.head = RPSPredictionHead(encoder_channels, target_t, num_rotors=num_rotors)

    def forward(self, audio):
        """
        audio: (B, samples) raw mono waveform.
        Returns: (B, 4, T_stft) predicted RPS per STFT frame.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # STFT → complex tensor (B, F, T, 2)
        X = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True, normalized=True,
        )
        X = torch.view_as_real(X)  # (B, F, T, 2)
        X = X.unsqueeze(1)  # (B, 1, F, T, 2)

        # Forward through encoder, collect features
        encoder_features = []
        h = X
        for encoder in self.encoders:
            h = encoder(h)
            encoder_features.append(h)

        return self.head(encoder_features)


# ─── Model factory ────────────────────────────────────────────────────────────


MODEL_REGISTRY = {
    "simple_conv": SimpleConv,
    "dcunet_enc_rps": DCUNetEncRPS,
    "dccrn_enc_rps": lambda **kw: DCCRNEncRPS(lite=False, **kw),
    "dccrn_lite_rps": lambda **kw: DCCRNEncRPS(lite=True, **kw),
}


def get_model(model_name, n_fft=2048, hop_length=512, num_rotors=4):
    """Create a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors)


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate(model, loader, device, dataset_len):
    """Run model on dataloader, return per-frame and per-clip metrics."""
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

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = total_loss / dataset_len
    mae_frame = (all_preds - all_targets).abs().mean().item()
    mae_per_rotor = (all_preds - all_targets).abs().mean(dim=(0, 2))

    # Per-clip (time-averaged)
    cp = all_preds.mean(dim=2)
    ct = all_targets.mean(dim=2)
    mae_clip = (cp - ct).abs().mean().item()

    # Macro-averaged per-sample R²: compute R² per sample using each
    # sample's own mean as the baseline, then average. This measures
    # within-sample temporal tracking quality without inflating the
    # metric with between-sample RPS variance.
    r2_per_sample = []
    for pred_i, tgt_i in zip(all_preds, all_targets):
        ss_res_i = ((pred_i - tgt_i) ** 2).sum()
        ss_tot_i = ((tgt_i - tgt_i.mean()) ** 2).sum()
        if ss_tot_i > 1e-6:
            r2_per_sample.append((1 - ss_res_i / ss_tot_i).item())
    r2_arr = torch.tensor(r2_per_sample) if r2_per_sample else torch.tensor([float('nan')])
    r2_mean   = float(r2_arr.mean())
    r2_median = float(r2_arr.median())

    return {
        "mse": mse,
        "mae_frame": mae_frame,
        "mae_per_rotor": mae_per_rotor.tolist(),
        "mae_clip": mae_clip,
        "r2": r2_mean,
        "r2_median": r2_median,
    }


# ─── Training loop ────────────────────────────────────────────────────────────


def train_model(model_name, args):
    """Train a single model."""
    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    n_fft, hop = args.n_fft, args.hop_length

    # Data
    train_ds = DREGONRPSDataset(os.path.join(args.data_root, "train"), n_fft, hop)
    valid_ds = DREGONRPSDataset(os.path.join(args.data_root, "valid"), n_fft, hop)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = get_model(model_name, n_fft=n_fft, hop_length=hop).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    scaler = torch.amp.GradScaler("cuda")

    # Naive baseline
    print("Computing naive baseline...")
    rps_sum = torch.zeros(4)
    for _, rps in train_loader:
        rps_sum += rps.sum(dim=(0, 2))
    rps_mean = rps_sum / (len(train_ds) * train_ds[0][1].shape[1])
    naive_mse = 0.0
    for _, rps in valid_loader:
        diff = rps - rps_mean.view(1, 4, 1)
        naive_mse += (diff**2).sum().item()
    naive_mse /= len(valid_ds) * valid_ds[0][1].shape[1]
    print(f"Naive MSE: {naive_mse:.4f} (RMSE: {naive_mse**0.5:.2f} RPS)")

    # Training
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_path = os.path.join(args.save_path, f"best_{model_name}.pt")

    print(f"\n{'Epoch':>5} {'Train MSE':>10} {'Val MSE':>10} {'MAE/frame':>10} {'MAE/clip':>10} {'R²':>8} {'LR':>10}")
    print("-" * 65)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
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

        train_mse = train_loss_sum / len(train_ds)
        metrics = evaluate(model, valid_loader, device, len(valid_ds))
        val_mse = metrics["mse"]
        scheduler.step(val_mse)
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} {train_mse:10.4f} {val_mse:10.4f} {metrics['mae_frame']:10.2f} {metrics['mae_clip']:10.2f} {metrics['r2']:8.4f} {lr:10.1e}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed / 60:.1f} min")

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — {model_name}")
    print("="*60)
    model.load_state_dict(torch.load(best_path, weights_only=True))
    m = evaluate(model, valid_loader, device, len(valid_ds))

    print(f"\nPer-frame: MSE={m['mse']:.4f} (RMSE={m['mse']**0.5:.2f}), MAE={m['mae_frame']:.2f}, R²={m['r2']:.4f}")
    print(f"Per-clip:   MAE={m['mae_clip']:.2f}")
    print(f"Improvement over naive: {(1 - m['mse']/naive_mse)*100:.1f}%")

    return {
        "model": model_name,
        "mse": m["mse"],
        "mae_frame": m["mae_frame"],
        "mae_clip": m["mae_clip"],
        "r2": m["r2"],
        "naive_mse": naive_mse,
        "best_path": best_path,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train RPS prediction models")
    parser.add_argument("--model", type=str, default="simple_conv",
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--data_root", default="datasets/DREGON-LM")
    parser.add_argument("--save_path", default="results/rps_predictor_comparison")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    results = {}

    if args.train_all:
        for model_name in MODEL_REGISTRY.keys():
            result = train_model(model_name, args)
            results[model_name] = result
    else:
        result = train_model(args.model, args)
        results[args.model] = result

    # Summary
    if len(results) > 1:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Model':<20} {'MSE':>10} {'RMSE':>8} {'MAE/clip':>10} {'R²':>8} {'vs Naive':>10}")
        print("-"*70)
        for name, r in sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True):
            improvement = (1 - r["mse"]/r["naive_mse"])*100
            print(f"{name:<20} {r['mse']:>10.4f} {r['mse']**0.5:>8.2f} {r['mae_clip']:>10.2f} {r['r2']:>8.4f} {improvement:>9.1f}%")

        import json
        with open(os.path.join(args.save_path, "comparison_results.json"), "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ['preds', 'targets']} 
                      for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
