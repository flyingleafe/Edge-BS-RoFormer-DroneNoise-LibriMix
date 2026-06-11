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
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import wandb
from torch.utils.data import DataLoader, Dataset

# Import new model variants
from models.rps_predictor import (
    SimpleConv,
    SimpleConvAttnPool,
    SimpleConvBiGRU,
    SimpleConvBiGRUV2,
    SimpleConvMagPhaseBiGRU,
    SimpleConvMultiScale,
    SimpleConvSENext,
    SimpleConvTCN,
    SimpleConvV2,
    SimpleConvWide,
)
from utils.paths import get_datasets_path, get_results_path

# Alias for utils.py compatibility
RPSPredictor = SimpleConv


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
        # audio: (C, T) from torchaudio.
        # - Mono (C=1): squeeze to (T,) for backward compatibility.
        # - Multichannel (C>1): keep (C, T) so the training loop can treat
        #   channels as additional batch items.
        if audio.shape[0] == 1:
            audio = audio[0]  # (T,)
        # else: (C, T) — channels stay

        rps = torch.from_numpy(np.load(os.path.join(d, "rps.npy"))).float()  # (4, rps_T)

        # Number of STFT frames — use last dim so (C, T) and (T,) both work.
        n_frames = audio.shape[-1] // self.hop_length + 1

        # Resample RPS to STFT time grid by *shape-stretch* (endpoint-to-endpoint).
        # GOTCHA: this ignores real timestamps and is only correct because here
        # audio and rps.npy are co-extensive (DREGON-LM slices both to the same
        # span). It silently misaligns when motor/audio spans differ (e.g. the
        # free-flight take: motor ~47s vs audio ~53s) — there, timestamp-based
        # np.interp(stft_times, motor_times, rps) is required. The eval/plot
        # refactor standardizes on the timestamp method (utils.data). These
        # training targets use shape-stretch; the two agree only sub-frame on
        # DREGON-LM. See .pi/plans/rps-eval-plot-refactor-plan.md §Alignment.
        rps = F.interpolate(
            rps.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
        ).squeeze(0)  # (4, n_frames)

        return audio, rps


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

    def __init__(
        self, encoder_channels: list[int], target_t: int, common_dim: int = 64, num_rotors: int = 4
    ):
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
            pooled = feat.mean(dim=2)  # (B, C, T_i, 2)
            pooled = pooled.reshape(B, C * 2, T_i)
            level_feats.append(proj(pooled))  # (B, common_dim, T_i)

        # Bottom-up merge: coarsest → finest
        merged = level_feats[-1]
        for i in range(len(level_feats) - 2, -1, -1):
            finer = level_feats[i]
            if merged.shape[-1] != finer.shape[-1]:
                merged = F.interpolate(
                    merged, size=finer.shape[-1], mode="linear", align_corners=False
                )
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
        from models.dcunet import CBatchNorm2d as DCUNetCBN
        from models.dcunet import CConv2d as DCUNetCConv

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
                    DCUNetCConv(ic, oc, k, s, p),  # pyright: ignore[reportArgumentType]
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
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,  # pyright: ignore[reportArgumentType]
            return_complex=True,
            normalized=True,
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
        from models.dccrn import CBatchNorm2d as DCCRN_CBN
        from models.dccrn import CConv2d as DCCRN_CConv

        encoder_channels = [16, 32, 64, 128] if lite else [32, 64, 128, 256, 256, 512]

        # DCCRN encoder: kernel (5,2), stride (2,1), padding (2,0) for all layers
        enc_kernel = (5, 2)
        enc_stride = (2, 1)
        enc_padding = (2, 0)

        in_channels = [1] + encoder_channels[:-1]

        self.encoders = nn.ModuleList()
        for ic, oc in zip(in_channels, encoder_channels):
            self.encoders.append(
                nn.Sequential(
                    DCCRN_CConv(ic, oc, enc_kernel, enc_stride, enc_padding),  # pyright: ignore[reportArgumentType]
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
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,  # pyright: ignore[reportArgumentType]
            return_complex=True,
            normalized=True,
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
    "simple_conv_v2": SimpleConvV2,
    "simple_conv_wide": SimpleConvWide,
    "simple_conv_tcn": SimpleConvTCN,
    "simple_conv_multiscale": SimpleConvMultiScale,
    "simple_conv_bigru": SimpleConvBiGRU,
    "simple_conv_bigru_v2": SimpleConvBiGRUV2,
    "simple_conv_magphase_bigru": SimpleConvMagPhaseBiGRU,
    "simple_conv_attn_pool": SimpleConvAttnPool,
    "simple_conv_se_next": SimpleConvSENext,
    "dcunet_enc_rps": DCUNetEncRPS,
    "dccrn_enc_rps": lambda **kw: DCCRNEncRPS(lite=False, **kw),
    "dccrn_lite_rps": lambda **kw: DCCRNEncRPS(lite=True, **kw),
}


def get_model(model_name, n_fft=2048, hop_length=512, num_rotors=4):
    """Create a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](n_fft=n_fft, hop_length=hop_length, num_rotors=num_rotors)


# ─── PIT (Permutation-Invariant) MSE Loss ───────────────────────────────────

# Pre-compute all permutations of 4 rotors (4! = 24)
_ROTOR_PERMS = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)  # (24, 4)


def pairwise_mse(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise MSE between each estimated and target rotor.

    Args:
        est: (B, 4, T) predicted RPS
        target: (B, 4, T) ground-truth RPS

    Returns:
        (B, 4, 4) pairwise MSE matrix where [b, i, j] = MSE(est[b,i], target[b,j])
    """
    # est: (B, 4, T) -> (B, 4, 1, T)
    # target: (B, 4, T) -> (B, 1, 4, T)
    # diff: (B, 4, 4, T)
    diff = est.unsqueeze(2) - target.unsqueeze(1)
    return diff.pow(2).mean(dim=-1)  # (B, 4, 4)


def pit_mse_loss(
    est: torch.Tensor,
    target: torch.Tensor,
    perms: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Permutation-invariant MSE loss for RPS prediction.

    Finds the best 1-to-1 matching between predicted and target rotors
    by minimizing total MSE over all 4! = 24 permutations.

    Args:
        est: (B, 4, T) predicted RPS
        target: (B, 4, T) ground-truth RPS
        perms: Pre-computed permutation tensor (24, 4). If None, computes on-the-fly.

    Returns:
        Scalar loss (mean over batch of best-permutation MSE).
    """
    if perms is None:
        perms = _ROTOR_PERMS.to(est.device)
    elif perms.device != est.device:
        perms = perms.to(est.device)

    # Pairwise MSE matrix: (B, 4, 4)
    pw = pairwise_mse(est, target)

    # For each permutation, sum the pairwise losses along the matched pairs
    # pw: (B, 4, 4), perms: (P, 4) -> gather: (B, P, 4)
    # For perm p = [j0, j1, j2, j3], loss = pw[b, 0, j0] + pw[b, 1, j1] + ...
    # More efficient: index with perms
    B = pw.size(0)
    P = perms.size(0)

    # Gather: for each batch and permutation, collect pw[b, src_idx, tgt_idx]
    src_idx = torch.arange(4, device=pw.device).view(1, 1, 4)  # (1, 1, 4)
    tgt_idx = perms.view(1, P, 4)  # (1, P, 4)

    # Advanced indexing: pw[b, src_idx, tgt_idx]
    # pw: (B, 4, 4), want (B, P, 4)
    b_idx = torch.arange(B, device=pw.device).view(B, 1, 1)  # (B, 1, 1)
    perm_losses = pw[b_idx, src_idx, tgt_idx]  # (B, P, 4)
    perm_losses = perm_losses.sum(dim=-1)  # (B, P)

    # Best permutation per batch element
    best_loss, _ = perm_losses.min(dim=1)  # (B,) — sum of 4 pairwise MSEs
    # Normalize by n_rotors so PIT loss is comparable to standard per-element MSE
    return best_loss.mean() / 4.0


# ─── WandB Initialization ────────────────────────────────────────────────────


def wandb_init(args: argparse.Namespace, model_name: str) -> None:
    """Initialize WandB logging for RPS prediction project."""
    wandb_key = getattr(args, "wandb_key", "") or os.environ.get("WANDB_API_KEY", "")
    if not wandb_key or wandb_key.strip() == "":
        wandb.init(mode="disabled")
        return

    wandb.login(key=wandb_key)
    run_name = f"{model_name}_DREGON-LM-V2"
    init_kwargs = dict(
        entity="flyingleafe",
        project="rps-prediction",
        name=run_name,
        config={
            "model": model_name,
            "data_root": args.data_root,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pit_loss": args.pit_loss,
            "smoothness_weight": args.smoothness_weight,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
        },
    )
    wandb.init(**init_kwargs)  # pyright: ignore[reportArgumentType]

    # Write run ID to save path
    if wandb.run is not None and wandb.run.id:
        run_id_path = os.path.join(args.save_path, "wandb_run_id.txt")
        os.makedirs(args.save_path, exist_ok=True)
        with open(run_id_path, "w") as f:
            f.write(wandb.run.id)


# ─── Evaluation ──────────────────────────────────────────────────────────────


def _flatten_channels(audio: torch.Tensor, rps: torch.Tensor):
    """Flatten a multichannel batch into a flat batch for RPS models.

    Args:
        audio:  (B, T) or (B, C, T)
        rps:    (B, 4, F)

    Returns:
        audio_flat: (B*C, T)  — C=1 for mono batches (no-op)
        rps_flat:   (B*C, 4, F)
        C:          number of channels (1 for mono)
    """
    if audio.dim() == 2:  # (B, T) — mono
        return audio, rps, 1
    B, C, T = audio.shape
    audio_flat = audio.reshape(B * C, T)
    # Broadcast RPS across channels: (B,4,F) → (B,C,4,F) → (B*C,4,F)
    rps_flat = rps.unsqueeze(1).expand(B, C, -1, -1).reshape(B * C, rps.shape[1], rps.shape[2])
    return audio_flat, rps_flat, C


def evaluate(model, loader, device, dataset_len, pit_eval: bool = True):
    """Run model on dataloader, return per-frame and per-clip metrics.

    Handles both mono (T,) and multichannel (C, T) samples transparently.
    When pit_eval=True, uses permutation-invariant MSE for the primary loss.
    """
    model.eval()
    total_pit_loss = 0.0
    total_std_loss = 0.0
    all_preds, all_targets = [], []
    total_items = 0  # count B*C items, not just B

    with torch.no_grad():
        for audio, rps_target in loader:
            audio, rps_target = audio.to(device), rps_target.to(device)
            audio, rps_target, C = _flatten_channels(audio, rps_target)
            with torch.amp.autocast("cuda"):  # pyright: ignore[reportArgumentType, reportPrivateImportUsage]
                rps_pred = model(audio)
                pit_loss = pit_mse_loss(rps_pred, rps_target)
                std_loss = F.mse_loss(rps_pred, rps_target)
            n = audio.size(0)
            total_pit_loss += pit_loss.item() * n
            total_std_loss += std_loss.item() * n
            total_items += n
            all_preds.append(rps_pred.cpu())
            all_targets.append(rps_target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Divide by actual number of processed items (B*C), not dataset_len (B).
    pit_mse_val = total_pit_loss / total_items
    std_mse_val = total_std_loss / total_items
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
    r2_arr = torch.tensor(r2_per_sample) if r2_per_sample else torch.tensor([float("nan")])
    r2_mean = float(r2_arr.mean())
    r2_median = float(r2_arr.median())

    return {
        "mse": pit_mse_val,  # primary metric (PIT-aware if pit_eval=True)
        "std_mse": std_mse_val,  # fixed-order MSE for reference
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
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
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

    # Model
    model = get_model(model_name, n_fft=n_fft, hop_length=hop).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    scaler = torch.amp.GradScaler("cuda")  # pyright: ignore[reportPrivateImportUsage]

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

    print(
        f"\n{'Epoch':>5} {'Train MSE':>10} {'Val PIT':>10} {'Val Std':>10} {'MAE/f':>8} {'MAE/c':>8} {'R²':>8} {'LR':>10}"
    )
    print("-" * 75)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for audio, rps_target in train_loader:
            audio, rps_target = audio.to(device), rps_target.to(device)
            # Flatten multichannel batch: (B, C, T) → (B*C, T)
            audio, rps_target, _C = _flatten_channels(audio, rps_target)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):  # pyright: ignore[reportPrivateImportUsage]
                rps_pred = model(audio)
                if args.pit_loss:
                    loss = pit_mse_loss(rps_pred, rps_target)
                else:
                    loss = F.mse_loss(rps_pred, rps_target)
                if args.smoothness_weight > 0:
                    # Second-order finite difference for smoothness
                    # (B, 4, T) -> diff along time
                    diff1 = rps_pred[:, :, 1:] - rps_pred[:, :, :-1]
                    diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
                    smoothness = diff2.pow(2).mean()
                    loss = loss + args.smoothness_weight * smoothness
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * audio.size(0)
            train_items += audio.size(0)

        train_mse = train_loss_sum / train_items
        metrics = evaluate(model, valid_loader, device, len(valid_ds))
        val_mse = metrics["mse"]
        scheduler.step(val_mse)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:5d} {train_mse:10.4f} {val_mse:10.4f} {metrics['std_mse']:10.4f} {metrics['mae_frame']:8.2f} {metrics['mae_clip']:8.2f} {metrics['r2']:8.4f} {lr:10.1e}"
        )

        # WandB logging
        if wandb.run is not None and not wandb.run.disabled:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/mse": train_mse,
                    "val/pit_mse": val_mse,
                    "val/std_mse": metrics["std_mse"],
                    "val/mae_frame": metrics["mae_frame"],
                    "val/mae_clip": metrics["mae_clip"],
                    "val/r2": metrics["r2"],
                    "val/r2_median": metrics["r2_median"],
                    "lr": lr,
                }
            )

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
    print(f"\n{'=' * 60}")
    print(f"FINAL EVALUATION — {model_name}")
    print("=" * 60)
    model.load_state_dict(torch.load(best_path, weights_only=True))
    m = evaluate(model, valid_loader, device, len(valid_ds))

    print(
        f"\nPer-frame: PIT MSE={m['mse']:.4f} (RMSE={m['mse'] ** 0.5:.2f}), Std MSE={m['std_mse']:.4f}, MAE={m['mae_frame']:.2f}, R²={m['r2']:.4f}"
    )
    print(f"Per-clip:   MAE={m['mae_clip']:.2f}")
    print(f"Improvement over naive (PIT): {(1 - m['mse'] / naive_mse) * 100:.1f}%")

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
    parser.add_argument(
        "--model", type=str, default="simple_conv", choices=list(MODEL_REGISTRY.keys())
    )
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--data_root", default=str(get_datasets_path("DREGON-LM-V2")))
    parser.add_argument("--save_path", default=str(get_results_path("rps_predictor_comparison")))
    parser.add_argument(
        "--pit_loss",
        action="store_true",
        default=True,
        help="Use permutation-invariant MSE loss (default: True)",
    )
    parser.add_argument(
        "--no_pit_loss",
        action="store_false",
        dest="pit_loss",
        help="Disable permutation-invariant loss",
    )
    parser.add_argument("--wandb_key", type=str, default="", help="WandB API key")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument(
        "--smoothness_weight",
        type=float,
        default=0.0,
        help="Weight for temporal smoothness loss (second-order diff)",
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    results = {}

    if args.train_all:
        for model_name in MODEL_REGISTRY:
            wandb_init(args, model_name)
            result = train_model(model_name, args)
            results[model_name] = result
            if wandb.run is not None:
                wandb.finish()
    else:
        wandb_init(args, args.model)
        result = train_model(args.model, args)
        results[args.model] = result
        if wandb.run is not None:
            wandb.finish()

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Model':<20} {'MSE':>10} {'RMSE':>8} {'MAE/clip':>10} {'R²':>8} {'vs Naive':>10}")
        print("-" * 70)
        for name, r in sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True):
            improvement = (1 - r["mse"] / r["naive_mse"]) * 100
            print(
                f"{name:<20} {r['mse']:>10.4f} {r['mse'] ** 0.5:>8.2f} {r['mae_clip']:>10.2f} {r['r2']:>8.4f} {improvement:>9.1f}%"
            )

        import json

        with open(os.path.join(args.save_path, "comparison_results.json"), "w") as f:
            json.dump(
                {
                    k: {kk: vv for kk, vv in v.items() if kk not in ["preds", "targets"]}
                    for k, v in results.items()
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
