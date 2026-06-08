#!/usr/bin/env python
"""
Training script for the sinusoidal + filtered-noise drone-noise generator.

Model: `models.generative.DroneNoisePlusFilterGen`
  - Per-rotor harmonic oscillator bank from RPS (sinusoidal modelling)
  - RPS-conditioned per-frame magnitude filter applied to white noise
  - Output = harmonic + filtered noise

Data: `data_processing.noise_rps_dataset.build_noise_rps_datasets`
  - DREGON `in_flight_noise` split (motor telemetry @ ~929 Hz)
  - Michael's recordings (data/new-drone-noises/, motor telemetry @ ~30 Hz)
  - Train/val split by time-axis hold-out within each recording (val_pct).

Loss: Multi-scale STFT (linear + log magnitude) — `MultiScaleSTFT`.

Usage
-----
    python train_noise_gen.py --config configs/noise_gen.yaml

Optional flags:
    --device {cuda,cpu,auto}
    --wandb-project <name>    # enables wandb logging
    --resume <ckpt>           # resume from a checkpoint
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_processing.noise_rps_dataset import build_noise_rps_datasets
from models.generative import DroneNoisePlusFilterGen, MultiScaleSTFT

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, default="configs/noise_gen.yaml")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, enables wandb logging under the given project.",
    )
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    p.add_argument(
        "--run-name", type=str, default=None, help="Run subdirectory name (default: timestamp)."
    )
    return p.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# ---------------------------------------------------------------------------
# Build pieces
# ---------------------------------------------------------------------------


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DictConfig]:
    train_ds, val_ds = build_noise_rps_datasets(
        dregon_dir=cfg.data.get("dregon_dir"),
        michaels_dir=cfg.data.get("michaels_dir"),
        sample_rate=cfg.audio.sample_rate,
        chunk_size=cfg.audio.chunk_size,
        train_samples=cfg.training.train_samples_per_epoch,
        val_samples=cfg.training.val_samples_per_epoch,
        val_pct=cfg.data.val_pct,
        seed=cfg.training.seed,
        cache_dir=cfg.data.get("cache_dir"),
        channel_policy=cfg.data.get("channel_policy", "first"),
        rps_normalize=cfg.data.get("filter_rps_scale"),
    )

    n_workers = int(cfg.training.num_workers)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        num_workers=n_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        persistent_workers=n_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        num_workers=n_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=n_workers > 0,
    )
    return train_dl, val_dl, cfg


def build_model(cfg: DictConfig) -> DroneNoisePlusFilterGen:
    return DroneNoisePlusFilterGen(
        n_motors=cfg.model.n_motors,
        n_harmonics=cfg.model.n_harmonics,
        sample_rate=cfg.audio.sample_rate,
        use_basis_gain=cfg.model.use_basis_gain,
        filter_n_freqs=cfg.model.filter_n_freqs,
        filter_n_frames=cfg.model.filter_n_frames,
        filter_hidden=cfg.model.filter_hidden,
        filter_n_layers=cfg.model.filter_n_layers,
    )


def build_loss(cfg: DictConfig) -> MultiScaleSTFT:
    return MultiScaleSTFT(
        n_ffts=list(cfg.loss.n_ffts),
        log_weight=cfg.loss.log_weight,
        loss_type=cfg.loss.loss_type,
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "rps": batch["rps"].to(device, non_blocking=True),
        "audio": batch["audio"].to(device, non_blocking=True),
    }


def _forward_loss(
    model: DroneNoisePlusFilterGen,
    loss_fn: MultiScaleSTFT,
    rps: torch.Tensor,
    target: torch.Tensor,
    harmonic_branch_weight: float,
) -> tuple[torch.Tensor, dict]:
    out = model(rps)
    pred = out["audio"]
    main_loss = loss_fn(pred, target)
    total = main_loss
    logs = {"loss": main_loss.detach().item()}
    if harmonic_branch_weight > 0:
        h_loss = loss_fn(out["harmonic"], target)
        total = total + harmonic_branch_weight * h_loss
        logs["harmonic_loss"] = h_loss.detach().item()
    return total, logs


def train_one_epoch(
    model,
    loss_fn,
    optimizer,
    dl,
    device,
    cfg,
    epoch,
    *,
    wandb=None,
    grad_clip: float | None = None,
):
    model.train()
    total_loss = 0.0
    n = 0
    h_loss_total = 0.0
    pbar = tqdm(dl, desc=f"train {epoch}", leave=False)
    for batch in pbar:
        b = _move_batch(batch, device)
        optimizer.zero_grad()
        loss, logs = _forward_loss(
            model,
            loss_fn,
            b["rps"],
            b["audio"],
            harmonic_branch_weight=cfg.loss.get("harmonic_branch_weight", 0.0),
        )
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += logs["loss"] * b["audio"].shape[0]
        if "harmonic_loss" in logs:
            h_loss_total += logs["harmonic_loss"] * b["audio"].shape[0]
        n += b["audio"].shape[0]
        pbar.set_postfix(loss=f"{logs['loss']:.3f}")
        if wandb is not None:
            wandb.log(
                {"train/loss_step": logs["loss"], **{k: v for k, v in logs.items() if k != "loss"}}
            )
    avg = total_loss / max(n, 1)
    out = {"train/loss": avg}
    if h_loss_total:
        out["train/harmonic_loss"] = h_loss_total / max(n, 1)
    return out


@torch.no_grad()
def validate(model, loss_fn, dl, device, cfg) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    h_loss_total = 0.0
    for batch in tqdm(dl, desc="val", leave=False):
        b = _move_batch(batch, device)
        loss, logs = _forward_loss(
            model,
            loss_fn,
            b["rps"],
            b["audio"],
            harmonic_branch_weight=cfg.loss.get("harmonic_branch_weight", 0.0),
        )
        total_loss += logs["loss"] * b["audio"].shape[0]
        if "harmonic_loss" in logs:
            h_loss_total += logs["harmonic_loss"] * b["audio"].shape[0]
        n += b["audio"].shape[0]
    avg = total_loss / max(n, 1)
    out = {"val/loss": avg}
    if h_loss_total:
        out["val/harmonic_loss"] = h_loss_total / max(n, 1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = pick_device(args.device)

    with open(args.config) as f:
        cfg = DictConfig(OmegaConf.load(f))

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Output dir
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg.training.log_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] logging to {out_dir}")

    # Save resolved config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f)

    # Optional wandb
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            config=cfg.to_dict(),
            dir=str(out_dir),
        )

    # Data
    print("[data] building datasets...")
    train_dl, val_dl, _ = build_dataloaders(cfg)
    print(
        f"[data] train sources: {len(train_dl.dataset.records)}, val sources: {len(val_dl.dataset.records)}"
    )  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    print(f"[data] train samples/epoch: {len(train_dl.dataset)}, val: {len(val_dl.dataset)}")  # pyright: ignore[reportArgumentType]

    # Model
    print("[model] building DroneNoisePlusFilterGen...")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=float(cfg.training.lr))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.training.scheduler_patience,
        factor=cfg.training.scheduler_factor,
    )
    loss_fn = build_loss(cfg).to(device)

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"[resume] from epoch {start_epoch}, best_val={best_val:.4f}")

    # Training
    epochs_no_improve = 0
    grad_clip = cfg.training.get("grad_clip") or None
    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        tlogs = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            train_dl,
            device,
            cfg,
            epoch,
            wandb=wandb_run,
            grad_clip=grad_clip,
        )
        vlogs = validate(model, loss_fn, val_dl, device, cfg)
        dt = time.time() - t0
        msg = f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in {**tlogs, **vlogs}.items())
        msg += f"  lr={optimizer.param_groups[0]['lr']:.2e}  dt={dt:.1f}s"
        print(msg)
        if wandb_run is not None:
            wandb_run.log({**tlogs, **vlogs, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})

        scheduler.step(vlogs["val/loss"])

        # Checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": cfg.to_dict(),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if vlogs["val/loss"] < best_val:
            best_val = vlogs["val/loss"]
            torch.save(ckpt, out_dir / "best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.training.early_stop_patience:
            print(f"[early stop] no improvement for {epochs_no_improve} epochs.")
            break

    # Final summary
    summary = {
        "best_val_loss": best_val,
        "epochs_run": epoch + 1,
        "run_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] best val loss: {best_val:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
