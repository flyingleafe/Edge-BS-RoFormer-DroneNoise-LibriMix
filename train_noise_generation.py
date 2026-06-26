#!/usr/bin/env python3
"""
Train position-aware noise-generation models on DREGON-LM datasets.

The inverse of ``train_rps_predictor.py``: instead of audio -> RPS, this learns
RPS + array geometry -> the drone noise observed at each microphone.

Model: ``models.generative.PositionalHarmonicNoiseGen`` — per-rotor harmonic +
filtered-noise emitter, propagated to every mic (1/r attenuation + delay) and
summed. All 8 DREGON mics are rendered jointly (native multi-observer).

Data: the **same** DREGON-LM ``sample_*`` chunk format as RPS prediction
(``rps.npy`` + a multichannel audio file), but the target is the clean
``noise.wav`` (no speech mixing) and microphone/rotor positions are attached
from the recording geometry (``data_processing.dregon.get_geometry`` — i.e. the
``TimeFrame.global_data`` convention).

Usage:
    python train_noise_generation.py \
        --data_root datasets/DREGON-LM-V4 --dregon_dir data/DREGON \
        --model positional_harmonic_gen
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import wandb as _wandb
from data_processing.dregon import get_geometry
from data_processing.michaels import get_geometry as get_michaels_geometry
from data_processing.online_mixing import (
    TimeFrameNoisePool,
    _extract_audio_array,
    interpolate_rps_to_stft_grid,
)
from models.generative import MultiScaleSTFT, PositionalHarmonicNoiseGen
from tasks.noise_generation import DroneCodebook, geometry_to_rel_pos

# wandb's type stubs omit run/init/log/login/finish; treat as untyped.
wandb: Any = _wandb

# ─── Dataset ──────────────────────────────────────────────────────────────────


class DREGONNoiseGenDataset(Dataset):
    """Load ``noise.wav`` + ``rps.npy`` from DREGON-LM chunks for noise generation.

    Yields ``(rps_audio_rate, rel_pos, target, drone_name)``:
    * ``rps_audio_rate`` : ``(R, T)`` per-rotor speed (Hz) upsampled to audio rate
    * ``rel_pos``        : ``(M, R, 3)`` rotor->mic vectors (constant per recording
      geometry; carried as ``TimeFrame.global_data`` semantics)
    * ``target``         : ``(M, T)`` clean multichannel drone noise
    * ``drone_name``     : ``str`` key into the external :class:`DroneCodebook`

    The geometry (``mic_positions``/``rotor_positions``) and ``drone_name`` are the
    same for every chunk in a single-source dataset, so they are shared. (Mixed
    multi-drone datasets need per-chunk geometry + ``drone_name`` — see the task
    AGENTS doc.)
    """

    def __init__(
        self,
        data_dir: str,
        mic_positions: np.ndarray,
        rotor_positions: np.ndarray,
        *,
        target_file: str = "noise.wav",
        drone_name: str = "default",
    ):
        self.samples = sorted(
            d
            for d in glob.glob(os.path.join(data_dir, "sample_*"))
            if os.path.isfile(os.path.join(d, target_file))
            and os.path.isfile(os.path.join(d, "rps.npy"))
        )
        if not self.samples:
            raise FileNotFoundError(
                f"No chunks with both '{target_file}' and 'rps.npy' under {data_dir}. "
                f"Noise generation needs a clean-noise target — build a dataset that "
                f"keeps '{target_file}' (e.g. a non --real_valid multichannel set)."
            )
        self.target_file = target_file
        self.drone_name = str(drone_name)
        # (M, R, 3) float32, shared across all samples.
        self.rel_pos = torch.from_numpy(geometry_to_rel_pos(mic_positions, rotor_positions))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        d = self.samples[idx]
        target, _ = torchaudio.load(os.path.join(d, self.target_file))  # (M, T)
        n_samples = target.shape[-1]

        rps = torch.from_numpy(np.load(os.path.join(d, "rps.npy"))).float()  # (R, M_motor)
        # Upsample RPS to the audio sample grid (shape-stretch; rps and audio are
        # co-extensive in DREGON-LM chunks — same gotcha/justification as
        # DREGONRPSDataset, but here to the audio rate, not the STFT grid).
        rps_up = F.interpolate(
            rps.unsqueeze(0), size=n_samples, mode="linear", align_corners=False
        ).squeeze(0)  # (R, T)

        rel_pos = self.rel_pos[: target.shape[0]]  # (M, R, 3) — match mic count
        return rps_up, rel_pos, target.float(), self.drone_name


# ─── Online streaming dataset (reuses the RPS online-mixing slicer) ──────────────


def _drone_name(tf) -> str:
    """Map a noise recording to a codebook drone name (DREGON vs Michael's)."""
    rid = str(tf.tags.get("recording_id", ""))
    return "michaels" if rid.startswith("michaels") else "dregon"


def _noise_item(tf, *, sample_rate: int, duration_s: float):
    """Turn a sliced noise ``TimeFrame`` into a noise-gen training item.

    Reuses the online-mixing extractors: clean multichannel noise as the target,
    RPS interpolated onto the audio grid (``hop_length=1`` => one value per
    sample), and the per-frame geometry from ``global_data``.
    """
    n_samples = int(round(duration_s * sample_rate))
    audio = _extract_audio_array(tf, target_len=n_samples)  # (C, T)
    rps = interpolate_rps_to_stft_grid(tf, n_frames=n_samples, hop_length=1)  # (R, T)
    gd = tf.global_data or {}
    rel = geometry_to_rel_pos(gd["mic_positions"], gd["rotor_positions"])  # (M, R, 3)
    rel = rel[: audio.shape[0]]  # match channel count
    return (
        torch.from_numpy(rps),
        torch.from_numpy(rel),
        torch.from_numpy(audio),
        _drone_name(tf),
    )


class OnlineNoiseGenDataset(IterableDataset):
    """Finite stream of randomly-sliced ``(rps, rel_pos, noise, drone_name)``.

    Wraps :class:`data_processing.online_mixing.TimeFrameNoisePool` (the existing
    on-the-fly slicer) — one "epoch" is ``length`` random slices. Each slice
    carries its own geometry and drone identity, so DREGON + Michael's stream
    together.
    """

    def __init__(self, pool, *, sample_rate: int, duration_s: float, base_seed: int, length: int):
        self.pool = pool
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        self.base_seed = base_seed
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        nworkers = 1 if info is None else info.num_workers
        rng = np.random.default_rng(self.base_seed + wid)
        # split the epoch across workers
        n = self.length // nworkers + (1 if wid < self.length % nworkers else 0)
        for _ in range(n):
            tf = self.pool.sample_timeframe(rng, self.duration_s)
            yield _noise_item(tf, sample_rate=self.sample_rate, duration_s=self.duration_s)


class FixedNoiseGenDataset(Dataset):
    """A deterministic, pre-sliced validation set (so eval is reproducible)."""

    def __init__(self, pool, *, sample_rate: int, duration_s: float, seed: int, n: int):
        rng = np.random.default_rng(seed)
        self.items = [
            _noise_item(
                pool.sample_timeframe(rng, duration_s),
                sample_rate=sample_rate,
                duration_s=duration_s,
            )
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def build_noise_pools(config_path: str, *, sample_rate: int, duration_s: float):
    """Build (train_pool, valid_pool) from the online noise-gen YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train_pool = TimeFrameNoisePool.from_config(
        cfg["noise_train"], duration_s=duration_s, sample_rate=sample_rate
    )
    valid_pool = TimeFrameNoisePool.from_config(
        cfg["noise_valid"], duration_s=duration_s, sample_rate=sample_rate
    )
    return cfg, train_pool, valid_pool


def _pool_drone_names(*pools) -> list[str]:
    names = set()
    for pool in pools:
        for rec in pool.records:
            names.add(_drone_name(rec["tf"]))
    return sorted(names)


# ─── Model factory ────────────────────────────────────────────────────────────


MODEL_REGISTRY = {
    "positional_harmonic_gen": PositionalHarmonicNoiseGen,
}


def get_model(
    model_name: str,
    *,
    sample_rate: int = 16000,
    n_harmonics: int = 100,
    use_diff_noise: bool = True,
    cond_dim: int = 0,
) -> nn.Module:
    """Construct a noise-generation model by name.

    ``cond_dim > 0`` FiLM-conditions the emitter on an external code ``z``
    ``(B, cond_dim)``; the per-drone codes live in a separate
    :class:`DroneCodebook` (not in the model), and the training loop supplies
    ``z`` each step.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](
        sample_rate=sample_rate,
        n_harmonics=n_harmonics,
        use_diff_noise=use_diff_noise,
        cond_dim=cond_dim,
    )


def build_loss(args: argparse.Namespace) -> MultiScaleSTFT:
    return MultiScaleSTFT(
        n_ffts=list(args.stft_sizes),
        log_weight=args.log_weight,
        loss_type=args.loss_type,
    )


def _spectral_loss(loss_fn: MultiScaleSTFT, pred: torch.Tensor, target: torch.Tensor):
    """Multi-scale STFT loss over a multichannel batch.

    ``pred``/``target`` are ``(B, M, T)``; the mic axis is folded into the batch
    so each (clip, mic) waveform is scored independently.
    """
    b, m, t = pred.shape
    return loss_fn(pred.reshape(b * m, t), target.reshape(b * m, t))


# ─── Checkpoint bundling (model + external codebook) ─────────────────────────────


def save_bundle(path: str, model: nn.Module, codebook: DroneCodebook | None) -> None:
    """Save the generator and its external codebook in one file.

    The codebook is separate from the model (so model weights never resize when
    drones are added), but a single bundle keeps a run self-contained.
    """
    bundle: dict[str, Any] = {"model": model.state_dict()}
    if codebook is not None:
        bundle["codebook"] = codebook.state_dict()
        bundle["cond_dim"] = codebook.dim
        bundle["drone_names"] = codebook.names()
    torch.save(bundle, path)


def _load_for_adaptation(
    path: str, model: nn.Module, codebook: DroneCodebook | None, device
) -> None:
    """Warm-start from a bundle for few-shot adaptation to a new drone.

    Loads the trained generator. Known drones' codes load by name; the current
    (new) drone keeps its fresh small-random code (``strict=False``), which is
    exactly what gets optimised when ``--freeze_emitter`` is set.
    """
    bundle = torch.load(path, map_location=device)
    model.load_state_dict(bundle["model"])
    if codebook is not None and "codebook" in bundle:
        codebook.load_state_dict(bundle["codebook"], strict=False)


# ─── Eval ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, *, codebook=None, progress=False) -> float:
    model.eval()
    total = 0.0
    n = 0
    for rps, rel_pos, target, drone_names in tqdm(
        loader, desc="eval", unit="batch", leave=False, disable=not progress
    ):
        rps, rel_pos, target = rps.to(device), rel_pos.to(device), target.to(device)
        z = codebook(list(drone_names)).to(device) if codebook is not None else None
        pred = model(rps, rel_pos, z)
        loss = _spectral_loss(loss_fn, pred, target)
        bs = target.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


# ─── Training ───────────────────────────────────────────────────────────────────


def train_model(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    if args.online_config:
        # Online stream: clean noise + RPS + per-frame geometry sliced on the fly
        # from long recordings (reuses the RPS online-mixing slicer). DREGON and
        # Michael's stream together, each frame carrying its own geometry + drone.
        cfg, train_pool, valid_pool = build_noise_pools(
            args.online_config, sample_rate=args.sample_rate, duration_s=args.duration_s
        )
        base_seed = int(cfg.get("base_seed", 0))
        codebook_names = _pool_drone_names(train_pool, valid_pool)
        print(
            f"Online | train recs {[r['tf'].tags.get('recording_id') for r in train_pool.records]}"
        )
        print(
            f"       | valid recs {[r['tf'].tags.get('recording_id') for r in valid_pool.records]}"
        )
        print(f"       | drones {codebook_names}")
        train_ds: Dataset | IterableDataset = OnlineNoiseGenDataset(
            train_pool,
            sample_rate=args.sample_rate,
            duration_s=args.duration_s,
            base_seed=base_seed,
            length=args.samples_per_epoch,
        )
        valid_ds: Dataset = FixedNoiseGenDataset(
            valid_pool,
            sample_rate=args.sample_rate,
            duration_s=args.duration_s,
            seed=base_seed + 1,
            n=args.num_valid,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True
        )
        print(f"Train: {len(train_ds)} stream | Valid: {len(valid_ds)} fixed samples")
    else:
        if args.geometry == "michaels":
            mic_pos, rotor_pos = get_michaels_geometry()
            geom_src = "michaels (DJI Matrice 100)"
        else:
            mic_pos, rotor_pos = get_geometry(args.dregon_dir)
            geom_src = args.dregon_dir
        print(f"Geometry: {mic_pos.shape[0]} mics, {rotor_pos.shape[0]} rotors (from {geom_src})")

        train_ds = DREGONNoiseGenDataset(
            os.path.join(args.data_root, "train"),
            mic_pos,
            rotor_pos,
            target_file=args.target_file,
            drone_name=args.drone_name,
        )
        valid_ds = DREGONNoiseGenDataset(
            os.path.join(args.data_root, "valid"),
            mic_pos,
            rotor_pos,
            target_file=args.target_file,
            drone_name=args.drone_name,
        )
        codebook_names = [args.drone_name]
        print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} samples")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = get_model(
        args.model,
        sample_rate=args.sample_rate,
        n_harmonics=args.n_harmonics,
        use_diff_noise=not args.no_diff_noise,
        cond_dim=args.cond_dim,
    ).to(device)

    # Per-drone conditioning codes live OUTSIDE the model (name-keyed), so adding
    # a drone never resizes model weights. cond_dim == 0 disables conditioning.
    codebook: DroneCodebook | None = None
    if args.cond_dim > 0:
        codebook = DroneCodebook(args.cond_dim, names=codebook_names).to(device)

    # Optional warm-start for few-shot adaptation to a new drone: load a trained
    # generator (+ codes), then either fine-tune all or freeze the emitter and
    # fit only the new drone's code.
    if args.init_checkpoint:
        _load_for_adaptation(args.init_checkpoint, model, codebook, device)
    if args.freeze_emitter:
        for p in model.parameters():
            p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | params: {n_params:,}")
    if codebook is not None:
        print(f"Codebook: dim {args.cond_dim} | drones {codebook.names()}")

    loss_fn = build_loss(args).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if codebook is not None:
        params += list(codebook.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience
    )

    best_val = float("inf")
    epochs_no_improve = 0
    best_path = os.path.join(args.save_path, f"best_{args.model}.pt")

    print(f"\n{'Epoch':>5} {'Train':>10} {'Val':>10} {'LR':>10}")
    print("-" * 40)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        n = 0
        for rps, rel_pos, target, drone_names in tqdm(
            train_loader,
            desc=f"train e{epoch}",
            unit="batch",
            leave=False,
            disable=not args.epoch_progress,
        ):
            rps, rel_pos, target = rps.to(device), rel_pos.to(device), target.to(device)
            z = codebook(list(drone_names)).to(device) if codebook is not None else None
            optimizer.zero_grad()
            pred = model(rps, rel_pos, z)
            loss = _spectral_loss(loss_fn, pred, target)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            run_loss += loss.item() * target.shape[0]
            n += target.shape[0]

        train_loss = run_loss / max(n, 1)
        val_loss = evaluate(
            model, valid_loader, loss_fn, device, codebook=codebook, progress=args.epoch_progress
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} {train_loss:10.4f} {val_loss:10.4f} {lr:10.1e}")

        if wandb.run is not None and not wandb.run.disabled:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss, "lr": lr})

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            save_bundle(best_path, model, codebook)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nTraining time: {(time.time() - t0) / 60:.1f} min | best val: {best_val:.4f}")
    return {"model": args.model, "best_val": best_val, "best_path": best_path}


def wandb_init(args: argparse.Namespace) -> None:
    key = getattr(args, "wandb_key", "") or os.environ.get("WANDB_API_KEY", "")
    if not key.strip():
        wandb.init(mode="disabled")
        return
    wandb.login(key=key)
    wandb.init(
        entity="flyingleafe",
        project="noise-generation",
        name=f"{args.model}_{os.path.basename(args.data_root.rstrip('/'))}",
        config=vars(args),
    )


def main():
    p = argparse.ArgumentParser(description="Train position-aware noise-generation models")
    p.add_argument("--model", default="positional_harmonic_gen", choices=list(MODEL_REGISTRY))
    p.add_argument("--data_root", default="datasets/DREGON-LM-V4")
    p.add_argument("--dregon_dir", default="data/DREGON", help="DREGON dir for array geometry.")
    p.add_argument(
        "--geometry",
        choices=["dregon", "michaels"],
        default="dregon",
        help="Array geometry source. 'michaels' uses the DJI Matrice 100 rig "
        "(data_processing.michaels.get_geometry); use it for Michael's-source datasets.",
    )
    p.add_argument("--target_file", default="noise.wav", help="Per-chunk clean-noise target file.")
    p.add_argument(
        "--online_config",
        default="",
        help="Online noise-gen YAML (noise_train/noise_valid source specs). When "
        "set, clean noise + RPS + per-frame geometry are sliced on the fly from "
        "long recordings (reuses the RPS online-mixing slicer) instead of "
        "precomputed chunks; DREGON + Michael's stream jointly with per-drone "
        "conditioning. Overrides --data_root/--geometry/--drone_name.",
    )
    p.add_argument(
        "--samples_per_epoch",
        type=int,
        default=2000,
        help="Online mode: random slices per epoch (the stream is infinite).",
    )
    p.add_argument(
        "--num_valid",
        type=int,
        default=128,
        help="Online mode: fixed validation slices (deterministic).",
    )
    p.add_argument(
        "--duration_s", type=float, default=1.0, help="Online mode: slice length (seconds)."
    )
    p.add_argument("--save_path", default="results/noise_generation")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--scheduler_patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_harmonics", type=int, default=100)
    p.add_argument(
        "--no_diff_noise", action="store_true", help="Disable the filtered-noise branch."
    )
    p.add_argument(
        "--cond_dim",
        type=int,
        default=0,
        help="Per-drone conditioning-code dim d (0 = single-drone, no "
        "conditioning). >0 FiLM-conditions the emitter on an external code z; "
        "the codes live in a name-keyed DroneCodebook, not in the model.",
    )
    p.add_argument(
        "--drone_name",
        default="default",
        help="Drone name (codebook key) for this single-source dataset.",
    )
    p.add_argument(
        "--init_checkpoint",
        default="",
        help="Warm-start bundle for few-shot adaptation: loads the trained "
        "generator (+ known codes by name); a new drone keeps a fresh code.",
    )
    p.add_argument(
        "--freeze_emitter",
        action="store_true",
        help="Freeze the generator and train only the codebook (few-shot "
        "adaptation of a new drone's code).",
    )
    p.add_argument("--stft_sizes", type=int, nargs="+", default=[2048, 1024, 512, 256, 128])
    p.add_argument("--log_weight", type=float, default=1.0)
    p.add_argument("--loss_type", choices=["L1", "L2"], default="L1")
    p.add_argument("--wandb_key", default="")
    p.add_argument(
        "--epoch-progress", "--epoch_progress", dest="epoch_progress", action="store_true"
    )
    args = p.parse_args()

    load_dotenv()
    os.makedirs(args.save_path, exist_ok=True)
    wandb_init(args)
    train_model(args)
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
