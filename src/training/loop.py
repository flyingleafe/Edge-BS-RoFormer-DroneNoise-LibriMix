"""One generic training loop (docs/refactor-unified-framework.md § "train.py
/ eval.py").

Covers the union of behaviors of the three old trainers (``train.py``,
``train_rps_predictor.py``, ``train_noise_gen.py``): map-style *or* iterable
(online-mixing) datasets, AMP autocast + ``GradScaler``, gradient
accumulation, grad-norm clipping, an optimizer factory ported from
``train.py::get_optimizer``, ``ReduceLROnPlateau`` + early stopping on a
configurable monitor metric, checkpointing (``best.ckpt`` + periodic
``ep{N}_{monitor}_{value:.4f}.ckpt``), and wandb logging (run name =
experiment name, git commit hash, dirty-tree guard, run-id file for the
job runner to pick up — mirrors ``train.py::wandb_init``).

Everything Frame-shaped funnels through the task's :class:`~tasks.codecs.Codec`
(``to_inputs`` / ``call_model`` / ``to_frame``) and
``data_processing.collate`` (``frame_collate`` for batching,
``slice_sample`` for the per-sample view :class:`~metrics.suite.MetricSuite`
needs) — the loop itself never inspects task-specific tensor shapes.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import tdseries as td
import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

import wandb
from data_processing.collate import batch_size as frame_batch_size
from data_processing.collate import frame_collate, slice_sample
from tasks.codecs import Codec
from tasks.task import Task
from training.config import (
    build_dataset,
    build_losses,
    build_metrics,
    build_task_and_codec,
    instantiate_model,
)

__all__ = ["run_training", "get_optimizer", "git_commit_hash", "is_git_dirty"]


# ─── Optimizer factory (ported from train.py::get_optimizer) ─────────────────


def get_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    extra_params: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer by name — adam/adamw/radam/rmsprop/sgd/prodigy/adamw8bit.

    ``weight_decay`` is a named kwarg so every optimizer gets it uniformly
    (the original ``train.py`` version relied on each optimizer accepting it
    via ``**optim_params``); ``extra_params`` are the optimizer-specific
    passthrough kwargs (``config.optimizer`` in the original).
    """
    params = dict(extra_params or {})
    params.setdefault("weight_decay", weight_decay)
    trainable = model.parameters()
    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr, **params)
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, **params)
    if name == "radam":
        return torch.optim.RAdam(trainable, lr=lr, **params)
    if name == "rmsprop":
        return torch.optim.RMSprop(trainable, lr=lr, **params)
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=lr, **params)
    if name == "prodigy":
        from prodigyopt import Prodigy

        return Prodigy(trainable, lr=lr, **params)
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(trainable, lr=lr, **params)
    raise ValueError(f"unknown optimizer {name!r}")


# ─── Git state (dirty-tree guard + commit hash for wandb) ────────────────────


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def is_git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return False


# ─── Small helpers ─────────────────────────────────────────────────────────────


def _to_device(frame: td.Frame, device: torch.device) -> td.Frame:
    return frame.map_data(lambda t: t.to(device))


def _iter_samples(frame: td.Frame) -> Iterable[td.Frame]:
    for i in range(frame_batch_size(frame)):
        yield slice_sample(frame, i)


def _make_loader(dataset: Any, *, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    iterable = isinstance(dataset, IterableDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and not iterable),
        num_workers=num_workers,
        collate_fn=frame_collate,
        persistent_workers=(num_workers > 0 and iterable),
        pin_memory=torch.cuda.is_available(),
    )


def _take(it: Iterator[td.Frame], n: int) -> Iterable[td.Frame]:
    """``next(it)`` ``n`` times — a plain function (not inline ``next()`` in a
    generator expression) so a preceding ``assert it is not None`` narrows
    the type at the call site."""
    for _ in range(n):
        yield next(it)


def _better(mode: str):
    if mode == "min":
        return lambda new, best: new < best
    if mode == "max":
        return lambda new, best: new > best
    raise ValueError(f"optim.monitor_mode must be 'min' or 'max', got {mode!r}")


# ─── Core epoch steps ──────────────────────────────────────────────────────────


def _forward(
    codec: Codec, model: torch.nn.Module, batch: td.Frame, *, device: torch.device, amp: bool
) -> td.Frame:
    inputs = codec.to_inputs(batch)
    with torch.autocast(device_type=device.type, enabled=amp):
        outputs = codec.call_model(model, inputs)
    return codec.to_frame(outputs, batch)


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    codec: Codec,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    batches: Iterable[td.Frame],
    n_batches: int | None,
    device: torch.device,
    amp: bool,
    grad_clip: float | None,
    grad_accum_steps: int,
    epoch: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    count = 0
    pbar = tqdm(batches, total=n_batches, desc=f"train e{epoch}", leave=False)
    for i, batch in enumerate(pbar):
        batch = _to_device(batch, device)
        pred_frame = _forward(codec, model, batch, device=device, amp=amp)
        loss = loss_fn(pred_frame, batch)
        scaler.scale(loss / grad_accum_steps).backward()

        is_last = n_batches is not None and i == n_batches - 1
        if (i + 1) % grad_accum_steps == 0 or is_last:
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_val = float(loss.detach().item())
        total_loss += loss_val
        count += 1
        pbar.set_postfix({"loss": loss_val})
    return total_loss / max(count, 1)


def _validate(
    *,
    model: torch.nn.Module,
    codec: Codec,
    valid_loader: DataLoader,
    metric_suite: Any,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    pairs: list[tuple[td.Frame, td.Frame]] = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = _to_device(batch, device)
            pred_frame = _forward(codec, model, batch, device=device, amp=amp)
            pred_cpu = pred_frame.map_data(lambda t: t.detach().cpu())
            batch_cpu = batch.map_data(lambda t: t.detach().cpu())
            for pred_i, target_i in zip(_iter_samples(pred_cpu), _iter_samples(batch_cpu)):
                pairs.append((pred_i, target_i))
    result = metric_suite.evaluate(pairs)
    return result.aggregate("mean")


# ─── Checkpointing ──────────────────────────────────────────────────────────────


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


# ─── Main entry point ──────────────────────────────────────────────────────────


def run_training(cfg: Any) -> dict[str, float]:
    """Train per ``cfg`` (a composed ``training.config.RootConfig``).

    Sets up the run dir + wandb, builds every component, then runs
    epochs/iterable-stream-chunks with early stopping. Returns the best
    monitored metric value and the epoch it stopped at.
    """
    if is_git_dirty() and not cfg.allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=true"
        )
    commit = git_commit_hash()

    run_dir = Path(cfg.results_root) / cfg.experiment_name
    if run_dir.exists() and not cfg.resume and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} already exists and is non-empty; pass resume=true to continue into it"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task: Task
    task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model).to(device)
    loss_fn = build_losses(cfg.loss).to(device)
    metric_suite = build_metrics(cfg.metrics)

    optimizer = get_optimizer(
        model,
        name=cfg.optim.optimizer,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        extra_params=dict(cfg.optim.optimizer_params or {}),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=cfg.optim.monitor_mode, patience=cfg.optim.patience, factor=cfg.optim.factor
    )
    better = _better(cfg.optim.monitor_mode)

    train_ds = build_dataset(cfg.data.train)
    valid_ds = build_dataset(cfg.data.valid)
    batch_size = cfg.data.batch_size or cfg.batch_size
    num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else cfg.num_workers
    train_loader = _make_loader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    valid_loader = _make_loader(
        valid_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    train_iterable = isinstance(train_ds, IterableDataset)
    train_iter: Iterator[td.Frame] | None = iter(train_loader) if train_iterable else None
    batches_per_epoch: int | None = None
    if train_iterable:
        if cfg.samples_per_validation is None:
            raise ValueError(
                "data.train is an IterableDataset (online-mixing style); "
                "cfg.samples_per_validation must be set"
            )
        batches_per_epoch = math.ceil(cfg.samples_per_validation / batch_size)

    scaler = GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))

    wandb_mode = (
        cfg.logging.mode if cfg.logging.mode else (None if cfg.logging.enabled else "disabled")
    )
    run = wandb.init(
        entity=cfg.logging.entity,
        project=cfg.logging.project,
        name=cfg.experiment_name,
        mode=wandb_mode,
        tags=[task.name, *list(cfg.logging.tags or [])],
        dir=str(run_dir),
        config={"git_commit": commit, "experiment_name": cfg.experiment_name},
    )
    if run is not None and getattr(run, "id", None):
        (run_dir / "wandb_run_id.txt").write_text(run.id)

    monitor = cfg.optim.monitor
    best_metric: float | None = None
    no_improve = 0
    epoch = 0
    for epoch in range(cfg.epochs):
        if train_iterable:
            assert train_iter is not None and batches_per_epoch is not None
            batches = _take(train_iter, batches_per_epoch)
            n_batches = batches_per_epoch
        else:
            batches = train_loader
            n_batches = len(train_loader)

        train_loss = _train_one_epoch(
            model=model,
            codec=codec,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            batches=batches,
            n_batches=n_batches,
            device=device,
            amp=cfg.amp,
            grad_clip=cfg.grad_clip,
            grad_accum_steps=max(1, cfg.grad_accum_steps),
            epoch=epoch,
        )
        val_metrics = _validate(
            model=model,
            codec=codec,
            valid_loader=valid_loader,
            metric_suite=metric_suite,
            device=device,
            amp=cfg.amp,
        )

        metric_value = train_loss if monitor == "loss" else val_metrics[monitor]
        scheduler.step(metric_value)
        lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "lr": lr,
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
        )

        improved = best_metric is None or better(metric_value, best_metric)
        if improved:
            best_metric = metric_value
            no_improve = 0
            _save_checkpoint(model, run_dir / "best.ckpt")
        else:
            no_improve += 1

        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            _save_checkpoint(model, run_dir / f"ep{epoch}_{monitor}_{metric_value:.4f}.ckpt")

        if no_improve >= cfg.patience:
            break

    wandb.finish()
    assert best_metric is not None
    return {f"best_{monitor}": best_metric, "final_epoch": float(epoch)}
