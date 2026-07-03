"""Shared config-building helper for the training-framework tests."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def make_tiny_config(
    *,
    results_root: str,
    experiment_name: str = "tiny_rps_test",
    frame_rate: list[int] | None = None,
    n_train: int = 6,
    n_valid: int = 4,
    epochs: int = 1,
    batch_size: int = 2,
    monitor: str = "mse",
) -> Any:
    """Build a minimal RootConfig-shaped ``DictConfig`` around
    ``tests.training._fixtures`` (``TinyRPSFrameDataset`` / ``TinyRPSModel``)
    — the same shape ``@hydra.main`` would compose from ``conf/``, but built
    directly so tests don't need a real dataset/checkpoint on disk.
    """
    frame_rate = list(frame_rate) if frame_rate is not None else [16000, 512]
    data = {
        "train": {
            "_target_": "tests.training._fixtures.TinyRPSFrameDataset",
            "params": {"n_samples": n_train, "duration_s": 0.5, "sample_rate": 16000, "seed": 0},
            "iterable": False,
        },
        "valid": {
            "_target_": "tests.training._fixtures.TinyRPSFrameDataset",
            "params": {"n_samples": n_valid, "duration_s": 0.5, "sample_rate": 16000, "seed": 1},
            "iterable": False,
        },
        "batch_size": None,
        "num_workers": None,
    }
    model = {
        "task": "rps_prediction",
        "task_params": {"n_channels": None, "sr": [16000, 1], "frame_rate": frame_rate},
        "_target_": "tests.training._fixtures.TinyRPSModel",
        "params": {"hop_length": 512, "num_rotors": 4, "hidden": 8},
        "model_type": None,
        "legacy_config_path": None,
    }
    loss = {
        "terms": [
            {
                "name": "pit_mse",
                "_target_": "losses.PITMSELoss",
                "weight": 1.0,
                "params": {"rate": [16000, 512]},
            }
        ]
    }
    metrics = {
        "terms": [
            {
                "name": "mse",
                "_target_": "metrics.RPSMetric",
                "params": {"stat": "mse", "rate": [16000, 512]},
            },
        ]
    }
    optim = {
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "optimizer_params": {},
        "patience": 5,
        "factor": 0.5,
        "monitor": monitor,
        "monitor_mode": "min",
    }
    logging = {
        "enabled": True,
        "entity": "test",
        "project": "test",
        "mode": "disabled",
        "tags": [],
    }
    return OmegaConf.create(
        {
            "experiment_name": experiment_name,
            "seed": 0,
            "validate_only": False,
            "allow_dirty": True,
            "resume": False,
            "results_root": results_root,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": 0,
            "patience": 5,
            "grad_clip": None,
            "grad_accum_steps": 1,
            "amp": False,
            "samples_per_validation": None,
            "checkpoint_every": 0,
            "checkpoint": None,
            "data": data,
            "model": model,
            "loss": loss,
            "metrics": metrics,
            "optim": optim,
            "logging": logging,
        }
    )
