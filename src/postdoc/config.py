from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LocalConfig:
    gpus: int = 2
    results_dir: str = "results"
    data_dir: str = "datasets"


@dataclass
class CloudConfig:
    clouds: list[str] = field(default_factory=lambda: ["gcp"])
    monthly_budget_soft: int = 100
    r2_bucket: str = ""
    r2_endpoint: str = ""


@dataclass
class WandbConfig:
    project: str = ""
    entity: str = ""


@dataclass
class PostdocConfig:
    backend: str = "local"
    local: LocalConfig = field(default_factory=LocalConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(path: Path) -> PostdocConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    local_raw = raw.get("local", {})
    cloud_raw = raw.get("cloud", {})
    wandb_raw = raw.get("wandb", {})

    config = PostdocConfig(
        backend=raw.get("backend", "local"),
        local=LocalConfig(**{k: v for k, v in local_raw.items() if k in LocalConfig.__dataclass_fields__}),
        cloud=CloudConfig(**{k: v for k, v in cloud_raw.items() if k in CloudConfig.__dataclass_fields__}),
        wandb=WandbConfig(**{k: v for k, v in wandb_raw.items() if k in WandbConfig.__dataclass_fields__}),
    )

    env_backend = os.environ.get("POSTDOC_BACKEND")
    if env_backend:
        config.backend = env_backend

    return config
