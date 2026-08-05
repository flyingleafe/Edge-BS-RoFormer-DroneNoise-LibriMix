"""Structured Hydra configs for the unified train/eval framework.

See docs/refactor-unified-framework.md § "Hydra config architecture". Every
dataclass here is registered in Hydra's ``ConfigStore`` so ``@hydra.main``
type-checks a composed config at compose time; the ``build_*`` /
``instantiate_*`` functions turn a composed (or hand-built, for tests)
config into the live Python objects the training loop needs.

Component dispatch convention: every component config (dataset, loss term,
metric term) carries a Hydra-style ``_target_`` dotted path + a free-form
``params`` dict, built via :func:`instantiate_target`
(``hydra.utils.instantiate`` under the hood). **Model** configs are the one
exception — see :func:`instantiate_model` — because a model may come from
either that same ``_target_`` convention (routed through
``models.registry.build_model`` for the RPS-model registry) *or* the legacy
``model_type`` + ``legacy_config_path`` path through
``models.registry.LEGACY_MODEL_BUILDERS`` (pre-existing architectures with
their own YAML config format). ``instantiate_model`` is a small manual dispatcher
rather than a bare ``hydra.utils.instantiate(cfg.model)`` call so that
``ModelConfig``'s descriptive fields (``task``, ``task_params``,
``model_type``, ``legacy_config_path``) never leak into a component
constructor's kwargs — see its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from typing import Any

import hydra.utils
import tdseries as td
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from torch.utils.data import Dataset, IterableDataset

from losses import CompositeLoss, LossTerm
from metrics import MetricSuite
from tasks.codecs import Codec, build_codec
from tasks.task import TASK_FACTORIES, Task

__all__ = [
    "DatasetSpec",
    "DataConfig",
    "ModelConfig",
    "LossTermConfig",
    "LossConfig",
    "MetricTermConfig",
    "MetricsConfig",
    "OptimConfig",
    "WandbConfig",
    "ArtifactsConfig",
    "LoraConfig",
    "RootConfig",
    "register_configs",
    "instantiate_target",
    "build_dataset",
    "instantiate_model",
    "build_task_and_codec",
    "build_losses",
    "build_metrics",
]


# ─── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class DatasetSpec:
    """One dataset (train or valid): a ``_target_`` dotted path + kwargs.

    ``_target_`` may name a class directly (``data_processing.frame_datasets.
    DregonLMFrameDataset``) or a classmethod (``...OnlineMixFrameDataset.
    from_yaml``) — both are plain callables from Hydra's point of view.
    """

    _target_: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)
    iterable: bool = False


@dataclass
class DataConfig:
    """Train/valid dataset pair, plus optional per-experiment loader overrides."""

    train: DatasetSpec = field(default_factory=DatasetSpec)
    valid: DatasetSpec = field(default_factory=DatasetSpec)
    batch_size: int | None = None
    num_workers: int | None = None


@dataclass
class ModelConfig:
    """Task + model. See module docstring for the two instantiation paths."""

    task: str = MISSING
    task_params: dict[str, Any] = field(default_factory=dict)
    # Path A: RPS-model registry (models.registry.build_model) or any other
    # plain-kwargs constructor.
    _target_: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # Path B: legacy models.registry.LEGACY_MODEL_BUILDERS dispatch.
    model_type: str | None = None
    legacy_config_path: str | None = None


@dataclass
class LossTermConfig:
    name: str = MISSING
    _target_: str = MISSING
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    terms: list[LossTermConfig] = field(default_factory=list)


@dataclass
class MetricTermConfig:
    name: str = MISSING
    _target_: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    terms: list[MetricTermConfig] = field(default_factory=list)


@dataclass
class OptimConfig:
    """Optimizer + ReduceLROnPlateau scheduler + the metric that drives both
    the scheduler and checkpointing/early-stopping (``monitor``)."""

    optimizer: str = "adamw"  # adam/adamw/radam/rmsprop/sgd/prodigy/adamw8bit
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer_params: dict[str, Any] = field(default_factory=dict)
    patience: int = 5  # ReduceLROnPlateau patience (epochs)
    factor: float = 0.5  # ReduceLROnPlateau reduce factor
    monitor: str = MISSING  # metric name from metrics.terms (or "loss")
    monitor_mode: str = "min"  # "min" or "max"


@dataclass
class WandbConfig:
    enabled: bool = True
    entity: str = "flyingleafe"
    project: str = "harmonic-noise-suppression"
    mode: str | None = None  # e.g. "disabled"/"offline" override
    tags: list[str] = field(default_factory=list)


@dataclass
class ArtifactsConfig:
    """Cloudflare R2 artifact upload — see ``training.artifacts.ArtifactStore``
    and docs/refactor-unified-framework.md § "Future expansions". Uploads are
    a no-op (with a log line) when ``enabled=False`` or R2 credentials
    (``.env``: ``R2_ACCOUNT_ID`` + AWS keys) are missing — headless/CI safe by
    default even with ``enabled=True``.
    """

    enabled: bool = True
    bucket: str = "ml-data"
    prefix: str = "artifacts"
    upload_checkpoints: bool = True
    upload_val_samples: bool = True
    num_val_samples: int = 6


@dataclass
class LoraConfig:
    """LoRA fine-tuning seam — see ``training.lora.maybe_apply_lora``. Off by
    default; the port from the old ``loralib``-based trainer is deliberately
    deferred (enabling raises ``NotImplementedError`` with a pointer to the
    old implementation), the config shape exists so a future PR doesn't need
    to touch the training loop's call site.
    """

    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] | None = None
    checkpoint: str | None = None


@dataclass
class RootConfig:
    experiment_name: str = MISSING
    seed: int = 0
    validate_only: bool = False
    allow_dirty: bool = False
    resume: bool = False
    results_root: str = "results"
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    patience: int = 10  # early-stop patience (epochs without monitor improvement)
    grad_clip: float | None = 5.0
    grad_accum_steps: int = 1
    amp: bool = True
    amp_dtype: str = "float16"  # autocast dtype on cuda: "float16" | "bfloat16"
    samples_per_validation: int | None = None  # iterable-dataset "epoch" size
    checkpoint_every: int = 0  # 0 = only best.ckpt; N>0 = also every N epochs
    checkpoint: str | None = None  # eval.py: explicit checkpoint path override
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    logging: WandbConfig = field(default_factory=WandbConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)


def register_configs() -> None:
    """Register :class:`RootConfig` (and friends) in Hydra's ConfigStore.

    Idempotent — safe to call from every entry point
    (``train.py``/``eval.py``/tests) before ``@hydra.main`` composes.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RootConfig)


# ─── Generic dict/target helpers ─────────────────────────────────────────────


def _to_dict(cfg: Any) -> dict[str, Any]:
    """Normalize a composed Hydra node / dataclass instance / plain dict to
    a plain ``dict`` — the one place that absorbs the DictConfig-vs-dataclass
    ambiguity between real Hydra runs and hand-built test configs."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"cannot coerce {type(cfg).__name__} to a plain dict")


# Params that carry an exact ``(num, den)`` rate: YAML ``[16000, 512]``
# composes as a plain Python ``list`` (``_to_dict`` already fully resolves
# any ``ListConfig``/``DictConfig`` to plain containers), but
# ``framespec.SeriesSpec.rate`` is compared with ``!=`` against a genuine
# *reduced* ``(num, den)`` tuple inferred from live data (``framespec.spec_of``
# reads ``GridIndex.sr_num``/``sr_den``, which ``tdseries`` always stores in
# lowest terms — e.g. ``(16000, 512)`` normalizes to ``(125, 4)``). A bare
# ``tuple(v)`` would fix the list-vs-tuple mismatch but not the
# reduced-vs-unreduced one, so every rate-bearing key is run through
# ``td.normalize_rate`` (the exact same reduction) — the one place every
# ``_target_``/task-factory call site funnels through.
_RATE_PARAM_KEYS = {"rate", "sr", "rps_rate", "frame_rate"}


def _coerce_rate_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (td.normalize_rate(tuple(v)) if k in _RATE_PARAM_KEYS and isinstance(v, list) else v)
        for k, v in params.items()
    }


def instantiate_target(target: str, params: dict[str, Any] | None = None) -> Any:
    """Call the callable named by ``target`` (a dotted path resolved via
    ``hydra.utils.get_method`` — works for a plain function, a class, or a
    classmethod) with ``params`` as kwargs (after :func:`_coerce_rate_params`).

    Deliberately not ``hydra.utils.instantiate`` on an OmegaConf node: that
    call re-wraps ``params`` into a fresh ``DictConfig``/``ListConfig`` tree,
    which would undo the plain-container normalization ``_to_dict`` already
    did on the whole config tree upstream (``build_dataset``/``build_losses``/
    ``build_metrics`` all call ``_to_dict`` before reaching here).
    """
    fn = hydra.utils.get_method(target)
    return fn(**_coerce_rate_params(dict(params or {})))


# ─── Dataset ──────────────────────────────────────────────────────────────────


def build_dataset(spec: Any) -> Dataset | IterableDataset:
    """Instantiate one :class:`DatasetSpec` (``cfg.data.train`` / ``.valid``)."""
    d = _to_dict(spec)
    target = d.get("_target_")
    if not target or target is MISSING:
        raise ValueError("dataset spec is missing _target_")
    return instantiate_target(target, d.get("params", {}))


# ─── Model + task + codec ─────────────────────────────────────────────────────


def instantiate_model(model_cfg: Any) -> torch.nn.Module:
    """Build the model for ``cfg.model`` — legacy path if ``model_type`` is
    set, else the ``_target_`` path (see module docstring)."""
    d = _to_dict(model_cfg)
    model_type = d.get("model_type")
    if model_type:
        legacy_config_path = d.get("legacy_config_path")
        if not legacy_config_path:
            raise ValueError("model.model_type is set but model.legacy_config_path is missing")
        from models.registry import build_legacy_model

        return build_legacy_model(model_type, legacy_config_path)

    target = d.get("_target_")
    if not target:
        raise ValueError(
            "model config needs either model_type+legacy_config_path (legacy) or "
            "_target_ (models.registry.build_model / any plain-kwargs constructor)"
        )
    return instantiate_target(target, d.get("params", {}))


def build_task_and_codec(model_cfg: Any) -> tuple[Task, Codec]:
    """Build the ``Task`` (spec) and matching ``Codec`` for ``cfg.model`` —
    both from the same ``task`` name + ``task_params`` dict."""
    d = _to_dict(model_cfg)
    task_name = d.get("task")
    if not task_name:
        raise ValueError("model config is missing 'task'")
    if task_name not in TASK_FACTORIES:
        raise ValueError(f"unknown task {task_name!r}; choose one of {sorted(TASK_FACTORIES)}")
    task_params = _coerce_rate_params(d.get("task_params", {}) or {})
    task = TASK_FACTORIES[task_name](**task_params)
    codec = build_codec(task_name, **task_params)
    return task, codec


# ─── Losses / metrics ─────────────────────────────────────────────────────────


def build_losses(loss_cfg: Any) -> CompositeLoss:
    d = _to_dict(loss_cfg)
    terms = d.get("terms") or []
    if not terms:
        raise ValueError("loss config has no terms")
    components: dict[str, LossTerm] = {}
    for term in terms:
        name = term["name"]
        weight = float(term.get("weight", 1.0))
        loss_obj = instantiate_target(term["_target_"], term.get("params", {}))
        components[name] = LossTerm(weight, loss_obj)
    return CompositeLoss(components)


def build_metrics(metrics_cfg: Any) -> MetricSuite:
    d = _to_dict(metrics_cfg)
    terms = d.get("terms") or []
    if not terms:
        raise ValueError("metrics config has no terms")
    metrics = {
        term["name"]: instantiate_target(term["_target_"], term.get("params", {})) for term in terms
    }
    return MetricSuite(metrics)
