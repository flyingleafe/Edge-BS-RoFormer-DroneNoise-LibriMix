"""Pre-run spec validation (docs/refactor-unified-framework.md § "Pre-run
validation").

Four checks, run once before any GPU time is spent (and standalone via
``python train.py ... validate_only=true``):

1. the train dataset's per-sample spec covers the model's (batch-stripped)
   input spec;
2. the model's batched output spec, unioned with the dataset's batched spec,
   covers every loss/metric's ``requires_pred`` *and* ``requires_target``;
3. the optimizer's scheduler ``monitor`` metric exists (a metric-suite name, or
   the literals ``"loss"`` = train loss / ``"val_loss"`` = the objective on
   held-out data);
4. a one-batch CPU forward pass actually runs, and its output spec covers
   the task's declared output spec.

:func:`validate_config` never raises — a failure to even build a component
(dataset/model/loss/metrics) becomes a problem string like any spec mismatch,
so a single call always returns the full list of problems it could find
(``[]`` means valid).
"""

from __future__ import annotations

from typing import Any

import torch

from data_processing.collate import collate_frames
from tasks.spec import FrameSpec, check_subsumes, merge_specs, spec_of, without_batch
from training.config import (
    build_dataset,
    build_losses,
    build_metrics,
    build_task_and_codec,
    instantiate_model,
)

__all__ = ["validate_config"]

# Enough samples to exercise collate_frames' stacking logic without paying
# for a real batch; 1 sample can't distinguish "stacks correctly" from
# "coincidentally works unbatched".
SMOKE_SAMPLES = 2


def _draw_samples(dataset: Any, n: int) -> list[Any]:
    """Pull up to ``n`` per-sample Frames from a map-style or iterable dataset."""
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        n = min(n, len(dataset))
        if n <= 0:
            return []
        return [dataset[i] for i in range(n)]
    it = iter(dataset)
    samples = []
    for _ in range(n):
        try:
            samples.append(next(it))
        except StopIteration:
            break
    return samples


def validate_config(cfg: Any) -> list[str]:  # noqa: C901 - linear checklist, not complex branching
    """Run all four pre-run checks against a composed root config.

    Returns a list of human-readable problem strings; ``[]`` means the
    pipeline is valid and safe to train/eval.
    """
    problems: list[str] = []

    try:
        train_ds = build_dataset(cfg.data.train)
    except Exception as exc:
        return [f"failed to build data.train dataset: {exc!r}"]

    try:
        task, codec = build_task_and_codec(cfg.model)
    except Exception as exc:
        return [f"failed to build model task/codec: {exc!r}"]

    try:
        samples = _draw_samples(train_ds, SMOKE_SAMPLES)
    except Exception as exc:
        return [f"failed to draw a sample from data.train: {exc!r}"]
    if not samples:
        return ["data.train yielded no samples"]

    # ── 1. dataset spec ⊇ without_batch(model input) ──────────────────────
    sample_spec = spec_of(samples[0])
    required_input = without_batch(task.input_spec)
    problems += [f"dataset input: {p}" for p in check_subsumes(sample_spec, required_input)]

    try:
        batch = collate_frames(samples if len(samples) > 1 else samples * 2)
    except Exception as exc:
        problems.append(f"failed to collate a batch from data.train samples: {exc!r}")
        batch = None

    loss = None
    try:
        loss = build_losses(cfg.loss)
    except Exception as exc:
        problems.append(f"failed to build loss: {exc!r}")

    metric_suite = None
    try:
        metric_suite = build_metrics(cfg.metrics)
    except Exception as exc:
        problems.append(f"failed to build metrics: {exc!r}")

    # ── 2. model output_spec ∪ dataset.spec ⊇ each loss/metric requirement ─
    if batch is not None:
        batch_spec = spec_of(batch)
        provided: FrameSpec = merge_specs(task.output_spec, batch_spec)
        if loss is not None:
            problems += [
                f"loss requires_pred: {p}" for p in check_subsumes(provided, loss.requires_pred)
            ]
            problems += [
                f"loss requires_target: {p}" for p in check_subsumes(provided, loss.requires_target)
            ]
        if metric_suite is not None:
            for name, metric in metric_suite.metrics.items():
                problems += [
                    f"metric {name!r} requires_pred: {p}"
                    for p in check_subsumes(provided, metric.requires_pred)
                ]
                problems += [
                    f"metric {name!r} requires_target: {p}"
                    for p in check_subsumes(provided, metric.requires_target)
                ]

    # ── 3. scheduler monitor metric exists ─────────────────────────────────
    try:
        monitor = cfg.optim.monitor
        metric_names = set(metric_suite.metrics) if metric_suite is not None else set()
        # "val_loss" is the objective on held-out data — the right monitor
        # whenever the metric suite is not aligned with the loss being trained
        # (see training.loop.run_training).
        if monitor not in ("loss", "val_loss") and monitor not in metric_names:
            problems.append(
                f"optim.monitor {monitor!r} is neither 'loss'/'val_loss' nor a "
                f"metrics.terms name {sorted(metric_names)}"
            )
    except Exception as exc:
        problems.append(f"failed to read optim.monitor: {exc!r}")

    # ── 4. one-batch CPU smoke test ─────────────────────────────────────────
    if batch is not None:
        try:
            model = instantiate_model(cfg.model)
            model.eval()
            inputs = codec.to_inputs(batch)
            with torch.no_grad():
                outputs = codec.call_model(model, inputs)
            out_frame = codec.to_frame(outputs, batch)
            problems += [
                f"smoke-test output: {p}"
                for p in check_subsumes(spec_of(out_frame), task.output_spec)
            ]
        except Exception as exc:
            problems.append(f"one-batch CPU smoke test failed: {exc!r}")

    return problems
