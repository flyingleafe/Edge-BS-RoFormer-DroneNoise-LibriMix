"""Performance metrics: real-time factor, FLOPs, peak GPU memory.

Ported from the inline measurement in ``final_valid.py::process_audio_files``
(RTF = inference wall-clock / audio duration; ``thop.profile`` for FLOPs;
``torch.cuda.max_memory_allocated`` for peak memory).

Unlike the separation/RPS metrics, these are not naturally a pure function of
a ``(pred, target)`` Frame pair — they depend on *how* the prediction was
produced (wall-clock time, the model + its input shapes), not just the
resulting tensors. This module therefore centres on measurement utilities
(:func:`measure_inference`, :func:`measure_flops`) that a training/eval loop
calls around the model's forward pass; :class:`RTFMetric` is a thin Frame
adapter for the common case where that loop has already stashed the timing
into Frame metadata (``data_processing.frames.with_meta``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import contextmanager

import tdseries as td
import torch
from torch import nn

from data_processing.frames import get_meta
from tasks.spec import SCALAR, FrameSpec

# ─── Pure measurement utilities ──────────────────────────────────────────────


def compute_rtf(elapsed_s: float, audio_duration_s: float) -> float:
    """Real-time factor: inference wall-clock time / audio duration.

    <1 means faster than real time. ``audio_duration_s`` of 0 raises
    ``ZeroDivisionError`` (matches the original's unguarded division).
    """
    return elapsed_s / audio_duration_s


@contextmanager
def measure_inference(device: torch.device | str = "cpu"):
    """Context manager measuring wall-clock time and peak CUDA memory.

    Yields a dict populated (in place) after the block exits, with keys
    ``elapsed_s`` and ``peak_mem_mb`` (``peak_mem_mb`` is 0.0 on non-CUDA
    devices).

    Example::

        with measure_inference(device) as stats:
            out = model(x)
        rtf = compute_rtf(stats["elapsed_s"], audio_duration_s)
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    stats: dict[str, float] = {}
    try:
        yield stats
    finally:
        stats["elapsed_s"] = time.perf_counter() - start
        stats["peak_mem_mb"] = (
            torch.cuda.max_memory_allocated(dev) / 1e6 if dev.type == "cuda" else 0.0
        )


def peak_gpu_mem_mb(device: torch.device | str) -> float:
    """Peak CUDA memory allocated on ``device``, in MB (0.0 if not CUDA)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(dev) / 1e6


def measure_flops(model: nn.Module, inputs: Iterable[torch.Tensor]) -> float:
    """Model FLOPs (in GFLOPs) via ``thop.profile``. 0.0 if profiling fails
    (e.g. an unsupported op) — same fallback as the original.

    ``thop`` is a profiling-only dependency, imported lazily here so modules
    that don't need FLOPs counting don't pay its import cost.
    """
    from thop import profile

    try:
        model_profile = profile(model, inputs=tuple(inputs), verbose=False)
        return float(model_profile[0]) / 1e9
    except Exception:
        return 0.0


# ─── Frame adapter ────────────────────────────────────────────────────────────


class RTFMetric:
    """Frame adapter around :func:`compute_rtf`.

    Expects the timing/duration to already be recorded as Frame metadata
    (e.g. by a training/eval loop wrapping the model call in
    :func:`measure_inference` and writing the result via
    ``data_processing.frames.with_meta``):

    - ``pred["meta"][elapsed_key]`` (default ``"elapsed_s"``): inference
      wall-clock time for this sample.
    - ``target["meta"][duration_key]`` (default ``"audio_duration_s"``): the
      reference audio's duration in seconds.
    """

    def __init__(
        self,
        *,
        elapsed_key: str = "elapsed_s",
        duration_key: str = "audio_duration_s",
    ) -> None:
        self.elapsed_key = elapsed_key
        self.duration_key = duration_key
        self.requires_pred = FrameSpec({"meta": FrameSpec({elapsed_key: SCALAR})})
        self.requires_target = FrameSpec({"meta": FrameSpec({duration_key: SCALAR})})

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        elapsed_s = get_meta(pred, self.elapsed_key)
        duration_s = get_meta(target, self.duration_key)
        if elapsed_s is None or duration_s is None:
            raise ValueError(
                f"RTFMetric requires pred.meta[{self.elapsed_key!r}] and "
                f"target.meta[{self.duration_key!r}] to be populated by the eval loop"
            )
        return compute_rtf(float(elapsed_s), float(duration_s))


__all__ = [
    "compute_rtf",
    "measure_inference",
    "peak_gpu_mem_mb",
    "measure_flops",
    "RTFMetric",
]
