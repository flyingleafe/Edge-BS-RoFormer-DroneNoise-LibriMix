"""Utilities for measuring PyTorch DataLoader throughput.

The benchmark intentionally accepts an already-constructed loader/iterable.  That
keeps the public surface identical for naive file-backed datasets, optimized
cache-backed datasets, finite map-style datasets, and infinite IterableDatasets.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import torch


def _maybe_cuda_synchronize(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _leading_dim(x: Any) -> int | None:
    if isinstance(x, torch.Tensor) and x.ndim >= 1:
        return int(x.shape[0])
    return None


def infer_num_examples(batch: Any) -> int:
    """Infer dataset examples in a collated batch.

    For the project's RPS loaders, batches are usually ``(audio, rps)`` where
    ``audio`` has shape ``(B, T)`` or ``(B, C, T)``.  In both cases the dataset
    example count is ``B``.
    """
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.ndim >= 1 else 1
    if isinstance(batch, (tuple, list)) and batch:
        dim = _leading_dim(batch[0])
        if dim is not None:
            return dim
    if isinstance(batch, dict):
        for value in batch.values():
            dim = _leading_dim(value)
            if dim is not None:
                return dim
    return 1


def infer_num_audio_clips(batch: Any) -> int:
    """Infer effective audio clips in a collated batch.

    Multichannel RPS training flattens ``(B, C, T)`` audio into ``B*C`` effective
    clips before the model.  This metric makes fixed multichannel datasets and
    future online mixers comparable to model-side throughput.
    """
    audio = batch[0] if isinstance(batch, (tuple, list)) and batch else batch
    if isinstance(audio, dict) and "audio" in audio:
        audio = audio["audio"]
    if not isinstance(audio, torch.Tensor) or audio.ndim == 0:
        return infer_num_examples(batch)
    if audio.ndim >= 3:
        return int(audio.shape[0] * audio.shape[1])
    if audio.ndim >= 1:
        return int(audio.shape[0])
    return 1


def batch_shape_summary(batch: Any) -> Any:
    """Return a lightweight shape summary for diagnostics."""
    if isinstance(batch, torch.Tensor):
        return {"shape": tuple(batch.shape), "dtype": str(batch.dtype), "device": str(batch.device)}
    if isinstance(batch, (tuple, list)):
        return [batch_shape_summary(x) for x in batch]
    if isinstance(batch, dict):
        return {k: batch_shape_summary(v) for k, v in batch.items()}
    return type(batch).__name__


def benchmark_dataloader(
    loader: Iterable[Any],
    *,
    seconds: float = 10.0,
    max_batches: int | None = None,
    warmup_batches: int = 0,
    sync_cuda: bool = False,
) -> dict[str, Any]:
    """Iterate over ``loader`` and measure throughput.

    Args:
        loader: Any iterable yielding collated batches, typically a PyTorch
            ``DataLoader``.  It may be finite or infinite.
        seconds: Target measured duration.  Iteration stops after this many
            seconds unless the loader is exhausted first.  The current batch is
            always allowed to finish.
        max_batches: Optional hard cap on measured batches.  Useful for tests or
            for benchmarking finite datasets without consuming all of them.
        warmup_batches: Batches to consume before starting the timer.  Warmup is
            not included in throughput counts.
        sync_cuda: If ``True``, call ``torch.cuda.synchronize()`` around timing
            points.  Usually unnecessary for CPU-only DataLoader benchmarks, but
            useful if a caller adds device transfer/transforms to the iterable.

    Returns:
        Plain dict with counts, elapsed seconds, throughput metrics, exhaustion
        status, and first-batch shape diagnostics.
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be > 0 when provided")
    if warmup_batches < 0:
        raise ValueError("warmup_batches must be >= 0")

    it = iter(loader)

    exhausted = False
    for _ in range(warmup_batches):
        try:
            next(it)
        except StopIteration:
            exhausted = True
            break

    batches = 0
    examples = 0
    audio_clips = 0
    first_batch_summary = None

    _maybe_cuda_synchronize(sync_cuda)
    start = time.perf_counter()
    end = start

    if not exhausted:
        while True:
            if max_batches is not None and batches >= max_batches:
                break
            if batches > 0 and (time.perf_counter() - start) >= seconds:
                break
            try:
                batch = next(it)
            except StopIteration:
                exhausted = True
                break

            if first_batch_summary is None:
                first_batch_summary = batch_shape_summary(batch)
            batches += 1
            examples += infer_num_examples(batch)
            audio_clips += infer_num_audio_clips(batch)
            _maybe_cuda_synchronize(sync_cuda)
            end = time.perf_counter()

    elapsed = max(end - start, 0.0)
    denom = elapsed if elapsed > 0 else float("nan")
    return {
        "elapsed_s": elapsed,
        "batches": batches,
        "examples": examples,
        "audio_clips": audio_clips,
        "batches_per_s": batches / denom,
        "examples_per_s": examples / denom,
        "audio_clips_per_s": audio_clips / denom,
        "exhausted": exhausted,
        "warmup_batches": warmup_batches,
        "seconds_target": seconds,
        "max_batches": max_batches,
        "first_batch": first_batch_summary,
    }


def format_benchmark_result(result: dict[str, Any]) -> str:
    """Human-readable one-line-plus-shapes summary."""
    status = "exhausted" if result.get("exhausted") else "time/cap reached"
    return (
        f"{result['batches']} batches, {result['examples']} examples, "
        f"{result['audio_clips']} audio clips in {result['elapsed_s']:.3f}s "
        f"({result['batches_per_s']:.2f} batch/s, "
        f"{result['examples_per_s']:.1f} example/s, "
        f"{result['audio_clips_per_s']:.1f} audio-clip/s; {status})\n"
        f"first_batch={result.get('first_batch')}"
    )
