"""Shared helpers for the Frame-adapter metrics in this package.

Canonical entry names (docs/refactor-unified-framework.md § "Task typing:
FrameSpec"): ``"mixture"``, ``"enhanced"``, ``"rps"``, ``"rps_pred"``,
``"salience"``, ``"audio"``. As in ``losses._common``, the clean-speech
ground truth (dataset-only, never produced by a model) is addressed by the
entry name ``"target"`` in the *target* Frame.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import tdseries as td
import torch

from framespec import FrameSpec, SeriesSpec

AUDIO_RATE: tuple[int, int] = (16000, 1)


def audio_dims(n_channels: int | None) -> tuple[str | None, ...]:
    """Batched audio dims: mono ``(batch, time)``; multi-mic ``(batch, mic, time)``."""
    return ("batch", "time") if n_channels is None else ("batch", "mic", "time")


def audio_series_spec(
    n_channels: int | None = None, rate: tuple[int, int] = AUDIO_RATE
) -> SeriesSpec:
    """The ``SeriesSpec`` for a batched audio entry, mono or multi-mic."""
    return SeriesSpec(dims=audio_dims(n_channels), time="grid", rate=rate)


def rps_series_spec(rate: tuple[int, int] | None = None) -> SeriesSpec:
    """The ``SeriesSpec`` for a batched per-rotor RPS entry ``(batch, rotor, time)``."""
    return SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=rate)


def to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Coerce a Frame entry's payload (numpy or torch) to a detached numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_array(frame: td.Frame, key: str) -> np.ndarray:
    """Pull the underlying data out of a Frame entry by canonical name, as numpy.

    Metrics are evaluation-time (no autograd needed), so unlike
    ``losses._common.get_tensor`` this always returns numpy.
    """
    entry = frame[key]
    if isinstance(entry, td.Series):
        data = entry.data
        if data is None:
            raise ValueError(f"Frame entry {key!r} is an index-only Series (no data)")
        return to_numpy(data)
    return to_numpy(entry)


@runtime_checkable
class Metric(Protocol):
    """Structural protocol every Frame-adapter metric in this package satisfies.

    Mirrors ``losses._common.Loss`` but returns a plain ``float`` (no
    autograd) — see docs/refactor-unified-framework.md § "Metrics".
    """

    requires_pred: FrameSpec
    requires_target: FrameSpec

    def __call__(self, pred: td.Frame, target: td.Frame) -> float: ...


__all__ = [
    "AUDIO_RATE",
    "audio_dims",
    "audio_series_spec",
    "rps_series_spec",
    "to_numpy",
    "get_array",
    "Metric",
]
