"""Shared helpers for the Frame-adapter losses in this package.

Canonical entry names (docs/refactor-unified-framework.md § "Task typing:
FrameSpec"): ``"mixture"``, ``"enhanced"``, ``"rps"``, ``"rps_pred"``,
``"salience"``, ``"audio"``. That list is the *model* input/output contract
and deliberately omits the clean-speech ground truth, which is dataset-only
(never produced by a model). This package adopts the entry name
``"target"`` for it in the *target* Frame (the dataset batch), mirroring the
mixture/target naming already used throughout the pre-refactor root scripts
(the dataset builders).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import tdseries as td
import torch

from tasks.spec import FrameSpec, SeriesSpec

AUDIO_RATE: tuple[int, int] = (16000, 1)


def audio_dims(n_channels: int | None) -> tuple[str | None, ...]:
    """Batched audio dims: mono ``(batch, time)``; multi-mic ``(batch, mic, time)``.

    Mirrors ``tasks.task._audio_dims`` (kept private there) so loss specs
    line up with task specs without a cross-package import of a private name.
    """
    return ("batch", "time") if n_channels is None else ("batch", "mic", "time")


def audio_series_spec(
    n_channels: int | None = None, rate: tuple[int, int] = AUDIO_RATE
) -> SeriesSpec:
    """The ``SeriesSpec`` for a batched audio entry, mono or multi-mic."""
    return SeriesSpec(dims=audio_dims(n_channels), time="grid", rate=rate)


def rps_series_spec(rate: tuple[int, int] | None = None) -> SeriesSpec:
    """The ``SeriesSpec`` for a batched per-rotor RPS entry ``(batch, rotor, time)``."""
    return SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=rate)


def to_torch(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Coerce a Frame entry's payload (numpy or torch) to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def get_tensor(frame: td.Frame, key: str) -> torch.Tensor:
    """Pull the underlying tensor out of a Frame entry by canonical name, as torch.

    Raises the same ``KeyError`` a plain ``frame[key]`` would if the entry is
    absent — Frame adapters are expected to declare that entry in
    ``requires_pred``/``requires_target`` so spec validation catches this
    earlier, at compose time.
    """
    entry = frame[key]
    if isinstance(entry, td.Series):
        data = entry.data
        if data is None:
            raise ValueError(f"Frame entry {key!r} is an index-only Series (no data)")
        return to_torch(data)
    return to_torch(entry)


@runtime_checkable
class Loss(Protocol):
    """Structural protocol every Frame-adapter loss in this package satisfies.

    See docs/refactor-unified-framework.md § "Losses/metrics are classes
    declaring what they consume".
    """

    requires_pred: FrameSpec
    requires_target: FrameSpec

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor: ...


__all__ = [
    "AUDIO_RATE",
    "audio_dims",
    "audio_series_spec",
    "rps_series_spec",
    "get_tensor",
    "to_torch",
    "Loss",
]
