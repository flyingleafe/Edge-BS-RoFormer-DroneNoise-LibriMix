"""Batching for :class:`tdseries.Frame` samples.

A dataset adapter yields one ``td.Frame`` per sample (no ``"batch"`` dim —
see ``tasks.spec.without_batch``). :func:`collate_frames` stacks a list of
*equal-shape* per-sample Frames into one batched Frame, matching a task's
batched ``FrameSpec`` (leading ``"batch"`` dim on every Series entry).

Batching design (see docs/refactor-unified-framework.md § "Task typing:
FrameSpec", "Batching: ``collate_frames``"):

- Every ``td.Series`` entry (temporal or invariant) is stacked along a new
  leading axis via ``np.stack``/``torch.stack`` (whichever the payload already
  uses); its ``dims`` gain a leading ``"batch"``. A temporal entry's
  ``GridIndex``/``StampIndex``/``SpanIndex`` is reused verbatim from the first
  sample (the "equal-shape" contract requires every sample to share it — this
  is checked via ``TimeIndex.equal``, not just shape, so silently
  misaligned rates/starts fail loudly instead of scrambling the batch).
- A nested (non-``"meta"``) ``td.Frame`` entry recurses through
  :func:`collate_frames`.
- The nested invariant ``"meta"`` Frame (see ``data_processing.frames``) is
  special-cased: it holds heterogeneous per-recording scalars (numbers,
  strings, ``None``), which are not all ``Series``-representable. Per key,
  if every sample's value is present and numeric, the batched entry becomes a
  ``td.wrap``-ed ``(batch,)`` array (so e.g. ``batched["meta"]["input_snr"]``
  is directly usable for ``MetricSuite`` per-SNR grouping / wandb logging);
  otherwise it stays a plain Python list of length ``batch`` (preserves
  strings / missing values without forcing a lossy cast).
- Any other plain (non-Series, non-Frame) entry follows the same numeric ->
  array / else -> list rule as ``"meta"`` values.

:func:`slice_sample` is the inverse, used by the training loop / eval to pull
one sample back out of a batched (prediction or target) Frame for
:class:`~metrics.suite.MetricSuite` (which scores one ``(pred, target)`` pair
at a time): it drops the ``"batch"`` axis via ``Frame.slice["batch", i]``
(which recurses into every ``Series`` leaf, including nested Frames — so a
numeric ``"meta"`` array is *already* correctly reduced to a 0-d Series by
that call) and then re-materializes ``"meta"`` back to plain per-sample
scalars: index list-valued keys at ``i``, and unwrap the now-0-d numeric
Series (via ``.item()``) rather than leaving a ``Series`` object sitting
where a plain float/str is expected (``get_meta``/``MetricSuite`` grouping
compares these by value, not by the ``Series``'s identity-based ``__hash__``).

:func:`frame_collate` is the ``torch.utils.data.DataLoader``-facing wrapper:
it calls :func:`collate_frames` and then coerces every Series leaf's payload
to a ``torch.Tensor`` (:meth:`tdseries.Frame.map_data`), since a raw
``collate_frames`` result may still hold numpy arrays when the dataset
yielded numpy-backed Frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import tdseries as td
import torch

__all__ = ["collate_frames", "frame_collate", "slice_sample", "batch_size"]

_NUMERIC_TYPES = (int, float, bool, np.integer, np.floating, np.bool_)


def _stack_arrays(arrays: Sequence[Any]) -> Any:
    first = arrays[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(list(arrays), dim=0)
    return np.stack([np.asarray(a) for a in arrays], axis=0)


def _is_all_numeric(values: Sequence[Any]) -> bool:
    return all(v is not None and isinstance(v, _NUMERIC_TYPES) for v in values)


def _collate_scalars(values: Sequence[Any]) -> Any:
    """Numeric -> ``(batch,)`` array entry; else a plain per-sample list."""
    if _is_all_numeric(values):
        return td.wrap(np.asarray(values), dims=("batch",))
    return list(values)


def _collate_meta(metas: Sequence[td.Frame | None]) -> td.Frame:
    # `metas` are already the extracted "meta" sub-Frames (collate_frames
    # passes `frame["meta"]`, not the parent) — data_processing.frames.meta_dict
    # expects the *parent* Frame (it does its own `frame["meta"]` lookup), so
    # it is not reusable here; pull each sub-Frame's own entries directly.
    dicts = [{k: m[k] for k in m} if m is not None else {} for m in metas]
    keys: set[str] = set()
    for d in dicts:
        keys.update(d)
    entries: dict[str, Any] = {}
    for key in sorted(keys):
        entries[key] = _collate_scalars([d.get(key) for d in dicts])
    return td.Frame(entries)


def _collate_series(key: str, values: Sequence[td.Series]) -> td.Series:
    dims = values[0].dims
    for v in values[1:]:
        if v.dims != dims:
            raise ValueError(
                f"collate_frames: entry {key!r} has inconsistent dims across samples "
                f"({dims} vs {v.dims}) — frames must be equal-shape"
            )
    data_list = [v.data for v in values]
    if any(d is None for d in data_list):
        raise ValueError(
            f"collate_frames: entry {key!r} is an index-only Series (data=None) in at "
            "least one sample; cannot stack"
        )
    stacked = _stack_arrays(data_list)
    new_dims = ("batch", *dims)
    if "time" in dims:
        ti0 = values[0].tindex
        for v in values[1:]:
            if not ti0.equal(v.tindex):
                raise ValueError(
                    f"collate_frames: entry {key!r} time index differs across samples "
                    "(rate/start/length mismatch) — frames must be equal-shape"
                )
        return td.Series(stacked, new_dims, {"time": ti0})
    return td.wrap(stacked, dims=new_dims)


def _collate_entry(key: str, values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, td.Series):
        return _collate_series(key, values)
    if isinstance(first, td.Frame):
        return collate_frames(list(values))
    return _collate_scalars(values)


def collate_frames(frames: Sequence[td.Frame]) -> td.Frame:
    """Stack ``frames`` (equal-shape, per-sample) into one batched Frame.

    Every entry gains a leading ``"batch"`` dim (see module docstring for the
    per-entry-kind rules). Raises ``ValueError`` if the frames don't share an
    identical entry-key set, or if a shared entry's shape/dims/time-index
    disagree between samples.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("collate_frames requires at least one frame")
    keys = list(frames[0].keys())
    key_set = set(keys)
    for f in frames[1:]:
        if set(f.keys()) != key_set:
            raise ValueError(
                "collate_frames requires identical entry keys across samples; got "
                f"{key_set} vs {set(f.keys())}"
            )
    entries: dict[str, Any] = {}
    for key in keys:
        values = [f[key] for f in frames]
        if key == "meta":
            entries[key] = _collate_meta(values)
        else:
            entries[key] = _collate_entry(key, values)
    return td.Frame(entries)


def frame_collate(frames: Sequence[td.Frame]) -> td.Frame:
    """``collate_fn`` for a ``torch.utils.data.DataLoader`` over Frame samples.

    Stacks via :func:`collate_frames`, then coerces every Series leaf's data
    to a ``torch.Tensor`` (numpy-backed dataset samples are the common case;
    torch-backed ones pass through ``torch.as_tensor`` as a no-op).
    """
    batched = collate_frames(frames)
    return batched.map_data(torch.as_tensor)


def batch_size(frame: td.Frame) -> int:
    """Size of the ``"batch"`` axis, read off the first top-level entry that
    carries one (a batched Frame always has at least one)."""
    for key in frame:
        entry = frame[key]
        if isinstance(entry, td.Series) and "batch" in entry.dims:
            return entry.dim_size("batch")
    raise ValueError("frame has no entry carrying a 'batch' dim")


def slice_sample(frame: td.Frame, i: int) -> td.Frame:
    """Pull sample ``i`` back out of a batched Frame — the inverse of
    :func:`collate_frames` (see module docstring)."""
    sliced = frame.slice["batch", i]
    if "meta" in sliced:
        meta = sliced["meta"]
        fixed: dict[str, Any] = {}
        for k in meta:
            v = meta[k]
            if isinstance(v, list):
                fixed[k] = v[i]
            elif isinstance(v, td.Series):
                data = v.data
                fixed[k] = data.item() if hasattr(data, "item") else data
            else:
                fixed[k] = v
        sliced = sliced.with_entry("meta", td.Frame(fixed))
    return sliced
