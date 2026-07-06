"""Structural specs for :class:`tdseries.Frame` pipelines.

A *task* is the function type of a model: which entries (and what kind of
series) it consumes and produces. ``FrameSpec`` describes the shape of a
Frame; :func:`check_subsumes` verifies that what one stage provides covers
what the next stage requires — before any GPU time is spent.

Conventions (see docs/refactor-unified-framework.md):

- dataset adapters declare the per-sample spec of the Frames they emit;
- models declare batched input/output specs (leading ``"batch"`` dim) —
  use :func:`without_batch` to compare against per-sample dataset specs;
- losses/metrics declare ``requires_pred`` / ``requires_target`` specs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import tdseries as td

TimeKind = Literal["grid", "stamps", "spans"]

_TIME_KIND_OF_INDEX: dict[type, TimeKind] = {
    td.GridIndex: "grid",
    td.StampIndex: "stamps",
    td.SpanIndex: "spans",
}


@dataclass(frozen=True)
class ScalarSpec:
    """A plain (non-Series) invariant entry, e.g. a metadata scalar."""


SCALAR = ScalarSpec()


@dataclass(frozen=True)
class SeriesSpec:
    """Requirements on a single :class:`tdseries.Series` entry.

    ``dims`` are ordered dim names (``None`` = anonymous axis). ``time`` is
    the required time-index kind (``None`` = atemporal). ``rate`` is an
    exact ``(num, den)`` sample-rate constraint for grid series (``None`` =
    any rate). ``dtype`` is ``None`` (any), a numpy dtype name
    (``"float32"``), or a kind (``"floating"`` / ``"integer"``).
    """

    dims: tuple[str | None, ...]
    time: TimeKind | None = "grid"
    rate: tuple[int, int] | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        has_time_dim = "time" in self.dims
        if has_time_dim != (self.time is not None):
            raise ValueError(
                f"SeriesSpec dims {self.dims} inconsistent with time={self.time!r}: "
                "a 'time' dim requires a time kind and vice versa"
            )
        if self.rate is not None and self.time != "grid":
            raise ValueError("rate constraints only apply to time='grid' series")


@dataclass(frozen=True)
class FrameSpec:
    """Requirements on a :class:`tdseries.Frame`: named entries, each a
    Series / nested Frame / scalar. Extra entries in the provided Frame are
    always allowed; ``optional`` names are checked only when present."""

    entries: Mapping[str, SeriesSpec | FrameSpec | ScalarSpec] = field(default_factory=dict)
    optional: frozenset[str] = frozenset()


EntrySpec = SeriesSpec | FrameSpec | ScalarSpec


def spec_of(frame: td.Frame) -> FrameSpec:
    """Infer the exact spec of a live Frame (used by the one-batch smoke
    test and for error messages)."""
    entries: dict[str, EntrySpec] = {}
    for name in frame:
        entries[name] = _spec_of_entry(frame[name])
    return FrameSpec(entries)


def _spec_of_entry(value: object) -> EntrySpec:
    if isinstance(value, td.Frame):
        return spec_of(value)
    if isinstance(value, td.Series):
        tk: TimeKind | None = None
        if value.has_time:
            tk = _TIME_KIND_OF_INDEX.get(type(value.tindex))
        rate: tuple[int, int] | None = None
        if tk == "grid":
            idx = value.tindex
            if isinstance(idx, td.GridIndex):
                rate = (idx.sr_num, idx.sr_den)
        return SeriesSpec(
            dims=tuple(value.dims),
            time=tk,
            rate=rate,
            dtype=_dtype_name(value.data),
        )
    return SCALAR


def _dtype_name(data: object) -> str | None:
    if data is None:
        return None
    dtype = getattr(data, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _dtype_matches(provided: str | None, required: str | None) -> bool:
    if required is None or provided is None:
        return True
    if required in ("floating", "integer"):
        kind = "f" if required == "floating" else "iu"
        try:
            return np.dtype(provided).kind in kind
        except TypeError:
            return provided.startswith("float" if required == "floating" else "int")
    return provided == required


def _dims_match(provided: tuple[str | None, ...], required: tuple[str | None, ...]) -> bool:
    if len(provided) != len(required):
        return False
    return all(r is None or p == r for p, r in zip(provided, required, strict=True))


def _check_series(path: str, provided: SeriesSpec, required: SeriesSpec) -> list[str]:
    problems: list[str] = []
    if not _dims_match(provided.dims, required.dims):
        problems.append(f"{path}: dims {provided.dims} do not match required {required.dims}")
    if provided.time != required.time:
        problems.append(
            f"{path}: time kind {provided.time!r} does not match required {required.time!r}"
        )
    if required.rate is not None and provided.rate != required.rate:
        problems.append(f"{path}: rate {provided.rate} != required {required.rate}")
    if not _dtype_matches(provided.dtype, required.dtype):
        problems.append(f"{path}: dtype {provided.dtype!r} incompatible with {required.dtype!r}")
    return problems


def check_subsumes(provided: FrameSpec, required: FrameSpec, *, path: str = "") -> list[str]:
    """Return human-readable mismatch messages ([] == ``provided`` covers
    ``required``). Extra provided entries are fine; required-optional
    entries are checked only if the provider has them."""
    problems: list[str] = []
    for name, req in required.entries.items():
        child_path = f"{path}.{name}" if path else name
        if name not in provided.entries:
            if name in required.optional:
                continue
            problems.append(f"{child_path}: required entry is missing")
            continue
        if name in provided.optional and name not in required.optional:
            problems.append(f"{child_path}: provider marks this entry optional but it is required")
        prov = provided.entries[name]
        problems.extend(_check_entry(child_path, prov, req))
    return problems


def _check_entry(path: str, prov: EntrySpec, req: EntrySpec) -> list[str]:
    match req:
        case ScalarSpec():
            return []  # any present entry satisfies a scalar requirement
        case SeriesSpec():
            if not isinstance(prov, SeriesSpec):
                return [f"{path}: expected a Series, provider has {type(prov).__name__}"]
            return _check_series(path, prov, req)
        case FrameSpec():
            if not isinstance(prov, FrameSpec):
                return [f"{path}: expected a nested Frame, provider has {type(prov).__name__}"]
            return check_subsumes(prov, req, path=path)


def without_batch(spec: FrameSpec) -> FrameSpec:
    """Strip the leading ``"batch"`` dim from every SeriesSpec — turns a
    batched model spec into the per-sample spec a dataset must provide."""
    entries: dict[str, EntrySpec] = {}
    for name, entry in spec.entries.items():
        match entry:
            case SeriesSpec(dims=("batch", *rest)):
                entries[name] = SeriesSpec(
                    dims=tuple(rest), time=entry.time, rate=entry.rate, dtype=entry.dtype
                )
            case FrameSpec():
                entries[name] = without_batch(entry)
            case _:
                entries[name] = entry
    return FrameSpec(entries, spec.optional)


def merge_specs(*specs: FrameSpec) -> FrameSpec:
    """Union of provider specs (e.g. model output ∪ dataset batch): later
    specs win on name clashes; an entry is optional only if optional in
    every spec that has it."""
    entries: dict[str, EntrySpec] = {}
    optional: set[str] = set()
    for spec in specs:
        for name, entry in spec.entries.items():
            entries[name] = entry
            if name in spec.optional:
                if name not in entries or name in optional:
                    optional.add(name)
            else:
                optional.discard(name)
    return FrameSpec(entries, frozenset(optional))
