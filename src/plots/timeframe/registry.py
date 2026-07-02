"""Renderer registry for timeframe tracks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import matplotlib.axes

Renderer = Callable[[Any, "TrackContext"], "RenderedTrack"]


@dataclass
class TrackContext:
    """Context passed to every renderer."""

    ax: matplotlib.axes.Axes
    name: str
    t_start: float
    t_end: float
    style: dict[str, Any]


@dataclass
class RenderedTrack:
    """Result returned by a renderer."""

    ax: matplotlib.axes.Axes
    legend_handles: list[Any]


_renderers: dict[str, Renderer] = {}


def register_renderer(key: str, fn: Renderer) -> None:
    """Register a renderer under ``key``."""
    if key in _renderers:
        raise ValueError(f"Duplicate renderer {key!r}")
    _renderers[key] = fn


def get_renderer(key: str) -> Renderer:
    """Return the renderer registered under ``key``."""
    try:
        return _renderers[key]
    except KeyError:
        known = sorted(_renderers)
        raise ValueError(f"Unknown renderer {key!r}. Known: {known}") from None


def list_renderers() -> list[str]:
    """Return all registered renderer keys."""
    return sorted(_renderers)


def resolve_renderer_key(
    name: str,
    series: Any,
    fallback_types: Mapping[type, str] | None = None,
) -> str:
    """Pick a renderer key for ``series`` (a TimeSeries subclass).

    Resolution order:
        1. ``series.tags.get("plot.renderer")`` if present.
        2. Exact track ``name`` if registered.
        3. The concrete class of ``series`` if registered.
        4. A registered base-class fallback from ``fallback_types``.
    """
    fallback_types = fallback_types or {}
    tags = getattr(series, "tags", {})
    explicit = tags.get("plot.renderer")
    if explicit is not None:
        return str(explicit)
    if name in _renderers:
        return name
    for cls in type(series).__mro__:
        if cls.__name__ in _renderers:
            return cls.__name__
        if cls in fallback_types and fallback_types[cls] in _renderers:
            return fallback_types[cls]
    known = sorted(_renderers)
    raise ValueError(
        f"No renderer for track {name!r} of type {type(series).__name__}. Known: {known}"
    )


def resolve_title(name: str, series: Any, channel: int | None = None) -> str | None:
    """Resolve subplot title for a track.

    Order: explicit ``plot.title`` tag, then track name, then channel suffix.
    ``plot.title`` can be ``None`` to suppress the title.
    """
    tags = getattr(series, "tags", {})
    if "plot.title" in tags:
        return tags["plot.title"]
    title = name
    if channel is not None:
        title = f"{title} — ch{channel}"
    return title
