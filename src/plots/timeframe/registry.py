"""Renderer registry for timeframe tracks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import matplotlib.axes
import tdseries as td

Renderer = Callable[[td.Series, "TrackContext"], "RenderedTrack"]


@dataclass(frozen=True)
class PlotTrack:
    """A plottable track: a ``tdseries.Series`` plus plot-only presentation metadata.

    Series-level plot tags (the old ``plot.renderer``/``plot.freqs``/``plot.rps_pred``
    tags on ``utils.data`` series) do not live on ``tdseries.Series`` — that library
    owns no plotting concept. This wrapper is where the plots package keeps them
    instead, so a ``Frame`` stays pure data:

    * ``series``   — the ``tdseries.Series`` to render.
    * ``renderer`` — explicit renderer-registry key; overrides dispatch-by-index
      when set (``None`` = dispatch by the series' time-index type, see
      :func:`resolve_renderer_key`).
    * ``hints``    — renderer-specific extras, e.g. ``"freqs"``/``"rps_pred"``
      (salience renderer), ``"freq_max_hz"`` (spectrogram renderer), ``"title"``
      (subplot title override, ``None`` suppresses it), ``"kind"`` (nudges
      dispatch toward ``"spectrogram"``/``"salience"`` for an unlabeled 2-D grid).
    """

    series: td.Series
    renderer: str | None = None
    hints: dict[str, Any] = field(default_factory=dict)


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


def resolve_renderer_key(track: PlotTrack) -> str:
    """Pick a renderer key for ``track``.

    Resolution order:
        1. ``track.renderer`` if explicitly set (e.g. set by
           :func:`~plots.timeframe.renderers.make_spectrogram_series` /
           :func:`~plots.timeframe.renderers.make_salience_series`).
        2. The series' time-index type: ``StampIndex`` -> ``"rps"``,
           ``SpanIndex`` -> ``"spans"``. For ``GridIndex``:
           ``track.hints["kind"]`` of ``"spectrogram"``/``"salience"`` picks
           the matching heatmap renderer; otherwise a single extra dim named
           ``"mic"``/``"channel"`` (or none at all — mono) dispatches to
           ``"audio"`` (channel-aware waveform; ``plot_timeframe`` fans this
           out into one row per channel), and any other extra dim dispatches
           to the generic ``"waveform"`` fallback (single row, one overlaid
           line per row of the non-time axis).
    """
    if track.renderer is not None:
        return track.renderer
    series = track.series
    try:
        tindex = series.tindex
    except ValueError:
        raise ValueError(
            f"Cannot resolve a renderer for an atemporal series (dims={series.dims})"
        ) from None
    if isinstance(tindex, td.StampIndex):
        return "rps"
    if isinstance(tindex, td.SpanIndex):
        return "spans"
    if isinstance(tindex, td.GridIndex):
        kind = track.hints.get("kind")
        if kind == "spectrogram":
            return "audio_spectrogram"
        if kind == "salience":
            return "salience"
        extra_dims = [d for d in series.dims if d != "time"]
        if not extra_dims or (len(extra_dims) == 1 and extra_dims[0] in ("mic", "channel")):
            return "audio"
        return "waveform"
    raise ValueError(f"Cannot resolve a renderer for time index type {type(tindex).__name__}")


def resolve_title(name: str | None, track: PlotTrack, channel: int | None = None) -> str | None:
    """Resolve subplot title for a track.

    Order: explicit ``hints["title"]`` (can be ``None`` to suppress the
    title), then the track name, then a channel suffix.
    """
    if "title" in track.hints:
        return track.hints["title"]
    title = name if name is not None else "track"
    if channel is not None:
        title = f"{title} — ch{channel}"
    return title
