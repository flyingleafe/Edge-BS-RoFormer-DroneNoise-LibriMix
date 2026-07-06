"""Generic ``tdseries.Frame`` layout engine."""

from __future__ import annotations

from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import tdseries as td

from .registry import PlotTrack, TrackContext, get_renderer, resolve_renderer_key, resolve_title
from .renderers import _select_channels


def plot_timeframe(
    frame: td.Frame,
    *,
    tracks: list[Any] | None = None,
    channel: int | list[int] | str | None = None,
    figsize: tuple[float, float] = (16, 12),
    sharex: bool = True,
    height_ratios: list[float] | None = None,
    **style,
) -> matplotlib.figure.Figure:
    """Render a ``Frame`` as a stack of time-aligned subplots.

    Parameters
    ----------
    frame
        The ``Frame`` to render. Also supplies the shared axis bounds
        (``frame.t_start``/``frame.t_end``) even for ``tracks`` items that
        are not entries of this frame.
    tracks
        Ordered list of tracks to plot. Each item is either a frame entry
        name (``str``), a raw ``tdseries.Series``, or a
        :class:`~plots.timeframe.registry.PlotTrack` (e.g. from
        :func:`~plots.timeframe.renderers.make_spectrogram_series`). Default:
        every temporal entry of ``frame``, ``"audio"`` first (if present),
        then the rest sorted alphabetically.
    channel
        Channel selection for audio tracks. ``None`` selects channel 0 for
        mono / every channel for multichannel, ``"all"`` selects every
        channel, or pass int/list[int].
    figsize
        Figure size in inches.
    sharex
        Share x-axis across all rows.
    height_ratios
        Optional ratios for the rows; auto-assigned if omitted.
    **style
        Extra style forwarded to renderers.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(frame) == 0:
        raise ValueError("Frame has no tracks")

    if tracks is None:
        keys = [k for k, v in frame.items() if isinstance(v, td.Series) and v.has_time]
        items: list[Any] = []
        if "audio" in keys:
            items.append("audio")
            keys.remove("audio")
        items.extend(sorted(keys))
    else:
        items = list(tracks)

    # Determine row count per track and build a flattened render plan.
    plan: list[tuple[str | None, PlotTrack, str, int | None]] = []
    for item in items:
        name, track = _resolve_track_item(item, frame)
        key = resolve_renderer_key(track)
        if key == "audio":
            for ch in _select_channels(track.series, channel):
                plan.append((name, track, key, ch))
        else:
            plan.append((name, track, key, None))

    n_rows = len(plan)
    if height_ratios is None:
        height_ratios = [1.0] * n_rows

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

    t_start = frame.t_start
    t_end = frame.t_end
    shared_ax: matplotlib.axes.Axes | None = None

    for idx, (name, track, renderer_key, ch) in enumerate(plan):
        ax = fig.add_subplot(
            gs[idx], sharex=shared_ax if sharex and shared_ax is not None else None
        )
        if shared_ax is None:
            shared_ax = ax

        ctx_style = dict(style)
        ctx_style["_frame"] = frame
        ctx_style["_hints"] = track.hints
        if ch is not None:
            ctx_style["_channel"] = ch

        ctx = TrackContext(
            ax=ax,
            name=name if name is not None else "track",
            t_start=t_start,
            t_end=t_end,
            style=ctx_style,
        )
        get_renderer(renderer_key)(track.series, ctx)

        title = resolve_title(name, track, channel=ch)
        if title is not None:
            ax.set_title(title)

        # Only bottom row shows x-axis label/ticks.
        if idx < n_rows - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s)")

    gs.tight_layout(fig)
    return fig


def _resolve_track_item(item: Any, frame: td.Frame) -> tuple[str | None, PlotTrack]:
    """Normalise one ``tracks=[...]`` item into ``(name, PlotTrack)``.

    ``name`` is ``None`` for items that were not looked up by frame-entry
    name (raw ``Series``/``PlotTrack`` passed directly) — titles then fall
    back to a generic placeholder unless the track sets ``hints["title"]``.
    """
    if isinstance(item, str):
        series = frame[item]  # raises KeyError for unknown entries
        if not isinstance(series, td.Series):
            raise TypeError(f"Track {item!r} is not a Series (got {type(series).__name__})")
        return item, PlotTrack(series=series)
    if isinstance(item, PlotTrack):
        return None, item
    if isinstance(item, td.Series):
        return None, PlotTrack(series=item)
    raise TypeError(f"Unsupported track item {item!r} of type {type(item).__name__}")
