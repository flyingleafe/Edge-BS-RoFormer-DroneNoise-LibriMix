"""Generic ``TimeFrame`` layout engine."""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt

from utils.data import TimeFrame, UniformSeries

from .registry import TrackContext, resolve_renderer_key, resolve_title
from .renderers import _select_channels


def _count_rows(series: Any, renderer_key: str, channel_sel: list[int] | None) -> int:
    """Return how many grid rows a track consumes."""
    if renderer_key == "audio" and isinstance(series, UniformSeries):
        return len(channel_sel) if channel_sel else 1
    return 1


def plot_timeframe(
    frame: TimeFrame,
    *,
    tracks: list[str] | None = None,
    channel: int | list[int] | str | None = None,
    figsize: tuple[float, float] = (16, 12),
    sharex: bool = True,
    height_ratios: list[float] | None = None,
    **style,
) -> matplotlib.figure.Figure:
    """Render a ``TimeFrame`` as a stack of time-aligned subplots.

    Parameters
    ----------
    frame
        The ``TimeFrame`` to render.
    tracks
        Ordered list of track names to plot.  Default: ``audio`` first (if
        present), then remaining tracks sorted alphabetically.
    channel
        Channel selection for audio tracks.  ``None`` selects channel 0,
        ``"all"`` selects every channel, or pass int/list[int].
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
    if not frame:
        raise ValueError("TimeFrame has no tracks")

    if tracks is None:
        keys = list(frame.keys())
        tracks = []
        if "audio" in keys:
            tracks.append("audio")
            keys.remove("audio")
        tracks.extend(sorted(keys))
    else:
        tracks = list(tracks)
        for name in tracks:
            if name not in frame:
                raise KeyError(name)

    # Determine row count per track and build a flattened render plan.
    plan: list[tuple[str, Any, str, Any]] = []  # name, series, renderer_key, channel_info
    for name in tracks:
        series = frame[name]
        key = resolve_renderer_key(name, series)
        if key == "audio" and isinstance(series, UniformSeries):
            ch_sel = _select_channels(series, channel)
            for ch in ch_sel:
                plan.append((name, series, key, ch))
        else:
            plan.append((name, series, key, None))

    n_rows = len(plan)
    if height_ratios is None:
        height_ratios = [1.0] * n_rows

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

    t_start = frame.t_start
    t_end = frame.t_end
    shared_ax: matplotlib.axes.Axes | None = None

    for idx, (name, series, renderer_key, ch_info) in enumerate(plan):
        from .registry import get_renderer

        ax = fig.add_subplot(
            gs[idx], sharex=shared_ax if sharex and shared_ax is not None else None
        )
        if shared_ax is None:
            shared_ax = ax

        ctx_style = dict(style)
        ctx_style["_frame"] = frame
        if ch_info is not None:
            ctx_style["_channel"] = ch_info

        ctx = TrackContext(
            ax=ax,
            name=name,
            t_start=t_start,
            t_end=t_end,
            style=ctx_style,
        )
        get_renderer(renderer_key)(series, ctx)

        title = resolve_title(name, series, channel=ch_info)
        if title is not None:
            ax.set_title(title)

        # Only bottom row shows x-axis label/ticks.
        if idx < n_rows - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s)")

    gs.tight_layout(fig)
    return fig
