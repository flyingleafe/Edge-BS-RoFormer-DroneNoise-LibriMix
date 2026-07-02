"""Generic ``TimeFrame`` plotting machinery.

Importing this package registers the default renderers so that
``plot_timeframe`` can dispatch tracks by type or explicit tag.
"""

from __future__ import annotations

from .layout import plot_timeframe
from .registry import RenderedTrack, TrackContext, get_renderer, list_renderers, register_renderer

__all__ = [
    "plot_timeframe",
    "register_renderer",
    "get_renderer",
    "list_renderers",
    "TrackContext",
    "RenderedTrack",
]
