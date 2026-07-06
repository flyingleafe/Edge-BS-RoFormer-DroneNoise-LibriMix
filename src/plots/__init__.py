# src/utils/plots/__init__.py — shared plot registry + make_plot CLI.
"""Task-separated plotting — shared infrastructure.

Each task sub-package exposes a ``PLOT_TYPES`` dict mapping dotted names
to plot functions::

    from plots.rps_prediction import PLOT_TYPES as RPS_PLOT_TYPES

Plot functions have signature::

    fn(*, samples=None, result=None, models=None, ax=None, **style) -> Figure
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.figure

# Type alias for plot functions.
PlotFn = Callable[..., matplotlib.figure.Figure]

# Global registry of all known plot types (populated at import time by
# each task sub-package).
_PLOT_TYPES: dict[str, PlotFn] = {}


def register(name: str, fn: PlotFn) -> None:
    """Register a plot function under a dotted name."""
    if name in _PLOT_TYPES:
        raise ValueError(f"Duplicate plot type: {name!r}")
    _PLOT_TYPES[name] = fn


def get_plot_fn(name: str) -> PlotFn:
    """Look up a registered plot function by dotted name."""
    try:
        return _PLOT_TYPES[name]
    except KeyError:
        known = sorted(_PLOT_TYPES)
        raise ValueError(f"Unknown plot type {name!r}. Known: {known}") from None


def list_plot_types() -> list[str]:
    """Return all registered plot type names."""
    return sorted(_PLOT_TYPES)
