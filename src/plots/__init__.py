# src/plots/__init__.py — shared plot registry + lazy dwym front door.
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


def __getattr__(name: str):
    """Lazy front-door exports: ``dwym`` / ``coerce_frame`` / ``comb_explorer``.

    Loaded on first access so ``import plots`` (registry only) stays light —
    ``dwym`` pulls in matplotlib renderers and tdseries.
    """
    if name == "dwym":
        from plots.dwym import dwym

        # Cache the *function* on the package: the submodule import above
        # also sets ``plots.dwym = <module>``, which would shadow the
        # function on the next attribute lookup. Function wins.
        globals()["dwym"] = dwym
        return dwym
    if name == "coerce_frame":
        from plots.coerce import coerce_frame

        globals()["coerce_frame"] = coerce_frame
        return coerce_frame
    if name in ("comb_explorer", "discover"):
        from plots import comb_widget

        fn = getattr(comb_widget, name)
        globals()[name] = fn
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
