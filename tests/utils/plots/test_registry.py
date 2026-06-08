"""Smoke tests for `utils.plots` — registry and plot functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import pytest

from utils.plots import get_plot_fn, list_plot_types, register

# ── Registry ─────────────────────────────────────────────────────────────


def test_register_adds_to_registry():
    def dummy_plot(**kw):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        return fig

    register("test.dummy", dummy_plot)
    assert get_plot_fn("test.dummy") is dummy_plot


def test_register_duplicate_raises():
    def dummy_plot(**kw):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        return fig

    register("test.dummy2", dummy_plot)
    with pytest.raises(ValueError, match="Duplicate"):
        register("test.dummy2", dummy_plot)


def test_get_plot_fn_unknown_raises():
    with pytest.raises(ValueError, match="Unknown plot type"):
        get_plot_fn("nonexistent.plot.type.xyz")


def test_list_plot_types_returns_sorted():
    names = list_plot_types()
    assert isinstance(names, list)
    assert names == sorted(names)
