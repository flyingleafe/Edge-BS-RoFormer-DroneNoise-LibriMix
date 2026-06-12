"""Tests for `_ticks.py` conversion helpers and `base.py` dunder methods."""

from __future__ import annotations

import numpy as np
import pytest

from utils.data import EventSeries, UniformSeries
from utils.data._ticks import TICKS_PER_SECOND, _c_to_ticks, secs_to_ticks, ticks_to_secs

# ── _c_to_ticks ──────────────────────────────────────────────────────────


def test_c_to_ticks_from_float():
    """_c_to_ticks(float) quantises via round(f * TPS)."""
    assert _c_to_ticks(1.0) == TICKS_PER_SECOND
    assert _c_to_ticks(0.5) == 500_000_000
    assert _c_to_ticks(0.0) == 0


def test_c_to_ticks_from_int_is_id():
    """_c_to_ticks(int) returns the int unchanged."""
    assert _c_to_ticks(42) == 42
    assert _c_to_ticks(0) == 0
    assert _c_to_ticks(TICKS_PER_SECOND) == TICKS_PER_SECOND


# ── secs_to_ticks / ticks_to_secs roundtrip ──────────────────────────────


def test_secs_to_ticks_roundtrip():
    """secs_to_ticks then ticks_to_secs recovers the original."""
    assert ticks_to_secs(secs_to_ticks(0.0)) == pytest.approx(0.0)
    assert ticks_to_secs(secs_to_ticks(1.0)) == pytest.approx(1.0)
    assert ticks_to_secs(secs_to_ticks(0.123456789)) == pytest.approx(0.123456789)


# ── base.py: __add__ ─────────────────────────────────────────────────────


def test_add_is_concat():
    """ts + other == ts.concat(other) for all concrete types."""
    a = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    b = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=1.0)
    assert (a + b).equal(a.concat(b))

    ea = EventSeries.from_events(np.array([0.1, 0.3]), np.array([1.0, 2.0]), t_start=0.0, t_end=0.5)
    eb = EventSeries.from_events(np.array([0.1, 0.3]), np.array([3.0, 4.0]), t_start=0.5, t_end=1.0)
    assert (ea + eb).equal(ea.concat(eb))


# ── base.py: __eq__ ──────────────────────────────────────────────────────


def test_eq_not_time_series_returns_notimplemented():
    """__eq__ with non-TimeSeries returns NotImplemented (Python falls back to False)."""
    from utils.data.base import TimeSeries

    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    # Call __eq__ directly to inspect the return value before Python's fallback.
    result = TimeSeries.__eq__(us, 42)
    assert result is NotImplemented


def test_eq_different_type_returns_false():
    """UniformSeries == EventSeries is False."""
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    es = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=0.5)
    assert (us == es) is False


def test_eq_delegates_to_equal():
    """(a == b) == a.equal(b) for same-type series."""
    a = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    b = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    c = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=100)
    assert (a == b) == a.equal(b)
    assert (a == c) == a.equal(c)


# ── base.py: __hash__ ────────────────────────────────────────────────────


def test_hash_is_id():
    """hash(ts) == id(ts) — identity-based for frozen dataclasses."""
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    assert hash(us) == id(us)
