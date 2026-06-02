"""Invariants for `EventSeries`."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from utils.data import DomainError, EventSeries, IncompatibleSeriesError

from .strategies import event_series


@st.composite
def _cuts(draw, es: EventSeries, k: int):
    raw = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=k, max_size=k,
            )
        )
    )
    return [es.t_start + f * (es.t_end - es.t_start) for f in raw]


@given(event_series())
def test_slice_identity(es):
    assert es.slice(es.t_start, es.t_end).equal(es)


@settings(max_examples=200)
@given(event_series(), st.data())
def test_slice_concat_no_op(es, data):
    [a, b, c] = data.draw(_cuts(es, 3))
    joined = es.slice(a, b).concat(es.slice(b, c))
    assert joined.equal(es.slice(a, c))


@settings(max_examples=100)
@given(event_series(), st.data())
def test_many_cuts_rejoin(es, data):
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [es.t_start, *data.draw(_cuts(es, k)), es.t_end]
    parts = [es.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(es)


def test_events_at_cut_go_right():
    # Half-open convention: event exactly at t_cut belongs to the right half.
    ts = np.array([0.0, 0.5, 0.9])
    vals = np.array([10.0, 20.0, 30.0])
    es = EventSeries.from_events(ts, vals, t_start=0.0, t_end=1.0)
    left = es.slice(0.0, 0.5)
    right = es.slice(0.5, 1.0)
    assert list(left.timestamps) == [0.0]
    assert list(right.timestamps) == [0.5, 0.9]
    assert left.concat(right).equal(es)


def test_empty_series():
    es = EventSeries.from_events(
        np.array([], dtype=np.float64), values=None, t_start=0.0, t_end=1.0,
    )
    assert len(es) == 0
    half = es.slice(0.0, 0.5)
    assert len(half) == 0
    assert half.t_end == 0.5


def test_slice_outside_domain_raises():
    es = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=1.0)
    with pytest.raises(DomainError):
        es.slice(-0.1, 0.5)


def test_concat_rejects_mismatched_seam():
    a = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=1.0)
    b = EventSeries.from_events(np.array([2.1]), np.array([2.0]), t_start=2.0, t_end=3.0)
    with pytest.raises(IncompatibleSeriesError):
        a.concat(b)


def test_concat_rejects_value_shape_mismatch():
    a = EventSeries.from_events(
        np.array([0.1]),
        np.array([[1.0, 2.0]]),
        t_start=0.0, t_end=1.0,
    )
    b = EventSeries.from_events(
        np.array([1.1]),
        np.array([[1.0, 2.0, 3.0]]),
        t_start=1.0, t_end=2.0,
    )
    with pytest.raises(IncompatibleSeriesError):
        a.concat(b)
