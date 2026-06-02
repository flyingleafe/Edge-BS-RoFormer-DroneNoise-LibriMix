"""Invariants for `SegmentSeries`."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from utils.data import IncompatibleSeriesError, SegmentSeries

from .strategies import segment_series


@st.composite
def _cuts(draw, ss: SegmentSeries, k: int):
    raw = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=k, max_size=k,
            )
        )
    )
    return [ss.t_start + f * (ss.t_end - ss.t_start) for f in raw]


@given(segment_series())
def test_slice_identity(ss):
    assert ss.slice(ss.t_start, ss.t_end).equal(ss)


@settings(max_examples=200)
@given(segment_series(), st.data())
def test_slice_concat_no_op(ss, data):
    [a, b, c] = data.draw(_cuts(ss, 3))
    joined = ss.slice(a, b).concat(ss.slice(b, c))
    assert joined.equal(ss.slice(a, c))


@settings(max_examples=100)
@given(segment_series(), st.data())
def test_many_cuts_rejoin(ss, data):
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [ss.t_start, *data.draw(_cuts(ss, k)), ss.t_end]
    parts = [ss.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(ss)


def test_splitting_a_straddling_segment():
    # One segment [0.2, 0.8); cut at 0.5.
    ss = SegmentSeries.from_segments(
        np.array([0.2]), np.array([0.8]), ids=np.array([42], dtype=np.int64),
        t_start=0.0, t_end=1.0,
    )
    left = ss.slice(0.0, 0.5)
    right = ss.slice(0.5, 1.0)
    assert len(left) == 1 and left.ends[0] == 0.5
    assert len(right) == 1 and right.starts[0] == 0.5
    assert int(left.ids[0]) == int(right.ids[0]) == 42
    # Re-merge on concat.
    rejoined = left.concat(right)
    assert len(rejoined) == 1
    assert rejoined.starts[0] == 0.2
    assert rejoined.ends[0] == 0.8


def test_unrelated_segments_meeting_at_seam_are_not_merged():
    # Two distinct segments that happen to touch at t=0.5 — no shared id.
    a = SegmentSeries.from_segments(
        np.array([0.2]), np.array([0.5]),
        ids=np.array([1], dtype=np.int64),
        t_start=0.0, t_end=0.5,
    )
    b = SegmentSeries.from_segments(
        np.array([0.5]), np.array([0.8]),
        ids=np.array([2], dtype=np.int64),
        t_start=0.5, t_end=1.0,
    )
    joined = a.concat(b)
    assert len(joined) == 2
    assert list(joined.starts) == [0.2, 0.5]
    assert list(joined.ends) == [0.5, 0.8]


def test_concat_rejects_seam_mismatch():
    a = SegmentSeries.from_segments(np.array([0.0]), np.array([0.3]), t_start=0.0, t_end=1.0)
    b = SegmentSeries.from_segments(np.array([2.0]), np.array([2.3]), t_start=2.0, t_end=3.0)
    with pytest.raises(IncompatibleSeriesError):
        a.concat(b)
