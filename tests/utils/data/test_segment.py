"""Invariants for `SegmentSeries` — exact int64 tick storage."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from utils.data import IncompatibleSeriesError, SegmentSeries

from .strategies import cut_points_ticks, segment_series


@st.composite
def _cuts(draw, ss: SegmentSeries, k: int) -> list[int]:
    return draw(cut_points_ticks(ss.t_start_ticks, ss.t_end_ticks, k))


@given(segment_series())
def test_slice_identity(ss):
    assert ss.slice(ss.t_start_ticks, ss.t_end_ticks).equal(ss)


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
    pts = [ss.t_start_ticks, *data.draw(_cuts(ss, k)), ss.t_end_ticks]
    parts = [ss.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(ss)


def test_splitting_a_straddling_segment():
    # One segment [200M, 800M] ns (0.2s .. 0.8s); cut at 500M ns (0.5s).
    dur = 1_000_000_000  # 1 s
    ss = SegmentSeries.from_ticks(
        np.array([200_000_000], dtype=np.int64),
        np.array([800_000_000], dtype=np.int64),
        ids=np.array([42], dtype=np.int64),
        t_start=0, dur=dur,
    )
    left = ss.slice(0, 500_000_000)
    right = ss.slice(500_000_000, dur)
    assert len(left) == 1 and left.ends_ticks[0] == 500_000_000
    assert len(right) == 1 and right.starts_ticks[0] == 0  # relative to right.t_start
    assert right.abs_starts_ticks[0] == 500_000_000
    assert int(left.ids[0]) == int(right.ids[0]) == 42
    rejoined = left.concat(right)
    assert len(rejoined) == 1
    assert rejoined.abs_starts_ticks[0] == 200_000_000
    assert rejoined.abs_ends_ticks[0] == 800_000_000


def test_unrelated_segments_meeting_at_seam_are_not_merged():
    # Two distinct segments that touch at t=0.5s — no shared id.
    a = SegmentSeries.from_ticks(
        np.array([200_000_000], dtype=np.int64),
        np.array([500_000_000], dtype=np.int64),
        ids=np.array([1], dtype=np.int64),
        t_start=0, dur=500_000_000,
    )
    b = SegmentSeries.from_ticks(
        np.array([0], dtype=np.int64),
        np.array([300_000_000], dtype=np.int64),
        ids=np.array([2], dtype=np.int64),
        t_start=500_000_000, dur=500_000_000,
    )
    joined = a.concat(b)
    assert len(joined) == 2
    assert list(joined.abs_starts_ticks) == [200_000_000, 500_000_000]
    assert list(joined.abs_ends_ticks) == [500_000_000, 800_000_000]


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------

def test_shift():
    ss = SegmentSeries.from_ticks(
        np.array([200_000_000, 500_000_000], dtype=np.int64),
        np.array([500_000_000, 800_000_000], dtype=np.int64),
        t_start=0, dur=1_000_000_000,
    )
    shifted = ss.shift(10_000_000_000)
    assert shifted.t_start_ticks == 10_000_000_000
    assert shifted.t_end_ticks == 11_000_000_000
    # Relative storage unchanged.
    assert list(shifted.starts_ticks) == [200_000_000, 500_000_000]
    assert list(shifted.abs_starts_ticks) == [10_200_000_000, 10_500_000_000]
    assert list(shifted.abs_ends_ticks) == [10_500_000_000, 10_800_000_000]


def test_shift_roundtrip():
    ss = SegmentSeries.from_ticks(
        np.array([200_000_000], dtype=np.int64),
        np.array([500_000_000], dtype=np.int64),
        t_start=0, dur=1_000_000_000,
    )
    assert ss.shift(5_000_000_000).shift(-5_000_000_000).equal(ss)


# ---------------------------------------------------------------------------
# Concat with auto-shift (gap allowed)
# ---------------------------------------------------------------------------

def test_concat_across_gap():
    a = SegmentSeries.from_ticks(
        np.array([0], dtype=np.int64),
        np.array([300_000_000], dtype=np.int64),
        t_start=0, dur=1_000_000_000,
    )
    b = SegmentSeries.from_ticks(
        np.array([0], dtype=np.int64),
        np.array([300_000_000], dtype=np.int64),
        t_start=2_000_000_000, dur=1_000_000_000,
    )
    joined = a.concat(b)
    assert joined.t_start_ticks == 0
    assert joined.t_end_ticks == 2_000_000_000
    assert list(joined.abs_starts_ticks) == [0, 1_000_000_000]
    assert list(joined.abs_ends_ticks) == [300_000_000, 1_300_000_000]
