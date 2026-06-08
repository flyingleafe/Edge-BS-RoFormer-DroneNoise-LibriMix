"""Invariants for `SegmentSeries` — exact int64 tick storage."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
        t_start=0,
        dur=dur,
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
        t_start=0,
        dur=500_000_000,
    )
    b = SegmentSeries.from_ticks(
        np.array([0], dtype=np.int64),
        np.array([300_000_000], dtype=np.int64),
        ids=np.array([2], dtype=np.int64),
        t_start=500_000_000,
        dur=500_000_000,
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
        t_start=0,
        dur=1_000_000_000,
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
        t_start=0,
        dur=1_000_000_000,
    )
    assert ss.shift(5_000_000_000).shift(-5_000_000_000).equal(ss)


# ---------------------------------------------------------------------------
# Concat with auto-shift (gap allowed)
# ---------------------------------------------------------------------------


def test_concat_across_gap():
    a = SegmentSeries.from_ticks(
        np.array([0], dtype=np.int64),
        np.array([300_000_000], dtype=np.int64),
        t_start=0,
        dur=1_000_000_000,
    )
    b = SegmentSeries.from_ticks(
        np.array([0], dtype=np.int64),
        np.array([300_000_000], dtype=np.int64),
        t_start=2_000_000_000,
        dur=1_000_000_000,
    )
    joined = a.concat(b)
    assert joined.t_start_ticks == 0
    assert joined.t_end_ticks == 2_000_000_000
    assert list(joined.abs_starts_ticks) == [0, 1_000_000_000]
    assert list(joined.abs_ends_ticks) == [300_000_000, 1_300_000_000]


# ═══════════════════════════════════════════════════════════════════════════
# Validation (__post_init__)
# ═══════════════════════════════════════════════════════════════════════════


def test_init_rejects_2d_starts():
    with pytest.raises(ValueError, match="1-D"):
        SegmentSeries(
            starts_ticks=np.array([[1], [2]]),
            ends_ticks=np.array([[2], [3]]),
            t_start_ticks=0,
            dur_ticks=100,
        )


def test_init_rejects_end_not_greater_than_start():
    with pytest.raises(ValueError, match="end > start"):
        SegmentSeries(
            starts_ticks=np.array([10]), ends_ticks=np.array([5]), t_start_ticks=0, dur_ticks=100
        )


def test_init_sorts_unsorted_segments():
    ss = SegmentSeries(
        starts_ticks=np.array([50, 10]),
        ends_ticks=np.array([60, 30]),
        t_start_ticks=0,
        dur_ticks=100,
    )
    assert list(ss.starts_ticks) == [10, 50]


def test_init_rejects_wrong_shape_ids():
    with pytest.raises(ValueError, match="ids must be 1-D"):
        SegmentSeries(
            starts_ticks=np.array([10]),
            ends_ticks=np.array([20]),
            ids=np.array([[1], [2]]),
            t_start_ticks=0,
            dur_ticks=100,
        )


def test_init_rejects_wrong_values_last_axis():
    with pytest.raises(ValueError, match="values.shape"):
        SegmentSeries(
            starts_ticks=np.array([10, 40]),
            ends_ticks=np.array([20, 60]),
            values=np.array([[1.0], [2.0], [3.0]]),
            t_start_ticks=0,
            dur_ticks=100,
        )


def test_init_rejects_negative_dur_ticks():
    with pytest.raises(ValueError, match="dur_ticks"):
        SegmentSeries(
            starts_ticks=np.array([], dtype=np.int64),
            ends_ticks=np.array([], dtype=np.int64),
            t_start_ticks=0,
            dur_ticks=-1,
        )


def test_init_rejects_start_before_t_start():
    with pytest.raises(ValueError, match="starts before t_start"):
        SegmentSeries(
            starts_ticks=np.array([-5]), ends_ticks=np.array([10]), t_start_ticks=0, dur_ticks=100
        )


def test_init_rejects_end_after_t_end():
    with pytest.raises(ValueError, match="ends after"):
        SegmentSeries(
            starts_ticks=np.array([50]), ends_ticks=np.array([150]), t_start_ticks=0, dur_ticks=100
        )


# ═══════════════════════════════════════════════════════════════════════════
# from_segments float-input path
# ═══════════════════════════════════════════════════════════════════════════


def test_from_segments_with_float_starts_ends():
    ss = SegmentSeries.from_segments(
        np.array([0.2, 0.6]), np.array([0.4, 0.9]), t_start=0.0, t_end=1.0
    )
    assert ss.t_start_ticks == 0
    assert len(ss) == 2


def test_from_segments_with_float_t_start_t_end():
    ss = SegmentSeries.from_segments(
        np.array([0.2, 0.6]), np.array([0.4, 0.9]), t_start=0.1, t_end=1.0
    )
    assert ss.t_start == pytest.approx(0.1)
    assert ss.t_end == pytest.approx(1.0)


def test_from_segments_empty_infers_zero_domain():
    ss = SegmentSeries.from_segments(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    assert ss.t_start_ticks == 0
    assert ss.dur_ticks == 0


# ═══════════════════════════════════════════════════════════════════════════
# Concat / equal error paths
# ═══════════════════════════════════════════════════════════════════════════


def test_concat_rejects_non_segment_series():
    from utils.data import UniformSeries

    ss = SegmentSeries.from_segments(np.array([0.2]), np.array([0.4]), t_start=0.0, t_end=1.0)
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    with pytest.raises(IncompatibleSeriesError, match="cannot concat"):
        ss.concat(us)


def test_concat_rejects_one_has_values_one_does_not():
    a = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        values=np.array([[1.0]]),
        t_start_ticks=0,
        dur_ticks=100,
    )
    b = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        values=None,
        t_start_ticks=100,
        dur_ticks=100,
    )
    with pytest.raises(IncompatibleSeriesError, match="one has values"):
        a.concat(b)


def test_equal_non_segment_series():
    from utils.data import UniformSeries

    ss = SegmentSeries(
        starts_ticks=np.array([10]), ends_ticks=np.array([20]), t_start_ticks=0, dur_ticks=100
    )
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    assert ss.equal(us) is False


def test_equal_different_ids():
    a = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        ids=np.array([1]),
        t_start_ticks=0,
        dur_ticks=100,
    )
    b = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        ids=np.array([2]),
        t_start_ticks=0,
        dur_ticks=100,
    )
    assert a.equal(b) is False


def test_equal_different_values():
    a = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        values=np.array([[5.0]]),
        t_start_ticks=0,
        dur_ticks=100,
    )
    b = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        values=np.array([[7.0]]),
        ids=a.ids,
        t_start_ticks=0,
        dur_ticks=100,
    )
    assert a.equal(b) is False


def test_equal_one_has_values_one_does_not():
    a = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        values=np.array([[5.0]]),
        t_start_ticks=0,
        dur_ticks=100,
    )
    b = SegmentSeries(
        starts_ticks=np.array([10]),
        ends_ticks=np.array([20]),
        values=None,
        ids=a.ids,
        t_start_ticks=0,
        dur_ticks=100,
    )
    assert a.equal(b) is False


# ═══════════════════════════════════════════════════════════════════════════
# __getitem__ with values / properties
# ═══════════════════════════════════════════════════════════════════════════


def test_getitem_with_values_returns_4_tuple():
    ss = SegmentSeries(
        starts_ticks=np.array([10, 40]),
        ends_ticks=np.array([20, 60]),
        values=np.array([[100.0, 200.0]]),
        t_start_ticks=1_000_000_000,
        dur_ticks=100,
    )
    s, e, v, i = ss[0]
    # start time = t_start + ticks_to_secs(10)
    assert s == pytest.approx(1.0 + 10 / 1e9, rel=1e-9)
    assert e == pytest.approx(1.0 + 20 / 1e9, rel=1e-9)
    assert v == pytest.approx(100.0)


def test_starts_returns_float_seconds():
    ss = SegmentSeries.from_ticks(
        np.array([100_000_000, 500_000_000]),
        np.array([200_000_000, 600_000_000]),
        t_start=0,
        dur=1_000_000_000,
    )
    assert isinstance(ss.starts[0], (float, np.floating))


def test_ends_returns_float_seconds():
    ss = SegmentSeries.from_ticks(
        np.array([100_000_000, 500_000_000]),
        np.array([200_000_000, 600_000_000]),
        t_start=0,
        dur=1_000_000_000,
    )
    assert isinstance(ss.ends[0], (float, np.floating))


def test_abs_starts_is_float_seconds():
    ss = SegmentSeries.from_ticks(
        np.array([100_000_000]), np.array([200_000_000]), t_start=1_000_000_000, dur=1_000_000_000
    )
    assert isinstance(ss.abs_starts[0], (float, np.floating))


def test_abs_ends_is_float_seconds():
    ss = SegmentSeries.from_ticks(
        np.array([100_000_000]), np.array([200_000_000]), t_start=1_000_000_000, dur=1_000_000_000
    )
    assert isinstance(ss.abs_ends[0], (float, np.floating))
