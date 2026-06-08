"""Invariants for `EventSeries` — exact int64 tick storage."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.data import DomainError, EventSeries, IncompatibleSeriesError, UniformSeries

from .strategies import cut_points_ticks, event_series


@st.composite
def _cuts(draw, es: EventSeries, k: int) -> list[int]:
    return draw(cut_points_ticks(es.t_start_ticks, es.t_end_ticks, k))


@given(event_series())
def test_slice_identity(es):
    assert es.slice(es.t_start_ticks, es.t_end_ticks).equal(es)


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
    pts = [es.t_start_ticks, *data.draw(_cuts(es, k)), es.t_end_ticks]
    parts = [es.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(es)


def test_events_at_cut_go_right():
    # Half-open: event exactly at t_cut belongs to the right half.
    t0 = 10_000_000_000  # 10 s in ns
    dur = 1_000_000_000  # 1 s
    ts = np.array([0, 500_000_000, 900_000_000], dtype=np.int64)  # relative
    vals = np.array([10.0, 20.0, 30.0])
    es = EventSeries.from_ticks(ts, vals, t_start=t0, dur=dur)
    t_cut = t0 + 500_000_000  # at the second event
    left = es.slice(t0, t_cut)
    right = es.slice(t_cut, t0 + dur)
    assert list(left.timestamp_ticks) == [0]
    assert list(right.timestamp_ticks) == [0, 400_000_000]
    assert left.concat(right).equal(es)


def test_getitem_returns_float_seconds():
    es = EventSeries.from_ticks(
        np.array([0, 500_000_000, 1_000_000_000], dtype=np.int64),
        np.array([1.0, 2.0, 3.0]),
        t_start=10_000_000_000,
        dur=2_000_000_000,
    )
    t0, v0 = es[0]
    assert t0 == pytest.approx(10.0)
    assert v0 == pytest.approx(1.0)
    t1, v1 = es[1]
    assert t1 == pytest.approx(10.5)
    assert v1 == pytest.approx(2.0)
    t2, v2 = es[2]
    assert t2 == pytest.approx(11.0)
    assert v2 == pytest.approx(3.0)


def test_empty_series():
    es = EventSeries.from_ticks(
        np.array([], dtype=np.int64),
        values=None,
        t_start=0,
        dur=1_000_000_000,
    )
    assert len(es) == 0
    half = es.slice(0, 500_000_000)
    assert len(half) == 0
    assert half.t_end_ticks == 500_000_000


def test_slice_outside_domain_raises():
    es = EventSeries.from_ticks(
        np.array([100_000_000], dtype=np.int64),
        np.array([1.0]),
        t_start=0,
        dur=1_000_000_000,
    )
    with pytest.raises(DomainError):
        es.slice(-100_000_000, 500_000_000)


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------


def test_shift_changes_t_start():
    es = EventSeries.from_ticks(
        np.array([0, 500_000_000], dtype=np.int64),
        np.array([1.0, 2.0]),
        t_start=0,
        dur=1_000_000_000,
    )
    shifted = es.shift(10_000_000_000)  # 10 s in ticks
    assert shifted.t_start_ticks == 10_000_000_000
    assert shifted.t_end_ticks == 11_000_000_000
    # Relative timestamps unchanged by shift.
    assert np.array_equal(shifted.timestamp_ticks, es.timestamp_ticks)
    # Absolute via __getitem__.
    t0, v0 = shifted[0]
    assert t0 == pytest.approx(10.0)
    assert v0 == pytest.approx(1.0)
    t1, v1 = shifted[1]
    assert t1 == pytest.approx(10.5)
    assert v1 == pytest.approx(2.0)


def test_shift_roundtrip():
    es = EventSeries.from_ticks(
        np.array([0, 500_000_000], dtype=np.int64),
        np.array([1.0, 2.0]),
        t_start=5_000_000_000,
        dur=1_000_000_000,
    )
    assert es.shift(3_000_000_000).shift(-3_000_000_000).equal(es)


# ---------------------------------------------------------------------------
# Concat with auto-shift (gap allowed)
# ---------------------------------------------------------------------------


def test_concat_across_gap():
    a = EventSeries.from_ticks(
        np.array([100_000_000], dtype=np.int64),
        np.array([1.0]),
        t_start=0,
        dur=1_000_000_000,
    )
    b = EventSeries.from_ticks(
        np.array([100_000_000], dtype=np.int64),
        np.array([2.0]),
        t_start=2_000_000_000,
        dur=1_000_000_000,
    )
    joined = a.concat(b)
    assert joined.t_start_ticks == 0
    assert joined.t_end_ticks == 2_000_000_000
    assert list(joined.abs_timestamps_ticks) == [100_000_000, 1_100_000_000]


def test_concat_rejects_value_shape_mismatch():
    # Values are time-last (..., M); 1 event => last axis length 1.
    a = EventSeries.from_ticks(
        np.array([100_000_000], dtype=np.int64),
        np.array([[1.0], [2.0]]),  # payload (2,), M=1
        t_start=0,
        dur=1_000_000_000,
    )
    b = EventSeries.from_ticks(
        np.array([100_000_000], dtype=np.int64),
        np.array([[1.0], [2.0], [3.0]]),  # payload (3,), M=1
        t_start=1_000_000_000,
        dur=1_000_000_000,
    )
    with pytest.raises(IncompatibleSeriesError):
        a.concat(b)


# ── Interpolation / interpolate_uniform ──────────────────────────────────


def test_slice_multidim_values_is_time_last():
    # Regression: 4-rotor RPS stored (4, M); slice must cut the M (last) axis
    # and keep all 4 channels — not silently produce (4, 0).
    M = 10
    ts = np.arange(M, dtype=np.int64) * 100_000_000  # 0,0.1,...,0.9 s
    vals = np.stack(
        [np.arange(M), np.arange(M) + 100, np.arange(M) + 200, np.arange(M) + 300]
    ).astype(np.float64)  # (4, M)
    es = EventSeries.from_ticks(ts, vals, t_start=0, dur=1_000_000_000)
    sl = es.slice(300_000_000, 700_000_000)  # events at 0.3,0.4,0.5,0.6
    assert sl.values.shape == (4, 4)  # pyright: ignore[reportOptionalMemberAccess]
    np.testing.assert_array_equal(sl.values[:, 0], [3, 103, 203, 303])  # pyright: ignore[reportOptionalSubscript]
    _, v = es[2]
    np.testing.assert_array_equal(v, [2, 102, 202, 302])
    rejoined = es.slice(0, 300_000_000).concat(es.slice(300_000_000, 1_000_000_000))
    assert rejoined.equal(es)


def test_interpolate_no_values_raises():
    es = EventSeries.from_ticks(
        np.array([100_000_000, 200_000_000], dtype=np.int64),
        values=None,
        t_start=0,
        dur=1_000_000_000,
    )
    with pytest.raises(ValueError, match="no values"):
        es.interpolate(np.array([0.1]))


def test_interpolate_at_event_times_1d():
    es = EventSeries.from_events(
        np.array([0.0, 0.5, 1.0]),
        values=np.array([10.0, 20.0, 30.0]),
        t_start=0.0,
        t_end=1.0,
    )
    vals = es.interpolate(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(vals, [10.0, 20.0, 30.0], atol=1e-12)


def test_interpolate_midpoint():
    es = EventSeries.from_events(
        np.array([0.0, 1.0]),
        values=np.array([0.0, 10.0]),
        t_start=0.0,
        t_end=1.0,
    )
    v = es.interpolate(np.array([0.5]))[0]
    np.testing.assert_allclose(v, 5.0, atol=1e-12)


def test_interpolate_multichannel():
    es = EventSeries.from_events(
        np.array([0.0, 0.5, 1.0]),
        values=np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        t_start=0.0,
        t_end=1.0,
    )
    vals = es.interpolate(np.array([0.25, 0.75]))
    assert vals.shape == (2, 2)
    np.testing.assert_allclose(vals[0], [1.5, 2.5], atol=1e-12)
    np.testing.assert_allclose(vals[1], [15.0, 25.0], atol=1e-12)


def test_interpolate_clamp_extrap():
    es = EventSeries.from_events(
        np.array([0.2, 0.8]),
        values=np.array([5.0, 15.0]),
        t_start=0.0,
        t_end=1.0,
    )
    v_left = es.interpolate(np.array([0.0]))[0]
    v_right = es.interpolate(np.array([1.0]))[0]
    assert v_left == 5.0
    assert v_right == 15.0


def test_interpolate_uniform_basic():
    es = EventSeries.from_events(
        np.array([0.0, 0.5, 1.0]),
        values=np.array([10.0, 20.0, 30.0]),
        t_start=0.0,
        t_end=1.0,
    )
    us = es.interpolate_uniform(sr=4.0)
    assert us.sr == 4.0
    assert us.n_samples == 4
    np.testing.assert_allclose(us.samples, [10.0, 15.0, 20.0, 25.0], atol=0.1)


def test_interpolate_uniform_phase_zero():
    es = EventSeries.from_events(
        np.array([1.0, 1.9]),
        values=np.array([10.0, 20.0]),
        t_start=1.0,
        t_end=2.0,
    )
    us = es.interpolate_uniform(sr=2.0)
    # phase=0: sample k at t_start + k/sr
    assert us.t_start_ticks == es.t_start_ticks
    # First sample at t=1.0
    np.testing.assert_allclose(us.samples[0], 10.0, atol=1e-12)


def test_interpolate_uniform_custom_domain():
    es = EventSeries.from_events(
        np.array([0.0, 0.5, 1.0]),
        values=np.array([10.0, 20.0, 30.0]),
        t_start=0.0,
        t_end=1.0,
    )
    us = es.interpolate_uniform(sr=2.0, t_start=0.2, t_end=0.8)
    # Domain 0.6 s at 2 Hz → 1 sample at t=0.2 (N=round(0.6*2)=1, dur_ticks=round(1/2*s))=0.5s)
    assert us.t_start == pytest.approx(0.2, abs=1e-9)
    assert us.n_samples >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Validation paths (__post_init__)
# ═══════════════════════════════════════════════════════════════════════════


def test_init_rejects_2d_timestamps():
    with pytest.raises(ValueError, match="1-D"):
        EventSeries(
            timestamp_ticks=np.array([[1, 2], [3, 4]], dtype=np.int64),
            values=None,
            t_start_ticks=0,
            dur_ticks=100,
        )


def test_init_rejects_value_shape_mismatch_last_axis():
    with pytest.raises(ValueError, match="values.shape"):
        EventSeries(
            timestamp_ticks=np.array([100, 200], dtype=np.int64),
            values=np.array([[1.0], [2.0]]),
            t_start_ticks=0,
            dur_ticks=300,
        )


def test_init_rejects_negative_dur_ticks():
    with pytest.raises(ValueError, match="dur_ticks"):
        EventSeries(
            timestamp_ticks=np.array([], dtype=np.int64), values=None, t_start_ticks=0, dur_ticks=-1
        )


def test_init_rejects_negative_relative_timestamp():
    with pytest.raises(ValueError, match="before t_start"):
        EventSeries(
            timestamp_ticks=np.array([-1], dtype=np.int64),
            values=np.array([1.0]),
            t_start_ticks=0,
            dur_ticks=100,
        )


def test_init_rejects_event_after_domain_end():
    with pytest.raises(ValueError, match="outside domain"):
        EventSeries(
            timestamp_ticks=np.array([200], dtype=np.int64),
            values=np.array([1.0]),
            t_start_ticks=0,
            dur_ticks=100,
        )


def test_init_rejects_unsorted_timestamps():
    with pytest.raises(ValueError, match="sorted"):
        EventSeries(
            timestamp_ticks=np.array([200, 100], dtype=np.int64),
            values=np.array([1.0, 2.0]),
            t_start_ticks=0,
            dur_ticks=300,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Constructor branches
# ═══════════════════════════════════════════════════════════════════════════


def test_from_events_with_float_timestamps():
    es = EventSeries.from_events(
        np.array([0.5, 0.8]), np.array([10.0, 20.0]), t_start=0.0, t_end=1.0
    )
    assert es.t_start_ticks == 0
    assert es.t_end_ticks == 1_000_000_000
    assert len(es) == 2


def test_from_events_empty_infers_zero_domain():
    es = EventSeries.from_events(np.array([], dtype=np.float64), values=None)
    assert es.t_start_ticks == 0
    assert es.dur_ticks == 0
    assert len(es) == 0


def test_from_ticks_with_explicit_dur():
    es = EventSeries.from_ticks(
        np.array([100, 500], dtype=np.int64), np.array([1.0, 2.0]), t_start=1000, dur=600
    )
    assert es.t_start_ticks == 1000
    assert es.dur_ticks == 600
    assert es.t_end_ticks == 1600


# ═══════════════════════════════════════════════════════════════════════════
# Interpolation edge paths
# ═══════════════════════════════════════════════════════════════════════════


def test_interpolate_with_int_query_times():
    es = EventSeries.from_events(
        np.array([0.0, 0.5, 1.0]), np.array([10.0, 20.0, 30.0]), t_start=0.0, t_end=1.0
    )
    vals = es.interpolate(np.array([500_000_000, 750_000_000], dtype=np.int64))
    np.testing.assert_allclose(vals, [20.0, 25.0], atol=1e-6)


def test_interpolate_empty_series_fill_error():
    es = EventSeries.from_events(
        np.array([], dtype=np.float64), np.array([], dtype=np.float64), t_start=0.0, t_end=1.0
    )
    with pytest.raises(DomainError, match="empty"):
        es.interpolate(np.array([0.5]), fill="error")


def test_interpolate_empty_series_fill_nan():
    es = EventSeries.from_events(
        np.array([], dtype=np.float64), np.array([], dtype=np.float64), t_start=0.0, t_end=1.0
    )
    vals = es.interpolate(np.array([0.5]), fill="nan")
    assert np.isnan(vals[0])


def test_interpolate_unsupported_kind():
    es = EventSeries.from_events(np.array([0.0, 1.0]), np.array([1.0, 2.0]), t_start=0.0, t_end=1.0)
    with pytest.raises(ValueError, match="unsupported interpolation kind"):
        es.interpolate(np.array([0.5]), kind="cubic")


def test_interpolate_fill_nan_extrapolation():
    es = EventSeries.from_events(
        np.array([0.2, 0.8]), np.array([5.0, 15.0]), t_start=0.0, t_end=1.0
    )
    vals = es.interpolate(np.array([0.0, 1.0]), fill="nan")
    assert np.isnan(vals[0])
    assert np.isnan(vals[1])


def test_interpolate_fill_error_extrapolation():
    es = EventSeries.from_events(
        np.array([0.2, 0.8]), np.array([5.0, 15.0]), t_start=0.0, t_end=1.0
    )
    with pytest.raises(DomainError, match="outside event span"):
        es.interpolate(np.array([0.0, 1.0]), fill="error")


def test_interpolate_fill_error_within_span():
    es = EventSeries.from_events(
        np.array([0.2, 0.8]), np.array([5.0, 15.0]), t_start=0.0, t_end=1.0
    )
    vals = es.interpolate(np.array([0.3, 0.7]), fill="error")
    assert len(vals) == 2


def test_interpolate_unsupported_fill_value():
    es = EventSeries.from_events(np.array([0.0, 1.0]), np.array([1.0, 2.0]), t_start=0.0, t_end=1.0)
    with pytest.raises(ValueError, match="unsupported fill"):
        es.interpolate(np.array([0.5]), fill="extrap")


# ═══════════════════════════════════════════════════════════════════════════
# Concat / equal error paths
# ═══════════════════════════════════════════════════════════════════════════


def test_concat_rejects_non_event_series():
    es = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=0.5)
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    with pytest.raises(IncompatibleSeriesError, match="cannot concat"):
        es.concat(us)


def test_concat_rejects_one_has_values_one_does_not():
    a = EventSeries.from_events(np.array([0.1]), values=np.array([1.0]), t_start=0.0, t_end=0.5)
    b = EventSeries.from_events(np.array([0.1]), values=None, t_start=0.5, t_end=1.0)
    with pytest.raises(IncompatibleSeriesError, match="one has values"):
        a.concat(b)


def test_equal_non_event_series_is_false():
    es = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=0.5)
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    assert es.equal(us) is False


def test_equal_different_t_start_is_false():
    a = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=0.5)
    b = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=1.0, t_end=1.5)
    assert a.equal(b) is False


def test_equal_one_none_values_one_not():
    a = EventSeries.from_events(np.array([0.1]), values=np.array([1.0]), t_start=0.0, t_end=0.5)
    b = EventSeries.from_events(np.array([0.1]), values=None, t_start=0.0, t_end=0.5)
    assert a.equal(b) is False


def test_equal_different_dur_is_false():
    a = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=0.5)
    b = EventSeries.from_events(np.array([0.1]), np.array([1.0]), t_start=0.0, t_end=0.8)
    assert a.equal(b) is False


# ═══════════════════════════════════════════════════════════════════════════
# interpolate_uniform guards
# ═══════════════════════════════════════════════════════════════════════════


def test_interpolate_uniform_rejects_nonpositive_sr():
    es = EventSeries.from_events(np.array([0.0, 1.0]), np.array([1.0, 2.0]), t_start=0.0, t_end=1.0)
    with pytest.raises(ValueError, match="sr must be > 0"):
        es.interpolate_uniform(0.0)
    with pytest.raises(ValueError, match="sr must be > 0"):
        es.interpolate_uniform(-1.0)


def test_interpolate_uniform_rejects_zero_dur():
    es = EventSeries.from_events(np.array([0.0, 1.0]), np.array([1.0, 2.0]), t_start=0.0, t_end=1.0)
    with pytest.raises(ValueError, match="duration must be > 0"):
        es.interpolate_uniform(10.0, t_start=100, t_end=100)
