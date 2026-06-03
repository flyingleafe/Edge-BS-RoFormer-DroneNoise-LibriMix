"""Invariants for `TimeFrame` — the column-keyed container."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from utils.data import (
    DomainError,
    EventSeries,
    IncompatibleSeriesError,
    SegmentSeries,
    TimeFrame,
    UniformSeries,
)

from .strategies import sample_rates, time_anchors


@st.composite
def time_frame(draw) -> TimeFrame:
    """Build a frame whose tracks share the same arbitrary [t0, t1)."""
    sr = draw(sample_rates)
    n = draw(st.integers(min_value=4, max_value=40))
    t0 = draw(time_anchors)
    samples = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
                min_size=n, max_size=n,
            )
        ),
        dtype=np.float64,
    )
    us = UniformSeries.from_samples(samples, sr=sr, t_start=t0)
    duration = us.duration
    t1 = us.t_end

    # An event series on the same domain (a few events).
    m = draw(st.integers(min_value=0, max_value=8))
    fracs = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=0.999, allow_nan=False),
                min_size=m, max_size=m,
            )
        )
    )
    ev_ts = np.array([t0 + f * duration for f in fracs], dtype=np.float64)
    ev_vals = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
                min_size=m, max_size=m,
            )
        ),
        dtype=np.float64,
    )
    es = EventSeries.from_events(ev_ts, values=ev_vals, t_start=t0, t_end=t1)

    # A segment series.
    k = draw(st.integers(min_value=0, max_value=3))
    if k > 0:
        seg_fracs = sorted(
            draw(
                st.lists(
                    st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
                    min_size=2 * k, max_size=2 * k, unique=True,
                )
            )
        )
        starts = np.array([t0 + seg_fracs[2 * i] * duration for i in range(k)])
        ends = np.array([t0 + seg_fracs[2 * i + 1] * duration for i in range(k)])
        # Filter out degenerate segments that collapsed under fp rounding.
        keep = ends > starts
        starts = starts[keep]
        ends = ends[keep]
        ss = SegmentSeries.from_segments(starts, ends, t_start=t0, t_end=t1)
    else:
        ss = SegmentSeries.from_segments(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64),
            t_start=t0, t_end=t1,
        )
    return TimeFrame(
        tracks={"audio": us, "rps": es, "vad": ss},
        t_start=t0, t_end=t1,
    )


@st.composite
def _cuts(draw, tf: TimeFrame, k: int):
    raw = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=k, max_size=k,
            )
        )
    )
    return [tf.t_start + f * (tf.t_end - tf.t_start) for f in raw]


@given(time_frame())
def test_slice_identity(tf):
    assert tf.slice(tf.t_start, tf.t_end).equal(tf)


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(time_frame(), st.data())
def test_slice_concat_no_op(tf, data):
    [a, b, c] = data.draw(_cuts(tf, 3))
    assert tf.slice(a, b).concat(tf.slice(b, c)).equal(tf.slice(a, c))


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(time_frame(), st.data())
def test_many_cuts_rejoin(tf, data):
    k = data.draw(st.integers(min_value=2, max_value=5))
    pts = [tf.t_start, *data.draw(_cuts(tf, k)), tf.t_end]
    parts = [tf.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(tf)


# ---------------------------------------------------------------------------
# Column ops
# ---------------------------------------------------------------------------

def _toy_frame(t0=0.0, dur=1.0, sr=10.0):
    n = int(round(dur * sr))
    us = UniformSeries.from_samples(np.arange(n, dtype=np.float64), sr=sr, t_start=t0)
    es = EventSeries.from_events(
        np.array([t0 + 0.1, t0 + 0.5]),
        np.array([1.0, 2.0]),
        t_start=t0, t_end=t0 + dur,
    )
    return TimeFrame(tracks={"audio": us, "rps": es}, t_start=t0, t_end=t0 + dur)


def test_select_and_drop():
    tf = _toy_frame()
    assert set(tf.select(["audio"]).keys()) == {"audio"}
    assert set(tf.drop(["rps"]).keys()) == {"audio"}


def test_select_missing_raises():
    tf = _toy_frame()
    with pytest.raises(KeyError):
        tf.select(["audio", "missing"])


def test_merge_two_frames():
    tf1 = _toy_frame()
    extra = EventSeries.from_events(
        np.array([0.2]), np.array([99.0]), t_start=0.0, t_end=1.0,
    )
    tf2 = TimeFrame(tracks={"imu": extra}, t_start=0.0, t_end=1.0)
    merged = tf1.merge(tf2)
    assert set(merged.keys()) == {"audio", "rps", "imu"}


def test_merge_key_collision_raises():
    tf1 = _toy_frame()
    tf2 = _toy_frame()
    with pytest.raises(ValueError):
        tf1.merge(tf2)


def test_merge_different_domains():
    tf1 = _toy_frame(t0=0.0)
    tf2 = _toy_frame(t0=2.0)
    merged = tf1.merge(tf2, overwrite=True)
    assert merged.t_start == pytest.approx(0.0)
    assert merged.t_end == pytest.approx(3.0)
    assert set(merged.keys()) == {"audio", "rps"}


def test_with_track_expands_domain():
    tf = _toy_frame(t0=0.0, dur=1.0)
    late = UniformSeries.from_samples(np.zeros(5), sr=10.0, t_start=5.0)
    extended = tf.with_track("late", late)
    assert extended.t_start == pytest.approx(0.0)
    assert extended.t_end == pytest.approx(5.5)
    assert "late" in extended


def test_slice_outside_domain_raises():
    tf = _toy_frame()
    with pytest.raises(DomainError):
        tf.slice(-0.5, 0.5)


def test_track_domain_need_not_match_frame():
    us = UniformSeries.from_samples(np.zeros(10), sr=10.0, t_start=0.0)  # [0, 1)
    # Frame domain larger than track — allowed.
    tf = TimeFrame(tracks={"audio": us}, t_start=0.0, t_end=2.0)
    assert tf.t_start == 0.0
    assert tf.t_end == 2.0


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------

def test_shift():
    tf = _toy_frame(t0=10.0)
    shifted = tf.shift(5.0)
    assert shifted.t_start == pytest.approx(15.0)
    assert shifted.t_end == pytest.approx(16.0)
    assert shifted["audio"].t_start == pytest.approx(15.0)
    t0, v0 = shifted["rps"][0]
    assert t0 == pytest.approx(15.1)
    assert v0 == pytest.approx(1.0)


def test_shift_roundtrip():
    tf = _toy_frame(t0=3.0)
    assert tf.shift(7.0).shift(-7.0).equal(tf)


# ---------------------------------------------------------------------------
# Concat with heterogeneous tracks / gaps
# ---------------------------------------------------------------------------

def test_concat_allows_different_keys():
    tf1 = _toy_frame(t0=0.0, dur=1.0)
    tf2 = _toy_frame(t0=1.0, dur=1.0).drop(["rps"])
    joined = tf1.concat(tf2)
    assert "audio" in joined
    assert "rps" in joined
    assert joined.t_start == pytest.approx(0.0)
    assert joined.t_end == pytest.approx(2.0)


def test_concat_with_gap():
    tf1 = _toy_frame(t0=0.0, dur=1.0)
    tf2 = _toy_frame(t0=3.0, dur=1.0)
    joined = tf1.concat(tf2)
    assert joined.t_start == pytest.approx(0.0)
    assert joined.t_end == pytest.approx(2.0)
    # Both audio tracks should be present and aligned.
    audio = joined["audio"]
    assert audio.t_start == pytest.approx(0.0)
    assert audio.t_end == pytest.approx(2.0)


def test_slice_heterogeneous_domains():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0.0)
    es = EventSeries.from_events(
        np.array([0.2, 0.5]), np.array([1.0, 2.0]), t_start=0.2, t_end=0.6,
    )
    tf = TimeFrame.from_tracks({"audio": us, "rps": es})
    # Slice across the whole hull [0.0, 1.0).
    sliced = tf.slice(0.0, 1.0)
    assert "audio" in sliced
    assert "rps" in sliced
    assert sliced["audio"].t_start == pytest.approx(0.0)
    assert sliced["rps"].t_start == pytest.approx(0.2)
