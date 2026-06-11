"""Invariants for `TimeFrame` — exact int64 tick storage."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from utils.data import (
    DomainError,
    EventSeries,
    IncompatibleSeriesError,
    SegmentSeries,
    TimeFrame,
    UniformSeries,
)

from .strategies import cut_points_ticks, sample_rates, time_anchors

_TPS = 1_000_000_000


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
                min_size=n,
                max_size=n,
            )
        ),
        dtype=np.float64,
    )
    us = UniformSeries.from_samples(samples, sr=sr, t_start=t0)
    dur_ticks = us.dur_ticks
    t1_ticks = us.t_end_ticks

    m = draw(st.integers(min_value=0, max_value=8))
    if m == 0:
        es = EventSeries.from_ticks(
            np.array([], dtype=np.int64),
            values=None,
            t_start=t0,
            dur=dur_ticks,
        )
    else:
        fracs = sorted(
            draw(
                st.lists(
                    st.floats(min_value=0.0, max_value=0.999, allow_nan=False),
                    min_size=m,
                    max_size=m,
                )
            )
        )
        ev_ts_ticks = np.array(
            [t0 + int(f * dur_ticks) for f in fracs],
            dtype=np.int64,
        )
        ev_vals = np.asarray(
            draw(
                st.lists(
                    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
                    min_size=m,
                    max_size=m,
                )
            ),
            dtype=np.float64,
        )
        es = EventSeries.from_events(ev_ts_ticks, values=ev_vals, t_start=t0, t_end=t1_ticks)

    k = draw(st.integers(min_value=0, max_value=3))
    if k > 0:
        seg_fracs = sorted(
            draw(
                st.lists(
                    st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
                    min_size=2 * k,
                    max_size=2 * k,
                    unique=True,
                )
            )
        )
        starts = np.array(
            [t0 + int(seg_fracs[2 * i] * dur_ticks) for i in range(k)],
            dtype=np.int64,
        )
        ends = np.array(
            [t0 + int(seg_fracs[2 * i + 1] * dur_ticks) for i in range(k)],
            dtype=np.int64,
        )
        keep = ends > starts
        starts = starts[keep]
        ends = ends[keep]
        ss = SegmentSeries.from_segments(starts, ends, t_start=t0, t_end=t1_ticks)
    else:
        ss = SegmentSeries.from_segments(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            t_start=t0,
            t_end=t1_ticks,
        )
    return TimeFrame(
        tracks={"audio": us, "rps": es, "vad": ss},
        t_start_ticks=t0,
        dur_ticks=dur_ticks,
    )


@st.composite
def _cuts(draw, tf: TimeFrame, k: int):
    return draw(cut_points_ticks(tf.t_start_ticks, tf.t_end_ticks, k))


# ---------------------------------------------------------------------------
# Slice/concat identity
# ---------------------------------------------------------------------------


@given(time_frame())
def test_slice_identity(tf):
    assert tf.slice(tf.t_start_ticks, tf.t_end_ticks).equal(tf)


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(time_frame(), st.data())
def test_slice_concat_no_op(tf, data):
    [a, b, c] = data.draw(_cuts(tf, 3))
    assert tf.slice(a, b).concat(tf.slice(b, c)).equal(tf.slice(a, c))


@given(time_frame())
def test_slice_concat_across_full_domain(tf):
    """Explicit boundary invariant: slice(t0,t1).concat(slice(t1,t2)) == slice(t0,t2)."""
    t0 = tf.t_start_ticks
    t2 = tf.t_end_ticks
    t1 = (t0 + t2) // 2
    assert tf.slice(t0, t1).concat(tf.slice(t1, t2)).equal(tf.slice(t0, t2))


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(time_frame(), st.data())
def test_many_cuts_rejoin(tf, data):
    k = data.draw(st.integers(min_value=2, max_value=5))
    pts = [tf.t_start_ticks, *data.draw(_cuts(tf, k)), tf.t_end_ticks]
    parts = [tf.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(tf)


# ---------------------------------------------------------------------------
# Column ops
# ---------------------------------------------------------------------------


def _toy_frame(t0=0, dur=1_000_000_000, sr=10.0):
    n = int(round(dur * sr / _TPS))
    us = UniformSeries.from_samples(np.arange(n, dtype=np.float64), sr=sr, t_start=t0)
    es = EventSeries.from_events(
        np.array([t0 + 100_000_000, t0 + 500_000_000], dtype=np.int64),
        np.array([1.0, 2.0]),
        t_start=t0,
        t_end=t0 + dur,
    )
    return TimeFrame(tracks={"audio": us, "rps": es}, t_start_ticks=t0, dur_ticks=dur)


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
        np.array([200_000_000], dtype=np.int64),
        np.array([99.0]),
        t_start=0,
        t_end=1_000_000_000,
    )
    tf2 = TimeFrame(tracks={"imu": extra}, t_start_ticks=0, dur_ticks=1_000_000_000)
    merged = tf1.merge(tf2)
    assert set(merged.keys()) == {"audio", "rps", "imu"}


def test_merge_key_collision_raises():
    tf1 = _toy_frame()
    tf2 = _toy_frame()
    with pytest.raises(ValueError):
        tf1.merge(tf2)


def test_merge_different_domains():
    tf1 = _toy_frame(t0=0)
    tf2 = _toy_frame(t0=2_000_000_000)
    merged = tf1.merge(tf2, overwrite=True)
    assert merged.t_start_ticks == 0
    assert merged.t_end_ticks == 3_000_000_000
    assert set(merged.keys()) == {"audio", "rps"}


def test_with_track_expands_domain():
    tf = _toy_frame(t0=0, dur=1_000_000_000)
    late = UniformSeries.from_samples(np.zeros(5), sr=10.0, t_start=5_000_000_000)
    extended = tf.with_track("late", late)
    assert extended.t_start_ticks == 0
    assert extended.t_end_ticks == 5_500_000_000
    assert "late" in extended


def test_slice_outside_domain_raises():
    tf = _toy_frame()
    with pytest.raises(DomainError):
        tf.slice(-500_000_000, 500_000_000)


def test_track_domain_need_not_match_frame():
    us = UniformSeries.from_samples(np.zeros(10), sr=10.0, t_start=0)
    tf = TimeFrame(tracks={"audio": us}, t_start_ticks=0, dur_ticks=2_000_000_000)
    assert tf.t_start_ticks == 0
    assert tf.t_end_ticks == 2_000_000_000


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------


def test_shift():
    tf = _toy_frame(t0=10_000_000_000)
    shifted = tf.shift(5_000_000_000)
    assert shifted.t_start_ticks == 15_000_000_000
    assert shifted.t_end_ticks == 16_000_000_000
    assert cast(Any, shifted["audio"]).t_start_ticks == 15_000_000_000
    t0, v0 = shifted["rps"][0]  # pyright: ignore[reportIndexIssue]
    assert t0 == pytest.approx(15.1)
    assert v0 == pytest.approx(1.0)


def test_shift_roundtrip():
    tf = _toy_frame(t0=3_000_000_000)
    assert tf.shift(7_000_000_000).shift(-7_000_000_000).equal(tf)


# ---------------------------------------------------------------------------
# Concat with heterogeneous tracks / gaps
# ---------------------------------------------------------------------------


def test_concat_allows_different_keys():
    tf1 = _toy_frame(t0=0, dur=1_000_000_000)
    tf2 = _toy_frame(t0=1_000_000_000, dur=1_000_000_000).drop(["rps"])
    joined = tf1.concat(tf2)
    assert "audio" in joined
    assert "rps" in joined
    assert joined.t_start_ticks == 0
    assert joined.t_end_ticks == 2_000_000_000


def test_concat_with_gap():
    tf1 = _toy_frame(t0=0, dur=1_000_000_000)
    tf2 = _toy_frame(t0=3_000_000_000, dur=1_000_000_000)
    joined = tf1.concat(tf2)
    assert joined.t_start_ticks == 0
    assert joined.t_end_ticks == 2_000_000_000
    audio = cast(Any, joined["audio"])
    assert audio.t_start_ticks == 0
    assert audio.t_end_ticks == 2_000_000_000


def test_slice_heterogeneous_domains():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    es = EventSeries.from_events(
        np.array([200_000_000, 500_000_000], dtype=np.int64),
        np.array([1.0, 2.0]),
        t_start=200_000_000,
        t_end=600_000_000,
    )
    tf = TimeFrame.from_tracks({"audio": us, "rps": es})
    sliced = tf.slice(0, 1_000_000_000)
    assert "audio" in sliced
    assert "rps" in sliced
    assert cast(Any, sliced["audio"]).t_start_ticks == 0
    assert cast(Any, sliced["rps"]).t_start_ticks == 200_000_000


# ── TimeFrame.tags ────────────────────────────────────────────────────────


def test_tags_default():
    tf = TimeFrame()
    assert tf.tags == {}


def test_tags_preserved_on_slice():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    tf = TimeFrame.from_tracks({"audio": us}, tags={"id": "s42", "snr": -10.0})
    s = tf.slice(0, 0.5)
    assert dict(s.tags) == {"id": "s42", "snr": -10.0}


def test_tags_preserved_on_shift():
    tf = TimeFrame(tags={"id": "s1"})
    assert dict(tf.shift(1.0).tags) == {"id": "s1"}


def test_tags_preserved_on_select():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    tf = TimeFrame.from_tracks({"audio": us, "rps": us}, tags={"id": "s1"})
    assert dict(tf.select(["audio"]).tags) == {"id": "s1"}


def test_tags_preserved_on_with_track():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    tf = TimeFrame.from_tracks({"audio": us}, tags={"id": "s1"})
    tf2 = tf.with_track("rps", us)
    assert dict(tf2.tags) == {"id": "s1"}


def test_tags_concat_disjoint_union():
    us1 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    us2 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=1.0)
    tf1 = TimeFrame.from_tracks({"audio": us1}, tags={"id": "a"})
    tf2 = TimeFrame.from_tracks({"audio": us2}, tags={"snr": -5.0})
    joined = tf1.concat(tf2)
    assert dict(joined.tags) == {"id": "a", "snr": -5.0}


def test_tags_concat_equal_shared_keys():
    us1 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    us2 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0.5)
    tf1 = TimeFrame.from_tracks({"audio": us1}, tags={"id": "x", "v": 1})
    tf2 = TimeFrame.from_tracks({"audio": us2}, tags={"id": "x"})
    joined = tf1.concat(tf2)
    assert dict(joined.tags) == {"id": "x", "v": 1}


def test_tags_concat_conflict_raises():
    us1 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    us2 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0.5)
    tf1 = TimeFrame.from_tracks({"audio": us1}, tags={"id": "a"})
    tf2 = TimeFrame.from_tracks({"audio": us2}, tags={"id": "b"})
    with pytest.raises(IncompatibleSeriesError, match="tag 'id' conflict"):
        tf1.concat(tf2)


def test_tags_merge_preserves():
    us1 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    us2 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us1}, tags={"id": "s1"})
    tf2 = TimeFrame.from_tracks({"rps": us2}, tags={"snr": -10.0})
    merged = tf1.merge(tf2)
    assert dict(merged.tags) == {"id": "s1", "snr": -10.0}


def test_tags_merge_conflict_raises():
    us1 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    us2 = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us1}, tags={"id": "a"})
    tf2 = TimeFrame.from_tracks({"rps": us2}, tags={"id": "b"})
    with pytest.raises(IncompatibleSeriesError, match="tag 'id' conflict"):
        tf1.merge(tf2)


# ═══════════════════════════════════════════════════════════════════════════
# __post_init__ validation + normalisation
# ═══════════════════════════════════════════════════════════════════════════


def test_init_rejects_negative_dur_ticks():
    with pytest.raises(ValueError, match="dur_ticks"):
        TimeFrame(t_start_ticks=0, dur_ticks=-1)


def test_init_rejects_non_timeseries_track():
    with pytest.raises(TypeError, match="not a TimeSeries"):
        TimeFrame(tracks={"bad": 42}, t_start_ticks=0, dur_ticks=100)  # type: ignore[arg-type]


def test_init_rejects_t_start_after_hull_start():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    with pytest.raises(ValueError, match="after hull start"):
        TimeFrame(tracks={"audio": us}, t_start_ticks=1_000_000_000, dur_ticks=100)


def test_init_rejects_t_end_before_hull_end():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    with pytest.raises(ValueError, match="before hull end"):
        TimeFrame(tracks={"audio": us}, t_start_ticks=0, dur_ticks=100)


def test_init_normalises_none_tags_to_empty_dict():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf = TimeFrame(tracks={"audio": us}, t_start_ticks=0, dur_ticks=500_000_000, tags=None)  # type: ignore[arg-type]
    assert tf.tags == {}


def test_init_normalises_none_global_data_to_empty_dict():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf = TimeFrame(tracks={"audio": us}, t_start_ticks=0, dur_ticks=500_000_000, global_data=None)  # type: ignore[arg-type]
    assert tf.global_data == {}


# ═══════════════════════════════════════════════════════════════════════════
# from_tracks error path
# ═══════════════════════════════════════════════════════════════════════════


def test_from_tracks_empty_without_domain_raises():
    with pytest.raises(ValueError, match="cannot infer domain"):
        TimeFrame.from_tracks({})


# ═══════════════════════════════════════════════════════════════════════════
# Dict-like protocol
# ═══════════════════════════════════════════════════════════════════════════


def test_len_is_track_count():
    tf = _toy_frame()
    assert len(tf) == 2


def test_iter_yields_keys():
    tf = _toy_frame()
    assert set(tf) == {"audio", "rps"}


def test_values_returns_abs_tracks():
    tf = _toy_frame(t0=1_000_000_000)
    for tr in tf.values():
        assert cast(Any, tr).t_start_ticks >= 1_000_000_000


def test_items_returns_abs_tracks():
    tf = _toy_frame(t0=1_000_000_000)
    for name, tr in tf.items():
        assert name in {"audio", "rps"}
        assert cast(Any, tr).t_start_ticks >= 1_000_000_000


def test_contains_key():
    tf = _toy_frame()
    assert "audio" in tf
    assert "missing" not in tf


# ═══════════════════════════════════════════════════════════════════════════
# Domain properties
# ═══════════════════════════════════════════════════════════════════════════


def test_t_start_is_float_seconds():
    tf = _toy_frame(t0=1_000_000_000)
    assert tf.t_start == pytest.approx(1.0)


def test_t_end_is_float_seconds():
    tf = _toy_frame(t0=1_000_000_000, dur=2_000_000_000)
    assert tf.t_end == pytest.approx(3.0)


def test_duration_is_float_seconds():
    tf = _toy_frame(dur=2_000_000_000)
    assert tf.duration == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# global_data semantics
# ═══════════════════════════════════════════════════════════════════════════


def test_merge_global_data_union():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us}, global_data={"mic": np.array([1.0])})
    tf2 = TimeFrame.from_tracks({"rps": us}, global_data={"rotor": np.array([2.0])})
    merged = tf1.merge(tf2)
    assert "mic" in merged.global_data
    assert "rotor" in merged.global_data


def test_merge_global_data_equal_shared_keys():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    arr = np.array([1.0, 2.0])
    tf1 = TimeFrame.from_tracks({"audio": us}, global_data={"pos": arr})
    tf2 = TimeFrame.from_tracks({"rps": us}, global_data={"pos": arr.copy()})
    merged = tf1.merge(tf2)
    assert np.array_equal(merged.global_data["pos"], arr)


def test_merge_global_data_conflict_raises():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us}, global_data={"pos": np.array([1.0])})
    tf2 = TimeFrame.from_tracks({"rps": us}, global_data={"pos": np.array([2.0])})
    with pytest.raises(IncompatibleSeriesError, match="global_data"):
        tf1.merge(tf2)


def test_concat_global_data_union():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us}, global_data={"a": np.array([1])})
    tf2 = TimeFrame.from_tracks(
        {"audio": UniformSeries.from_samples(np.arange(3.0), sr=10.0, t_start=1.0)},
        global_data={"b": np.array([2])},
    )
    joined = tf1.concat(tf2)
    assert "a" in joined.global_data
    assert "b" in joined.global_data


def test_concat_global_data_conflict_raises():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us}, global_data={"x": np.array([1])})
    tf2 = TimeFrame.from_tracks(
        {"audio": UniformSeries.from_samples(np.arange(3.0), sr=10.0, t_start=1.0)},
        global_data={"x": np.array([2])},
    )
    with pytest.raises(IncompatibleSeriesError, match="global_data"):
        tf1.concat(tf2)


# ═══════════════════════════════════════════════════════════════════════════
# concat missing-track path
# ═══════════════════════════════════════════════════════════════════════════


def test_concat_other_has_track_self_does_not():
    tf1 = _toy_frame(t0=0, dur=1_000_000_000).drop(["rps"])
    extra = EventSeries.from_events(
        np.array([200_000_000], dtype=np.int64),
        np.array([99.0]),
        t_start=1_000_000_000,
        t_end=2_000_000_000,
    )
    tf2 = TimeFrame.from_tracks({"imu": extra})
    joined = tf1.concat(tf2)
    assert "audio" in joined
    assert "imu" in joined


# ═══════════════════════════════════════════════════════════════════════════
# shift zero-delta fast path
# ═══════════════════════════════════════════════════════════════════════════


def test_shift_zero_is_identity():
    tf = _toy_frame()
    assert tf.shift(0).equal(tf)


# ═══════════════════════════════════════════════════════════════════════════
# Dunders + equal
# ═══════════════════════════════════════════════════════════════════════════


def test_add_is_concat():
    tf1 = _toy_frame(t0=0, dur=1_000_000_000)
    tf2 = _toy_frame(t0=1_000_000_000, dur=1_000_000_000)
    assert (tf1 + tf2).equal(tf1.concat(tf2))


def test_eq_non_timeframe_returns_notimplemented():
    tf = _toy_frame()
    assert (tf.__eq__(42)) is NotImplemented


def test_eq_delegates_to_equal():
    tf1 = _toy_frame()
    tf2 = _toy_frame()
    assert (tf1 == tf2) == tf1.equal(tf2)


def test_hash_is_id():
    tf = _toy_frame()
    assert hash(tf) == id(tf)


def test_equal_different_t_start():
    tf1 = _toy_frame(t0=0)
    tf2 = _toy_frame(t0=1_000_000_000)
    assert not tf1.equal(tf2)


def test_equal_different_track_set():
    tf1 = _toy_frame()
    tf2 = _toy_frame().drop(["rps"])
    assert not tf1.equal(tf2)


def test_equal_different_tags():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us}, tags={"id": "a"})
    tf2 = TimeFrame.from_tracks({"audio": us}, tags={"id": "b"})
    assert not tf1.equal(tf2)


def test_equal_different_global_data():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tf1 = TimeFrame.from_tracks({"audio": us}, global_data={"x": np.array([1])})
    tf2 = TimeFrame.from_tracks({"audio": us}, global_data={"x": np.array([2])})
    assert not tf1.equal(tf2)


def test_global_leaf_equal_numpy_arrays():
    from utils.data.frame import _global_leaf_equal

    assert _global_leaf_equal(np.array([1, 2]), np.array([1, 2])) is True
    assert _global_leaf_equal(np.array([1, 2]), np.array([1, 3])) is False


def test_global_leaf_equal_different_shapes():
    from utils.data.frame import _global_leaf_equal

    assert _global_leaf_equal(np.array([1, 2]), np.array([1, 2, 3])) is False
