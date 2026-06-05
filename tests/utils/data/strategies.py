"""Hypothesis strategies for `utils.data` time series.

Strategies generate well-formed series whose declared domain matches their
content.  Cut points for slicing are drawn inside the domain.

All time is int64 ticks.  Strategies draw exact tick anchors (including
Unix-magnitude ~1.6e18 ticks) and construct series via ``from_ticks`` /
``from_events`` with int arguments to guarantee exactness.
"""
from __future__ import annotations

import numpy as np
from hypothesis import assume, strategies as st

from utils.data import EventSeries, SegmentSeries, UniformSeries

# ---- tick anchors -------------------------------------------------------
# Small magnitudes (up to ~3 hours in ns) and Unix-like (year ~2033).

_TPS = 1_000_000_000

_small_ticks = st.integers(min_value=0, max_value=10_000_000_000_000)
_large_ticks = st.integers(min_value=1_600_000_000_000_000_000,
                           max_value=1_800_000_000_000_000_000)
time_anchors = st.one_of(_small_ticks, _large_ticks)

# Sample rates plausible for audio / telemetry.
sample_rates = st.sampled_from([10.0, 100.0, 1000.0, 8000.0, 16000.0, 44100.0])


# ---- UniformSeries ------------------------------------------------------

@st.composite
def uniform_series(draw, min_n: int = 1, max_n: int = 64) -> UniformSeries:
    sr = draw(sample_rates)
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    t0 = draw(time_anchors)
    samples = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n,
        )
    )
    return UniformSeries.from_samples(
        np.asarray(samples, dtype=np.float64), sr=sr, t_start=t0,
    )


@st.composite
def cut_points(draw, ts: UniformSeries, k: int) -> list[float]:
    """`k` ordered cut points strictly inside `[ts.t_start, ts.t_end]`."""
    raw = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=k, max_size=k,
        )
    )
    raw.sort()
    return [ts.t_start + f * (ts.t_end - ts.t_start) for f in raw]


# ---- cut points in ticks (for int64 series) ----------------------------

@st.composite
def cut_points_ticks(draw, t_start: int, t_end: int, k: int) -> list[int]:
    """`k` distinct, ordered int64 tick cut points strictly inside ``[t_start, t_end)``."""
    if t_end <= t_start:
        return [t_start] * k
    # Ensure room for k distinct points.
    span = t_end - t_start
    if span < k:
        # Not enough room: return t_start repeated.  The caller should handle
        # degenerate (identical) cut points gracefully.
        return [t_start] * k
    pts = sorted(draw(
        st.lists(
            st.integers(min_value=t_start, max_value=t_end - 1),
            min_size=k, max_size=k, unique=True,
        )
    ))
    return pts


# ---- EventSeries --------------------------------------------------------

@st.composite
def event_series(draw, max_m: int = 32) -> EventSeries:
    t0 = draw(time_anchors)
    dur_ticks = draw(st.integers(min_value=1, max_value=10_000_000_000))  # ≤ 10 s
    m = draw(st.integers(min_value=0, max_value=max_m))
    if m == 0:
        return EventSeries.from_ticks(
            np.array([], dtype=np.int64), values=None, t_start=t0, dur=dur_ticks,
        )
    # Place events strictly inside the domain as exact ticks.
    assume(dur_ticks > m)
    raw_ts = sorted(draw(
        st.lists(
            st.integers(min_value=0, max_value=dur_ticks - 1),
            min_size=m, max_size=m, unique=True,
        )
    ))
    timestamps = np.array(raw_ts, dtype=np.int64)  # relative to t0
    vals = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
                min_size=m, max_size=m,
            )
        ),
        dtype=np.float64,
    )
    return EventSeries.from_ticks(
        timestamps=timestamps, values=vals, t_start=t0, dur=dur_ticks,
    )


# ---- SegmentSeries ------------------------------------------------------

@st.composite
def segment_series(draw, max_m: int = 8) -> SegmentSeries:
    t0 = draw(time_anchors)
    dur_ticks = draw(st.integers(min_value=100, max_value=10_000_000_000))  # ≥ 100 ns
    m = draw(st.integers(min_value=0, max_value=max_m))
    if m == 0:
        return SegmentSeries.from_segments(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64),
            t_start=t0, t_end=t0 + dur_ticks,
        )
    # Disjoint segments strictly inside [0, dur_ticks).
    # Pick 2*m sorted int ticks, take pairs as (start, end).
    assume(dur_ticks >= 2 * m + 1)
    cuts = sorted(draw(
        st.lists(
            st.integers(min_value=1, max_value=dur_ticks - 1),
            min_size=2 * m, max_size=2 * m, unique=True,
        )
    ))
    starts = np.array(cuts[0::2], dtype=np.int64) + t0
    ends = np.array(cuts[1::2], dtype=np.int64) + t0
    return SegmentSeries.from_segments(
        starts, ends, t_start=t0, t_end=t0 + dur_ticks,
    )
