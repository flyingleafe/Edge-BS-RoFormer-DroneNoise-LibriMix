"""Hypothesis strategies for `utils.data` time series.

Strategies generate well-formed series whose declared domain matches their
content. Cut points for slicing are drawn inside the domain.
"""
from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from utils.data import EventSeries, SegmentSeries, UniformSeries

# Time anchors. Cover small relative-time numbers and "Unix-like" magnitudes
# to flush out float-precision bugs.
finite_floats = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64
)
unix_like = st.floats(
    min_value=1.6e9, max_value=1.7e9, allow_nan=False, allow_infinity=False, width=64
)
time_anchors = st.one_of(finite_floats, unix_like)

# Sample rates plausible for audio / telemetry.
sample_rates = st.sampled_from([10.0, 100.0, 1000.0, 8000.0, 16000.0, 44100.0])


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
    return UniformSeries.from_samples(np.asarray(samples, dtype=np.float64), sr=sr, t_start=t0)


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


@st.composite
def event_series(draw, max_m: int = 32) -> EventSeries:
    t_start = draw(time_anchors)
    dur = draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False))
    t_end = t_start + dur
    m = draw(st.integers(min_value=0, max_value=max_m))
    if m == 0:
        return EventSeries.from_events(
            np.array([], dtype=np.float64), values=None, t_start=t_start, t_end=t_end,
        )
    # Place events strictly inside the domain.
    fracs = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=0.999999, allow_nan=False),
                min_size=m, max_size=m,
            )
        )
    )
    ts = np.array([t_start + f * dur for f in fracs], dtype=np.float64)
    vals = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
                min_size=m, max_size=m,
            )
        ),
        dtype=np.float64,
    )
    return EventSeries(timestamps=ts, values=vals, t_start=t_start, t_end=t_end)


@st.composite
def segment_series(draw, max_m: int = 8) -> SegmentSeries:
    t_start = draw(time_anchors)
    dur = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False))
    t_end = t_start + dur
    m = draw(st.integers(min_value=0, max_value=max_m))
    if m == 0:
        return SegmentSeries.from_segments(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64),
            t_start=t_start, t_end=t_end,
        )
    # Disjoint segments inside [t_start, t_end].
    # Pick 2m sorted fracs, take pairs as (start, end).
    fracs = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
                min_size=2 * m, max_size=2 * m, unique=True,
            )
        )
    )
    starts = np.array([t_start + fracs[2 * i] * dur for i in range(m)])
    ends = np.array([t_start + fracs[2 * i + 1] * dur for i in range(m)])
    # Skip the degenerate case where rounding makes a zero-width segment.
    keep = ends > starts
    starts = starts[keep]
    ends = ends[keep]
    return SegmentSeries.from_segments(starts, ends, t_start=t_start, t_end=t_end)
