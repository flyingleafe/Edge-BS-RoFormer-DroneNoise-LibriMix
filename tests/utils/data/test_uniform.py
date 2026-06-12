"""Invariants for `UniformSeries` — the trickiest of the three concrete types
because of sub-sample cuts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.data import DomainError, IncompatibleSeriesError, UniformSeries

from .strategies import cut_points_ticks, uniform_series

# ---------------------------------------------------------------------------
# Identity: slicing the whole domain is a no-op
# ---------------------------------------------------------------------------


@given(uniform_series())
def test_slice_identity(us: UniformSeries):
    sliced = us.slice(us.t_start_ticks, us.t_end_ticks)
    assert sliced.equal(us)


# ---------------------------------------------------------------------------
# Reported domain matches the request
# ---------------------------------------------------------------------------


@given(uniform_series(), st.data())
def test_slice_reports_exact_domain(us: UniformSeries, data):
    [a, b] = data.draw(_cuts_ticks(us, 2))
    s = us.slice(a, b)
    assert s.t_start_ticks == a
    assert s.t_end_ticks == b
    assert s.dur_ticks == b - a


# ---------------------------------------------------------------------------
# The big one: slice(a,b) ⊕ slice(b,c) == slice(a,c)
# ---------------------------------------------------------------------------


@st.composite
def _cuts_ticks(draw, us: UniformSeries, k: int) -> list[int]:
    return draw(cut_points_ticks(us.t_start_ticks, us.t_end_ticks, k))


@settings(max_examples=200)
@given(uniform_series(), st.data())
def test_slice_concat_is_no_op(us: UniformSeries, data):
    [a, b, c] = data.draw(_cuts_ticks(us, 3))
    left = us.slice(a, b)
    right = us.slice(b, c)
    whole = us.slice(a, c)
    joined = left.concat(right)
    assert joined.equal(whole)


# Multiple consecutive cuts must still rejoin to the original.
@settings(max_examples=100)
@given(uniform_series(min_n=4), st.data())
def test_many_slices_concat_no_op(us: UniformSeries, data):
    k = data.draw(st.integers(min_value=2, max_value=6))
    pts = [us.t_start_ticks, *data.draw(_cuts_ticks(us, k)), us.t_end_ticks]
    parts = [us.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(us)


# ---------------------------------------------------------------------------
# Slicing at exact-sample boundary: no overlap, no duplicated sample
# ---------------------------------------------------------------------------


def test_exact_boundary_cut_no_overlap():
    us = UniformSeries.from_samples(np.arange(10, dtype=np.float64), sr=10.0, t_start=0)
    left = us.slice(0, 500_000_000)  # exactly on sample 5's left edge at 0.5s
    right = us.slice(500_000_000, 1_000_000_000)
    assert left.n_samples == 5
    assert right.n_samples == 5
    assert left.concat(right).equal(us)


def test_sub_sample_cut_overlaps_one_sample():
    us = UniformSeries.from_samples(np.arange(10, dtype=np.float64), sr=10.0, t_start=0)
    # cut at 0.43s = 430_000_000 ticks — falls inside sample 4's cell [0.4, 0.5)
    left = us.slice(0, 430_000_000)
    right = us.slice(430_000_000, 1_000_000_000)
    assert left.n_samples == 5  # 0..4 inclusive
    assert right.n_samples == 6  # 4..9 inclusive
    # The shared sample is index 4 on the left and index 0 on the right.
    assert left.samples[-1] == right.samples[0] == 4.0
    assert left.concat(right).equal(us)


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


def test_slice_outside_domain_raises():
    us = UniformSeries.from_samples(np.arange(10, dtype=np.float64), sr=10.0, t_start=0)
    from utils.data import DomainError

    with pytest.raises(DomainError):
        us.slice(-500_000_000, 500_000_000)
    with pytest.raises(DomainError):
        us.slice(0, 2_000_000_000)


# ---------------------------------------------------------------------------
# Concat rejects incompatible series
# ---------------------------------------------------------------------------


def test_concat_rejects_rate_mismatch():
    a = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    b = UniformSeries.from_samples(np.arange(10.0), sr=20.0, t_start=1_000_000_000)
    from utils.data import IncompatibleSeriesError

    with pytest.raises(IncompatibleSeriesError):
        a.concat(b)


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------


def test_shift_preserves_samples():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    shifted = us.shift(5_000_000_000)
    assert shifted.t_start_ticks == 5_000_000_000
    assert shifted.t_end_ticks == 6_000_000_000
    assert shifted.t_first_edge_ticks == 5_000_000_000
    assert np.array_equal(shifted.samples, us.samples)


def test_shift_roundtrip():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=3_000_000_000)
    assert us.shift(7_000_000_000).shift(-7_000_000_000).equal(us)


# ---------------------------------------------------------------------------
# Concat with auto-shift (gap allowed)
# ---------------------------------------------------------------------------


def test_concat_across_gap():
    a = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    b = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=2_000_000_000)
    joined = a.concat(b)
    assert joined.t_start_ticks == 0
    assert joined.t_end_ticks == 1_500_000_000
    assert joined.n_samples == 15


# ---------------------------------------------------------------------------
# Multi-channel samples
# ---------------------------------------------------------------------------


def test_multichannel_slice_concat():
    samples = np.random.RandomState(0).randn(4, 100).astype(np.float32)
    us = UniformSeries.from_samples(samples, sr=100.0, t_start=10_000_000_000)
    a = us.slice(10_000_000_000, 10_370_000_000)
    b = us.slice(10_370_000_000, 10_830_000_000)
    c = us.slice(10_830_000_000, 11_000_000_000)
    assert a.concat(b).concat(c).equal(us)


# ── Interpolation / resampling ───────────────────────────────────────────


def test_interpolate_at_sample_points_1d():
    us = UniformSeries.from_samples(np.array([1.0, 3.0, 2.0, 0.0]), sr=4.0, t_start=0)
    t = us.sample_times()
    vals = us.interpolate(t)
    np.testing.assert_allclose(vals, us.samples, atol=1e-12)


def test_interpolate_midpoint():
    us = UniformSeries.from_samples(np.array([0.0, 4.0]), sr=2.0, t_start=0)
    v = us.interpolate(np.array([0.25]))[0]
    np.testing.assert_allclose(v, 2.0, atol=1e-12)


def test_interpolate_multichannel():
    vals = np.array([[0.0, 2.0, 4.0], [10.0, 8.0, 6.0]])  # (2, 3)
    us = UniformSeries.from_samples(vals, sr=3.0, t_start=0)
    result = us.interpolate(np.array([0.5]))
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result[0, 0], 3.0, atol=1e-12)
    np.testing.assert_allclose(result[1, 0], 7.0, atol=1e-12)


def test_interpolate_clamp_extrap():
    us = UniformSeries.from_samples(np.array([5.0, 3.0]), sr=2.0, t_start=0)
    v_left = us.interpolate(np.array([-0.5]))[0]
    v_right = us.interpolate(np.array([1.5]))[0]
    assert v_left == 5.0
    assert v_right == 3.0


def test_interpolate_nan_extrap():
    us = UniformSeries.from_samples(np.array([5.0, 3.0]), sr=2.0, t_start=0)
    v = us.interpolate(np.array([-0.5]), fill="nan")[0]
    assert np.isnan(v)


def test_resample_same_rate_approx_identity():
    us = UniformSeries.from_samples(np.sin(np.linspace(0, 2 * np.pi, 128)), sr=128.0, t_start=0)
    rs = us.resample(128.0)
    # Same rate, phase=0 → should be close.
    np.testing.assert_allclose(rs.samples, us.samples, atol=0.05)


def test_resample_half_rate():
    us = UniformSeries.from_samples(np.arange(10.0, dtype=np.float64), sr=10.0, t_start=0)
    rs = us.resample(5.0)
    assert rs.sr == 5.0
    assert rs.n_samples == 5


def test_resample_phase_zero():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=1.0)
    rs = us.resample(20.0)
    # phase=0 means rs.t_start ≈ us.t_start
    np.testing.assert_allclose(rs.t_first_edge, us.t_start, atol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════


def test_init_rejects_0d_samples():
    with pytest.raises(ValueError, match="at least one axis"):
        UniformSeries(samples=np.float64(5.0), sr=10.0, t_start_ticks=0, dur_ticks=100, phase=0.0)  # type: ignore[arg-type]


def test_init_rejects_nonpositive_sr():
    with pytest.raises(ValueError, match="sr must be > 0"):
        UniformSeries(samples=np.arange(5.0), sr=0.0, t_start_ticks=0, dur_ticks=100, phase=0.0)


def test_init_rejects_phase_out_of_range():
    with pytest.raises(ValueError, match="phase"):
        UniformSeries(samples=np.arange(5.0), sr=10.0, t_start_ticks=0, dur_ticks=100, phase=1.5)


# ═══════════════════════════════════════════════════════════════════════════
# Tool methods
# ═══════════════════════════════════════════════════════════════════════════


def test_timestamps_returns_float_seconds():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    ts = us.timestamps
    assert isinstance(ts[0], (float, np.floating))


def test_timestamp_ticks_returns_int64():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    tts = us.timestamp_ticks
    assert tts.dtype == np.int64


def test_channel_shape_for_mono():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    assert us.channel_shape == ()


def test_channel_shape_for_stereo():
    samples = np.random.randn(2, 10).astype(np.float64)
    us = UniformSeries.from_samples(samples, sr=10.0, t_start=0)
    assert us.channel_shape == (2,)


def test_values_is_samples():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    assert np.array_equal(us.values, us.samples)


def test_getitem_returns_along_last_axis():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    assert us[3] == 3.0


def test_sample_times_ticks_returns_int64():
    us = UniformSeries.from_samples(np.arange(5.0), sr=10.0, t_start=0)
    assert us.sample_times_ticks().dtype == np.int64


def test_time_to_index_integer_sr():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    assert us.time_to_index(0.0) == 0
    assert us.time_to_index(0.35) == 3


def test_time_to_index_non_integer_sr():
    us = UniformSeries.from_samples(np.arange(100.0), sr=44.1, t_start=0)
    idx = us.time_to_index(0.5)
    assert 20 <= idx <= 24  # ~22.05 samples at 0.5s


# ═══════════════════════════════════════════════════════════════════════════
# Concat incompatible offset paths
# ═══════════════════════════════════════════════════════════════════════════


def test_concat_rejects_incompatible_grid_float_offset():
    # b has a sample grid offset that doesn't align with a's grid
    a = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    # b starts exactly at a.t_end but with phase=-0.3 (misaligned sample grid)
    b = UniformSeries(
        samples=np.arange(5.0),
        sr=10.0,
        t_start_ticks=1_000_000_000,
        dur_ticks=500_000_000,
        phase=-0.3,
    )
    # concat alignment: k_offset = 10 + 0 + (-0.3) - 0 = 9.7, round to 10
    # |9.7-10|=0.3 > 0.1 → IncompatibleSeriesError
    with pytest.raises(IncompatibleSeriesError, match="incompatible sample grids"):
        a.concat(b)


# ═══════════════════════════════════════════════════════════════════════════
# Interpolation error paths
# ═══════════════════════════════════════════════════════════════════════════


def test_interpolate_with_int_query_times():
    us = UniformSeries.from_samples(np.array([0.0, 2.0, 4.0]), sr=3.0, t_start=0)
    vals = us.interpolate(np.array([500_000_000], dtype=np.int64))
    # 0.5s: between 2.0@0.333s and 4.0@0.666s → linear interpolation = 3.0
    np.testing.assert_allclose(vals[0], 3.0, atol=1e-6)


def test_interpolate_empty_series_fill_error():
    us = UniformSeries.from_samples(np.array([], dtype=np.float64), sr=10.0, t_start=0)
    with pytest.raises(DomainError, match="empty"):
        us.interpolate(np.array([0.5]), fill="error")


def test_interpolate_empty_series_fill_nan():
    us = UniformSeries.from_samples(np.array([], dtype=np.float64), sr=10.0, t_start=0)
    vals = us.interpolate(np.array([0.5]), fill="nan")
    assert np.isnan(vals[0])


def test_interpolate_unsupported_kind():
    us = UniformSeries.from_samples(np.array([0.0, 2.0]), sr=2.0, t_start=0)
    with pytest.raises(ValueError, match="unsupported interpolation kind"):
        us.interpolate(np.array([0.5]), kind="cubic")


def test_interpolate_fill_nan_extrapolation_multichannel():
    vals = np.array([[0.0, 2.0, 4.0], [10.0, 8.0, 6.0]])
    us = UniformSeries.from_samples(vals, sr=3.0, t_start=0)
    result = us.interpolate(np.array([-0.5, 1.5]), fill="nan")
    assert result.shape == (2, 2)
    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 0])
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 1])


def test_interpolate_fill_error_extrapolation():
    us = UniformSeries.from_samples(np.array([5.0, 3.0]), sr=2.0, t_start=0)
    with pytest.raises(DomainError, match="outside data span"):
        us.interpolate(np.array([-0.5, 1.5]), fill="error")


def test_interpolate_fill_error_within_span():
    us = UniformSeries.from_samples(np.array([5.0, 3.0]), sr=2.0, t_start=0)
    # grid_t = [0.0, 0.5], queries at [0.1, 0.4] are within span
    vals = us.interpolate(np.array([0.1, 0.4]), fill="error")
    assert len(vals) == 2


def test_interpolate_unsupported_fill():
    us = UniformSeries.from_samples(np.array([5.0, 3.0]), sr=2.0, t_start=0)
    with pytest.raises(ValueError, match="unsupported fill"):
        us.interpolate(np.array([0.5]), fill="extrap")


def test_resample_rejects_nonpositive_sr():
    us = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0)
    with pytest.raises(ValueError, match="new_sr must be > 0"):
        us.resample(0.0)
