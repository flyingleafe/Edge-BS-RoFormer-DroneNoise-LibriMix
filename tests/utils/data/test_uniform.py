"""Invariants for `UniformSeries` — the trickiest of the three concrete types
because of sub-sample cuts."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from utils.data import UniformSeries
from utils.data._floats import grid_atol, tclose

from .strategies import cut_points, sample_rates, time_anchors, uniform_series


# ---------------------------------------------------------------------------
# Identity: slicing the whole domain is a no-op
# ---------------------------------------------------------------------------

@given(uniform_series())
def test_slice_identity(us: UniformSeries):
    sliced = us.slice(us.t_start, us.t_end)
    assert sliced.equal(us)


# ---------------------------------------------------------------------------
# Reported domain matches the request
# ---------------------------------------------------------------------------

@given(uniform_series(), st.data())
def test_slice_reports_exact_domain(us: UniformSeries, data):
    [a, b] = data.draw(cut_points(us, 2))
    s = us.slice(a, b)
    assert tclose(s.t_start, a)
    assert tclose(s.t_end, b)
    assert s.duration == pytest.approx(b - a, abs=1e-9)


# ---------------------------------------------------------------------------
# The big one: slice(a,b) ⊕ slice(b,c) == slice(a,c)
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(uniform_series(), st.data())
def test_slice_concat_is_no_op(us: UniformSeries, data):
    [a, b, c] = data.draw(cut_points(us, 3))
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
    pts = [us.t_start, *data.draw(cut_points(us, k)), us.t_end]
    parts = [us.slice(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.concat(p)
    assert joined.equal(us)


# ---------------------------------------------------------------------------
# Slicing at exact-sample boundary: no overlap, no duplicated sample
# ---------------------------------------------------------------------------

def test_exact_boundary_cut_no_overlap():
    us = UniformSeries.from_samples(np.arange(10, dtype=np.float64), sr=10.0, t_start=0.0)
    left = us.slice(0.0, 0.5)  # exactly on sample 5's left edge
    right = us.slice(0.5, 1.0)
    assert left.n_samples == 5
    assert right.n_samples == 5
    assert left.concat(right).equal(us)


def test_sub_sample_cut_overlaps_one_sample():
    us = UniformSeries.from_samples(np.arange(10, dtype=np.float64), sr=10.0, t_start=0.0)
    # cut at 0.43 — falls inside sample 4's cell [0.4, 0.5)
    left = us.slice(0.0, 0.43)
    right = us.slice(0.43, 1.0)
    assert left.n_samples == 5  # 0..4 inclusive
    assert right.n_samples == 6  # 4..9 inclusive
    # The shared sample is index 4 on the left and index 0 on the right.
    assert left.samples[-1] == right.samples[0] == 4.0
    assert left.concat(right).equal(us)


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------

def test_slice_outside_domain_raises():
    us = UniformSeries.from_samples(np.arange(10, dtype=np.float64), sr=10.0, t_start=0.0)
    from utils.data import DomainError
    with pytest.raises(DomainError):
        us.slice(-0.5, 0.5)
    with pytest.raises(DomainError):
        us.slice(0.0, 2.0)


# ---------------------------------------------------------------------------
# Concat rejects incompatible series
# ---------------------------------------------------------------------------

def test_concat_rejects_rate_mismatch():
    a = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0.0)
    b = UniformSeries.from_samples(np.arange(10.0), sr=20.0, t_start=1.0)
    from utils.data import IncompatibleSeriesError
    with pytest.raises(IncompatibleSeriesError):
        a.concat(b)


def test_concat_rejects_seam_mismatch():
    a = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=0.0)
    b = UniformSeries.from_samples(np.arange(10.0), sr=10.0, t_start=2.0)  # seam should be 1.0
    from utils.data import IncompatibleSeriesError
    with pytest.raises(IncompatibleSeriesError):
        a.concat(b)


# ---------------------------------------------------------------------------
# Multi-channel samples
# ---------------------------------------------------------------------------

def test_multichannel_slice_concat():
    samples = np.random.RandomState(0).randn(100, 4).astype(np.float32)
    us = UniformSeries.from_samples(samples, sr=100.0, t_start=10.0)
    a = us.slice(10.0, 10.37)
    b = us.slice(10.37, 10.83)
    c = us.slice(10.83, 11.0)
    assert a.concat(b).concat(c).equal(us)
