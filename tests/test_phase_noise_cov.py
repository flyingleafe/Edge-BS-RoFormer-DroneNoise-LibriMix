"""WP18 — the rank-one-plus-diagonal moment estimator.

The measurement it backs (``scripts/phase_noise_cov/``) claims to separate a
harmonic-common term ``sigma_J^2`` from per-harmonic terms ``v_k`` out of one
K x K covariance.  These tests construct data where both are known.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "phase_noise_cov"))
sys.path.insert(0, str(REPO / "src"))

import estimate as E  # noqa: E402


def _make(n=20000, k_n=12, sigma_j=0.3, seed=0):
    rng = np.random.default_rng(seed)
    v = 0.05 + 0.4 / np.arange(1, k_n + 1) ** 2  # a k-dependent diagonal
    j = rng.normal(0.0, sigma_j, n)
    x = j[None, :] + rng.normal(0.0, 1.0, (k_n, n)) * np.sqrt(v)[:, None]
    return x, v


def test_moment_fit_recovers_common_and_diagonal():
    x, v = _make()
    valid = np.ones(x.shape, dtype=bool)
    c, _ = E._pairwise_cov(x, valid)
    fit = E.fit_rank_one(c, np.full(x.shape[0], 6.0))
    assert fit["sigma_c2_mean"] == pytest.approx(0.09, rel=0.08)
    assert fit["sigma_c2_median"] == pytest.approx(0.09, rel=0.15)
    got = np.asarray(fit["v_k"])
    assert np.allclose(got, v, rtol=0.15, atol=0.01)
    # a genuine rank-one plus diagonal: uniform loading, small residual
    assert abs(fit["loading_beta"]) < 0.05
    assert fit["offdiag_resid_rel"] < 0.35
    assert fit["rank1_energy_frac"] > 0.9


def test_no_common_term_gives_zero_sigma():
    rng = np.random.default_rng(1)
    k_n = 12
    v = 0.05 + 0.4 / np.arange(1, k_n + 1) ** 2
    x = rng.normal(0.0, 1.0, (k_n, 20000)) * np.sqrt(v)[:, None]
    c, _ = E._pairwise_cov(x, np.ones(x.shape, dtype=bool))
    fit = E.fit_rank_one(c, np.full(k_n, 6.0))
    assert abs(fit["sigma_c2_mean"]) < 0.005
    assert np.allclose(np.asarray(fit["v_k"]), v, rtol=0.1, atol=0.01)


def test_pairwise_complete_matches_full_when_masked_at_random():
    x, v = _make(n=40000)
    rng = np.random.default_rng(3)
    valid = rng.random(x.shape) > 0.3  # per-harmonic dropout, as the twin gate does
    c, n = E._pairwise_cov(x, valid)
    fit = E.fit_rank_one(c, np.full(x.shape[0], 6.0))
    assert (n[~np.eye(x.shape[0], dtype=bool)] > 1000).all()
    assert fit["sigma_c2_mean"] == pytest.approx(0.09, rel=0.12)
    assert np.allclose(np.asarray(fit["v_k"]), v, rtol=0.2, atol=0.015)


def test_structured_common_term_is_flagged():
    """A common term whose loading falls as 1/k (a shared *phase*, not a shared
    delay) must NOT read as the uniform-loading model."""
    rng = np.random.default_rng(5)
    k_n, n = 12, 20000
    a = 1.0 / np.arange(1, k_n + 1)
    v = np.full(k_n, 0.05)
    x = (
        a[:, None] * rng.normal(0.0, 0.5, n)[None, :]
        + rng.normal(0, 1, (k_n, n)) * np.sqrt(v)[:, None]
    )
    c, _ = E._pairwise_cov(x, np.ones((k_n, n), dtype=bool))
    fit = E.fit_rank_one(c, np.full(k_n, 6.0))
    assert fit["loading_beta"] < -0.7  # recovers the 1/k loading
    assert fit["offdiag_resid_rel"] > 0.5  # and the constant-offdiag fit is bad


def test_kscaled_common_estimate_uses_wider_partners_only():
    """With a monotone band schedule the common power is cumulative in the
    band, so ``C_ij`` measures it up to ``min(B_i, B_j)``; the estimator must
    read harmonic k's common term off partners with a band at least as wide."""
    k_n, n = 10, 40000
    rng = np.random.default_rng(7)
    bands = np.arange(1, k_n + 1) * 1.0
    # common power grows with the band -> per-harmonic common contribution
    cum = 0.02 * bands
    inc = np.diff(np.concatenate(([0.0], cum)))
    comps = rng.normal(0.0, 1.0, (k_n, n)) * np.sqrt(inc)[:, None]
    # harmonic i sees components 0..i (everything inside its band)
    common = np.cumsum(comps, axis=0)
    v = np.full(k_n, 0.05)
    x = common + rng.normal(0, 1, (k_n, n)) * np.sqrt(v)[:, None]
    c, _ = E._pairwise_cov(x, np.ones((k_n, n), dtype=bool))
    fit = E.fit_rank_one(c, bands)
    # every harmonic but the widest (which has no wider partner and is
    # extrapolated, hence flagged) recovers its own diagonal
    assert np.allclose(np.asarray(fit["v_k"])[:-1], v[:-1], rtol=0.25, atol=0.01)
    assert fit["common_extrapolated"][-1] is True
    assert not any(fit["common_extrapolated"][:-1])
    got = np.asarray(fit["common_k"], dtype=float)
    assert np.allclose(got[:-1], cum[:-1], rtol=0.3, atol=0.012)


def test_brickwall_highpass_removes_slow_common_term():
    fs, n = 62.5, 4000
    t = np.arange(n) / fs
    slow = 2.0 * np.sin(2 * np.pi * 0.7 * t)
    fast = np.random.default_rng(9).normal(0, 0.2, n)
    y = E._brickwall(slow + fast, 4.0, fs, high=True)
    assert np.var(y) < 0.1 * np.var(slow)
    assert np.var(y) == pytest.approx(np.var(fast) * (1 - 4.0 / (fs / 2)), rel=0.25)
