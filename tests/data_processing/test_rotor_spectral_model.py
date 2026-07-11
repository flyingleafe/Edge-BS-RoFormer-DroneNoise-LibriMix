"""Tests for the analytic static-comb rotor-spectral noise model (E8)."""

from __future__ import annotations

import numpy as np
import tdseries as td

from data_processing.online_mixing import build_noise_pool
from data_processing.rotor_spectral_model import (
    ProfileRanges,
    StaticCombNoisePool,
    _comb_waveform,
    sample_profile,
)


def test_profile_enforces_min_fraction_above_floor():
    rng = np.random.default_rng(0)
    for _ in range(50):
        prof = sample_profile(
            rng,
            ProfileRanges(),
            n_harmonics=100,
            ref_rps=80.0,
            sample_rate=16000,
            min_harm_above_floor=0.30,
        )
        assert prof.frac_above_floor >= 0.30 - 1e-9
        assert prof.a_k.shape == (100,)
        assert abs(prof.a_k[0] - 1.0) < 1e-6  # fundamental normalised to 1


def test_comb_amplitudes_are_static_in_time():
    # A constant-RPS comb must not drift in per-harmonic magnitude over time —
    # this is the whole point (no amplitude->RPS shortcut).
    rng = np.random.default_rng(1)
    sr = 16000
    prof = sample_profile(
        rng,
        ProfileRanges(),
        n_harmonics=100,
        ref_rps=90.0,
        sample_rate=sr,
        min_harm_above_floor=0.30,
    )
    w = _comb_waveform(np.full(sr, 90.0), prof.a_k, sr, rng)
    win = np.hanning(1024)
    cols = np.array([np.abs(np.fft.rfft(w[i : i + 1024] * win)) for i in range(0, sr - 1024, 512)])
    cov = cols.std(0) / (cols.mean(0) + 1e-9)
    assert float(np.median(cov)) < 0.1  # magnitudes essentially constant across frames


def test_static_comb_pool_yields_wellformed_timeframe():
    pool = StaticCombNoisePool(
        sample_rate=16000, duration_s=0.5, n_harmonics=64, n_mics=8, n_rotors=4, seed=3
    )
    tf = pool.sample_timeframe(np.random.default_rng(0), 0.5)
    assert isinstance(tf, td.Frame)
    audio = tf["audio"]
    assert audio.data.shape == (8, int(round(0.5 * 16000)))
    assert np.isfinite(np.asarray(audio.data)).all()
    rps = np.asarray(tf["rps"].data)
    assert rps.shape[0] == 4
    assert rps.min() >= 20.0 and rps.max() <= 200.0


def test_build_noise_pool_dispatches_static_comb():
    pool = build_noise_pool(
        {"kind": "static_comb", "n_harmonics": 32, "n_mics": 8, "n_rotors": 4},
        duration_s=0.5,
        sample_rate=16000,
    )
    assert isinstance(pool, StaticCombNoisePool)
