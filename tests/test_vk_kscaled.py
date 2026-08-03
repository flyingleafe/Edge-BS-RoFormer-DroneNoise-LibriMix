"""Tests for the k-scaled per-track VK bandwidth (``VKConfig.bw_rps``) and the
WP18 frequency-update weight (``VKConfig.freq_weight = "k_beta"``).

Small synthetic fixtures in the ``test_vk_tracking.py`` style: seeded,
CPU-only, sized for speed.

Run:  pytest tests/test_vk_kscaled.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_processing.vk_tracking import (  # noqa: E402
    VKConfig,
    _tuma_bw_min,
    vk_envelopes,
    vk_track,
)

FS = 16000.0


def comb_audio(dur: float, r_true: np.ndarray, k_max: int, snr_db: float, seed: int) -> np.ndarray:
    """Locked-phase harmonic comb (k = 1..k_max, amps 1/sqrt(k)) + white noise."""
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * np.cumsum(r_true) / FS
    sig = np.zeros(int(dur * FS))
    psi = rng.uniform(0, 2 * np.pi, k_max)
    for k in range(1, k_max + 1):
        sig += (1.0 / np.sqrt(k)) * np.cos(k * phase + psi[k - 1])
    noise = rng.standard_normal(len(sig))
    noise *= np.sqrt(np.mean(sig**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return sig + noise


def make_grid(dur: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(int(dur * FS)) / FS
    frame_times = np.arange(0, dur, 0.032)
    edge = (frame_times > 0.5) & (frame_times < dur - 0.5)
    return t, frame_times, edge


def test_config_rejects_bad_bw_rps_and_freq_weight():
    with pytest.raises(ValueError, match="bw_rps"):
        VKConfig(bw_rps=0.0)
    with pytest.raises(ValueError, match="freq_weight"):
        VKConfig(freq_weight="amp_only")


def test_kscaled_bands_reported_in_bw_track():
    """Uncoupled single-rotor config: ``bw_track[m] == clip(k_m * bw_rps, lo, hi)``
    exactly (the sep clamp never fires when no coupling group forms)."""
    dur = 2.0
    t, _, _ = make_grid(dur)
    r_true = np.full_like(t, 45.0)
    y = comb_audio(dur, r_true, k_max=6, snr_db=20.0, seed=0)
    # couple_hz below the ~45 Hz harmonic spacing: every track is its own group
    cfg = VKConfig(fs=FS, k_max=6, couple_hz=20.0, bw_rps=0.25)
    env = vk_envelopes(y, r_true[None, :], cfg)
    assert all(len(g) == 1 for g in env.groups)
    # only solved tracks report a band (k=1 sits below f_min and is skipped)
    solved = sorted(m for g in env.groups for m in g)
    assert len(solved) >= 4
    fs_env = env.fs_env
    lo = _tuma_bw_min(fs_env, cfg.p)
    expected = np.clip(env.k.astype(np.float64) * 0.25, lo, 0.9 * fs_env)
    assert np.allclose(env.bw_track[solved], expected[solved], rtol=0, atol=1e-12)

    # a huge bw_rps must clip at the demod-lowpass ceiling 0.9 * fs_env
    cfg_hi = VKConfig(fs=FS, k_max=6, couple_hz=20.0, bw_rps=20.0)
    env_hi = vk_envelopes(y, r_true[None, :], cfg_hi)
    expected_hi = np.clip(env_hi.k.astype(np.float64) * 20.0, lo, 0.9 * fs_env)
    assert np.allclose(env_hi.bw_track[solved], expected_hi[solved], rtol=0, atol=1e-12)
    assert float(env_hi.bw_track[solved].max()) == pytest.approx(0.9 * fs_env)


def test_k_beta_keeps_no_comb_gate_on_white_noise():
    """freq_weight='k_beta' must not weaken the no-comb gate: on white noise
    the trajectory stays at the init (design test 5 invariant)."""
    dur = 2.0
    _, frame_times, _ = make_grid(dur)
    rng = np.random.default_rng(2)
    y = rng.standard_normal(int(dur * FS))
    r_init = np.full((1, len(frame_times)), 45.0)
    cfg = VKConfig(fs=FS, k_max=6, couple_hz=20.0, n_outer=4, freq_weight="k_beta")
    res = vk_track(y, r_init, frame_times, cfg)
    drift = float(np.max(np.abs(res.r_refined - 45.0)))
    assert drift < 0.05, f"k_beta weight hallucinated {drift:.3f} rev/s from white noise"
    assert res.max_deltas == [0.0] * cfg.n_outer


def test_k_beta_updates_on_locked_comb():
    """On a locked comb the k_beta weight still produces finite, nonzero deltas."""
    dur = 2.0
    t, frame_times, _ = make_grid(dur)
    r_true = np.full_like(t, 45.0)
    y = comb_audio(dur, r_true, k_max=6, snr_db=20.0, seed=3)
    r_init = np.full((1, len(frame_times)), 45.3)
    cfg = VKConfig(fs=FS, k_max=6, couple_hz=20.0, n_outer=4, freq_weight="k_beta")
    res = vk_track(y, r_init, frame_times, cfg)
    assert np.all(np.isfinite(res.r_refined))
    assert np.all(np.isfinite(res.max_deltas))
    assert max(res.max_deltas) > 0.0


def test_kscaled_k_beta_capture_smoke():
    """Capture smoke: 4-harmonic comb at 80 rev/s, 20 dB SNR, +0.5 rev/s init
    offset; ``bw_rps=0.25, n_outer=8, freq_weight='k_beta'`` recovers at least
    half of the offset."""
    dur = 4.0
    t, frame_times, edge = make_grid(dur)
    r_true = np.full_like(t, 80.0)
    y = comb_audio(dur, r_true, k_max=4, snr_db=20.0, seed=5)
    r_init = np.full((1, len(frame_times)), 80.5)
    cfg = VKConfig(
        fs=FS,
        k_max=4,
        couple_hz=20.0,
        n_outer=8,
        bw_rps=0.25,
        freq_weight="k_beta",
    )
    res = vk_track(y, r_init, frame_times, cfg)
    err = float(np.mean(np.abs(res.r_refined[0, edge] - 80.0)))
    assert err < 0.25, f"mean |error| {err:.3f} rev/s did not recover half the offset"
