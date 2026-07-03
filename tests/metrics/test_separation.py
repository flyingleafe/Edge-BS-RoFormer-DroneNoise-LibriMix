"""Tests for metrics.separation (sdr, si_sdr, l1_freq, ..., Frame adapters)."""

import numpy as np
import tdseries as td

from metrics.separation import (
    ESTOIMetric,
    PESQMetric,
    SDRMetric,
    SISDRMetric,
    STOIMetric,
    bleed_full,
    bleedless,
    estoi,
    fullness,
    l1_freq,
    pesq,
    sdr,
    si_sdr,
    stoi,
)

SR = 16000


def _mono_frame(entry: str, x: np.ndarray) -> td.Frame:
    return td.Frame({entry: td.uniform(x.astype(np.float32), SR, dims=("time",))})


def _tone(freq: float = 440.0, seconds: float = 1.0, sr: int = SR, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ─── Hand-computed sanity checks ────────────────────────────────────────────


def test_si_sdr_scale_invariance():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((1, 4000)).astype(np.float32)
    scaled = ref * 2.5
    # A pure scaling has ~zero distortion after optimal rescaling -> very high SI-SDR.
    assert si_sdr(ref, scaled) > 100.0


def test_si_sdr_matches_hand_computation():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((1, 4000)).astype(np.float64)
    noise = rng.standard_normal(ref.shape).astype(np.float64) * 0.1
    est = ref + noise

    scale = np.dot(est.flatten(), ref.flatten()) / np.dot(ref.flatten(), ref.flatten())
    ref_scaled = ref * scale
    expected = 10 * np.log10(np.sum(ref_scaled**2) / np.sum((ref_scaled - est) ** 2))

    got = si_sdr(ref.astype(np.float32), est.astype(np.float32))
    assert abs(got - expected) < 1e-2


def test_sdr_matches_hand_computation():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((1, 1, 2000)).astype(np.float32)
    noise = rng.standard_normal(ref.shape).astype(np.float32) * 0.05
    est = ref + noise

    num = np.sum(ref**2)
    den = np.sum((ref - est) ** 2)
    expected = 10 * np.log10(num / den)

    got = sdr(ref, est)[0]
    assert abs(got - expected) < 1e-2


def test_sdr_higher_for_less_distorted_estimate():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((1, 1, 2000)).astype(np.float32)
    small_noise = ref + rng.standard_normal(ref.shape).astype(np.float32) * 0.01
    big_noise = ref + rng.standard_normal(ref.shape).astype(np.float32) * 1.0
    assert sdr(ref, small_noise)[0] > sdr(ref, big_noise)[0]


def test_l1_freq_zero_distance_gives_max_score():
    x = _tone()
    assert l1_freq(x[None, :], x[None, :]) == 100.0


def test_bleed_full_identical_signal_is_perfect():
    x = _tone(seconds=1.0)
    bl, fu = bleed_full(x[None, :], x[None, :], sr=SR)
    assert bl == 100.0
    assert fu == 100.0
    assert bleedless(x[None, :], x[None, :], sr=SR) == 100.0
    assert fullness(x[None, :], x[None, :], sr=SR) == 100.0


def test_bleed_full_bleeding_estimate_lowers_bleedless_not_fullness():
    ref = _tone(freq=440.0)
    # Estimate has extra energy the reference doesn't (bleed): scale up.
    bleeding = ref * 3.0
    bl, fu = bleed_full(ref[None, :], bleeding[None, :], sr=SR)
    assert bl < 100.0
    assert fu == 100.0  # estimate has strictly more energy everywhere -> no incompleteness


# ─── Perceptual metrics (smoke-tested, fast enough to run every time) ───────


def test_pesq_high_for_near_identical_speech_like_signal():
    x = _tone(freq=200.0, seconds=1.0)
    score = pesq(x[None, :], (x * 0.99)[None, :], SR)
    assert score > 3.5  # PESQ range is roughly [-0.5, 4.5]


def test_stoi_high_for_near_identical_signal():
    x = _tone(freq=200.0, seconds=1.0)
    score = stoi(x[None, :], (x * 0.99)[None, :], SR)
    assert score > 0.9
    assert estoi(x[None, :], (x * 0.99)[None, :], SR) > 0.9


# ─── Frame adapters ──────────────────────────────────────────────────────────


def test_sdr_metric_frame_adapter():
    x = _tone()
    pred = _mono_frame("enhanced", x)
    target = _mono_frame("target", x)
    metric = SDRMetric()
    value = metric(pred, target)
    assert isinstance(value, float)
    assert value > 50.0  # identical signal, only eps floor limits this
    assert "enhanced" in metric.requires_pred.entries
    assert "target" in metric.requires_target.entries


def test_si_sdr_metric_frame_adapter():
    x = _tone()
    pred = _mono_frame("enhanced", x)
    target = _mono_frame("target", x)
    metric = SISDRMetric()
    assert metric(pred, target) > 50.0


def test_pesq_and_stoi_metrics_frame_adapters():
    x = _tone(freq=200.0)
    pred = _mono_frame("enhanced", (x * 0.99).copy())
    target = _mono_frame("target", x)

    pesq_metric = PESQMetric(sample_rate=SR)
    stoi_metric = STOIMetric(sample_rate=SR)
    estoi_metric = ESTOIMetric(sample_rate=SR)

    assert pesq_metric(pred, target) > 3.5
    assert stoi_metric(pred, target) > 0.9
    assert estoi_metric(pred, target) > 0.9
