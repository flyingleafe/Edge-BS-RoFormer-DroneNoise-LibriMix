"""Tests for data_processing.harmonicity — torch-free, tiny signals.

A synthetic harmonic comb must read as strongly harmonic; white noise must read
as flat/non-harmonic. These are the ordering properties the metric exists to
provide (see the module docstring).
"""

from __future__ import annotations

import numpy as np

from data_processing.harmonicity import measure_harmonicity

SR = 16000


def _harmonic_tone(f0: float, n_harmonics: int, seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
    return (x / np.max(np.abs(x))).astype(np.float32)


def test_harmonic_tone_reads_as_harmonic():
    h = measure_harmonicity(_harmonic_tone(120.0, 6), SR)
    assert abs(h.f0_hz - 120.0) < 5.0
    assert h.harmonic_energy_ratio > 0.5
    assert h.harmonic_to_noise_db > 6.0
    assert h.n_prominent_harmonics >= 3
    assert h.spectral_flatness < 0.2


def test_white_noise_reads_as_non_harmonic():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(4.0 * SR)).astype(np.float32)
    h = measure_harmonicity(noise, SR)
    assert h.harmonic_energy_ratio < 0.25
    assert h.spectral_flatness > 0.3


def test_harmonic_beats_noise_on_ordering():
    rng = np.random.default_rng(1)
    tone = _harmonic_tone(90.0, 5)
    noise = rng.standard_normal(tone.shape).astype(np.float32)
    ht = measure_harmonicity(tone, SR)
    hn = measure_harmonicity(noise, SR)
    assert ht.harmonic_energy_ratio > hn.harmonic_energy_ratio
    assert ht.spectral_flatness < hn.spectral_flatness


def test_silence_is_safe():
    h = measure_harmonicity(np.zeros(SR, dtype=np.float32), SR)
    assert h.f0_hz == 0.0
    assert h.harmonic_energy_ratio == 0.0
    assert h.spectral_flatness == 1.0


def test_multichannel_averages_channels():
    tone = _harmonic_tone(150.0, 5)
    stereo = np.stack([tone, tone])  # (2, T)
    h = measure_harmonicity(stereo, SR)
    assert abs(h.f0_hz - 150.0) < 5.0
    assert h.harmonic_energy_ratio > 0.5


def test_as_dict_is_json_scalars():
    h = measure_harmonicity(_harmonic_tone(100.0, 4), SR)
    d = h.as_dict()
    assert isinstance(d["n_prominent_harmonics"], int)
    assert isinstance(d["f0_hz"], float)
    assert set(d) == {
        "f0_hz",
        "harmonic_energy_ratio",
        "harmonic_to_noise_db",
        "n_prominent_harmonics",
        "spectral_flatness",
    }
