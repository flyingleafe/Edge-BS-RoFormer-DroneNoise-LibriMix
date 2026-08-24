"""The restored classical RPS baselines (commit ``00753c4``).

These estimators are known to be bad on real four-rotor mixtures — that is the
point of keeping them as a baseline. So the tests pin down the contract only:
the shape, the finiteness, and the search band of the output, plus the fact
that every entry of ``CLASSICAL_TRACKERS`` runs on an arbitrary waveform.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.classical_rps.predictors import (
    CLASSICAL_TRACKERS,
    HOP_LENGTH,
    N_BLADES,
    N_ROTORS,
    RPS_MAX,
    RPS_MIN,
    SR,
    matched_filter_tracker,
    nmf_tracker,
)

FOUR_RPS = (71.0, 78.5, 84.0, 89.5)


def _four_comb(duration_s: float = 2.0, n_harm: int = 6, seed: int = 0) -> np.ndarray:
    """Four harmonic combs at ``FOUR_RPS``, plus a little white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * SR)) / SR
    x = 0.01 * rng.standard_normal(t.size)
    for rps in FOUR_RPS:
        f0 = N_BLADES * rps
        for k in range(1, n_harm + 1):
            x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return (x / np.max(np.abs(x))).astype(np.float32)


@pytest.mark.parametrize("tracker", [matched_filter_tracker, nmf_tracker])
def test_a_comb_tracker_returns_a_full_rotor_track_inside_the_search_band(tracker):
    audio = _four_comb()
    pred = tracker(audio)

    expected_frames = len(audio) // HOP_LENGTH + 1
    assert pred.shape == (N_ROTORS, expected_frames)
    assert np.isfinite(pred).all()
    assert (pred >= RPS_MIN).all()
    assert (pred <= RPS_MAX).all()


def test_every_classical_tracker_runs_on_white_noise():
    rng = np.random.default_rng(1)
    audio = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    expected_frames = len(audio) // HOP_LENGTH + 1

    for name, tracker in CLASSICAL_TRACKERS.items():
        pred = tracker(audio)
        assert pred.shape == (N_ROTORS, expected_frames), name
        assert np.isfinite(pred).all(), name
