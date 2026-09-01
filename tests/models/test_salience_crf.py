"""The encode/decode pair must lose nothing on real-shaped trajectories."""
from __future__ import annotations

import numpy as np
import torch

from models.salience_crf import band_for_rev_s, crf_decode_layers, gaussian_layer_target

GRID = np.arange(0.0, 150.0 + 1e-9, 0.5)


def _tracks(seed=0, n=3, r=4, t=200):
    """Trajectories with the dynamics real telemetry has: OU-like, reaching 0."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        base = np.cumsum(rng.standard_normal(t)) * 0.6 + 60.0
        g = np.stack([np.clip(base + rng.standard_normal(t).cumsum() * 0.2 + k * 3.0,
                              0.0, 149.0) for k in range(r)])
        out.append(g)
    return torch.tensor(np.stack(out))


def test_round_trip_is_exact():
    rps = _tracks()
    span, pen = band_for_rev_s(16.0, 0.5, dtype=torch.float64)
    sal = gaussian_layer_target(rps.double(), GRID, sigma_bins=1.0)
    rec = crf_decode_layers(sal, GRID, span, pen)
    assert float((rec - rps.double()).abs().max()) < 1e-9


def test_a_stopped_rotor_decodes_to_zero():
    """Zero is a value at bin 0, not an absence needing a threshold."""
    rps = torch.zeros(1, 4, 40, dtype=torch.float64)
    rps[0, 1] = 61.0
    span, pen = band_for_rev_s(16.0, 0.5, dtype=torch.float64)
    rec = crf_decode_layers(gaussian_layer_target(rps, GRID, 1.0), GRID, span, pen)
    assert float(rec[0, 0].abs().max()) < 1e-9
    assert abs(float(rec[0, 1].mean()) - 61.0) < 1e-9


def test_the_band_admits_what_real_rotors_do():
    """15.3 rev/s in one frame is 31 bins at 0.5 rev/s; 3 bins cannot hold it."""
    span, _ = band_for_rev_s(15.3, 0.5)
    assert span >= 31
