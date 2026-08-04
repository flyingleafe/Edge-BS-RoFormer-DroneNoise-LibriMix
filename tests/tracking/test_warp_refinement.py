"""Tests for the iterated time-warp (generalized-demodulation) IF refiner.

Reuses the phase-validation ladder's S0 synth helpers
(``scripts/vk_phase_validation.py``: ``build_s0_cell`` / ``_synth_rps``) so
the test signals are exactly the ladder's cells, shortened to 12 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "scripts"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vk_phase_validation as ladder  # noqa: E402

from tracking.warp_refinement import iter_warp_refine  # noqa: E402

SR = ladder.SR
DUR_S = 12.0
EDGE_S = 0.5


def _build_s0(jitter: str, seed: int = 1000):
    """Ladder S0 cell (clean SNR) at the test's shorter duration."""
    old = ladder.S0_DUR_S
    ladder.S0_DUR_S = DUR_S
    try:
        return ladder.build_s0_cell("clean", jitter, seed)
    finally:
        ladder.S0_DUR_S = old


def _mae(cell, r_hat: np.ndarray) -> float:
    t_aud = np.arange(cell.r_true_aud.shape[-1]) / SR
    truth_ft = np.stack(
        [np.interp(cell.ft, t_aud, cell.r_true_aud[i]) for i in range(cell.r_true_aud.shape[0])]
    )
    edge = (cell.ft > EDGE_S) & (cell.ft < cell.ft[-1] - EDGE_S)
    return float(np.mean(np.abs((r_hat - truth_ft)[:, edge])))


def test_converges_from_one_rev_offset_clean():
    """From truth+1.0 rev/s on a clean single-rotor comb: IF MAE < 0.05."""
    cell = _build_s0("none")
    r, diag = iter_warp_refine(cell.audio, cell.r_init_base + 1.0, cell.ft, sr=SR)
    assert r.shape == cell.r_init_base.shape
    assert _mae(cell, r) < 0.05
    # Diagnostics carry per-round per-order lock quality.
    rounds = diag["rotors"][0]["rounds"]
    assert len(rounds) == 4
    assert all("orders" in rd for rd in rounds)
    assert any(o["n_locked"] > 0 for o in rounds[0]["orders"])


def test_truth_init_does_not_degrade():
    """From exact truth the refiner must add < 0.02 rev/s error."""
    cell = _build_s0("none")
    base = _mae(cell, cell.r_init_base)  # frame-grid representation floor (~0)
    r, _ = iter_warp_refine(cell.audio, cell.r_init_base.copy(), cell.ft, sr=SR)
    assert _mae(cell, r) - base < 0.02


def test_perharm_jitter_still_converges_from_offset():
    """Per-harmonic OU jitter (the arm where stage D stalls): from +1.0 the
    common shaft error must still be pulled out via the low-order rungs."""
    cell = _build_s0("perharm")
    r, _ = iter_warp_refine(cell.audio, cell.r_init_base + 1.0, cell.ft, sr=SR)
    assert _mae(cell, r) < 0.15


def test_twin_pair_no_cross_capture():
    """Two rotors 1 rev/s apart, comb up to k=40: truth-init refinement must
    stay on its own rotor (twin rejection keeps colliding orders out)."""
    rng = np.random.default_rng(7)
    r_aud = ladder._synth_rps(DUR_S, 42, (80.0, 81.0))  # (2, T)
    n = r_aud.shape[-1]
    sig = np.zeros(n)
    for i in range(2):
        phase = 2.0 * np.pi * np.cumsum(r_aud[i]) / SR
        for k in range(1, 41):
            sig += (1.0 / k) * np.cos(k * phase + rng.uniform(0.0, 2.0 * np.pi))
    sig += 0.01 * rng.standard_normal(n)
    ft = ladder._frame_grid(n)
    t_aud = np.arange(n) / SR
    truth_ft = ladder._interp_rows(ft, t_aud, r_aud)
    edge = (ft > EDGE_S) & (ft < ft[-1] - EDGE_S)

    r, _ = iter_warp_refine(sig[None], truth_ft.copy(), ft, sr=SR)
    for i in range(2):
        maes = [float(np.mean(np.abs(r[i] - truth_ft[j])[edge])) for j in range(2)]
        assert int(np.argmin(maes)) == i  # no swap toward the twin
        assert maes[i] < 0.1
