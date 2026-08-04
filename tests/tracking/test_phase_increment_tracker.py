"""Tests for the phase-increment ML instantaneous-frequency tracker.

Mirrors ``test_warp_refinement.py``: reuses the phase-validation ladder's S0
synth helpers (``scripts/vk_phase_validation.py``) so the test signals are
exactly the ladder's cells, shortened to 12 s. Adds the tracker's key
property — recovery of a slowly-varying r(t) under HEAVY per-harmonic phase
diffusion (coherence time ~0.1 s at k=10), the regime that defeats
long-window coherent methods.
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

from tracking.phase_increment_tracker import pi_kalman_refine  # noqa: E402

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
    r, diag = pi_kalman_refine(cell.audio, cell.r_init_base + 1.0, cell.ft, sr=SR)
    assert r.shape == cell.r_init_base.shape
    assert _mae(cell, r) < 0.05
    # Diagnostics carry the per-iteration harmonic sets and q_k budgets.
    iters = diag["rotors"][0]["iters"]
    assert len(iters) == 3
    assert iters[0]["k_cap"] == 8
    assert all("q_k" in d for d in iters)
    assert iters[0]["n_meas"] > 0


def test_truth_init_does_not_degrade():
    """From exact truth the refiner must add < 0.02 rev/s error."""
    cell = _build_s0("none")
    base = _mae(cell, cell.r_init_base)  # frame-grid representation floor (~0)
    r, _ = pi_kalman_refine(cell.audio, cell.r_init_base.copy(), cell.ft, sr=SR)
    assert _mae(cell, r) - base < 0.02


def test_perharm_jitter_still_converges_from_offset():
    """Per-harmonic OU jitter (the arm where stage D stalls): from +1.0 the
    common shaft error must still be pulled out — the diffusion lands in the
    per-harmonic measurement-noise term, not in the signal model."""
    cell = _build_s0("perharm")
    r, _ = pi_kalman_refine(cell.audio, cell.r_init_base + 1.0, cell.ft, sr=SR)
    assert _mae(cell, r) < 0.15


def _build_heavy_diffusion_cell(seed: int = 5, q1: float = 0.2):
    """Comb with per-harmonic Brownian phase, q_k = q1 * k^2 rad^2/s.

    At q1 = 0.2 the coherence time tau_k = 2 / q_k is 0.1 s at k=10 (the
    spec's regime): a long coherent window sees only smeared lines at
    mid/high k, while frame-to-frame phase increments remain informative.
    """
    rng = np.random.default_rng(seed)
    r_aud = ladder._synth_rps(DUR_S, seed, (80.0,))  # (1, T)
    n = r_aud.shape[-1]
    phi = 2.0 * np.pi * np.cumsum(r_aud[0]) / SR
    sig = np.zeros(n)
    for k in range(1, 41):
        b = np.cumsum(rng.standard_normal(n)) * np.sqrt(q1 * k * k / SR)
        sig += (1.0 / k) * np.cos(k * phi + rng.uniform(0.0, 2.0 * np.pi) + b)
    sig += 0.01 * rng.standard_normal(n)
    ft = ladder._frame_grid(n)
    t_aud = np.arange(n) / SR
    truth_ft = ladder._interp_rows(ft, t_aud, r_aud)
    return sig[None], ft, truth_ft


def test_heavy_diffusion_recovers_slow_rate():
    """THE key property: with tau ~ 0.1 s at k=10 (q_k = 0.2 k^2 rad^2/s)
    the slowly-varying r(t) is still recovered to < 0.2 rev/s from +1.0,
    and the data-driven q_k estimate lands in the right decade."""
    sig, ft, truth_ft = _build_heavy_diffusion_cell()
    r, diag = pi_kalman_refine(sig, truth_ft + 1.0, ft, sr=SR)
    edge = (ft > EDGE_S) & (ft < ft[-1] - EDGE_S)
    mae = float(np.mean(np.abs((r - truth_ft)[:, edge])))
    assert mae < 0.2
    q10 = diag["rotors"][0]["iters"][-1]["q_k"].get("10")
    assert q10 is not None
    assert 4.0 < q10 < 120.0  # truth 20 rad^2/s; order-of-magnitude calibration


def test_joint_mode_tracks_fully_collided_twins():
    """Tight twin pair at a CONSTANT split of 0.7 rev/s (one shaft
    trajectory, twin = shaft + 0.7) with a comb of ONLY k <= 10: from a
    differential init (+0.3 / -0.2 — track split 0.2) every signal-bearing
    harmonic is twin-collided, so gate mode has nothing to measure and
    plateaus at the init error; joint mode must resolve the two-tone
    mixtures and track BOTH rotors to < 0.15."""
    rng = np.random.default_rng(11)
    shaft = ladder._synth_rps(DUR_S, 21, (80.0,))[0]
    r_aud = np.stack([shaft, shaft + 0.7])  # (2, T), split exactly 0.7
    n = r_aud.shape[-1]
    sig = np.zeros(n)
    for i in range(2):
        phase = 2.0 * np.pi * np.cumsum(r_aud[i]) / SR
        for k in range(1, 11):
            sig += (1.0 / k) * np.cos(k * phase + rng.uniform(0.0, 2.0 * np.pi))
    sig += 0.01 * rng.standard_normal(n)
    ft = ladder._frame_grid(n)
    t_aud = np.arange(n) / SR
    truth_ft = ladder._interp_rows(ft, t_aud, r_aud)
    edge = (ft > EDGE_S) & (ft < ft[-1] - EDGE_S)
    init = truth_ft + np.asarray([0.3, -0.2])[:, None]  # differential error

    r_gate, _ = pi_kalman_refine(sig[None], init.copy(), ft, sr=SR)
    r_joint, diag = pi_kalman_refine(sig[None], init.copy(), ft, sr=SR, pair_mode="joint")
    for i in range(2):
        mae_gate = float(np.mean(np.abs(r_gate[i] - truth_ft[i])[edge]))
        mae_joint = float(np.mean(np.abs(r_joint[i] - truth_ft[i])[edge]))
        assert mae_joint < 0.15
        assert mae_joint < mae_gate  # gate mode plateaus (near-no-op here)
        maes = [float(np.mean(np.abs(r_joint[i] - truth_ft[j])[edge])) for j in range(2)]
        assert int(np.argmin(maes)) == i  # order assignment keeps identity
    # Pair diagnostics report the measured split near the true 0.7 rev/s.
    last_pair = diag["pairs"][-1][0]
    assert last_pair["n_windows_locked"] > 0
    assert 0.55 < last_pair["split_meas_med"] < 0.85


def test_twin_pair_no_cross_capture():
    """Two rotors 1 rev/s apart, comb up to k=40: truth-init refinement must
    stay on its own rotor (the twin guard excludes colliding harmonics)."""
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

    r, _ = pi_kalman_refine(sig[None], truth_ft.copy(), ft, sr=SR)
    for i in range(2):
        maes = [float(np.mean(np.abs(r[i] - truth_ft[j])[edge])) for j in range(2)]
        assert int(np.argmin(maes)) == i  # no swap toward the twin
        assert maes[i] < 0.1
