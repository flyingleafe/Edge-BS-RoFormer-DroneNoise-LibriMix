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


def _twin_cell(split: float = 0.7, k_top: int = 10, seed: int = 11):
    """Tight twin pair at a constant ``split``, comb of only ``k <= k_top``."""
    rng = np.random.default_rng(seed)
    shaft = ladder._synth_rps(DUR_S, 21, (80.0,))[0]
    r_aud = np.stack([shaft, shaft + split])
    n = r_aud.shape[-1]
    sig = np.zeros(n)
    for i in range(2):
        phase = 2.0 * np.pi * np.cumsum(r_aud[i]) / SR
        for k in range(1, k_top + 1):
            sig += (1.0 / k) * np.cos(k * phase + rng.uniform(0.0, 2.0 * np.pi))
    sig += 0.01 * rng.standard_normal(n)
    ft = ladder._frame_grid(n)
    t_aud = np.arange(n) / SR
    truth_ft = ladder._interp_rows(ft, t_aud, r_aud)
    edge = (ft > EDGE_S) & (ft < ft[-1] - EDGE_S)
    return sig, ft, truth_ft, edge


def test_k_scaled_converges_within_capture():
    """band_mode='k_scaled': from +0.25 (inside the B0 = 0.35 trust region)
    the clean single-rotor comb converges as tightly as the fixed band."""
    cell = _build_s0("none")
    r, diag = pi_kalman_refine(
        cell.audio, cell.r_init_base + 0.25, cell.ft, sr=SR, band_mode="k_scaled"
    )
    assert _mae(cell, r) < 0.05
    it0 = diag["rotors"][0]["iters"][0]
    assert "band_hz_k" in it0
    # bands scale with k: k=8 band = 8 * 0.35 = 2.8 Hz
    assert abs(it0["band_hz_k"]["8"] - 2.8) < 0.01
    assert "n_band_clamped" not in it0  # no Nyquist clamp below k ~ 80


def test_k_scaled_out_of_capture_never_worsens():
    """From +1.0 (outside the nominal B0 = 0.35 capture range at every k)
    the k-scaled stage must never move AWAY from truth — band-edge leakage
    may pull it partially in, but the error must not grow."""
    cell = _build_s0("none")
    init = cell.r_init_base + 1.0
    mae_init = _mae(cell, init)
    r, _ = pi_kalman_refine(cell.audio, init.copy(), cell.ft, sr=SR, band_mode="k_scaled")
    assert _mae(cell, r) <= mae_init + 0.05


def test_k_scaled_unmasks_twin_low_harmonics():
    """Twin split 0.7 with a comb of only k <= 10: the fixed 6 Hz band
    twin-collides every harmonic (sep 7 Hz -> k < 10), so gate mode is a
    no-op; the k-scaled separation k*0.35 + 1 un-masks k >= 3
    (0.7k > 0.35k + 1 for k > 2.9) and gate mode converges."""
    sig, ft, truth_ft, edge = _twin_cell()
    init = truth_ft + np.asarray([0.15, -0.1])[:, None]

    r_fix, diag_fix = pi_kalman_refine(sig[None], init.copy(), ft, sr=SR)
    r_ks, diag_ks = pi_kalman_refine(sig[None], init.copy(), ft, sr=SR, band_mode="k_scaled")
    it_fix = diag_fix["rotors"][0]["iters"][-1]
    it_ks = diag_ks["rotors"][0]["iters"][-1]
    # fixed band: every signal-bearing harmonic fully collided or gated
    assert it_fix.get("n_meas", 0) == 0 or it_fix["n_twin_excluded"] >= 9
    # k-scaled: k >= 3 carry measurements
    n_k = it_ks.get("n_meas_k", {})
    assert sum(n_k.get(str(k), 0) for k in range(3, 11)) > 0
    for i in range(2):
        mae_fix = float(np.mean(np.abs(r_fix[i] - truth_ft[i])[edge]))
        mae_ks = float(np.mean(np.abs(r_ks[i] - truth_ft[i])[edge]))
        assert mae_ks < 0.08
        assert mae_ks < mae_fix


def test_band_anneal_shrinks_and_converges():
    """band_anneal='posterior': B0 shrinks across iterations (recorded per
    iteration and in band_b0_final) without hurting convergence."""
    cell = _build_s0("none")
    r, diag = pi_kalman_refine(
        cell.audio,
        cell.r_init_base + 0.25,
        cell.ft,
        sr=SR,
        band_mode="k_scaled",
        band_anneal="posterior",
    )
    assert _mae(cell, r) < 0.05
    iters = diag["rotors"][0]["iters"]
    b0s = [d["band_b0"] for d in iters]
    assert b0s[0] == 0.35
    assert b0s[-1] < 0.35  # annealed down
    final = diag["band_b0_final"][0]
    assert 0.12 <= final <= 0.35


def _displaced_cell(delta: float = 0.4, k_split: int = 13, seed: int = 31):
    """DREGON-like displaced comb: harmonics k < k_split ride at r - delta,
    k >= k_split on the true mechanical rate r."""
    rng = np.random.default_rng(seed)
    r_aud = ladder._synth_rps(DUR_S, 77, (80.0,))  # (1, T)
    n = r_aud.shape[-1]
    phi_true = 2.0 * np.pi * np.cumsum(r_aud[0]) / SR
    phi_disp = 2.0 * np.pi * np.cumsum(r_aud[0] - delta) / SR
    sig = np.zeros(n)
    for k in range(1, 31):
        phase = phi_disp if k < k_split else phi_true
        sig += (1.0 / k) * np.cos(k * phase + rng.uniform(0.0, 2.0 * np.pi))
    sig += 0.01 * rng.standard_normal(n)
    ft = ladder._frame_grid(n)
    t_aud = np.arange(n) / SR
    truth_ft = ladder._interp_rows(ft, t_aud, r_aud)
    return sig, ft, truth_ft


def test_lowk_consistency_gate_blocks_displaced_pull():
    """On a displaced comb (k < 13 at r - 0.4) truth-init refinement is
    pulled below truth by the displaced low harmonics; the consistency gate
    detects the low-vs-high disagreement and blocks most of the pull."""
    sig, ft, truth_ft = _displaced_cell()
    edge = (ft > EDGE_S) & (ft < ft[-1] - EDGE_S)
    r_def, _ = pi_kalman_refine(sig[None], truth_ft.copy(), ft, sr=SR)
    r_gated, diag = pi_kalman_refine(sig[None], truth_ft.copy(), ft, sr=SR, lowk_gate="consistency")
    mae_def = float(np.mean(np.abs(r_def - truth_ft)[:, edge]))
    mae_gated = float(np.mean(np.abs(r_gated - truth_ft)[:, edge]))
    fired = any(d.get("lowk", {}).get("fired", False) for d in diag["rotors"][0]["iters"])
    assert fired
    assert mae_def > 0.1  # the displaced pull is real on this cell
    assert mae_gated < 0.6 * mae_def


def test_lowk_gate_is_noop_on_consistent_comb():
    """On a clean on-grid comb the gate must not fire, and the output must
    be BIT-IDENTICAL to the default (the FLY124 no-op requirement)."""
    cell = _build_s0("none")
    r_def, _ = pi_kalman_refine(cell.audio, cell.r_init_base + 0.25, cell.ft, sr=SR)
    r_gated, diag = pi_kalman_refine(
        cell.audio, cell.r_init_base + 0.25, cell.ft, sr=SR, lowk_gate="consistency"
    )
    assert not any(d.get("lowk", {}).get("fired", False) for d in diag["rotors"][0]["iters"])
    assert np.array_equal(r_def, r_gated)


def test_clean_probe_avoids_other_rotor_lines():
    """Two rotors 2.75 rev/s apart: the fixed +11 Hz probe of rotor 0's
    k = 4 sits exactly on rotor 1's k = 4 line (4 * 2.75 = 11); probe_mode
    'clean' must move it, log zero fallbacks, and still converge."""
    rng = np.random.default_rng(13)
    n = int(DUR_S * SR)
    t = np.arange(n) / SR
    wobble = 0.05 * np.sin(2.0 * np.pi * 0.3 * t)  # near-steady cruise rates
    r_aud = np.stack([80.0 + wobble, 82.75 + wobble])
    sig = np.zeros(n)
    for i in range(2):
        phase = 2.0 * np.pi * np.cumsum(r_aud[i]) / SR
        for k in range(1, 31):
            sig += (1.0 / k) * np.cos(k * phase + rng.uniform(0.0, 2.0 * np.pi))
    sig += 0.01 * rng.standard_normal(n)
    ft = ladder._frame_grid(n)
    t_aud = np.arange(n) / SR
    truth_ft = ladder._interp_rows(ft, t_aud, r_aud)
    edge = (ft > EDGE_S) & (ft < ft[-1] - EDGE_S)

    r, diag = pi_kalman_refine(sig[None], truth_ft + 0.2, ft, sr=SR, probe_mode="clean")
    it_last = diag["rotors"][0]["iters"][-1]
    assert it_last["probe_fallbacks"] == 0
    offs = it_last["probe_off_k"]
    assert offs  # per-k offsets recorded
    for i in range(2):
        assert float(np.mean(np.abs(r[i] - truth_ft[i])[edge])) < 0.1


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


# ---------------------------------------------------------------------------
# demodulation-bank internals (issue #16 Tier 0)


def _demod_fixture(sr=16000, dur=4.0, n_ch=3, k_top=24, seed=11):
    """A 3-channel comb clip plus the args ``_demod_bank`` takes."""
    rng = np.random.default_rng(seed)
    n_t = int(dur * sr)
    t = np.arange(n_t) / sr
    r = 60.0 + 0.5 * np.sin(2.0 * np.pi * 0.3 * t)
    phi = 2.0 * np.pi * np.cumsum(r) / sr
    y = np.zeros((n_ch, n_t), np.float32)
    for k in range(1, k_top + 6):
        y += (np.cos(k * phi + rng.uniform(0.0, 2.0 * np.pi)) / k).astype(np.float32)
    y += (rng.standard_normal((n_ch, n_t)) * 0.05).astype(np.float32)
    stride = 256
    return y, phi, t, list(range(1, k_top + 1)), stride, len(range(0, n_t, stride)), sr


def _explicit_off_comb(y, phi, t, ks, off_hz, stride, n_env, band_cyc, band_cyc_k=None):
    """The pre-optimization off-comb bank: demodulate the clip a SECOND time."""
    from tracking.phase_increment_tracker import zoom_lp_decimate

    c1 = np.exp(-1j * phi).astype(np.complex64)
    ramp = np.exp(-2j * np.pi * off_hz * t).astype(np.complex64)
    out = np.empty((y.shape[0], len(ks), n_env), dtype=np.complex128)
    cur, cur_k = np.ones_like(c1), 0
    for a, k in enumerate(ks):
        for _ in range(k - cur_k):
            cur = cur * c1
        cur_k = k
        rows = None if band_cyc_k is None else band_cyc_k[a : a + 1]
        buf = (np.asarray(y * cur, dtype=np.complex64) * ramp)[:, None, :]
        out[:, a : a + 1] = zoom_lp_decimate(buf, stride, n_env, band_cyc, rows)
    return out


def test_off_comb_probe_matches_a_second_demodulation():
    """The probe sliced out of the on-comb spectrum IS the second demodulation.

    A constant frequency offset is a pure bin shift, so one FFT serves both
    bands (issue #16 Tier 0 item 2); with the offset on the bin grid the only
    difference left is complex64 rounding.
    """
    from tracking.phase_increment_tracker import _demod_bank

    y, phi, t, ks, stride, n_env, sr = _demod_fixture()
    band_cyc, off_hz = 6.0 / sr, 11.0
    assert (off_hz * stride * n_env / sr).is_integer()  # exactly on the bin grid
    z_on, z_off = _demod_bank(y, phi, t, ks, off_hz, stride, n_env, band_cyc, sr=sr)
    ref = _explicit_off_comb(y, phi, t, ks, off_hz, stride, n_env, band_cyc)
    assert np.abs(z_off - ref).max() < 1e-6 * np.abs(ref).max()
    # The noise floor the gates actually consume agrees to ~1e-6 relative.
    p_new = np.mean(np.abs(z_off) ** 2, axis=-1)
    p_ref = np.mean(np.abs(ref) ** 2, axis=-1)
    assert np.max(np.abs(p_new - p_ref) / p_ref) < 1e-5
    assert z_on.shape == z_off.shape == (y.shape[0], len(ks), n_env)


def test_off_comb_probe_per_harmonic_offsets():
    """The per-k signed probe offsets (probe_mode='clean') shift per row."""
    from tracking.phase_increment_tracker import _demod_bank

    y, phi, t, ks, stride, n_env, sr = _demod_fixture()
    band_cyc = 6.0 / sr
    bin_hz = sr / (stride * n_env)
    offs = np.array([(9.0 if a % 2 else -12.0) for a in range(len(ks))])
    assert np.allclose(offs / bin_hz, np.rint(offs / bin_hz))  # on the grid
    _, z_off = _demod_bank(y, phi, t, ks, 11.0, stride, n_env, band_cyc, None, offs, sr)
    for a in (0, 1, len(ks) - 1):
        ref = _explicit_off_comb(y, phi, t, [ks[a]], float(offs[a]), stride, n_env, band_cyc)
        assert np.abs(z_off[:, a] - ref[:, 0]).max() < 1e-5 * np.abs(ref).max()  # complex64


def test_zoom_lp_decimate_bank_on_band_is_unchanged():
    """The refactor must not touch the on-comb envelope: bit-identical."""
    from tracking.phase_increment_tracker import _zoom_lp_decimate_bank, zoom_lp_decimate

    rng = np.random.default_rng(3)
    x = (rng.standard_normal((2, 5, 4096)) + 1j * rng.standard_normal((2, 5, 4096))).astype(
        np.complex64
    )
    rows = np.array([2.0, 4.0, 6.0, 8.0, 10.0]) / 16000
    for band_rows in (None, rows):
        plain = zoom_lp_decimate(x, 64, 64, 6.0 / 16000, band_rows)
        on, off = _zoom_lp_decimate_bank(x, 64, 64, 6.0 / 16000, band_rows, 40)
        assert np.array_equal(plain, on)
        assert off is not None and off.shape == on.shape
