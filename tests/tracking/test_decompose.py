"""Unit tests for :mod:`tracking.decompose` and :func:`tracking.decompose_stage`.

The array core promoted out of ``scripts/vk_decompose.py``. Four things are
pinned here that the driver's own tests cannot see:

- The windowed-application primitives are parameterized (no campaign constant
  baked in), so the telemetry refiner and the decomposition share them.
- :func:`tracking.decompose.reconstruct` follows the SAME rule as
  :func:`tracking.vk_reconstruct` — two implementations of one contract, so
  they are diffed against each other.
- The phase re-reference of :func:`tracking.decompose.stitch_bank` inverts the
  solver's per-window phase origin exactly.
- :func:`tracking.decompose_stage` obeys the Stage contract: the trajectory is
  untouched, the products travel in ``meta``, the ledger is the log entry.
"""

from __future__ import annotations

import numpy as np
import pytest

import tracking as trk
from tracking import decompose as D
from tracking.fitness_vk import FVKConfig, auto_smooth_lambda, to_audio_grid
from tracking.vk_tracking import vk_reconstruct

SR = 8000
HOP_S = 0.032


# ---------------------------------------------------------------------------
# the windowed-application primitives


def test_window_bounds_tiles_and_right_aligns() -> None:
    bounds = D.window_bounds(1996, 16.0, 12.0, HOP_S)
    assert bounds[0] == (0, 500)
    assert bounds[-1] == (1496, 1996)
    covered = np.zeros(1996, dtype=bool)
    for i0, i1 in bounds:
        covered[i0:i1] = True
    assert covered.all()


def test_window_bounds_reads_its_own_frame_hop() -> None:
    # Twice the frame hop halves the frame count of the same window length.
    assert D.window_bounds(1000, 16.0, 16.0, 0.064)[0] == (0, 250)
    assert D.window_bounds(120, 16.0, 12.0, HOP_S) == [(0, 120)]


def test_window_span_snaps_both_ends_to_the_stride() -> None:
    ft = D.frame_grid(16000 * 40, 16000, HOP_S)
    a0, a1 = D.window_span(ft, 125, 250, 16000 * 40, 160, 16000, HOP_S)
    assert a0 % 160 == 0
    assert (a1 - a0) % 160 == 0
    assert a0 == 64000


def test_fade_weights_ramp_floor_and_symmetry() -> None:
    w = D.fade_weights(10, 4)
    assert w[0] == pytest.approx(0.2)
    assert w[5] == pytest.approx(1.0)
    assert (w > 0).all()
    assert np.allclose(w, w[::-1])
    assert np.allclose(D.fade_weights(6, 0), 1.0)


def test_interp_rps_drops_duplicate_stamps_and_clips() -> None:
    got = D.interp_rps(
        np.array([[10.0, 20.0, 99.0, 30.0]]),
        np.array([0.0, 1.0, 1.0, 2.0]),
        np.array([-1.0, 0.5, 1.5, 5.0]),
    )
    assert got.dtype == np.float64
    assert got[0].tolist() == pytest.approx([10.0, 15.0, 25.0, 30.0])


def test_to_audio_grid_is_the_solvers_own_carrier() -> None:
    # decompose re-exports the objective module's interpolation on purpose: the
    # carrier a window is decomposed at must be the carrier it was scored at.
    assert D.to_audio_grid is to_audio_grid
    r = np.array([[50.0, 60.0]])
    got = D.to_audio_grid(r, np.array([0.0, 1.0]), 5, 4.0)
    assert got[0].tolist() == pytest.approx([50.0, 52.5, 55.0, 57.5, 60.0])


# ---------------------------------------------------------------------------
# the solve


def _synth(dur_s: float = 1.0, rate: float = 60.0, k_max: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """``(audio (1, T), rates (1, T))`` — one rotor, a flat comb, a white floor."""
    rng = np.random.default_rng(0)
    n_t = int(dur_s * SR)
    rates = np.full((1, n_t), rate)
    phase = 2.0 * np.pi * np.cumsum(rates, axis=-1) / float(SR)
    audio = rng.normal(scale=0.01, size=(1, n_t))
    for k in range(1, k_max + 1):
        audio[0] += (0.4 / k) * np.cos(k * phase[0] + 0.2 * k)
    return audio, rates


def test_group_plan_reports_the_memory_law() -> None:
    rates = np.stack([np.full(2 * SR, r) for r in (50.0, 61.0)])
    cfg = D.solve_config(8, sr=SR, mics=2)
    plan = D.group_plan(rates, 8, cfg)
    assert plan["n_tracks"] == 16
    g, n_env = plan["max_group"], plan["n_env"]
    assert plan["banded_gb"] == round(2 * (2 * g + 1) * g * n_env * 16 / 1e9, 3)


def test_solve_config_keeps_a_modelled_harmonic_under_nyquist() -> None:
    # The 6 kHz campaign ceiling must not sit on top of a low sample rate.
    assert D.solve_config(80, sr=SR, mics=1).f_max == pytest.approx(0.375 * SR)
    assert D.solve_config(80, sr=16000, mics=1).f_max == pytest.approx(6000.0)


def test_reconstruct_follows_the_vk_reconstruct_rule() -> None:
    # ONE interpolation contract, two implementations: the stitched one takes
    # the global phase and returns per-track energies, but must otherwise agree
    # sample for sample with the solver's own reconstruction.
    audio, rates = _synth()
    cfg = D.solve_config(5, sr=SR, mics=1)
    env = D.solve_window(audio, rates, cfg, k_hi=5)
    stride = int(round(SR / env.fs_env))
    recon, energy = D.reconstruct(env.x, env.k, env.rotor, env.phase, stride)
    ref = vk_reconstruct(env, n_samples=recon.shape[-1])
    assert np.abs(recon - ref).max() < 1e-4 * float(np.abs(ref).max())
    assert energy.shape == (len(env.k),)
    assert energy.sum() > 0


def test_reconstruct_is_a_linear_interpolation_of_the_envelope() -> None:
    stride, n_env = 10, 5
    x = np.ones((1, 1, n_env), dtype=np.complex64)
    phase = np.linspace(0.0, 3.0, n_env * stride)[None, :]
    recon, energy = D.reconstruct(x, np.array([1]), np.array([0]), phase, stride)
    assert recon[0] == pytest.approx(np.cos(phase[0]), abs=1e-6)
    assert energy[0] == pytest.approx(float((np.cos(phase[0]) ** 2).sum()), rel=1e-5)


def test_stitch_bank_inverts_the_per_window_phase_origin() -> None:
    # The solver's phase starts at the window, so a window at sample a0 holds
    # the global track times exp(+j k Phi(a0 - 1)). Two overlapping windows
    # built that way must stitch back to the global bank; without the
    # re-reference the cross-fade cancels them.
    stride, n_env = 100, 40
    rates = np.full((1, n_env * stride), 60.0)
    phi = D.shaft_phase(rates, SR)
    k = np.array([1, 2], dtype=np.int64)
    rotor = np.zeros(2, dtype=np.int64)
    rng = np.random.default_rng(1)
    x_global = (rng.normal(size=(1, 2, n_env)) + 1j * rng.normal(size=(1, 2, n_env))).astype(
        np.complex64
    )

    windows = []
    for j0, j1 in ((0, 25), (15, 40)):
        a0 = j0 * stride
        shift = np.zeros(1) if a0 == 0 else phi[:, a0 - 1]
        local = x_global[:, :, j0:j1] * np.exp(1j * k[None, :, None] * shift[rotor][None, :, None])
        windows.append(
            {
                "a0": a0,
                "x": local.astype(np.complex64),
                "valid": np.ones((2, j1 - j0), dtype=bool),
                "rotor": rotor,
                "k": k,
            }
        )
    st = D.stitch_bank(windows, phi, stride, ramp=10)
    assert st["n_env"] == n_env
    assert np.abs(st["x"] - x_global).max() < 2e-3 * float(np.abs(x_global).max())
    assert st["covered"].all()


def test_phase_reference_deviation_is_zero_for_the_global_carrier() -> None:
    rates = np.full((1, 4 * SR), 60.0)
    phi = D.shaft_phase(rates, SR)
    dev = D.phase_reference_deviation(rates, phi, SR, SR, seconds=1.0)
    assert dev < 1e-8


# ---------------------------------------------------------------------------
# the readings


def test_track_bands_partition_the_harmonics() -> None:
    masks = D.track_bands(np.arange(1, 81))
    assert sorted(masks) == ["k1-9", "k10-24", "k25-49", "k50-80"]
    assert sum(int(m.sum()) for m in masks.values()) == 80


def test_reference_mic_picks_the_loudest_channel() -> None:
    audio = np.stack([np.ones(100), 3.0 * np.ones(100), np.zeros(100)])
    assert D.reference_mic(audio, -1) == 1
    assert D.reference_mic(audio, 0) == 0


def test_energy_ledger_names_the_cross_term() -> None:
    led = D.energy_ledger(
        np.array([[1.0, 1.0, 1.0, 1.0]]),
        np.array([[0.5, 0.5, 0.5, 0.5]]),
        np.array([0.5, 0.6]),
        np.array([1, 12]),
    )
    assert led["total"] == pytest.approx(4.0)
    assert led["residual"] == pytest.approx(1.0)
    assert led["cross_term"] == pytest.approx(4.0 - 1.0 - 1.1)
    assert led["band_share_of_tracks"]["k50-80"] == 0.0


def test_rank_one_share_separates_common_from_independent_drift() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(size=(1, 4000))
    k = np.arange(1, 13)[:, None]
    assert D.rank_one_share(k * common)["lambda1_share"] == pytest.approx(1.0, abs=1e-6)
    assert D.rank_one_share(rng.normal(size=(12, 4000)))["lambda1_share"] < 0.15
    assert D.rank_one_share(np.zeros((10, 4)))["lambda1_share"] is None


# ---------------------------------------------------------------------------
# the portable smoothness weight


def test_auto_smooth_lambda_holds_the_prior_to_half_the_data_term() -> None:
    n_t, stride = 4000, 160
    flat = np.full((1, n_t), 60.0)
    lam, p0 = auto_smooth_lambda(flat, stride)
    assert p0 == pytest.approx(0.0, abs=1e-12)
    assert lam == 1.0  # a quiet init keeps the calibrated weight

    ramp = np.linspace(5.0, 90.0, n_t)[None, :]
    lam, p0 = auto_smooth_lambda(ramp, stride)
    assert p0 > 0.5
    assert lam == pytest.approx(0.5 / p0)
    assert lam * p0 == pytest.approx(0.5)


def test_optimize_trajectory_reports_the_auto_weight() -> None:
    audio, rates = _synth(dur_s=0.5)
    ft = np.arange(0.0, 0.5, 0.032)
    r = np.full((1, ft.size), 60.0)
    cfg = FVKConfig(sr=SR, k_max=3, max_channels=1, f_max=0.375 * SR)
    _, diag = trk.optimize_trajectory(
        audio, SR, r, ft, cfg, schedule=(trk.FVKStage(3, 1.0, 1),), smooth_lambda="auto"
    )
    assert diag["prior_init"] is not None
    assert diag["smooth_lambda"] == pytest.approx(
        auto_smooth_lambda(to_audio_grid(r, ft, audio.shape[-1], SR), cfg.stride)[0]
    )
    with pytest.raises(ValueError, match="'auto'"):
        trk.optimize_trajectory(audio, SR, r, ft, cfg, smooth_lambda="nope")


# ---------------------------------------------------------------------------
# the Stage


def test_decompose_stage_leaves_the_trajectory_and_reports_the_ledger() -> None:
    audio, rates = _synth(dur_s=1.0)
    ft = np.arange(0.0, 1.0, 0.032)
    r = np.full((1, ft.size), 60.0)
    frame = trk.tracking_frame(audio, SR, rps=r, frame_times=ft, rps_meas=r, dtype=np.float64)
    out = trk.decompose_stage(D.solve_config(5, sr=SR, mics=1))(frame)

    r_out, ft_out = trk.get_rps(out)
    assert np.array_equal(r_out, r)  # the trajectory is NOT changed
    assert np.allclose(ft_out, ft)

    entry = out["meta"]["tracking"][-1]
    assert entry["stage"] == "decompose"
    assert entry["k_hi"] == 5
    assert entry["n_tracks"] == 5
    assert 0.0 < entry["track_fraction"] <= 1.0
    assert entry["residual_fraction"] < 0.2  # a clean synthetic comb is captured

    seam = out["meta"]["decompose"]
    assert set(seam) == {"envelopes", "phase", "recon", "track_energy"}
    # The split is EXACT: audio = recon + (audio - recon), by definition.
    y = np.asarray(trk.get_audio(out)[0], dtype=np.float64)
    residual = y - np.asarray(seam["recon"], dtype=np.float64)
    assert np.abs(y - (seam["recon"] + residual)).max() < 1e-9


def test_decompose_stage_composes_after_a_trajectory_stage() -> None:
    audio, rates = _synth(dur_s=1.0)
    ft = np.arange(0.0, 1.0, 0.032)
    r = np.full((1, ft.size), 60.0)
    frame = trk.tracking_frame(audio, SR, rps=r, frame_times=ft, rps_meas=r, dtype=np.float64)
    run = trk.pipeline(
        trk.presmooth_stage(cut_hz=5.0),
        trk.decompose_stage(D.solve_config(4, sr=SR, mics=1), k_hi=4),
    )
    out = run(frame)
    assert [e["stage"] for e in out["meta"]["tracking"]] == ["presmooth", "decompose"]
    assert out["meta"]["tracking"][-1]["k_hi"] == 4
