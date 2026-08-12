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


# ---------------------------------------------------------------------------
# the v2 bandwidth schedule


def test_bandwidth_schedule_parses_and_round_trips() -> None:
    sched = D.BandwidthSchedule.parse("3,0.05,1.5,10")
    assert sched is not None
    assert sched == D.BandwidthSchedule(3.0, 0.05, 1.5, 10.0)
    assert D.BandwidthSchedule.parse(sched.text()) == sched
    assert D.BandwidthSchedule.parse("") is None
    assert D.BandwidthSchedule.parse("  ") is None
    with pytest.raises(ValueError, match="4 comma-separated"):
        D.BandwidthSchedule.parse("3,0.05")
    with pytest.raises(ValueError, match="must be positive"):
        D.BandwidthSchedule(0.0, 0.1, 1.0, 6.0)


def test_line_separations_read_the_nearest_neighbouring_line() -> None:
    # Two rotors at 50 and 61 rev/s: the k-th line of one sits k Hz from the
    # k-th of the other only when the combs interleave, so the answer is the
    # minimum over EVERY other line, not over the same harmonic.
    rotor = np.array([0, 0, 0, 1, 1, 1])
    k = np.array([1, 2, 3, 1, 2, 3])
    r = np.stack([np.full(8, 50.0), np.full(8, 61.0)])
    sep = D.line_separations(r, rotor, k)
    # lines 50, 100, 150 | 61, 122, 183. The third line of rotor 0 (150 Hz) is
    # closest to the SECOND of rotor 1 (122 Hz), not to its third — which is the
    # whole reason the reading is a minimum over every line and not over k.
    assert sep.tolist() == pytest.approx([11.0, 22.0, 28.0, 11.0, 22.0, 33.0])
    assert D.line_separations(r[:1], np.array([0]), np.array([1]))[0] == np.inf


def test_schedule_bandwidths_obey_the_floor_and_both_caps() -> None:
    sched = D.BandwidthSchedule(2.0, 0.5, 1.0, 6.0)
    k = np.array([1, 4, 20, 20])
    sep = np.array([100.0, 100.0, 100.0, 1.5])  # the last track is crowded
    got = D.schedule_bandwidths(k, sep, sched, 3.0)
    # k=1: the law asks 2.5 but the base 3.0 is a FLOOR, never a target
    # k=4: 4.0, free; k=20: 12.0 clipped by bw_abs_max; last: by 1.0 * sep
    assert got.tolist() == pytest.approx([3.0, 4.0, 6.0, 3.0])


def test_base_bandwidths_reproduce_the_solvers_own_clamp() -> None:
    # A second implementation of tracking.vk_envelopes' per-group bandwidth
    # clamp, so it is diffed against the solver instead of trusted. Two combs:
    # a SPARSE one (the clamp does not bite, the k-scaled request survives) and
    # a DENSE one (the clamp floors every track at VKConfig.bw_hz).
    for rates, k_hi in (((50.0, 61.0), 6), ((60.0, 60.4, 60.8, 61.2), 12)):
        r = np.stack([np.full(SR, v) for v in rates])
        audio = np.zeros((1, SR))
        cfg = D.solve_config(k_hi, sr=SR, mics=1)
        env = D.solve_window(audio, r, cfg, k_hi=k_hi)
        want = D.base_bandwidths(r, k_hi, cfg)
        assert want.tolist() == pytest.approx(env.bw_track.tolist(), rel=1e-9)


def test_schedule_reaches_the_bandwidth_it_asks_for() -> None:
    # The Tuma round trip: the schedule is applied through rho^2, so the band
    # the solver ACTUALLY used (Envelopes.bw_track) must come back as the
    # scheduled one, on a dense comb and on a sparse one alike.
    for rates, k_hi in (((50.0, 61.0), 6), ((60.0, 60.4, 60.8, 61.2), 12)):
        r = np.stack([np.full(SR, v) for v in rates])
        audio = np.zeros((1, SR))
        cfg = D.solve_config(k_hi, sr=SR, mics=1)
        sched = D.BandwidthSchedule(3.0, 0.2, 1.5, 8.0)
        env = D.solve_window(audio, r, cfg, k_hi=k_hi, bw_schedule=sched)
        want = D.schedule_bandwidths(
            env.k,
            D.line_separations(r, env.rotor, env.k),
            sched,
            D.base_bandwidths(r, k_hi, cfg),
        )
        assert env.bw_track.tolist() == pytest.approx(want.tolist(), rel=1e-6)
        assert (env.bw_track >= D.base_bandwidths(r, k_hi, cfg) - 1e-9).all()


#: Four rotor rates whose 80 lines fall in ONE coupling group AND contain a
#: near-coincidence, so the solver's separation clamp floors the whole group at
#: ``VKConfig.bw_hz`` = 1 Hz. That is the DREGON regime the v2 schedule exists
#: for; a sparser comb keeps the k-scaled band and needs no schedule. The
#: fundamentals stay 4.4 Hz apart, so the low band is not degenerate as well.
DENSE_RATES = (35.28, 40.75, 45.41, 49.81)


def _jittered_comb(
    k_max: int = 20, dur_s: float = 4.0, sigma: float = 0.005, floor: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """``(audio (2, T), rates (4, T))`` — a comb whose LINEWIDTH scales with k.

    Each shaft phase carries its own Ornstein-Uhlenbeck jitter, so harmonic
    ``k`` sees ``k`` times that phase and its line is ``k`` times as wide. That
    is the physics the v2 schedule is matched to. The rates handed back are the
    CLEAN ones, so the jitter is exactly what the envelopes have to absorb.
    """
    rng = np.random.default_rng(7)
    n_t = int(dur_s * SR)
    rates = np.stack([np.full(n_t, v) for v in DENSE_RATES])
    phase = 2.0 * np.pi * np.cumsum(rates, axis=-1) / float(SR)
    audio = rng.normal(scale=floor, size=(2, n_t))
    theta = 8.0 / SR  # OU reversion: a finite linewidth, not a random walk's
    for r in range(len(DENSE_RATES)):
        noise = rng.normal(scale=sigma, size=n_t)
        jit = np.zeros(n_t)
        for i in range(1, n_t):  # the recursion is the definition; n_t is small
            jit[i] = jit[i - 1] * (1.0 - theta) + noise[i]
        for k in range(1, k_max + 1):
            for c, gain in enumerate((1.0, 0.7)):
                audio[c] += gain * (0.5 / k) * np.cos(k * (phase[r] + jit) + 0.3 * c)
    return audio, rates


def _order_contrast(
    resid: np.ndarray, rates: np.ndarray, bands: tuple[tuple[int, int], ...]
) -> dict[str, float]:
    """On-order over half-order power, in decibels, per harmonic band.

    The tuning campaign's probe (``docs/experiments/vk-decomposition.md`` v2):
    a 2048 / 512 Hann spectrogram interpolated onto the ORDER grid, on-order
    within +-0.06 of a whole order against half-order within +-0.06 of a half.
    Zero means the residual carries no comb structure in that band — neither
    leaked (positive) nor over-subtracted (negative).
    """
    n_fft, hop = 2048, 512
    w = np.hanning(n_fft + 1)[:n_fft]
    y = np.atleast_2d(resid)
    starts = np.arange(0, y.shape[-1] - n_fft + 1, hop)
    spec = np.fft.rfft(np.stack([y[:, s : s + n_fft] * w for s in starts], axis=1), axis=-1)
    power = (np.abs(spec) ** 2).mean(axis=0).T
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    grid = np.arange(0.5, bands[-1][1] + 0.5 + 1e-9, 0.01)
    prof = np.zeros_like(grid)
    for row in np.atleast_2d(rates):
        f0 = row[np.clip(starts + n_fft // 2, 0, y.shape[-1] - 1)]
        for j, rate in enumerate(f0):
            prof += np.interp(grid * float(rate), freqs, power[:, j], left=0.0, right=0.0)
    near_int = np.abs(grid - np.round(grid))
    near_half = np.abs((grid - 0.5) - np.round(grid - 0.5))
    out = {}
    for lo, hi in bands:
        on = (near_int <= 0.06) & (np.round(grid) >= lo) & (np.round(grid) <= hi)
        half = (near_half <= 0.06) & (np.round(grid - 0.5) >= lo) & (np.round(grid - 0.5) < hi)
        out[f"k{lo}-{hi}"] = 10.0 * np.log10(prof[on].mean() / prof[half].mean())
    return out


def test_tuned_schedule_neither_leaks_nor_over_subtracts() -> None:
    # THE guard on both failure modes at once. A flat 1 Hz band cannot follow a
    # k-scaled linewidth, so comb structure LEAKS into the residual (positive
    # contrast); a band far wider than the line swallows the floor around it and
    # NOTCHES the residual (negative contrast). The tuned schedule must land on
    # zero in EVERY band, between those two.
    bands = ((1, 9), (10, 20))
    audio, rates = _jittered_comb()
    cfg = D.solve_config(20, sr=SR, mics=2)
    stride = int(round(SR / 100.0))
    # The premise: this comb reproduces the v1 regime, one flat 1 Hz band for
    # every track. Without it the arms below would measure nothing.
    assert D.base_bandwidths(rates, 20, cfg).tolist() == pytest.approx([1.0] * 80)

    def residual_of(sched: D.BandwidthSchedule | None) -> dict[str, float]:
        env = D.solve_window(audio, rates, cfg, k_hi=20, mics=2, bw_schedule=sched)
        recon, _ = D.reconstruct(env.x, env.k, env.rotor, D.shaft_phase(rates, SR), stride)
        return _order_contrast(audio - np.asarray(recon, dtype=np.float64), rates, bands)

    flat = residual_of(None)
    tuned = residual_of(D.BandwidthSchedule(3.0, 0.0, 1.5, 3.0))  # the tuned v2 schedule
    over = residual_of(D.BandwidthSchedule(8.0, 0.0, 3.0, 16.0))  # deliberately too wide

    assert flat["k10-20"] > 1.0, f"the flat band must LEAK a k-scaled linewidth: {flat}"
    for name, got in tuned.items():
        assert abs(got) < 0.5, f"{name}: tuned residual contrast {got:.2f} dB, want 0 +- 0.5"
    assert abs(tuned["k10-20"]) < abs(flat["k10-20"])
    assert over["k10-20"] < -1.0, f"a band far wider than the line must NOTCH: {over}"


# ---------------------------------------------------------------------------
# the readings, continued


def test_residual_tones_finds_an_injected_foreign_tone() -> None:
    # A tone that is NOT a rotor order must come back with a large prominence
    # and a non-integer order — the discrimination the report is built on.
    rng = np.random.default_rng(3)
    n_t = 8 * SR
    t = np.arange(n_t) / float(SR)
    rates = np.full((1, n_t), 60.0)
    foreign = 60.0 * 7.37  # order 7.37 of the rotor: not a harmonic
    audio = rng.normal(scale=0.01, size=(1, n_t)) + 0.3 * np.cos(2.0 * np.pi * foreign * t)
    got = D.residual_tones(audio, SR, rates, segment_s=8.0, nperseg=8192)

    assert len(got["segments"]) == 1
    peaks = got["segments"][0]["peaks"]
    top = peaks[0]
    assert top["freq_hz"] == pytest.approx(foreign, abs=3.0)
    assert top["prominence_db"] > 6.0
    assert top["order"] == pytest.approx(7.37, abs=0.05)
    assert top["order_dist"] > 0.3  # squarely between two harmonics
    assert "non-comb" in got["note"].lower() or "NOT rotor" in got["note"]


def test_residual_tones_reports_a_rotor_harmonic_as_on_order() -> None:
    rng = np.random.default_rng(4)
    n_t = 8 * SR
    t = np.arange(n_t) / float(SR)
    rates = np.full((1, n_t), 60.0)
    audio = rng.normal(scale=0.01, size=(1, n_t)) + 0.3 * np.cos(2.0 * np.pi * 5.0 * 60.0 * t)
    got = D.residual_tones(audio, SR, rates, segment_s=8.0, nperseg=8192)
    top = got["segments"][0]["peaks"][0]
    assert top["order"] == pytest.approx(5.0, abs=0.05)
    assert top["order_dist"] < 0.05


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
