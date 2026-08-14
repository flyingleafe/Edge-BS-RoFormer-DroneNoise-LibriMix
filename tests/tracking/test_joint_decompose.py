"""Tests for the v3 JOINT decomposition (:mod:`tracking.joint_decompose`).

Three layers:

- The pure pieces — the Whittaker-Henderson smoother, the order-cell fold, the
  comb mask, the whitening weight, the stitch arithmetic.
- The v2 GUARD: with every joint hook at its neutral value the solver's output
  is bit for bit the v2 output, so the seam cannot have moved the old path.
- The acceptance test (``slow``): one synthetic recording with a KNOWN shaft
  wander, per-track phase noise growing with ``k`` and a smooth colored floor,
  decomposed by v2 and by v3, then judged by the order-cell instrument.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _joint_fixture import K_MAX, N_MIC, SR, make_fixture

from tracking import joint_solve_window
from tracking.decompose import BandwidthSchedule, reconstruct, solve_config, solve_window
from tracking.joint_decompose import (
    JointConfig,
    _order_trend,
    cell_profile,
    corrected_phase,
    global_rate_correction,
    masked_smooth_psd,
    order_cell_profile,
    split_phases,
    theta_rate,
    upsample_env,
    wh_lambda,
    wh_smooth,
    whiten_weights,
    whitened_flatness,
    window_extra_phase,
)
from tracking.vk_tracking import _track_table, env_stride, vk_envelopes

SCHEDULE = "3,0,1.5,3"
#: The ladder the fixture uses. It starts at 3 for the reason in
#: ``JointConfig.k_trust`` and ends at the fixture's own harmonic cap.
LADDER = (3, 12, K_MAX)


# ---------------------------------------------------------------------------
# the pure pieces


def test_wh_smooth_passes_a_constant_and_a_ramp() -> None:
    # A second-difference prior has a two-dimensional null space: a constant and
    # a straight line pass any weight untouched.
    lam = wh_lambda(1.0, 100.0)
    n = 200
    assert wh_smooth(np.full(n, 3.0), lam) == pytest.approx(np.full(n, 3.0), abs=1e-6)
    ramp = np.arange(n, dtype=float) * 0.01
    assert wh_smooth(ramp, lam) == pytest.approx(ramp, abs=1e-5)


def test_wh_lambda_is_the_solver_bandwidth_relation() -> None:
    # Narrower band -> stronger prior, and the smoother must actually reduce a
    # fast wiggle far more than a slow one.
    assert wh_lambda(0.5, 100.0) > wh_lambda(5.0, 100.0)
    t = np.arange(1000) / 100.0
    lam = wh_lambda(1.0, 100.0)
    slow = wh_smooth(np.sin(2 * np.pi * 0.2 * t), lam)
    fast = wh_smooth(np.sin(2 * np.pi * 20.0 * t), lam)
    assert float(np.abs(slow).max()) > 0.8
    assert float(np.abs(fast).max()) < 0.05


def test_wh_smooth_returns_a_short_row_unchanged() -> None:
    y = np.array([1.0, 5.0])
    assert wh_smooth(y, 10.0) == pytest.approx(y)


def test_upsample_env_holds_the_tail() -> None:
    out = upsample_env(np.array([[0.0, 1.0]]), 25, 10)
    assert out.shape == (1, 25)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 10] == pytest.approx(1.0)
    assert out[0, 24] == pytest.approx(1.0)  # held beyond the last knot


def test_cell_profile_reads_a_planted_comb() -> None:
    # A profile that is flat except for a bump at offset +0.1 in every cell must
    # come back with that offset and a depth of the bump's height.
    step = 0.005
    grid = np.arange(0.0, 10.5 + 0.5 * step, step)
    prof = np.ones_like(grid)
    for m in range(1, 10):
        j = int(round((m + 0.1) / step))
        prof[j] = 10.0  # 10 dB over the flat cell
    got = cell_profile(prof, grid, 1, 9, step)
    assert got["peak_offset"] == pytest.approx(0.1, abs=step)
    assert got["depth_db"] == pytest.approx(10.0, abs=0.5)
    assert got["n_cells"] == 9


def test_cell_profile_reads_zero_depth_on_a_flat_profile() -> None:
    step = 0.005
    grid = np.arange(0.0, 10.5 + 0.5 * step, step)
    got = cell_profile(np.ones_like(grid), grid, 1, 9, step)
    assert got["depth_db"] == pytest.approx(0.0, abs=1e-9)
    assert got["excess_db"] is not None


def test_masked_smooth_psd_recovers_a_colored_floor() -> None:
    # No comb at all: the fit must return the injected shape to a fraction of a
    # decibel, which is the null the residual reading is judged against.
    rng = np.random.default_rng(3)
    n_t = 8 * SR
    f = np.fft.rfftfreq(n_t, d=1.0 / SR)
    shape = (1.0 + (f / 200.0) ** 2) ** -0.7
    y = np.stack(
        [np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * shape, n=n_t) for _ in range(2)]
    )
    rates = np.full((2, n_t), 80.0)
    psd = masked_smooth_psd(y, SR, rates, 20, n_fft=4096, n_blocks=2)
    true = np.interp(psd.freq, f, 2.0 * np.log(np.maximum(shape, 1e-30)))
    band = (psd.freq >= 60.0) & (psd.freq <= 0.45 * SR)
    for c in range(2):
        d = (psd.log_s[c].mean(axis=0) - true)[band]
        rms_db = float(np.sqrt(np.mean((d - d.mean()) ** 2))) * 10.0 / np.log(10.0)
        assert rms_db < 1.0, f"mic {c}: {rms_db} dB"
    assert 0.0 < psd.n_masked_frac < 0.6


def test_whiten_weights_are_normalized_and_clamped() -> None:
    freq = np.linspace(0.0, 8000.0, 513)
    log_s = np.tile(-np.log1p(freq / 50.0) * 4.0, (2, 1, 1))
    from tracking.joint_decompose import SmoothPSD

    psd = SmoothPSD(freq=freq, t_block=np.array([1.0]), log_s=log_s)
    rotor, k = _track_table(2, 1, 20)
    r_env = np.full((2, 50), 80.0)
    u = whiten_weights(psd, k, rotor, r_env, np.arange(50) / 100.0, clamp_db=9.0)
    assert u.shape == (len(k), 50)
    assert float(np.mean(np.log(u))) == pytest.approx(0.0, abs=1e-9)
    # The clamp bounds the SPREAD, because the weight is re-centered after it.
    assert float(20 * np.log10(u.max() / u.min())) <= 2 * 9.0 + 1e-6
    # A loud floor must weigh the track DOWN: the low harmonics sit where the
    # planted spectrum is loudest, so their weight is the smallest.
    assert float(u[:5].mean()) < float(u[-5:].mean())


def test_split_phases_recovers_a_planted_shaft_phase() -> None:
    # Envelopes built by hand: unit amplitude, phase exactly k * theta. The
    # k-weighted regression must give theta back.
    fs_env = 100.0
    n = 400
    t = np.arange(n) / fs_env
    theta = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    rotor, k = _track_table(2, 1, 8)
    x = np.exp(1j * k[None, :, None] * theta[None, None, :]) * np.ones((2, len(k), n))
    got = split_phases(
        x, k, rotor, np.ones((len(k), n), dtype=bool), fs_env, k_trust=8, with_psi=False
    )
    for r in range(2):
        rec = got.theta[r] - got.theta[r].mean()
        assert rec == pytest.approx(theta - theta.mean(), abs=0.02)
    assert got.diag["n_trust"] == len(k)


def test_split_phases_drops_a_noise_dominated_track() -> None:
    fs_env = 100.0
    n = 400
    rng = np.random.default_rng(1)
    rotor, k = _track_table(1, 1, 4)
    x = np.ones((1, len(k), n), dtype=complex)
    x[0, 3] = np.exp(1j * rng.uniform(-np.pi, np.pi, n))  # pure noise phase
    got = split_phases(x, k, rotor, np.ones((len(k), n), dtype=bool), fs_env, k_trust=4)
    assert got.diag["n_trust"] == 3
    assert float(np.abs(got.psi[3]).max()) == 0.0


def test_theta_rate_is_the_gauge_free_form() -> None:
    fs_env = 100.0
    n = 500
    ramp = 2.0 * np.pi * 0.25 * np.arange(n) / fs_env  # a constant 0.25 rev/s error
    assert theta_rate(ramp, fs_env)[10:-10] == pytest.approx(0.25, abs=1e-6)
    # Adding any constant changes nothing, which is why the stitch carries this.
    assert theta_rate(ramp + 17.0, fs_env) == pytest.approx(theta_rate(ramp, fs_env))


def test_stitch_arithmetic_puts_two_windows_on_one_carrier() -> None:
    # Two overlapping windows, each holding the SAME physical shaft correction
    # but at its own phase origin. The rate stitch plus the per-window rotation
    # must map both onto the global carrier with no residual phase.
    sr, stride = 16000, 160
    n_t = 16000 * 4
    r_audio = np.full((1, n_t), 80.0)
    phi_hat = 2.0 * np.pi * np.cumsum(r_audio, axis=-1) / sr
    dr_true = 0.2 * np.sin(2 * np.pi * 0.3 * np.arange(n_t) / sr)
    theta_true = 2.0 * np.pi * np.cumsum(dr_true) / sr

    windows = []
    for a0 in (0, 16000 * 2):
        n_env_w = (n_t - a0) // stride
        idx = a0 + np.arange(n_env_w) * stride
        th = theta_true[idx] - theta_true[max(a0 - 1, 0)]
        windows.append(
            {
                "a0": a0,
                "theta": th[None, :],
                "dr": theta_rate(th[None, :], sr / stride),
                "x": np.ones((1, 1, n_env_w), dtype=np.complex64),
            }
        )
    dr_g = global_rate_correction(windows, stride, 0, n_t, ramp=12)
    _, phi_t = corrected_phase(r_audio, dr_g, sr, stride, 0, n_t)
    for w in windows:
        e = window_extra_phase(
            w["theta"], phi_hat, phi_t, int(w["a0"]), stride, int(w["x"].shape[-1])
        )
        # The two carriers agree to a constant, so the rotation is nearly flat.
        assert float(np.abs(e - e.mean()).max()) < 0.05


# ---------------------------------------------------------------------------
# the v2 guard


def test_neutral_hooks_reproduce_the_v2_solve_exactly() -> None:
    rng = np.random.default_rng(7)
    sr, n_t = 8000, 8000
    r = np.full((2, n_t), 50.0)
    phase = 2 * np.pi * np.cumsum(r, axis=-1) / sr
    y = rng.normal(scale=0.05, size=(2, n_t))
    for kk in (1, 2, 3):
        y += (0.4 / kk) * np.cos(kk * phase[0]) + (0.3 / kk) * np.cos(kk * phase[1])
    cfg = solve_config(6, sr=sr, mics=2).vk_config(6)
    base = vk_envelopes(y, r, cfg, k_hi=6)
    n_env = base.x.shape[-1]
    neutral: dict[str, Any] = {
        "phase_offset": np.zeros_like(r),
        "env_rotation": np.zeros((len(base.k), n_env)),
        "data_weight": np.ones((len(base.k), n_env)),
    }
    hooked = vk_envelopes(y, r, cfg, k_hi=6, **neutral)
    assert np.array_equal(hooked.x, base.x)
    assert np.array_equal(hooked.phase, base.phase)
    assert np.array_equal(hooked.bw_track, base.bw_track)


def test_hooks_reject_a_mis_shaped_input() -> None:
    sr, n_t = 8000, 4000
    r = np.full((2, n_t), 50.0)
    y = np.zeros((1, n_t))
    cfg = solve_config(4, sr=sr, mics=1).vk_config(4)
    bad: dict[str, Any] = {"phase_offset": np.zeros((3, n_t))}
    with pytest.raises(ValueError, match="phase_offset"):
        vk_envelopes(y, r, cfg, k_hi=4, **bad)
    bad = {"env_rotation": np.zeros((2, 5))}
    with pytest.raises(ValueError, match="env_rotation"):
        vk_envelopes(y, r, cfg, k_hi=4, **bad)
    bad = {"data_weight": -np.ones((8, 50))}
    with pytest.raises(ValueError, match="non-negative"):
        vk_envelopes(y, r, cfg, k_hi=4, **bad)


# ---------------------------------------------------------------------------
# the acceptance test
#
# Four solves on a 20 s fixture, about 17 s in all. It is NOT ``slow`` marked on
# purpose: it is the only thing that says the alternation works, so it has to
# run by default.


@pytest.fixture(scope="module")
def joint_arms() -> dict[str, object]:
    """v2 and v3 on the same fixture, plus every reading the assertions need."""
    fx = make_fixture()
    y = np.asarray(fx["audio"])
    r_hat = np.asarray(fx["r_hat"])
    cfg = solve_config(K_MAX, sr=SR, mics=N_MIC, bw_rps=1.0, f_max=6000.0)
    sched = BandwidthSchedule.parse(SCHEDULE)
    stride, _ = env_stride(cfg.vk_config(K_MAX))

    env2 = solve_window(y, r_hat, cfg, k_hi=K_MAX, mics=N_MIC, bw_schedule=sched)
    rec2, _ = reconstruct(env2.x, env2.k, env2.rotor, env2.phase, stride)
    res2 = y - rec2

    v3 = joint_solve_window(
        y,
        r_hat,
        cfg,
        k_hi=K_MAX,
        mics=N_MIC,
        jcfg=JointConfig(iters=3, k_trust=LADDER, profile_n_fft=4096),
        bw_schedule=sched,
    )
    prof = {
        "original": order_cell_profile(y, SR, r_hat, n_fft=4096, k_max=K_MAX)["bands"],
        "v2": order_cell_profile(res2, SR, r_hat, n_fft=4096, k_max=K_MAX)["bands"],
        "v3": order_cell_profile(v3.residual, SR, r_hat, n_fft=4096, k_max=K_MAX)["bands"],
    }
    e_tot = float((y**2).sum())
    return {
        "fx": fx,
        "stride": stride,
        "v2_residual_fraction": float((res2**2).sum() / e_tot),
        "v3_residual_fraction": float((v3.residual**2).sum() / e_tot),
        "profiles": prof,
        "v3": v3,
    }


def test_v3_removes_far_more_comb_than_v2(joint_arms: dict[str, object]) -> None:
    prof = joint_arms["profiles"]
    assert isinstance(prof, dict)
    # excess_db is the ABSOLUTE comb power left, in the input's own units, so
    # the difference between two arms is decibels of comb removed. depth_db is a
    # RATIO and it is not the reading to gate on: it can rise while the comb
    # falls, because the floor it is measured against falls faster.
    for band, floor_db in (("k1-9", 12.0), ("k10-24", 8.0)):
        orig = float(prof["original"][band]["excess_db"])
        v2 = float(prof["v2"][band]["excess_db"])
        v3 = float(prof["v3"][band]["excess_db"])
        assert v2 - v3 > floor_db, f"{band}: v2 {v2} dB, v3 {v3} dB"
        assert orig - v3 > orig - v2, f"{band}: v3 must remove more than v2"
    # At k1-9 the ratio reading agrees, and there v2 is the one that leaks.
    assert float(prof["v2"]["k1-9"]["depth_db"]) > 3.5
    assert float(prof["v3"]["k1-9"]["depth_db"]) < 2.0


def test_v3_residual_energy_approaches_the_floor(joint_arms: dict[str, object]) -> None:
    v2 = float(joint_arms["v2_residual_fraction"])  # type: ignore[arg-type]
    v3 = float(joint_arms["v3_residual_fraction"])  # type: ignore[arg-type]
    assert v3 < v2 / 10.0, f"v2 {v2}, v3 {v3}"


def test_recovered_shaft_phase_matches_the_truth(joint_arms: dict[str, object]) -> None:
    fx = joint_arms["fx"]
    assert isinstance(fx, dict)
    res = joint_arms["v3"]
    theta_true = np.asarray(fx["theta"])
    n_t = theta_true.shape[-1]
    theta_hat = upsample_env(res.theta_env, n_t, int(joint_arms["stride"]))  # type: ignore[union-attr,arg-type]
    for i in range(theta_true.shape[0]):
        a = theta_true[i] - theta_true[i].mean()
        b = theta_hat[i] - theta_hat[i].mean()
        corr = float(np.corrcoef(a, b)[0, 1])
        assert corr > 0.95, f"rotor {i}: correlation {corr}"
        # And the SIZE has to be right too, not only the shape.
        assert float(b.std()) == pytest.approx(float(a.std()), rel=0.15)


def test_recovered_floor_matches_the_truth(joint_arms: dict[str, object]) -> None:
    fx = joint_arms["fx"]
    assert isinstance(fx, dict)
    res = joint_arms["v3"]
    psd = res.psd  # type: ignore[union-attr]
    n_t = np.asarray(fx["audio"]).shape[-1]
    f_full = np.fft.rfftfreq(n_t, d=1.0 / SR)
    true = np.interp(
        psd.freq, f_full, 2.0 * np.log(np.maximum(np.asarray(fx["floor_shape"]), 1e-30))
    )
    band = (psd.freq >= 60.0) & (psd.freq <= 0.45 * SR)
    for c in range(psd.log_s.shape[0]):
        d = (psd.log_s[c].mean(axis=0) - true)[band]
        rms_db = float(np.sqrt(np.mean((d - d.mean()) ** 2))) * 10.0 / np.log(10.0)
        assert rms_db < 1.5, f"mic {c}: {rms_db} dB"


def test_whitening_flattens_the_residual(joint_arms: dict[str, object]) -> None:
    res = joint_arms["v3"]
    flat = whitened_flatness(res.residual, SR, res.psd)  # type: ignore[union-attr]
    assert flat["flatness_whitened_mean"] > 10.0 * flat["flatness_raw_mean"]


def test_every_iteration_lowers_the_residual(joint_arms: dict[str, object]) -> None:
    res = joint_arms["v3"]
    fracs = [s["residual_fraction"] for s in res.iterations]  # type: ignore[union-attr]
    assert fracs == sorted(fracs, reverse=True), fracs
    assert res.iterations[0]["k_trust"] == LADDER[0]  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# the instrument's own audit: the detrend


def test_cell_profile_reads_zero_on_a_pure_spectral_tilt() -> None:
    """A smooth falling spectrum must NOT read as a comb at offset -0.5.

    This is the bias that produced (and then withdrew) a "half-order comb"
    verdict on both rigs. One cell spans a whole order, the floor falls across
    it, and normalizing by the cell's own scalar median leaves that ramp in — so
    the fold peaks at the cell's low edge, which is the half-integer position.
    """
    step = 0.005
    grid = np.arange(0.0, 10.5 + 0.5 * step, step)
    tilt = np.exp(-0.6 * grid)  # a smooth 2.6 dB-per-order fall, no line at all
    naive = cell_profile(tilt, grid, 1, 9, step)
    assert naive["peak_offset"] == pytest.approx(-0.5, abs=0.02)
    assert naive["depth_db"] > 1.0  # the artefact the campaign chased
    trend = _order_trend(tilt, step, 1.0)
    fixed = cell_profile(tilt, grid, 1, 9, step, trend=trend)
    assert fixed["depth_db"] < 0.05, fixed["depth_db"]


def test_detrend_keeps_a_real_line() -> None:
    # The same tilt with a planted line at +0.1 orders: the detrend must remove
    # the ramp and leave the line standing.
    step = 0.005
    grid = np.arange(0.0, 10.5 + 0.5 * step, step)
    prof = np.exp(-0.6 * grid)
    for m in range(1, 10):
        prof[int(round((m + 0.1) / step))] *= 6.0
    got = cell_profile(prof, grid, 1, 9, step, trend=_order_trend(prof, step, 1.0))
    assert got["peak_offset"] == pytest.approx(0.1, abs=2 * step)
    assert got["depth_db"] > 5.0


def test_order_cell_profile_detrends_by_default() -> None:
    # End to end on white noise shaped by a steep smooth tilt: no comb anywhere,
    # so every band must read about zero depth.
    rng = np.random.default_rng(5)
    n_t = 6 * SR
    f = np.fft.rfftfreq(n_t, d=1.0 / SR)
    shape = (1.0 + (f / 80.0) ** 2) ** -0.9
    y = np.stack(
        [np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * shape, n=n_t) for _ in range(2)]
    )
    rates = np.full((2, n_t), 80.0)
    prof = order_cell_profile(y, SR, rates, n_fft=4096, k_max=20, bands=((1, 9), (10, 20)))
    for band, d in prof["bands"].items():
        assert d["depth_db"] < 0.35, f"{band}: {d['depth_db']} at {d['peak_offset']}"


# ---------------------------------------------------------------------------
# the stitch seam


def test_theta_rate_holds_its_edges() -> None:
    # np.gradient is one-sided at the two ends, so the edge frames carry a
    # different estimator from the interior. They are held instead, because the
    # stitch is built on this array and one wrong frame becomes a seam.
    fs_env = 100.0
    t = np.arange(300) / fs_env
    theta = np.sin(2 * np.pi * 0.3 * t)[None, :]
    dr = theta_rate(theta, fs_env)
    assert dr[0, 0] == pytest.approx(dr[0, 1])
    assert dr[0, -1] == pytest.approx(dr[0, -2])
    assert dr.shape == theta.shape


def test_split_phases_extrapolates_over_the_tapered_edges() -> None:
    """The shaft estimate must not fit the solver's tapered window ends.

    A planted shaft phase with a fabricated spike in the first frames of the
    envelope (what a tapered edge produces) must come back without the spike,
    because the smoother's data weight fades there and the prior extrapolates.
    """
    fs_env = 100.0
    n = 400
    t = np.arange(n) / fs_env
    theta = 0.2 * np.sin(2 * np.pi * 0.15 * t)
    rotor, k = _track_table(1, 1, 6)
    x = np.exp(1j * k[None, :, None] * theta[None, None, :]) * np.ones((2, len(k), n))
    x[:, :, :4] *= np.exp(1j * 1.5)  # a fabricated jump inside the taper
    got = split_phases(
        x, k, rotor, np.ones((len(k), n), dtype=bool), fs_env, k_trust=6, with_psi=False
    )
    dr = theta_rate(got.theta[0], fs_env)
    interior = float(np.abs(dr[10:-10]).max())
    assert float(np.abs(dr).max()) < 4.0 * max(interior, 1e-6)
