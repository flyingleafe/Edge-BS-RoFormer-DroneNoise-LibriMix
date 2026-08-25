"""Tests for the stochastic rotor-noise model."""

from __future__ import annotations

import numpy as np
import pytest

from data_processing import stochastic_rotor_noise as srn

SR = 16000


def _params(seed: int = 0, **overrides):
    params = srn.sample_params(
        np.random.default_rng(seed), n_rotors=4, n_harmonics=40, sample_rate=SR
    )
    return params.with_(**overrides) if overrides else params


def _cruise(n_seconds: float = 2.0, speed: float = 80.0) -> np.ndarray:
    n = int(n_seconds * SR)
    return np.tile(np.array([[speed]]), (4, n)) + np.array([0.0, 1.5, -2.0, 0.7])[:, None]


def _power_spectrum(x: np.ndarray, n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    hop = n_fft // 4
    window = np.hanning(n_fft + 1)[:n_fft]
    n_frames = (x.size - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    power = (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(axis=0)
    return power, np.fft.rfftfreq(n_fft, 1.0 / SR)


def test_gp_statistics_match_the_requested_kernel():
    rng = np.random.default_rng(0)
    series = srn.sample_gp(rng, 400, 256, dt=0.05, tau=1.0, std=3.0)
    assert series.shape == (400, 256)
    assert np.std(series) == pytest.approx(3.0, rel=0.1)
    # A process with a one-second correlation time and a 50 ms step is strongly
    # correlated between adjacent samples and weakly correlated 40 steps out.
    near = np.corrcoef(series[:, :-1].ravel(), series[:, 1:].ravel())[0, 1]
    far = np.corrcoef(series[:, :-40].ravel(), series[:, 40:].ravel())[0, 1]
    assert near > 0.99
    assert abs(far) < 0.3


def test_zero_std_gives_a_constant_process():
    out = srn.sample_gp(np.random.default_rng(0), 3, 64, dt=0.1, tau=1.0, std=0.0)
    assert np.all(out == 0.0)


def test_realized_spectrum_follows_the_model():
    params = _params(1)
    audio, diag = srn.synthesize(params, _cruise(), rng=np.random.default_rng(2), n_mics=1)
    power, freqs = _power_spectrum(audio[0].astype(np.float64))
    model = 10.0 ** (srn.model_psd_db(diag, 0).mean(axis=0) / 10.0)
    band = (freqs > 60.0) & (freqs < 7000.0)
    realized_db = 10.0 * np.log10(power[band] / power[band].max())
    model_db = 10.0 * np.log10(model[band] / model[band].max())
    assert np.corrcoef(realized_db, model_db)[0, 1] > 0.95


def test_the_comb_sits_at_multiples_of_the_rotor_speed():
    # One rotor, a clean profile, and a floor far below it: the spectrum's peaks
    # must land on multiples of the speed.
    params = _params(3, floor_mean_db=-45.0).with_(n_rotors=1)
    params = params.with_(profile_db=params.profile_db[:1], gamma0=params.gamma0[:1])
    params = params.with_(gamma_slope=params.gamma_slope[:1] * 0.0 + 0.1)
    speed = 70.0
    rps = np.full((1, 2 * SR), speed)
    audio, _ = srn.synthesize(params, rps, rng=np.random.default_rng(4), n_mics=1)
    power, freqs = _power_spectrum(audio[0].astype(np.float64))
    band = (freqs > 50.0) & (freqs < 3000.0)
    peak_freqs = freqs[band][power[band] > np.percentile(power[band], 99.0)]
    residual = np.abs(peak_freqs / speed - np.round(peak_freqs / speed))
    assert np.median(residual) < 0.12


def test_amplitudes_carry_no_speed_information():
    # The same parameters under two different constant speeds must give the
    # same harmonic levels: the model's amplitudes are drawn independently of
    # the trajectory, which is what keeps the amplitude shortcut closed.
    params = _params(5, amp_rps_exponent=0.0)
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    slow = srn.build_psd(
        params, np.full((4, 60), 60.0), freqs, dt=0.032, rng=np.random.default_rng(6)
    )
    fast = srn.build_psd(
        params, np.full((4, 60), 95.0), freqs, dt=0.032, rng=np.random.default_rng(6)
    )
    assert slow["lines"].sum() == pytest.approx(fast["lines"].sum(), rel=0.05)
    assert np.allclose(slow["floor"], fast["floor"])


def test_stopped_rotors_are_silent():
    params = _params(7)
    rps = np.zeros((4, SR))
    audio, _ = srn.synthesize(
        params, rps, rng=np.random.default_rng(8), n_mics=1, normalize_rms=None
    )
    assert float(np.abs(audio).max()) < 1e-6


def test_line_power_scales_with_the_level_knob():
    params = _params(9)
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    rps = np.full((4, 40), 80.0)
    base = srn.build_psd(params, rps, freqs, dt=0.032, rng=np.random.default_rng(10))
    louder = srn.build_psd(
        params.with_(harm_mean_db=params.harm_mean_db + 10.0),
        rps,
        freqs,
        dt=0.032,
        rng=np.random.default_rng(10),
    )
    assert louder["lines"].sum() / base["lines"].sum() == pytest.approx(10.0, rel=0.02)


def test_floor_shape_is_smooth():
    params = _params(11)
    freqs = np.linspace(30.0, 8000.0, 4000)
    shape = srn.floor_shape_db(params, freqs)
    # No step in the interpolated curve: the largest jump between neighbouring
    # points stays far below the curve's own range.
    assert np.abs(np.diff(shape)).max() < 0.1 * (shape.max() - shape.min())


def test_pool_emits_a_usable_frame():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=1.0, n_harmonics=30, n_mics=2, n_rotors=4
    )
    frame = pool.sample_timeframe(np.random.default_rng(12), 1.0)
    audio = np.asarray(frame["audio"].data)
    rps = np.asarray(frame["rps"].data)
    assert audio.shape == (2, SR)
    assert rps.shape[0] == 4
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.0


def test_pool_is_seed_reproducible():
    pool = srn.StochasticNoisePool(sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1)
    a = pool.sample_timeframe(np.random.default_rng(13), 0.5)
    b = pool.sample_timeframe(np.random.default_rng(13), 0.5)
    assert np.array_equal(np.asarray(a["audio"].data), np.asarray(b["audio"].data))


def test_pool_draws_a_fresh_parameter_set_per_window():
    pool = srn.StochasticNoisePool(sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1)
    rng = np.random.default_rng(14)
    first = pool.render(rng, 0.5)[2]
    second = pool.render(rng, 0.5)[2]
    assert not np.allclose(first.profile_db, second.profile_db)


def test_full_flight_windows_move_along_one_flight():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=1.0, n_harmonics=20, n_mics=1, rps_kind="full_flight"
    )
    rng = np.random.default_rng(15)
    windows = [pool.sample_rps(rng, 1.0) for _ in range(12)]
    # One flight is cached and windowed, so the windows differ and the flight
    # itself runs from the ground to cruise.
    assert len({round(float(w.mean()), 3) for w in windows}) > 6
    assert pool._flight is not None
    assert float(pool._flight.rps.min()) < 5.0
    assert float(pool._flight.rps.max()) > 60.0


def test_registered_as_an_online_mix_engine():
    from data_processing.online_mixing import _build_engine

    engine = _build_engine(
        {"kind": "stochastic", "n_harmonics": 20, "n_mics": 1}, window_s=0.5, sample_rate=SR
    )
    assert isinstance(engine, srn.StochasticNoisePool)
    assert engine.n_harmonics == 20


def test_rps_scale_range_moves_the_whole_trajectory():
    slow = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1, rps_scale_range=(0.5, 0.5)
    )
    plain = srn.StochasticNoisePool(sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1)
    a = slow.sample_rps(np.random.default_rng(20), 0.5)
    b = plain.sample_rps(np.random.default_rng(20), 0.5)
    # The same draw, halved: a stopped rotor stays stopped and every other
    # speed halves, so the comb moves with the label.
    assert np.allclose(a, 0.5 * b)


def test_rps_scale_range_spreads_the_speed_prior():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1, rps_scale_range=(0.6, 1.5)
    )
    rng = np.random.default_rng(21)
    means = np.array([float(pool.sample_rps(rng, 0.5).mean()) for _ in range(24)])
    assert means.max() / max(means.min(), 1e-6) > 1.8


def test_normalize_rms_range_spreads_the_output_level():
    pool = srn.StochasticNoisePool(
        sample_rate=SR,
        duration_s=0.5,
        n_harmonics=20,
        n_mics=1,
        normalize_rms=(0.005, 0.2),
    )
    rng = np.random.default_rng(22)
    levels = []
    for _ in range(16):
        audio = pool.render(rng, 0.5)[0]
        levels.append(float(np.sqrt(np.mean(np.square(audio)))))
    levels = np.array(levels)
    assert levels.min() >= 0.004
    assert levels.max() <= 0.21
    assert levels.max() / levels.min() > 5.0


def test_scalar_normalize_rms_is_exact():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1, normalize_rms=0.04
    )
    audio = pool.render(np.random.default_rng(23), 0.5)[0]
    assert float(np.sqrt(np.mean(np.square(audio)))) == pytest.approx(0.04, rel=1e-4)


@pytest.mark.parametrize("speed", [95.0, 130.0, 220.0])
def test_harmonics_above_nyquist_do_not_escape_the_grid(speed):
    # k * rps runs far past Nyquist at the top of the speed range. Those lines
    # carry no power, and they must not carry an index either: an out-of-range
    # scatter used to make bincount return a longer array than the accumulator.
    params = _params(30, harm_mean_db=0.0)
    rps = np.full((4, SR), speed)
    audio, _ = srn.synthesize(params, rps, rng=np.random.default_rng(31), n_mics=1)
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.0


def test_high_speed_pool_windows_render():
    pool = srn.StochasticNoisePool(
        sample_rate=SR,
        duration_s=1.0,
        n_harmonics=80,
        n_mics=2,
        rps_kind="full_flight",
        rps_scale_range=(0.4, 1.5),
        aggressiveness=(0.8, 2.5),
    )
    rng = np.random.default_rng(32)
    for _ in range(8):
        frame = pool.sample_timeframe(rng, 1.0)
        assert np.isfinite(np.asarray(frame["audio"].data)).all()
