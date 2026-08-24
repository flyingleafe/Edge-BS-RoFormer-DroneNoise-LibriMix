"""Tests for the rotors-off noise engine and the SNR reference-power floor."""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from data_processing.mixing import scale_source_to_snr
from data_processing.online_mixing import build_noise_stream
from data_processing.silence_noise import DEFAULT_RANGES, SilenceNoisePool

SR = 16000


def _pool(**kw) -> SilenceNoisePool:
    kw.setdefault("sample_rate", SR)
    kw.setdefault("duration_s", 1.0)
    return SilenceNoisePool(**kw)


# ─── construction ─────────────────────────────────────────────────────────────


def test_from_config_reads_the_documented_keys():
    pool = SilenceNoisePool.from_config(
        {"kind": "silence", "n_channels": 4, "types": {"colored": 1.0}},
        duration_s=0.5,
        sample_rate=SR,
    )
    assert pool.n_mics == 4
    assert pool.types == ("colored",)
    assert pool.type_probs.tolist() == [1.0]
    assert pool.sample_rate == SR


def test_from_config_defaults_to_eight_channels_and_three_types():
    pool = SilenceNoisePool.from_config({"kind": "silence"}, duration_s=1.0, sample_rate=SR)
    assert pool.n_mics == 8
    assert set(pool.types) == {"room_tone", "colored", "lf_rumble"}
    assert pool.type_probs.sum() == pytest.approx(1.0)


def test_unknown_type_name_is_rejected():
    with pytest.raises(ValueError, match="unknown silence floor type"):
        _pool(types={"pink_elephant": 1.0})
    with pytest.raises(ValueError, match="unknown silence floor type"):
        _pool(ranges={"pink_elephant": {"rms": [1.0, 2.0]}})
    with pytest.raises(ValueError, match="unknown range key"):
        _pool(ranges={"colored": {"loudness": [1.0, 2.0]}})


def test_pool_is_picklable():
    import pickle

    pool = pickle.loads(pickle.dumps(_pool(n_channels=2)))
    audio, rps, _ = pool.render(np.random.default_rng(0), 0.25)
    assert audio.shape == (2, SR // 4)
    assert rps.shape[0] == 4


# ─── the sampled Frame ────────────────────────────────────────────────────────


def test_sample_timeframe_is_wellformed():
    pool = _pool(n_channels=8)
    tf = pool.sample_timeframe(np.random.default_rng(0), 0.5)
    assert isinstance(tf, td.Frame)
    audio = tf["audio"]
    assert audio.data.shape == (8, int(round(0.5 * SR)))
    assert np.asarray(audio.data).dtype == np.float32
    assert np.isfinite(np.asarray(audio.data)).all()


def test_audio_sample_rate_matches_the_pool():
    tf = _pool(n_channels=2).sample_timeframe(np.random.default_rng(1), 0.25)
    assert int(round(tf["audio"].tindex.sr)) == SR


def test_rps_track_is_always_all_zeros():
    pool = _pool(n_channels=2)
    rng = np.random.default_rng(7)
    for _ in range(12):
        tf = pool.sample_timeframe(rng, 0.25)
        rps = np.asarray(tf["rps"].data)
        assert rps.shape[0] == 4
        assert rps.dtype == np.float32
        assert not rps.any()


@pytest.mark.parametrize("floor_type", ["room_tone", "colored", "lf_rumble"])
def test_per_type_rms_lands_inside_the_configured_range(floor_type: str):
    pool = _pool(n_channels=3, types={floor_type: 1.0})
    lo, hi = DEFAULT_RANGES[floor_type]["rms"]
    rng = np.random.default_rng(11)
    seen = []
    for _ in range(24):
        audio, _rps, params = pool.render(rng, 0.5)
        assert params["floor_type"] == floor_type
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        # The realized RMS is the drawn target: each channel is scaled to it.
        # lf_rumble adds two independent components, whose finite-sample cross
        # term moves the total by a fraction of a percent.
        tol = 2e-2 if floor_type == "lf_rumble" else 1e-3
        assert rms == pytest.approx(params["rms"], rel=tol)
        seen.append(rms)
    assert min(seen) >= lo * 0.95
    assert max(seen) <= hi * 1.05


def test_lf_rumble_energy_sits_below_200_hz():
    pool = _pool(n_channels=2, types={"lf_rumble": 1.0})
    rng = np.random.default_rng(3)
    for _ in range(8):
        audio, _rps, _params = pool.render(rng, 1.0)
        spec = np.abs(np.fft.rfft(audio[0].astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(audio.shape[-1], d=1.0 / SR)
        frac = float(spec[freqs < 200.0].sum() / spec.sum())
        assert frac > 0.9


def test_colored_floor_is_broadband_unlike_the_rumble():
    rng = np.random.default_rng(5)
    colored = _pool(n_channels=1, types={"colored": 1.0})
    audio, _rps, _params = colored.render(rng, 1.0)
    spec = np.abs(np.fft.rfft(audio[0].astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(audio.shape[-1], d=1.0 / SR)
    assert float(spec[freqs >= 200.0].sum() / spec.sum()) > 0.05


def test_channels_are_decorrelated():
    pool = _pool(n_channels=8, types={"colored": 1.0})
    audio, _rps, _params = pool.render(np.random.default_rng(2), 1.0)
    corr = np.corrcoef(audio.astype(np.float64))
    off = corr[~np.eye(8, dtype=bool)]
    assert float(np.abs(off).max()) < 0.2


def test_build_noise_stream_dispatches_silence():
    stream, ceiling = build_noise_stream(
        {"kind": "silence", "n_channels": 8},
        sample_rate=SR,
        window_s=0.5,
        seed=0,
    )
    frame = next(iter(stream))
    assert frame["audio"].data.shape == (8, int(round(0.5 * SR)))
    assert not np.asarray(frame["rps"].data).any()
    assert ceiling == 8


# ─── the SNR reference-power floor ────────────────────────────────────────────


def _noise(rms: float, n: int = 2048, channels: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((channels, n))
    x /= np.sqrt(np.mean(x**2, axis=1, keepdims=True))
    return (x * rms).astype(np.float32)


def test_floor_none_reproduces_the_unfloored_scale():
    source = _noise(0.5, seed=1)
    noise = _noise(0.001, seed=2)
    for per_channel in (False, True):
        base = scale_source_to_snr(source, noise, -6.0, per_channel=per_channel)
        same = scale_source_to_snr(
            source, noise, -6.0, per_channel=per_channel, ref_power_floor=None
        )
        assert np.array_equal(base, same)


def test_quiet_noise_gets_the_floor_and_loud_noise_does_not():
    source = _noise(0.5, seed=1)
    floor_rms = 0.02
    floor = floor_rms**2
    snr_db = -10.0
    expected_power = floor * 10.0 ** (snr_db / 10.0)

    quiet = _noise(0.001, seed=2)
    scaled = scale_source_to_snr(source, quiet, snr_db, ref_power_floor=floor)
    assert float(np.mean(scaled.astype(np.float64) ** 2)) == pytest.approx(expected_power, rel=1e-4)
    # Without the floor the very same draw is far quieter.
    unfloored = scale_source_to_snr(source, quiet, snr_db)
    assert float(np.mean(unfloored.astype(np.float64) ** 2)) < 0.01 * expected_power

    loud = _noise(0.08, seed=3)
    assert np.array_equal(
        scale_source_to_snr(source, loud, snr_db, ref_power_floor=floor),
        scale_source_to_snr(source, loud, snr_db),
    )


def test_floor_applies_per_channel():
    source = _noise(0.5, channels=2, seed=1)
    # Channel 0 quiet (floored), channel 1 loud (untouched).
    noise = np.stack([_noise(0.001, seed=2)[0], _noise(0.08, seed=3)[0]]).astype(np.float32)
    floor = 0.02**2
    snr_db = -10.0
    scaled = scale_source_to_snr(source, noise, snr_db, per_channel=True, ref_power_floor=floor)
    powers = np.mean(scaled.astype(np.float64) ** 2, axis=1)
    assert powers[0] == pytest.approx(floor * 10.0 ** (snr_db / 10.0), rel=1e-4)
    plain = scale_source_to_snr(source, noise, snr_db, per_channel=True)
    assert powers[1] == pytest.approx(float(np.mean(plain[1].astype(np.float64) ** 2)), rel=1e-6)
