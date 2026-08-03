"""Tests for the time-varying time-warp augmentation of the noise+RPS pair.

Covers the pure-warp maths (constant-alpha exactness, time-varying label
consistency, parameter bounds) plus the ``OnlineMixIterableDataset`` wiring
(no-op path is byte-identical to the un-warped stream; probability-1.0 path
yields the right shapes / scaled labels; multichannel coherence).
"""

from __future__ import annotations

import numpy as np
import tdseries as td

from data_processing.time_warp import (
    DEFAULT_DEV_CONST,
    DEFAULT_DEV_SINE,
    WarpParams,
    apply_time_warp,
    sample_warp_params,
    source_duration_s,
)

SR = 16000
LABEL_RATE = 100.0


def _make_noise_frame(
    *,
    f0: float = 400.0,
    rps_base: np.ndarray | None = None,
    duration_s: float = 2.0,
    channels: int = 1,
    sr: int = SR,
    n_motor: int = 400,
    rid: str = "synth",
) -> td.Frame:
    """A synthetic noise Frame: pure-sine audio + a (constant) rps track."""
    if rps_base is None:
        rps_base = np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tone = np.sin(2.0 * np.pi * f0 * t).astype(np.float32)
    if channels == 1:
        audio = td.uniform(tone, sr, dims=("time",), t_start=0.0)
    else:
        stacked = np.stack([tone * (1.0 + 0.1 * ch) for ch in range(channels)], axis=0)
        audio = td.uniform(stacked, sr, dims=("mic", "time"), t_start=0.0)

    motor_t = np.linspace(0.0, duration_s - 0.01, n_motor)
    values = np.tile(np.asarray(rps_base, dtype=np.float64)[:, None], (1, n_motor))
    rps = td.events(motor_t, values, dims=("rotor", "time"), t_start=0.0, t_end=duration_s)
    return td.Frame({"audio": audio, "rps": rps, "meta": td.Frame({"recording_id": rid})})


def _fft_peak_hz(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return float(freqs[int(np.argmax(spec))])


# ─── 1. Constant-alpha exactness ────────────────────────────────────────────────


def test_constant_alpha_scales_frequency_and_label_and_length():
    f0 = 400.0
    rps_base = np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    frame = _make_noise_frame(f0=f0, rps_base=rps_base, duration_s=2.0)
    c = 1.07
    params = WarpParams(c=c, a=0.0, f=0.5, phi=0.0, dev_const=0.08, dev_sine=0.04)

    target_len = SR  # 1 s
    warped = apply_time_warp(frame, params, target_len=target_len, sample_rate=SR)

    audio = np.asarray(warped["audio"].data, dtype=np.float32)
    assert audio.shape == (target_len,)
    # Frequency scaled by c: dominant peak ~ c * f0.
    peak = _fft_peak_hz(audio, SR)
    assert abs(peak - c * f0) < 2.0

    # Constant alpha => rps scaled by exactly c.
    rps = np.asarray(warped["rps"].data, dtype=np.float64)
    expected = np.tile((c * rps_base)[:, None], (1, rps.shape[1]))
    np.testing.assert_allclose(rps, expected, rtol=0, atol=1e-3)


# ─── 2. Time-varying label consistency + monotone in-bounds tau ─────────────────


def test_time_varying_label_matches_independent_computation():
    rps_base = np.array([45.0, 55.0, 65.0, 75.0], dtype=np.float64)
    frame = _make_noise_frame(rps_base=rps_base, duration_s=2.0, n_motor=400)
    params = WarpParams(c=1.05, a=0.03, f=0.7, phi=1.1, dev_const=0.08, dev_sine=0.04)
    target_len = SR

    warped = apply_time_warp(
        frame, params, target_len=target_len, sample_rate=SR, label_rate_hz=LABEL_RATE
    )
    rps = np.asarray(warped["rps"].data, dtype=np.float64)

    n_label = rps.shape[1]
    t_label = np.arange(n_label, dtype=np.float64) / LABEL_RATE
    alpha = params.alpha(t_label)
    # Constant source rps => r(tau) == rps_base regardless of tau.
    expected = alpha[None, :] * rps_base[:, None]
    np.testing.assert_allclose(rps, expected, rtol=0, atol=1e-3)

    # tau strictly increasing (monotone) and source indexing stays in-bounds.
    t_audio = np.arange(target_len, dtype=np.float64) / SR
    tau_audio = params.tau(t_audio)
    assert np.all(np.diff(tau_audio) > 0)
    src_pos = tau_audio * SR
    assert src_pos[0] >= 0.0
    assert src_pos[-1] <= frame["audio"].dim_size("time") - 1


def test_time_varying_label_tracks_source_trajectory():
    # Non-constant source trajectory: r(tau) genuinely depends on tau.
    duration_s = 2.0
    n_motor = 400
    motor_t = np.linspace(0.0, duration_s - 0.01, n_motor)
    # Linear ramp per rotor: r_i(s) = base_i + slope * s.
    slope = 5.0
    base = np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    values = base[:, None] + slope * motor_t[None, :]
    n = int(duration_s * SR)
    tone = np.sin(2.0 * np.pi * 300.0 * np.arange(n) / SR).astype(np.float32)
    frame = td.Frame(
        {
            "audio": td.uniform(tone, SR, dims=("time",), t_start=0.0),
            "rps": td.events(
                motor_t, values, dims=("rotor", "time"), t_start=0.0, t_end=duration_s
            ),
        }
    )
    params = WarpParams(c=1.04, a=0.03, f=0.6, phi=0.4, dev_const=0.08, dev_sine=0.04)
    target_len = SR
    warped = apply_time_warp(
        frame, params, target_len=target_len, sample_rate=SR, label_rate_hz=LABEL_RATE
    )
    rps = np.asarray(warped["rps"].data, dtype=np.float64)

    n_label = rps.shape[1]
    t_label = np.arange(n_label, dtype=np.float64) / LABEL_RATE
    tau = params.tau(t_label)
    alpha = params.alpha(t_label)
    r = base[:, None] + slope * tau[None, :]  # r(tau)
    expected = alpha[None, :] * r
    np.testing.assert_allclose(rps, expected, rtol=0, atol=1e-2)


# ─── 3. Parameter bounds ────────────────────────────────────────────────────────


def test_default_alpha_stays_within_bounds_over_many_draws():
    rng = np.random.default_rng(0)
    spec = {
        "dev_const": DEFAULT_DEV_CONST,
        "dev_sine": DEFAULT_DEV_SINE,
        "f_low": 0.1,
        "f_high": 1.0,
    }
    bound = DEFAULT_DEV_CONST + DEFAULT_DEV_SINE  # 0.12
    t = np.linspace(0.0, 1.0, 2000)
    for _ in range(500):
        params = sample_warp_params(spec, rng)
        alpha = params.alpha(t)
        assert np.all(alpha >= 1.0 - bound - 1e-9)
        assert np.all(alpha <= 1.0 + bound + 1e-9)
        assert params.a >= 0.0
        assert 0.1 <= params.f <= 1.0
        assert 0.0 <= params.phi < 2.0 * np.pi


# ─── 4. No-op path: byte-identical to the un-warped stream ──────────────────────


def _online_frames(policy: dict, base_seed: int = 4321, n: int = 6, repo=None, rid="pool"):
    """The first ``n`` sample Frames of the compiled online-mix pipeline over a
    one-recording synthetic frames dataset (local dload repo)."""
    import data_processing.streams as streams
    from data_processing.frame_datasets import OnlineMixFrameDataset

    if repo is None:
        raise ValueError("pass the patched_repo fixture")
    frame = _make_noise_frame(channels=2, duration_s=3.0, rid=rid)
    repo.commit(
        f"TW-{rid}",
        [(rid, streams.frame_to_sample(frame))],
        meta={"layout": streams.TDFRAME_LAYOUT},
    )
    cfg = {
        "sample_rate": SR,
        "duration_s": 1.0,
        "base_seed": base_seed,
        "sources": {"noise": [{"kind": "frames", "dataset": f"TW-{rid}", "min_motor_rps": 0.0}]},
        "policy": policy,
    }
    import itertools

    return list(itertools.islice(iter(OnlineMixFrameDataset.from_config(cfg)), n))


def test_absent_key_is_bit_identical_to_baseline(patched_repo):
    # A policy with an augmentation but NO noise_time_warp key must not consume
    # any warp RNG; two datasets built identically must produce identical output.
    policy = {
        "source_prob": 0.0,
        "augmentations": {
            "probability": 0.5,
            "choices": [{"random_gain": {"min_db": -6.0, "max_db": 6.0}}],
        },
    }
    run_a = _online_frames(policy, repo=patched_repo, rid="a")
    run_b = _online_frames(dict(policy), repo=patched_repo, rid="a")
    for fa, fb in zip(run_a, run_b):
        np.testing.assert_array_equal(
            np.asarray(fa["mixture"].data), np.asarray(fb["mixture"].data)
        )
        np.testing.assert_array_equal(np.asarray(fa["rps"].data), np.asarray(fb["rps"].data))


def test_probability_zero_warp_is_bit_identical_to_no_key(patched_repo):
    # noise_time_warp with probability 0 must draw no RNG => identical to no key.
    base_policy = {"source_prob": 0.0}
    warp_policy = {"source_prob": 0.0, "noise_time_warp": {"probability": 0.0}}
    run_base = _online_frames(base_policy, repo=patched_repo, rid="b")
    run_warp = _online_frames(warp_policy, repo=patched_repo, rid="b")
    for fa, fb in zip(run_base, run_warp):
        np.testing.assert_array_equal(
            np.asarray(fa["mixture"].data), np.asarray(fb["mixture"].data)
        )
        np.testing.assert_array_equal(np.asarray(fa["rps"].data), np.asarray(fb["rps"].data))


# ─── 5. Integration: probability 1.0 over a synthetic pool ──────────────────────


def test_integration_prob_one_shapes_and_scaled_labels(patched_repo):
    rps_base = np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    import data_processing.streams as streams

    frame = _make_noise_frame(channels=2, rps_base=rps_base, duration_s=3.0, rid="warp1")
    patched_repo.commit(
        "TW-warp1", [("warp1", streams.frame_to_sample(frame))], meta={"layout": "tdframe-v1"}
    )

    def _cfg(policy):
        return {
            "sample_rate": SR,
            "duration_s": 1.0,
            "base_seed": 99,
            "sources": {"noise": [{"kind": "frames", "dataset": "TW-warp1", "min_motor_rps": 0.0}]},
            "policy": policy,
        }

    from data_processing.frame_datasets import OnlineMixFrameDataset

    warp_frame = next(
        iter(
            OnlineMixFrameDataset.from_config(
                _cfg({"source_prob": 0.0, "noise_time_warp": {"probability": 1.0}})
            )
        )
    )
    plain_frame = next(iter(OnlineMixFrameDataset.from_config(_cfg({"source_prob": 0.0}))))

    audio = np.asarray(warp_frame["mixture"].data)
    rps_np = np.asarray(warp_frame["rps"].data)
    assert audio.shape == (2, SR)
    assert rps_np.shape == (4, SR // 512 + 1)
    bound = DEFAULT_DEV_CONST + DEFAULT_DEV_SINE
    # Constant source rps => warped label is alpha(t) * rps_base, so per-rotor
    # values stay within the +-12% band of the source constant.
    for i, base_val in enumerate(rps_base):
        assert np.all(rps_np[i] >= base_val * (1.0 - bound) - 1e-3)
        assert np.all(rps_np[i] <= base_val * (1.0 + bound) + 1e-3)

    # Warped audio differs from the un-warped audio.
    plain_audio = np.asarray(plain_frame["mixture"].data)
    assert not np.allclose(audio, plain_audio)


# ─── 6. Multichannel coherence ──────────────────────────────────────────────────


def test_multichannel_warp_is_channel_coherent():
    # Warping each channel independently must equal warping the whole array.
    rng = np.random.default_rng(7)
    n = int(2.0 * SR)
    base_tone = np.sin(2.0 * np.pi * 350.0 * np.arange(n) / SR)
    multi = np.stack([(base_tone * (1.0 + 0.2 * ch)).astype(np.float32) for ch in range(4)], axis=0)
    rps_base = np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    motor_t = np.linspace(0.0, 2.0 - 0.01, 400)
    values = np.tile(rps_base[:, None], (1, 400))
    frame_multi = td.Frame(
        {
            "audio": td.uniform(multi, SR, dims=("mic", "time"), t_start=0.0),
            "rps": td.events(motor_t, values, dims=("rotor", "time"), t_start=0.0, t_end=2.0),
        }
    )
    params = sample_warp_params({}, rng)
    target_len = SR
    warped_multi = apply_time_warp(frame_multi, params, target_len=target_len, sample_rate=SR)
    warped_audio = np.asarray(warped_multi["audio"].data, dtype=np.float32)

    for ch in range(4):
        frame_ch = td.Frame(
            {
                "audio": td.uniform(multi[ch], SR, dims=("time",), t_start=0.0),
                "rps": td.events(motor_t, values, dims=("rotor", "time"), t_start=0.0, t_end=2.0),
            }
        )
        warped_ch = apply_time_warp(frame_ch, params, target_len=target_len, sample_rate=SR)
        np.testing.assert_array_equal(
            warped_audio[ch], np.asarray(warped_ch["audio"].data, dtype=np.float32)
        )


def test_source_duration_request_covers_worst_case():
    params = WarpParams(
        c=1.0 + DEFAULT_DEV_CONST,
        a=DEFAULT_DEV_SINE,
        f=0.1,
        phi=0.0,
        dev_const=DEFAULT_DEV_CONST,
        dev_sine=DEFAULT_DEV_SINE,
    )
    base_duration = 1.0
    requested = source_duration_s(base_duration, params)
    # tau(T) must never exceed the requested source seconds.
    t = np.linspace(0.0, base_duration, 5000)
    assert float(params.tau(t)[-1]) <= requested
