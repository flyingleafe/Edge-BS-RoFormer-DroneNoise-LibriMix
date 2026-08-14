"""Tests for the front door (``tracking.top``): plumbing, stages, recipes.

Covers: (1) the frame construction / accessor round-trip and the append-only
``meta["tracking"]`` log; (2) a blind-seed -> VK pipeline on a synthetic
two-rotor signal; (3) the stage guard leaving a good trajectory unvetoed;
(4) every adapter appending its diagnostics entry; (5) the peel/pi_kalman
SPLIT — the composed recipe must reproduce the fused application exactly, and
the candidate stages must reproduce their array-level formulas. Synthetic-signal helpers
mirror ``tests/tracking/test_vk_tracking.py`` (short signals, small ``k_max``
— the whole module stays CPU-fast).

Note: base speeds are 70 / 82 rev/s so their doubles fall OUTSIDE the blind
scan band (30-120 rev/s) — with the test comb's flat ``1/sqrt(k)`` amplitude
profile, a base at 45 gets out-promoted by its 2x subharmonic-up alias.
"""

import numpy as np
import pytest

from tracking.phase_increment_tracker import pi_kalman_refine
from tracking.pipelines import make_peels
from tracking.rps_refinement import RefineConfig
from tracking.telemetry_refit import presmooth
from tracking.top import (
    PeelConfig,
    PiConfig,
    blind_seed_stage,
    flagship,
    get_audio,
    get_rps,
    guarded,
    pi_kalman_arm_stage,
    pi_kalman_stage,
    pipeline,
    presmooth_stage,
    refine_coherent_stage,
    scale_stage,
    shift_stage,
    tracking_frame,
    vk_stage,
    warp_stage,
    with_rps,
)
from tracking.vk_tracking import VKConfig

FS = 16000.0
K_MAX = 20


def synth_comb(
    t: np.ndarray, r_true_list: list[np.ndarray], snr_db: float, seed: int
) -> np.ndarray:
    """Sum of harmonic combs (k = 1..K_MAX, amps 1/sqrt(k), random phases) + noise."""
    rng = np.random.default_rng(seed)
    sig = np.zeros_like(t)
    for r_true in r_true_list:
        phase = 2 * np.pi * np.cumsum(r_true) / FS
        for k in range(1, K_MAX + 1):
            sig += (1.0 / np.sqrt(k)) * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
    noise = rng.standard_normal(len(t))
    noise *= np.sqrt(np.mean(sig**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return sig + noise


DUR = 8.0
T_AUD = np.arange(int(DUR * FS)) / FS
R1_TRUE = 70.0 + 0.8 * np.sin(2 * np.pi * 0.2 * T_AUD)
R2_TRUE = 82.0 + 0.6 * np.sin(2 * np.pi * 0.3 * T_AUD + 1.0)


def make_cfg(**overrides) -> VKConfig:
    defaults: dict = dict(fs=FS, k_max=K_MAX, n_outer=8, couple_hz=20.0)
    defaults.update(overrides)
    return VKConfig(**defaults)


@pytest.fixture(scope="module")
def two_rotor_frame():
    """Two wobbling rotors at 70 / 82 rev/s, 10 dB SNR, no rps entry yet."""
    y = synth_comb(T_AUD, [R1_TRUE, R2_TRUE], snr_db=10.0, seed=1)
    return tracking_frame(y, 16000, meta={"recording_id": "synth2"})


@pytest.fixture(scope="module")
def seeded_frame(two_rotor_frame):
    return blind_seed_stage(2)(two_rotor_frame)


def _truth_on(ft: np.ndarray) -> np.ndarray:
    """(2, N) truth, rows sorted by base speed (the seed-bases convention)."""
    return np.stack([np.interp(ft, T_AUD, R1_TRUE), np.interp(ft, T_AUD, R2_TRUE)])


# ---------------------------------------------------------------------------
# 1. frame construction / accessors / meta log


def test_tracking_frame_roundtrip():
    sr = 16000
    audio = np.sin(2 * np.pi * 45.0 * np.arange(sr) / sr).astype(np.float32)
    ft = np.arange(0.0, 1.0 - 0.016, 0.032)
    r = np.stack([np.full(len(ft), 45.0), np.full(len(ft), 52.0)])
    frame = tracking_frame(
        audio, sr, rps=r, frame_times=ft, rps_meas=r + 1.0, meta={"recording_id": "x"}
    )

    a, sr_out = get_audio(frame)
    assert a.shape == (1, sr) and a.dtype == np.float32  # (T,) -> (1, T)
    assert sr_out == float(sr)

    r_out, t_out = get_rps(frame)
    assert r_out.shape == r.shape
    np.testing.assert_allclose(r_out, r)
    np.testing.assert_allclose(t_out, ft, atol=2e-9)  # nanosecond-tick rounding

    r_meas, _ = get_rps(frame, "rps_meas")
    np.testing.assert_allclose(r_meas, r + 1.0)
    assert frame["meta"]["recording_id"] == "x"


def test_tracking_frame_requires_frame_times():
    audio = np.zeros(1600, dtype=np.float32)
    with pytest.raises(ValueError, match="frame_times"):
        tracking_frame(audio, 16000, rps=np.zeros((1, 10)))


def test_with_rps_appends_meta_without_mutation():
    audio = np.zeros(3200, dtype=np.float32)
    ft = np.arange(0.0, 0.2 - 0.016, 0.032)
    r = np.full((1, len(ft)), 45.0)
    f0 = tracking_frame(audio, 16000, rps=r, frame_times=ft, meta={"recording_id": "x"})

    f1 = with_rps(f0, r + 0.5, ft, stage="a", info={"foo": 1})
    f2 = with_rps(f1, r + 1.0, ft, stage="b", info={"bar": 2})

    assert [e["stage"] for e in f2["meta"]["tracking"]] == ["a", "b"]
    assert f2["meta"]["tracking"][0]["foo"] == 1
    assert f2["meta"]["recording_id"] == "x"  # existing meta preserved
    r2, _ = get_rps(f2)
    np.testing.assert_allclose(r2, r + 1.0)
    # append-only: earlier frames keep their own (shorter) logs
    assert [e["stage"] for e in f1["meta"]["tracking"]] == ["a"]
    assert "tracking" not in set(f0["meta"])
    r0, _ = get_rps(f0)
    np.testing.assert_allclose(r0, r)


# ---------------------------------------------------------------------------
# 2. blind seed -> VK pipeline


def test_blind_seed_vk_pipeline_improves(two_rotor_frame, seeded_frame):
    out = vk_stage(make_cfg())(seeded_frame)

    r_seed, ft = get_rps(seeded_frame)
    r_vk, ft_vk = get_rps(out)
    np.testing.assert_allclose(ft_vk, ft, atol=2e-9)
    edge = (ft > 0.5) & (ft < DUR - 0.5)
    truth = _truth_on(ft)

    err_seed = float(np.mean(np.abs(r_seed[:, edge] - truth[:, edge])))
    err_vk = float(np.mean(np.abs(r_vk[:, edge] - truth[:, edge])))
    assert err_seed < 1.0, f"blind seed missed the bases (mean err {err_seed:.2f})"
    assert err_vk < 0.1, f"vk mean err {err_vk:.3f} exceeds 0.1"
    assert err_vk < err_seed / 3.0, (
        f"vk ({err_vk:.3f}) did not improve on the seed ({err_seed:.3f})"
    )

    log = out["meta"]["tracking"]
    assert [e["stage"] for e in log] == ["blind_seed", "vk"]
    assert len(log[0]["bases"]) == 2
    assert np.isfinite(log[1]["confidence_mean"])
    assert len(log[1]["residual_ratios"]) == make_cfg().n_outer
    # the input frames are untouched
    assert "rps" not in two_rotor_frame
    assert [e["stage"] for e in seeded_frame["meta"]["tracking"]] == ["blind_seed"]


def test_pipeline_composes_left_to_right(two_rotor_frame):
    cfg = make_cfg()
    a = pipeline(blind_seed_stage(2), vk_stage(cfg))
    # composition == manual chaining, including the meta log order
    out = a(two_rotor_frame)
    assert [e["stage"] for e in out["meta"]["tracking"]] == ["blind_seed", "vk"]


def test_vk_stage_rejects_rate_mismatch(seeded_frame):
    with pytest.raises(ValueError, match="does not match"):
        vk_stage(make_cfg(fs=8000.0))(seeded_frame)


# ---------------------------------------------------------------------------
# 3. guard


def test_guarded_leaves_good_trajectory_unvetoed(seeded_frame):
    out = guarded(vk_stage(make_cfg()))(seeded_frame)

    log = out["meta"]["tracking"]
    assert [e["stage"] for e in log] == ["blind_seed", "vk", "guard"]
    assert log[-1]["reverted"] == []
    assert log[-1]["reasons"] == []
    assert len(log[-1]["conf_before"]) == 2

    # unvetoed -> the guard keeps the vk trajectories (still near truth)
    r_g, ft = get_rps(out)
    edge = (ft > 0.5) & (ft < DUR - 0.5)
    err = float(np.mean(np.abs(r_g[:, edge] - _truth_on(ft)[:, edge])))
    assert err < 0.1, f"guarded output drifted from truth (mean err {err:.3f})"


# ---------------------------------------------------------------------------
# 4. refiner adapters append diagnostics


@pytest.fixture(scope="module")
def one_rotor_frame():
    """4 s single rotor at a constant 45 rev/s, init biased +0.3 rev/s."""
    dur = 4.0
    t = np.arange(int(dur * FS)) / FS
    y = synth_comb(t, [np.full_like(t, 45.0)], snr_db=10.0, seed=0)
    ft = np.arange(0.0, dur - 0.016, 0.032)
    r0 = np.full((1, len(ft)), 45.3)
    return tracking_frame(y, 16000, rps=r0, frame_times=ft)


@pytest.mark.parametrize(
    ("stage_factory", "expected_name", "info_key"),
    [
        (lambda: pi_kalman_stage(n_iter=1, k_max=8, k_caps=(8,)), "pi_kalman", "diagnostics"),
        (lambda: warp_stage(rounds=1), "warp", "diagnostics"),
        (lambda: refine_coherent_stage(n_iter=1, k_min=4, k_max=10), "stage_d", "params"),
    ],
)
def test_refiner_adapters_append_diagnostics(
    one_rotor_frame, stage_factory, expected_name, info_key
):
    out = stage_factory()(one_rotor_frame)

    entry = out["meta"]["tracking"][-1]
    assert entry["stage"] == expected_name
    assert info_key in entry

    r_in, ft_in = get_rps(one_rotor_frame)
    r_out, ft_out = get_rps(out)
    assert r_out.shape == r_in.shape
    np.testing.assert_allclose(ft_out, ft_in, atol=2e-9)
    err_in = float(np.mean(np.abs(r_in - 45.0)))
    err_out = float(np.mean(np.abs(r_out - 45.0)))
    assert err_out < err_in, f"{expected_name} made the trajectory worse ({err_in} -> {err_out})"


def test_refine_coherent_stage_rejects_rate_mismatch(one_rotor_frame):
    with pytest.raises(ValueError, match="does not match"):
        refine_coherent_stage(RefineConfig(sample_rate=8000))(one_rotor_frame)


# ---------------------------------------------------------------------------
# 5. the recipes: the peel/pi_kalman split reproduces the fused application


PEEL_KW = {"n_rotors": 2, "peel_k_max": 8}
PI_KW = {"n_iter": 1, "k_max": 8, "band_hz": 6.0}


@pytest.fixture(scope="module")
def alt_frame(two_rotor_frame):
    """The two-rotor frame with a slightly detuned constant init."""
    ft = np.arange(0.0, 2.0 - 0.016, 0.032)
    r0 = np.stack([np.full(len(ft), 69.6), np.full(len(ft), 82.4)])
    return with_rps(two_rotor_frame, r0, ft, stage="init", info={})


def test_flagship_recipe_equals_the_fused_application(alt_frame):
    """``peel_stage -> pi_kalman_stage`` IS one ``pi_kalman_arm_stage``.

    The split must be a re-composition and not a re-implementation, so the
    trajectory it produces is compared against the two array cores called by
    hand, bit for bit.
    """
    audio, sr = get_audio(alt_frame)
    r0, ft = get_rps(alt_frame)
    clip = np.asarray(audio, dtype=np.float64)
    peel_audio, pair_audio, _ = make_peels(clip, r0, ft, sr, "ls", n_rotors=2, k_max=8)
    r_ref, _ = pi_kalman_refine(
        clip, r0, ft, sr=int(sr), peel_audio=peel_audio, pair_audio=pair_audio, **PI_KW
    )

    composed = flagship(1, peel=PeelConfig(n_rotors=2, k_max=8), pi=PiConfig(extra=PI_KW))
    fused = pi_kalman_arm_stage(**PEEL_KW, **PI_KW)

    r_composed, _ = get_rps(composed(alt_frame))
    r_fused, _ = get_rps(fused(alt_frame))
    assert np.array_equal(r_composed, r_ref)
    assert np.array_equal(r_fused, r_ref)


def test_one_application_is_one_log_entry(alt_frame):
    """The peel leaves a seam, not a log entry — so an application logs once."""
    out = flagship(2, peel=PeelConfig(n_rotors=2, k_max=8), pi=PiConfig(extra=PI_KW))(alt_frame)
    log = [e for e in out["meta"]["tracking"] if e["stage"] == "peeled"]
    assert len(log) == 2
    assert all(e["peel"]["mode"] == "ls" and e["wall_peel_s"] >= 0.0 for e in log)
    # the seam is consumed: it must not survive into the output frame
    assert "peel_seam" not in out["meta"]


def test_naive_arm_is_the_same_recipe_without_the_peel(alt_frame):
    out = flagship(1, peel=None, pi=PiConfig(extra=PI_KW))(alt_frame)
    entry = out["meta"]["tracking"][-1]
    assert entry["stage"] == "naive"
    assert "peel" not in entry and entry["wall_peel_s"] == 0.0


def test_candidate_stages_are_their_formulas(alt_frame):
    r0, ft = get_rps(alt_frame)
    assert np.array_equal(get_rps(scale_stage(0.99458)(alt_frame))[0], r0 * 0.99458)
    assert np.array_equal(get_rps(presmooth_stage(5.0)(alt_frame))[0], presmooth(r0, ft, 5.0))
    shifted = get_rps(shift_stage(0.02)(alt_frame))[0]
    assert np.array_equal(shifted, np.stack([np.interp(ft + 0.02, ft, row) for row in r0]))


# ---------------------------------------------------------------------------
# the joint blocks as stages, and the two combinators
#
# The v3b arithmetic is pinned by tests/tracking/test_joint_regression.py. What
# is pinned HERE is the composition: that a hand-built pipeline of the three
# blocks is the shipped recipe, that the repetition and the windowing
# combinators do what they say, and that a joint stage refuses to run without
# its seam.

JOINT_SR = 8000
JOINT_K = 6


@pytest.fixture(scope="module")
def joint_clip():
    """``(audio, rates)`` — a two-rotor comb with a slow shaft wander."""
    rng = np.random.default_rng(4)
    n_t = 3 * JOINT_SR
    rates = np.stack([np.full(n_t, 55.0), np.full(n_t, 67.0)])
    drift = np.cumsum(rng.normal(scale=1.0, size=n_t)) / np.sqrt(n_t)
    phase = 2 * np.pi * np.cumsum(rates, axis=-1) / JOINT_SR + 0.5 * drift[None, :]
    audio = rng.normal(scale=0.02, size=(2, n_t))
    for r in range(2):
        for k in range(1, JOINT_K + 1):
            audio += (0.4 / k) * np.cos(k * phase[r])[None, :] * np.array([[1.0], [0.7]])
    return audio, rates


def _joint_frame(audio, rates, hop: float = 0.032):
    """The frame a windowed run starts from: audio plus the trajectory grid."""
    step = int(hop * JOINT_SR)
    r = rates[:, ::step]
    return tracking_frame(
        np.asarray(audio, dtype=np.float64),
        JOINT_SR,
        rps=r,
        frame_times=np.arange(r.shape[-1]) * hop,
        dtype=np.float64,
    )


def _joint_cfg():
    from tracking.decompose import solve_config

    return solve_config(JOINT_K, sr=JOINT_SR, mics=2, f_max=3000.0)


def test_hand_composed_blocks_are_the_shipped_recipe(joint_clip):
    """The recipe is a composition, so composing it by hand must give it back."""
    from tracking.joint_decompose import JointConfig, joint_result, joint_state
    from tracking.top import (
        floor_stage,
        iterate,
        joint_iterations,
        joint_solve_window,
        joint_state_of,
        phase_split_stage,
        vk_solve_stage,
        with_meta,
    )

    audio, rates = joint_clip
    cfg, jc = (
        _joint_cfg(),
        JointConfig(iters=2, k_trust=(2, JOINT_K), psd_n_fft=2048, profile_n_fft=2048),
    )
    want = joint_solve_window(audio, rates, cfg, k_hi=JOINT_K, mics=2, jcfg=jc)

    state = joint_state(rates, cfg, k_hi=JOINT_K, n_t=audio.shape[-1], jcfg=jc)
    stride, _ = state.grid
    frame = tracking_frame(
        np.asarray(audio, dtype=np.float64),
        JOINT_SR,
        rps=state.carrier[:, ::stride][:, : state.n_env],
        frame_times=np.arange(state.n_env) * stride / JOINT_SR,
        dtype=np.float64,
    )
    run = pipeline(
        floor_stage(),
        iterate(pipeline(vk_solve_stage(), phase_split_stage(), floor_stage()), jc.iters - 1),
        vk_solve_stage(profile=True),
    )
    out = run(with_meta(frame, joint=state))
    got = joint_result(joint_state_of(out), joint_iterations(out))

    assert np.array_equal(got.env.x, want.env.x)
    assert np.array_equal(got.theta_env, want.theta_env)
    assert np.array_equal(got.residual, want.residual)
    assert got.iterations == want.iterations
    # And the log is the method, block by block, in order.
    assert [e["stage"] for e in out["meta"]["tracking"]] == [
        "joint_floor",
        "joint_solve",
        "joint_phase_split",
        "joint_floor",
        "joint_solve",
    ]


def test_joint_init_stage_derives_the_carrier_from_the_frame(joint_clip):
    from tracking.joint_decompose import JointConfig
    from tracking.top import floor_stage, joint_init_stage, joint_state_of, vk_solve_stage

    audio, rates = joint_clip
    n_t = audio.shape[-1]
    frame = tracking_frame(
        np.asarray(audio, dtype=np.float64),
        JOINT_SR,
        rps=rates[:, ::80],
        frame_times=np.arange(n_t // 80) * 80 / JOINT_SR,
        dtype=np.float64,
    )
    run = pipeline(
        joint_init_stage(
            _joint_cfg(),
            k_hi=JOINT_K,
            jcfg=JointConfig(iters=1, psd_n_fft=2048, profile_n_fft=2048),
        ),
        floor_stage(),
        vk_solve_stage(),
    )
    state = joint_state_of(run(frame))
    assert state.carrier.shape == (2, n_t)
    assert state.n_solves == 1
    assert state.residual is not None
    # The tracks explain most of a clip that IS a comb.
    left = float((state.residual**2).sum() / (np.asarray(audio, dtype=np.float64) ** 2).sum())
    assert left < 0.2, left


def test_a_joint_stage_refuses_to_run_without_its_seam(joint_clip):
    from tracking.top import vk_solve_stage

    audio, rates = joint_clip
    frame = tracking_frame(
        audio,
        JOINT_SR,
        rps=rates[:, ::80],
        frame_times=np.arange(audio.shape[-1] // 80) * 80 / JOINT_SR,
    )
    with pytest.raises(ValueError, match="joint_init_stage"):
        vk_solve_stage()(frame)


def test_iterate_repeats_and_zero_is_the_identity():
    from tracking.top import iterate

    def bump(frame):
        r, ft = get_rps(frame)
        return with_rps(frame, r + 1.0, ft, stage="bump", info={})

    frame = tracking_frame(
        np.zeros(800), 8000, rps=np.full((1, 8), 10.0), frame_times=np.arange(8) * 0.0125
    )
    assert float(get_rps(iterate(bump, 3)(frame))[0].mean()) == pytest.approx(13.0)
    assert float(get_rps(iterate(bump, 0)(frame))[0].mean()) == pytest.approx(10.0)
    assert len(iterate(bump, 0)(frame)["meta"].keys() & {"tracking"}) == 0


def test_windowed_stitches_a_two_window_recording(joint_clip):
    """Two overlapping windows, stitched, still explain the clip."""
    from tracking.decompose import reconstruct
    from tracking.top import decompose_stage, windowed

    audio, rates = joint_clip
    frame = _joint_frame(audio, rates)
    run = windowed(decompose_stage(_joint_cfg(), k_hi=JOINT_K), window_s=2.0, hop_s=1.0)
    out = run(frame)
    entry = out["meta"]["tracking"][-1]
    assert entry["stage"] == "windowed"
    assert entry["n_windows"] >= 2
    assert entry["joint"] is False
    st = out["meta"]["decompose"]
    a_min, a_max = int(st["a_min"]), int(st["a_max"])
    recon, _ = reconstruct(
        st["x"], st["k"], st["rotor"], np.asarray(st["phi"])[:, a_min:a_max], entry["stride"]
    )
    clip = np.asarray(audio, dtype=np.float64)[:, a_min:a_max]
    assert float(((clip - recon) ** 2).sum() / (clip**2).sum()) < 0.2


def test_windowed_carries_a_joint_windows_shaft_correction(joint_clip):
    """A joint inner stage makes the stitch the RATE stitch, and it reports it."""
    from tracking.joint_decompose import JointConfig
    from tracking.top import floor_stage, joint_init_stage, vk_solve_stage, windowed

    audio, rates = joint_clip
    frame = _joint_frame(audio, rates)
    jc = JointConfig(iters=1, psd_n_fft=1024, profile_n_fft=1024, profile_every_iter=False)
    inner = pipeline(
        joint_init_stage(_joint_cfg(), k_hi=JOINT_K, jcfg=jc), floor_stage(), vk_solve_stage()
    )
    out = windowed(inner, window_s=2.0, hop_s=1.0)(frame)
    entry = out["meta"]["tracking"][-1]
    assert entry["joint"] is True
    assert entry["theta_stitch_max_rate_hz"] < 50.0
    assert out["meta"]["decompose"]["r_corrected"].shape == rates.shape
