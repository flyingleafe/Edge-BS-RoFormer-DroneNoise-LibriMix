"""Tests for the canonical blind-annotation ladder (``tracking.pipelines``).

Covers: (1) the frozen config registry — spot-checks of the calibrated values
so an accidental recalibration fails loudly (the published annotations depend
on them); (2) a ``vit2dsp_stage`` smoke run on a short synthetic two-rotor
signal (helpers mirror ``test_stages.py``), with reduced VK configs passed as
explicit overrides — the frozen values themselves stay untouched.
"""

from dataclasses import replace

import numpy as np
import pytest
import tdseries as td

from tracking.pipelines import (
    CAPTURE_CFG,
    DEFAULT_PEEL_MODE,
    LADDER_N_ROTORS,
    MIDBAND_CFG,
    MIDBAND_CFGS,
    PAIRSCAN_HOP_S,
    PAIRSCAN_WIN_S,
    PEEL_MODES,
    REFINE_CFG,
    SEED_CFG,
    TRACK_CFG,
    VIT2D_BEAM,
    VIT2D_DELTA,
    VIT2D_STEP,
    VIT_DELTA,
    VIT_DSTEP,
    VIT_GAMMA_MULT,
    make_peels,
)
from tracking.top import (
    get_audio,
    get_rps,
    peel_alternation,
    tracking_frame,
    vit2dsp_stage,
)
from tracking.vk_blind_seeding import SeedConfig

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


# ---------------------------------------------------------------------------
# 1. frozen config registry (guards against accidental recalibration)


def test_refine_cfg_frozen_values():
    assert REFINE_CFG.fs == 16000.0
    assert REFINE_CFG.k_min == 6
    assert REFINE_CFG.k_max == 30
    assert REFINE_CFG.bw_hz == 1.5
    assert REFINE_CFG.n_outer == 5
    assert REFINE_CFG.max_step == 0.3
    assert REFINE_CFG.k_schedule == "fixed"
    assert REFINE_CFG.couple_hz == 20.0


def test_capture_cfg_is_refine_with_grow_schedule():
    assert replace(REFINE_CFG, k_schedule="grow", n_outer=12) == CAPTURE_CFG


def test_track_cfg_frozen_values():
    assert (TRACK_CFG.k_min, TRACK_CFG.k_max) == (6, 12)
    assert TRACK_CFG.bw_hz == 7.0
    assert TRACK_CFG.n_outer == 8
    assert TRACK_CFG.max_step == 0.5
    assert TRACK_CFG.update_gate == 8.0


def test_midband_cfgs_frozen_values():
    assert (MIDBAND_CFG.k_min, MIDBAND_CFG.k_max) == (6, 10)
    assert MIDBAND_CFG.bw_hz == 4.0
    assert MIDBAND_CFG.n_outer == 6
    assert (
        replace(MIDBAND_CFG, bw_hz=6.0, n_outer=4),
        replace(MIDBAND_CFG, bw_hz=4.0, n_outer=4),
    ) == MIDBAND_CFGS


def test_seed_cfg_frozen_values():
    assert (SEED_CFG.scan_lo, SEED_CFG.scan_hi, SEED_CFG.scan_step) == (30.0, 120.0, 0.05)
    assert SEED_CFG.k_scan == 40
    assert SEED_CFG.whiten_hz == 150.0
    assert SEED_CFG.blind_offsets == (-1.5, -0.5, 0.5, 1.5)
    # The registry entry pins the SeedConfig defaults — a default drift in
    # vk_blind_seeding without a registry update must fail here.
    assert SeedConfig() == SEED_CFG


def test_ladder_constants_frozen_values():
    assert (PAIRSCAN_WIN_S, PAIRSCAN_HOP_S) == (1.0, 0.25)
    assert (VIT_DELTA, VIT_DSTEP, VIT_GAMMA_MULT) == (6.0, 0.05, 0.3)
    assert (VIT2D_DELTA, VIT2D_STEP, VIT2D_BEAM) == (6.0, 0.1, 3)


# ---------------------------------------------------------------------------
# 2. vit2dsp_stage smoke (short signal, reduced VK overrides)

# Reduced-cost VK overrides (explicit arguments — the frozen configs are the
# defaults and stay untouched): one outer round each is enough for a smoke.
FAST_MID = replace(MIDBAND_CFGS[0], n_outer=1)
FAST_REF = replace(REFINE_CFG, n_outer=1, k_max=12)


@pytest.fixture(scope="module")
def two_rotor_frame():
    """5 s, two wobbling rotors at 70 / 82 rev/s, 10 dB SNR, no rps entry."""
    dur = 5.0
    t = np.arange(int(dur * FS)) / FS
    r1 = 70.0 + 0.8 * np.sin(2 * np.pi * 0.2 * t)
    r2 = 82.0 + 0.6 * np.sin(2 * np.pi * 0.3 * t + 1.0)
    y = synth_comb(t, [r1, r2], snr_db=10.0, seed=1)
    return tracking_frame(y, 16000, meta={"recording_id": "synth2"})


def test_vit2dsp_stage_self_seeds_and_logs(two_rotor_frame):
    out = vit2dsp_stage(midband_cfg=FAST_MID, refine_cfg=FAST_REF)(two_rotor_frame)

    r, ft = get_rps(out)
    assert r.shape[0] == LADDER_N_ROTORS  # the ladder is a 4-track construction
    assert r.shape[1] == len(ft)
    assert np.all(np.isfinite(r))
    # every track stays inside the blind-scan band the seed came from
    assert np.all(r > 20.0) and np.all(r < 130.0)

    log = out["meta"]["tracking"]
    assert [e["stage"] for e in log] == ["vit2dsp_seed", "vit2dsp"]
    entry = log[-1]
    assert entry["stages"] == ["init", "viterbi_c", "vit2dsp", "midband_bw6", "refine"]
    assert entry["guard_reverted"] == {}  # stage_guard=False -> no guard entries
    assert np.isfinite(entry["wall_scan_s"]) and np.isfinite(entry["wall_vk_s"])
    # the input frame is untouched
    assert "rps" not in two_rotor_frame


def test_vit2dsp_stage_uses_existing_rps(two_rotor_frame):
    ft = np.arange(0.0, 5.0 - 0.016, 0.032)
    r0 = np.stack([np.full(len(ft), v) for v in (69.5, 70.5, 81.5, 82.5)])
    frame = two_rotor_frame.with_entry("rps", td.events(ft, r0, dims=("rotor", "time")))

    out = vit2dsp_stage(midband_cfg=FAST_MID, refine_cfg=FAST_REF, stage_guard=True)(frame)

    r, ft_out = get_rps(out)
    assert r.shape == r0.shape
    np.testing.assert_allclose(ft_out, ft, atol=2e-9)
    log = out["meta"]["tracking"]
    # no seed stage — the existing rps entry is the ladder init
    assert [e["stage"] for e in log] == ["vit2dsp"]
    assert set(log[-1]["guard_reverted"]) == {"viterbi_c", "vit2dsp", "midband_bw6", "refine"}


def test_vit2dsp_stage_rejects_wrong_track_count(two_rotor_frame):
    ft = np.arange(0.0, 5.0 - 0.016, 0.032)
    r0 = np.stack([np.full(len(ft), 70.0), np.full(len(ft), 82.0)])
    frame = two_rotor_frame.with_entry("rps", td.events(ft, r0, dims=("rotor", "time")))
    with pytest.raises(ValueError, match="4-track"):
        vit2dsp_stage(midband_cfg=FAST_MID, refine_cfg=FAST_REF)(frame)


# ---------------------------------------------------------------------------
# 3. the peeled alternation (make_peels + pi_kalman_arm_stage + the driver)


@pytest.fixture(scope="module")
def peel_frame(two_rotor_frame):
    """The two-rotor frame with a slightly detuned constant init on the 32 ms grid."""
    ft = np.arange(0.0, 5.0 - 0.016, 0.032)
    r0 = np.stack([np.full(len(ft), 69.6), np.full(len(ft), 82.4)])
    return two_rotor_frame.with_entry("rps", td.events(ft, r0, dims=("rotor", "time")))


@pytest.mark.parametrize("peel_mode", list(PEEL_MODES))
def test_make_peels_removes_energy(peel_frame, peel_mode):
    audio, sr = get_audio(peel_frame)
    r, ft = get_rps(peel_frame)
    peel_audio, pair_audio, diag = make_peels(
        np.asarray(audio, dtype=np.float64), r, ft, sr, peel_mode, n_rotors=2, k_max=K_MAX
    )
    assert set(peel_audio) == {0, 1}
    assert set(pair_audio) == {(0, 1), (1, 0)}
    assert diag["mode"] == peel_mode
    # the gate: a correctly-phased peel takes energy OUT of the clip
    assert diag["energy_ok"] and diag["e_resid_all_ratio"] < 1.0
    # rotor i is never peeled of its OWN comb, so its residual keeps that energy
    assert all(0.0 < d["e_removed_frac"] < 1.0 for d in diag["per_rotor"])


def test_peel_alternation_logs_one_entry_per_application(peel_frame):
    frames = peel_alternation(peel_frame, 2, arm="peeled", n_rotors=2, tag="test", verbose=False)

    assert len(frames) == 3
    assert frames[0] is peel_frame  # the init is returned untouched
    log = frames[-1]["meta"]["tracking"]
    assert [e["stage"] for e in log] == ["peeled", "peeled"]
    for entry in log:
        assert entry["peel"]["mode"] == DEFAULT_PEEL_MODE
        assert len(entry["step_rms"]) == 2
        assert np.isfinite(entry["wall_peel_s"]) and np.isfinite(entry["wall_pi_s"])
    r_init, _ = get_rps(frames[0])
    r_final, _ = get_rps(frames[-1])
    assert np.all(np.isfinite(r_final)) and not np.allclose(r_final, r_init)


def test_naive_arm_skips_the_peel(peel_frame):
    frames = peel_alternation(peel_frame, 1, arm="naive", n_rotors=2, verbose=False)
    entry = frames[-1]["meta"]["tracking"][-1]
    assert entry["stage"] == "naive"
    assert "peel" not in entry


def test_peel_alternation_rejects_unknown_arm(peel_frame):
    with pytest.raises(ValueError, match="unknown arm"):
        peel_alternation(peel_frame, 1, arm="peel", verbose=False)
