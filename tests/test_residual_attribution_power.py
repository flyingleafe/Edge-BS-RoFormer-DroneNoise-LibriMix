"""The band-power attribution estimator, checked against data it CAN solve.

The in-flight verdict is negative, so the estimator has to be shown working
somewhere before the negative means anything: these tests give it four rotors
with genuinely independent speeds and a known basis, and require recovery. The
last test is the mirror image — collinear rotor speeds, where recovery must
fail — which is the flight condition.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.residual_attribution import power


def _synth(rng, *, n_t=600, independent=True, n_band=3, n_ch=8, n_rot=4):
    """Band powers from a known basis and known per-rotor speeds."""
    common = 60 + 20 * np.sin(np.linspace(0, 6, n_t))
    if independent:
        rps = common + 12 * rng.standard_normal((n_rot, n_t)).cumsum(-1) / np.sqrt(n_t) * 3
    else:
        rps = common + 0.4 * rng.standard_normal((n_rot, n_t))
    basis = rng.random((n_ch, n_rot, n_band)) + 0.1
    basis /= basis.sum(0, keepdims=True)
    s = power.modulation_regressors(rps, exponent=5.0)
    s /= s.mean(-1, keepdims=True)
    level = np.array([1.0, 0.6, 0.3, 0.1])[:, None]  # (R, 1) per-rotor strength
    clean = np.einsum("crb,rt->cbt", basis * level[None, :, :], s)
    floor = 0.05 * clean.mean(-1)[:, :, None]
    obs = (clean + floor) * (1 + 0.05 * rng.standard_normal(clean.shape))
    return obs, rps, s, basis, level


def test_band_power_tracks_a_known_level():
    sr = 16000
    t = np.arange(sr) / sr
    x = np.stack([np.sqrt(2.0) * np.sin(2 * np.pi * 700 * t), np.zeros(sr)])
    x[1] = 0.1 * x[0]
    bp = power.band_power(x, sr, [(500, 1000), (2000, 4000)], frame_s=0.1, n_fft=512)
    assert bp.power.shape[:2] == (2, 2)
    # the tone sits in band 0; mic 1 is 20 dB down on mic 0 in that band
    ratio = bp.power[0, 0].mean() / bp.power[1, 0].mean()
    assert 50 < ratio < 200


def test_free_modulation_recovers_shares_when_rotors_move_apart():
    rng = np.random.default_rng(0)
    obs, _, s, basis, level = _synth(rng, independent=True)
    fit = power.fit_free_modulation(obs, s)
    truth = basis * level[None, :, :]
    truth = np.transpose(truth, (1, 0, 2))
    truth = truth / truth.sum(0, keepdims=True)
    assert fit["r2"].mean() > 0.9
    assert np.abs(fit["share"] - truth).mean() < 0.1


def test_pattern_agreement_prefers_the_true_assignment():
    rng = np.random.default_rng(1)
    obs, _, s, basis, _ = _synth(rng, independent=True)
    fit = power.fit_free_modulation(obs, s)
    agree = power.pattern_agreement(fit["gain"], basis)
    assert agree["cos_mean"].mean() > agree["cos_perm"][1:].max()


def test_basis_fit_beats_its_own_permutation():
    rng = np.random.default_rng(2)
    obs, _, s, basis, _ = _synth(rng, independent=True)
    good = power.fit_basis_modulation(obs, s, basis)["r2"]
    bad = power.fit_basis_modulation(obs, s, np.roll(basis, 1, axis=1))["r2"]
    assert (good > bad + 0.01).all()


def test_mode_information_sees_differential_drive_and_not_noise():
    from tracking.rotors import MIXER, NUM_ROTORS

    rng = np.random.default_rng(3)
    obs, rps, _, _, _ = _synth(rng, independent=True)
    design, names = power.mode_design(rps, MIXER.T / NUM_ROTORS)
    assert list(names) == ["const", "common", "roll", "pitch", "yaw"]
    info = power.mode_information(obs, design, n_boot=16, block_frames=50)
    assert (info["delta_r2"] > info["delta_r2_null_q95"]).mean() > 0.8

    obs2, rps2, _, _, _ = _synth(rng, independent=False)
    design2, _ = power.mode_design(rps2, MIXER.T / NUM_ROTORS)
    info2 = power.mode_information(obs2, design2, n_boot=16, block_frames=50)
    assert (info2["delta_r2"] > info2["delta_r2_null_q95"]).mean() < 0.5


def test_identifiability_flags_a_duplicated_column():
    rng = np.random.default_rng(5)
    b = rng.random((8, 4)) + 0.1
    b[:, 2] = b[:, 1]  # two rotors the array cannot tell apart
    d = power.basis_identifiability(b, with_floor=False)
    assert float(d["max_cos"]) == pytest.approx(1.0, abs=1e-9)
    assert float(d["cond"]) > 1e7
    assert d["vif"][1] > 1e6


def test_additivity_reads_zero_on_incoherent_sums():
    rng = np.random.default_rng(4)
    sr = 16000
    single = {r: rng.standard_normal((4, sr)) * (0.5 + r) for r in range(4)}
    combined = sum(single.values())
    add = power.additivity(single, combined, sr, [(500, 1000), (2000, 4000)])
    assert np.abs(add["excess_db"]).max() < 1.0
