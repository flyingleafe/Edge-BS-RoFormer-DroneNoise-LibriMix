"""The amplitude-only propagation head: geometry gains x a learnable per-mic EQ.

Covers the three things the head must get right — the EQ curve itself (knot
interpolation, clamping, the rendered twin), the rig routing (a batch mixes
rigs, and each sample must take its own rig's curve), and the composition with
the geometry gains inside ``amp_stats``.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from losses import smoothness_penalty
from models.generative.propagation import MicEQ
from models.registry import build_noise_gen_model

RIGS = ["dregon", "michaels"]


def _eq(n_mics: int = 3, n_knots: int = 8) -> MicEQ:
    return MicEQ(RIGS, n_mics, n_knots=n_knots, f_min=20.0, f_max=8000.0)


# ---------------------------------------------------------------------------
# the curve


def test_zero_init_is_unity_gain():
    """An untrained EQ must be the plain 1/r law, so an arm starts where v2 did."""
    eq = _eq()
    freq = torch.rand(2, 4, 10, 5) * 4000.0
    torch.testing.assert_close(eq.gain(freq, ["dregon", "michaels"]), torch.ones(2, 3, 4, 10, 5))


def test_gain_is_exact_at_the_knots():
    eq = _eq(n_mics=2, n_knots=6)
    with torch.no_grad():
        eq.log_eq["dregon"].copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) * 0.1)
    knot_f = eq.knot_freqs().reshape(1, -1)  # [1, K]
    got = eq.log_gain(knot_f, ["dregon"])[0]  # [M, K]
    torch.testing.assert_close(got, eq.log_eq["dregon"].detach(), atol=1e-5, rtol=1e-5)


def test_between_knots_is_linear_in_log_frequency():
    eq = _eq(n_mics=1, n_knots=4)
    with torch.no_grad():
        eq.log_eq["dregon"].copy_(torch.tensor([[0.0, 1.0, 0.0, 2.0]]))
    knots = eq.knot_freqs()
    mid = float(math.sqrt(float(knots[0]) * float(knots[1])))  # geometric mean = midpoint in log-f
    got = eq.log_gain(torch.tensor([[mid]]), ["dregon"])
    torch.testing.assert_close(got.reshape(()), torch.tensor(0.5), atol=1e-5, rtol=1e-5)


def test_outside_the_span_the_response_is_held_not_extrapolated():
    """DC (a stopped rotor) and above-Nyquist must be finite and clamped."""
    eq = _eq(n_mics=1, n_knots=4)
    with torch.no_grad():
        eq.log_eq["dregon"].copy_(torch.tensor([[-1.0, 0.0, 0.0, 3.0]]))
    freq = torch.tensor([[0.0, 1.0, 1e6]])
    got = eq.log_gain(freq, ["dregon"]).reshape(-1)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got[0], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)  # f=0 -> f_min
    torch.testing.assert_close(got[1], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got[2], torch.tensor(3.0), atol=1e-5, rtol=1e-5)  # -> f_max


def test_gain_shape_follows_the_frequency_grid():
    eq = _eq(n_mics=8, n_knots=16)
    freq = torch.rand(4, 4, 100, 31) * 4000.0  # [B, R, H, t]
    assert eq.gain(freq, ["dregon"] * 4).shape == (4, 8, 4, 100, 31)
    # A batch with fewer observers takes the leading microphones.
    assert eq.gain(freq, ["dregon"] * 4, n_mics=2).shape == (4, 2, 4, 100, 31)


# ---------------------------------------------------------------------------
# rig routing


def test_each_sample_takes_its_own_rig_curve():
    eq = _eq(n_mics=2, n_knots=5)
    with torch.no_grad():
        eq.log_eq["dregon"].fill_(0.5)
        eq.log_eq["michaels"].fill_(-0.5)
    freq = torch.full((3, 1), 1000.0)
    got = eq.log_gain(freq, ["michaels", "dregon", "michaels"])[:, 0, 0]
    torch.testing.assert_close(got, torch.tensor([-0.5, 0.5, -0.5]))


def test_an_unknown_rig_is_an_error_not_a_silent_default():
    eq = _eq()
    with pytest.raises(KeyError, match="unknown rig"):
        eq.log_gain(torch.full((1, 1), 100.0), ["fly999"])


def test_more_microphones_than_the_head_was_built_for_is_an_error():
    eq = _eq(n_mics=2)
    with pytest.raises(ValueError, match="built for 2 microphones"):
        eq.gain(torch.full((1, 1), 100.0), ["dregon"], n_mics=4)


def test_batch_size_must_match_the_rig_list():
    eq = _eq()
    with pytest.raises(ValueError, match="disagrees"):
        eq.log_gain(torch.full((3, 1), 100.0), ["dregon"])


# ---------------------------------------------------------------------------
# the smoothness prior


def test_curvature_penalty_ignores_a_ramp_and_catches_alternation():
    """The penalty must price knot-to-knot ALTERNATION, not frequency structure."""
    ramp = torch.linspace(0.0, 1.0, 8).reshape(1, 1, 8)
    zigzag = torch.tensor([0.0, 0.35, 0.0, 0.35, 0.0, 0.35, 0.0, 0.35]).reshape(1, 1, 8)
    assert float(smoothness_penalty(ramp, dims=(-1,))) == pytest.approx(0.0, abs=1e-6)
    assert float(smoothness_penalty(zigzag, dims=(-1,))) > 0.4


# ---------------------------------------------------------------------------
# the rendered twin


def test_filter_audio_applies_the_same_response_as_the_amplitude_path():
    """A pure tone must come out scaled by exactly the EQ at its frequency."""
    sr, n = 16000, 16000
    eq = _eq(n_mics=2, n_knots=6)
    with torch.no_grad():
        eq.log_eq["dregon"].copy_(torch.tensor([[0.0, 0.0, 0.7, 0.7, 0.0, 0.0]] * 2))
    f = float(eq.knot_freqs()[2])
    t = torch.arange(n, dtype=torch.float32) / sr
    tone = torch.sin(2 * math.pi * f * t).reshape(1, 1, n).expand(1, 2, n).contiguous()
    out = eq.filter_audio(tone, ["dregon"], sr)
    ratio = out.abs().max() / tone.abs().max()
    expected = float(eq.gain(torch.tensor([[f]]), ["dregon"]).reshape(-1)[0])
    assert float(ratio) == pytest.approx(expected, rel=0.02)


def test_filter_audio_is_the_identity_at_zero_init():
    eq = _eq(n_mics=2)
    audio = torch.randn(2, 2, 4096)
    torch.testing.assert_close(
        eq.filter_audio(audio, ["dregon", "michaels"], 16000.0), audio, atol=1e-5, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# composition inside the model


def _model(*, knots: int = 16, per_rotor: bool = False, n_mics: int = 4):
    params = {
        "n_harmonics": 12,
        "cond_dim": 8,
        "drone_names": RIGS,
        "amp_calibration": True,
        "noise_floor_bands": 60,
        "n_mics": n_mics,
        "mic_eq_knots": knots,
        "per_rotor_deltas": per_rotor,
        "n_rotors": 4,
    }
    return build_noise_gen_model("positional_harmonic_gen", **params)


def _inputs(b: int = 2, r: int = 4, m: int = 4, t: int = 8000):
    rps = torch.full((b, r, t), 70.0)
    rel = torch.full((b, m, r, 3), 0.3)
    for mi in range(m):  # distinct distances, so the 1/r factor is not degenerate
        rel[:, mi] = 0.2 + 0.1 * mi
    return rps, rel


def test_the_eq_replaces_the_flat_per_mic_scalar():
    keys = dict(_model(knots=16).named_parameters()).keys()
    assert any(k.startswith("mic_eq.log_eq.") for k in keys)
    assert not any(k.startswith("log_mic_gain.") for k in keys)
    # ... and without it the v2 arm's scalar is still what is built.
    keys_v2 = dict(_model(knots=0).named_parameters()).keys()
    assert any(k.startswith("log_mic_gain.") for k in keys_v2)
    assert not any(k.startswith("mic_eq.") for k in keys_v2)
    # The broadband branch keeps its own per-mic gain and floor either way.
    for k in (keys, keys_v2):
        assert any(x.startswith("log_mic_gain_noise.") for x in k)
        assert any(x.startswith("log_floor_psd.") for x in k)


def test_the_eq_needs_the_absolute_level_calibration():
    params = {"cond_dim": 8, "drone_names": RIGS, "amp_calibration": False, "mic_eq_knots": 8}
    with pytest.raises(ValueError, match="requires amp_calibration"):
        build_noise_gen_model("positional_harmonic_gen", **params)


def test_amp_stats_exposes_the_frequency_of_every_cell():
    model = _model()
    model.eval()
    rps, rel = _inputs()
    out = model.amp_stats(rps, rel, ["dregon", "michaels"])
    freq = out["freq"]  # [B, R, H, t_a]
    assert freq.shape[:3] == (2, 4, 12)
    # f = (h+1) * rps, the same series the oscillator bank builds.
    torch.testing.assert_close(freq[0, 0, 0], torch.full_like(freq[0, 0, 0], 70.0))
    torch.testing.assert_close(freq[0, 0, 11], torch.full_like(freq[0, 0, 11], 70.0 * 12))


def test_amp_stats_multiplies_geometry_gain_by_the_eq_at_each_cell_frequency():
    model = _model()
    model.eval()
    eq = cast(MicEQ, model.mic_eq)
    with torch.no_grad():
        # A curve that is far from flat across the harmonics AND different per
        # microphone, so neither effect can hide the other.
        ramp = torch.linspace(-1.0, 1.0, eq.n_knots)
        eq.log_eq["dregon"].copy_(torch.stack([ramp * s for s in (1.0, -1.0, 0.4, -0.7)]))
    rps, rel = _inputs()
    names = ["dregon", "dregon"]
    out = model.amp_stats(rps, rel, names)
    expected_eq = eq.gain(out["freq"], names, n_mics=4)
    # Divide the EQ back out: what remains must be flat across microphones up to
    # the 1/r ratio, which is what the head promises.
    bare = out["amp"] / expected_eq
    ratio = bare[:, 1] / bare[:, 0]
    gain_ratio = (out["gain"][:, 1] / out["gain"][:, 0])[..., None, None]
    torch.testing.assert_close(ratio, gain_ratio.expand_as(ratio), atol=1e-4, rtol=1e-3)
    # The EQ really did something: the raw amplitudes are NOT in the 1/r ratio.
    raw = out["amp"][:, 1] / out["amp"][:, 0]
    assert not torch.allclose(raw, gain_ratio.expand_as(raw), atol=1e-3)


def test_amp_stats_emits_the_knot_curve_for_the_curvature_penalty():
    model = _model()
    model.eval()
    rps, rel = _inputs()
    out = model.amp_stats(rps, rel, ["dregon", "michaels"])
    assert out["mic_eq"].shape == (2, 4, 16)
    assert out["mic_eq"].requires_grad


def test_the_rendering_path_carries_the_same_eq():
    """A checkpoint must render with the response it was fit with."""
    model = _model(n_mics=2)
    model.eval()
    rps, rel = _inputs(b=1, m=2, t=8000)
    # The emitter's broadband branch draws noise on every call, so the two
    # renderings are compared on the SAME draw.
    torch.manual_seed(0)
    flat = model(rps, rel, ["dregon"])
    eq = cast(MicEQ, model.mic_eq)
    with torch.no_grad():  # boost microphone 0 by ~6 dB across the whole band
        eq.log_eq["dregon"][0].fill_(0.7)
    torch.manual_seed(0)
    boosted = model(rps, rel, ["dregon"])
    ratio = boosted[0, 0].abs().mean() / flat[0, 0].abs().mean()
    assert float(ratio) == pytest.approx(math.exp(0.7), rel=0.05)
    torch.testing.assert_close(boosted[0, 1], flat[0, 1], atol=1e-5, rtol=1e-4)


def test_both_embedding_arms_run_through_the_same_head():
    """Single embedding and per-rotor sub-embeddings must both reach the EQ."""
    rps, rel = _inputs()
    for per_rotor in (False, True):
        model = _model(per_rotor=per_rotor)
        model.eval()
        out = model.amp_stats(rps, rel, ["dregon", "michaels"])
        assert out["amp"].shape == (2, 4, 4, 12, out["amp"].shape[-1])
        assert "mic_eq" in out and "freq" in out
        out["amp"].sum().backward()
        assert cast(MicEQ, model.mic_eq).log_eq["dregon"].grad is not None
