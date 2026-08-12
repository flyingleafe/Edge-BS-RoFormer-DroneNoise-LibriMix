"""Unit tests for the amplitude-target path (model -> codec -> loss).

Four properties, one per stage of the path:

- the amplitude gain map is exactly ``propagate``'s ``1/r`` weight — measured
  against the distances it drops the delays from;
- the codec round-trips a real batch into the entries the loss declares;
- the loss is zero on an exact target, ignores masked cells, and is blind
  below its floor;
- the calibration gains recover a KNOWN per-microphone level pattern by
  gradient descent — the toy version of what the objective must do on the real
  arrays, where the per-mic residual spread is far wider than ``1/r`` predicts.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import tdseries as td
import torch

from losses.amplitude_target import AmplitudeTarget, band_powers, resample_time
from models.generative.positional_harmonic_gen import amplitude_gains, propagate
from models.registry import build_noise_gen_model
from tasks.codecs import NoiseGenerationCodec
from tasks.task import TASK_FACTORIES

SR = 16000
DRONES = ["dregon", "michaels"]


def _model(n_harmonics: int = 32, n_mics: int = 4) -> Any:
    """The conditioned generator with the amplitude-path extras switched on.

    Returned as ``Any``: the builder is typed ``nn.Module`` (it returns either
    the bare generator or the codebook wrapper), and these tests deliberately
    reach for the wrapper's own surface (``amp_stats``, the gain tables).
    """
    m = build_noise_gen_model(
        "positional_harmonic_gen",
        n_harmonics=n_harmonics,
        cond_dim=8,
        drone_names=DRONES,
        amp_calibration=True,
        n_mics=n_mics,
        noise_floor_bands=60,
    )
    m.eval()
    return cast(Any, m)


def _geometry(n_mics: int = 4, n_rotors: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    mic = rng.normal(scale=0.15, size=(n_mics, 3)).astype(np.float32)
    rotor = rng.normal(scale=0.25, size=(n_rotors, 3)).astype(np.float32)
    return mic, rotor


# ---------------------------------------------------------------------------
# 1. the gain map


def test_amplitude_gains_match_propagate_distances():
    """``amplitude_gains`` is ``propagate``'s attenuation with nothing else in it."""
    mic, rotor = _geometry()
    rel = torch.as_tensor(mic[None, :, None, :] - rotor[None, None, :, :])  # [1, M, R, 3]
    gains = amplitude_gains(rel, ref_distance=1.0, eps=1e-6)
    dist = np.linalg.norm(mic[:, None, :] - rotor[None, :, :], axis=-1)
    assert gains.shape == (1, 4, 4)
    np.testing.assert_allclose(gains[0].numpy(), 1.0 / dist, rtol=1e-6)

    # And it IS the weight propagate applies: a single rotor of DC (no delay
    # effect on a constant) is scaled by exactly that factor at each mic.
    src = torch.zeros(1, 4, 512)
    src[:, 0] = 1.0
    obs = propagate(src, rel, sample_rate=SR, ref_distance=1.0)
    np.testing.assert_allclose(obs[0, :, 256].numpy(), gains[0, :, 0].numpy(), rtol=1e-4, atol=1e-6)


def test_amp_stats_shapes_and_rotor_separation():
    """One rotor at a time: the envelope bank keeps the rotors apart."""
    model = _model()
    mic, rotor = _geometry()
    rel = torch.as_tensor(mic[None, :, None, :] - rotor[None, None, :, :])
    rps = torch.full((1, 4, SR), 60.0)
    rps[:, 2:] = 0.0  # two rotors stopped
    out = model.amp_stats(rps, rel, ["dregon"])
    assert out["amp"].shape[:3] == (1, 4, 4)
    assert out["amp"].shape[-2] == 32  # n_harmonics
    assert out["noise_psd"].shape[0] == 1 and out["noise_psd"].shape[-1] == 60
    # The stopped rotors' envelopes are silenced by the fade, the running ones
    # are not — no rotor sum happened anywhere.
    assert float(out["amp"][:, :, 2:].abs().max()) == 0.0
    assert float(out["amp"][:, :, :2].abs().max()) > 0.0


# ---------------------------------------------------------------------------
# 2. the codec


def _batch(n: int = 2, n_mics: int = 4, chunk: int = 4096) -> td.Frame:
    """A batched Frame in ``DecompFrameDataset``'s layout (no dataset needed)."""
    from data_processing.collate import frame_collate

    mic, rotor = _geometry(n_mics)
    n_env = chunk // 160
    frames = []
    for i in range(n):
        frames.append(
            td.Frame(
                {
                    "rps": td.uniform(
                        np.full((4, chunk), 60.0, dtype=np.float32),
                        SR,
                        dims=("rotor", "time"),
                        t_start=0.0,
                    ),
                    "residual": td.uniform(
                        np.zeros((n_mics, chunk), dtype=np.float32),
                        SR,
                        dims=("mic", "time"),
                        t_start=0.0,
                    ),
                    "amp": td.Series(
                        np.full((n_mics, 4, 80, n_env), 1e-3, dtype=np.float32),
                        ("mic", "rotor", "k", "time"),
                        {"time": td.GridIndex.create((SR, 160), n_env, t_start=0.0)},
                    ),
                    "amp_valid": td.Series(
                        np.ones((4, 80, n_env), dtype=bool),
                        ("rotor", "k", "time"),
                        {"time": td.GridIndex.create((SR, 160), n_env, t_start=0.0)},
                    ),
                    "mic_pos": td.wrap(mic, dims=("mic", None)),
                    "rotor_pos": td.wrap(rotor, dims=("rotor", None)),
                    "meta": td.Frame({"drone": DRONES[i % len(DRONES)]}),
                }
            )
        )
    return frame_collate(frames)


def test_codec_round_trip_matches_task_spec():
    """to_inputs -> call_model -> to_frame produces exactly the task's outputs."""
    from framespec import check_subsumes, spec_of

    batch = _batch()
    codec = NoiseGenerationCodec(conditioned=True, amplitude=True)
    task = TASK_FACTORIES["noise_generation"](conditioned=True, amplitude=True)
    inputs = codec.to_inputs(batch)
    assert inputs["drone_names"] == DRONES
    assert inputs["rel_pos"].shape == (2, 4, 4, 3)
    with torch.no_grad():
        pred = codec.to_frame(codec.call_model(_model(), inputs), batch)
    assert set(pred) == {"amp_pred", "noise_psd"}
    assert check_subsumes(spec_of(pred), task.output_spec) == []


# ---------------------------------------------------------------------------
# 3. the loss


def test_loss_zero_on_exact_target_and_respects_the_mask():
    core = AmplitudeTarget(eps=1e-8, psd_weight=0.0)
    tgt = torch.rand(2, 3, 4, 20, 10) * 1e-2 + 1e-4
    valid = torch.ones(2, 4, 20, 10, dtype=torch.bool)
    assert float(core.amplitude_term(tgt, tgt, valid)) == pytest.approx(0.0, abs=1e-7)

    # A wrong prediction only counts where the mask says the track was solved.
    wrong = tgt.clone()
    wrong[:, :, :, 10:] *= 100.0
    valid[:, :, 10:] = False
    assert float(core.amplitude_term(wrong, tgt, valid)) == pytest.approx(0.0, abs=1e-6)
    valid[:, :, 10:] = True
    assert float(core.amplitude_term(wrong, tgt, valid)) > 1.0


def test_loss_floor_hides_sub_floor_differences():
    """Below ``eps`` the log is flat, so floor noise stops driving the fit."""
    core = AmplitudeTarget(eps=1e-4, psd_weight=0.0)
    valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    tgt = torch.full((1, 1, 1, 1, 4), 1e-9)
    pred = torch.full((1, 1, 1, 1, 4), 1e-7)  # 100x off, but both far below eps
    assert float(core.amplitude_term(pred, tgt, valid)) < 1e-3
    loud = torch.full((1, 1, 1, 1, 4), 1e-2)
    assert float(core.amplitude_term(loud, tgt, valid)) > 3.0


def test_target_is_resampled_onto_the_prediction_grid():
    """A 100 Hz target and a 31.25 Hz prediction are compared, not rejected."""
    core = AmplitudeTarget(eps=1e-8, psd_weight=0.0)
    ramp = torch.linspace(1e-3, 1e-2, 100).reshape(1, 1, 1, 1, 100)
    coarse = resample_time(ramp, 32)
    valid = torch.ones(1, 1, 1, 100, dtype=torch.bool)
    assert coarse.shape[-1] == 32
    assert float(core.amplitude_term(coarse, ramp, valid)) < 0.02


def test_band_powers_locate_a_tone_and_follow_the_power_law():
    """A tone lands in its own band, and band power follows amplitude squared.

    Only the SHAPE and the power law are pinned. The absolute constant relating
    ``|STFT|^2 / ||w||^2`` to the generator's magnitude response is a units
    convention, and it is exactly what the per-drone ``log_gain_noise`` absorbs —
    asserting a particular value here would be asserting the convention, not the
    measurement.
    """
    t = torch.arange(SR, dtype=torch.float32) / SR
    tone = (0.5 * torch.sin(2 * torch.pi * 2500.0 * t)).reshape(1, -1)
    per_band = band_powers(tone, 8).mean(dim=-2)[0]
    assert int(per_band.argmax()) == 2  # 2500 Hz in 0..8000 Hz over 8 bands

    louder = band_powers(2.0 * tone, 8).mean(dim=-2)[0]
    assert float(louder.sum() / per_band.sum()) == pytest.approx(4.0, rel=1e-3)

    torch.manual_seed(0)
    white = band_powers(torch.randn(1, SR), 8).mean(dim=-2)[0]
    assert float(white.max() / white.min()) < 1.5  # flat input -> flat bands


# ---------------------------------------------------------------------------
# 4. the calibration gains


def test_calibration_gains_learn_a_known_per_mic_level():
    """Fit a synthetic target that is the model's own output times known gains.

    This is the toy of the real problem: the decomposition's targets are in the
    recording's absolute units, and the per-microphone pattern is NOT the ``1/r``
    one (docs/experiments/residual-attribution.md). The gains must be able to
    absorb both.
    """
    torch.manual_seed(0)
    model = _model()
    mic, rotor = _geometry()
    rel = torch.as_tensor(mic[None, :, None, :] - rotor[None, None, :, :])
    rps = torch.full((1, 4, 4096), 60.0)
    with torch.no_grad():
        base = model.amp_stats(rps, rel, ["dregon"])["amp"]
    true_global, true_mic = 3.0, torch.tensor([0.5, -0.4, 1.2, -0.8])
    target = base * torch.exp(true_global + true_mic)[None, :, None, None, None]
    valid = torch.ones(1, 4, base.shape[-2], base.shape[-1], dtype=torch.bool)

    core = AmplitudeTarget(eps=1e-12, psd_weight=0.0)
    gains = [p for n, p in model.named_parameters() if "log_gain" in n or "log_mic_gain" in n]
    opt = torch.optim.Adam(gains, lr=0.1)
    for _ in range(400):
        opt.zero_grad()
        pred = model.amp_stats(rps, rel, ["dregon"])["amp"]
        loss = core.amplitude_term(pred, target, valid)
        loss.backward()
        opt.step()
    assert float(loss) < 0.02
    gain_tables = cast(Any, model)
    learned = (
        gain_tables.log_gain["dregon"] + gain_tables.log_mic_gain["dregon"][:4]
    ).detach()
    np.testing.assert_allclose(learned.numpy(), (true_global + true_mic).numpy(), atol=0.05)


def test_calibration_is_off_by_default_and_leaves_rendering_alone():
    """A model built without the flag has no gains and renders as before."""
    plain = cast(
        Any,
        build_noise_gen_model(
            "positional_harmonic_gen", n_harmonics=8, cond_dim=8, drone_names=DRONES
        ),
    )
    assert not any("log_gain" in n or "log_mic_gain" in n for n, _ in plain.named_parameters())
    mic, rotor = _geometry(2)
    rel = torch.as_tensor(mic[None, :, None, :] - rotor[None, None, :, :])
    plain.eval()
    with torch.no_grad():
        audio = plain(torch.full((1, 4, 2048), 60.0), rel, ["dregon"])
    assert audio.shape == (1, 2, 2048)
