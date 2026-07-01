"""Smoke + wiring tests for the noise-generation task and training script.

Builds tiny synthetic DREGON-LM-style chunks (clean ``noise.wav`` + ``rps.npy``),
then exercises: the geometry->rel_pos helper, the TimeFrame loader's
``global_data`` positions, the dataset item contract, and one end-to-end
forward+backward training step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchaudio

from data_processing.michaels import ARRAY_OFFSET_FORWARD, MIC_ARRAY_RADIUS, WHEELBASE
from data_processing.michaels import NUM_ROTORS as MICHAELS_NUM_ROTORS
from data_processing.michaels import get_geometry as get_michaels_geometry
from tasks.noise_generation import DroneCodebook, geometry_to_rel_pos, load_input_set
from train_noise_generation import (
    DREGONNoiseGenDataset,
    MultiScaleSTFT,
    _smoothness_loss,
    _spectral_loss,
    get_model,
)

SR = 16000
N_MICS = 8
N_ROTORS = 4


def _fake_geometry():
    rng = np.random.default_rng(0)
    mic_pos = rng.normal(scale=0.1, size=(N_MICS, 3))
    rotor_pos = rng.normal(scale=0.1, size=(N_ROTORS, 3)) + np.array([0.0, 0.0, 0.2])
    return mic_pos, rotor_pos


def _make_dataset(root, n=3, n_samples=4096, n_motor=40):
    rng = np.random.default_rng(1)
    for i in range(n):
        d = root / f"sample_{i:05d}"
        d.mkdir()
        noise = torch.from_numpy(rng.normal(scale=0.1, size=(N_MICS, n_samples)).astype("float32"))
        torchaudio.save(str(d / "noise.wav"), noise, SR)
        rps = (rng.uniform(60, 90, size=(N_ROTORS, n_motor))).astype("float32")
        np.save(d / "rps.npy", rps)


# ── geometry helper ─────────────────────────────────────────────────────────


def test_geometry_to_rel_pos():
    mic_pos, rotor_pos = _fake_geometry()
    rel = geometry_to_rel_pos(mic_pos, rotor_pos)
    assert rel.shape == (N_MICS, N_ROTORS, 3)
    # rel[m, r] == mic[m] - rotor[r]
    for m in range(N_MICS):
        for r in range(N_ROTORS):
            assert np.allclose(rel[m, r], mic_pos[m] - rotor_pos[r])


# ── TimeFrame loader carries positions in global_data ───────────────────────


def test_load_input_set_global_data(tmp_path):
    _make_dataset(tmp_path)
    mic_pos, rotor_pos = _fake_geometry()
    frames = list(load_input_set(tmp_path, mic_pos, rotor_pos))
    assert len(frames) == 3
    tf = frames[0]
    assert "audio" in tf and "rps" in tf
    assert np.allclose(tf.global_data["mic_positions"], mic_pos)
    assert np.allclose(tf.global_data["rotor_positions"], rotor_pos)


# ── dataset item contract ────────────────────────────────────────────────────


def test_dataset_item_shapes(tmp_path):
    _make_dataset(tmp_path, n_samples=4096)
    mic_pos, rotor_pos = _fake_geometry()
    ds = DREGONNoiseGenDataset(str(tmp_path), mic_pos, rotor_pos, drone_name="dregon")
    rps, rel_pos, target, drone_name = ds[0]
    assert rps.shape == (N_ROTORS, 4096)  # upsampled to audio rate
    assert rel_pos.shape == (N_MICS, N_ROTORS, 3)
    assert target.shape == (N_MICS, 4096)
    assert drone_name == "dregon"


def test_dataset_missing_target_raises(tmp_path):
    # rps.npy present but no noise.wav -> clear error.
    d = tmp_path / "sample_00000"
    d.mkdir()
    np.save(d / "rps.npy", np.ones((N_ROTORS, 10), dtype="float32"))
    mic_pos, rotor_pos = _fake_geometry()
    try:
        DREGONNoiseGenDataset(str(tmp_path), mic_pos, rotor_pos)
    except FileNotFoundError as e:
        assert "noise.wav" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError for missing target")


# ── end-to-end training step ─────────────────────────────────────────────────


def test_one_training_step(tmp_path):
    _make_dataset(tmp_path, n=4, n_samples=4096)
    mic_pos, rotor_pos = _fake_geometry()
    ds = DREGONNoiseGenDataset(str(tmp_path), mic_pos, rotor_pos)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    model = get_model("positional_harmonic_gen", sample_rate=SR, n_harmonics=8)
    loss_fn = MultiScaleSTFT(n_ffts=[512, 256, 128], log_weight=1.0, loss_type="L1")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rps, rel_pos, target, _drone_name = next(iter(loader))
    pred = model(rps, rel_pos)
    assert pred.shape == target.shape == (2, N_MICS, 4096)

    loss = _spectral_loss(loss_fn, pred, target)
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    # gradients reach the emitter's learned parameters
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()


def _smooth_args(harm=0.0, noise=0.0, no_diff_noise=False):
    import argparse

    return argparse.Namespace(
        harm_smooth_weight=harm, noise_smooth_weight=noise, no_diff_noise=no_diff_noise
    )


def test_smoothness_loss_zero_when_disabled():
    model = get_model("positional_harmonic_gen", sample_rate=SR, n_harmonics=8)
    rps = torch.full((1, N_ROTORS, 4096), 80.0)
    out = model(rps, _rel(1), return_dict=True)
    assert _smoothness_loss(out, _smooth_args(0.0, 0.0)).item() == 0.0


def test_smoothness_loss_positive_and_backprops():
    # Enabling either weight adds a positive penalty whose gradient reaches the
    # emitter (it is computed from the emitter's own control curves).
    model = get_model("positional_harmonic_gen", sample_rate=SR, n_harmonics=8)
    rps = torch.full((2, N_ROTORS, 4096), 80.0)
    out = model(rps, _rel(2), return_dict=True)
    pen = _smoothness_loss(out, _smooth_args(harm=1e-2, noise=1e-2))
    assert pen.item() > 0
    pen.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_noise_smoothness_ignored_without_diff_noise():
    # --no_diff_noise removes the noise branch, so its smoothness term is skipped
    # even if the weight is set (only the harmonic term remains).
    model = get_model(
        "positional_harmonic_gen", sample_rate=SR, n_harmonics=8, use_diff_noise=False
    )
    rps = torch.full((1, N_ROTORS, 4096), 80.0)
    out = model(rps, _rel(1), return_dict=True)
    only_noise = _smoothness_loss(out, _smooth_args(harm=0.0, noise=1e-2, no_diff_noise=True))
    assert only_noise.item() == 0.0
    with_harm = _smoothness_loss(out, _smooth_args(harm=1e-2, noise=1e-2, no_diff_noise=True))
    assert with_harm.item() > 0


def test_diff_noise_toggle(tmp_path):
    _make_dataset(tmp_path, n=2)
    mic_pos, rotor_pos = _fake_geometry()
    ds = DREGONNoiseGenDataset(str(tmp_path), mic_pos, rotor_pos)
    model = get_model(
        "positional_harmonic_gen", sample_rate=SR, n_harmonics=8, use_diff_noise=False
    )
    rps, rel_pos, target, _drone_name = ds[0]
    out = model(rps.unsqueeze(0), rel_pos.unsqueeze(0))
    assert out.shape == (1, N_MICS, target.shape[-1])


# ── Per-drone conditioning (external DroneCodebook + FiLM) ────────────────────


def _rel(n=1):
    rng = np.random.default_rng(2)
    return torch.from_numpy(
        (rng.normal(scale=0.1, size=(n, N_MICS, N_ROTORS, 3)) + 0.3).astype("float32")
    )


def test_codebook_lookup_and_growth():
    cb = DroneCodebook(8, names=["dregon"])
    assert cb.names() == ["dregon"] and "dregon" in cb
    z = cb(["dregon"])
    assert z.shape == (1, 8)
    # Growable by name without touching anything else; idempotent add.
    cb.add("michaels")
    assert set(cb.names()) == {"dregon", "michaels"}
    assert cb(["michaels", "dregon"]).shape == (2, 8)
    with pytest.raises(KeyError):
        cb(["unknown"])


def test_drone_conditioning_changes_output():
    # Two drone codes -> different audio (deterministic emitter + eval mode so
    # the difference is the code, not the random noise branch or random phases).
    model = get_model(
        "positional_harmonic_gen", sample_rate=SR, n_harmonics=8, use_diff_noise=False, cond_dim=4
    ).eval()
    cb = DroneCodebook(4, names=["a", "b"], init_std=0.5)
    rps = torch.full((1, N_ROTORS, 4096), 80.0)
    rel = _rel(1)
    out0 = model(rps, rel, cb(["a"]))
    out1 = model(rps, rel, cb(["b"]))
    assert out0.shape == (1, N_MICS, 4096)
    assert not torch.allclose(out0, out1)


def test_codebook_receives_gradient():
    # The external code is what we optimise; gradient must reach it.
    model = get_model(
        "positional_harmonic_gen", sample_rate=SR, n_harmonics=8, use_diff_noise=False, cond_dim=4
    )
    cb = DroneCodebook(4, names=["a", "b"], init_std=0.5)
    rps = torch.full((2, N_ROTORS, 4096), 80.0)
    out = model(rps, _rel(2), cb(["a", "b"]))
    out.pow(2).mean().backward()
    grads = [p.grad for p in cb.parameters()]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


def test_z_required_when_conditioned():
    model = get_model("positional_harmonic_gen", sample_rate=SR, n_harmonics=8, cond_dim=4)
    rps = torch.full((1, N_ROTORS, 4096), 80.0)
    with pytest.raises(ValueError, match="cond_dim"):
        model(rps, _rel(1))


def test_unconditioned_ignores_z():
    # cond_dim=0 (default): a code passed in is accepted but unused. Eval mode so
    # the two calls share zero phases (train mode would resample per call).
    model = get_model(
        "positional_harmonic_gen", sample_rate=SR, n_harmonics=8, use_diff_noise=False
    ).eval()
    rps = torch.full((1, N_ROTORS, 4096), 80.0)
    rel = _rel(1)
    assert torch.allclose(model(rps, rel), model(rps, rel, torch.zeros(1, 4)))


def test_few_shot_freeze_emitter_adapts_only_code():
    # Few-shot adaptation: freeze the generator, optimise only a fresh code.
    model = get_model(
        "positional_harmonic_gen", sample_rate=SR, n_harmonics=8, use_diff_noise=False, cond_dim=4
    )
    for p in model.parameters():
        p.requires_grad_(False)
    cb = DroneCodebook(4, names=["new_drone"], init_std=0.5)
    before = cb(["new_drone"]).detach().clone()
    opt = torch.optim.Adam(cb.parameters(), lr=1e-1)
    target = torch.randn(1, N_MICS, 4096)
    for _ in range(3):
        opt.zero_grad()
        pred = model(torch.full((1, N_ROTORS, 4096), 80.0), _rel(1), cb(["new_drone"]))
        (pred - target).pow(2).mean().backward()
        opt.step()
    # the code moved; the (frozen) model did not need any trainable params
    assert not torch.allclose(before, cb(["new_drone"]))
    assert all(not p.requires_grad for p in model.parameters())


# ── Michael's array geometry (from the rig photos + DJI Matrice 100) ──────────


def test_michaels_geometry():
    mic_pos, rotor_pos = get_michaels_geometry()
    assert mic_pos.shape == (8, 3)
    assert rotor_pos.shape == (MICHAELS_NUM_ROTORS, 3)

    # All 8 mics lie on the ring: same forward (X) offset, and radius
    # MIC_ARRAY_RADIUS about the centre in the Y-Z (lateral-vertical) plane.
    fwd = ARRAY_OFFSET_FORWARD
    center = np.array([fwd, 0.0, 0.33])
    assert np.allclose(mic_pos[:, 0], fwd)  # all at the forward offset
    radii = np.linalg.norm(mic_pos[:, 1:] - center[1:], axis=-1)
    assert np.allclose(radii, MIC_ARRAY_RADIUS)
    # adjacent-mic spacing ~ 60 mm (the spec's measured value)
    assert np.linalg.norm(mic_pos[0] - mic_pos[7]) == pytest.approx(0.060, abs=0.004)
    # The array centre sits forward of the front rotors (boom sticks out front),
    # not behind them — a front-edge-referenced offset, not body-centre 20 cm.
    front_rotor_x = rotor_pos[:, 0].max()
    assert fwd > front_rotor_x

    # Rotors: opposite motors are the diagonal == wheelbase; all at body height.
    assert np.allclose(rotor_pos[:, 2], 0.0)
    assert np.linalg.norm(rotor_pos[0] - rotor_pos[2]) == pytest.approx(
        WHEELBASE, abs=1e-6
    )  # RF-LB
    assert np.linalg.norm(rotor_pos[1] - rotor_pos[3]) == pytest.approx(
        WHEELBASE, abs=1e-6
    )  # LF-RB


def test_michaels_geometry_feeds_model():
    # Michael's geometry -> rel_pos -> model renders all 8 mics.
    mic_pos, rotor_pos = get_michaels_geometry()
    rel = torch.from_numpy(geometry_to_rel_pos(mic_pos, rotor_pos)).unsqueeze(0)  # (1, 8, 4, 3)
    rps = torch.full((1, MICHAELS_NUM_ROTORS, 4096), 80.0)
    model = get_model("positional_harmonic_gen", sample_rate=SR, n_harmonics=8)
    out = model(rps, rel)
    assert out.shape == (1, 8, 4096)
    assert torch.isfinite(out).all()
