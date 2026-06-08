"""Tests for `train_rps_predictor` — pure functions, dataset, model shapes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── stft_time_frames ─────────────────────────────────────────────────────


def test_stft_time_frames_typical():
    from train_rps_predictor import stft_time_frames

    assert stft_time_frames(16000, 512, 2048) > 0


# ── get_model ────────────────────────────────────────────────────────────


def test_get_model_known_names():
    from train_rps_predictor import MODEL_REGISTRY, get_model

    for name in MODEL_REGISTRY:
        model = get_model(name, n_fft=256, hop_length=64, num_rotors=4)
        assert model is not None


def test_get_model_unknown_name_raises():
    from train_rps_predictor import get_model

    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model_xyz")


# ── pairwise_mse ────────────────────────────────────────────────────────


def test_pairwise_mse_shape():
    from train_rps_predictor import pairwise_mse

    est = torch.randn(2, 4, 32)
    tgt = torch.randn(2, 4, 32)
    pw = pairwise_mse(est, tgt)
    assert pw.shape == (2, 4, 4)


def test_pairwise_mse_identity():
    from train_rps_predictor import pairwise_mse

    x = torch.randn(2, 4, 32)
    pw = pairwise_mse(x, x)
    # Diagonal (i=j) should be near-zero.
    for b in range(2):
        for i in range(4):
            assert pw[b, i, i] < 1e-6


# ── pit_mse_loss ────────────────────────────────────────────────────────


def test_pit_mse_loss_identity():
    from train_rps_predictor import pit_mse_loss

    x = torch.randn(2, 4, 32)
    loss = pit_mse_loss(x, x)
    assert loss.item() < 1e-6


def test_pit_mse_loss_permutation_invariant():
    from train_rps_predictor import pit_mse_loss

    x = torch.randn(2, 4, 32)
    # Permute rotors
    perm = torch.tensor([2, 0, 3, 1])
    x_perm = x[:, perm, :]
    loss = pit_mse_loss(x, x_perm)
    assert loss.item() < 1e-6


def test_pit_mse_loss_increases_with_noise():
    from train_rps_predictor import pit_mse_loss

    x = torch.randn(2, 4, 32)
    loss_clean = pit_mse_loss(x, x)
    loss_noisy = pit_mse_loss(x, x + 0.1 * torch.randn(2, 4, 32))
    assert loss_noisy > loss_clean


# ── _flatten_channels ────────────────────────────────────────────────────


def test_flatten_channels_mono_noop():
    from train_rps_predictor import _flatten_channels

    audio = torch.randn(8, 16000)  # (B, T)
    rps = torch.randn(8, 4, 32)  # (B, 4, F)
    a_flat, r_flat, C = _flatten_channels(audio, rps)
    assert C == 1
    assert a_flat.shape == (8, 16000)
    assert r_flat.shape == (8, 4, 32)


def test_flatten_channels_multichannel():
    from train_rps_predictor import _flatten_channels

    audio = torch.randn(4, 3, 16000)  # (B, C, T)
    rps = torch.randn(4, 4, 32)  # (B, 4, F)
    a_flat, r_flat, C = _flatten_channels(audio, rps)
    assert C == 3
    assert a_flat.shape == (12, 16000)
    assert r_flat.shape == (12, 4, 32)


# ── model forward shapes ─────────────────────────────────────────────────


def test_simple_conv_forward_shape():
    from train_rps_predictor import get_model

    model = get_model("simple_conv", n_fft=256, hop_length=64, num_rotors=4)
    audio = torch.randn(2, 16000)
    with torch.no_grad():
        out = model(audio)
    assert out.shape[0] == 2
    assert out.shape[1] == 4


def test_dcunet_enc_rps_forward_shape():
    from train_rps_predictor import get_model

    model = get_model("dcunet_enc_rps", n_fft=256, hop_length=64, num_rotors=4)
    audio = torch.randn(2, 16000)
    with torch.no_grad():
        out = model(audio)
    assert out.shape[0] == 2
    assert out.shape[1] == 4


def test_dccrn_enc_rps_forward_shape():
    from train_rps_predictor import get_model

    model = get_model("dccrn_enc_rps", n_fft=256, hop_length=64, num_rotors=4)
    audio = torch.randn(2, 16000)
    with torch.no_grad():
        out = model(audio)
    assert out.shape[0] == 2
    assert out.shape[1] == 4


def test_dccrn_lite_rps_forward_shape():
    from train_rps_predictor import get_model

    model = get_model("dccrn_lite_rps", n_fft=256, hop_length=64, num_rotors=4)
    audio = torch.randn(2, 16000)
    with torch.no_grad():
        out = model(audio)
    assert out.shape[0] == 2
    assert out.shape[1] == 4


# ── DREGONRPSDataset (mock filesystem) ───────────────────────────────────


def test_dataset_len(tmp_path):
    import soundfile as sf

    from train_rps_predictor import DREGONRPSDataset

    for i in range(3):
        d = tmp_path / f"sample_{i:05d}"
        d.mkdir()
        audio = np.zeros(8000, dtype=np.float32)
        sf.write(str(d / "mixture.wav"), audio, 16000)
        np.save(str(d / "rps.npy"), np.zeros((4, 100), dtype=np.float32))

    ds = DREGONRPSDataset(str(tmp_path), n_fft=256, hop_length=64)
    assert len(ds) == 3


def test_dataset_getitem_shape(tmp_path):
    import soundfile as sf

    from train_rps_predictor import DREGONRPSDataset

    d = tmp_path / "sample_00000"
    d.mkdir()
    audio = np.zeros(8000, dtype=np.float32)
    sf.write(str(d / "mixture.wav"), audio, 16000)
    np.save(str(d / "rps.npy"), np.zeros((4, 100), dtype=np.float32))

    ds = DREGONRPSDataset(str(tmp_path), n_fft=256, hop_length=64)
    a, r = ds[0]
    assert a.dim() == 1
    assert r.shape[0] == 4


# ── evaluate with mock model ─────────────────────────────────────────────


def test_evaluate_with_identity_model(tmp_path):
    import soundfile as sf

    from train_rps_predictor import DREGONRPSDataset, evaluate

    for i in range(2):
        d = tmp_path / f"sample_{i:05d}"
        d.mkdir()
        audio = np.zeros(4000, dtype=np.float32)
        sf.write(str(d / "mixture.wav"), audio, 16000)
        rps = np.random.randn(4, 50).astype(np.float32)
        np.save(str(d / "rps.npy"), rps)

    ds = DREGONRPSDataset(str(tmp_path), n_fft=256, hop_length=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    class IdentityModel(torch.nn.Module):
        def forward(self, audio):
            n_frames = audio.shape[-1] // 64 + 1
            return torch.randn(audio.shape[0], 4, n_frames)

    model = IdentityModel()
    metrics = evaluate(model, loader, torch.device("cpu"), len(ds))
    assert "mse" in metrics
    assert "mae_frame" in metrics
    assert "mae_clip" in metrics
    assert "r2" in metrics


# ── wandb_init ───────────────────────────────────────────────────────────


def test_wandb_init_no_key_disables(monkeypatch):
    import argparse

    from train_rps_predictor import wandb_init

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args = argparse.Namespace(
        wandb_key="",
        save_path="/tmp/fake",
        epochs=10,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        pit_loss=True,
        smoothness_weight=0.0,
        n_fft=256,
        hop_length=64,
        data_root="/tmp",
    )
    wandb_init(args, "test_model")
    import wandb

    assert wandb.run is None or wandb.run.disabled


# ── EvalResult methods ───────────────────────────────────────────────────


def test_eval_result_per_snr_stratifies():
    from tasks.rps_prediction import EvalResult

    rows = [
        {
            "sample": "s1",
            "mse": 1.0,
            "mae_frame": 0.5,
            "mae_clip": 0.3,
            "r2": 0.8,
            "input_snr": -28.0,
        },
        {
            "sample": "s2",
            "mse": 2.0,
            "mae_frame": 0.7,
            "mae_clip": 0.4,
            "r2": 0.6,
            "input_snr": -12.0,
        },
        {
            "sample": "s3",
            "mse": 0.5,
            "mae_frame": 0.3,
            "mae_clip": 0.2,
            "r2": 0.9,
            "input_snr": -3.0,
        },
    ]
    result = EvalResult(per_sample=rows, model_spec="test")
    snr_rows = result.per_snr()
    assert len(snr_rows) >= 3  # at least [-30,-25), [-15,-10), [-5,0) + Overall


def test_eval_result_to_json_writes_file(tmp_path):
    from tasks.rps_prediction import EvalResult

    rows = [{"sample": "s1", "mse": 1.0, "mae_frame": 0.5, "mae_clip": 0.3, "r2": 0.8}]
    result = EvalResult(per_sample=rows, aggregate={"mse": 1.0}, model_spec="test")
    p = tmp_path / "out.json"
    result.to_json(p)
    assert p.exists()
    import json

    data = json.loads(p.read_text())
    assert data["model_spec"] == "test"


# ── train_model smoke test ───────────────────────────────────────────────


def test_train_model_smoke(tmp_path, monkeypatch):
    """train_model runs for 1 epoch without crashing."""
    import argparse

    import soundfile as sf

    from train_rps_predictor import train_model

    # Create tiny dataset
    for split in ["train", "valid"]:
        split_dir = tmp_path / split
        split_dir.mkdir()
        for i in range(3):
            d = split_dir / f"sample_{i:05d}"
            d.mkdir()
            audio = np.random.randn(4000).astype(np.float32)
            sf.write(str(d / "mixture.wav"), audio, 16000)
            rps = np.random.randn(4, 50).astype(np.float32)
            np.save(str(d / "rps.npy"), rps)

    save_path = tmp_path / "checkpoints"
    save_path.mkdir()

    # Disable wandb
    monkeypatch.setenv("WANDB_MODE", "disabled")

    args = argparse.Namespace(
        device="cpu",
        epochs=1,
        patience=10,
        batch_size=2,
        lr=1e-4,
        weight_decay=0.0,
        grad_clip=5.0,
        data_root=str(tmp_path),
        save_path=str(save_path),
        pit_loss=True,
        smoothness_weight=0.0,
        n_fft=256,
        hop_length=64,
    )
    result = train_model("simple_conv", args)
    assert "mse" in result
    assert "best_path" in result
