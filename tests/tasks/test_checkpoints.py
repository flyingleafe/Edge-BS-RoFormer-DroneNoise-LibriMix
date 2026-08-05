"""Tests for `tasks.checkpoints.load_model` — one round trip plus the error paths."""

from __future__ import annotations

import pytest
import torch


def test_load_model_round_trip_runs_a_forward_pass(tmp_path):
    """Save a registry model's ``state_dict``, load it back through the
    ``Type@ckpt`` spec, and make sure the loaded module is the same weights,
    in eval mode, on CPU, and callable on the task's audio contract."""
    from models.registry import build_model
    from tasks.checkpoints import load_model

    ref = build_model("simple_conv", n_fft=2048, hop_length=512, num_rotors=4)
    ckpt = tmp_path / "best.pt"
    torch.save(ref.state_dict(), ckpt)

    model = load_model(f"simple_conv@{ckpt}")

    assert not model.training  # load_model calls .eval()
    assert next(model.parameters()).device.type == "cpu"
    got, want = model.state_dict(), ref.state_dict()
    assert got.keys() == want.keys()
    assert all(torch.equal(got[k], want[k]) for k in want)

    audio = torch.zeros(2, 16000)
    with torch.no_grad():
        out = model(audio)
    assert out.shape == (2, 4, 16000 // 512 + 1)  # (B, rotors, T_stft)


def test_load_model_accepts_a_wrapped_state_dict(tmp_path):
    """Training checkpoints nest the weights under ``"state_dict"``."""
    from models.registry import build_model
    from tasks.checkpoints import load_model

    ref = build_model("simple_conv", n_fft=2048, hop_length=512, num_rotors=4)
    ckpt = tmp_path / "epoch12.pt"
    torch.save({"state_dict": ref.state_dict(), "epoch": 12}, ckpt)

    model = load_model(f"simple_conv@{ckpt}")
    got, want = model.state_dict(), ref.state_dict()
    assert all(torch.equal(got[k], want[k]) for k in want)


def test_load_model_no_at_symbol():
    from tasks.checkpoints import load_model

    with pytest.raises(ValueError, match="no '@' found"):
        load_model("no_at_symbol")


def test_load_model_empty_type():
    from tasks.checkpoints import load_model

    with pytest.raises(ValueError, match="missing model type"):
        load_model("@/path/to/ckpt.pt")


def test_load_model_empty_path():
    from tasks.checkpoints import load_model

    with pytest.raises(ValueError, match="missing checkpoint path"):
        load_model("simple_conv@")


def test_load_model_missing_checkpoint():
    from tasks.checkpoints import load_model

    with pytest.raises(FileNotFoundError, match="not found"):
        load_model("simple_conv@/nonexistent/path.pt")


def test_load_model_unknown_type(tmp_path):
    from tasks.checkpoints import load_model

    dummy = tmp_path / "fake.pt"
    dummy.write_text("dummy")
    with pytest.raises(ValueError, match="Unknown model type"):
        load_model(f"unknown_model_type_xyz@{dummy}")


def test_load_model_legacy_type_not_supported(tmp_path):
    from tasks.checkpoints import load_model

    dummy = tmp_path / "fake.pt"
    dummy.write_text("dummy")
    with pytest.raises(ValueError, match="not yet supported"):
        load_model(f"dcunet@{dummy}")
