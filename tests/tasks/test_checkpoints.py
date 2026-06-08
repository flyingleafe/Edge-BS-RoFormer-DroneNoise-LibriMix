"""Tests for `tasks.checkpoints.load_model` error paths."""

from __future__ import annotations

import pytest


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
    with pytest.raises(ValueError, match="not yet supported"):
        load_model(f"unknown_model_type_xyz@{dummy}")
