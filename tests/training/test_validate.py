"""Tests for training.validate.validate_config."""

from __future__ import annotations

from tests.training.conftest import make_tiny_config
from training.validate import validate_config


def test_validate_config_passes_a_matched_pipeline(tmp_path):
    cfg = make_tiny_config(results_root=str(tmp_path))
    problems = validate_config(cfg)
    assert problems == []


def test_validate_config_catches_a_rate_mismatch(tmp_path):
    # The model/task declares frame_rate=8000/512 while the dataset (and
    # loss/metrics, built at 16000/512 in make_tiny_config) actually produces
    # RPS on the 16000/512 grid — a deliberately mismatched pipeline.
    cfg = make_tiny_config(results_root=str(tmp_path), frame_rate=[8000, 512])
    problems = validate_config(cfg)
    assert problems != []
    assert any("rate" in p for p in problems)


def test_validate_config_reports_missing_dataset_entry(tmp_path):
    cfg = make_tiny_config(results_root=str(tmp_path))
    # speech_enhancement's masked_mse loss needs a "target" entry the tiny
    # rps_prediction dataset never provides.
    cfg.loss.terms = [
        {
            "name": "masked_mse",
            "_target_": "losses.MaskedLoss",
            "weight": 1.0,
            "params": {"n_channels": None, "sr": [16000, 1]},
        }
    ]
    problems = validate_config(cfg)
    assert problems != []
    assert any("target" in p for p in problems)
