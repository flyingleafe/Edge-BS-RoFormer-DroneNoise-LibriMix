"""Tests for training.validate.validate_config."""

from __future__ import annotations

from omegaconf import OmegaConf

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


# ─── C7/C8 — salience_rps end-to-end (proves the gap is actually closed) ─────


def test_validate_config_passes_a_salience_pipeline(tmp_path):
    """A basic_pitch_salience model + SalienceRPSBCELoss + SalienceBCEMetric
    pipeline over a tiny synthetic ``rps_prediction``-shaped dataset (the
    salience_rps task's input spec is identical: just ``mixture``/``rps``) —
    proves the BCE-target-from-rps derivation (REPLICATION.md § C7/C8) wires
    together end-to-end through the same ``validate_config`` gate
    ``train.py``/``eval.py`` run."""
    grid = {"fmin": 27.5, "n_bins": 264, "bins_per_octave": 36}
    cfg = OmegaConf.create(
        {
            "experiment_name": "tiny_salience_test",
            "seed": 0,
            "validate_only": True,
            "allow_dirty": True,
            "resume": False,
            "results_root": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "patience": 5,
            "grad_clip": None,
            "grad_accum_steps": 1,
            "amp": False,
            "samples_per_validation": None,
            "checkpoint_every": 0,
            "checkpoint": None,
            "data": {
                "train": {
                    "_target_": "tests.training._fixtures.TinyRPSFrameDataset",
                    "params": {"n_samples": 4, "duration_s": 0.5, "sample_rate": 16000, "seed": 0},
                    "iterable": False,
                },
                "valid": {
                    "_target_": "tests.training._fixtures.TinyRPSFrameDataset",
                    "params": {"n_samples": 2, "duration_s": 0.5, "sample_rate": 16000, "seed": 1},
                    "iterable": False,
                },
                "batch_size": None,
                "num_workers": None,
            },
            "model": {
                "task": "salience_rps",
                "task_params": {"n_channels": None, "sr": [16000, 1], "frame_rate": [16000, 256]},
                "_target_": "models.registry.build_model",
                "params": {
                    "name": "basic_pitch_salience",
                    "n_fft": 512,
                    "hop_length": 256,
                    "num_rotors": 4,
                    "sr": 16000,
                    "n_harmonics": 2,
                },
                "model_type": None,
                "legacy_config_path": None,
            },
            "loss": {
                "terms": [
                    {
                        "name": "bce",
                        "_target_": "losses.SalienceRPSBCELoss",
                        "weight": 1.0,
                        "params": {**grid, "blur_bins": 1, "pos_weight": 2.0, "rate": [16000, 256]},
                    }
                ]
            },
            "metrics": {
                "terms": [
                    {
                        "name": "bce",
                        "_target_": "metrics.SalienceBCEMetric",
                        "params": {**grid, "blur_bins": 1, "pos_weight": 2.0, "rate": [16000, 256]},
                    }
                ]
            },
            "optim": {
                "optimizer": "adamw",
                "lr": 1.0e-3,
                "weight_decay": 0.0,
                "optimizer_params": {},
                "patience": 5,
                "factor": 0.5,
                "monitor": "bce",
                "monitor_mode": "min",
            },
            "logging": {
                "enabled": True,
                "entity": "test",
                "project": "test",
                "mode": "disabled",
                "tags": [],
            },
            "artifacts": {
                "enabled": False,
                "bucket": "ml-data",
                "prefix": "artifacts",
                "upload_checkpoints": True,
                "upload_val_samples": True,
                "num_val_samples": 0,
            },
            "lora": {
                "enabled": False,
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": None,
                "checkpoint": None,
            },
        }
    )
    problems = validate_config(cfg)
    assert problems == []


# ─── E2/E3 — noise_generation end-to-end (proves the codec fix works) ────────


def test_validate_config_passes_a_conditioned_noise_generation_pipeline(tmp_path):
    """positional_harmonic_gen (cond_dim>0, DroneCodebook-wrapped) +
    MultiScaleSTFTLoss + AuraMRSTFTMetric over a tiny synthetic
    ``noise_generation`` dataset — proves the codec's rel_pos fix and
    conditioned drone-name resolution (REPLICATION.md § E1/E2/E3) actually
    forward-pass end-to-end (this is exactly what ``NoiseGenerationCodec``
    used to get wrong: mic_pos/rotor_pos/drone_id passed straight through
    as kwargs the model doesn't accept)."""
    cfg = OmegaConf.create(
        {
            "experiment_name": "tiny_noise_gen_test",
            "seed": 0,
            "validate_only": True,
            "allow_dirty": True,
            "resume": False,
            "results_root": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "patience": 5,
            "grad_clip": None,
            "grad_accum_steps": 1,
            "amp": False,
            "samples_per_validation": None,
            "checkpoint_every": 0,
            "checkpoint": None,
            "data": {
                "train": {
                    "_target_": "tests.training._fixtures.TinyNoiseGenFrameDataset",
                    "params": {"n_samples": 4, "duration_s": 0.5, "sample_rate": 16000, "seed": 0},
                    "iterable": False,
                },
                "valid": {
                    "_target_": "tests.training._fixtures.TinyNoiseGenFrameDataset",
                    "params": {"n_samples": 2, "duration_s": 0.5, "sample_rate": 16000, "seed": 1},
                    "iterable": False,
                },
                "batch_size": None,
                "num_workers": None,
            },
            "model": {
                "task": "noise_generation",
                "task_params": {"sr": [16000, 1], "conditioned": True, "return_dict": False},
                "_target_": "models.registry.build_noise_gen_model",
                "params": {
                    "model_name": "positional_harmonic_gen",
                    "sample_rate": 16000,
                    "n_harmonics": 4,
                    "use_diff_noise": True,
                    "cond_dim": 4,
                    "drone_names": ["dregon", "michaels"],
                },
                "model_type": None,
                "legacy_config_path": None,
            },
            "loss": {
                "terms": [
                    {
                        "name": "spectral",
                        "_target_": "losses.MultiScaleSTFTLoss",
                        "weight": 1.0,
                        "params": {
                            "n_channels": 1,
                            "sr": [16000, 1],
                            "pred_key": "audio",
                            "target_key": "audio",
                            "n_ffts": [512, 256],
                        },
                    }
                ]
            },
            "metrics": {
                "terms": [
                    {
                        "name": "mrstft",
                        "_target_": "metrics.AuraMRSTFTMetric",
                        "params": {
                            "n_channels": 1,
                            "sr": [16000, 1],
                            "sample_rate": 16000,
                            "pred_key": "audio",
                            "target_key": "audio",
                        },
                    }
                ]
            },
            "optim": {
                "optimizer": "adam",
                "lr": 1.0e-3,
                "weight_decay": 0.0,
                "optimizer_params": {},
                "patience": 5,
                "factor": 0.5,
                "monitor": "mrstft",
                "monitor_mode": "max",
            },
            "logging": {
                "enabled": True,
                "entity": "test",
                "project": "test",
                "mode": "disabled",
                "tags": [],
            },
            "artifacts": {
                "enabled": False,
                "bucket": "ml-data",
                "prefix": "artifacts",
                "upload_checkpoints": True,
                "upload_val_samples": True,
                "num_val_samples": 0,
            },
            "lora": {
                "enabled": False,
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": None,
                "checkpoint": None,
            },
        }
    )
    problems = validate_config(cfg)
    assert problems == []
