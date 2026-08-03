"""Tests for the online-mixing pipeline compiler (``build_online_mix_pipeline``).

The policy-resolution unit tests are unchanged from the class-based mixer (the
policy semantics are preserved verbatim); the behavior tests exercise the
compiled dload pipeline over synthetic local-repo datasets (see
``tests/data_processing/conftest.py`` for the shared fixtures).
"""

from __future__ import annotations

import itertools

import numpy as np
from omegaconf import OmegaConf

from data_processing.frames import get_meta
from data_processing.online_mixing import _resolve_policy, _resolve_probability

from .conftest import SPEECH_DATASET, SR  # noqa: F401  (constants; fixtures auto-discovered)


def _rps_cfg(**over):
    cfg = {
        "sample_rate": SR,
        "duration_s": 1.0,
        "base_seed": 123,
        "sources": {
            "noise": [
                {
                    "kind": "frames",
                    "dataset": "MICHAELS-FRAMES-TEST",
                    "min_motor_rps": 0.0,
                }
            ],
            "speech": [{"dataset": SPEECH_DATASET, "subpath": "LibriSpeech/train-clean-100"}],
        },
        "policy": {"source_prob": 1.0, "snr_db": -10.0, "speech_per_channel": "shared"},
    }
    cfg.update(over)
    return cfg


# ─── policy resolution (unchanged semantics) ──────────────────────────────────


def test_resolve_policy_uses_sample_id_stages():
    policy = {
        "stages": [
            {"until": 50_000, "source_prob": 1.0},
            {"until": None, "source_prob": 1.0, "augmentations": {"probability": 0.5}},
        ]
    }

    assert "augmentations" not in _resolve_policy(policy, 49_999)
    assert _resolve_policy(policy, 50_000)["augmentations"]["probability"] == 0.5


def test_resolve_probability_ramp_interpolation():
    ramp = {"ramp": {"from": 25_000, "until": 125_000, "start": 0.0, "end": 0.7}}

    assert _resolve_probability(0.5, 0) == 0.5
    assert _resolve_probability(1, 0) == 1.0
    # Before/at the window start: clamped to `start` but FLOORED at 1e-9 — a
    # ramp never resolves to exactly 0.0 (the fire draw must stay consumed).
    assert _resolve_probability(ramp, 0) == 1e-9
    assert _resolve_probability(ramp, 25_000) == 1e-9
    # Inside: linear interpolation.
    assert abs(_resolve_probability(ramp, 75_000) - 0.35) < 1e-12
    assert abs(_resolve_probability(ramp, 100_000) - 0.525) < 1e-12
    # At/after the window end: clamped to `end`.
    assert _resolve_probability(ramp, 125_000) == 0.7
    assert _resolve_probability(ramp, 10_000_000) == 0.7


def test_resolve_policy_materializes_ramps_without_mutation():
    ramp = {"ramp": {"from": 100, "until": 200, "start": 0.0, "end": 1.0}}
    policy = {
        "stages": [
            {
                "until": None,
                "augmentations": {"probability": ramp, "choices": ["random_polarity"]},
                "noise_augmentations": [
                    {"probability": 1.0, "choices": ["spec_mask"]},
                    {"probability": ramp, "choices": ["floor_inject"]},
                ],
            }
        ]
    }

    resolved = _resolve_policy(policy, 150)

    assert resolved["augmentations"]["probability"] == 0.5
    assert resolved["noise_augmentations"][0]["probability"] == 1.0
    assert resolved["noise_augmentations"][1]["probability"] == 0.5
    # The source policy is never mutated (it is shared across samples/workers).
    stage = policy["stages"][0]
    assert stage["augmentations"]["probability"] is ramp
    assert stage["noise_augmentations"][1]["probability"] is ramp
    # Non-ramped blocks are reused as-is; ramped ones are fresh copies.
    assert resolved["noise_augmentations"][0] is stage["noise_augmentations"][0]
    assert resolved["noise_augmentations"][1] is not stage["noise_augmentations"][1]


# ─── compiled-pipeline behavior ────────────────────────────────────────────────


def _take(cfg, n, start=0):
    from data_processing.online_mixing import build_online_mix_pipeline

    pipe = build_online_mix_pipeline({**cfg, "start_sample_id": start})
    return list(itertools.islice(iter(pipe), n))


def test_pipeline_sample_shapes_determinism_and_ids(michaels_frames_dataset, speech_dataset):
    cfg = _rps_cfg()
    run_a = _take(cfg, 4)
    run_b = _take(cfg, 4)

    for fa, fb in zip(run_a, run_b):
        np.testing.assert_array_equal(
            np.asarray(fa["mixture"].data), np.asarray(fb["mixture"].data)
        )
        np.testing.assert_array_equal(np.asarray(fa["rps"].data), np.asarray(fb["rps"].data))

    first = run_a[0]
    assert first["mixture"].shape == (2, SR)  # the 2-mic synthetic recording
    assert first["rps"].shape == (4, SR // 512 + 1)
    assert [get_meta(f, "sample_id") for f in run_a] == [0, 1, 2, 3]
    # different samples -> different content
    assert not np.array_equal(
        np.asarray(run_a[0]["mixture"].data), np.asarray(run_a[1]["mixture"].data)
    )


def test_pipeline_start_sample_id_offsets_the_id_stream(michaels_frames_dataset, speech_dataset):
    frames = _take(_rps_cfg(), 3, start=41)
    assert [get_meta(f, "sample_id") for f in frames] == [41, 42, 43]


def test_pipeline_speech_excludes_held_out_speakers(michaels_frames_dataset, speech_dataset):
    # The speech fixture keys carry speaker ids 19/103/200; excluding 200 must
    # leave the stream functional (it simply never draws 200/*).
    cfg = _rps_cfg()
    cfg["sources"]["speech"] = [
        {
            "dataset": SPEECH_DATASET,
            "subpath": "LibriSpeech/train-clean-100",
            "exclude": ["200"],
        }
    ]
    frames = _take(cfg, 3)
    assert all(f["mixture"].shape[-1] == SR for f in frames)


def test_online_mix_frame_dataset_dataloader_batches(michaels_frames_dataset, speech_dataset):
    from torch.utils.data import DataLoader

    from data_processing.collate import frame_collate
    from data_processing.frame_datasets import OnlineMixFrameDataset

    ds = OnlineMixFrameDataset.from_config(_rps_cfg())
    loader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=frame_collate)
    batch = next(iter(loader))
    assert batch["mixture"].data.shape == (4, 2, SR)
    assert batch["rps"].data.shape == (4, 4, SR // 512 + 1)


def test_noise_augmentation_list_blocks_apply_sequentially(michaels_frames_dataset, speech_dataset):
    # Three policies sharing every seed: the underlying chunk at a given stream
    # position is identical; only the aug blocks differ.
    def cfg_with(p_fs: float, p_floor: float):
        cfg = _rps_cfg(base_seed=7)
        cfg["policy"] = {
            "source_prob": 1.0,
            "snr_db": -10.0,
            "speech_per_channel": "shared",
            "noise_augmentations": [
                {
                    "probability": p_fs,
                    "choices": [{"freq_scale": {"alpha_low": 1.2, "alpha_high": 1.2}}],
                },
                {
                    "probability": p_floor,
                    "choices": [{"floor_inject": {"level_low_db": -6.0, "level_high_db": -6.0}}],
                },
            ],
        }
        return cfg

    pos = 5
    both = _take(cfg_with(1.0, 1.0), 1, start=pos)[0]
    fs_only = _take(cfg_with(1.0, 1e-9), 1, start=pos)[0]
    neither = _take(cfg_with(1e-9, 1e-9), 1, start=pos)[0]

    both_audio = np.asarray(both["mixture"].data)
    fs_audio = np.asarray(fs_only["mixture"].data)
    # Block 2 (floor_inject) fires ON TOP of block 1's output: audio differs,
    # labels are preserved (floor_inject is label-preserving).
    assert not np.array_equal(both_audio, fs_audio)
    np.testing.assert_allclose(
        np.asarray(both["rps"].data), np.asarray(fs_only["rps"].data), rtol=1e-4, atol=1e-4
    )
    # Block 1 (freq_scale, fixed alpha 1.2) rescales the labels vs the no-fire
    # stream.
    ratio = np.asarray(fs_only["rps"].data) / np.asarray(neither["rps"].data)
    assert np.allclose(ratio, 1.2, atol=0.02)


def test_v4_michaels_online_config_uses_librispeech_dataset_form():
    cfg = OmegaConf.load("conf/online_mix/online_mix_v4_michaels_train_no_room1.yaml")
    speech = cfg.sources.speech[0]

    assert speech.dataset == "librispeech"
    assert speech.subpath == "LibriSpeech/train-clean-100"
    assert "root" not in speech and "cache" not in speech
