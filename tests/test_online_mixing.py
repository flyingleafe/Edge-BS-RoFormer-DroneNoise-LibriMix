from __future__ import annotations

import numpy as np
import soundfile as sf
import tdseries as td
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_processing.frames import get_meta, make_recording_frame
from data_processing.online_mixing import (
    AudioFileSourcePool,
    OnlineMixIterableDataset,
    TimeFrameNoisePool,
    _resolve_policy,
    _resolve_probability,
)


def _make_noise_tf(
    duration_s: float = 2.0,
    sr: int = 16000,
    recording_id: str = "synthetic",
) -> td.Frame:
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.stack(
        [0.01 * np.sin(2 * np.pi * (100 + 10 * ch) * t) for ch in range(8)], axis=0
    ).astype(np.float32)
    audio_us = td.uniform(audio, sr, dims=("mic", "time"), t_start=0.0)

    motor_t = np.linspace(0.0, duration_s - 0.01, 200, dtype=np.float64)
    rps = np.stack([60 + i + 0.5 * motor_t for i in range(4)], axis=0).astype(np.float32)
    rps_es = td.events(motor_t, rps, dims=("rotor", "time"), t_start=0.0, t_end=duration_s)
    return make_recording_frame(
        {"audio": audio_us, "rps": rps_es}, meta={"recording_id": recording_id}
    )


def _make_source_pool(tmp_path, sr: int = 16000) -> AudioFileSourcePool:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for i in range(3):
        t = np.arange(sr * 2, dtype=np.float32) / sr
        audio = (0.02 * np.sin(2 * np.pi * (220 + i * 30) * t)).astype(np.float32)
        sf.write(source_dir / f"src_{i}.wav", audio, sr)
    return AudioFileSourcePool.from_config(
        {"root": str(source_dir), "glob": "*.wav"}, duration_s=1.0, sample_rate=sr
    )


def test_noise_pool_from_config_excludes_validation_recording(monkeypatch):
    import data_processing.online_mixing as om

    frames = [
        _make_noise_tf(recording_id="free-flight_nosource_room1"),
        _make_noise_tf(recording_id="free-flight_nosource_room2"),
    ]
    monkeypatch.setattr(om, "load_dregon_timeframes", lambda *args, **kwargs: frames)

    pool = TimeFrameNoisePool.from_config(
        {
            "kind": "dregon",
            "root": "data",
            "split": "in_flight_noise",
            "exclude_recording_ids": ["free-flight_nosource_room1"],
            "min_motor_rps": 0.0,
        },
        duration_s=1.0,
        sample_rate=16000,
    )

    ids = [get_meta(r["tf"], "recording_id") for r in pool.records]
    assert ids == ["free-flight_nosource_room2"]


def test_resolve_policy_uses_sample_id_stages():
    policy = {
        "stages": [
            {"until": 50_000, "source_prob": 1.0},
            {"until": None, "source_prob": 1.0, "augmentations": {"probability": 0.5}},
        ]
    }

    assert "augmentations" not in _resolve_policy(policy, 49_999)
    assert _resolve_policy(policy, 50_000)["augmentations"]["probability"] == 0.5


def test_online_mix_generate_sample_shape_and_determinism(tmp_path):
    noise_pool = TimeFrameNoisePool([_make_noise_tf()], min_motor_rps=0.0, duration_s=1.0)
    source_pool = _make_source_pool(tmp_path)
    ds = OnlineMixIterableDataset(
        noise_pool,
        source_pool,
        policy={"source_prob": 1.0, "snr_db": -10.0, "speech_per_channel": "shared"},
        base_seed=123,
        duration_s=1.0,
        sample_rate=16000,
        hop_length=512,
    )

    audio1, rps1 = ds.generate_sample(42)
    audio2, rps2 = ds.generate_sample(42)
    audio3, rps3 = ds.generate_sample(43)

    assert audio1.shape == (8, 16000)
    assert rps1.shape == (4, 32)
    assert torch.equal(audio1, audio2)
    assert torch.equal(rps1, rps2)
    assert not torch.equal(audio1, audio3)
    assert not torch.equal(rps1, rps3)


def test_audio_source_pool_packed_int16_cache_is_reused(tmp_path, capsys):
    source_dir = tmp_path / "sources_packed"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    sr = 16000
    sf.write(source_dir / "a.wav", np.linspace(-0.1, 0.1, sr, dtype=np.float32), sr)
    cfg = {
        "root": str(source_dir),
        "glob": "*.wav",
        "cache": {"mode": "packed_int16", "dir": str(cache_dir)},
    }

    pool1 = AudioFileSourcePool.from_config(cfg, duration_s=1.0, sample_rate=sr)
    first_out = capsys.readouterr().out
    pool2 = AudioFileSourcePool.from_config(cfg, duration_s=1.0, sample_rate=sr)
    second_out = capsys.readouterr().out
    sample = pool2.sample_array(np.random.default_rng(0), channels=2, mode="shared")

    assert "Creating source cache" in first_out
    assert "Reusing source cache" in second_out
    assert sample.shape == (2, sr)
    assert pool1._packed_index is not None
    assert pool2._packed_data is not None


def test_audio_source_pool_memory_cache_keeps_sampling_interface(tmp_path):
    source_dir = tmp_path / "sources_cache"
    source_dir.mkdir()
    sr = 16000
    sf.write(source_dir / "a.wav", np.ones(sr, dtype=np.float32) * 0.01, sr)

    pool = AudioFileSourcePool.from_config(
        {"root": str(source_dir), "glob": "*.wav", "cache": {"mode": "memory"}},
        duration_s=1.0,
        sample_rate=sr,
    )

    sample = pool.sample_array(np.random.default_rng(0), channels=8, mode="shared")

    assert sample.shape == (8, sr)
    assert pool._memory_cache is not None


def test_v4_michaels_online_config_uses_original_librispeech_not_generated_vocals():
    cfg = OmegaConf.load("conf/online_mix/online_mix_v4_michaels_train_no_room1.yaml")
    speech = cfg.sources.speech[0]

    assert speech.root == "data/librispeech/LibriSpeech/train-clean-100"
    assert speech.glob == "**/*.flac"
    assert "datasets/DREGON-LM-V4-michaels/train" not in speech.root


def test_v4_michaels_online_config_cache_dir_is_env_configurable(monkeypatch, tmp_path):
    override = tmp_path / "online-cache"
    monkeypatch.setenv("ONLINE_MIX_SOURCE_CACHE_DIR", str(override))

    cfg = OmegaConf.load("conf/online_mix/online_mix_v4_michaels_train_no_room1.yaml")
    speech = cfg.sources.speech[0]

    assert speech.cache.mode == "packed_int16"
    assert speech.cache.dir == str(override)


def test_online_mix_dataloader_batches_multichannel_tensors(tmp_path):
    noise_pool = TimeFrameNoisePool([_make_noise_tf()], min_motor_rps=0.0, duration_s=1.0)
    ds = OnlineMixIterableDataset(
        noise_pool,
        _make_source_pool(tmp_path),
        policy={"source_prob": 0.5, "snr_db": {"uniform": {"low": -30, "high": 0}}},
        base_seed=1,
        duration_s=1.0,
        sample_rate=16000,
        hop_length=512,
    )
    loader = DataLoader(ds, batch_size=4, num_workers=2)

    audio, rps = next(iter(loader))

    assert audio.shape == (4, 8, 16000)
    assert rps.shape == (4, 4, 32)
    assert audio.dtype == torch.float32
    assert rps.dtype == torch.float32


def test_audio_source_pool_packed_cache_skips_unreadable_files(tmp_path):
    source_dir = tmp_path / "sources_corrupt"
    cache_dir = tmp_path / "cache_corrupt"
    source_dir.mkdir()
    sr = 16000
    for i in range(3):
        sf.write(source_dir / f"ok_{i}.wav", np.linspace(-0.1, 0.1, sr, dtype=np.float32), sr)
    # Deliberately corrupt file (garbage bytes, .wav extension) — must be
    # skipped with a warning, not kill the cache build (corrupt-flac-in-
    # librispeech scenario).
    (source_dir / "bad.wav").write_bytes(b"RIFFgarbage-not-audio")
    cfg = {
        "root": str(source_dir),
        "glob": "*.wav",
        "cache": {"mode": "packed_int16", "dir": str(cache_dir)},
    }

    pool1 = AudioFileSourcePool.from_config(cfg, duration_s=1.0, sample_rate=sr)
    assert len(pool1.files) == 3
    assert all(p.name.startswith("ok_") for p in pool1.files)
    assert pool1._packed_index is not None and len(pool1._packed_index) == 3

    # Reuse path must drop the same files via the manifest, keeping the packed
    # index aligned with self.files.
    pool2 = AudioFileSourcePool.from_config(cfg, duration_s=1.0, sample_rate=sr)
    assert len(pool2.files) == 3
    sample = pool2.sample_array(np.random.default_rng(0), channels=2, mode="shared")
    assert sample.shape == (2, sr)


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


def test_noise_augmentation_list_blocks_apply_sequentially(tmp_path):
    noise_pool = TimeFrameNoisePool([_make_noise_tf()], min_motor_rps=0.0, duration_s=1.0)
    source_pool = _make_source_pool(tmp_path)

    def make_ds(p_fs: float, p_floor: float) -> OnlineMixIterableDataset:
        return OnlineMixIterableDataset(
            noise_pool,
            source_pool,
            policy={
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
                        "choices": [
                            {"floor_inject": {"level_low_db": -6.0, "level_high_db": -6.0}}
                        ],
                    },
                ],
            },
            base_seed=7,
            duration_s=1.0,
            sample_rate=16000,
            hop_length=512,
        )

    both = make_ds(1.0, 1.0).generate_sample(5)
    fs_only = make_ds(1.0, 1e-9).generate_sample(5)
    neither = make_ds(1e-9, 1e-9).generate_sample(5)

    # Block 2 (floor_inject) fires ON TOP of block 1's output: audio differs,
    # labels are preserved (floor_inject is label-preserving).
    assert not torch.equal(both[0], fs_only[0])
    assert torch.allclose(both[1], fs_only[1], rtol=1e-4, atol=1e-4)
    # Block 1 (freq_scale, fixed alpha 1.2) rescales the labels vs the no-fire
    # stream (same noise slice — the fire decision is drawn after sourcing).
    assert not torch.equal(fs_only[1], neither[1])
    ratio = (fs_only[1] / neither[1]).numpy()
    assert np.allclose(ratio, 1.2, atol=0.02)
