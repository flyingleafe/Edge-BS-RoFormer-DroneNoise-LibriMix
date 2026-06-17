from __future__ import annotations

from omegaconf import OmegaConf
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from data_processing.online_mixing import (
    AudioFileSourcePool,
    OnlineMixIterableDataset,
    TimeFrameNoisePool,
    _resolve_policy,
)
from utils.data import EventSeries, TimeFrame, UniformSeries


def _make_noise_tf(
    duration_s: float = 2.0,
    sr: int = 16000,
    recording_id: str = "synthetic",
) -> TimeFrame:
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.stack(
        [0.01 * np.sin(2 * np.pi * (100 + 10 * ch) * t) for ch in range(8)], axis=0
    ).astype(np.float32)
    audio_us = UniformSeries.from_samples(audio, sr, t_start=0.0)

    motor_t = np.linspace(0.0, duration_s - 0.01, 200, dtype=np.float64)
    rps = np.stack([60 + i + 0.5 * motor_t for i in range(4)], axis=0).astype(np.float32)
    rps_es = EventSeries.from_events(motor_t, rps, t_start=0.0, t_end=duration_s)
    return TimeFrame.from_tracks(
        {"audio": audio_us, "rps": rps_es},
        t_start=0.0,
        t_end=duration_s,
        tags={"recording_id": recording_id},
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

    ids = [r["tf"].tags["recording_id"] for r in pool.records]
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
    cfg = OmegaConf.load("configs/online_mix_v4_michaels_train_no_room1.yaml")
    speech = cfg.sources.speech[0]

    assert speech.root == "data/librispeech/LibriSpeech/train-clean-100"
    assert speech.glob == "**/*.flac"
    assert "datasets/DREGON-LM-V4-michaels/train" not in speech.root


def test_v4_michaels_online_config_cache_dir_is_env_configurable(monkeypatch, tmp_path):
    override = tmp_path / "online-cache"
    monkeypatch.setenv("ONLINE_MIX_SOURCE_CACHE_DIR", str(override))

    cfg = OmegaConf.load("configs/online_mix_v4_michaels_train_no_room1.yaml")
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
