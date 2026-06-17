from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from data_processing.online_mixing import (
    AudioFileSourcePool,
    OnlineMixIterableDataset,
    TimeFrameNoisePool,
)
from utils.data import EventSeries, TimeFrame, UniformSeries


def _make_noise_tf(duration_s: float = 2.0, sr: int = 16000) -> TimeFrame:
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
        tags={"recording_id": "synthetic"},
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
