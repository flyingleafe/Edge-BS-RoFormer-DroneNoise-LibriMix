"""Tests for data_processing.frame_datasets: DregonLMFrameDataset's
`flatten_channels` (C9, channel-as-extra-batch-item) and NoiseGenFrameDataset
(noise-generation Frame adapter around NoiseRPSDataset, E1-E3)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from data_processing.frame_datasets import DregonLMFrameDataset, NoiseGenFrameDataset
from data_processing.frames import get_meta
from data_processing.noise_rps_dataset import NoiseRPSDataset, _ChunkSource

SR = 16000


def _write_sample(root, name: str, *, n_channels: int, n_audio: int, rng) -> None:
    d = root / name
    d.mkdir()
    audio = (rng.standard_normal((n_audio, n_channels)) * 0.1).astype(np.float32)
    sf.write(str(d / "mixture.wav"), audio, SR)
    n_frames = n_audio // 512 + 1
    rps = rng.uniform(20.0, 100.0, size=(4, n_frames)).astype(np.float32)
    np.save(str(d / "rps.npy"), rps)


def _make_split(tmp_path, *, n_samples: int, n_channels: int, split: str = "train"):
    root = tmp_path / split
    root.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        _write_sample(root, f"sample_{i:05d}", n_channels=n_channels, n_audio=8000, rng=rng)
    return root


# ─── DregonLMFrameDataset.flatten_channels (C9) ───────────────────────────────


def test_flatten_channels_expands_index_space(tmp_path):
    root = _make_split(tmp_path, n_samples=3, n_channels=8)
    ds = DregonLMFrameDataset(root, flatten_channels=True)
    assert len(ds) == 3 * 8


def test_flatten_channels_item_is_mono_with_channel_meta(tmp_path):
    root = _make_split(tmp_path, n_samples=2, n_channels=4)
    ds = DregonLMFrameDataset(root, flatten_channels=True)

    frame0 = ds[0]
    assert frame0["mixture"].dims == ("time",)
    assert get_meta(frame0, "channel") == 0
    assert get_meta(frame0, "recording_id") == "sample_00000"

    frame3 = ds[3]  # sample_00000, channel 3 (last channel of the first recording)
    assert get_meta(frame3, "channel") == 3
    assert get_meta(frame3, "recording_id") == "sample_00000"

    frame4 = ds[4]  # sample_00001, channel 0
    assert get_meta(frame4, "channel") == 0
    assert get_meta(frame4, "recording_id") == "sample_00001"


def test_flatten_channels_broadcasts_same_rps_across_channels(tmp_path):
    root = _make_split(tmp_path, n_samples=1, n_channels=4)
    ds = DregonLMFrameDataset(root, flatten_channels=True)
    rps0 = np.asarray(ds[0]["rps"].data)
    rps2 = np.asarray(ds[2]["rps"].data)
    np.testing.assert_array_equal(rps0, rps2)


def test_flatten_channels_rejects_explicit_channel(tmp_path):
    import pytest

    root = _make_split(tmp_path, n_samples=1, n_channels=4)
    with pytest.raises(ValueError, match="incompatible with channel"):
        DregonLMFrameDataset(root, flatten_channels=True, channel=0)


def test_flatten_channels_rejects_mono_files(tmp_path):
    import pytest

    root = _make_split(tmp_path, n_samples=1, n_channels=1)
    with pytest.raises(ValueError, match="multichannel"):
        DregonLMFrameDataset(root, flatten_channels=True)


def test_default_dataset_unaffected_by_flatten_channels_flag(tmp_path):
    root = _make_split(tmp_path, n_samples=3, n_channels=8)
    ds = DregonLMFrameDataset(root, channel=0)
    assert len(ds) == 3
    assert "channel" not in ds[0]["meta"]


# ─── NoiseGenFrameDataset (E1-E3) ──────────────────────────────────────────────


def _michaels_chunk_source(*, duration_s: float = 2.0, seed: int = 0) -> _ChunkSource:
    """A tiny in-memory 'michaels'-origin _ChunkSource — no data/ files needed
    (michaels.get_geometry() is pure constants, unlike dregon's which reads
    micPos.txt/coordinates.mat)."""
    import tdseries as td

    rng = np.random.default_rng(seed)
    n = int(round(duration_s * SR))
    audio = (rng.standard_normal((8, n)) * 0.1).astype(np.float32)
    n_motor = 40
    rps = rng.uniform(20.0, 90.0, size=(4, n_motor)).astype(np.float32)
    audio_series = td.uniform(audio, SR, dims=("mic", "time"), t_start=0.0)
    rps_series = td.events(
        np.linspace(0, duration_s, n_motor, endpoint=False),
        rps,
        dims=("rotor", "time"),
        t_start=0.0,
        t_end=duration_s,
    )
    frame = td.Frame({"audio": audio_series, "rps": rps_series})
    return _ChunkSource(
        frame=frame, origin="michaels", rps_key="rps", n_channels=8, duration=duration_s
    )


def _make_noise_rps_dataset(**kwargs) -> NoiseRPSDataset:
    src = _michaels_chunk_source()
    return NoiseRPSDataset(
        [src], chunk_size=8000, sample_rate=SR, samples_per_epoch=6, seed=0, **kwargs
    )


def test_noise_gen_frame_dataset_emits_expected_entries():
    inner = _make_noise_rps_dataset()
    ds = NoiseGenFrameDataset(inner)
    assert len(ds) == 6

    frame = ds[0]
    assert set(frame) == {"audio", "rps", "mic_pos", "rotor_pos", "meta"}
    assert frame["audio"].dims == ("mic", "time")
    assert frame["audio"].data.shape == (1, 8000)  # single mic (channel_policy='first')
    assert frame["rps"].dims == ("rotor", "time")
    assert frame["rps"].data.shape[0] == 4
    assert frame["mic_pos"].data.shape == (1, 3)
    assert frame["rotor_pos"].data.shape == (4, 3)
    assert get_meta(frame, "drone") == "michaels"


def test_noise_gen_frame_dataset_rejects_random_channel_policy():
    import pytest

    inner = _make_noise_rps_dataset(channel_policy="random")
    with pytest.raises(ValueError, match="channel_policy='first'"):
        NoiseGenFrameDataset(inner)


def test_noise_gen_frame_dataset_build_train_valid_classmethods(tmp_path, monkeypatch):
    # Patch load_michaels_noise_sources so build_train/build_valid don't need
    # real data/new-drone-noises files on disk.
    import data_processing.noise_rps_dataset as nrd

    src = _michaels_chunk_source(duration_s=4.0)

    def _fake_michaels_sources(_michaels_dir, _sample_rate):
        return [src]

    monkeypatch.setattr(nrd, "load_michaels_noise_sources", _fake_michaels_sources)

    train_ds = NoiseGenFrameDataset.build_train(
        dregon_dir=None,
        michaels_dir=str(tmp_path),
        chunk_size=8000,
        train_samples=4,
        val_samples=2,
        val_pct=0.25,
    )
    valid_ds = NoiseGenFrameDataset.build_valid(
        dregon_dir=None,
        michaels_dir=str(tmp_path),
        chunk_size=8000,
        train_samples=4,
        val_samples=2,
        val_pct=0.25,
    )
    assert len(train_ds) == 4
    assert len(valid_ds) == 2
    assert get_meta(train_ds[0], "drone") == "michaels"
    assert get_meta(valid_ds[0], "drone") == "michaels"
