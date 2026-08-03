"""Tests for data_processing.frame_datasets: DregonLMFrameDataset's
`flatten_channels` (C9, channel-as-extra-batch-item) and NoiseGenFrameDataset
(noise-generation Frame adapter around NoiseRPSDataset, E1-E3)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from data_processing.frame_datasets import (
    DregonLMFrameDataset,
    NoiseGenFrameDataset,
    OnlineMixFrameDataset,
)
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


# ─── OnlineMixFrameDataset.flatten_channels ────────────────────────────────────


_published_by_repo: dict[int, set[str]] = {}


def _publish_synthetic_recording(repo, *, channels: int, dataset_name: str):
    key = id(repo)
    rset = _published_by_repo.setdefault(key, set())
    if dataset_name in rset:
        return
    rset.add(dataset_name)
    """Publish a one-recording synthetic frames dataset (deterministic seed)."""
    import tdseries as td

    import data_processing.streams as streams

    rng = np.random.default_rng(0)  # fixed seed: the same audio every call
    dur_s = 3.0
    n = int(dur_s * SR)
    if channels == 1:
        audio = td.uniform(
            (0.1 * rng.standard_normal(n)).astype(np.float32), SR, dims=("time",), t_start=0.0
        )
    else:
        audio = td.uniform(
            (0.1 * rng.standard_normal((channels, n))).astype(np.float32),
            SR,
            dims=("mic", "time"),
            t_start=0.0,
        )
    n_motor = 60
    rps = td.events(
        np.linspace(0.0, dur_s, n_motor, endpoint=False),
        np.tile(np.arange(4, dtype=np.float32)[:, None] + 50.0, (1, n_motor)),
        dims=("rotor", "time"),
        t_start=0.0,
        t_end=dur_s,
    )
    rid = f"rec_{channels}ch"
    frame = td.Frame({"audio": audio, "rps": rps, "meta": td.Frame({"recording_id": rid})})
    repo.commit(
        dataset_name,
        [(rid, streams.frame_to_sample(frame))],
        meta={"layout": streams.TDFRAME_LAYOUT},
    )


def _online_mix_frame_dataset(
    repo, *, channels: int, flatten_channels: bool
) -> OnlineMixFrameDataset:
    """Build an OnlineMixFrameDataset over a previously published frames dataset."""
    dataset = f"SYNTH-FLAT-{channels}CH"
    _publish_synthetic_recording(repo, channels=channels, dataset_name=dataset)
    cfg = {
        "sample_rate": SR,
        "duration_s": 1.0,
        "base_seed": 11,
        "sources": {
            "noise": [{"kind": "frames", "dataset": dataset, "min_motor_rps": 0.0}],
        },
        "policy": {"source_prob": 0.0},
    }
    return OnlineMixFrameDataset.from_config(cfg, flatten_channels=flatten_channels)


def test_online_mix_flatten_yields_c_mono_frames_per_chunk(patched_repo):
    import itertools

    ds = _online_mix_frame_dataset(patched_repo, channels=4, flatten_channels=True)
    frames = list(itertools.islice(iter(ds), 8))  # 2 chunks x 4 mics

    for i, frame in enumerate(frames):
        assert frame["mixture"].dims == ("time",)
        assert get_meta(frame, "channel") == i % 4

    # All 4 mono views of one chunk broadcast the identical rps target.
    rps_first = np.asarray(frames[0]["rps"].data)
    for frame in frames[1:4]:
        np.testing.assert_array_equal(np.asarray(frame["rps"].data), rps_first)

    # The mono views come from different mics (audio differs across channels).
    assert not np.array_equal(
        np.asarray(frames[0]["mixture"].data), np.asarray(frames[1]["mixture"].data)
    )


def test_online_mix_flatten_matches_unflattened_channels(patched_repo):
    """The flatten function, applied to a raw frame, yields the per-channel
    mono views as separate frames (audio and rps byte-identical)."""
    import itertools

    from data_processing.frame_datasets import _flatten_frame_channels

    raw = _online_mix_frame_dataset(patched_repo, channels=4, flatten_channels=False)
    (raw_frame,) = list(itertools.islice(iter(raw), 1))

    raw_audio = np.asarray(raw_frame["mixture"].data)  # (4, T)
    assert raw_frame["mixture"].dims == ("mic", "time")
    flat_frames = _flatten_frame_channels(raw_frame)
    for ch, frame in enumerate(flat_frames):
        np.testing.assert_array_equal(np.asarray(frame["mixture"].data), raw_audio[ch])
        np.testing.assert_array_equal(
            np.asarray(frame["rps"].data), np.asarray(raw_frame["rps"].data)
        )


def test_online_mix_default_false_is_unchanged_multichannel_stream(patched_repo):
    import itertools

    ds = _online_mix_frame_dataset(patched_repo, channels=4, flatten_channels=False)
    frames = list(itertools.islice(iter(ds), 2))
    for frame in frames:
        assert frame["mixture"].dims == ("mic", "time")
        assert frame["mixture"].data.shape == (4, SR)
        assert "channel" not in frame["meta"]


def test_online_mix_flatten_mono_passthrough(patched_repo):
    import itertools

    ds = _online_mix_frame_dataset(patched_repo, channels=1, flatten_channels=True)
    frames = list(itertools.islice(iter(ds), 2))
    for frame in frames:
        assert frame["mixture"].dims == ("time",)
        assert "channel" not in frame["meta"]


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
