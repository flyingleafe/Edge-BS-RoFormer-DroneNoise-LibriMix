"""Tests for data_processing.streams — the dload ↔ tdseries bridge.

All tests run against a ``dload.remote.LocalRemote`` (directory-backed bucket
under ``tmp_path``) + a local ``ShardCache``; no network, no repo-root
``dload.toml``. Where code under test goes through
``streams.open_repository()`` (DloadFrameDataset, ensure_local,
resolve_source), the module-level repository cache is monkeypatched to the
test Repository.
"""

from __future__ import annotations

import io
from functools import partial

import dload
import numpy as np
import pytest
import soundfile as sf
import tdseries as td
import torch
from dload.cache import ShardCache
from dload.remote import LocalRemote
from dload.repo import Repository

import data_processing.streams as streams
from data_processing.collate import frame_collate
from data_processing.frames import get_meta

SR = 16000
N_CHANNELS = 2
N_SAMPLES = 3
HOP = 512
DURATION_T = SR  # 1 s per sample
N_FRAMES = DURATION_T // HOP + 1


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository(
        LocalRemote(tmp_path / "remote"),
        ShardCache(tmp_path / "cache", None),
        lock_path=tmp_path / "dload.lock",
    )


@pytest.fixture
def patched_repo(repo, monkeypatch) -> Repository:
    """Route streams.open_repository() to the tmp_path-backed Repository."""
    monkeypatch.setattr(streams, "_repository", repo)
    return repo


def _wav_bytes(audio_tc: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_tc, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _dregon_lm_samples(n: int = N_SAMPLES, channels: int = N_CHANNELS):
    """Deterministic DREGON-LM-convention samples + a `_meta` bookkeeping key."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        audio = (rng.standard_normal((DURATION_T, channels)) * 0.1).astype(np.float32)
        rps = rng.uniform(20.0, 100.0, size=(4, 29)).astype(np.float32)
        out.append(
            (
                f"sample_{i:05d}",
                {
                    "mixture": _wav_bytes(audio),
                    "rps": dload.codecs.npy_bytes(rps),
                    "meta": dload.codecs.json_bytes({"input_snr": -5.0 - i}),
                },
            )
        )
    out.append(("_meta", {"json": dload.codecs.json_bytes({"note": "bookkeeping"})}))
    return out


@pytest.fixture
def dregon_lm_dataset(repo) -> dload.Dataset:
    manifest = repo.commit(
        "DREGON-LM-TEST-train",
        _dregon_lm_samples(),
        meta={"fields": {"mixture": "wav", "rps": "npy", "meta": "json"}},
    )
    return dload.Dataset(repo, manifest)


# ─── to_frames / decode_dregon_lm ──────────────────────────────────────────────


def test_to_frames_round_trip(dregon_lm_dataset):
    pipe = streams.to_frames(
        dregon_lm_dataset.samples(),
        partial(streams.decode_dregon_lm, sample_rate=SR, hop_length=HOP),
    )
    frames = list(pipe)
    assert len(frames) == N_SAMPLES  # `_meta` dropped

    frame = frames[0]
    assert frame["mixture"].dims == ("mic", "time")
    assert frame["mixture"].shape == (N_CHANNELS, DURATION_T)
    assert float(frame["mixture"].tindex.sr) == SR
    assert frame["rps"].dims == ("rotor", "time")
    assert frame["rps"].shape == (4, N_FRAMES)
    assert frame["rps"].tindex.rate == td.GridIndex.create((SR, HOP), N_FRAMES).rate
    assert get_meta(frame, "recording_id") == "sample_00000"
    assert get_meta(frame, "input_snr") == -5.0

    # WAV FLOAT subtype -> exact float32 round trip of the audio payload.
    original = _dregon_lm_samples()[0][1]["mixture"]
    raw, _ = sf.read(io.BytesIO(original), dtype="float32", always_2d=True)
    np.testing.assert_array_equal(np.asarray(frame["mixture"].data), raw.T)


def test_decode_dregon_lm_channel_selection(dregon_lm_dataset):
    pipe = streams.to_frames(
        dregon_lm_dataset.samples(),
        partial(streams.decode_dregon_lm, sample_rate=SR, hop_length=HOP, channel=1),
    )
    frame = next(iter(pipe))
    assert frame["mixture"].dims == ("time",)
    assert frame["mixture"].shape == (DURATION_T,)


def test_decode_dregon_lm_rejects_wrong_rate(dregon_lm_dataset):
    pipe = streams.to_frames(
        dregon_lm_dataset.samples(),
        partial(streams.decode_dregon_lm, sample_rate=8000),
    )
    with pytest.raises(ValueError, match="sample_rate"):
        next(iter(pipe))


# ─── generic Frame codec (tdframe-v1) ──────────────────────────────────────────


def _rich_frame() -> td.Frame:
    """8ch audio + 4-rotor RPS at another rate + irregular telemetry + meta."""
    rng = np.random.default_rng(7)
    audio = td.uniform(
        rng.standard_normal((8, SR)).astype(np.float32), SR, dims=("mic", "time"), t_start=0.0
    )
    rps = td.uniform(
        rng.uniform(20, 100, size=(4, 929)).astype(np.float64),
        929,
        dims=("rotor", "time"),
        t_start=0.0,
    )
    stamps = np.sort(rng.uniform(0.0, 1.0, size=57))
    imu = td.events(
        stamps,
        rng.standard_normal((3, 57)).astype(np.float32),
        dims=("axis", "time"),
        t_start=0.0,
        t_end=1.0,
    )
    vad = td.spans(
        np.array([0.1, 0.6]), np.array([0.3, 0.9]), np.array([1.0, 1.0], dtype=np.float32)
    )
    mic_pos = td.wrap(rng.standard_normal((8, 3)), dims=("mic", None))
    return td.Frame(
        {
            "audio": audio,
            "motors_measured": rps,
            "imu": imu,
            "vad": vad,
            "mic_pos": mic_pos,
            "meta": td.Frame({"recording_id": "rec1", "room": "room1", "input_snr": -12.5}),
        }
    )


def test_frame_codec_round_trip_exact():
    frame = _rich_frame()
    fields = streams.frame_to_sample(frame)
    assert all(isinstance(v, bytes) for v in fields.values())
    restored = streams.sample_to_frame(fields)

    assert set(restored.keys()) == set(frame.keys())
    for key in ("audio", "motors_measured", "imu", "vad", "mic_pos"):
        orig, back = frame[key], restored[key]
        assert back.dims == orig.dims
        np.testing.assert_array_equal(np.asarray(back.data), np.asarray(orig.data))
        assert np.asarray(back.data).dtype == np.asarray(orig.data).dtype
        if orig.has_time:
            assert back.tindex.equal(orig.tindex)
        assert back.equal(orig)
    # Irregular timestamps round-trip exactly (tick-exact StampIndex fields).
    np.testing.assert_array_equal(
        restored["imu"].tindex.abs_stamps_ticks, frame["imu"].tindex.abs_stamps_ticks
    )
    for k in ("recording_id", "room", "input_snr"):
        assert restored["meta"][k] == frame["meta"][k]
    assert restored.t_start_ticks == frame.t_start_ticks
    assert restored.t_end_ticks == frame.t_end_ticks


def test_frame_codec_round_trip_after_slicing():
    """Sliced views (nonzero t_start, grid phase) must survive the codec."""
    window = _rich_frame().time[0.25:0.75]
    restored = streams.sample_to_frame(streams.frame_to_sample(window))
    assert restored["audio"].tindex.equal(window["audio"].tindex)
    assert restored["audio"].t_start == window["audio"].t_start
    np.testing.assert_array_equal(
        np.asarray(restored["audio"].data), np.asarray(window["audio"].data)
    )
    assert restored["imu"].tindex.equal(window["imu"].tindex)


def test_decode_tdframe_defaults_recording_id():
    frame = td.Frame({"audio": td.uniform(np.zeros(64, dtype=np.float32), 64, dims=("time",))})
    fields = streams.frame_to_sample(frame)
    decoded = streams.decode_tdframe(("rec42", fields))
    assert get_meta(decoded, "recording_id") == "rec42"


# ─── frame combinators ─────────────────────────────────────────────────────────


def _frame_pipe(dataset: dload.Dataset, **decoder_kwargs) -> dload.Pipeline:
    return streams.to_frames(
        dataset.samples(),
        partial(streams.decode_dregon_lm, sample_rate=SR, hop_length=HOP, **decoder_kwargs),
    )


def test_frame_windows_shapes_and_copies(dregon_lm_dataset):
    windows = list(streams.frame_windows(_frame_pipe(dregon_lm_dataset), win_s=0.25))
    assert len(windows) == N_SAMPLES * 4  # 1 s frames, non-overlapping 0.25 s
    for w in windows:
        assert w["mixture"].shape == (N_CHANNELS, SR // 4)
        data = np.asarray(w["mixture"].data)
        assert data.base is None  # copied out of the parent recording
    # Overlapping hop
    hopped = list(streams.frame_windows(_frame_pipe(dregon_lm_dataset), win_s=0.5, hop_s=0.25))
    assert len(hopped) == N_SAMPLES * 3


def test_mix_frames_at_fixed_snr(dregon_lm_dataset):
    rng = np.random.default_rng(3)
    speech_audio = rng.standard_normal(DURATION_T).astype(np.float32) * 0.05

    def speech_frames():
        while True:
            yield td.Frame(
                {
                    "mixture": td.uniform(speech_audio, SR, dims=("time",), t_start=0.0),
                    "meta": td.Frame({"recording_id": "speech0"}),
                }
            )

    noise_pipe = _frame_pipe(dregon_lm_dataset)
    mixed = next(
        iter(streams.mix_frames(dload.from_iterable(speech_frames), noise_pipe, snr_db=-10.0))
    )
    assert mixed["mixture"].shape == (N_CHANNELS, DURATION_T)
    assert get_meta(mixed, "input_snr") == -10.0
    assert get_meta(mixed, "speech_id") == "speech0"
    assert "rps" in mixed  # noise frame's aligned tracks are kept

    # mixture == noise + speech scaled to source-to-noise SNR of -10 dB.
    noise = np.asarray(next(iter(_frame_pipe(dregon_lm_dataset)))["mixture"].data)
    residual = np.asarray(mixed["mixture"].data) - noise
    snr = 10.0 * np.log10(np.mean(residual**2) / np.mean(noise**2))
    assert snr == pytest.approx(-10.0, abs=0.1)


def test_resample_frames(dregon_lm_dataset):
    frame = next(iter(streams.resample_frames(_frame_pipe(dregon_lm_dataset), 8000)))
    assert frame["mixture"].shape == (N_CHANNELS, 8000)
    assert float(frame["mixture"].tindex.sr) == 8000.0
    # RPS grid (not listed in `entries`) is untouched.
    assert frame["rps"].shape == (4, N_FRAMES)


# ─── DloadFrameDataset ─────────────────────────────────────────────────────────


def test_dload_frame_dataset_yields_collatable_frames(patched_repo, dregon_lm_dataset):
    ds = streams.DloadFrameDataset(
        "DREGON-LM-TEST-train", sample_rate=SR, hop_length=HOP, channel=0, take=2
    )
    assert ds.n_samples == N_SAMPLES + 1  # raw manifest count (incl. `_meta`)
    frames = list(ds)
    assert len(frames) == 2
    batched = frame_collate(frames)
    assert isinstance(batched["mixture"].data, torch.Tensor)
    assert tuple(batched["mixture"].data.shape) == (2, DURATION_T)
    assert tuple(batched["rps"].data.shape) == (2, 4, N_FRAMES)


def test_dload_frame_dataset_shuffle_and_repeat(patched_repo, dregon_lm_dataset):
    ds = streams.DloadFrameDataset(
        "DREGON-LM-TEST-train",
        sample_rate=SR,
        hop_length=HOP,
        channel=0,
        shuffle=7,
        shuffle_buffer=8,
        repeat=True,
    )
    it = iter(ds)
    keys = [get_meta(next(it), "recording_id") for _ in range(2 * N_SAMPLES)]
    # An infinite stream that covers the dataset across cycles.
    assert set(keys) == {f"sample_{i:05d}" for i in range(N_SAMPLES)}


def test_dload_frame_dataset_repeat_with_more_workers_than_shards(patched_repo, dregon_lm_dataset):
    """Regression: with repeat=True and num_workers > shard count (tiny
    datasets pack into one shard), the empty-stripe worker must retire
    instead of spinning forever and deadlocking the DataLoader."""
    from torch.utils.data import DataLoader

    ds = streams.DloadFrameDataset(
        "DREGON-LM-TEST-train", sample_rate=SR, hop_length=HOP, channel=0, shuffle=5, repeat=True
    )
    loader = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=frame_collate, timeout=60)
    it = iter(loader)
    seen = set()
    for _ in range(4):  # > one pass over the 3 samples: the stream repeats
        batch = next(it)
        seen.update(batch["meta"]["recording_id"])
    assert seen == {f"sample_{i:05d}" for i in range(N_SAMPLES)}


def test_dload_frame_dataset_dispatches_tdframe_layout(patched_repo):
    frame = _rich_frame()
    patched_repo.commit(
        "RICH-FRAMES-TEST",
        [("rec1", streams.frame_to_sample(frame))],
        meta={"layout": streams.TDFRAME_LAYOUT},
    )
    ds = streams.DloadFrameDataset("RICH-FRAMES-TEST")
    frames = list(ds)
    assert len(frames) == 1
    assert frames[0]["audio"].equal(frame["audio"])
    assert frames[0]["imu"].tindex.equal(frame["imu"].tindex)


# ─── ensure_local / resolve_source ─────────────────────────────────────────────


def test_ensure_local_reconstructs_raw_cli_layout(patched_repo):
    """Raw datasets (CLI convention): key = relpath minus ext, field = ext."""
    wav = _wav_bytes(np.zeros((256, 1), dtype=np.float32))
    patched_repo.commit(
        "raw-test",
        [
            ("recA/audio", {"wav": wav}),
            ("recA/motors", {"csv": b"t,rps\n0,42\n"}),
            ("README", {"data": b"extensionless"}),
        ],
    )
    root = streams.ensure_local("raw-test")
    assert (root / "recA" / "audio.wav").read_bytes() == wav
    assert (root / "recA" / "motors.csv").read_bytes() == b"t,rps\n0,42\n"
    assert (root / "README").read_bytes() == b"extensionless"
    assert (root / ".complete").exists()
    # Idempotent: second call short-circuits to the same tree.
    assert streams.ensure_local("raw-test") == root


def test_ensure_local_uses_manifest_fields_map(patched_repo, dregon_lm_dataset):
    """Processed datasets: manifest meta maps field stems to extensions."""
    root = streams.ensure_local("DREGON-LM-TEST-train")
    sample_dir = root / "sample_00000"
    assert (sample_dir / "mixture.wav").is_file()
    assert (sample_dir / "rps.npy").is_file()
    assert (sample_dir / "meta.json").is_file()
    assert not (root / "_meta").exists()  # bookkeeping samples are skipped
    rps = np.load(sample_dir / "rps.npy")
    assert rps.shape == (4, 29)


def test_resolve_source_passthrough_and_uri(patched_repo, dregon_lm_dataset):
    assert streams.resolve_source("datasets/DREGON-LM-V4/train") == streams.Path(
        "datasets/DREGON-LM-V4/train"
    )
    p = streams.Path("/abs/path")
    assert streams.resolve_source(p) is p

    resolved = streams.resolve_source("dload:DREGON-LM-TEST-train/sample_00001")
    assert resolved.is_dir()
    assert (resolved / "mixture.wav").is_file()
    root = streams.resolve_source("dload:DREGON-LM-TEST-train")
    assert resolved.parent == root

    with pytest.raises(ValueError, match="invalid dload URI"):
        streams.resolve_source("dload:")
