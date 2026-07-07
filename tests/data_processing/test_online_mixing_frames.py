"""`kind: frames` noise sources — published tdframe-v1 datasets feeding the pools.

Runs against a ``dload.remote.LocalRemote`` (directory-backed bucket under
``tmp_path``), same setup as ``test_streams.py``. Verifies that
``TimeFrameNoisePool.from_config`` / ``build_noise_rps_datasets`` consume the
published rich frames directly: the ``meta.split`` filter, entry subsetting,
byte-for-byte pass-through of the already-fixed rotor track (no re-applied
``clean_command_spikes``), audio resampling, ``take``/exclude filters, and the
end-to-end online-mix sample.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td
from dload.cache import ShardCache
from dload.remote import LocalRemote
from dload.repo import Repository

import data_processing.streams as streams
from data_processing.dregon import clean_command_spikes
from data_processing.frames import get_meta, make_recording_frame
from data_processing.noise_rps_dataset import build_noise_rps_datasets
from data_processing.online_mixing import OnlineMixIterableDataset, TimeFrameNoisePool

SR = 16000
MICHAELS_SR = 32000  # published at a different rate -> exercises resampling
DUR_S = 4.0
N_MOTOR = 200

DREGON_DATASET = "DREGON-FRAMES-TEST"
MICHAELS_DATASET = "MICHAELS-FRAMES-TEST"


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_repo(tmp_path, monkeypatch) -> Repository:
    repo = Repository(
        LocalRemote(tmp_path / "remote"),
        ShardCache(tmp_path / "cache", None),
        lock_path=tmp_path / "dload.lock",
    )
    monkeypatch.setattr(streams, "_repository", repo)
    return repo


def _audio_series(sr: int, seed: int) -> td.Series:
    rng = np.random.default_rng(seed)
    data = (0.05 * rng.standard_normal((2, int(DUR_S * sr)))).astype(np.float32)
    return td.uniform(data, sr, dims=("mic", "time"), t_start=0.0)


def _motor_values(base: float) -> np.ndarray:
    """Per-rotor constants ``base + rotor`` plus one impulsive spike.

    The spike is the fingerprint: a (re-)applied ``clean_command_spikes``
    median filter would remove it, so its survival proves the published values
    pass through untouched.
    """
    values = np.tile(np.arange(4, dtype=np.float32)[:, None] + base, (1, N_MOTOR))
    values[0, N_MOTOR // 2] = 200.0
    return values


def _motor_series(values: np.ndarray) -> td.Series:
    t = np.linspace(0.0, DUR_S - 0.01, N_MOTOR)
    return td.events(t, values, dims=("rotor", "time"), t_start=0.0, t_end=DUR_S)


@pytest.fixture
def dregon_command_values() -> np.ndarray:
    return _motor_values(60.0)


@pytest.fixture
def dregon_frames_dataset(patched_repo, dregon_command_values):
    """DREGON-frames-style dataset: cleaned command + raw + measured + splits."""

    def rec(rid: str, split: str, *, seed: int, measured: bool) -> td.Frame:
        tracks: dict[str, td.Series] = {
            "audio": _audio_series(SR, seed),
            "motors_command": _motor_series(dregon_command_values),
            "motors_command_raw": _motor_series(dregon_command_values + 100.0),
        }
        if measured:
            tracks["motors_measured"] = _motor_series(_motor_values(55.0))
        return make_recording_frame(tracks, meta={"recording_id": rid, "split": split})

    patched_repo.commit(
        DREGON_DATASET,
        [
            (
                "rec_flight_a",
                streams.frame_to_sample(
                    rec("rec_flight_a", "in_flight_noise", seed=0, measured=True)
                ),
            ),
            (
                "rec_flight_b",
                streams.frame_to_sample(
                    rec("rec_flight_b", "in_flight_noise", seed=1, measured=False)
                ),
            ),
            (
                "rec_motor",
                streams.frame_to_sample(rec("rec_motor", "motor", seed=2, measured=False)),
            ),
        ],
        meta={"layout": streams.TDFRAME_LAYOUT},
    )


@pytest.fixture
def michaels_rps_values() -> np.ndarray:
    return np.tile(np.arange(4, dtype=np.float32)[:, None] + 60.0, (1, N_MOTOR))


@pytest.fixture
def michaels_frames_dataset(patched_repo, michaels_rps_values):
    """michaels-frames-style dataset: 32 kHz audio + already-aligned `rps`."""
    frame = make_recording_frame(
        {"audio": _audio_series(MICHAELS_SR, seed=3), "rps": _motor_series(michaels_rps_values)},
        meta={"recording_id": "FLY124"},
    )
    patched_repo.commit(
        MICHAELS_DATASET,
        [("FLY124", streams.frame_to_sample(frame))],
        meta={"layout": streams.TDFRAME_LAYOUT},
    )


# ─── TimeFrameNoisePool `kind: frames` ─────────────────────────────────────────


def test_frames_kind_split_filter_subsetting_and_no_recleaning(
    dregon_frames_dataset, dregon_command_values
):
    pool = TimeFrameNoisePool.from_config(
        {"kind": "frames", "dataset": DREGON_DATASET, "splits": ["in_flight_noise"]},
        duration_s=1.0,
        sample_rate=SR,
    )

    ids = sorted(get_meta(r["tf"], "recording_id") for r in pool.records)
    assert ids == ["rec_flight_a", "rec_flight_b"]  # `motor` split filtered out

    tf = next(r["tf"] for r in pool.records if get_meta(r["tf"], "recording_id") == "rec_flight_a")
    # Only what the pool slices survives; imu/raw/measured/geometry are dropped.
    assert set(tf.keys()) == {"audio", "rps", "meta"}
    # The rotor track is the published (already-fixed) motors_command,
    # byte-for-byte — NOT the raw track, and NOT re-cleaned.
    np.testing.assert_array_equal(np.asarray(tf["rps"].data), dregon_command_values)
    # Guard: a fresh clean_command_spikes pass would have altered these values,
    # so exact equality above genuinely detects double-cleaning.
    assert not np.array_equal(clean_command_spikes(dregon_command_values), dregon_command_values)


def test_frames_kind_take_and_exclude_and_ids(dregon_frames_dataset):
    base = {"kind": "frames", "dataset": DREGON_DATASET, "splits": ["in_flight_noise"]}

    pool = TimeFrameNoisePool.from_config({**base, "take": 1}, duration_s=1.0, sample_rate=SR)
    assert len(pool.records) == 1

    pool = TimeFrameNoisePool.from_config(
        {**base, "exclude_recording_ids": ["rec_flight_a"]}, duration_s=1.0, sample_rate=SR
    )
    assert [get_meta(r["tf"], "recording_id") for r in pool.records] == ["rec_flight_b"]

    pool = TimeFrameNoisePool.from_config(
        {"kind": "frames", "dataset": DREGON_DATASET, "recording_ids": ["rec_flight_a"]},
        duration_s=1.0,
        sample_rate=SR,
    )
    assert [get_meta(r["tf"], "recording_id") for r in pool.records] == ["rec_flight_a"]


def test_frames_kind_resamples_audio_and_passes_rps_through(
    michaels_frames_dataset, michaels_rps_values
):
    pool = TimeFrameNoisePool.from_config(
        {"kind": "frames", "dataset": MICHAELS_DATASET},
        duration_s=1.0,
        sample_rate=SR,
    )

    tf = pool.records[0]["tf"]
    assert float(tf["audio"].tindex.sr) == SR  # 32 kHz publish -> 16 kHz pool
    assert tf["audio"].dim_size("time") == int(DUR_S * SR)
    np.testing.assert_array_equal(np.asarray(tf["rps"].data), michaels_rps_values)


def test_frames_kind_rejects_non_tdframe_dataset(patched_repo):
    patched_repo.commit("NOT-FRAMES", [("a", {"data": b"x"})])
    with pytest.raises(ValueError, match="tdframe-v1"):
        TimeFrameNoisePool.from_config(
            {"kind": "frames", "dataset": "NOT-FRAMES"}, duration_s=1.0, sample_rate=SR
        )


def test_frames_kind_end_to_end_online_mix_sample(michaels_frames_dataset, michaels_rps_values):
    pool = TimeFrameNoisePool.from_config(
        {"kind": "frames", "dataset": MICHAELS_DATASET},
        duration_s=1.0,
        sample_rate=SR,
    )
    ds = OnlineMixIterableDataset(
        pool, None, policy={}, base_seed=7, duration_s=1.0, sample_rate=SR, hop_length=512
    )

    audio, rps = ds.generate_sample(0)

    assert audio.shape == (2, SR)
    assert rps.shape == (4, SR // 512 + 1)
    # Constant per-rotor published speeds come back exactly (no cleaning /
    # distortion in the interpolation path).
    expected = np.arange(4, dtype=np.float32) + 60.0
    np.testing.assert_allclose(rps.numpy(), np.tile(expected[:, None], (1, rps.shape[1])))


# ─── build_noise_rps_datasets `frames:` specs ──────────────────────────────────


def test_build_noise_rps_datasets_from_frames_specs(dregon_frames_dataset, michaels_frames_dataset):
    train_ds, val_ds = build_noise_rps_datasets(
        dregon_dir=f"frames:{DREGON_DATASET}",
        michaels_dir=f"frames:{MICHAELS_DATASET}",
        sample_rate=SR,
        chunk_size=SR // 2,
        train_samples=4,
        val_samples=2,
        val_pct=0.25,
        seed=0,
    )

    # DREGON side mirrors the folder loader: only recordings with
    # motors_measured qualify (rec_flight_a), plus the one michaels recording.
    assert {src.origin for src in train_ds.records} == {"dregon", "michaels"}
    assert len(train_ds.records) == 2
    dregon_src = next(s for s in train_ds.records if s.origin == "dregon")
    assert dregon_src.rps_key == "motors_measured"

    item = train_ds[0]
    assert item["audio"].shape == (SR // 2,)
    assert item["rps"].shape == (4, SR // 2)
    assert len(val_ds) == 2
