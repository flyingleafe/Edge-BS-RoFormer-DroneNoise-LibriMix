"""Shared fixtures for the data_processing stream tests: synthetic dload
datasets (recording frames + raw speech audio) committed to a local
file-backed repository, injected as the process-wide streams repository."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import tdseries as td
from dload.cache import ShardCache
from dload.remote import LocalRemote
from dload.repo import Repository

import data_processing.streams as streams
from data_processing.frames import make_recording_frame

SR = 16000
MICHAELS_SR = 32000  # published at a different rate -> exercises resampling
DUR_S = 4.0
N_MOTOR = 200

DREGON_DATASET = "DREGON-FRAMES-TEST"
MICHAELS_DATASET = "MICHAELS-FRAMES-TEST"
SPEECH_DATASET = "SPEECH-TEST"


@pytest.fixture
def patched_repo(tmp_path, monkeypatch) -> Repository:
    repo = Repository(
        LocalRemote(tmp_path / "remote"),
        ShardCache(tmp_path / "cache", None),
        lock_path=tmp_path / "dload.lock",
    )
    monkeypatch.setattr(streams, "_repository", repo)
    return repo


def _audio_series(sr: int, seed: int, channels: int = 2, dur_s: float = DUR_S) -> td.Series:
    rng = np.random.default_rng(seed)
    data = (0.05 * rng.standard_normal((channels, int(dur_s * sr)))).astype(np.float32)
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


def _motor_series(values: np.ndarray, dur_s: float = DUR_S) -> td.Series:
    t = np.linspace(0.0, dur_s - 0.01, N_MOTOR)
    return td.events(t, values, dims=("rotor", "time"), t_start=0.0, t_end=dur_s)


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


def _wav_bytes(seed: int, dur_s: float = 2.0, sr: int = SR) -> bytes:
    rng = np.random.default_rng(seed)
    x = (0.1 * rng.standard_normal(int(dur_s * sr))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def speech_dataset(patched_repo):
    """A raw wav-per-sample speech dataset in the librispeech key layout."""
    samples = []
    for spk in ("19", "103", "200"):
        for i in range(4):
            key = f"LibriSpeech/train-clean-100/{spk}/chap/{spk}-chap-{i:04d}"
            samples.append((key, {"wav": _wav_bytes(hash((spk, i)) % 10000)}))
    patched_repo.commit(SPEECH_DATASET, samples, meta={})
