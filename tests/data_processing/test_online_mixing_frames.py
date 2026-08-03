"""Tests for the online-mix noise stream over published ``tdframe-v1`` frames
datasets (``kind: frames``) — the *only* real-recording source kind.

The published frames carry their fixes baked in (DREGON ``motors_command`` is
``clean_command_spikes``-cleaned at derivation time), so the stream must NOT
re-clean; the loader reduces each recording to the canonical
``(audio, rps, meta)`` noise frame (:func:`frames.adapt_recording_frame`).

Fixtures (synthetic datasets in a local dload repo) live in
``tests/data_processing/conftest.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import data_processing.streams as streams
from data_processing.frames import get_meta
from data_processing.online_mixing import (
    _load_real_records,
    build_noise_stream,
    build_online_mix_pipeline,
)
from data_processing.sources.dregon import clean_command_spikes

from .conftest import DREGON_DATASET, DUR_S, MICHAELS_DATASET, SR


def test_frames_kind_split_filter_subsetting_and_no_recleaning(
    dregon_frames_dataset, dregon_command_values
):
    records = _load_real_records(
        {"kind": "frames", "dataset": DREGON_DATASET, "splits": ["in_flight_noise"]},
        sample_rate=SR,
        window_s=1.0,
    )

    ids = sorted(get_meta(r["tf"], "recording_id") for r in records)
    assert ids == ["rec_flight_a", "rec_flight_b"]  # `motor` split filtered out

    tf = next(r["tf"] for r in records if get_meta(r["tf"], "recording_id") == "rec_flight_a")
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

    records = _load_real_records({**base, "take": 1}, sample_rate=SR, window_s=1.0)
    assert len(records) == 1

    records = _load_real_records(
        {**base, "exclude_recording_ids": ["rec_flight_a"]}, sample_rate=SR, window_s=1.0
    )
    assert [get_meta(r["tf"], "recording_id") for r in records] == ["rec_flight_b"]

    records = _load_real_records(
        {"kind": "frames", "dataset": DREGON_DATASET, "recording_ids": ["rec_flight_a"]},
        sample_rate=SR,
        window_s=1.0,
    )
    assert [get_meta(r["tf"], "recording_id") for r in records] == ["rec_flight_a"]


def test_frames_kind_resamples_audio_and_passes_rps_through(
    michaels_frames_dataset, michaels_rps_values
):
    records = _load_real_records(
        {"kind": "frames", "dataset": MICHAELS_DATASET}, sample_rate=SR, window_s=1.0
    )

    tf = records[0]["tf"]
    assert float(tf["audio"].tindex.sr) == SR  # 32 kHz publish -> 16 kHz pool
    assert tf["audio"].dim_size("time") == int(DUR_S * SR)
    np.testing.assert_array_equal(np.asarray(tf["rps"].data), michaels_rps_values)


def test_frames_kind_rejects_non_tdframe_dataset(patched_repo):
    patched_repo.commit("NOT-FRAMES", [("a", {"data": b"x"})])
    with pytest.raises(ValueError, match="tdframe-v1"):
        _load_real_records(
            {"kind": "frames", "dataset": "NOT-FRAMES"}, sample_rate=SR, window_s=1.0
        )


def test_frames_kind_min_motor_rps_window(michaels_frames_dataset):
    # Constant 60..63 RPS telemetry: a 70 RPS floor leaves no valid window.
    with pytest.raises(ValueError, match="no usable noise recordings"):
        _load_real_records(
            {"kind": "frames", "dataset": MICHAELS_DATASET, "min_motor_rps": 70.0},
            sample_rate=SR,
            window_s=1.0,
        )


def test_frames_kind_end_to_end_online_mix_sample(michaels_frames_dataset, michaels_rps_values):
    pipe = build_online_mix_pipeline(
        {
            "sample_rate": SR,
            "duration_s": 1.0,
            "base_seed": 7,
            "sources": {
                "noise": [{"kind": "frames", "dataset": MICHAELS_DATASET, "min_motor_rps": 0.0}]
            },
            "policy": {},
        }
    )
    frame = next(iter(pipe))

    assert frame["mixture"].dims == ("mic", "time")
    assert frame["mixture"].shape == (2, SR)
    assert frame["rps"].shape == (4, SR // 512 + 1)
    # Constant per-rotor published speeds come back exactly (no cleaning /
    # distortion in the interpolation path).
    expected = np.arange(4, dtype=np.float32) + 60.0
    np.testing.assert_allclose(
        np.asarray(frame["rps"].data), np.tile(expected[:, None], (1, frame["rps"].shape[1]))
    )
    assert get_meta(frame, "sample_id") == 0


def test_noise_stream_is_deterministic_and_varies(michaels_frames_dataset):
    spec = [{"kind": "frames", "dataset": MICHAELS_DATASET, "min_motor_rps": 0.0}]
    s1, _ = build_noise_stream(spec, sample_rate=SR, window_s=1.0, seed=5)
    s2, _ = build_noise_stream(spec, sample_rate=SR, window_s=1.0, seed=5)
    a = list(itertools.islice(iter(s1), 4))
    b = list(itertools.islice(iter(s2), 4))
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(np.asarray(fa["audio"].data), np.asarray(fb["audio"].data))
    # successive windows differ (random starts over a 4 s recording)
    assert not np.array_equal(np.asarray(a[0]["audio"].data), np.asarray(a[1]["audio"].data))
