"""The per-source microphone filter of a ``kind: frames`` noise source.

``channels: [0]`` keeps those audio channels of every decoded chunk, BEFORE
any augmentation or mixing. The speech-lane ceiling must follow, so a mono
source gets one lane instead of the rig's eight, and a
``flatten_channels=True`` dataset then yields ONE training frame per chunk.

Fixtures (a synthetic 2-channel frames dataset in a local dload repo) live in
``tests/data_processing/conftest.py``.
"""

from __future__ import annotations

import itertools

import numpy as np

from data_processing.frame_datasets import OnlineMixFrameDataset
from data_processing.online_mixing import (
    _load_real_records,
    build_noise_stream,
    build_online_mix_pipeline,
)

from .conftest import MICHAELS_DATASET, SR

SPEC = {"kind": "frames", "dataset": MICHAELS_DATASET, "min_motor_rps": 0.0}


def _policy(channels: list[int] | None, *, silence_channels: int | None = None) -> dict:
    noise: list[dict] = [{**SPEC, **({"channels": channels} if channels else {})}]
    if silence_channels is not None:
        noise.append({"kind": "silence", "weight": 0.2, "n_channels": silence_channels})
    return {
        "sample_rate": SR,
        "duration_s": 1.0,
        "base_seed": 11,
        "sources": {"noise": noise},
        "policy": {},
    }


def test_channels_selects_mics_of_every_chunk(michaels_frames_dataset):
    """The kept channel is the SAME data the unfiltered stream has at index 0."""
    full, ceiling_full = build_noise_stream([SPEC], sample_rate=SR, window_s=1.0, seed=5)
    mono, ceiling_mono = build_noise_stream(
        [{**SPEC, "channels": [0]}], sample_rate=SR, window_s=1.0, seed=5
    )
    assert (ceiling_full, ceiling_mono) == (2, 1)

    for chunk_full, chunk_mono in itertools.islice(zip(iter(full), iter(mono)), 3):
        assert chunk_full["audio"].dim_size("mic") == 2
        assert chunk_mono["audio"].dim_size("mic") == 1
        np.testing.assert_array_equal(
            np.asarray(chunk_mono["audio"].data)[0],
            np.asarray(chunk_full["audio"].data)[0],
        )


def test_channels_accepts_a_bare_int_and_a_later_mic(michaels_frames_dataset):
    records = _load_real_records({**SPEC, "channels": 1}, sample_rate=SR, window_s=1.0)
    assert records[0]["channels"] == (1,)

    stream, ceiling = build_noise_stream(
        [{**SPEC, "channels": [1]}], sample_rate=SR, window_s=1.0, seed=5
    )
    full, _ = build_noise_stream([SPEC], sample_rate=SR, window_s=1.0, seed=5)
    assert ceiling == 1
    np.testing.assert_array_equal(
        np.asarray(next(iter(stream))["audio"].data)[0],
        np.asarray(next(iter(full))["audio"].data)[1],
    )


def test_absent_key_keeps_every_channel(michaels_frames_dataset):
    records = _load_real_records(SPEC, sample_rate=SR, window_s=1.0)
    assert records[0]["channels"] is None
    _, ceiling = build_noise_stream([SPEC], sample_rate=SR, window_s=1.0, seed=5)
    assert ceiling == 2


def test_mono_silence_arm_does_not_raise_the_lane_ceiling(michaels_frames_dataset):
    """A `kind: silence` arm sized to the source keeps the pool mono."""
    _, ceiling = build_noise_stream(
        _policy([0], silence_channels=1)["sources"]["noise"],
        sample_rate=SR,
        window_s=1.0,
        seed=5,
    )
    assert ceiling == 1
    # Left at the 8-mic default, the silence arm would force 8 speech lanes.
    _, ceiling_default = build_noise_stream(
        _policy([0], silence_channels=8)["sources"]["noise"],
        sample_rate=SR,
        window_s=1.0,
        seed=5,
    )
    assert ceiling_default == 8


def test_mono_pipeline_yields_one_frame_per_chunk(michaels_frames_dataset, tmp_path):
    pipe = build_online_mix_pipeline(_policy([0], silence_channels=1))
    frame = next(iter(pipe))
    # One microphone: the mixture Series is mono `(time,)`, with no `mic` dim.
    assert frame["mixture"].shape == (SR,)
    assert "mic" not in frame["mixture"].dims

    ds = OnlineMixFrameDataset.from_config(_policy([0], silence_channels=1), flatten_channels=True)
    frames = list(itertools.islice(iter(ds), 4))
    assert [f["mixture"].shape for f in frames] == [(SR,)] * 4
    # One frame per chunk: the sample ids advance by one, not by the mic count.
    assert [int(f["meta"]["sample_id"]) for f in frames] == [0, 1, 2, 3]
