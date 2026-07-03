"""Tests for data_processing.collate (collate_frames / frame_collate / slice_sample)."""

from __future__ import annotations

import numpy as np
import tdseries as td
import torch

from data_processing.collate import batch_size, collate_frames, frame_collate, slice_sample
from tests.training._fixtures import make_tiny_frame


def _three_frames() -> list[td.Frame]:
    return [
        make_tiny_frame(recording_id=f"r{i}", input_snr=-10.0 * i, rng=np.random.default_rng(i))
        for i in range(3)
    ]


def test_collate_frames_stacks_series_with_leading_batch_dim():
    frames = _three_frames()
    batched = collate_frames(frames)

    mixture = batched["mixture"]
    assert isinstance(mixture, td.Series)
    assert mixture.dims == ("batch", "time")
    assert mixture.dim_size("batch") == 3
    assert mixture.dim_size("time") == frames[0]["mixture"].dim_size("time")

    rps = batched["rps"]
    assert rps.dims == ("batch", "rotor", "time")
    assert rps.dim_size("batch") == 3
    assert rps.dim_size("rotor") == 4
    np.testing.assert_allclose(rps.data[1], frames[1]["rps"].data)


def test_collate_frames_meta_numeric_becomes_array_and_string_stays_list():
    frames = _three_frames()
    batched = collate_frames(frames)
    meta = batched["meta"]

    input_snr = meta["input_snr"]
    assert isinstance(input_snr, td.Series)
    assert input_snr.dims == ("batch",)
    np.testing.assert_allclose(np.asarray(input_snr.data), [0.0, -10.0, -20.0])

    recording_id = meta["recording_id"]
    assert recording_id == ["r0", "r1", "r2"]


def test_collate_frames_rejects_mismatched_keys():
    frames = _three_frames()
    dropped = frames[1].drop(["rps"])
    try:
        collate_frames([frames[0], dropped, frames[2]])
    except ValueError as exc:
        assert "identical entry keys" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched entry keys")


def test_frame_collate_produces_torch_tensors():
    frames = _three_frames()
    batched = frame_collate(frames)
    assert isinstance(batched["mixture"].data, torch.Tensor)
    assert isinstance(batched["rps"].data, torch.Tensor)


def test_slice_sample_roundtrips_collate_frames():
    frames = _three_frames()
    batched = collate_frames(frames)
    assert batch_size(batched) == 3

    for i, original in enumerate(frames):
        sample = slice_sample(batched, i)
        np.testing.assert_allclose(sample["mixture"].data, original["mixture"].data)
        np.testing.assert_allclose(sample["rps"].data, original["rps"].data)
        assert sample["meta"]["recording_id"] == original["meta"]["recording_id"]
        assert sample["meta"]["input_snr"] == original["meta"]["input_snr"]
