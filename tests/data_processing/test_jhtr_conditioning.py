"""Conditioning must not perturb the fixed validation samples or their ordering.

Use the existing local recording/speech repository fixtures to exercise the real
pipeline and corruption adapter, without downloading the full campaign sets.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Any, TypedDict

import numpy as np
import pytest
import tdseries as td
from omegaconf import OmegaConf

from data_processing.frame_datasets import (
    FixedSynthFrameDataset,
    OnlineMixFrameDataset,
    SpeechPairedSynthValidDataset,
)
from data_processing.frames import get_meta, meta_dict
from data_processing.rps_corruption import RPSCorruption
from losses import PITMSELoss, RPSMSELoss

from .conftest import MICHAELS_DATASET, SPEECH_DATASET, SR


class _SynthOptions(TypedDict):
    path: str
    n: int
    base_seed: int
    duration_s: float


@pytest.fixture
def conditioning_policy(tmp_path, michaels_frames_dataset, speech_dataset):
    cfg = OmegaConf.create(
        {
            "sample_rate": SR,
            "duration_s": 1.0,
            "base_seed": 880101,
            "sources": {
                "noise": [{"kind": "frames", "dataset": MICHAELS_DATASET}],
                "speech": [{"dataset": SPEECH_DATASET}],
            },
            "policy": {
                "stages": [
                    {
                        "until": None,
                        "source_prob": 1.0,
                        "snr_db": -10.0,
                        "speech_per_channel": "shared",
                    }
                ]
            },
        }
    )
    path = tmp_path / "conditioning.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def _same_series(left, right):
    assert left.dims == right.dims
    np.testing.assert_array_equal(left.data, right.data)
    np.testing.assert_array_equal(left.tindex.sample_times(), right.tindex.sample_times())


def _same_sample(left, right, *, aligned_targets=False):
    _same_series(left["mixture"], right["mixture"])
    assert meta_dict(left) == meta_dict(right)
    if not aligned_targets:
        _same_series(left["rps"], right["rps"])
    else:
        np.testing.assert_array_equal(
            left["rps"].tindex.sample_times(), right["rps"].tindex.sample_times()
        )
        # Whole rows may move; neither the target values nor their time order may.
        assert sorted(map(tuple, left["rps"].data)) == sorted(map(tuple, right["rps"].data))


@pytest.mark.parametrize("paired", [False, True])
def test_default_none_keeps_unconditioned_samples(conditioning_policy: str, paired: bool):
    cls = SpeechPairedSynthValidDataset if paired else FixedSynthFrameDataset
    options = _SynthOptions(path=conditioning_policy, n=8, base_seed=880101, duration_s=1.0)
    baseline = cls(**options)
    explicit_none = cls(**options, rps_corruption=None)
    assert len(baseline) == len(explicit_none) == 8
    for i in range(len(baseline)):
        left, right = baseline[i], explicit_none[i]
        assert "rps_cond" not in left and "rps_cond" not in right
        _same_sample(left, right)
    # Check against the pre-existing stream, not merely two adapter invocations.
    stream = OnlineMixFrameDataset.from_yaml(
        conditioning_policy, flatten_channels=True, speech=False if paired else None
    )
    original = list(itertools.islice(stream, 4 if paired else 8))
    for i, frame in enumerate(original):
        _same_sample(baseline[i], frame)


def test_fixed_corruption_preserves_samples_and_aligns_swaps(conditioning_policy: str):
    options = _SynthOptions(path=conditioning_policy, n=16, base_seed=880101, duration_s=1.0)
    # Force pair events solely to cover both alignment branches in this unit test;
    # campaign configs retain the established default probability and seeds.
    corruption: dict[str, Any] = {"seed": 777, "pair_event_prob": 1.0}
    baseline = FixedSynthFrameDataset(**options)
    conditioned = FixedSynthFrameDataset(**options, rps_corruption=corruption)
    repeated = FixedSynthFrameDataset(**options, rps_corruption=corruption)
    sampler = RPSCorruption(**corruption)
    saw_swap = False
    assert len(baseline) == len(conditioned) == len(repeated) == 16
    for i in range(len(baseline)):
        original, current, again = baseline[i], conditioned[i], repeated[i]
        _same_sample(original, current, aligned_targets=True)
        _same_sample(current, again)
        sample_id = int(get_meta(original, "sample_id"))
        channel = int(get_meta(original, "channel"))
        expected_cond, expected_target = sampler(original["rps"].data, sample_id * 256 + channel)
        np.testing.assert_array_equal(current["rps_cond"].data, expected_cond)
        np.testing.assert_array_equal(current["rps"].data, expected_target)
        _same_series(current["rps_cond"], again["rps_cond"])
        saw_swap |= not np.array_equal(original["rps"].data, current["rps"].data)
    assert saw_swap, "fixture must exercise the non-identity target-alignment branch"


def test_paired_speech_corruptions_are_identical_and_repeatable(conditioning_policy: str):
    options = _SynthOptions(path=conditioning_policy, n=16, base_seed=880101, duration_s=1.0)
    baseline = SpeechPairedSynthValidDataset(**options)
    conditioned = SpeechPairedSynthValidDataset(**options, rps_corruption={"seed": 777})
    repeated = SpeechPairedSynthValidDataset(**options, rps_corruption={"seed": 777})
    assert len(conditioned) == len(baseline) == len(repeated) == 16
    for i in range(len(conditioned.no_speech)):
        quiet = conditioned.no_speech[i]
        speech = conditioned.with_speech[i]
        _same_series(quiet["rps_cond"], speech["rps_cond"])
        _same_series(quiet["rps"], speech["rps"])
        assert not np.array_equal(quiet["mixture"].data, speech["mixture"].data)
    for i in range(len(conditioned)):
        _same_sample(baseline[i], conditioned[i], aligned_targets=True)
        _same_sample(conditioned[i], repeated[i])
        _same_series(conditioned[i]["rps_cond"], repeated[i]["rps_cond"])


def test_conditional_loss_penalizes_row_swap_while_pit_matches():
    rates = np.tile(np.array([0.0, 20.0, 60.0, 100.0], dtype=np.float32)[:, None], (1, 32))
    swapped = rates[[1, 0, 2, 3]]
    frame_rate = Fraction(SR, 512)
    target = td.Frame({"rps": td.uniform(rates, frame_rate, dims=("rotor", "time"))})
    pred = td.Frame({"rps_pred": td.uniform(swapped, frame_rate, dims=("rotor", "time"))})
    ordered = RPSMSELoss(rate=(SR, 512))(pred, target)
    pit = PITMSELoss(rate=(SR, 512))(pred, target)
    assert ordered.item() == pytest.approx(200.0)
    assert pit.item() == pytest.approx(0.0, abs=1e-6)
