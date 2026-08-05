"""Tests for the FrameSpec structural typing layer (framespec)."""

import numpy as np
import tdseries as td

from framespec import (
    SCALAR,
    FrameSpec,
    SeriesSpec,
    check_subsumes,
    merge_specs,
    spec_of,
    without_batch,
)
from tasks.task import rps_prediction, speech_enhancement


def _sample_frame() -> td.Frame:
    return td.Frame(
        {
            "mixture": td.uniform(
                np.zeros((8, 16000), dtype=np.float32), 16000, dims=("mic", "time")
            ),
            "rps": td.events(
                np.linspace(0.0, 1.0, 30),
                np.zeros((4, 30), dtype=np.float64),
                dims=("rotor", "time"),
            ),
            "mic_pos": td.wrap(np.zeros((8, 3)), dims=("mic", None)),
            "meta": td.Frame({"recording_id": "r1", "input_snr": -15.0}),
        }
    )


def test_spec_of_infers_kinds_and_rates():
    spec = spec_of(_sample_frame())
    mix = spec.entries["mixture"]
    assert isinstance(mix, SeriesSpec)
    assert mix.dims == ("mic", "time")
    assert mix.time == "grid"
    assert mix.rate == (16000, 1)
    assert mix.dtype == "float32"
    rps = spec.entries["rps"]
    assert isinstance(rps, SeriesSpec)
    assert rps.time == "stamps"
    pos = spec.entries["mic_pos"]
    assert isinstance(pos, SeriesSpec)
    assert pos.time is None
    assert isinstance(spec.entries["meta"], FrameSpec)


def test_subsumes_accepts_matching_and_extra_entries():
    provided = spec_of(_sample_frame())
    required = FrameSpec(
        {
            "mixture": SeriesSpec(dims=("mic", "time"), time="grid", rate=(16000, 1)),
            "meta": FrameSpec({"recording_id": SCALAR}),
        }
    )
    assert check_subsumes(provided, required) == []


def test_subsumes_reports_missing_and_mismatched():
    provided = spec_of(_sample_frame())
    required = FrameSpec(
        {
            "enhanced": SeriesSpec(dims=("time",)),
            "mixture": SeriesSpec(dims=("mic", "time"), time="grid", rate=(44100, 1)),
        }
    )
    problems = check_subsumes(provided, required)
    assert any("enhanced" in p and "missing" in p for p in problems)
    assert any("rate" in p for p in problems)


def test_optional_required_entries_are_skipped_when_absent():
    provided = FrameSpec({"mixture": SeriesSpec(dims=("time",))})
    required = FrameSpec(
        {
            "mixture": SeriesSpec(dims=("time",)),
            "rps": SeriesSpec(dims=("rotor", "time")),
        },
        optional=frozenset({"rps"}),
    )
    assert check_subsumes(provided, required) == []


def test_anonymous_required_dim_matches_any_name():
    provided = FrameSpec({"x": SeriesSpec(dims=("mic", None), time=None)})
    required = FrameSpec({"x": SeriesSpec(dims=(None, None), time=None)})
    assert check_subsumes(provided, required) == []
    # but a named requirement does not match a differently-named dim
    required2 = FrameSpec({"x": SeriesSpec(dims=("rotor", None), time=None)})
    assert check_subsumes(provided, required2) != []


def test_without_batch_strips_leading_batch_dim():
    task = rps_prediction(n_channels=8, frame_rate=(16000, 512))
    per_sample = without_batch(task.input_spec)
    mix = per_sample.entries["mixture"]
    assert isinstance(mix, SeriesSpec)
    assert mix.dims == ("mic", "time")


def test_task_pipeline_check_end_to_end():
    """Dataset spec covers model input; output+data covers a loss spec."""
    dataset_spec = spec_of(_sample_frame())
    task = rps_prediction(n_channels=8)
    assert check_subsumes(dataset_spec, without_batch(task.input_spec)) == []

    loss_requires_pred = FrameSpec({"rps_pred": SeriesSpec(dims=("batch", "rotor", "time"))})
    available = merge_specs(task.output_spec)
    assert check_subsumes(available, loss_requires_pred) == []


def test_speech_enhancement_optional_outputs():
    task = speech_enhancement(n_channels=None, use_rps=True, predict_rps=True)
    assert "rps" in task.input_spec.entries
    assert "rps_pred" in task.output_spec.entries
    mix = task.input_spec.entries["mixture"]
    assert isinstance(mix, SeriesSpec)
    assert mix.dims == ("batch", "time")
