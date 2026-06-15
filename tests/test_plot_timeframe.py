"""Tests for generic ``plot_timeframe`` machinery."""

from __future__ import annotations

import numpy as np
import pytest

from utils.data import EventSeries, SegmentSeries, TimeFrame, UniformSeries
from utils.plots.timeframe import plot_timeframe
from utils.plots.timeframe.registry import resolve_renderer_key, resolve_title


def test_mono_audio_and_rps():
    sr = 16000
    audio = UniformSeries.from_samples(np.random.randn(sr), sr=sr)
    rps = EventSeries.from_events(
        timestamps=np.linspace(0, 1.0, 100),
        values=np.random.rand(4, 100),
    )
    frame = TimeFrame.from_tracks({"audio": audio, "rps": rps})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 2  # one audio row + one rps row


def test_multi_channel_audio_defaults_to_one_row_per_channel():
    sr = 16000
    audio = UniformSeries.from_samples(np.random.randn(4, sr), sr=sr)
    frame = TimeFrame.from_tracks({"audio": audio})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 4


def test_channel_all_selects_every_channel():
    sr = 16000
    audio = UniformSeries.from_samples(np.random.randn(3, sr), sr=sr)
    frame = TimeFrame.from_tracks({"audio": audio})
    fig = plot_timeframe(frame, channel="all")
    assert len(fig.axes) == 3


def test_prediction_track_renders_as_overlay():
    sr = 16000
    audio = UniformSeries.from_samples(np.random.randn(sr), sr=sr)
    pred = UniformSeries.from_samples(np.random.randn(4, 100), sr=100.0, t_start=0.0)
    frame = TimeFrame.from_tracks({"audio": audio, "pred_M": pred})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 2  # audio (1 row) + pred (1 row)


def test_unknown_track_raises():
    frame = TimeFrame.from_tracks({"audio": UniformSeries.from_samples(np.zeros(100), sr=16000)})
    with pytest.raises(KeyError):
        plot_timeframe(frame, tracks=["not_a_track"])


def test_explicit_renderer_tag_overrides_name():
    audio = UniformSeries.from_samples(
        np.zeros(100),
        sr=16000,
        tags={"plot.renderer": "UniformSeries"},
    )
    frame = TimeFrame.from_tracks({"audio": audio})
    key = resolve_renderer_key("audio", frame["audio"])
    assert key == "UniformSeries"


def test_explicit_title_tag_used():
    audio = UniformSeries.from_samples(
        np.zeros(100),
        sr=16000,
        tags={"plot.title": "Custom Title"},
    )
    assert resolve_title("audio", audio) == "Custom Title"


def test_none_title_tag_suppresses():
    audio = UniformSeries.from_samples(
        np.zeros(100),
        sr=16000,
        tags={"plot.title": None},
    )
    assert resolve_title("audio", audio) is None


def test_segment_series_fallback_renders():
    segs = SegmentSeries.from_segments(
        starts=[0.0, 0.3, 0.6],
        ends=[0.2, 0.5, 0.9],
        t_start=0.0,
        t_end=1.0,
    )
    frame = TimeFrame.from_tracks({"segments": segs})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 1


def test_event_rug_renders_without_values():
    ev = EventSeries.from_events(
        timestamps=[0.1, 0.5, 0.9],
        t_start=0.0,
        t_end=1.0,
    )
    frame = TimeFrame.from_tracks({"events": ev})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 1


def test_tags_propagate_through_slice():
    audio = UniformSeries.from_samples(
        np.random.randn(16000),
        sr=16000,
        tags={"unit": "test"},
    )
    sliced = audio.slice(0, 0.5)
    assert sliced.tags == {"unit": "test"}


def test_tags_propagate_through_concat():
    a = UniformSeries.from_samples(np.random.randn(8000), sr=16000, tags={"a": 1})
    b = UniformSeries.from_samples(np.random.randn(8000), sr=16000, tags={"b": 2})
    c = a.concat(b)
    assert c.tags == {"a": 1, "b": 2}


def test_conflicting_tags_on_concat_raise():
    a = UniformSeries.from_samples(np.random.randn(8000), sr=16000, tags={"x": 1})
    b = UniformSeries.from_samples(np.random.randn(8000), sr=16000, tags={"x": 2})
    with pytest.raises(ValueError):
        a.concat(b)
