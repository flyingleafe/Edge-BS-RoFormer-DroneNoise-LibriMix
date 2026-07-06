"""Tests for generic ``plot_timeframe`` machinery."""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.registry import resolve_renderer_key, resolve_title


def test_mono_audio_and_rps():
    sr = 16000
    audio = td.uniform(np.random.randn(sr).astype(np.float32), sr, dims=("time",))
    rps = td.events(
        np.linspace(0, 1.0, 100, endpoint=False),
        np.random.rand(4, 100),
        dims=("rotor", "time"),
        t_start=0.0,
        t_end=1.0,
    )
    frame = td.Frame({"audio": audio, "rps": rps})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 2  # one audio row + one rps row


def test_multi_channel_audio_defaults_to_one_row_per_channel():
    sr = 16000
    audio = td.uniform(np.random.randn(4, sr).astype(np.float32), sr, dims=("mic", "time"))
    frame = td.Frame({"audio": audio})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 4


def test_channel_all_selects_every_channel():
    sr = 16000
    audio = td.uniform(np.random.randn(3, sr).astype(np.float32), sr, dims=("mic", "time"))
    frame = td.Frame({"audio": audio})
    fig = plot_timeframe(frame, channel="all")
    assert len(fig.axes) == 3


def test_prediction_track_renders_as_overlay():
    sr = 16000
    audio = td.uniform(np.random.randn(sr).astype(np.float32), sr, dims=("time",))
    pred = td.uniform(np.random.randn(4, 100).astype(np.float32), 100, dims=("rotor", "time"))
    frame = td.Frame({"audio": audio, "pred_M": pred})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 2  # audio (1 row) + pred (1 row, all rotors overlaid)


def test_unknown_track_raises():
    frame = td.Frame({"audio": td.uniform(np.zeros(100, dtype=np.float32), 16000, dims=("time",))})
    with pytest.raises(KeyError):
        plot_timeframe(frame, tracks=["not_a_track"])


def test_explicit_renderer_overrides_default_dispatch():
    audio = td.uniform(np.zeros(100, dtype=np.float32), 16000, dims=("time",))
    track = PlotTrack(series=audio, renderer="waveform")
    assert resolve_renderer_key(track) == "waveform"


def test_default_dispatch_by_index_type():
    audio = td.uniform(np.zeros(100, dtype=np.float32), 16000, dims=("time",))
    assert resolve_renderer_key(PlotTrack(series=audio)) == "audio"

    rps = td.events(
        [0.1, 0.5, 0.9], np.zeros((4, 3)), dims=("rotor", "time"), t_start=0.0, t_end=1.0
    )
    assert resolve_renderer_key(PlotTrack(series=rps)) == "rps"

    spans = td.spans([0.0, 0.3], [0.2, 0.5], t_start=0.0, t_end=1.0)
    assert resolve_renderer_key(PlotTrack(series=spans)) == "spans"


def test_explicit_title_hint_used():
    audio = td.uniform(np.zeros(100, dtype=np.float32), 16000, dims=("time",))
    track = PlotTrack(series=audio, hints={"title": "Custom Title"})
    assert resolve_title("audio", track) == "Custom Title"


def test_none_title_hint_suppresses():
    audio = td.uniform(np.zeros(100, dtype=np.float32), 16000, dims=("time",))
    track = PlotTrack(series=audio, hints={"title": None})
    assert resolve_title("audio", track) is None


def test_spans_series_renders():
    segs = td.spans([0.0, 0.3, 0.6], [0.2, 0.5, 0.9], t_start=0.0, t_end=1.0)
    frame = td.Frame({"segments": segs})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 1


def test_event_rug_renders_without_values():
    ev = td.events([0.1, 0.5, 0.9], t_start=0.0, t_end=1.0)
    frame = td.Frame({"events": ev})
    fig = plot_timeframe(frame)
    assert len(fig.axes) == 1
