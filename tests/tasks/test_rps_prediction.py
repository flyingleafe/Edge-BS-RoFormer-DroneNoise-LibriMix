"""Tests for `tasks.rps_prediction` — load_predictor, align, evaluate error paths."""

from __future__ import annotations

import numpy as np
import pytest

from utils.data import EventSeries, TimeFrame, UniformSeries

# ── load_predictor ───────────────────────────────────────────────────────


def test_load_predictor_returns_existing_predictor_as_is():
    from tasks.rps_prediction import load_predictor

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 10))

    p = FakePredictor()
    result = load_predictor(p)
    assert result is p


def test_load_predictor_unknown_string():
    from tasks.rps_prediction import load_predictor

    with pytest.raises(ValueError, match="Unknown predictor spec"):
        load_predictor("not_a_valid_spec_xyz")


def test_load_predictor_rejects_non_string_non_predictor():
    from tasks.rps_prediction import load_predictor

    with pytest.raises(TypeError, match="expects str or RPSPredictor"):
        load_predictor(42)


# ── evaluate error paths ─────────────────────────────────────────────────


def test_evaluate_missing_audio_track():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    es = EventSeries.from_events(np.array([0.0, 1.0]), np.zeros((4, 2)), t_start=0.0, t_end=1.0)
    tf = TimeFrame.from_tracks({"rps": es})  # no "audio"
    p = FakePredictor()
    with pytest.raises(KeyError, match="audio"):
        evaluate(p, [tf], verbose=False)  # type: ignore[arg-type]


def test_evaluate_missing_rps_track():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    us = UniformSeries.from_samples(np.zeros(16000), sr=16000.0, t_start=0)
    tf = TimeFrame.from_tracks({"audio": us})  # no "rps"
    p = FakePredictor()
    with pytest.raises(KeyError, match="rps"):
        evaluate(p, [tf], verbose=False)  # type: ignore[arg-type]


def test_evaluate_audio_not_uniform_series():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    es = EventSeries.from_events(np.array([0.0]), np.array([1.0]), t_start=0.0, t_end=1.0)
    rps_es = EventSeries.from_events(np.array([0.0, 1.0]), np.zeros((4, 2)), t_start=0.0, t_end=1.0)
    tf = TimeFrame.from_tracks({"audio": es, "rps": rps_es})
    p = FakePredictor()
    with pytest.raises(TypeError, match="UniformSeries"):
        evaluate(p, [tf], verbose=False)  # type: ignore[arg-type]


def test_evaluate_rps_not_event_series():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    us = UniformSeries.from_samples(np.zeros(16000), sr=16000.0, t_start=0)
    tf = TimeFrame.from_tracks({"audio": us, "rps": us})  # rps is UniformSeries
    p = FakePredictor()
    with pytest.raises(TypeError, match="EventSeries"):
        evaluate(p, [tf], verbose=False)  # type: ignore[arg-type]


def test_evaluate_with_unknown_alignment():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    us = UniformSeries.from_samples(np.zeros(16000), sr=16000.0, t_start=0)
    es = EventSeries.from_events(np.array([0.0, 1.0]), np.zeros((4, 2)), t_start=0.0, t_end=1.0)
    tf = TimeFrame.from_tracks({"audio": us, "rps": es})
    p = FakePredictor()
    with pytest.raises(ValueError, match="Unknown alignment"):
        evaluate(p, [tf], alignment="bogus", verbose=False)  # type: ignore[arg-type]


def test_align_shape_stretch_no_values_raises():
    from tasks.rps_prediction import _align_shape_stretch

    es = EventSeries.from_events(np.array([0.0, 1.0]), values=None, t_start=0.0, t_end=1.0)
    audio = np.zeros(16000)
    with pytest.raises(ValueError, match="no values"):
        _align_shape_stretch(audio, es)
