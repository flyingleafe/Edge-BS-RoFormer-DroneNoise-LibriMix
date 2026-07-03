"""Tests for `tasks.rps_prediction` — load_predictor, align, evaluate error paths."""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

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


# ── align_rps_to_gt ──────────────────────────────────────────────────────


def test_align_rps_to_gt_rejects_transposed_input():
    from tasks.rps_prediction import align_rps_to_gt

    # A transposed (F, R) array reads as R = 100 rotors — must fail fast
    # instead of materializing a huge pairwise cost over frames.
    pred = np.random.rand(100, 4)
    gt = np.random.rand(100, 4)
    with pytest.raises(ValueError, match="rotor axis first"):
        align_rps_to_gt(pred, gt)


def test_align_rps_to_gt_permutes_rows_back():
    from tasks.rps_prediction import align_rps_to_gt

    gt = np.stack([np.full(32, 10.0 * (r + 1)) for r in range(4)])  # (4, 32)
    perm = [2, 0, 3, 1]
    pred = gt[perm]
    aligned = align_rps_to_gt(pred, gt)
    np.testing.assert_allclose(aligned, gt)


# ── evaluate error paths ─────────────────────────────────────────────────


def test_evaluate_missing_audio_track():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    rps = td.events(
        np.array([0.0, 1.0]), np.zeros((4, 2)), dims=("rotor", "time"), t_start=0.0, t_end=2.0
    )
    frame = td.Frame({"rps": rps})  # no "audio"
    p = FakePredictor()
    with pytest.raises(KeyError, match="audio"):
        evaluate(p, [frame], verbose=False)  # type: ignore[arg-type]


def test_evaluate_missing_rps_track():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    audio = td.uniform(np.zeros(16000, dtype=np.float32), 16000, dims=("time",), t_start=0)
    frame = td.Frame({"audio": audio})  # no "rps"
    p = FakePredictor()
    with pytest.raises(KeyError, match="rps"):
        evaluate(p, [frame], verbose=False)  # type: ignore[arg-type]


def test_evaluate_audio_not_grid_series():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    audio_es = td.events(
        np.array([0.0]), np.zeros((4, 1)), dims=("rotor", "time"), t_start=0.0, t_end=1.0
    )
    rps_es = td.events(
        np.array([0.0, 1.0]), np.zeros((4, 2)), dims=("rotor", "time"), t_start=0.0, t_end=2.0
    )
    frame = td.Frame({"audio": audio_es, "rps": rps_es})
    p = FakePredictor()
    with pytest.raises(TypeError, match="GridIndex"):
        evaluate(p, [frame], verbose=False)  # type: ignore[arg-type]


def test_evaluate_rps_not_stamp_series():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    audio = td.uniform(np.zeros(16000, dtype=np.float32), 16000, dims=("time",), t_start=0)
    frame = td.Frame({"audio": audio, "rps": audio})  # rps is a GridIndex Series
    p = FakePredictor()
    with pytest.raises(TypeError, match="StampIndex"):
        evaluate(p, [frame], verbose=False)  # type: ignore[arg-type]


def test_evaluate_with_unknown_alignment():
    from tasks.rps_prediction import evaluate

    class FakePredictor:
        def predict(self, audio, sr=16000):
            return np.zeros((4, 32))

    audio = td.uniform(np.zeros(16000, dtype=np.float32), 16000, dims=("time",), t_start=0)
    rps = td.events(
        np.array([0.0, 1.0]), np.zeros((4, 2)), dims=("rotor", "time"), t_start=0.0, t_end=2.0
    )
    frame = td.Frame({"audio": audio, "rps": rps})
    p = FakePredictor()
    with pytest.raises(ValueError, match="Unknown alignment"):
        evaluate(p, [frame], alignment="bogus", verbose=False)  # type: ignore[arg-type]


def test_align_shape_stretch_no_values_raises():
    from tasks.rps_prediction import _align_shape_stretch

    es = td.events(np.array([0.0, 0.5]), values=None, dims=("time",), t_start=0.0, t_end=1.0)
    audio = np.zeros(16000)
    with pytest.raises(ValueError, match="no values"):
        _align_shape_stretch(audio, es)
