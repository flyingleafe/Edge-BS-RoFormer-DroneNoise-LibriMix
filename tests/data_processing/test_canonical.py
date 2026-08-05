"""Tests for entry-name canonicalization (``data_processing.canonical``)."""

import warnings

import numpy as np
import pytest
import tdseries as td

from data_processing.canonical import coerce_frame

SR = 8000
DUR_S = 0.5


def _audio_series(seed: int = 0, channels: int | None = None) -> td.Series:
    rng = np.random.default_rng(seed)
    n = int(SR * DUR_S)
    if channels is None:
        data = rng.standard_normal(n).astype(np.float32) * 0.05
        return td.uniform(data, SR, dims=("time",), t_start=0.0)
    data = rng.standard_normal((channels, n)).astype(np.float32) * 0.05
    return td.uniform(data, SR, dims=("mic", "time"), t_start=0.0)


def _rps_series(seed: int = 1, n: int = 40) -> td.Series:
    rng = np.random.default_rng(seed)
    values = rng.uniform(40.0, 90.0, size=(4, n))
    times = np.linspace(0.0, DUR_S, n, endpoint=False)  # events domain is half-open
    return td.events(times, values, dims=("rotor", "time"), t_start=0.0, t_end=DUR_S)


def test_coerce_motor_alias_becomes_rps_with_warning():
    frame = td.Frame({"audio": _audio_series(), "motors_command": _rps_series()})
    with pytest.warns(UserWarning, match="motors_command.*rps"):
        coerced = coerce_frame(frame)
    assert "rps" in coerced
    assert "motors_command" not in coerced


def test_coerce_override_is_silent_and_wins():
    frame = td.Frame({"audio": _audio_series(), "motor_speed": _rps_series()})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coerced = coerce_frame(frame, rps="motor_speed")
    assert "rps" in coerced
    assert "motor_speed" not in coerced


def test_coerce_keeps_unknown_entries():
    imu = td.events(
        np.linspace(0.0, DUR_S, 20, endpoint=False),
        np.random.default_rng(4).standard_normal((3, 20)),
        dims=(None, "time"),
        t_start=0.0,
        t_end=DUR_S,
    )
    frame = td.Frame({"audio": _audio_series(), "motors_measured": _rps_series(), "imu_accel": imu})
    with pytest.warns(UserWarning):
        coerced = coerce_frame(frame)
    assert "imu_accel" in coerced
    assert "rps" in coerced


def test_coerce_sole_waveform_becomes_audio():
    frame = td.Frame({"recording": _audio_series(channels=2), "rps": _rps_series()})
    with pytest.warns(UserWarning, match="recording.*audio"):
        coerced = coerce_frame(frame)
    assert "audio" in coerced
    assert "recording" not in coerced


def test_coerce_missing_override_raises():
    frame = td.Frame({"audio": _audio_series()})
    with pytest.raises(ValueError, match="no such entry"):
        coerce_frame(frame, rps="nope")
