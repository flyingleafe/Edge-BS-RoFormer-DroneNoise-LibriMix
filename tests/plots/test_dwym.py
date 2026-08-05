"""Tests for ``plots.dwym`` dispatch and ``data_processing.canonical`` coercion.

Small synthetic frames per dispatch shape; every figure is closed at the
end of each test (Agg backend, no display).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import numpy as np
import pytest
import tdseries as td

from plots.dwym import DwymResult, dwym

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


def _salience_series(seed: int = 2, n_bins: int = 16, n_frames: int = 25) -> td.Series:
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 1.0, size=(n_bins, n_frames)).astype(np.float32)
    return td.uniform(data, (SR, 512), dims=("freq", "time"), t_start=0.0)


# ── dispatch table ────────────────────────────────────────────────────────


def test_dwym_se_triple_routes_to_se():
    frame = td.Frame(
        {
            "mixture": _audio_series(0),
            "target": _audio_series(1),
            "enhanced": _audio_series(2),
        }
    )
    result = dwym(frame)
    assert result.route == "se"
    assert isinstance(result.figure, matplotlib.figure.Figure)
    # Three spectrogram rows.
    assert len([ax for ax in result.figure.axes]) >= 3
    assert set(result.audio) == {"mixture", "target", "enhanced"}
    result.close()


def test_dwym_audio_plus_rps_routes_to_rps():
    frame = td.Frame({"audio": _audio_series(), "rps": _rps_series()})
    result = dwym(frame)
    assert result.route == "rps"
    assert "audio" in result.audio
    result.close()


def test_dwym_audio_rps_and_pred():
    pred = td.uniform(
        np.random.default_rng(3).uniform(40.0, 90.0, size=(4, 30)),
        60,  # 60 Hz frame rate over 0.5 s
        dims=("rotor", "time"),
        t_start=0.0,
    )
    frame = td.Frame({"audio": _audio_series(), "rps": _rps_series(), "rps_pred": pred})
    result = dwym(frame)
    assert result.route == "rps"
    result.close()


def test_dwym_salience_routes_to_salience():
    frame = td.Frame(
        {"audio": _audio_series(), "salience": _salience_series(), "rps": _rps_series()}
    )
    with pytest.warns(UserWarning, match="freqs"):
        result = dwym(frame)
    assert result.route == "salience"
    result.close()


def test_dwym_noise_gen_pair_in_one_frame():
    frame = td.Frame({"audio": _audio_series(0), "generated": _audio_series(1)})
    result = dwym(frame)
    assert result.route == "noise_gen"
    assert set(result.audio) == {"audio", "generated"}
    result.close()


def test_dwym_dict_of_two_bare_audio_frames_is_noise_gen():
    frames = {
        "real": td.Frame({"audio": _audio_series(0)}),
        "generated": td.Frame({"audio": _audio_series(1)}),
    }
    result = dwym(frames)
    assert result.route == "noise_gen"
    assert set(result.audio) == {"real/audio", "generated/audio"}
    result.close()


def test_dwym_bare_audio_routes_to_audio():
    frame = td.Frame({"audio": _audio_series()})
    result = dwym(frame)
    assert result.route == "audio"
    # Spectrogram row + waveform row.
    assert len(result.figure.axes) == 2
    result.close()


def test_dwym_unknown_mix_falls_through_to_timeframe():
    imu = td.events(
        np.linspace(0.0, DUR_S, 20, endpoint=False),
        np.random.default_rng(4).standard_normal((3, 20)),
        dims=(None, "time"),
        t_start=0.0,
        t_end=DUR_S,
    )
    frame = td.Frame({"audio": _audio_series(), "imu_accel": imu})
    result = dwym(frame)
    assert result.route == "timeframe"
    result.close()


def test_dwym_dict_of_two_rps_frames_renders_multi():
    frames = {
        "model A": td.Frame({"audio": _audio_series(0), "rps": _rps_series(1)}),
        "model B": td.Frame({"audio": _audio_series(1), "rps": _rps_series(2)}),
    }
    result = dwym(frames)
    assert result.route == "multi:rps"
    assert len(result.figures) == 1
    # One row block (spectrogram + rps) per label.
    assert len(result.figure.axes) >= 4
    result.close()


def test_dwym_forced_renderer_wins():
    frame = td.Frame({"audio": _audio_series(), "rps": _rps_series()})
    result = dwym(frame, renderer="timeframe")
    assert result.route == "timeframe"
    result.close()


def test_dwym_rejects_unknown_renderer_and_input():
    frame = td.Frame({"audio": _audio_series()})
    with pytest.raises(ValueError, match="renderer"):
        dwym(frame, renderer="bogus")
    with pytest.raises(TypeError, match="dwym expects"):
        dwym(42)  # type: ignore[arg-type]


def test_dwym_result_save(tmp_path):
    frame = td.Frame({"audio": _audio_series()})
    result = dwym(frame)
    paths = result.save(tmp_path / "fig.png")
    assert len(paths) == 1
    assert paths[0].exists()
    result.close()


def test_dwym_result_is_dwymresult():
    frame = td.Frame({"audio": _audio_series()})
    result = dwym(frame)
    assert isinstance(result, DwymResult)
    result.close()


# ── coercion ──────────────────────────────────────────────────────────────


def test_dwym_remap_hint_silences_and_routes():
    frame = td.Frame({"audio": _audio_series(), "motor_speed": _rps_series()})
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = dwym(frame, rps="motor_speed")
    assert result.route == "rps"
    result.close()
