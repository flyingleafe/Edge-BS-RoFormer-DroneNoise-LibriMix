"""The held-out Michael's TEST recordings (FLY103/FLY108, `michaels-test`).

Two layers:

- **Synthetic** (always runs): a tiny wav + DatCon-shaped CSV written to
  ``tmp_path`` proves the ANCHORED alignment path is exactly
  ``(t_log - time_offset) * time_dilation``, that mono audio becomes a
  ``(time,)`` Series with no ``mic_pos``, and that the legacy FLY124/FLY125
  path is untouched.
- **Real data** (skipped when ``data/new-drone-noises`` is absent): the real
  CSVs parse into four motor channels, and ``build_test`` yields the two
  frames at their real durations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import tdseries as td

from data_processing import sources, streams
from data_processing.derivations import SPECS
from data_processing.sources import michaels

SR = 8000
DUR_S = 6.0
#: the real recordings, as ``MICHAELS_TEST_FILES`` declares them.
EXPECTED = {"FLY103": ("103_2.wav", "FLY103.csv"), "FLY108": ("108_2.wav", "FLY108.csv")}


# ─── registry / declaration ──────────────────────────────────────────────────


def test_test_files_declared():
    assert len(michaels.MICHAELS_TEST_FILES) == 2
    got = {csv.split(".")[0]: (wav, csv) for wav, csv, _o, _d in michaels.MICHAELS_TEST_FILES}
    assert got == EXPECTED
    for _wav, _csv, offset, dilation in michaels.MICHAELS_TEST_FILES:
        assert isinstance(offset, float)
        assert 0.9 < dilation < 1.1


def test_registry_entry_resolves():
    entry = sources.get("michaels-test")
    assert entry.builder is michaels.build_test
    assert entry.raw_dataset == "new-drone-noises"
    assert entry.frames_dataset == "michaels-test-frames"
    assert "TEST" in entry.provenance["description"]
    meta = sources.dataset_meta("michaels-test")
    assert meta[streams.LAYOUT_META_KEY] == "tdframe-v1"


def test_derivation_spec_is_derivable():
    spec = SPECS["michaels-test-frames"]
    assert spec["generator"] == "source_frames"
    assert not spec.get("adopt_only")
    assert spec["gen"]["source"] == "michaels-test"
    assert spec["gen"]["raw"]["uri"].startswith("dload:new-drone-noises@")
    # the published michaels-frames spec must not have been touched
    assert SPECS["michaels-frames"]["adopt_only"] is True


# ─── synthetic tree ──────────────────────────────────────────────────────────


def _write_recording(root, *, wav_name: str, csv_name: str, log_t0: float, n_ch: int = 1):
    """A mono/multichannel wav plus a DatCon-shaped CSV covering more time.

    The motor speeds are a distinct ramp per rotor, so a misaligned read shows
    up as a wrong value rather than a wrong shape.
    """
    n = int(DUR_S * SR)
    audio = (0.05 * np.random.default_rng(0).standard_normal((n, n_ch))).astype(np.float32)
    sf.write(str(root / wav_name), audio if n_ch > 1 else audio[:, 0], SR)

    # the log runs from log_t0 and is 3x longer than the audio (the real case)
    t = np.round(np.arange(log_t0, log_t0 + 3 * DUR_S, 0.05), 6)
    frame = pd.DataFrame({"Clock:offsetTime": t})
    for i, rotor in enumerate(michaels.ROTOR_ORDER):
        frame[f"Motor:Speed:{rotor}"] = 60.0 * (i + 1) + np.arange(len(t))
    frame["IMU_ATTI(0):roll:C"] = np.linspace(0.0, 1.0, len(t))
    frame["Attribute|Value"] = ["ACType|M100"] + [np.nan] * (len(t) - 1)
    frame.to_csv(root / csv_name, index=False)
    return t


def test_read_motor_speeds_converts_rpm(tmp_path):
    t = _write_recording(tmp_path, wav_name="x.wav", csv_name="FLYTEST.csv", log_t0=-30.0)
    t_log, rps = michaels.read_motor_speeds(tmp_path / "FLYTEST.csv")
    assert rps.shape == (4, len(t))
    np.testing.assert_allclose(t_log, t)
    # RPM -> rev/s, and rotor r starts at 60*(r+1) RPM
    np.testing.assert_allclose(rps[:, 0], np.array([60.0, 120.0, 180.0, 240.0]) / 60.0)


def test_anchored_stamps_are_the_audio_clock(tmp_path):
    _write_recording(tmp_path, wav_name="x.wav", csv_name="FLYTEST.csv", log_t0=-30.0)
    offset, dilation = -20.0, 1.01
    wav, ts, ms, sr = michaels.load_raw_aligned(
        tmp_path / "x.wav",
        tmp_path / "FLYTEST.csv",
        time_offset=offset,
        time_dilation=dilation,
        sr=None,
        anchor=True,
    )
    assert sr == SR
    assert wav.shape == (1, int(DUR_S * SR))  # anchored path never crops audio
    assert ms.shape[0] == 4
    # stamp 0 is the audio's first sample, and the span is the audio duration
    assert 0.0 <= ts[0] < 0.05 * dilation
    assert abs(ts[-1] - DUR_S) < 0.05 * dilation
    assert np.all(np.diff(ts) > 0)


def test_anchor_and_legacy_paths_differ_as_designed(tmp_path):
    """The legacy path crops the audio head; the anchored one never does."""
    _write_recording(tmp_path, wav_name="x.wav", csv_name="FLYTEST.csv", log_t0=-30.0)
    legacy, ts_legacy, _ms, _sr = michaels.load_raw_aligned(
        tmp_path / "x.wav", tmp_path / "FLYTEST.csv", time_offset=-30.0, time_dilation=1.0, sr=None
    )
    anchored, ts_anchored, _ms2, _sr2 = michaels.load_raw_aligned(
        tmp_path / "x.wav",
        tmp_path / "FLYTEST.csv",
        time_offset=-30.0,
        time_dilation=1.0,
        sr=None,
        anchor=True,
    )
    assert anchored.shape[-1] == int(DUR_S * SR)
    assert legacy.shape[-1] <= anchored.shape[-1]
    # legacy keeps the raw log clock, anchored re-anchors it on the audio
    assert ts_legacy[0] < -29.0
    assert ts_anchored[0] >= 0.0


def test_build_frame_mono_layout(tmp_path):
    _write_recording(tmp_path, wav_name="x.wav", csv_name="FLYTEST.csv", log_t0=-30.0)
    frame = michaels.build_frame(tmp_path, "x.wav", "FLYTEST.csv", -20.0, 1.0, anchor=True)

    assert frame["audio"].dims == ("time",)  # mono, per frames.audio_series
    assert frame["audio"].data.shape == (int(DUR_S * SR),)
    assert frame["rps"].dims == ("rotor", "time")
    assert frame["rps"].data.shape[0] == 4
    assert "motor_speed" in frame  # the raw RPM block survives
    assert "mic_pos" not in frame  # one microphone is not an array
    assert frame["rotor_pos"].data.shape == (4, 3)

    meta = td.Frame(frame["meta"])
    assert meta["recording_id"] == "FLYTEST"
    assert meta["n_channels"] == 1
    assert meta["time_offset"] == -20.0
    assert "anchored" in meta["provenance"]["builder"] or "audio clock" in str(
        meta["provenance"]["alignment"]
    )
    assert "TEST" in str(meta["provenance"]["calibration"])

    # and it survives the published tdframe-v1 codec unchanged
    back = streams.sample_to_frame(streams.frame_to_sample(frame))
    np.testing.assert_allclose(np.asarray(back["audio"].data), np.asarray(frame["audio"].data))
    np.testing.assert_allclose(np.asarray(back["rps"].data), np.asarray(frame["rps"].data))


def test_multichannel_still_uses_the_mic_dim(tmp_path):
    """The 8-channel FLY124/FLY125 layout must not have changed."""
    _write_recording(
        tmp_path, wav_name="m.wav", csv_name="FLYMULTI.csv", log_t0=-30.0, n_ch=michaels.N_MICS
    )
    frame = michaels.build_frame(tmp_path, "m.wav", "FLYMULTI.csv", -20.0, 1.0, anchor=True)
    assert frame["audio"].dims == ("mic", "time")
    assert frame["mic_pos"].data.shape[0] == michaels.N_MICS


# ─── real data ───────────────────────────────────────────────────────────────


def _raw_root():
    from pathlib import Path

    for candidate in (Path("data/new-drone-noises"), Path("data")):
        if (candidate / "FLY103.csv").exists():
            return candidate
        if (candidate / "new-drone-noises" / "FLY103.csv").exists():
            return candidate / "new-drone-noises"
    return None


REAL_ROOT = _raw_root()
needs_raw = pytest.mark.skipif(REAL_ROOT is None, reason="data/new-drone-noises not present")


@needs_raw
@pytest.mark.parametrize("rid", sorted(EXPECTED))
def test_real_csv_has_four_motor_channels(rid):
    assert REAL_ROOT is not None
    t_log, rps = michaels.read_motor_speeds(REAL_ROOT / EXPECTED[rid][1])
    assert rps.shape[0] == 4
    assert len(t_log) == rps.shape[1] > 3000
    finite = rps[:, np.isfinite(rps).all(axis=0)]
    assert finite.shape[1] > 3000  # only the one all-NaN header row is dropped
    assert 60.0 < finite.max() < 120.0  # cruise rev/s, i.e. the RPM/60 conversion
    assert t_log[-1] - t_log[0] > 150.0  # the log outlasts the audio


@needs_raw
def test_build_test_yields_two_mono_frames():
    assert REAL_ROOT is not None
    frames = dict(michaels.build_test(REAL_ROOT))
    assert set(frames) == set(EXPECTED)
    for rid, expected_dur in (("FLY103", 106.48), ("FLY108", 99.44)):
        frame = frames[rid]
        audio = frame["audio"]
        assert audio.dims == ("time",)
        assert audio.tindex.sr == 48000  # published at the native rate
        assert abs(audio.data.shape[0] / 48000 - expected_dur) < 0.05
        rps = frame["rps"]
        assert rps.dims == ("rotor", "time")
        assert rps.data.shape[0] == 4
        assert np.isfinite(np.asarray(rps.data)).all()
        stamps = rps.tindex.abs_stamps
        assert -0.1 < stamps[0] < 0.1  # anchored on the audio
        assert abs(stamps[-1] - expected_dur) < 0.5
        assert np.asarray(rps.data).max() > 60.0
        assert "mic_pos" not in frame
