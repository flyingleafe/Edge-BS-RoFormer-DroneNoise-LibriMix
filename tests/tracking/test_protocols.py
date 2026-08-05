"""Tests for the declarative protocol window specs (``tracking.protocols``).

Covers: (1) the window tables are non-empty and their fields sane; (2) the
frozen constants match the values the scripts shipped with (spot-checked as
literals, NOT re-imported from the scripts — the scripts now read them from
here); (3) pool membership; (4) the ``to_frame`` round-trip. CPU-fast, no
data: the beatvk manifest is a synthetic stub.
"""

from __future__ import annotations

import numpy as np
import pytest

import tracking.protocols as P

# A minimal beatvk manifest stub covering BOTH accepted shapes:
# {rid: {"windows": [...]}} and {rid: [...]}.
_STUB_MANIFEST = {
    "free-flight_nosource_room1": {
        "windows": [
            {"index": 0, "start_s": 4.0, "end_s": 20.0, "regime": "warmup", "mean_rps": 22.0},
            {"index": 1, "start_s": 20.0, "end_s": 36.0, "regime": "cruise", "mean_rps": 80.5},
        ]
    },
    "FLY124": [
        {"index": 0, "start_s": 0.0, "end_s": 16.0, "regime": "cruise", "mean_rps": 74.0},
    ],
}


def test_vk37_window_table() -> None:
    specs = list(P.iter_windows("vk37"))
    assert len(specs) == 5  # the 5 DREGON recordings carrying motors_measured
    assert [s.recording_id for s in specs] == list(P.VK37.recordings)
    for s in specs:
        assert s.protocol == "vk37" and s.index == 0 and s.regime == "cruise"
        assert s.start_s is None and s.end_s is None  # loader-derived bounds
        assert s.name == f"{s.recording_id}__w00"


def test_vk37_frozen_constants() -> None:
    # the scripts/vk_validation.py protocol, frozen (design §5.1)
    assert P.VK37.dataset == "DREGON"
    assert P.VK37.sr == 16000
    assert P.VK37.hop_s == 0.032
    assert P.VK37.window_s == 25.0
    assert P.VK37.edge_trim_s == 0.5
    assert P.VK37.min_motor_rps == 30.0
    assert P.VK37.smooth_frames == 8
    assert "free-flight_nosource_room1" in P.VK37.recordings
    assert "free-flight_whitenoise-high_room1" in P.VK37.recordings


def test_beatvk_frozen_constants() -> None:
    # the scripts/beatvk_eval.py protocol, frozen
    assert P.BEATVK.dataset == "beatvk-valid-raw"
    assert P.BEATVK.sr == 16000
    assert P.BEATVK.hop_samples == 512  # 0.032 s grid
    assert P.BEATVK.window_s == 16.0
    assert P.BEATVK.edge_trim_s == 0.5
    assert P.BEATVK.n_rotors == 4
    assert set(P.BEATVK_DREGON_RECS) == {
        "free-flight_nosource_room1",
        "free-flight_speech-low_room1",
        "free-flight_whitenoise-low_room1",
    }
    assert P.BEATVK.recordings == (*P.BEATVK_DREGON_RECS, "FLY124")
    assert P.FROZEN_FLY124_ALIGNMENT == (-20.84, 1.001)
    # publish-time regime tagging rule
    assert P.regime_of(3.0) == "ground"
    assert P.regime_of(30.0) == "warmup"
    assert P.regime_of(50.0) == "cruise"


def test_beatvk_windows_from_manifest_and_pools() -> None:
    specs = list(P.iter_windows("beatvk", _STUB_MANIFEST))
    assert len(specs) == 3
    dregon_cruise = [s for s in specs if P.BEATVK.pools["dregon_cruise"].contains(s)]
    fly124_cruise = [s for s in specs if P.BEATVK.pools["fly124_cruise"].contains(s)]
    assert [s.name for s in dregon_cruise] == ["free-flight_nosource_room1__w01"]
    assert [s.name for s in fly124_cruise] == ["FLY124__w00"]
    warmup = specs[0]
    assert warmup.regime == "warmup" and warmup.start_s == 4.0 and warmup.end_s == 20.0
    assert P.BEATVK.pools_of(warmup) == ("warmup", "all")

    sub = list(P.iter_windows("beatvk", _STUB_MANIFEST, recordings={"FLY124"}))
    assert [s.recording_id for s in sub] == ["FLY124"]


def test_beatvk_requires_manifest_and_validates_recordings() -> None:
    with pytest.raises(ValueError, match="manifest"):
        list(P.iter_windows("beatvk"))
    with pytest.raises(KeyError, match="unknown recordings"):
        list(P.iter_windows("beatvk", _STUB_MANIFEST, recordings={"NOPE"}))
    with pytest.raises(KeyError, match="unknown protocol"):
        P.get_protocol("nope")


def test_to_frame_round_trip() -> None:
    from tracking.stages import get_audio, get_rps

    rng = np.random.default_rng(0)
    sr = 16000
    audio = rng.standard_normal((2, sr)).astype(np.float32)
    ft = np.arange(0.0, 1.0 - 0.016, 0.032)
    rps = np.stack([np.full(len(ft), 70.0), np.full(len(ft), 82.0)])
    spec = P.WindowSpec(
        protocol="beatvk",
        recording_id="FLY124",
        index=3,
        start_s=48.0,
        end_s=64.0,
        regime="cruise",
    )

    frame = P.to_frame(audio, sr, spec, rps=rps, frame_times=ft, rps_meas=rps + 0.5)
    a, got_sr = get_audio(frame)
    assert got_sr == sr and a.shape == audio.shape
    np.testing.assert_allclose(a, audio, rtol=0, atol=0)
    r, times = get_rps(frame)
    np.testing.assert_allclose(r, rps)
    np.testing.assert_allclose(times, ft, atol=1e-9)
    r_meas, _ = get_rps(frame, "rps_meas")
    np.testing.assert_allclose(r_meas, rps + 0.5)
    meta = frame["meta"]
    assert meta["protocol"] == "beatvk"
    assert meta["recording_id"] == "FLY124"
    assert meta["window_index"] == 3
    assert meta["regime"] == "cruise"
    assert meta["start_s"] == 48.0 and meta["end_s"] == 64.0
