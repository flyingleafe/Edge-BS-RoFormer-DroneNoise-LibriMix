"""Unit tests for the importable core of ``scripts/refine_dregon_rps.py``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import refine_dregon_rps as R


def test_window_bounds_tiles_and_right_aligns() -> None:
    bounds = R.window_bounds(1996, 16.0, 12.0)
    assert bounds[0] == (0, 500)
    assert bounds[-1] == (1496, 1996)  # the last window is right-aligned
    assert all(i1 - i0 == 500 for i0, i1 in bounds)
    # every frame is covered
    covered = np.zeros(1996, dtype=bool)
    for i0, i1 in bounds:
        covered[i0:i1] = True
    assert covered.all()


def test_window_bounds_short_recording() -> None:
    assert R.window_bounds(120, 16.0, 12.0) == [(0, 120)]


def test_fade_weights_ramp_and_floor() -> None:
    w = R.fade_weights(10, 4)
    assert w[0] == pytest.approx(0.2)
    assert w[-1] == pytest.approx(0.2)
    assert w[5] == pytest.approx(1.0)
    assert (w > 0).all()
    assert np.allclose(w, w[::-1])
    assert np.allclose(R.fade_weights(6, 0), 1.0)


def test_interp_rps_drops_duplicate_stamps_and_clips() -> None:
    stamps = np.array([0.0, 1.0, 1.0, 2.0])
    vals = np.array([[10.0, 20.0, 99.0, 30.0]])
    got = R.interp_rps(vals, stamps, np.array([-1.0, 0.5, 1.5, 5.0]))
    assert got.dtype == np.float64
    assert got[0].tolist() == pytest.approx([10.0, 15.0, 25.0, 30.0])


def test_scale_shift_pct() -> None:
    init = np.full((2, 4), 80.0)
    refined = np.stack([np.full(4, 79.2), np.full(4, 80.8)])
    assert R.scale_shift_pct(refined, init) == pytest.approx([-1.0, 1.0])


OFFSET_S = 5.48


def _fake_recording(rid: str, n_frames: int) -> dict[str, object]:
    ft = np.arange(n_frames) * R.HOP_S
    r_tel = np.full((4, n_frames), 80.0)
    return {
        "recording_id": rid,
        "audio": np.zeros((1, int(n_frames * R.HOP_S * R.SR)), dtype=np.float32),
        "ft": ft,
        "r_tel": r_tel,
        "t0_offset_s": OFFSET_S,
    }


def test_stitch_crossfades_and_writes_sidecar(tmp_path: Path, monkeypatch) -> None:
    rid = "rec"
    monkeypatch.setitem(R._RECORDINGS, rid, _fake_recording(rid, 300))
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    # Two overlapping windows: the first refined to 79.0, the second rejected
    # (telemetry). The overlap must fade between the two values.
    for i0, i1, val, used in ((0, 200, 79.0, True), (100, 300, 80.0, False)):
        (raw / f"{rid}__f{i0:06d}.json").write_text(
            json.dumps(
                {
                    "recording": rid,
                    "i0": i0,
                    "i1": i1,
                    "used": used,
                    "reason": "ok" if used else "no_improvement",
                    "mean_rev_s": 80.0,
                    "scale_pct_per_rotor": [-1.25] * 4,
                    "r_window": np.full((4, i1 - i0), val).tolist(),
                }
            )
        )
    params = {"window_s": 6.4, "hop_s": 3.2, "k_max": 40, "channels": 4, "knot_s": 0.25, "lr": 1.0}
    written = R.stitch(tmp_path, tmp_path / "labels", "frames:TEST", params)

    assert written == [tmp_path / "labels" / f"{rid}.npz"]
    with np.load(written[0]) as z:
        r_ref = z["r_refined"]
        # The sidecar's times are the PUBLISHED recording's, not the trimmed
        # frame's: the loader's overlap trim is added back.
        assert float(z["t0_offset_s"]) == pytest.approx(OFFSET_S)
        assert z["ft"][0] == pytest.approx(OFFSET_S)
        assert z["ft"][1] == pytest.approx(OFFSET_S + R.HOP_S)
        assert z["r_telemetry"].shape == r_ref.shape == (4, 300)
        assert z["window_used"].tolist() == [True, False]
        assert z["window_starts"].tolist() == [0, 100]
        assert int(z["k_max"]) == 40
    assert r_ref[:, 0] == pytest.approx(79.0)  # only the refined window covers it
    assert r_ref[:, -1] == pytest.approx(80.0)  # only the telemetry window
    mid = r_ref[0, 100:200]
    assert (mid >= 79.0).all() and (mid <= 80.0).all()
    assert np.all(np.diff(mid) >= -1e-12)  # the fade is monotone across the overlap

    report = json.loads((tmp_path / "labels" / f"{rid}.report.json").read_text())
    assert report["n_windows"] == 2
    assert report["n_used"] == 1
    # The headline stat reads the STITCHED labels, so the crossfade toward the
    # rejected window dilutes the raw -1.25 % optimizer movement.
    assert report["cruise_scale_pct"] == pytest.approx(-0.9375)
    assert report["cruise_scale_pct_raw_optimizer"] == pytest.approx(-1.25)
    assert "r_window" not in report["windows"][0]
