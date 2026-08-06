"""Tests for the declarative protocol window specs (``tracking.protocols``).

Covers: (1) the window tables are non-empty and their fields sane; (2) the
frozen constants match the values the scripts shipped with (spot-checked as
literals, NOT re-imported from the scripts — the scripts now read them from
here); (3) pool membership; (4) the ``to_frame`` round-trip; (5) the prep-cache
reader (``resolve_prep_dir`` / ``load_prep_window``). CPU-fast, no data: the
beatvk manifest is a synthetic stub and the prep window is written into a
temporary directory.
"""

from __future__ import annotations

from pathlib import Path

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
    from tracking.top import get_audio, get_rps

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


# ---------------------------------------------------------------------------
# 5. the shared window slicer, the PIT assignment and the pooling rule


def test_slice_window_grid_and_edge_mask() -> None:
    spec = next(s for s in P.iter_windows("beatvk", _STUB_MANIFEST) if s.index == 1)
    sr = P.BEATVK.sr
    audio = np.arange(40 * sr, dtype=np.float32)[None, :]  # 40 s, one channel
    ts = np.arange(0.0, 40.0, 0.01)
    vals = np.stack([80.0 + i + 0.0 * ts for i in range(4)])

    seg, ft, r_meas, edge = P.slice_window(audio, sr, spec, ts, vals)

    assert seg.shape == (1, 16 * sr)  # [20, 36) s
    assert seg[0, 0] == 20.0 * sr  # sliced by sample index off the window start
    np.testing.assert_allclose(np.diff(ft), P.BEATVK.hop_s)
    assert ft[0] == 0.0 and ft[-1] < 16.0  # window-relative grid
    assert r_meas is not None and r_meas.shape == (4, len(ft))
    np.testing.assert_allclose(r_meas[:, 0], [80.0, 81.0, 82.0, 83.0])
    # the protocol's 0.5 s edge trim, both ends
    assert not edge[0] and not edge[-1]
    assert edge.sum() == int(((ft > 0.5) & (ft < ft[-1] - 0.5)).sum())


def test_slice_window_truncation_and_bounds() -> None:
    spec = next(s for s in P.iter_windows("beatvk", _STUB_MANIFEST) if s.index == 1)
    sr = P.BEATVK.sr
    audio = np.zeros((2, 40 * sr), dtype=np.float32)
    seg, ft, r_meas, _ = P.slice_window(audio, sr, spec, seconds=4.0)
    assert seg.shape == (2, 4 * sr) and r_meas is None and len(ft) == 125
    with pytest.raises(ValueError, match="outside"):
        P.slice_window(np.zeros((2, 25 * sr), dtype=np.float32), sr, spec)


def test_pit_align_recovers_the_permutation() -> None:
    rng = np.random.default_rng(0)
    gt = np.stack([np.full(50, v) for v in (70.0, 75.0, 80.0, 85.0)])
    shuffle = [2, 0, 3, 1]
    pred = gt[shuffle] + rng.normal(0.0, 0.05, gt.shape)

    aligned, perm = P.pit_align(pred, gt)

    # perm undoes the shuffle: pred row perm[i] belongs to gt rotor i
    assert [shuffle[i] for i in perm] == [0, 1, 2, 3]
    np.testing.assert_allclose(aligned, pred[perm])
    assert np.abs(aligned - gt).max() < 0.5
    # the same assignment losses.pit.align_rps_to_gt exposes (it delegates here)
    from losses.pit import align_rps_to_gt

    np.testing.assert_array_equal(align_rps_to_gt(pred, gt), aligned)


def test_pool_means_filters_by_recording_regime_and_window() -> None:
    rows = [
        {"recording": "free-flight_nosource_room1", "regime": "cruise", "window": 0, "mae": 2.0},
        {"recording": "free-flight_nosource_room1", "regime": "cruise", "window": 1, "mae": 4.0},
        {"recording": "FLY124", "regime": "warmup", "window": 0, "mae": 9.0},
    ]
    pooled = P.pool_means(rows, P.BEATVK_REPORT_POOLS, ndigits=4)
    assert pooled["dregon_cruise"] == 3.0
    assert pooled["dregon_ramp"] == 2.0  # window 0 = the takeoff ramp
    assert pooled["dregon_steady"] == 4.0  # windows 1-2
    assert pooled["fly124_warmup"] == 9.0
    assert pooled["fly124_cruise"] is None  # no member window
    assert pooled["all"] == 5.0


def _write_prep(dst: Path, key: str) -> dict[str, np.ndarray]:
    """One synthetic prep-cache window, in the frozen ``.npz`` layout."""
    arrays = {
        "audio": np.zeros((2, 320), dtype=np.float32),
        "ft": np.arange(10, dtype=np.float32) * 0.032,
        "r_meas": np.full((4, 10), 80.0, dtype=np.float32),
    }
    np.savez(dst / f"{key}.npz", regime=np.str_("cruise"), **arrays)
    return arrays


def test_resolve_prep_dir_honours_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(P.PREP_DIR_ENV, str(tmp_path))
    assert P.resolve_prep_dir() == tmp_path
    # No env var, no pulled cache -> the ``--build-preps`` output.
    monkeypatch.delenv(P.PREP_DIR_ENV)
    monkeypatch.setattr(P, "PULLED_PREP_SUBPATH", "no/such/prep_cache")
    assert P.resolve_prep_dir() == P.BUILT_PREP


def test_load_prep_window_reads_the_frozen_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _write_prep(tmp_path, "FLY124__w02")
    monkeypatch.setenv(P.PREP_DIR_ENV, str(tmp_path))

    win = P.load_prep_window("FLY124__w02")

    assert set(win) == {"audio", "ft", "r", "regime"}
    assert win["regime"] == "cruise"
    for name, entry in (("audio", "audio"), ("ft", "ft"), ("r", "r_meas")):
        assert win[name].dtype == np.float64  # the reader widens every array
        np.testing.assert_allclose(win[name], arrays[entry])
    # An explicit directory overrides the resolution order.
    other = tmp_path / "other"
    other.mkdir()
    _write_prep(other, "FLY124__w03")
    assert P.load_prep_window("FLY124__w03", other)["r"].shape == (4, 10)
