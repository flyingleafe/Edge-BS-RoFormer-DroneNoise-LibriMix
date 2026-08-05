"""Tests for ``plots.explore`` — meta_table / grid / pick over synthetic
frames and a fake map-style dataset. No dload network access."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import tdseries as td

from plots.dwym import DwymResult
from plots.explore import datasets, grid, meta_table, pick

SR = 8000
DUR_S = 0.4


def _audio(seed: int) -> td.Series:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(int(SR * DUR_S)).astype(np.float32) * 0.05
    return td.uniform(data, SR, dims=("time",), t_start=0.0)


def _rps(seed: int, n: int = 30) -> td.Series:
    rng = np.random.default_rng(seed)
    values = rng.uniform(40.0, 90.0, size=(4, n))
    times = np.linspace(0.0, DUR_S, n, endpoint=False)
    return td.events(times, values, dims=("rotor", "time"), t_start=0.0, t_end=DUR_S)


def make_frame(i: int, *, alias_entries: bool = False) -> td.Frame:
    """One synthetic sample; ``alias_entries`` uses raw (un-coerced) names."""
    audio_key = "waveform" if alias_entries else "audio"
    rps_key = "motors_command" if alias_entries else "rps"
    return td.Frame(
        {
            audio_key: _audio(i),
            rps_key: _rps(i + 100),
            "meta": td.Frame(
                {"recording_id": f"rec_{i:02d}", "split": "train", "snr_db": -10.0 - i}
            ),
        }
    )


class FakeDataset:
    """Minimal map-style dataset (no dload, no torch)."""

    def __init__(self, frames: list[td.Frame]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, i: int) -> td.Frame:
        return self._frames[i]


@pytest.fixture()
def frames() -> list[td.Frame]:
    return [make_frame(i) for i in range(6)]


# ── meta_table ─────────────────────────────────────────────────────────


def test_meta_table_columns_and_rows(frames):
    table = meta_table(frames)
    assert len(table) == 6
    for column in ("recording_id", "split", "snr_db", "entries", "duration_s"):
        assert column in table.columns
    assert table["recording_id"].tolist() == [f"rec_{i:02d}" for i in range(6)]
    assert table["entries"].iloc[0] == "audio, rps"
    assert table["duration_s"].iloc[0] == pytest.approx(DUR_S, abs=1e-3)


def test_meta_table_limit_and_fields(frames):
    table = meta_table(frames, fields=["split"], limit=2)
    assert len(table) == 2
    assert "snr_db" not in table.columns
    assert table["split"].tolist() == ["train", "train"]


def test_meta_table_accepts_map_style_dataset(frames):
    table = meta_table(FakeDataset(frames), limit=3)
    assert len(table) == 3


# ── grid ───────────────────────────────────────────────────────────────


def test_grid_returns_dwym_result(frames):
    result = grid(frames, n=4, seed=0)
    assert isinstance(result, DwymResult)
    assert result.route == "grid"
    assert result.audio == {}
    assert len(result.figure.axes) >= 4
    result.close()


def test_grid_is_deterministic_with_seed(frames):
    def titles(result):
        out = [ax.get_title() for ax in result.figure.axes if ax.get_title()]
        result.close()
        return out

    assert titles(grid(frames, n=3, seed=7)) == titles(grid(frames, n=3, seed=7))


def test_grid_handles_fewer_frames_than_n(frames):
    result = grid(frames[:2], n=8)
    titled = [ax for ax in result.figure.axes if ax.get_title()]
    assert len(titled) == 2
    result.close()


def test_grid_coerces_alias_entries():
    aliased = [make_frame(i, alias_entries=True) for i in range(3)]
    with pytest.warns(UserWarning, match="coerce"):
        result = grid(aliased, n=2, seed=0)
    assert result.route == "grid"
    result.close()


def test_grid_rejects_unknown_hints(frames):
    with pytest.raises(TypeError, match="unsupported hints"):
        grid(frames, n=2, nonsense=True)


# ── pick ───────────────────────────────────────────────────────────────


def test_pick_by_index_map_style(frames):
    frame = pick(FakeDataset(frames), 4)
    assert frame["meta"]["recording_id"] == "rec_04"


def test_pick_by_index_stream(frames):
    frame = pick(iter(frames), 2)
    assert frame["meta"]["recording_id"] == "rec_02"


def test_pick_negative_index(frames):
    assert pick(FakeDataset(frames), -1)["meta"]["recording_id"] == "rec_05"
    with pytest.raises(IndexError, match="map-style"):
        pick(iter(frames), -1)


def test_pick_by_query_string(frames):
    frame = pick(frames, "rec_03")
    assert frame["meta"]["recording_id"] == "rec_03"


def test_pick_by_predicate(frames):
    frame = pick(frames, lambda f: f["meta"]["snr_db"] < -14.0)
    assert frame["meta"]["recording_id"] == "rec_05"


def test_pick_coerces_aliases():
    aliased = [make_frame(i, alias_entries=True) for i in range(2)]
    with pytest.warns(UserWarning, match="coerce"):
        frame = pick(aliased, 1)
    assert "audio" in frame and "rps" in frame


def test_pick_remap_hint_is_silent():
    frame = td.Frame({"audio": _audio(0), "my_track": _rps(1), "meta": td.Frame({})})
    picked = pick([frame], 0, rps="my_track")
    assert "rps" in picked and "my_track" not in dict(picked.items())


def test_pick_no_match_raises(frames):
    with pytest.raises(ValueError, match="no sample matched"):
        pick(frames, "does_not_exist")


# ── datasets ───────────────────────────────────────────────────────────


def test_datasets_reads_the_lock_offline():
    table = datasets()
    assert {"name", "version"} <= set(table.columns)
    assert "DREGON-frames" in set(table["name"])
    assert all(len(v) == 12 for v in table["version"])
