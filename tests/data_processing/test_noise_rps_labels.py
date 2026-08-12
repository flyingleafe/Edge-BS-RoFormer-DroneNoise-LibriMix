"""Tests for the DREGON rotor-speed label knobs of ``noise_rps_dataset``.

The two knobs are the refined-label sidecar (``dregon_rps_override_dir``) and
the constant gain (``dregon_rps_scale``). Both replace the VALUES of a
recording's rotor-speed track and must not touch its timebase. The tests use a
synthetic Frame at an absolute epoch ``t_start``, which is where a float
subtraction of the epoch would lose the offsets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tdseries as td

from data_processing.frames import make_recording_frame
from data_processing.noise_rps_dataset import (
    apply_rps_override,
    apply_rps_scale,
    build_noise_rps_datasets,
    resolve_override_dir,
)

SR = 16000
#: An absolute epoch ``t_start`` of the size the published frames carry
#: (~1.7e9 s = ~1.7e18 ticks), so a float64 epoch subtraction would be wrong.
EPOCH = 1_700_000_000.0
DURATION = 10.0
MOTOR_RATE = 100.0
RECORDING_ID = "free-flight_nosource_room1"


def _synthetic_frame(recording_id: str = RECORDING_ID) -> td.Frame:
    """Audio + a 4-rotor ``motors_measured`` track at an absolute epoch."""
    n_audio = int(DURATION * SR)
    n_motor = int(DURATION * MOTOR_RATE)
    offsets = np.arange(n_motor) / MOTOR_RATE
    values = np.stack([50.0 + 2.0 * r + offsets for r in range(4)]).astype(np.float32)
    audio = np.zeros((1, n_audio), dtype=np.float32)
    return make_recording_frame(
        {
            "audio": td.uniform(audio, SR, dims=("mic", "time"), t_start=EPOCH),
            "motors_measured": td.events(
                EPOCH + offsets, values, dims=("rotor", "time"), t_start=EPOCH
            ),
        },
        meta={"recording_id": recording_id},
        mic_pos=np.zeros((1, 3)),
        rotor_pos=np.zeros((4, 3)),
    )


def _write_sidecar(directory: Path, recording_id: str, *, slope: float = 3.0) -> np.ndarray:
    """A known ramp per rotor: ``r_refined[r] = 10*r + slope*ft``."""
    directory.mkdir(parents=True, exist_ok=True)
    ft = np.linspace(0.0, DURATION, 51)
    refined = np.stack([10.0 * r + slope * ft for r in range(4)])
    np.savez(directory / f"{recording_id}.npz", ft=ft, r_refined=refined, r_telemetry=refined)
    return ft


def test_override_replaces_values_against_a_known_ramp(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    _write_sidecar(tmp_path, RECORDING_ID)

    out = apply_rps_override(frame, "motors_measured", tmp_path)

    stamps = np.asarray(out["motors_measured"].tindex.abs_stamps_ticks, dtype=np.int64)
    offsets = (stamps - out["audio"].t_start_ticks) / float(td.TICKS_PER_SECOND)
    expected = np.stack([10.0 * r + 3.0 * offsets for r in range(4)])
    assert np.allclose(np.asarray(out["motors_measured"].data), expected, atol=1e-4)


def test_override_keeps_the_timebase_dims_and_dtype(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    _write_sidecar(tmp_path, RECORDING_ID)
    before = frame["motors_measured"]

    after = apply_rps_override(frame, "motors_measured", tmp_path)["motors_measured"]

    assert after.tindex.equal(before.tindex)
    assert after.dims == before.dims
    assert after.data.dtype == before.data.dtype
    assert after.data.shape == before.data.shape
    assert after.t_start_ticks == before.t_start_ticks


def test_override_clips_at_the_edges_of_the_sidecar(tmp_path: Path) -> None:
    """Stamps outside ``ft`` take the first/last refined value, not NaN."""
    frame = _synthetic_frame()
    directory = tmp_path / "short"
    directory.mkdir()
    ft = np.linspace(2.0, 5.0, 31)
    refined = np.stack([np.full_like(ft, 40.0 + r) for r in range(4)])
    np.savez(directory / f"{RECORDING_ID}.npz", ft=ft, r_refined=refined)

    out = apply_rps_override(frame, "motors_measured", directory)

    values = np.asarray(out["motors_measured"].data)
    assert np.isfinite(values).all()
    assert np.allclose(values, np.array([40.0, 41.0, 42.0, 43.0])[:, None], atol=1e-4)


def test_missing_sidecar_raises_with_the_available_ids(tmp_path: Path) -> None:
    _write_sidecar(tmp_path, "some_other_recording")
    frame = _synthetic_frame()

    with pytest.raises(FileNotFoundError) as err:
        apply_rps_override(frame, "motors_measured", tmp_path)
    assert RECORDING_ID in str(err.value)
    assert "some_other_recording" in str(err.value)


def test_scale_multiplies_the_values_and_keeps_the_timebase() -> None:
    frame = _synthetic_frame()
    before = frame["motors_measured"]

    after = apply_rps_scale(frame, "motors_measured", 0.99458)["motors_measured"]

    assert np.allclose(np.asarray(after.data), np.asarray(before.data) * 0.99458, atol=1e-4)
    assert after.tindex.equal(before.tindex)
    assert after.data.dtype == before.data.dtype


def test_both_knobs_together_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_noise_rps_datasets(
            dregon_dir="frames:DREGON-frames",
            michaels_dir=None,
            dregon_rps_override_dir=tmp_path,
            dregon_rps_scale=0.99458,
        )


def test_knobs_without_dregon_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DREGON only"):
        build_noise_rps_datasets(
            dregon_dir=None,
            michaels_dir="frames:michaels-frames",
            dregon_rps_override_dir=tmp_path,
        )


def test_knobs_need_a_frames_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="published-frames spec"):
        build_noise_rps_datasets(
            dregon_dir="data/DREGON",
            michaels_dir=None,
            dregon_rps_scale=0.99458,
        )


def _michaels_frame() -> td.Frame:
    """The same shape as ``_synthetic_frame``, but with a generic ``rps``
    track — the michaels convention."""
    frame = _synthetic_frame("FLY125")
    series = frame["motors_measured"]
    return td.Frame(
        {
            "audio": frame["audio"],
            "rps": series,
            "meta": frame["meta"],
        }
    )


def test_override_applies_to_dregon_only_through_the_frames_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole loader path, with the published-frames stream faked out.

    Also pins WHERE the replacement happens: the sidecar times are relative to
    the FULL frame's audio t_start, so the values must already be refined
    before the loader trims the audio/telemetry overlap.
    """
    _write_sidecar(tmp_path, RECORDING_ID)

    def fake_iter(name, version, splits=None):
        yield _synthetic_frame() if name == "DREGON-frames" else _michaels_frame()

    monkeypatch.setattr("data_processing.noise_rps_dataset.iter_published_frames", fake_iter)

    train_ds, _valid_ds = build_noise_rps_datasets(
        dregon_dir="frames:DREGON-frames",
        michaels_dir="frames:michaels-frames",
        chunk_size=SR,
        train_samples=1,
        val_samples=1,
        val_pct=0.2,
        dregon_rps_override_dir=tmp_path,
    )

    by_origin = {record.origin: record for record in train_ds.records}
    assert set(by_origin) == {"dregon", "michaels"}

    dregon = by_origin["dregon"].frame
    stamps = np.asarray(dregon["motors_measured"].tindex.abs_stamps_ticks, dtype=np.int64)
    offsets = (stamps - int(EPOCH * td.TICKS_PER_SECOND)) / float(td.TICKS_PER_SECOND)
    expected = np.stack([10.0 * r + 3.0 * offsets for r in range(4)])
    assert np.allclose(np.asarray(dregon["motors_measured"].data), expected, atol=1e-3)

    # Michael's keeps the original synthetic ramp (50 + 2*rotor + offset).
    michaels = by_origin["michaels"].frame
    m_stamps = np.asarray(michaels["rps"].tindex.abs_stamps_ticks, dtype=np.int64)
    m_offsets = (m_stamps - int(EPOCH * td.TICKS_PER_SECOND)) / float(td.TICKS_PER_SECOND)
    m_expected = np.stack([50.0 + 2.0 * r + m_offsets for r in range(4)])
    assert np.allclose(np.asarray(michaels["rps"].data), m_expected, atol=1e-3)


def test_relative_override_dir_resolves_against_the_repo_root() -> None:
    resolved = resolve_override_dir("src/data_processing/refined_labels")
    assert resolved.is_absolute()
    assert resolved.parts[-3:] == ("src", "data_processing", "refined_labels")
    assert (resolved.parent.parent.parent / "pyproject.toml").is_file()
