#!/usr/bin/env python3
"""Publish rich per-recording ``td.Frame`` datasets to dload (R2).

Two datasets, one sample per recording, serialized with the generic Frame
codec (``data_processing.streams.frame_to_sample``, layout ``tdframe-v1``) so
``DloadFrameDataset`` decodes them back to ``td.Frame`` automatically:

- **DREGON-frames** — every recording ``discover_recordings`` finds under
  ``data/DREGON`` (in-flight, motor, clean-source splits), loaded via
  ``dregon.load_timeframe`` at native 44.1 kHz. On top of what the loader
  yields, the commanded rotor speeds are stored twice: ``motors_command`` is
  the *fixed* track (``clean_command_spikes`` — the canonical entry downstream
  code reads) and ``motors_command_raw`` keeps the untouched telemetry. The
  raw per-sample audio clock (``audio_timestamps`` from ``*_audiots.mat``) is
  preserved as an invariant entry.

- **michaels-frames** — FLY124/FLY125 (``michaels.MICHAELS_FILES``), audio at
  native 44.1 kHz realigned to the flight-log clock exactly as
  ``load_michaels_timeframe`` does (window cut + leading-gap fix +
  ``time_offset``/``time_dilation``), plus *every* remaining CSV column as
  aligned series on the same time base: numeric columns grouped per logical
  sensor block (``IMU_ATTI(0):*`` -> ``imu_atti``, ``Motor:Speed:*`` ->
  ``motor_speed``, ...) as ``(channel, time)`` Series with the original column
  names as channel labels; boolean/string columns as one ``(time,)`` Series
  each. Rows where a block logged nothing (DatCon merges sensors of different
  rates into one table) are dropped per block, so each block keeps its true
  sample times. The non-temporal ``Attribute|Value`` pairs go into the frame
  meta.

Idempotent: shards are content-addressed, so re-running re-uploads nothing
that already exists. Run, then pin::

    python scripts/publish_frame_datasets.py --dataset both
    dload pin DREGON-frames && dload pin michaels-frames

Memory note: frames are built lazily inside the generator that
``repo.commit`` consumes — exactly one recording is in memory at a time.
"""

from __future__ import annotations

import argparse
import gc
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
import tdseries as td

from data_processing import streams
from data_processing.dregon import (
    clean_command_spikes,
    discover_recordings,
    get_geometry,
    load_timeframe,
)
from data_processing.frames import make_recording_frame, with_meta
from data_processing.michaels import (
    MICHAELS_FILES,
    _load_michaels_data_raw,
)
from data_processing.michaels import (
    get_geometry as michaels_geometry,
)

Sample = tuple[str, dict[str, bytes]]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DREGON_DATASET = "DREGON-frames"
MICHAELS_DATASET = "michaels-frames"
CLEAN_KERNEL = 21  # clean_command_spikes median-filter kernel (its default)

# ─── DREGON ────────────────────────────────────────────────────────────────────


def _clean_command(values: np.ndarray) -> np.ndarray:
    return clean_command_spikes(values, kernel=CLEAN_KERNEL)


def build_dregon_frame(sample: dict, geometry: tuple[np.ndarray, np.ndarray]) -> td.Frame:
    """One DREGON recording -> rich Frame (audio + all telemetry + fixes)."""
    frame = load_timeframe(sample, geometry=geometry)

    fixes: dict[str, Any] = {}
    if "motors_command" in frame:
        raw = frame["motors_command"]
        frame = frame.with_entry("motors_command_raw", raw)
        frame = frame.with_entry("motors_command", raw.map_data(_clean_command))
        fixes["motors_command"] = (
            f"clean_command_spikes(kernel={CLEAN_KERNEL}): leading constant-value "
            "logging artifact zeroed + median filter along time; raw values kept "
            "in 'motors_command_raw'"
        )

    if sample.get("audiots_path"):
        # load_timeframe only uses audiots[0] as the time anchor; keep the full
        # per-sample clock (one Unix timestamp per audio sample) as-is.
        stamps = scipy.io.loadmat(sample["audiots_path"])["audio_timestamps"]
        frame = frame.with_entry(
            "audio_timestamps", td.wrap(stamps.flatten().astype(np.float64), dims=(None,))
        )

    return with_meta(
        frame,
        provenance={
            "script": "scripts/publish_frame_datasets.py",
            "loader": "data_processing.dregon.load_timeframe (native sr, no resample)",
            "fixes": fixes,
        },
    )


def iter_dregon_samples(dregon_dir: Path) -> Iterator[Sample]:
    geometry = get_geometry(dregon_dir)
    samples = sorted(discover_recordings(dregon_dir), key=lambda s: str(s["recording_id"]))
    ids = [s["recording_id"] for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate DREGON recording ids in discover_recordings output")
    for sample in samples:
        rid = str(sample["recording_id"])
        frame = build_dregon_frame(sample, geometry)
        fields = streams.frame_to_sample(frame)
        del frame
        print(f"  built {rid} ({sum(len(v) for v in fields.values()) / 1e6:.1f} MB)")
        yield rid, fields
        del fields
        gc.collect()


# ─── Michael's ─────────────────────────────────────────────────────────────────

_T_COL = "Clock:offsetTime"
_ATTR_COL = "Attribute|Value"
# Two-level grouping for the per-rotor motor blocks ("Motor:Speed:RFront" ->
# group "Motor:Speed"); everything else groups on the first ":" segment.
_TWO_LEVEL_PREFIXES = ("Motor:", "MotorCtrl:")


def _sanitize(name: str) -> str:
    """CSV header -> Frame entry name (codec forbids '/', '#' and '_frame')."""
    name = name.replace("(0)", "")
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    if not name or name == "_frame":
        raise ValueError(f"cannot derive an entry name from column {name!r}")
    return name


def _group_key(col: str) -> str:
    if col.startswith(_TWO_LEVEL_PREFIXES) and col.count(":") >= 2:
        return ":".join(col.split(":")[:2])
    return col.split(":")[0]


def _aligned_csv(
    csv_path: Path, time_offset: float, time_dilation: float, wav_duration: float
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """The exact row cut + timestamp fix of ``_load_michaels_data_raw``,
    applied to the *whole* CSV. Returns (cut rows, aligned stamps, full csv)."""
    csv = pd.read_csv(csv_path, low_memory=False)
    t = csv[_T_COL]
    mask = (t >= time_offset) & (t <= wav_duration + time_offset)
    cut = cast(pd.DataFrame, csv[mask]).reset_index(drop=True)
    ts = np.asarray(cut[_T_COL], dtype=np.float64)
    jump_idx = int(np.argmax(np.diff(ts)))
    ts[0 : jump_idx + 2] = np.linspace(ts[0], ts[jump_idx + 2], jump_idx + 2)
    ts = ts * time_dilation
    return cut, ts, csv


def _column_kind(col: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(col):
        return "numeric"
    values = col.dropna()
    if len(values) and all(isinstance(v, (bool, np.bool_)) for v in values):
        return "bool"
    return "string"


def _telemetry_entries(cut: pd.DataFrame, ts: np.ndarray) -> dict[str, td.Series]:
    """Every CSV column (minus time + attributes) as aligned Series entries."""
    entries: dict[str, td.Series] = {}
    numeric_groups: dict[str, list[str]] = {}
    for col in map(str, cut.columns):
        if col in (_T_COL, _ATTR_COL):
            continue
        column = cast(pd.Series, cut[col])
        kind = _column_kind(column)
        if kind == "numeric":
            numeric_groups.setdefault(_group_key(col), []).append(col)
            continue
        # bool / string columns: one (time,) Series each, on the rows where
        # the column actually has a value.
        mask = column.notna().to_numpy()
        if not mask.any():
            continue
        values = column.to_numpy()[mask]
        data = values.astype(bool) if kind == "bool" else values.astype(str)
        entries[_sanitize(col)] = td.events(ts[mask], data, dims=("time",))

    for group, cols in numeric_groups.items():
        arr = cut[cols].to_numpy(dtype=np.float64).T  # (C, M), time-last
        mask = ~np.all(np.isnan(arr), axis=0)  # rows where this block logged
        if not mask.any():
            continue
        ev = td.events(ts[mask], arr[:, mask], dims=("channel", "time"))
        entries[_sanitize(group)] = td.Series(
            ev.data,
            ("channel", "time"),
            {"channel": td.LabelIndex(tuple(cols)), "time": ev.tindex},
        )
    return entries


def _wav_duration_s(wav_path: Path) -> float:
    """Original (pre-crop) wav duration — what the loader's CSV window uses."""
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def build_michaels_frame(
    wav_rel: str, csv_rel: str, time_offset: float, time_dilation: float
) -> td.Frame:
    """One Michael's recording -> rich Frame (realigned audio + rps + full CSV)."""
    wav_path, csv_path = DATA_DIR / wav_rel, DATA_DIR / csv_rel
    # Audio + canonical rps exactly as load_michaels_timeframe builds them,
    # but at the native sample rate (sr=None -> no resampling).
    wav, ts_rps, ms, sample_rate = _load_michaels_data_raw(
        wav_path, csv_path, time_offset=time_offset, time_dilation=time_dilation, sr=None
    )
    tracks: dict[str, td.Series] = {
        "audio": td.uniform(wav, sample_rate, dims=("mic", "time")),
        "rps": td.events(ts_rps, ms, dims=("rotor", "time")),
    }
    del wav

    cut, ts, full_csv = _aligned_csv(
        csv_path, time_offset, time_dilation, wav_duration=_wav_duration_s(wav_path)
    )
    # The telemetry row cut must land on the very same rows (hence stamps) the
    # loader's rps came from — otherwise the entries are not aligned.
    if not np.array_equal(ts, ts_rps):
        raise AssertionError(f"{csv_rel}: aligned CSV stamps diverge from the loader's rps stamps")
    tracks.update(_telemetry_entries(cut, ts))

    attributes = [str(v) for v in full_csv[_ATTR_COL].dropna()] if _ATTR_COL in full_csv else []
    meta = {
        "recording_id": Path(csv_rel).stem,  # FLY124 / FLY125
        "wav": wav_rel,
        "csv": csv_rel,
        "time_offset": float(time_offset),
        "time_dilation": float(time_dilation),
        "sample_rate": int(sample_rate),
        "n_csv_rows": int(len(full_csv)),
        "n_csv_rows_aligned": int(len(cut)),
        "dat_attributes": attributes,
        "provenance": {
            "script": "scripts/publish_frame_datasets.py",
            "alignment": (
                "load_michaels_timeframe fixes: CSV rows cut to the audio window, "
                "leading timestamp gap linearly interpolated, stamps scaled by "
                "time_dilation, audio cropped to the covered span (native sr)"
            ),
            "telemetry": (
                "numeric CSV columns grouped per sensor block as (channel, time) "
                "Series labelled with the original column names; bool/string "
                "columns as one (time,) Series each; all-NaN rows dropped per "
                "block (DatCon logs sensors at different rates)"
            ),
        },
    }
    mic_pos, rotor_pos = michaels_geometry()
    return make_recording_frame(tracks, meta=meta, mic_pos=mic_pos, rotor_pos=rotor_pos)


def iter_michaels_samples() -> Iterator[Sample]:
    for wav_rel, csv_rel, time_offset, time_dilation in MICHAELS_FILES:
        rid = Path(csv_rel).stem
        frame = build_michaels_frame(wav_rel, csv_rel, time_offset, time_dilation)
        fields = streams.frame_to_sample(frame)
        del frame
        print(f"  built {rid} ({sum(len(v) for v in fields.values()) / 1e6:.1f} MB)")
        yield rid, fields
        del fields
        gc.collect()


# ─── Publishing ────────────────────────────────────────────────────────────────


def publish(dataset: str) -> None:
    repo = streams.open_repository()
    recipe = Path(__file__).read_text(encoding="utf-8")

    if dataset in ("dregon", "both"):
        print(f"Publishing {DREGON_DATASET} ...")
        manifest = repo.commit(
            DREGON_DATASET,
            iter_dregon_samples(DATA_DIR / "DREGON"),
            meta={
                streams.LAYOUT_META_KEY: streams.TDFRAME_LAYOUT,
                "description": (
                    "DREGON recordings as rich td.Frames: 8ch 44.1kHz audio, "
                    "motors_command (clean_command_spikes-fixed) + motors_command_raw, "
                    "motors_measured/imu/source_position on true timestamps, "
                    "audio_timestamps, geometry, per-recording meta"
                ),
                "source": "data/DREGON (dregon.discover_recordings, all splits)",
            },
            recipe=recipe,
            progress=print,
        )
        print(f"{DREGON_DATASET}@{manifest.version[:12]}: {manifest.num_samples} samples")

    if dataset in ("michaels", "both"):
        print(f"Publishing {MICHAELS_DATASET} ...")
        manifest = repo.commit(
            MICHAELS_DATASET,
            iter_michaels_samples(),
            meta={
                streams.LAYOUT_META_KEY: streams.TDFRAME_LAYOUT,
                "description": (
                    "Michael's FLY124/FLY125 as rich td.Frames: realigned native-sr "
                    "8ch audio, rps (rev/s), and every DJI flight-log CSV column as "
                    "aligned series grouped per sensor block"
                ),
                "source": "data/recording_with_motor_speed (michaels.MICHAELS_FILES)",
            },
            recipe=recipe,
            progress=print,
        )
        print(f"{MICHAELS_DATASET}@{manifest.version[:12]}: {manifest.num_samples} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish DREGON/michaels rich-frame datasets.")
    parser.add_argument(
        "--dataset",
        choices=("dregon", "michaels", "both"),
        default="both",
        help="which dataset(s) to publish (default: both)",
    )
    args = parser.parse_args()
    publish(args.dataset)


if __name__ == "__main__":
    main()
