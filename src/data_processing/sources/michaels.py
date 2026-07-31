"""Michael's drone-noise source — uniform registry entry.

8-channel DJI Matrice 100 in-flight recordings + flight-controller CSV logs
(per-motor rotation speed in RPM). The audio and telemetry clocks are *not*
aligned by default — a per-file ``time_offset`` (seconds) plus a
``time_dilation`` clock-rate factor brings them into register, and the logged
speeds themselves are low by a small multiplicative factor
(``MICHAELS_RPS_SCALE``). All three constants are **measured**, not hand-tuned:
see the block comment above :data:`MICHAELS_FILES` and
``docs/experiments/rps-refine-precision.md`` §§ WP13 (timing + the model form)
and WP14 (the rev/s magnitudes, refit on 13 windows).

Raw layout (dload dataset ``recording_with_motor_speed``):
  - ``recording_1/124.wav`` + ``recording_1/FLY124.csv``
  - ``recording_2/125.wav`` + ``recording_2/FLY125.csv``

:func:`build` yields one rich ``tdframe-v1`` Frame per recording: native-sr
8-ch audio realigned to the flight-log clock (window cut + leading-gap fix +
``time_offset``/``time_dilation``), the canonical ``rps`` track (rev/s), plus
*every* remaining CSV column as aligned series on the same time base — numeric
columns grouped per logical sensor block (``IMU_ATTI(0):*`` → ``imu_atti``,
``Motor:Speed:*`` → ``motor_speed``, ...) as ``(channel, time)`` Series with
the original column names as channel labels; boolean/string columns as one
``(time,)`` Series each. Rows where a block logged nothing (DatCon merges
sensors of different rates into one table) are dropped per block, so each
block keeps its true sample times. The non-temporal ``Attribute|Value`` pairs
go into the frame meta.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import librosa as lr
import numpy as np
import pandas as pd
import soundfile as sf
import tdseries as td

from data_processing.frames import make_recording_frame

# ── Measured alignment / label calibration (2026-07-31) ─────────────────────
#
# Referee: the LABEL-FREE Vold-Kalman reconstruction residual ||x - x_hat||/||x||
# (k = 1..30) of the harmonic model built ALONG a candidate telemetry
# trajectory, scored on contiguous 16 s windows of the frozen beat-VK window
# protocol. Only the TELEMETRY is shifted/scaled; our own blind RPS estimate is
# never consulted.
#
# TIMING — both recordings show a clock DILATION error (the audio-optimal lag
# drifts linearly with time), not a constant offset. Folding the fitted
# lag(t) = a + b*t into this loader's parameterisation gives
#     time_dilation_new = time_dilation_old / (1 - b)
#     time_offset_new   = time_offset_old  - a / (1 - b)
#
#   FLY124 — 4 cruise windows, OLS b = +0.65356 ms/s, a = -86.131 ms,
#     R^2 = 0.942, residual RMS 2.90 ms vs 12.04 ms for a constant-lag model.
#     (-20.84, 1.001) -> (-20.753813, 1.001654644).
#   FLY125 — 9 cruise windows, OLS b = +0.37656 ms/s, a = -172.086 ms,
#     R^2 = 0.923, residual RMS 4.49 ms vs 16.19 ms constant-lag.
#     (-26.51, 1.0048) -> (-26.337849, 1.005178509).
#
# VALUE — the logged speeds are LOW by ~0.55-0.65 rev/s at cruise (DREGON shows
# the opposite sign, so this is a Michael's-rig property, not a referee bias).
# Additive vs multiplicative is statistically unresolved in the cruise band; we
# ship the MULTIPLICATIVE form on physical grounds, because these frames cover
# the WHOLE recording including warm-up and ground idle, where an additive
# +0.6 rev/s would corrupt a near-stationary rotor's label (and manufacture
# harmonics at a standstill) while a scale correctly vanishes as rps -> 0.
#
# DEGENERACY — the global gain is not separable from a sample-clock error, so
# this constant is a *label-for-this-audio* correction, not proof that the ESC
# is miscalibrated. Per-rotor constants are NOT identifiable (between-rotor
# spread < within-rotor scatter) and a per-rotor lag is refuted three ways.
#
# (wav_path, csv_path, time_offset_sec, time_dilation) — paths relative to the
# raw root (the `recording_with_motor_speed` tree).
MICHAELS_FILES = [
    ("recording_1/124.wav", "recording_1/FLY124.csv", -20.753813, 1.001654644),
    ("recording_2/125.wav", "recording_2/FLY125.csv", -26.337849, 1.005178509),
]

#: Per-recording MULTIPLICATIVE rev/s correction, keyed by CSV stem (the
#: recording id). Applied to the rotor speeds in :func:`load_raw_aligned`, so
#: every consumer of ``rps`` gets calibrated labels. Recordings without a
#: measured constant (the 103 unaligned ``new-drone-noises`` logs) fall back to
#: 1.0. Both values are the WP14 13-window global refit over non-twin rotors;
#: they supersede 1.00839 / 1.00690 (WP13, 2-4 windows, twin-contaminated).
MICHAELS_RPS_SCALE: dict[str, float] = {
    "FLY124": 1.00698,  # g = 0.698 % +- 0.069 -> +0.558 rev/s at 80 rev/s
    "FLY125": 1.00706,  # g = 0.706 % +- 0.034 -> +0.565 rev/s at 80 rev/s
}


def rps_scale_for(csv_path: str | Path) -> float:
    """Calibrated rev/s scale for a recording, from its CSV stem (1.0 if none)."""
    return MICHAELS_RPS_SCALE.get(Path(csv_path).stem.upper(), 1.0)


# ── Array / airframe geometry ───────────────────────────────────────────────
# Body frame: X = forward, Y = left, Z = up; origin at the drone body centre
# (rotor plane). The microphone array is a HORIZONTAL ring (plane = X-Y, normal
# +Z) mounted forward of and above the body. Constants read from
# `data/recording_with_motor_speed/Microphone_Array_Configuration.jpeg`;
# wheelbase from the DJI Matrice 100 (the airframe in the rig photos).

N_MICS = 8
NUM_ROTORS = 4
MIC_ARRAY_RADIUS = 0.0825  # m (Ø165 mm)
# The rig spec's horizontal "20cm" is "drone body to centre of array" — i.e. the
# gap from the body's FRONT EDGE to the array centre, NOT from the body centre.
# 200 + ~100 ≈ 300 mm clears the front-rotor reach (a ≈ 230 mm).
ARRAY_GAP_FROM_BODY_EDGE = 0.20  # m: measured front-edge -> array centre (spec)
BODY_HALF_FORWARD = 0.10  # m: DJI Matrice 100 body centre -> front edge (estimate)
ARRAY_OFFSET_FORWARD = ARRAY_GAP_FROM_BODY_EDGE + BODY_HALF_FORWARD  # +X, body-centre frame
ARRAY_OFFSET_UP = 0.33  # m (+Z): above drone body, vertical
WHEELBASE = 0.650  # m: DJI Matrice 100 motor-to-motor diagonal
# rps-row order — michaels CSV "Motor:Speed:*" columns, first 4; rotor_positions
# below is in the SAME order.
ROTOR_ORDER = ("RFront", "LFront", "LBack", "RBack")


def get_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mic_positions (8, 3), rotor_positions (4, 3))`` in metres.

    Microphones: 8 evenly-spaced points on a HORIZONTAL ring (radius 82.5 mm)
    in the X-Y plane (normal +Z), numbered counter-clockwise starting 22.5°
    left of top, ring centre ``ARRAY_OFFSET_FORWARD`` forward and 330 mm above
    the body. Rotors: X-quad at body height, ordered as ``ROTOR_ORDER``.
    """
    theta = np.deg2rad(112.5 + 45.0 * np.arange(N_MICS))  # mic 1 at 112.5°, CCW
    fwd = MIC_ARRAY_RADIUS * np.sin(theta)  # image-up component -> +X (forward)
    lat = MIC_ARRAY_RADIUS * np.cos(theta)  # image-right component -> -Y (lateral)
    mic_positions = np.stack(
        [ARRAY_OFFSET_FORWARD + fwd, -lat, np.full(N_MICS, ARRAY_OFFSET_UP)], axis=-1
    )  # (8, 3)

    a = (WHEELBASE / 2.0) * np.cos(np.deg2rad(45.0))
    rotor_positions = np.array(
        [
            [+a, -a, 0.0],  # RFront
            [+a, +a, 0.0],  # LFront
            [-a, +a, 0.0],  # LBack
            [-a, -a, 0.0],  # RBack
        ]
    )
    return mic_positions, rotor_positions


# ─── Alignment + loading ──────────────────────────────────────────────────────


def load_raw_aligned(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float = 0.0,
    time_dilation: float = 1.0,
    sr: int | None = None,
    rps_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load + align one recording — port of the notebook function.

    ``rps_scale`` defaults to the recording's calibrated
    :data:`MICHAELS_RPS_SCALE` entry (``None`` -> looked up from the CSV stem);
    pass an explicit float (``1.0``) to score a hypothesis against the
    uncalibrated telemetry.

    Returns ``(wav (C, N) at sr, ts (M,) aligned motor timestamps, ms (4, M)
    motor speeds in rev/s (calibrated), sr)``.
    """
    scale = rps_scale_for(csv_path) if rps_scale is None else float(rps_scale)
    wav, sample_rate = lr.load(str(wav_path), sr=sr, mono=False)
    if len(wav.shape) == 1:
        wav = wav[None, :]

    csv = pd.read_csv(csv_path)
    ms_cols = [c for c in csv.columns if "Motor" in c][:4]
    t_col = "Clock:offsetTime"
    small_csv = csv[[t_col] + ms_cols]

    wav_duration = wav.shape[-1] / sample_rate
    cut_csv = small_csv[
        (small_csv[t_col] >= time_offset) & (small_csv[t_col] <= wav_duration + time_offset)
    ]
    ts = np.asarray(cut_csv[t_col], dtype=np.float64)

    if ts[0] > time_offset:
        wav = wav[:, int((ts[0] - time_offset) * sample_rate) :]
        time_offset = ts[0]
        wav_duration = wav.shape[-1] / sample_rate

    if ts[-1] < wav_duration + time_offset:
        wav = wav[:, : int((ts[-1] - ts[0]) * sample_rate)]

    jump_idx = np.argmax(np.diff(ts))
    ts[0 : jump_idx + 2] = np.linspace(ts[0], ts[jump_idx + 2], jump_idx + 2)
    ts *= time_dilation

    ms = np.asarray(cut_csv[ms_cols], dtype=np.float64).T / 60 * scale
    return wav, ts, ms, int(sample_rate)


# ─── Builder ──────────────────────────────────────────────────────────────────

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
    """The exact row cut + timestamp fix of :func:`load_raw_aligned`, applied to
    the *whole* CSV. Returns (cut rows, aligned stamps, full csv)."""
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


def build_frame(
    raw_root: Path, wav_rel: str, csv_rel: str, time_offset: float, time_dilation: float
) -> td.Frame:
    """One Michael's recording -> rich Frame (realigned audio + rps + full CSV)."""
    wav_path, csv_path = Path(raw_root) / wav_rel, Path(raw_root) / csv_rel
    # Audio + canonical rps exactly as load_raw_aligned builds them, at the
    # native sample rate (sr=None -> no resampling).
    wav, ts_rps, ms, sample_rate = load_raw_aligned(
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
        # The measured rev/s calibration baked into `rps` (2026-07-31). Frames
        # published before that date carry 1.0 (uncalibrated) labels; this field
        # is how a consumer tells the two apart.
        "rps_scale": rps_scale_for(csv_path),
        "sample_rate": int(sample_rate),
        "n_csv_rows": int(len(full_csv)),
        "n_csv_rows_aligned": int(len(cut)),
        "dat_attributes": attributes,
        "provenance": {
            "builder": "data_processing.sources.michaels.build_frame",
            "alignment": (
                "CSV rows cut to the audio window, leading timestamp gap "
                "linearly interpolated, stamps scaled by time_dilation, audio "
                "cropped to the covered span (native sr)"
            ),
            "telemetry": (
                "numeric CSV columns grouped per sensor block as (channel, time) "
                "Series labelled with the original column names; bool/string "
                "columns as one (time,) Series each; all-NaN rows dropped per "
                "block (DatCon logs sensors at different rates)"
            ),
            "calibration": (
                "time_offset/time_dilation/rps_scale are MEASURED constants "
                "(MICHAELS_FILES + MICHAELS_RPS_SCALE, 2026-07-31): the "
                "audio-optimal telemetry lag was scanned per 16 s cruise window "
                "with the label-free VK reconstruction residual and regressed on "
                "window time (dilation, R^2 0.94/0.92, residual RMS 2.9/4.5 ms vs "
                "12.0/16.2 ms for a constant lag), and the rev/s labels carry a "
                "multiplicative correction (additive-vs-multiplicative was a "
                "statistical tie, resolved for multiplicative so the correction "
                "vanishes at rps -> 0). The canonical `rps` entry is CORRECTED; "
                "the `motor_speed` block holds the raw uncalibrated "
                "`Motor:Speed:*` CSV columns (RPM). See "
                "docs/experiments/rps-refine-precision.md sec. WP13."
            ),
        },
    }
    mic_pos, rotor_pos = get_geometry()
    return make_recording_frame(tracks, meta=meta, mic_pos=mic_pos, rotor_pos=rotor_pos)


def resolve_raw_root(data_root: str | Path | None = None) -> Path:
    """The tree ``MICHAELS_FILES``' relative paths resolve against.

    ``None`` -> the dload raw pin (``sources.raw_root("michaels")``). A given
    root may be the ``recording_with_motor_speed`` tree itself, a ``dload:``
    URI, or an enclosing checkout ``data/`` dir — the nested tree is descended
    into when present, so the historical ``data/``-relative call convention
    keeps working. Mirrors :func:`sources.dregon._dregon_dir`.
    """
    if data_root is None:
        from data_processing import sources

        return Path(sources.raw_root("michaels"))
    from data_processing.streams import resolve_source

    root = Path(resolve_source(data_root))
    nested = root / "recording_with_motor_speed"
    return nested if nested.is_dir() else root


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(recording_id, frame)`` for each aligned recording."""
    raw_dir = resolve_raw_root(raw_dir)
    for wav_rel, csv_rel, time_offset, time_dilation in MICHAELS_FILES:
        rid = Path(csv_rel).stem
        yield rid, build_frame(raw_dir, wav_rel, csv_rel, time_offset, time_dilation)


def load_michaels_timeframes(
    data_root: str | Path | None = None,
    sr: int | None = 16000,
) -> list[td.Frame]:
    """Load FLY124/FLY125 as aligned recording ``td.Frame``s at sample rate
    ``sr`` (or native rate when ``None``). ``data_root`` is resolved by
    :func:`resolve_raw_root` (default: the dload raw pin).

    Each frame holds 8-channel ``audio`` + ``rps`` (aligned, rev/s) + ``meta``.
    """
    root = resolve_raw_root(data_root)
    frames: list[td.Frame] = []
    for wav_rel, csv_rel, time_offset, time_dilation in MICHAELS_FILES:
        wav, ts, ms, raw_sr = load_raw_aligned(
            root / wav_rel,
            root / csv_rel,
            time_offset=time_offset,
            time_dilation=time_dilation,
            sr=sr,
        )
        meta = {"recording_id": Path(csv_rel).stem}
        mic_pos, rotor_pos = get_geometry()
        frames.append(
            make_recording_frame(
                {
                    "audio": td.uniform(wav, raw_sr, dims=("mic", "time"), t_start=0.0),
                    "rps": td.events(ts, ms, dims=("rotor", "time"), t_start=0.0),
                },
                meta=meta,
                mic_pos=mic_pos,
                rotor_pos=rotor_pos,
            )
        )
    return frames


# ─── Registry provenance ──────────────────────────────────────────────────────
# (entry assembled in sources/__init__.py)

PROVENANCE: dict[str, Any] = {
    "citation": "Michael's DJI Matrice 100 rig recordings (project-local, unpublished).",
    "description": (
        "FLY124/FLY125 as rich td.Frames: realigned native-sr 8ch audio, rps "
        "(rev/s), and every DJI flight-log CSV column as aligned series grouped "
        "per sensor block. 103 further recordings in `new-drone-noises` have no "
        "alignment constants and exist raw-only."
    ),
    "sample_rate": 44100,
    "channels": 8,
}
