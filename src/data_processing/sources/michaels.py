"""Michael's drone-noise source — uniform registry entry.

8-channel DJI Matrice 100 in-flight recordings + flight-controller CSV logs
(per-motor rotation speed in RPM), aligned via per-file ``time_offset`` /
``time_dilation`` constants and value-calibrated via ``MICHAELS_RPS_SCALE``.
All three are MEASURED, not hand-tuned — see the block comment above
``MICHAELS_FILES`` and ``docs/experiments/rps-refine-precision.md`` §§ WP13/WP14.

Raw layout (dload dataset ``recording_with_motor_speed``):
  - ``recording_1/124.wav`` + ``recording_1/FLY124.csv``
  - ``recording_2/125.wav`` + ``recording_2/FLY125.csv``

:func:`build` yields one rich ``tdframe-v1`` Frame per recording: native-sr
8-ch audio realigned to the flight-log clock (window cut + leading-gap fix +
``time_offset``/``time_dilation``), the canonical ``rps`` track (rev/s,
carrying the measured ``MICHAELS_RPS_SCALE`` calibration), plus
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

from data_processing.frames import audio_series, make_recording_frame

# ── Measured alignment / label calibration (2026-07-31) ─────────────────────
#
# Referee: the LABEL-FREE Vold-Kalman reconstruction residual
# ||x - x_hat||/||x|| (k = 1..30, `scripts/michaels_calib/calib.py:RECON_CFG`,
# identical to `rps_refine_lab.RECON_CFG`) of the harmonic model built ALONG a
# candidate telemetry trajectory, scored on contiguous 16 s windows of the
# frozen beat-VK window protocol. Only the TELEMETRY is shifted/scaled; our own
# blind RPS estimate is never consulted. Sweep: `scripts/michaels_calib/`.
#
# TIMING — both recordings show a clock DILATION error (the audio-optimal lag
# drifts linearly with time), not a constant offset. Folding lag(t) = a + b·t
# into the loader's parameterisation gives
#     time_dilation_new = time_dilation_old / (1 - b)
#     time_offset_new   = time_offset_old  - a / (1 - b)
# FLY124: 4 cruise windows, b = +0.65356 ms/s, R^2 0.942, residual RMS 2.90 ms
# vs 12.04 ms constant-lag. FLY125: 9 windows, b = +0.37656 ms/s, R^2 0.923,
# residual RMS 4.49 ms vs 16.19 ms.
#
# VALUE — the logged speeds are LOW by ~0.55-0.65 rev/s at cruise. Additive vs
# multiplicative is statistically UNRESOLVED; we ship the MULTIPLICATIVE form
# on physical grounds: these frames cover the WHOLE recording including warm-up
# and ground idle, where an additive +0.6 rev/s would corrupt a near-stationary
# rotor's label while a scale correctly vanishes as rps -> 0.
#
# MAGNITUDE (refit, WP14) — one global fit per recording over ALL 13 cruise
# windows, non-twin rotors only: g = 0.698 % ± 0.069 (FLY124), 0.706 % ± 0.034
# (FLY125). The two agree to 0.008 pp. Per-rotor constants are NOT identifiable.
#
# DEGENERACY — the global gain is not separable from a sample-clock error, so
# it is a *label-for-this-audio* correction, not proof the ESC is miscalibrated.
#
# Full write-up: `docs/experiments/rps-refine-precision.md` §§ WP13, WP14.
#
# (wav_path, csv_path, time_offset_sec, time_dilation) — paths are relative to
# the raw root (the `recording_with_motor_speed` tree).
MICHAELS_FILES = [
    ("recording_1/124.wav", "recording_1/FLY124.csv", -20.753813, 1.001654644),
    ("recording_2/125.wav", "recording_2/FLY125.csv", -26.337849, 1.005178509),
]

# ── Held-out TEST recordings (raw dataset ``new-drone-noises``) ─────────────
#
# FLY103 / FLY108 are the only two further recordings of the same rig and the
# same airframe (DatCon attribute ``ACType|M100``). They are **MONO** — one
# microphone, not the 8-channel ring — and their flight logs are ~2x longer
# than the audio, so the audio is a sub-window of the log.
#
# They use the ANCHORED alignment path (``anchor=True`` of
# :func:`load_raw_aligned`): ``time_offset`` is the flight-log clock time of
# the audio's first sample and the telemetry stamps become
# ``(t_log - time_offset) * time_dilation`` — the audio clock, exactly. The
# legacy path of FLY124/FLY125 instead crops the audio head and then uses the
# log stamps as-is, which only agrees with the audio clock because those two
# logs happen to start ~0.15 s before their audio. Here the logs start 50-78 s
# early, so that identity does not hold and the anchored path is mandatory.
#
# (wav_path, csv_path, time_offset_sec, time_dilation) — paths are relative to
# the raw root (the `new-drone-noises` tree).
# COARSE constants (`scripts/michaels_calib/coarse_align.py`, 2026-08-24):
# the comb-score line fit over 5 segments, residual RMS 1.9 ms (FLY103) and
# 3.4 ms (FLY108), whole-recording score +5.6 % / +6.1 % over the best single
# offset at dilation 1. Both recordings agree on a 1.19 % clock dilation to 5
# decimal places, which is 7x FLY125's and 20x FLY124's.
# PENDING CALIBRATION: `scripts/michaels_calib/fit_new.py` (the VK
# reconstruction residual) has not refined these, and MICHAELS_RPS_SCALE has
# no entry for either recording yet, so the rev/s labels are UNCALIBRATED.
# The coarse rev/s scale hints are 1.0037 (FLY103) and 1.0053 (FLY108).
MICHAELS_TEST_FILES = [
    ("103_2.wav", "FLY103.csv", -0.8915, 1.0119078),
    ("108_2.wav", "FLY108.csv", -0.3956, 1.0119673),
]

#: Per-recording MULTIPLICATIVE rev/s correction, keyed by CSV stem (the
#: recording id). Applied in :func:`load_raw_aligned`, so every consumer of
#: ``rps`` gets calibrated labels. Recordings without a measured constant (the
#: 103 unaligned `new-drone-noises` logs) fall back to 1.0. Both values are the
#: WP14 13-window global refit; they supersede 1.00839 / 1.00690 (WP13, 2-4
#: windows, twin-contaminated).
MICHAELS_RPS_SCALE: dict[str, float] = {
    "FLY124": 1.00698,  # g = 0.698 % ± 0.069 -> +0.558 rev/s at 80 rev/s
    "FLY125": 1.00706,  # g = 0.706 % ± 0.034 -> +0.565 rev/s at 80 rev/s
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

_T_COL = "Clock:offsetTime"
_ATTR_COL = "Attribute|Value"


def read_motor_speeds(
    csv_path: str | Path, rps_scale: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The raw DatCon motor-speed table: ``(t (M,), rps (4, M))``.

    ``t`` is the flight-log clock (``Clock:offsetTime``, seconds, NOT aligned
    to the audio); ``rps`` is the first four ``Motor:*`` columns converted from
    RPM to rev/s and multiplied by ``rps_scale`` (``None`` -> the recording's
    :data:`MICHAELS_RPS_SCALE` entry, 1.0 when it has none). This is the one
    copy of the column selection + unit conversion; every caller that needs
    rotor speeds off a Michael's CSV uses it.
    """
    scale = rps_scale_for(csv_path) if rps_scale is None else float(rps_scale)
    csv = pd.read_csv(csv_path, low_memory=False)
    ms_cols = [c for c in csv.columns if "Motor" in c][:4]
    t = np.asarray(csv[_T_COL], dtype=np.float64)
    return t, np.asarray(csv[ms_cols], dtype=np.float64).T / 60 * scale


def _anchored_stamps(
    t_log: np.ndarray, time_offset: float, time_dilation: float, wav_duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(row mask, stamps on the audio clock)`` for the anchored path.

    The audio is never cropped: ``time_offset`` is the log-clock time of its
    first sample, so ``(t_log - time_offset) * time_dilation`` is audio time.
    The row cut is ``wav_duration / time_dilation`` of LOG time, which is
    exactly ``wav_duration`` of audio time — so the stamps span the audio and
    nothing beyond it (an overhang would let an in-flight-window search select
    telemetry the audio does not cover).
    """
    mask = (t_log >= time_offset) & (t_log <= time_offset + wav_duration / time_dilation)
    return mask, (np.asarray(t_log[mask], dtype=np.float64) - time_offset) * time_dilation


def load_raw_aligned(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float = 0.0,
    time_dilation: float = 1.0,
    sr: int | None = None,
    rps_scale: float | None = None,
    anchor: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load + align one recording — port of the notebook function.

    ``rps_scale`` defaults to the recording's calibrated
    :data:`MICHAELS_RPS_SCALE` entry (``None`` -> looked up from the CSV stem);
        # The measured rev/s calibration baked into `rps` (2026-07-31). Frames
        # published before that date carry 1.0 (uncalibrated) labels; this field
        # is how a consumer tells the two apart.
    pass an explicit float (e.g. ``1.0``) to score a hypothesis against the
    uncalibrated telemetry.

    ``anchor`` selects the ANCHORED alignment (the held-out
    :data:`MICHAELS_TEST_FILES` recordings): the audio is kept whole and the
    stamps become ``(t_log - time_offset) * time_dilation``, so stamp 0 is the
    audio's first sample. The default (``False``) is the legacy FLY124/FLY125
    path: crop the audio head by ``ts[0] - time_offset``, interpolate the
    leading log gap, and return the log stamps scaled by ``time_dilation``.

    Returns ``(wav (C, N) at sr, ts (M,) aligned motor timestamps, ms (4, M)
    motor speeds in rev/s (calibrated), sr)``.
    """
    wav, sample_rate = lr.load(str(wav_path), sr=sr, mono=False)
    if len(wav.shape) == 1:
        wav = wav[None, :]

    t_log, ms_all = read_motor_speeds(csv_path, rps_scale)
    wav_duration = wav.shape[-1] / sample_rate

    if anchor:
        mask, ts = _anchored_stamps(t_log, time_offset, time_dilation, wav_duration)
        ms = ms_all[:, mask]
        if not np.isfinite(ms).all():
            raise ValueError(f"{Path(csv_path).name}: NaN motor speeds inside the audio window")
        return wav, ts, ms, int(sample_rate)

    mask = (t_log >= time_offset) & (t_log <= wav_duration + time_offset)
    ts = np.asarray(t_log[mask], dtype=np.float64)
    ms = ms_all[:, mask]

    if ts[0] > time_offset:
        wav = wav[:, int((ts[0] - time_offset) * sample_rate) :]
        time_offset = ts[0]
        wav_duration = wav.shape[-1] / sample_rate

    if ts[-1] < wav_duration + time_offset:
        wav = wav[:, : int((ts[-1] - ts[0]) * sample_rate)]

    jump_idx = np.argmax(np.diff(ts))
    ts[0 : jump_idx + 2] = np.linspace(ts[0], ts[jump_idx + 2], jump_idx + 2)
    ts *= time_dilation
    return wav, ts, ms, int(sample_rate)


# ─── Builder ──────────────────────────────────────────────────────────────────

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
    csv_path: Path,
    time_offset: float,
    time_dilation: float,
    wav_duration: float,
    anchor: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """The exact row cut + timestamp fix of :func:`load_raw_aligned`, applied to
    the *whole* CSV. Returns (cut rows, aligned stamps, full csv)."""
    csv = pd.read_csv(csv_path, low_memory=False)
    t = np.asarray(csv[_T_COL], dtype=np.float64)
    if anchor:
        mask_a, ts = _anchored_stamps(t, time_offset, time_dilation, wav_duration)
        cut = cast(pd.DataFrame, csv[mask_a]).reset_index(drop=True)
        return cut, ts, csv
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


#: Provenance note of the held-out TEST recordings. Kept next to
#: :data:`MICHAELS_TEST_FILES`, whose constants it describes.
_TEST_CALIBRATION_NOTE = (
    "Held-out TEST recordings (FLY103/FLY108, raw tree new-drone-noises), "
    "calibrated 2026-08 with the same two-stage procedure as FLY124/FLY125: "
    "scripts/michaels_calib/coarse_align.py seeds the log-clock offset from "
    "the telemetry-predicted comb in one STFT (the flight log is ~2x longer "
    "than the audio, so a seed is mandatory), then "
    "scripts/michaels_calib/fit_new.py scans the audio-optimal telemetry lag "
    "per 16 s cruise window with the label-free VK reconstruction residual, "
    "regresses it on window time (dilation) and fits one global multiplicative "
    "rev/s scale on the non-twin rotors (WP14 convention). These recordings "
    "are MONO, so the frame carries no mic_pos. See "
    "docs/experiments/rps-refine-precision.md WP13/WP14 for the procedure."
)


def _wav_duration_s(wav_path: Path) -> float:
    """Original (pre-crop) wav duration — what the loader's CSV window uses."""
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def build_frame(
    raw_root: Path,
    wav_rel: str,
    csv_rel: str,
    time_offset: float,
    time_dilation: float,
    anchor: bool = False,
) -> td.Frame:
    """One Michael's recording -> rich Frame (realigned audio + rps + full CSV).

    ``anchor`` selects the anchored alignment of the held-out
    :data:`MICHAELS_TEST_FILES` recordings (see :func:`load_raw_aligned`).
    Those are mono, so their ``audio`` entry is a ``(time,)`` Series and the
    frame carries no ``mic_pos`` — one microphone is not the 8-mic ring, and
    its position on the airframe was never recorded.
    """
    wav_path, csv_path = Path(raw_root) / wav_rel, Path(raw_root) / csv_rel
    # Audio + canonical rps exactly as load_raw_aligned builds them, at the
    # native sample rate (sr=None -> no resampling). The rev/s calibration
    # (MICHAELS_RPS_SCALE) is applied inside the loader, so the canonical `rps`
    # entry is CORRECTED while the `motor_speed` block below (the raw
    # `Motor:Speed:*` CSV columns, in RPM) stays untouched — the same fixed/raw
    # pairing DREGON gets with motors_command/motors_command_raw.
    rps_scale = rps_scale_for(csv_path)
    wav, ts_rps, ms, sample_rate = load_raw_aligned(
        wav_path,
        csv_path,
        time_offset=time_offset,
        time_dilation=time_dilation,
        sr=None,
        anchor=anchor,
    )
    tracks: dict[str, td.Series] = {
        "audio": audio_series(wav, sample_rate)
        if wav.shape[0] == 1
        else td.uniform(wav, sample_rate, dims=("mic", "time")),
        "rps": td.events(ts_rps, ms, dims=("rotor", "time")),
    }
    n_mics = int(wav.shape[0])
    del wav

    cut, ts, full_csv = _aligned_csv(
        csv_path,
        time_offset,
        time_dilation,
        wav_duration=_wav_duration_s(wav_path),
        anchor=anchor,
    )
    # The telemetry row cut must land on the very same rows (hence stamps) the
    # loader's rps came from — otherwise the entries are not aligned.
    if not np.array_equal(ts, ts_rps):
        raise AssertionError(f"{csv_rel}: aligned CSV stamps diverge from the loader's rps stamps")
    tracks.update(_telemetry_entries(cut, ts))

    attributes = [str(v) for v in full_csv[_ATTR_COL].dropna()] if _ATTR_COL in full_csv else []
    meta = {
        "recording_id": Path(csv_rel).stem,  # FLY124 / FLY125 / FLY103 / FLY108
        "wav": wav_rel,
        "csv": csv_rel,
        "time_offset": float(time_offset),
        "time_dilation": float(time_dilation),
        "rps_scale": float(rps_scale),
        "sample_rate": int(sample_rate),
        "n_channels": n_mics,
        "n_csv_rows": int(len(full_csv)),
        "n_csv_rows_aligned": int(len(cut)),
        "dat_attributes": attributes,
        "provenance": {
            "builder": "data_processing.sources.michaels.build_frame",
            "alignment": (
                "CSV rows cut to the audio window, stamps re-anchored to the "
                "audio clock as (t_log - time_offset) * time_dilation; the "
                "audio is kept whole (native sr)"
                if anchor
                else "CSV rows cut to the audio window, leading timestamp gap "
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
                _TEST_CALIBRATION_NOTE
                if anchor
                else "time_offset/time_dilation/rps_scale are MEASURED constants "
                "(sources.michaels.MICHAELS_FILES + MICHAELS_RPS_SCALE, "
                "2026-07-31): the audio-optimal telemetry lag was scanned per "
                "16 s cruise window with the label-free VK reconstruction "
                "residual and regressed on window time (dilation, R^2 "
                "0.94/0.92, residual RMS 2.9/4.5 ms vs 12.0/16.2 ms for a "
                "constant lag), and the rev/s labels carry a multiplicative "
                "correction (+0.558/+0.565 rev/s at 80 rev/s; "
                "additive-vs-multiplicative was a statistical tie, resolved "
                "for multiplicative so the correction vanishes at rps -> 0). "
                "The canonical `rps` entry is CORRECTED; the `motor_speed` "
                "block holds the raw uncalibrated `Motor:Speed:*` CSV columns "
                "(RPM). See docs/experiments/rps-refine-precision.md WP13/WP14."
            ),
        },
    }
    mic_pos, rotor_pos = get_geometry()
    return make_recording_frame(
        tracks,
        meta=meta,
        mic_pos=None if n_mics == 1 else mic_pos,
        rotor_pos=rotor_pos,
    )


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


def resolve_test_raw_root(data_root: str | Path | None = None) -> Path:
    """The tree ``MICHAELS_TEST_FILES``' relative paths resolve against.

    The held-out recordings live in a **different** raw dataset than
    FLY124/FLY125: ``new-drone-noises``. ``None`` -> the dload raw pin. A given
    root may be that tree itself, a ``dload:`` URI, or an enclosing checkout
    ``data/`` dir. Mirrors :func:`resolve_raw_root`.
    """
    if data_root is None:
        from data_processing import sources

        return Path(sources.raw_root("new-drone-noises"))
    from data_processing.streams import resolve_source

    root = Path(resolve_source(data_root))
    nested = root / "new-drone-noises"
    return nested if nested.is_dir() else root


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(recording_id, frame)`` for each aligned recording."""
    raw_dir = resolve_raw_root(raw_dir)
    for wav_rel, csv_rel, time_offset, time_dilation in MICHAELS_FILES:
        rid = Path(csv_rel).stem
        yield rid, build_frame(raw_dir, wav_rel, csv_rel, time_offset, time_dilation)


def build_test(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(recording_id, frame)`` for each held-out TEST recording.

    Same rich layout as :func:`build` (rps + every CSV column as an aligned
    series block + provenance meta), on the ``new-drone-noises`` raw tree and
    through the anchored alignment path. The audio is MONO and native 48 kHz.
    """
    raw_dir = resolve_test_raw_root(raw_dir)
    for wav_rel, csv_rel, time_offset, time_dilation in MICHAELS_TEST_FILES:
        rid = Path(csv_rel).stem
        yield rid, build_frame(raw_dir, wav_rel, csv_rel, time_offset, time_dilation, anchor=True)


def load_michaels_timeframe(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float = 0.0,
    time_dilation: float = 1.0,
    sr: int | None = 16000,
    recording_id: str | None = None,
    rps_scale: float | None = None,
) -> td.Frame:
    """Load one Michael's recording as an aligned ``td.Frame``.

    The frame holds an 8-channel ``audio`` Series (dims ``("mic", "time")``)
    and an ``rps`` Series (dims ``("rotor", "time")``): the audio is anchored
    at t=0, the RPS timestamps are aligned to it via ``time_offset`` /
    ``time_dilation``, and the speeds carry the calibrated ``rps_scale``
    (``None`` -> the recording's :data:`MICHAELS_RPS_SCALE` entry).
    """
    wav, ts, ms, raw_sr = load_raw_aligned(
        wav_path,
        csv_path,
        time_offset=time_offset,
        time_dilation=time_dilation,
        sr=sr,
        rps_scale=rps_scale,
    )
    mic_pos, rotor_pos = get_geometry()
    return make_recording_frame(
        {
            "audio": td.uniform(wav, raw_sr, dims=("mic", "time"), t_start=0.0),
            "rps": td.events(ts, ms, dims=("rotor", "time"), t_start=0.0),
        },
        meta={"recording_id": recording_id or Path(csv_path).stem},
        mic_pos=mic_pos,
        rotor_pos=rotor_pos,
    )


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
    return [
        load_michaels_timeframe(
            root / wav_rel,
            root / csv_rel,
            time_offset=time_offset,
            time_dilation=time_dilation,
            sr=sr,
        )
        for wav_rel, csv_rel, time_offset, time_dilation in MICHAELS_FILES
    ]


# ─── Registry provenance ──────────────────────────────────────────────────────
# (entry assembled in sources/__init__.py)

PROVENANCE: dict[str, Any] = {
    "citation": "Michael's DJI Matrice 100 rig recordings (project-local, unpublished).",
    "description": (
        "FLY124/FLY125 as rich td.Frames: realigned native-sr 8ch audio, "
        "calibrated rps (rev/s), and every DJI flight-log CSV column as aligned series grouped "
        "per sensor block. The two further recordings in `new-drone-noises` "
        "(FLY103/FLY108, mono) are the held-out TEST set — see the "
        "`michaels-test` entry."
    ),
    "sample_rate": 44100,
    "channels": 8,
}

TEST_PROVENANCE: dict[str, Any] = {
    "citation": "Michael's DJI Matrice 100 rig recordings (project-local, unpublished).",
    "description": (
        "HELD-OUT TEST recordings FLY103/FLY108 (raw tree `new-drone-noises`), "
        "same airframe and rig as FLY124/FLY125 but MONO: one microphone, "
        "native 48 kHz, 106.5 s and 99.4 s. Rich td.Frames with the same layout "
        "as `michaels` — audio + calibrated rps (rev/s) + every DJI flight-log "
        "CSV column as aligned series grouped per sensor block — minus mic_pos, "
        "which a single microphone does not have. Calibrated 2026-08 "
        "(coarse comb alignment + the WP13/WP14 VK procedure); their flight "
        "logs are ~2x longer than the audio, so they use the anchored "
        "alignment path. Reserved as a TEST set: do not train on them."
    ),
    "sample_rate": 48000,
    "channels": 1,
}
