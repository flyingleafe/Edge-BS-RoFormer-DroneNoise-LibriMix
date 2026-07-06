"""
External Drone Recording Loader — tdseries-native.

Loads drone noise recordings with DJI flight-log CSVs (FLY*.csv) and
multi-channel WAV audio into ``td.Frame`` objects, making them compatible
with the DREGON data processing pipeline.

Expected directory layout::

    data_root/
        recording_1/
            124.wav          # 8-ch, 44100 Hz
            FLY124.csv       # DJI flight log (230 columns)
        recording_2/
            125.wav
            FLY125.csv

CSV columns used
────────────────
  - Clock:offsetTime  (seconds from flight-controller start; 0 ≈ audio start)
  - Motor:Speed:RFront / LFront / LBack / RBack  (RPM → converted to RPS)
  - IMU_ATTI(0):accelX / accelY / accelZ
  - IMU_ATTI(0):gyroX  / gyroY  / gyroZ
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import tdseries as td

from data_processing.frames import make_recording_frame

# Column names in DJI flight-log CSVs
_TIME_COL = "Clock:offsetTime"
_MOTOR_SPEED_COLS = [
    "Motor:Speed:RFront",
    "Motor:Speed:LFront",
    "Motor:Speed:LBack",
    "Motor:Speed:RBack",
]
_ACCEL_COLS = [
    "IMU_ATTI(0):accelX",
    "IMU_ATTI(0):accelY",
    "IMU_ATTI(0):accelZ",
]
_GYRO_COLS = [
    "IMU_ATTI(0):gyroX",
    "IMU_ATTI(0):gyroY",
    "IMU_ATTI(0):gyroZ",
]


def _find_col_indices(header: list[str], names: list[str]) -> list[int]:
    lookup = {name: i for i, name in enumerate(header)}
    indices = []
    for name in names:
        if name not in lookup:
            raise KeyError(f"Column '{name}' not found in CSV header")
        indices.append(lookup[name])
    return indices


def _parse_dji_csv(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Parse a DJI flight-log CSV, return aligned arrays.

    Returns dict with keys: ``time``, ``motor_rps``, ``accel``, ``gyro``.
    """
    csv_path = Path(csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        time_idx = _find_col_indices(header, [_TIME_COL])[0]
        speed_idxs = _find_col_indices(header, _MOTOR_SPEED_COLS)

        try:
            accel_idxs = _find_col_indices(header, _ACCEL_COLS)
            gyro_idxs = _find_col_indices(header, _GYRO_COLS)
            has_imu = True
        except KeyError:
            has_imu = False

        times, speeds, accels, gyros = [], [], [], []
        for row in reader:
            try:
                t = float(row[time_idx])
                spd = [float(row[i]) if row[i] else float("nan") for i in speed_idxs]
                if any(np.isnan(spd)):
                    continue
                times.append(t)
                speeds.append(spd)
                if has_imu:
                    a = [float(row[i]) if row[i] else 0.0 for i in accel_idxs]
                    g = [float(row[i]) if row[i] else 0.0 for i in gyro_idxs]
                    accels.append(a)
                    gyros.append(g)
            except (ValueError, IndexError):
                continue

    result: dict[str, np.ndarray] = {
        "time": np.array(times, dtype=np.float64),
        "motor_rps": np.array(speeds, dtype=np.float32) / 60.0,  # RPM → RPS
    }
    if has_imu and accels:
        result["accel"] = np.array(accels, dtype=np.float32)
        result["gyro"] = np.array(gyros, dtype=np.float32)
    return result


def load_external_timeframe(
    wav_path: str | Path,
    csv_path: str | Path,
    recording_id: str | None = None,
) -> td.Frame:
    """Load an external drone recording as a ``td.Frame``.

    Alignment assumption: ``Clock:offsetTime = 0`` → audio sample 0.

    Returns
    -------
    td.Frame
        Entries: ``"audio"``, ``"motors_measured"``, and optionally
        ``"imu_accel"``, ``"imu_gyro"``, plus ``"meta"`` (``recording_id``,
        ``split``, ``flight_type``, ``sample_rate``).
    """
    wav_path = Path(wav_path)
    csv_path = Path(csv_path)
    if recording_id is None:
        recording_id = wav_path.stem

    # ── audio ──────────────────────────────────────────────────────────
    audio, sr = sf.read(str(wav_path))  # (N,) or (N, n_ch)
    audio = audio[np.newaxis, :] if audio.ndim == 1 else audio.T  # (n_ch, N)

    t_start = 0.0

    tracks: dict[str, td.Series] = {
        "audio": td.uniform(
            audio.astype(np.float32),
            sr,
            dims=("mic", "time"),
            t_start=t_start,
        ),
    }

    # ── CSV telemetry ──────────────────────────────────────────────────
    csv_data = _parse_dji_csv(csv_path)
    csv_time = csv_data["time"]  # may start negative — events handle that
    motor_rps = csv_data["motor_rps"]  # (M, 4)

    # values are time-last (..., M).
    tracks["motors_measured"] = td.events(
        csv_time,
        motor_rps.T,
        dims=("rotor", "time"),
        t_start=t_start,
    )

    # ── IMU (optional) ─────────────────────────────────────────────────
    if "accel" in csv_data:
        tracks["imu_accel"] = td.events(
            csv_time,
            csv_data["accel"].T,
            dims=(None, "time"),
            t_start=t_start,
        )
        tracks["imu_gyro"] = td.events(
            csv_time,
            csv_data["gyro"].T,
            dims=(None, "time"),
            t_start=t_start,
        )

    # ── meta ───────────────────────────────────────────────────────────
    meta = {
        "recording_id": recording_id,
        "split": "in_flight_noise",
        "flight_type": "free-flight",
        "sample_rate": int(sr),
    }

    return make_recording_frame(tracks, meta=meta)


def discover_external_recordings(
    data_root: str | Path,
) -> list[tuple[Path, Path, str]]:
    """Discover (wav_path, csv_path, recording_id) triples under *data_root*.

    Searches for directories containing exactly one .wav and one FLY*.csv.
    """
    data_root = Path(data_root)
    results = []
    for d in sorted(data_root.rglob("*")):
        if not d.is_dir():
            continue
        wavs = list(d.glob("*.wav"))
        csvs = list(d.glob("FLY*.csv"))
        if len(wavs) == 1 and len(csvs) == 1:
            rec_id = f"ext_{d.name}_{wavs[0].stem}"
            results.append((wavs[0], csvs[0], rec_id))
    return results


def load_all_external_timeframes(
    data_root: str | Path,
) -> list[td.Frame]:
    """Load every external recording found under *data_root* as a ``td.Frame``."""
    triples = discover_external_recordings(data_root)
    frames = []
    for wav_path, csv_path, rec_id in triples:
        tf = load_external_timeframe(wav_path, csv_path, recording_id=rec_id)
        frames.append(tf)
        audio_dur = tf["audio"].duration
        n_motor = tf["motors_measured"].dim_size("time") if "motors_measured" in tf else 0
        n_ch = tf["audio"].dim_size("mic")
        print(f"  Loaded {rec_id}: {audio_dur:.1f}s, {n_ch}ch, {n_motor} motor samples")
    return frames
