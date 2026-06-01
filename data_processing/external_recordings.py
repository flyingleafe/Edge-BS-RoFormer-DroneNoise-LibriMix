"""
External Drone Recording Loader

Loads drone noise recordings with DJI flight-log CSVs (FLY*.csv) and
multi-channel WAV audio into DREGONRecord objects, making them compatible
with the existing DREGON data processing pipeline.

Expected directory layout
─────────────────────────
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
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from data_processing.dregon import (
    DREGONRecord,
    DREGONSampleDict,
    IMUData,
    MotorData,
)

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
    """Map column names to indices; raise if any is missing."""
    lookup = {name: i for i, name in enumerate(header)}
    indices = []
    for name in names:
        if name not in lookup:
            raise KeyError(f"Column '{name}' not found in CSV header")
        indices.append(lookup[name])
    return indices


def parse_dji_csv(csv_path: str | Path) -> dict[str, np.ndarray]:
    """
    Parse a DJI flight-log CSV and return aligned arrays.

    Returns dict with keys:
        time       – (N,) seconds from offsetTime=0
        motor_rps  – (N, 4) rotor speeds in revolutions per second
        accel      – (N, 3) acceleration in m/s²  (or None)
        gyro       – (N, 3) angular velocity in rad/s (or None)
    """
    csv_path = Path(csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        time_idx = _find_col_indices(header, [_TIME_COL])[0]
        speed_idxs = _find_col_indices(header, _MOTOR_SPEED_COLS)

        # IMU columns are optional (may be absent or empty)
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


def load_external_recording(
    wav_path: str | Path,
    csv_path: str | Path,
    recording_id: str | None = None,
) -> DREGONRecord:
    """
    Load an external drone recording as a DREGONRecord.

    Alignment assumption: ``Clock:offsetTime = 0`` corresponds to audio
    sample 0.  Negative offset times are before the audio starts.

    Parameters
    ----------
    wav_path : path to multi-channel WAV
    csv_path : path to DJI flight-log CSV
    recording_id : identifier string (defaults to WAV stem)

    Returns
    -------
    DREGONRecord fully compatible with DREGON slicing / resampling API.
    """
    wav_path = Path(wav_path)
    csv_path = Path(csv_path)
    if recording_id is None:
        recording_id = wav_path.stem

    # ── audio ──────────────────────────────────────────────────────────
    audio, sr = sf.read(str(wav_path))  # (n_samples, n_channels) or (n_samples,)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    n_samples = audio.shape[0]

    # Synthetic timestamps: offsetTime=0 → sample 0
    audio_timestamps = np.arange(n_samples, dtype=np.float64) / sr

    # ── CSV telemetry ──────────────────────────────────────────────────
    csv_data = parse_dji_csv(csv_path)
    csv_time = csv_data["time"]  # may start negative
    motor_rps = csv_data["motor_rps"]  # (N, 4) already RPS

    # Build MotorData with timestamps relative to audio (== offsetTime)
    motors = MotorData(
        timestamps=csv_time,
        measured=motor_rps,
        command=motor_rps,  # no separate command channel in DJI logs
    )

    # ── IMU (optional) ─────────────────────────────────────────────────
    imu = None
    if "accel" in csv_data:
        imu = IMUData(
            timestamps=csv_time,
            angular_velocity=csv_data["gyro"],
            acceleration=csv_data["accel"],
        )

    # ── geometry (placeholder – no mic/rotor position data available) ──
    mic_positions = np.zeros((8, 3), dtype=np.float64)
    rotor_positions = np.zeros((4, 3), dtype=np.float64)

    return DREGONRecord(
        recording_id=recording_id,
        split="in_flight_noise",
        mic_positions=mic_positions,
        rotor_positions=rotor_positions,
        audio=audio,
        audio_timestamps=audio_timestamps,
        flight_type="free-flight",
        source_type=None,
        source_level=None,
        room=None,
        motor_id=None,
        motor_speed=None,
        sample_rate=sr,
        imu=imu,
        motors=motors,
        source_position=None,
    )


def discover_external_recordings(
    data_root: str | Path,
) -> list[tuple[Path, Path, str]]:
    """
    Discover (wav_path, csv_path, recording_id) triples under *data_root*.

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


def load_all_external_recordings(
    data_root: str | Path,
) -> list[DREGONRecord]:
    """Load every external recording found under *data_root*."""
    triples = discover_external_recordings(data_root)
    records = []
    for wav_path, csv_path, rec_id in triples:
        rec = load_external_recording(wav_path, csv_path, recording_id=rec_id)
        records.append(rec)
        print(
            f"  Loaded {rec_id}: {rec.duration:.1f}s, {rec.n_channels}ch, "
            f"{len(rec.motors)} motor samples"
        )
    return records
