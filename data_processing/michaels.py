"""
Michael's Dataset Loader — drone noise + rotor speed telemetry.

The dataset in `data/new-drone-noises/` consists of drone in-flight WAV
recordings plus DJI flight-controller CSV logs containing per-motor rotation
speeds. The CSV and WAV timelines are not aligned by default — a per-file
`time_offset` (in seconds) is needed, plus an interior valid range `[l_s, r_s]`
to clip out takeoff/landing and silent regions.

Alignment table (`MichaelsDataset.dataset_files`) is the empirical offsets from
the original legacy `drone_audition` repo. Files present in this repo:
  - 103_2.wav + FLY103.csv  (offset −0.94 s, valid 12.0–100.0 s)
  - 108_2.wav + FLY108.csv  (offset −0.40 s, valid  9.0–88.0 s)
The legacy repo also referenced 124.wav and 125.wav (offsets −20.63 and −26.27);
they are not shipped here but are kept in the table behind an existence check.

This module provides a `MichaelsRecord` class that is *duck-type compatible*
with `data_processing.dregon.DREGONRecord` for the subset of fields used by
the noise+RPS chunk extractor — i.e. `audio`, `audio_timestamps`, `motors`,
`sample_rate`, `recording_id`, `slice_by_time(start_sec, end_sec)`. So the
same `extract_noise_chunk_with_rps` function works on both.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .dregon import MotorData


# ---------------------------------------------------------------------------
# File registry (empirical offsets from legacy repo)
# ---------------------------------------------------------------------------

# (wav_filename, csv_filename, time_offset_sec, l_s, r_s)
MICHAELS_FILES = [
    ("103_2.wav", "FLY103.csv", -0.94, 12.0, 100.0),
    ("108_2.wav", "FLY108.csv", -0.40, 9.0, 88.0),
    ("124.wav",   "FLY124.csv", -20.63, 33.0, 105.0),
    ("125.wav",   "FLY125.csv", -26.27, 17.0, 170.0),
]


# ---------------------------------------------------------------------------
# Record class (DREGONRecord-compatible subset)
# ---------------------------------------------------------------------------

@dataclass
class MichaelsRecord:
    """A Michael's recording with aligned audio + motor speeds.

    Duck-type compatible with `DREGONRecord` for the fields used by
    `extract_noise_chunk_with_rps`: `audio`, `audio_timestamps`, `motors`,
    `sample_rate`, `recording_id`, `slice_by_time`.

    NOTE
    ----
    Motor speeds are stored in **revolutions per second** (Hz), to match
    DREGON's `MotorData.measured` units. The DJI CSVs log motor speed in RPM
    so the loader divides by 60.
    """

    recording_id: str
    split: str = "in_flight_noise"  # always
    audio: np.ndarray = field(repr=False, default=None)         # (n_samples, n_channels)
    audio_timestamps: np.ndarray = field(repr=False, default=None)  # (n_samples,)
    motors: MotorData | None = None
    sample_rate: int = 44100

    # Source paths (for traceability)
    wav_path: str | None = None
    csv_path: str | None = None
    time_offset: float = 0.0   # csv-time shift applied
    l_s: float = 0.0           # used valid-range start (sec, rel to record start)
    r_s: float = 0.0           # used valid-range end   (sec, rel to record start)

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def n_channels(self) -> int:
        return self.audio.shape[1] if self.audio.ndim > 1 else 1

    @property
    def start_time(self) -> float:
        return float(self.audio_timestamps[0])

    @property
    def end_time(self) -> float:
        return float(self.audio_timestamps[-1])

    def slice_by_time(self, start_sec: float, end_sec: float) -> "MichaelsRecord":
        """Slice all time series by *relative* time (0 == record start)."""
        abs_start = self.start_time + start_sec
        abs_end = self.start_time + end_sec
        start_idx = int(np.searchsorted(self.audio_timestamps, abs_start))
        end_idx = int(np.searchsorted(self.audio_timestamps, abs_end, side="right"))
        new_audio = self.audio[start_idx:end_idx]
        new_audio_ts = self.audio_timestamps[start_idx:end_idx]
        new_motors = (
            self.motors.slice_by_time(abs_start, abs_end) if self.motors else None
        )
        return replace(
            self,
            audio=new_audio,
            audio_timestamps=new_audio_ts,
            motors=new_motors,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_raw(
    wav_path: Path,
    csv_path: Path,
    time_offset: float,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicates legacy `_load_michaels_data_raw` (drone_audition.datasets.michaels).

    Returns
    -------
    wav        : (n_channels, n_samples) audio at `sample_rate`
    csv_ts     : (M,) CSV timestamps (in seconds, shifted by time_offset removal)
    motor_rps  : (4, M) motor speeds in Hz (RPM/60)
    """
    wav, _ = librosa.load(str(wav_path), sr=sample_rate, mono=False, dtype="float32")
    if wav.ndim == 1:
        wav = wav[None, :]  # (1, T)

    df = pd.read_csv(csv_path, low_memory=False)
    # The CSV has many "Motor*" columns; pick rotation speeds specifically.
    speed_cols = [c for c in df.columns if c.startswith("Motor:Speed")]
    if len(speed_cols) < 4:
        # Fall back to legacy heuristic ("first 4 columns containing 'Motor'")
        speed_cols = [c for c in df.columns if "Motor" in c][:4]
    speed_cols = speed_cols[:4]
    t_col = "Clock:offsetTime"
    small = df[[t_col] + speed_cols].dropna()

    wav_duration = wav.shape[-1] / sample_rate
    keep = (small[t_col] >= time_offset) & (small[t_col] <= wav_duration + time_offset)
    small = small[keep]
    ts = small[t_col].values.astype(np.float64)

    # If CSV starts later than time_offset, trim the front of the wav.
    if ts[0] > time_offset:
        trim_samples = int((ts[0] - time_offset) * sample_rate)
        wav = wav[:, trim_samples:]
        time_offset = float(ts[0])
        wav_duration = wav.shape[-1] / sample_rate

    # If CSV ends earlier than wav, trim the tail.
    if ts[-1] < wav_duration + time_offset:
        keep_samples = int((ts[-1] - time_offset) * sample_rate)
        wav = wav[:, :keep_samples]

    # RPM -> RPS
    motor_rps = small[speed_cols].values.T.astype(np.float32) / 60.0  # (4, M)
    return wav, ts, motor_rps


def load_michaels_record(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float,
    sample_rate: int = 16000,
    valid_l_s: float = 0.0,
    valid_r_s: float | None = None,
    recording_id: str | None = None,
) -> MichaelsRecord:
    """Load a single Michael's recording into a `MichaelsRecord`.

    Args:
        wav_path: path to the WAV file.
        csv_path: path to the DJI CSV log.
        time_offset: empirical CSV-time shift (sec). Positive = CSV time runs
            ahead of audio time; negative = behind.
        sample_rate: target audio sample rate (resamples via librosa).
        valid_l_s, valid_r_s: optional inner clip range (seconds, relative to
            record start) for trimming takeoff/landing. If `valid_r_s` is None,
            keeps to end.
        recording_id: optional id (defaults to wav filename stem).

    Returns:
        MichaelsRecord with audio (n_samples, 1) and synthetic audio
        timestamps anchored at the first CSV timestamp.
    """
    wav_path = Path(wav_path)
    csv_path = Path(csv_path)
    if recording_id is None:
        recording_id = wav_path.stem

    wav, ts, motor_rps = _load_raw(wav_path, csv_path, time_offset, sample_rate)

    # Build synthetic audio timestamps anchored at first CSV timestamp.
    n_samples = wav.shape[-1]
    audiots = np.arange(n_samples, dtype=np.float64) / sample_rate + ts[0]

    # Build a MotorData with the same units as DREGON (rev/s).
    motors = MotorData(
        timestamps=ts,
        measured=motor_rps.T,  # (M_motor, 4) — DREGON convention
        command=motor_rps.T,   # no command logged here; reuse measured
    )

    # Audio shape: (n_samples, n_channels)
    audio = wav.T.astype(np.float32)

    rec = MichaelsRecord(
        recording_id=recording_id,
        audio=audio,
        audio_timestamps=audiots,
        motors=motors,
        sample_rate=sample_rate,
        wav_path=str(wav_path),
        csv_path=str(csv_path),
        time_offset=time_offset,
        l_s=valid_l_s,
        r_s=valid_r_s if valid_r_s is not None else len(audio) / sample_rate,
    )

    # Apply the inner valid-range clip (cuts out takeoff/landing).
    if valid_r_s is None:
        valid_r_s = rec.duration
    rec = rec.slice_by_time(valid_l_s, valid_r_s)
    # Re-attach metadata that `replace` doesn't touch (it does — but set ranges
    # to reflect the post-slice frame of reference: now starts at 0).
    rec.l_s = 0.0
    rec.r_s = float(rec.duration)
    return rec


def load_all_michaels_records(
    data_dir: str | Path,
    sample_rate: int = 16000,
    files: list[tuple] | None = None,
) -> list[MichaelsRecord]:
    """Load every Michael's recording whose files exist in `data_dir`.

    Args:
        data_dir: directory containing the WAVs and CSVs (e.g.
            `data/new-drone-noises/`).
        sample_rate: target sample rate.
        files: optional override of `MICHAELS_FILES` (same 5-tuple format).
    Returns:
        list of `MichaelsRecord`.
    """
    data_dir = Path(data_dir)
    files = files if files is not None else MICHAELS_FILES
    records: list[MichaelsRecord] = []
    for wav_name, csv_name, t_off, l_s, r_s in files:
        wav_p = data_dir / wav_name
        csv_p = data_dir / csv_name
        if not (wav_p.exists() and csv_p.exists()):
            continue
        rec = load_michaels_record(
            wav_p, csv_p,
            time_offset=t_off,
            sample_rate=sample_rate,
            valid_l_s=l_s,
            valid_r_s=r_s,
            recording_id=wav_p.stem,
        )
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Noise + RPS chunk extraction (DREGON-API-compatible)
# ---------------------------------------------------------------------------

def extract_noise_chunk_with_rps(
    record: MichaelsRecord,
    duration_sec: float,
    channel: int = 0,
    start_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract a random (or specified) noise chunk + aligned RPS from a record.

    Mirrors `create_dregon_librimix.extract_noise_chunk_with_rps` so callers
    can treat DREGON and Michael's records uniformly.

    Returns
    -------
    audio    : (n_samples,) mono audio at `record.sample_rate`
    rps      : (4, n_motor_samples) RPS in Hz at motor logging rate (~30 Hz here)
    metadata : dict with `recording_id`, `start_time`, `duration`,
               `motor_sample_rate`, `channel`.
    """
    if record.motors is None:
        raise ValueError(f"record {record.recording_id} has no motor data")

    audio_start = record.audio_timestamps[0]
    audio_end = record.audio_timestamps[-1]
    motor_start = record.motors.timestamps[0]
    motor_end = record.motors.timestamps[-1]
    valid_start = max(audio_start, motor_start)
    valid_end = min(audio_end, motor_end)
    valid_duration = valid_end - valid_start
    if valid_duration < duration_sec:
        raise ValueError(
            f"record {record.recording_id} has insufficient overlap: "
            f"{valid_duration:.2f}s < {duration_sec}s"
        )

    rel_lo = valid_start - audio_start
    rel_hi = valid_end - audio_start - duration_sec
    if start_sec is None:
        start_sec = float(np.random.uniform(rel_lo, rel_hi))
    else:
        start_sec = float(np.clip(start_sec, rel_lo, rel_hi))

    sliced = record.slice_by_time(start_sec, start_sec + duration_sec)

    audio = sliced.audio[:, channel] if sliced.audio.ndim > 1 else sliced.audio
    rps = sliced.motors.measured.T  # (4, n_motor_samples)

    if len(sliced.motors.timestamps) > 1:
        motor_sr = 1.0 / float(np.median(np.diff(sliced.motors.timestamps)))
    else:
        motor_sr = 30.0  # DJI motor log is typically ~30 Hz

    metadata = {
        "recording_id": record.recording_id,
        "start_time": start_sec,
        "duration": duration_sec,
        "motor_sample_rate": motor_sr,
        "channel": channel,
    }
    return audio.astype(np.float32), rps.astype(np.float32), metadata
