"""
DREGON Dataset Loader

DREGON: Dataset and Methods for UAV-Embedded Sound Source Localization
https://dregon.inria.fr/datasets/dregon/

This module provides utilities to load the DREGON dataset as a HuggingFace dataset
with proper handling of multi-rate time series data and aligned slicing.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import scipy.io
import soundfile as sf
from scipy.ndimage import median_filter

# =============================================================================
# Constants
# =============================================================================

DREGON_BASE_URL = "http://dregon.inria.fr"
DREGON_DATA_URL = f"{DREGON_BASE_URL}/DREGON_data"

AUDIO_SAMPLE_RATE = 44100

# Download URLs for various components
DOWNLOAD_URLS = {
    # Microphone positions
    "micPos.txt": f"{DREGON_DATA_URL}/micPos.txt",
    # Emitted source signals
    "emitted_speech": f"{DREGON_DATA_URL}/emitted_signals/2min_TIMIT.wav",
    "emitted_whitenoise": f"{DREGON_DATA_URL}/emitted_signals/2min_white_noise.wav",
    # Clean source recordings (zip files via download IDs)
    "clean_speech": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=375",
    "clean_whitenoise": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=376",
    "clean_chirps": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=377",
    # Individual motor recordings
    "motors": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=378",
    # In-flight recordings (complete zips)
    "free-flight_speech-high_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=317",
    "free-flight_speech-low_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=323",
    "free-flight_whitenoise-high_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=329",
    "free-flight_whitenoise-low_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=335",
    "silent-flight_whitenoise-low_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=341",
    # Noise-only recordings
    "free-flight_nosource_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=314",
    "free-flight_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=350",
    "hovering_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=355",
    "updown_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=360",
    "rectangle_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=365",
    "spinning_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=370",
}

# Split definitions
SplitName = Literal["in_flight_source", "in_flight_noise", "noise_free", "motor", "clean_source"]

SPLIT_RECORDINGS: dict[SplitName, list[str]] = {
    "in_flight_source": [
        "free-flight_speech-high_room1",
        "free-flight_speech-low_room1",
        "free-flight_whitenoise-high_room1",
        "free-flight_whitenoise-low_room1",
    ],
    "in_flight_noise": [
        "free-flight_nosource_room1",
        "free-flight_nosource_room2",
        "hovering_nosource_room2",
        "updown_nosource_room2",
        "rectangle_nosource_room2",
        "spinning_nosource_room2",
    ],
    "noise_free": [
        "silent-flight_whitenoise-low_room1",
    ],
    "motor": [],  # Handled separately (individual motor files)
    "clean_source": [],  # Handled separately (clean recordings)
}


# =============================================================================
# Data Classes for Time Series
# =============================================================================


@dataclass(frozen=True)
class IMUData:
    """IMU (Inertial Measurement Unit) time series data."""

    timestamps: np.ndarray  # (n_samples,) Unix timestamps
    angular_velocity: np.ndarray  # (n_samples, 3) rad/s
    acceleration: np.ndarray  # (n_samples, 3) m/s^2

    def slice_by_time(self, start_time: float, end_time: float) -> "IMUData":
        """Slice to a time range (Unix timestamps)."""
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        return IMUData(
            timestamps=self.timestamps[mask],
            angular_velocity=self.angular_velocity[mask],
            acceleration=self.acceleration[mask],
        )

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class MotorData:
    """Motor telemetry time series data."""

    timestamps: np.ndarray  # (n_samples,) Unix timestamps
    measured: np.ndarray  # (n_samples, 4) measured rotor speeds
    command: np.ndarray  # (n_samples, 4) commanded rotor speeds

    def slice_by_time(self, start_time: float, end_time: float) -> "MotorData":
        """Slice to a time range (Unix timestamps)."""
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        return MotorData(
            timestamps=self.timestamps[mask],
            measured=self.measured[mask],
            command=self.command[mask],
        )

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class SourcePositionData:
    """Source position time series data (from Vicon motion capture)."""

    timestamps: np.ndarray  # (n_samples,) Unix timestamps
    azimuth: np.ndarray  # (n_samples,) radians
    elevation: np.ndarray  # (n_samples,) radians
    distance: np.ndarray  # (n_samples,) meters

    def slice_by_time(self, start_time: float, end_time: float) -> "SourcePositionData":
        """Slice to a time range (Unix timestamps)."""
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        return SourcePositionData(
            timestamps=self.timestamps[mask],
            azimuth=self.azimuth[mask],
            elevation=self.elevation[mask],
            distance=self.distance[mask],
        )

    def __len__(self) -> int:
        return len(self.timestamps)


# =============================================================================
# Main Record Class
# =============================================================================


@dataclass
class DREGONRecord:
    """
    A DREGON recording with aligned multi-rate time series data.

    This class supports slicing by time or audio samples while maintaining
    alignment across all time series (audio, IMU, motors, source position).
    """

    # Identifiers
    recording_id: str
    split: SplitName

    # Global geometry (shared across all samples) - required fields first
    mic_positions: np.ndarray = field(repr=False)  # (8, 3)
    rotor_positions: np.ndarray = field(repr=False)  # (4, 3)

    # Audio data - required fields
    audio: np.ndarray = field(repr=False)  # (n_samples, n_channels)
    audio_timestamps: np.ndarray = field(repr=False)  # (n_samples,) Unix timestamps

    # Recording metadata - optional fields with defaults
    flight_type: str | None = None  # 'free-flight', 'hovering', 'silent-flight', etc.
    source_type: str | None = None  # 'speech', 'whitenoise', 'chirp', None
    source_level: str | None = None  # 'high', 'low', None
    room: str | None = None  # 'room1', 'room2'
    motor_id: int | None = None  # 1-4 for individual motor recordings
    motor_speed: int | None = None  # 50-90 turns/sec
    sample_rate: int = AUDIO_SAMPLE_RATE

    # Optional time series data
    imu: IMUData | None = None
    motors: MotorData | None = None
    source_position: SourcePositionData | None = None

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.audio) / self.sample_rate

    @property
    def n_channels(self) -> int:
        """Number of audio channels."""
        return self.audio.shape[1] if len(self.audio.shape) > 1 else 1

    @property
    def start_time(self) -> float:
        """Start time (Unix timestamp)."""
        return float(self.audio_timestamps[0])

    @property
    def end_time(self) -> float:
        """End time (Unix timestamp)."""
        return float(self.audio_timestamps[-1])

    def slice_by_time(self, start_sec: float, end_sec: float) -> "DREGONRecord":
        """
        Slice all time series to a relative time range.

        Args:
            start_sec: Start time in seconds from beginning of recording
            end_sec: End time in seconds from beginning of recording

        Returns:
            New DREGONRecord with sliced data
        """
        # Convert relative time to absolute Unix timestamps
        abs_start = self.start_time + start_sec
        abs_end = self.start_time + end_sec

        return self._slice_by_absolute_time(abs_start, abs_end)

    def slice_by_audio_samples(self, start_idx: int, end_idx: int) -> "DREGONRecord":
        """
        Slice by audio sample indices, aligning other time series.

        Args:
            start_idx: Start audio sample index
            end_idx: End audio sample index (exclusive)

        Returns:
            New DREGONRecord with sliced data
        """
        # Get absolute time range from audio indices
        abs_start = self.audio_timestamps[start_idx]
        abs_end = self.audio_timestamps[min(end_idx, len(self.audio_timestamps) - 1)]

        # Slice audio directly by indices for precision
        new_audio = self.audio[start_idx:end_idx]
        new_audio_ts = self.audio_timestamps[start_idx:end_idx]

        # Slice other time series by time
        new_imu = self.imu.slice_by_time(abs_start, abs_end) if self.imu else None
        new_motors = self.motors.slice_by_time(abs_start, abs_end) if self.motors else None
        new_sourcepos = (
            self.source_position.slice_by_time(abs_start, abs_end)
            if self.source_position
            else None
        )

        return replace(
            self,
            audio=new_audio,
            audio_timestamps=new_audio_ts,
            imu=new_imu,
            motors=new_motors,
            source_position=new_sourcepos,
        )

    def _slice_by_absolute_time(self, abs_start: float, abs_end: float) -> "DREGONRecord":
        """Slice all time series by absolute Unix timestamps."""
        # Find audio indices
        start_idx = int(np.searchsorted(self.audio_timestamps, abs_start))
        end_idx = int(np.searchsorted(self.audio_timestamps, abs_end, side="right"))

        new_audio = self.audio[start_idx:end_idx]
        new_audio_ts = self.audio_timestamps[start_idx:end_idx]

        # Slice other time series
        new_imu = self.imu.slice_by_time(abs_start, abs_end) if self.imu else None
        new_motors = self.motors.slice_by_time(abs_start, abs_end) if self.motors else None
        new_sourcepos = (
            self.source_position.slice_by_time(abs_start, abs_end)
            if self.source_position
            else None
        )

        return replace(
            self,
            audio=new_audio,
            audio_timestamps=new_audio_ts,
            imu=new_imu,
            motors=new_motors,
            source_position=new_sourcepos,
        )

    def resample_audio(self, target_sr: int, cache_dir: Path | None = None) -> "DREGONRecord":
        """
        Resample audio to a target sample rate.

        Args:
            target_sr: Target sample rate in Hz
            cache_dir: Directory to cache resampled audio. If None, no caching.

        Returns:
            New DREGONRecord with resampled audio
        """
        if target_sr == self.sample_rate:
            return self

        # Check cache
        if cache_dir is not None:
            cache_path = self._get_cache_path(cache_dir, target_sr)
            if cache_path.exists():
                cached = np.load(cache_path)
                new_audio = cached["audio"]
                new_timestamps = cached["timestamps"]
                return replace(
                    self,
                    audio=new_audio,
                    audio_timestamps=new_timestamps,
                    sample_rate=target_sr,
                )

        # Resample using librosa
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for resampling. Install with: pip install librosa")

        # Resample each channel
        if len(self.audio.shape) == 1:
            new_audio = librosa.resample(self.audio, orig_sr=self.sample_rate, target_sr=target_sr)
        else:
            # (n_samples, n_channels) -> transpose for librosa -> transpose back
            new_audio = librosa.resample(
                self.audio.T, orig_sr=self.sample_rate, target_sr=target_sr
            ).T

        # Interpolate timestamps
        new_n_samples = len(new_audio)
        new_timestamps = np.linspace(
            self.audio_timestamps[0],
            self.audio_timestamps[-1],
            new_n_samples,
        )

        # Cache if requested
        if cache_dir is not None:
            cache_path = self._get_cache_path(cache_dir, target_sr)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, audio=new_audio, timestamps=new_timestamps)

        return replace(
            self,
            audio=new_audio,
            audio_timestamps=new_timestamps,
            sample_rate=target_sr,
        )

    def _get_cache_path(self, cache_dir: Path, target_sr: int) -> Path:
        """Generate a unique cache path for resampled audio."""
        # Hash based on recording ID, sample rate, and time range
        key = f"{self.recording_id}_{self.start_time}_{self.end_time}_{target_sr}"
        hash_str = hashlib.md5(key.encode()).hexdigest()[:12]
        return cache_dir / f"{self.recording_id}_{target_sr}hz_{hash_str}.npz"


# =============================================================================
# Lazy Loading Sample (for HuggingFace Dataset)
# =============================================================================


class DREGONSampleDict(TypedDict, total=False):
    """Type definition for HuggingFace dataset samples (lazy loading)."""

    recording_id: str
    split: str
    flight_type: str | None
    source_type: str | None
    source_level: str | None
    room: str | None
    motor_id: int | None
    motor_speed: int | None
    # File paths for lazy loading
    audio_path: str
    audiots_path: str | None
    imu_path: str | None
    motors_path: str | None
    sourcepos_path: str | None
    # Geometry paths
    mic_positions_path: str
    rotor_positions_path: str


def load_record_from_sample(
    sample: DREGONSampleDict,
    geometry: tuple[np.ndarray, np.ndarray] | None = None,
) -> DREGONRecord:
    """
    Load a full DREGONRecord from a HuggingFace dataset sample.

    Args:
        sample: Dictionary from HuggingFace dataset
        geometry: Optional pre-loaded (mic_positions, rotor_positions) tuple

    Returns:
        Fully loaded DREGONRecord
    """
    # Load geometry
    if geometry is not None:
        mic_pos, rotor_pos = geometry
    else:
        mic_pos_path = Path(sample["mic_positions_path"])
        if mic_pos_path.suffix == ".txt":
            mic_pos = _parse_mic_positions_txt(mic_pos_path)
        else:
            mic_pos = np.loadtxt(mic_pos_path)
        rotor_pos = _load_rotor_positions(sample["rotor_positions_path"])

    # Load audio
    audio, sr = sf.read(sample["audio_path"])
    if len(audio.shape) == 1:
        audio = audio[:, np.newaxis]

    # Load audio timestamps
    if sample.get("audiots_path"):
        audiots_mat = scipy.io.loadmat(sample["audiots_path"])
        audio_timestamps = audiots_mat["audio_timestamps"].flatten()
    else:
        # Generate synthetic timestamps for motor recordings
        audio_timestamps = np.arange(len(audio)) / sr

    # Load optional time series
    imu = None
    if sample.get("imu_path"):
        imu_mat = scipy.io.loadmat(sample["imu_path"])
        imu_struct = imu_mat["imu"]
        imu = IMUData(
            timestamps=imu_struct["timestamps"][0, 0].flatten(),
            angular_velocity=imu_struct["angular_velocity"][0, 0],
            acceleration=imu_struct["acceleration"][0, 0],
        )

    motors = None
    if sample.get("motors_path"):
        motors_mat = scipy.io.loadmat(sample["motors_path"])
        motors_struct = motors_mat["motor"]
        # Some recordings (room2) only have 'command', not 'measured'
        # Use 'measured' if available, otherwise fall back to 'command'
        if "measured" in motors_struct.dtype.names:
            measured = motors_struct["measured"][0, 0]
        else:
            measured = motors_struct["command"][0, 0]  # Use command as proxy
        motors = MotorData(
            timestamps=motors_struct["timestamps"][0, 0].flatten(),
            measured=measured,
            command=motors_struct["command"][0, 0],
        )

    source_position = None
    if sample.get("sourcepos_path"):
        sourcepos_mat = scipy.io.loadmat(sample["sourcepos_path"])
        sourcepos_struct = sourcepos_mat["source_position"]
        source_position = SourcePositionData(
            timestamps=sourcepos_struct["timestamps"][0, 0].flatten(),
            azimuth=sourcepos_struct["azimuth"][0, 0].flatten(),
            elevation=sourcepos_struct["elevation"][0, 0].flatten(),
            distance=sourcepos_struct["distance"][0, 0].flatten(),
        )

    return DREGONRecord(
        recording_id=sample["recording_id"],
        split=sample["split"],
        mic_positions=mic_pos,
        rotor_positions=rotor_pos,
        audio=audio,
        audio_timestamps=audio_timestamps,
        flight_type=sample.get("flight_type"),
        source_type=sample.get("source_type"),
        source_level=sample.get("source_level"),
        room=sample.get("room"),
        motor_id=sample.get("motor_id"),
        motor_speed=sample.get("motor_speed"),
        sample_rate=sr,
        imu=imu,
        motors=motors,
        source_position=source_position,
    )


# =============================================================================
# Download Utilities
# =============================================================================


def _download_file(url: str, dest: Path, desc: str | None = None) -> Path:
    """Download a file if it doesn't exist."""
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    print(f"Downloading {desc or dest.name}...")
    try:
        urllib.request.urlretrieve(url, tmp_dest)
        tmp_dest.rename(dest)
    except Exception:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise

    return dest


def _unpack_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Unpack a zip file if not already unpacked.

    Some DREGON download IDs serve raw files (e.g. WAV) instead of a zip
    archive. When that happens the file is placed directly in dest_dir with
    an appropriate extension (detected from magic bytes).
    """
    import shutil
    import zipfile

    marker = dest_dir / ".unzipped"
    if marker.exists():
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile:
        # Server returned a raw file instead of a zip archive.
        # Detect format from magic bytes and copy to dest_dir.
        with open(zip_path, "rb") as f:
            magic = f.read(4)
        ext = ".wav" if magic == b"RIFF" else zip_path.suffix
        dest_file = dest_dir / (zip_path.stem + ext)
        if not dest_file.exists():
            shutil.copy2(zip_path, dest_file)
        print(f"  (not a zip; placed as raw file {dest_file.name})")

    marker.write_text("ok", encoding="utf-8")
    return dest_dir


def _load_rotor_positions(path: str | Path) -> np.ndarray:
    """Load rotor positions from coordinates.mat."""
    mat = scipy.io.loadmat(str(path))
    return mat["rotorsPos"]


def _parse_mic_positions_txt(path: Path) -> np.ndarray:
    """
    Parse micPos.txt which is in MATLAB format.

    Format example:
    % comments
    micPos = [  0.0420    0.0615   -0.0410;  % mic 1
               -0.0420    0.0615    0.0410;  % mic 2
               ...
               0.0615    0.0420    0.0410]; % mic 8
    """
    content = path.read_text()

    # Extract the matrix content between [ and ]
    match = re.search(r'\[\s*([\s\S]*?)\s*\]', content)
    if not match:
        raise ValueError(f"Could not parse micPos.txt: no matrix found in {path}")

    matrix_content = match.group(1)

    # Parse each row
    rows = []
    for line in matrix_content.split(';'):
        # Remove comments (everything after %)
        line = re.sub(r'%.*', '', line)
        # Extract numbers
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if len(numbers) >= 3:
            rows.append([float(n) for n in numbers[:3]])

    return np.array(rows)


def download_dregon_dataset(
    data_dir: Path,
    *,
    download_clean_sources: bool = True,
    download_emitted_signals: bool = True,
) -> Path:
    """
    Download the DREGON dataset.

    Args:
        data_dir: Root directory for the dataset
        download_clean_sources: Whether to download clean source recordings
        download_emitted_signals: Whether to download emitted source signals

    Returns:
        Path to the DREGON dataset directory
    """
    dregon_dir = data_dir / "DREGON"
    dregon_dir.mkdir(parents=True, exist_ok=True)

    # Download micPos.txt
    mic_pos_path = dregon_dir / "micPos.txt"
    _download_file(DOWNLOAD_URLS["micPos.txt"], mic_pos_path, "microphone positions")

    # Download emitted signals
    if download_emitted_signals:
        emitted_dir = dregon_dir / "emitted_signals"
        emitted_dir.mkdir(exist_ok=True)
        _download_file(
            DOWNLOAD_URLS["emitted_speech"],
            emitted_dir / "2min_TIMIT.wav",
            "emitted speech signal",
        )
        _download_file(
            DOWNLOAD_URLS["emitted_whitenoise"],
            emitted_dir / "2min_white_noise.wav",
            "emitted white noise signal",
        )

    # Download clean source recordings
    if download_clean_sources:
        clean_dir = dregon_dir / "clean_sources"
        for source_type in ["speech", "whitenoise", "chirps"]:
            zip_path = clean_dir / f"clean_{source_type}.zip"
            _download_file(
                DOWNLOAD_URLS[f"clean_{source_type}"],
                zip_path,
                f"clean {source_type} recordings",
            )
            _unpack_zip(zip_path, clean_dir / source_type)

    # Download in-flight recordings
    for recording_id, url in DOWNLOAD_URLS.items():
        if recording_id in ["micPos.txt", "emitted_speech", "emitted_whitenoise", "motors"]:
            continue
        if recording_id.startswith("clean_"):
            continue

        recording_dir = dregon_dir / f"DREGON_{recording_id}"
        if recording_dir.exists() and any(recording_dir.glob("*.wav")):
            continue  # Already downloaded

        zip_path = dregon_dir / f"DREGON_{recording_id}.zip"
        try:
            _download_file(url, zip_path, f"recording {recording_id}")
            _unpack_zip(zip_path, recording_dir)
        except Exception as e:
            print(f"Warning: failed to download {recording_id}: {e}")
            continue

    # Download motor recordings
    motors_dir = dregon_dir / "DREGON_individual_motors_recordings"
    if not motors_dir.exists() or not any(motors_dir.glob("*.wav")):
        zip_path = dregon_dir / "motors.zip"
        _download_file(DOWNLOAD_URLS["motors"], zip_path, "motor recordings")
        _unpack_zip(zip_path, motors_dir)

    return dregon_dir


# =============================================================================
# Dataset Discovery
# =============================================================================


def _parse_recording_id(recording_id: str) -> dict:
    """Parse recording ID into metadata components."""
    # Pattern: flight-type_source-level_room
    # Examples: free-flight_speech-high_room1, hovering_nosource_room2

    parts = recording_id.split("_")

    result = {
        "flight_type": None,
        "source_type": None,
        "source_level": None,
        "room": None,
    }

    for part in parts:
        if part in ["free-flight", "hovering", "updown", "rectangle", "spinning", "silent-flight"]:
            result["flight_type"] = part
        elif part.startswith("room"):
            result["room"] = part
        elif "-" in part and part not in ["free-flight", "silent-flight"]:
            # Could be source-level like "speech-high"
            subparts = part.split("-")
            if len(subparts) == 2:
                if subparts[0] in ["speech", "whitenoise"]:
                    result["source_type"] = subparts[0]
                    result["source_level"] = subparts[1]
        elif part == "nosource":
            result["source_type"] = None

    return result


def _parse_motor_filename(filename: str) -> tuple[int | None, int | None]:
    """Parse motor recording filename into motor_id and speed."""
    # Examples: Motor1_50.wav, allMotors_70.wav
    match = re.match(r"Motor(\d)_(\d+)\.wav", filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.match(r"allMotors_(\d+)\.wav", filename)
    if match:
        return None, int(match.group(1))  # None means all motors

    return None, None


def _parse_clean_source_filename(filename: str) -> tuple[int, int, float]:
    """
    Parse clean source recording filename into azimuth, elevation, distance.

    Format: XX_YY_ZZ.wav where XX=azimuth, YY=elevation, ZZ=distance
    """
    match = re.match(r"(\d+)_(\d+)_(\d+)\.wav", filename)
    if match:
        azimuth = int(match.group(1))
        elevation = int(match.group(2))
        distance = int(match.group(3)) / 10.0  # Assuming distance is in decimeters
        return azimuth, elevation, distance
    return 0, 0, 0.0


def discover_recordings(dregon_dir: Path) -> list[DREGONSampleDict]:
    """
    Discover all recordings in the DREGON dataset directory.

    Returns:
        List of sample dictionaries for HuggingFace dataset
    """
    samples: list[DREGONSampleDict] = []

    # Geometry paths
    mic_pos_path = dregon_dir / "micPos.txt"
    coord_mat_path = dregon_dir / "coordinates.mat"

    # Use micPos.txt if available, otherwise fall back to coordinates.mat
    if not mic_pos_path.exists() and coord_mat_path.exists():
        # Extract from coordinates.mat
        mat = scipy.io.loadmat(str(coord_mat_path))
        mic_pos_path = dregon_dir / "micPos_extracted.txt"
        np.savetxt(mic_pos_path, mat["micPos"])

    rotor_pos_path = str(coord_mat_path) if coord_mat_path.exists() else ""

    # Discover in-flight recordings
    for split_name, recording_ids in SPLIT_RECORDINGS.items():
        for recording_id in recording_ids:
            recording_dir = dregon_dir / f"DREGON_{recording_id}"
            if not recording_dir.exists():
                continue

            wav_files = list(recording_dir.glob("*.wav"))
            if not wav_files:
                continue

            wav_path = wav_files[0]
            metadata = _parse_recording_id(recording_id)

            sample: DREGONSampleDict = {
                "recording_id": recording_id,
                "split": split_name,
                "flight_type": metadata["flight_type"],
                "source_type": metadata["source_type"],
                "source_level": metadata["source_level"],
                "room": metadata["room"],
                "motor_id": None,
                "motor_speed": None,
                "audio_path": str(wav_path),
                "audiots_path": None,
                "imu_path": None,
                "motors_path": None,
                "sourcepos_path": None,
                "mic_positions_path": str(mic_pos_path),
                "rotor_positions_path": rotor_pos_path,
            }

            # Find associated .mat files
            base_name = f"DREGON_{recording_id}"
            audiots = recording_dir / f"{base_name}_audiots.mat"
            if audiots.exists():
                sample["audiots_path"] = str(audiots)

            imu = recording_dir / f"{base_name}_imu.mat"
            if imu.exists():
                sample["imu_path"] = str(imu)

            motors = recording_dir / f"{base_name}_motors.mat"
            if motors.exists():
                sample["motors_path"] = str(motors)

            sourcepos = recording_dir / f"{base_name}_sourcepos.mat"
            if sourcepos.exists():
                sample["sourcepos_path"] = str(sourcepos)

            samples.append(sample)

    # Discover motor recordings
    motors_dir = dregon_dir / "DREGON_individual_motors_recordings"
    if motors_dir.exists():
        for wav_file in motors_dir.glob("*.wav"):
            motor_id, motor_speed = _parse_motor_filename(wav_file.name)

            sample: DREGONSampleDict = {
                "recording_id": f"motor_{wav_file.stem}",
                "split": "motor",
                "flight_type": None,
                "source_type": None,
                "source_level": None,
                "room": None,
                "motor_id": motor_id,
                "motor_speed": motor_speed,
                "audio_path": str(wav_file),
                "audiots_path": None,
                "imu_path": None,
                "motors_path": None,
                "sourcepos_path": None,
                "mic_positions_path": str(mic_pos_path),
                "rotor_positions_path": rotor_pos_path,
            }
            samples.append(sample)

    # Discover clean source recordings
    clean_dir = dregon_dir / "clean_sources"
    if clean_dir.exists():
        for source_type in ["speech", "whitenoise", "chirps"]:
            source_dir = clean_dir / source_type
            if not source_dir.exists():
                continue

            for wav_file in source_dir.rglob("*.wav"):
                azimuth, elevation, distance = _parse_clean_source_filename(wav_file.name)

                sample: DREGONSampleDict = {
                    "recording_id": f"clean_{source_type}_{wav_file.stem}",
                    "split": "clean_source",
                    "flight_type": None,
                    "source_type": source_type.rstrip("s"),  # chirps -> chirp
                    "source_level": None,
                    "room": None,
                    "motor_id": None,
                    "motor_speed": None,
                    "audio_path": str(wav_file),
                    "audiots_path": None,
                    "imu_path": None,
                    "motors_path": None,
                    "sourcepos_path": None,
                    "mic_positions_path": str(mic_pos_path),
                    "rotor_positions_path": rotor_pos_path,
                }
                samples.append(sample)

    return samples


# =============================================================================
# HuggingFace Dataset Integration
# =============================================================================


def load_dregon_dataset(
    data_dir: Path | str,
    *,
    splits: list[SplitName] | None = None,
    download: bool = True,
) -> "DatasetDict":
    """
    Load the DREGON dataset as a HuggingFace DatasetDict.

    Args:
        data_dir: Root data directory containing DREGON folder
        splits: List of splits to load. If None, load all splits.
        download: Whether to download missing data

    Returns:
        HuggingFace DatasetDict with lazy-loading samples
    """
    try:
        from datasets import Dataset, DatasetDict, Features, Value
    except ImportError:
        raise ImportError(
            "datasets is required to load the DREGON dataset. "
            "Install with: pip install datasets"
        )

    data_dir = Path(data_dir)
    dregon_dir = data_dir / "DREGON"

    # Download if needed (inner helpers are idempotent: they skip existing files)
    if download:
        download_dregon_dataset(data_dir)

    # Discover all recordings
    all_samples = discover_recordings(dregon_dir)

    if not all_samples:
        raise ValueError(f"No recordings found in {dregon_dir}")

    # Group by split
    if splits is None:
        splits = list(set(s["split"] for s in all_samples))

    split_samples: dict[str, list[DREGONSampleDict]] = {s: [] for s in splits}
    for sample in all_samples:
        if sample["split"] in split_samples:
            split_samples[sample["split"]].append(sample)

    # Define features
    features = Features(
        {
            "recording_id": Value("string"),
            "split": Value("string"),
            "flight_type": Value("string"),
            "source_type": Value("string"),
            "source_level": Value("string"),
            "room": Value("string"),
            "motor_id": Value("int32"),
            "motor_speed": Value("int32"),
            "audio_path": Value("string"),
            "audiots_path": Value("string"),
            "imu_path": Value("string"),
            "motors_path": Value("string"),
            "sourcepos_path": Value("string"),
            "mic_positions_path": Value("string"),
            "rotor_positions_path": Value("string"),
        }
    )

    # Create datasets
    datasets = {}
    for split_name, samples in split_samples.items():
        if not samples:
            continue

        # Convert to columnar format
        data = {key: [] for key in features.keys()}
        for sample in samples:
            for key in features.keys():
                val = sample.get(key)
                # Convert None to empty string for string types, -1 for int types
                if val is None:
                    if "int" in str(features[key]):
                        val = -1
                    else:
                        val = ""
                data[key].append(val)

        datasets[split_name] = Dataset.from_dict(data, features=features)

    return DatasetDict(datasets)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_geometry(dregon_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load microphone and rotor positions.

    Returns:
        Tuple of (mic_positions, rotor_positions) arrays
    """
    mic_pos_path = dregon_dir / "micPos.txt"
    coord_mat_path = dregon_dir / "coordinates.mat"

    if mic_pos_path.exists():
        # micPos.txt is in MATLAB format, needs special parsing
        mic_pos = _parse_mic_positions_txt(mic_pos_path)
    elif coord_mat_path.exists():
        mat = scipy.io.loadmat(str(coord_mat_path))
        mic_pos = mat["micPos"]
    else:
        raise FileNotFoundError("Neither micPos.txt nor coordinates.mat found")

    if coord_mat_path.exists():
        mat = scipy.io.loadmat(str(coord_mat_path))
        rotor_pos = mat["rotorsPos"]
    else:
        # Use default rotor positions from plot_structure.m
        r = 0.485 / 2
        angles = np.array([45, 135, 225, 315]) + 90
        offset = np.array([0.0115, 0.0, 0.1915])
        rotor_pos = np.zeros((4, 3))
        for i, angle in enumerate(angles):
            rotor_pos[i] = [
                r * np.cos(np.radians(angle)),
                r * np.sin(np.radians(angle)),
                0.1915,
            ]
            rotor_pos[i] += offset

    return mic_pos, rotor_pos


def create_sliced_dataset(
    dataset: "Dataset",
    segment_duration: float,
    *,
    overlap: float = 0.0,
    target_sr: int | None = None,
    cache_dir: Path | None = None,
    geometry: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[DREGONRecord]:
    """
    Create a list of fixed-duration DREGONRecord slices from a dataset.

    Args:
        dataset: HuggingFace dataset split
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments in seconds
        target_sr: Target sample rate for resampling (None = keep original)
        cache_dir: Directory for caching resampled audio
        geometry: Pre-loaded geometry tuple

    Returns:
        List of DREGONRecord slices
    """
    slices = []
    step = segment_duration - overlap

    for sample in dataset:
        record = load_record_from_sample(sample, geometry=geometry)

        if target_sr is not None:
            record = record.resample_audio(target_sr, cache_dir=cache_dir)

        # Create slices
        start = 0.0
        while start + segment_duration <= record.duration:
            slice_record = record.slice_by_time(start, start + segment_duration)
            slices.append(slice_record)
            start += step

    return slices


# =============================================================================
# RPS (Rotor Speed) Extraction Utilities
# =============================================================================


@dataclass
class NoiseSegment:
    """
    A segment of drone noise with associated RPS data.

    Used for creating synthetic mixtures with real rotor speed information.
    """

    audio: np.ndarray  # (n_samples,) mono audio
    rps: np.ndarray  # (4, n_motor_samples) rotor speeds at native rate
    sample_rate: int
    motor_sample_rate: float  # Approximate motor logging rate

    # Metadata
    recording_id: str
    start_time: float  # Relative start time in original recording
    duration: float

    @classmethod
    def from_record(
        cls,
        record: DREGONRecord,
        start_sec: float,
        duration_sec: float,
        channel: int = 0,
    ) -> "NoiseSegment":
        """
        Extract a noise segment with aligned RPS from a DREGONRecord.

        Args:
            record: Source DREGON record (must have motors data)
            start_sec: Start time in seconds from beginning of recording
            duration_sec: Duration in seconds
            channel: Which microphone channel to use (0-7)

        Returns:
            NoiseSegment with audio and RPS at native rates
        """
        if record.motors is None:
            raise ValueError(f"Record {record.recording_id} has no motor data")

        # Slice the record
        sliced = record.slice_by_time(start_sec, start_sec + duration_sec)

        # Extract mono audio from specified channel
        audio = sliced.audio[:, channel] if sliced.audio.ndim > 1 else sliced.audio

        # Extract RPS (already sliced by time in slice_by_time)
        rps = sliced.motors.measured.T  # (4, n_motor_samples)

        # Estimate motor sample rate
        if len(sliced.motors.timestamps) > 1:
            motor_dt = np.median(np.diff(sliced.motors.timestamps))
            motor_sr = 1.0 / motor_dt
        else:
            motor_sr = 929.0  # Default DREGON rate

        return cls(
            audio=audio,
            rps=rps,
            sample_rate=sliced.sample_rate,
            motor_sample_rate=motor_sr,
            recording_id=record.recording_id,
            start_time=start_sec,
            duration=duration_sec,
        )


def extract_noise_segments(
    dregon_dir: Path,
    segment_duration: float = 1.0,
    overlap: float = 0.0,
    splits: list[str] | None = None,
    target_sr: int | None = None,
    channel: int = 0,
) -> list[NoiseSegment]:
    """
    Extract all noise segments with RPS from DREGON dataset.

    Args:
        dregon_dir: Path to DREGON dataset directory
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments in seconds
        splits: Which splits to use (default: ["in_flight_noise"])
        target_sr: Target audio sample rate (None = keep 44100)
        channel: Which microphone channel to use

    Returns:
        List of NoiseSegment objects
    """
    if splits is None:
        splits = ["in_flight_noise"]

    # Load dataset
    dataset = load_dregon_dataset(dregon_dir.parent, splits=splits, download=False)
    geometry = get_geometry(dregon_dir)

    segments = []
    step = segment_duration - overlap

    for split_name in splits:
        if split_name not in dataset:
            continue

        for sample in dataset[split_name]:
            record = load_record_from_sample(sample, geometry=geometry)

            # Skip if no motor data
            if record.motors is None:
                continue

            # Optionally resample audio
            if target_sr is not None and target_sr != record.sample_rate:
                record = record.resample_audio(target_sr)

            # Extract segments
            start = 0.0
            while start + segment_duration <= record.duration:
                try:
                    segment = NoiseSegment.from_record(
                        record, start, segment_duration, channel=channel
                    )
                    segments.append(segment)
                except Exception as e:
                    print(f"Warning: Failed to extract segment from {record.recording_id} "
                          f"at {start:.1f}s: {e}")
                start += step

    return segments


def get_constant_rps_for_motor_recording(
    motor_id: int | None,
    motor_speed: int,
    duration_sec: float,
    motor_sample_rate: float = 929.0,
) -> np.ndarray:
    """
    Generate constant RPS array for individual motor recordings.

    Individual motor recordings have a fixed speed (no time-varying data).
    This creates a synthetic RPS array with the known constant speed.

    Args:
        motor_id: Which motor (1-4), or None for all motors
        motor_speed: Speed in turns/second (50-90)
        duration_sec: Duration to generate
        motor_sample_rate: Sample rate for the output

    Returns:
        (4, n_samples) array with constant speeds
    """
    n_samples = int(duration_sec * motor_sample_rate)
    rps = np.zeros((4, n_samples), dtype=np.float32)

    if motor_id is not None:
        # Single motor active
        rps[motor_id - 1, :] = motor_speed
    else:
        # All motors at same speed
        rps[:, :] = motor_speed

    return rps


def _find_step_artifact_length(command: np.ndarray) -> int:
    """Detect initial constant-value logging artifact and return its length.

    Some DREGON recordings start with a block of bit-identical command values
    (stale flight-controller initialization data) before the real signal begins.
    All four motors change at the same sample index when the real data starts.

    Returns the number of leading samples to discard, or 0 if no artifact.
    """
    n_samples = len(command)
    if n_samples < 2:
        return 0

    # Find how long the first sample's value is repeated exactly.
    # Use motor 0 as reference; all motors change at the same index.
    init_val = command[0, 0]
    if init_val < 30.0:  # Step artifacts always start at high values (>60 Hz)
        return 0

    n_same = 0
    for j in range(n_samples):
        if command[j, 0] == init_val:
            n_same += 1
        else:
            break

    # Require at least 100 identical samples (~0.1 s) to confirm it's an artifact
    if n_same >= 100:
        return n_same
    return 0


def clean_command_spikes(
    command: np.ndarray,
    kernel: int = 21,
) -> np.ndarray:
    """Clean DREGON commanded rotor speeds by removing known artifacts.

    Two artifacts are removed:

    1. **Step artifact** — some recordings start with a block of bit-identical
       high command values (stale FC initialization data, 0.6–3.1 s long).
       These are zeroed out.

    2. **Takeoff spikes** — during the real ramp-up the command oscillates
       between a smooth lower envelope (the real target) and short upward
       spikes. A median filter removes these impulsive outliers.

    Applied per-motor (operates along axis 0 independently for each column).

    Args:
        command: (n_samples, 4) array of commanded rotor speeds (RPS, Hz)
        kernel: Median filter kernel size along time axis (odd int, default 21).
                 At the ~929 Hz DREGON motor rate this covers ~23 ms — enough
                 to remove 1–3 sample spike bursts while preserving genuine
                 speed changes.

    Returns:
        (n_samples, 4) cleaned command array, same shape and dtype as input.
    """
    cleaned = command.copy()

    # Step 1: remove step artifact (zero out the constant-initial-value block)
    n_step = _find_step_artifact_length(command)
    if n_step > 0:
        cleaned[:n_step] = 0.0

    # Step 2: remove takeoff spikes via median filter
    cleaned = median_filter(cleaned, size=(kernel, 1), mode="reflect")

    return cleaned
