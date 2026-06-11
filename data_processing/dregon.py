"""
DREGON Dataset Loader — TimeFrame-native interface.

DREGON: Dataset and Methods for UAV-Embedded Sound Source Localization
https://dregon.inria.fr/datasets/dregon/

Each recording is loaded as a `TimeFrame` with tracks:
  - ``"audio"``            : UniformSeries  — (n_samples, n_channels) at 44100 Hz
  - ``"motors_measured"``  : EventSeries    — timestamps + values (N, 4)  [if available]
  - ``"motors_command"``   : EventSeries    — timestamps + values (N, 4)  [if available]
  - ``"imu_accel"``        : EventSeries    — timestamps + values (N, 3)  [if available]
  - ``"imu_gyro"``         : EventSeries    — timestamps + values (N, 3)  [if available]
  - ``"source_position"``  : EventSeries    — timestamps + values (N, 3)  [if available]

``tags`` carry scalar metadata (``recording_id``, ``split``, ``flight_type``, …).
``global_data`` stores array metadata: ``mic_positions`` (8, 3), ``rotor_positions`` (4, 3).

All time-series are aligned to a common absolute time base (Unix timestamps),
so ``tf.slice(t_a, t_b)`` simultaneously cuts every track.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import scipy.io
import soundfile as sf
from scipy.ndimage import median_filter

from utils.data import EventSeries, TimeFrame, UniformSeries

# =============================================================================
# Constants
# =============================================================================

AUDIO_SAMPLE_RATE = 44100
MOTOR_SAMPLE_RATE = 929.0  # approximate DREGON motor logging rate
NUM_ROTORS = 4
N_MICS = 8

# -- download URL map -------------------------------------------------------

DREGON_BASE_URL = "http://dregon.inria.fr"
DREGON_DATA_URL = f"{DREGON_BASE_URL}/DREGON_data"

DOWNLOAD_URLS = {
    "micPos.txt": f"{DREGON_DATA_URL}/micPos.txt",
    "emitted_speech": f"{DREGON_DATA_URL}/emitted_signals/2min_TIMIT.wav",
    "emitted_whitenoise": f"{DREGON_DATA_URL}/emitted_signals/2min_white_noise.wav",
    "clean_speech": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=375",
    "clean_whitenoise": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=376",
    "clean_chirps": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=377",
    "motors": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=378",
    "free-flight_speech-high_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=317",
    "free-flight_speech-low_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=323",
    "free-flight_whitenoise-high_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=329",
    "free-flight_whitenoise-low_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=335",
    "silent-flight_whitenoise-low_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=341",
    "free-flight_nosource_room1": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=314",
    "free-flight_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=350",
    "hovering_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=355",
    "updown_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=360",
    "rectangle_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=365",
    "spinning_nosource_room2": f"{DREGON_BASE_URL}/?smd_process_download=1&download_id=370",
}

# -- split definitions -------------------------------------------------------

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
    "noise_free": ["silent-flight_whitenoise-low_room1"],
    "motor": [],
    "clean_source": [],
}


# =============================================================================
# Download & discovery (unchanged from legacy)
# =============================================================================


def _download_file(url: str, dest: Path, desc: str | None = None) -> Path:
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
        with open(zip_path, "rb") as f:
            magic = f.read(4)
        ext = ".wav" if magic == b"RIFF" else zip_path.suffix
        dest_file = dest_dir / (zip_path.stem + ext)
        if not dest_file.exists():
            shutil.copy2(zip_path, dest_file)
        print(f"  (not a zip; placed as raw file {dest_file.name})")
    marker.write_text("ok", encoding="utf-8")
    return dest_dir


def _parse_mic_positions_txt(path: Path) -> np.ndarray:
    content = path.read_text()
    match = re.search(r"\[\s*([\s\S]*?)\s*\]", content)
    if not match:
        raise ValueError(f"Could not parse micPos.txt: no matrix found in {path}")
    matrix_content = match.group(1)
    rows = []
    for line in matrix_content.split(";"):
        line = re.sub(r"%.*", "", line)
        numbers = re.findall(r"-?\d+\.?\d*", line)
        if len(numbers) >= 3:
            rows.append([float(n) for n in numbers[:3]])
    return np.array(rows)


def _load_rotor_positions(path: str | Path) -> np.ndarray:
    mat = scipy.io.loadmat(str(path))
    return mat["rotorsPos"]


def _parse_recording_id(recording_id: str) -> dict[str, str | None]:
    parts = recording_id.split("_")
    result: dict[str, str | None] = {
        "flight_type": None,
        "source_type": None,
        "source_level": None,
        "room": None,
    }
    for part in parts:
        if part in {
            "free-flight",
            "hovering",
            "updown",
            "rectangle",
            "spinning",
            "silent-flight",
        }:
            result["flight_type"] = part
        elif part.startswith("room"):
            result["room"] = part
        elif "-" in part and part not in ("free-flight", "silent-flight"):
            subparts = part.split("-")
            if len(subparts) == 2 and subparts[0] in ("speech", "whitenoise"):
                result["source_type"] = subparts[0]
                result["source_level"] = subparts[1]
        elif part == "nosource":
            result["source_type"] = None
    return result


def _parse_motor_filename(filename: str) -> tuple[int | None, int | None]:
    match = re.match(r"Motor(\d)_(\d+)\.wav", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.match(r"allMotors_(\d+)\.wav", filename)
    if match:
        return None, int(match.group(1))
    return None, None


def discover_recordings(dregon_dir: Path) -> list[dict]:
    """Return list of sample dicts for all recordings found under *dregon_dir*.

    Each dict has string keys matching the old HuggingFace schema
    (``recording_id``, ``split``, ``audio_path``, ``audiots_path``,
    ``imu_path``, ``motors_path``, ``sourcepos_path``, …).
    """
    samples: list[dict] = []
    mic_pos_path = dregon_dir / "micPos.txt"
    coord_mat_path = dregon_dir / "coordinates.mat"
    if not mic_pos_path.exists() and coord_mat_path.exists():
        mat = scipy.io.loadmat(str(coord_mat_path))
        mic_pos_path = dregon_dir / "micPos_extracted.txt"
        np.savetxt(mic_pos_path, mat["micPos"])
    rotor_pos_path = str(coord_mat_path) if coord_mat_path.exists() else ""

    # In-flight recordings
    for split_name, recording_ids in SPLIT_RECORDINGS.items():
        for rid in recording_ids:
            rec_dir = dregon_dir / f"DREGON_{rid}"
            if not rec_dir.exists():
                continue
            wavs = list(rec_dir.glob("*.wav"))
            if not wavs:
                continue
            meta = _parse_recording_id(rid)
            sample: dict = {
                "recording_id": rid,
                "split": split_name,
                "flight_type": meta["flight_type"],
                "source_type": meta["source_type"],
                "source_level": meta["source_level"],
                "room": meta["room"],
                "motor_id": None,
                "motor_speed": None,
                "audio_path": str(wavs[0]),
                "audiots_path": None,
                "imu_path": None,
                "motors_path": None,
                "sourcepos_path": None,
                "mic_positions_path": str(mic_pos_path),
                "rotor_positions_path": rotor_pos_path,
            }
            base = f"DREGON_{rid}"
            for key, fname in [
                ("audiots_path", f"{base}_audiots.mat"),
                ("imu_path", f"{base}_imu.mat"),
                ("motors_path", f"{base}_motors.mat"),
                ("sourcepos_path", f"{base}_sourcepos.mat"),
            ]:
                p = rec_dir / fname
                if p.exists():
                    sample[key] = str(p)
            samples.append(sample)

    # Motor recordings
    motors_dir = dregon_dir / "DREGON_individual_motors_recordings"
    if motors_dir.exists():
        for wav_file in motors_dir.rglob("*.wav"):
            motor_id, motor_speed = _parse_motor_filename(wav_file.name)
            samples.append(
                {
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
            )

    # Clean source recordings
    clean_dir = dregon_dir / "clean_sources"
    if clean_dir.exists():
        for source_type in ("speech", "whitenoise", "chirps"):
            src_dir = clean_dir / source_type
            if not src_dir.exists():
                continue
            for wav_file in src_dir.rglob("*.wav"):
                samples.append(
                    {
                        "recording_id": f"clean_{source_type}_{wav_file.stem}",
                        "split": "clean_source",
                        "flight_type": None,
                        "source_type": source_type.rstrip("s"),
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
                )

    return samples


def download_dregon_dataset(
    data_dir: Path,
    *,
    download_clean_sources: bool = True,
    download_emitted_signals: bool = True,
) -> Path:
    """Download the full DREGON dataset (idempotent — skips existing files)."""
    dregon_dir = data_dir / "DREGON"
    dregon_dir.mkdir(parents=True, exist_ok=True)

    _download_file(DOWNLOAD_URLS["micPos.txt"], dregon_dir / "micPos.txt", "mic positions")

    if download_emitted_signals:
        ed = dregon_dir / "emitted_signals"
        ed.mkdir(exist_ok=True)
        _download_file(DOWNLOAD_URLS["emitted_speech"], ed / "2min_TIMIT.wav", "emitted speech")
        _download_file(
            DOWNLOAD_URLS["emitted_whitenoise"], ed / "2min_white_noise.wav", "emitted whitenoise"
        )

    if download_clean_sources:
        cd = dregon_dir / "clean_sources"
        for st in ("speech", "whitenoise", "chirps"):
            zp = cd / f"clean_{st}.zip"
            _download_file(DOWNLOAD_URLS[f"clean_{st}"], zp, f"clean {st}")
            _unpack_zip(zp, cd / st)

    for rid, url in DOWNLOAD_URLS.items():
        if rid in (
            "micPos.txt",
            "emitted_speech",
            "emitted_whitenoise",
            "motors",
        ) or rid.startswith("clean_"):
            continue
        rd = dregon_dir / f"DREGON_{rid}"
        if rd.exists() and any(rd.glob("*.wav")):
            continue
        zp = dregon_dir / f"DREGON_{rid}.zip"
        try:
            _download_file(url, zp, f"recording {rid}")
            _unpack_zip(zp, rd)
        except Exception as e:
            print(f"Warning: failed to download {rid}: {e}")

    motors_dir = dregon_dir / "DREGON_individual_motors_recordings"
    if not motors_dir.exists() or not any(motors_dir.rglob("*.wav")):
        zp = dregon_dir / "motors.zip"
        _download_file(DOWNLOAD_URLS["motors"], zp, "motor recordings")
        _unpack_zip(zp, motors_dir)

    return dregon_dir


# =============================================================================
# Geometry
# =============================================================================


def get_geometry(dregon_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mic_positions, rotor_positions)``.

    mic_positions: (8, 3)  — microphone xyz
    rotor_positions: (4, 3) — rotor xyz
    """
    mic_pos_path = dregon_dir / "micPos.txt"
    coord_mat_path = dregon_dir / "coordinates.mat"

    if mic_pos_path.exists():
        mic_pos = _parse_mic_positions_txt(mic_pos_path)
    elif coord_mat_path.exists():
        mic_pos = scipy.io.loadmat(str(coord_mat_path))["micPos"]
    else:
        raise FileNotFoundError("Neither micPos.txt nor coordinates.mat found")

    if coord_mat_path.exists():
        rotor_pos = scipy.io.loadmat(str(coord_mat_path))["rotorsPos"]
    else:
        r = 0.485 / 2
        angles = np.array([45, 135, 225, 315]) + 90
        offset = np.array([0.0115, 0.0, 0.1915])
        rotor_pos = np.zeros((4, 3))
        for i, angle in enumerate(angles):
            rotor_pos[i] = [r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle)), 0.1915]
        rotor_pos += offset

    return mic_pos, rotor_pos


# =============================================================================
# TimeFrame-native loading
# =============================================================================


def _mat_timestamps(mat_path: str | Path, field: str = "timestamps") -> np.ndarray:
    """Load a 1-D timestamp array from a .mat file, flattening as needed."""
    data = scipy.io.loadmat(str(mat_path))
    return data[field].flatten().astype(np.float64)


def load_timeframe(
    sample: dict,
    *,
    geometry: tuple[np.ndarray, np.ndarray] | None = None,
    target_sr: int | None = None,
) -> TimeFrame:
    """Load a single DREGON recording as a ``TimeFrame``.

    Parameters
    ----------
    sample : dict
        Sample dict as returned by ``discover_recordings``, with keys
        ``audio_path``, ``mic_positions_path``, ``rotor_positions_path``,
        and optional ``audiots_path``, ``imu_path``, ``motors_path``,
        ``sourcepos_path``.
    geometry : (mic_pos, rotor_pos) | None
        Pre-loaded geometry.  Loaded from paths in *sample* if ``None``.
    target_sr : int | None
        If given, resample audio to this rate before wrapping in UniformSeries.

    Returns
    -------
    TimeFrame
        Tracks: ``"audio"``, and optionally ``"motors_measured"``,
        ``"motors_command"``, ``"imu_accel"``, ``"imu_gyro"``,
        ``"source_position"``.
        Tags: ``recording_id``, ``split``, ``flight_type``, ``source_type``,
        ``source_level``, ``room``, ``motor_id``, ``motor_speed``,
        ``sample_rate``.
        global_data: ``mic_positions``, ``rotor_positions``.
    """
    # --- geometry -----------------------------------------------------------
    if geometry is not None:
        mic_pos, rotor_pos = geometry
    else:
        mic_pp = Path(sample["mic_positions_path"])
        mic_pos = (
            _parse_mic_positions_txt(mic_pp) if mic_pp.suffix == ".txt" else np.loadtxt(mic_pp)
        )
        rotor_pos = _load_rotor_positions(sample["rotor_positions_path"])

    # --- audio ---------------------------------------------------------------
    audio, sr = sf.read(sample["audio_path"])  # (N,) or (N, n_ch)
    audio = audio[np.newaxis, :] if audio.ndim == 1 else audio.T  # (n_ch, N), axis -1 = time

    # Resample if requested.  ``audio`` is (n_ch, N) with time on axis=-1, so
    # resample along axis=-1 directly — do NOT transpose, or librosa resamples
    # the (short) channel axis and loops over millions of tiny signals.
    if target_sr is not None and target_sr != sr:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=target_sr, axis=-1, res_type="soxr_hq"
        )
        sr = target_sr

    # Absolute start time — from audiots.mat if available, else 0
    if sample.get("audiots_path"):
        audiots = _mat_timestamps(sample["audiots_path"], "audio_timestamps")
        t0 = float(audiots[0])
    else:
        t0 = 0.0

    tracks: dict = {
        "audio": UniformSeries.from_samples(audio.astype(np.float32), sr, t_start=t0),
    }

    # --- motor data ----------------------------------------------------------
    if sample.get("motors_path"):
        motors_mat = scipy.io.loadmat(sample["motors_path"])
        ms = motors_mat["motor"]
        motor_ts = ms["timestamps"][0, 0].flatten().astype(np.float64)
        # Some recordings (room2) have only 'command', not 'measured'
        has_measured = "measured" in ms.dtype.names
        # EventSeries values are time-last (..., M); .mat stores (M, K) -> .T
        if has_measured:
            measured = ms["measured"][0, 0].astype(np.float32)  # (M, 4)
            tracks["motors_measured"] = EventSeries.from_events(
                motor_ts,
                values=measured.T,
                t_start=t0,  # (4, M)
            )
        command = ms["command"][0, 0].astype(np.float32)  # (M, 4)
        tracks["motors_command"] = EventSeries.from_events(
            motor_ts,
            values=command.T,
            t_start=t0,  # (4, M)
        )

    # --- IMU -----------------------------------------------------------------
    if sample.get("imu_path"):
        imu = scipy.io.loadmat(sample["imu_path"])["imu"]
        imu_ts = imu["timestamps"][0, 0].flatten().astype(np.float64)
        accel = imu["acceleration"][0, 0].astype(np.float32)  # (M, 3)
        gyro = imu["angular_velocity"][0, 0].astype(np.float32)  # (M, 3)
        tracks["imu_accel"] = EventSeries.from_events(imu_ts, values=accel.T, t_start=t0)
        tracks["imu_gyro"] = EventSeries.from_events(imu_ts, values=gyro.T, t_start=t0)

    # --- source position -----------------------------------------------------
    if sample.get("sourcepos_path"):
        sp = scipy.io.loadmat(sample["sourcepos_path"])["source_position"]
        sp_ts = sp["timestamps"][0, 0].flatten().astype(np.float64)
        sp_vals = np.column_stack(
            [
                sp["azimuth"][0, 0].flatten(),
                sp["elevation"][0, 0].flatten(),
                sp["distance"][0, 0].flatten(),
            ]
        ).astype(np.float32)  # (M, 3)
        tracks["source_position"] = EventSeries.from_events(sp_ts, values=sp_vals.T, t_start=t0)

    # --- tags ----------------------------------------------------------------
    tags: dict = {
        "recording_id": str(sample.get("recording_id", "")),
        "split": str(sample.get("split", "")),
        "flight_type": str(sample.get("flight_type") or ""),
        "source_type": str(sample.get("source_type") or ""),
        "source_level": str(sample.get("source_level") or ""),
        "room": str(sample.get("room") or ""),
        "motor_id": int(sample.get("motor_id") or -1),
        "motor_speed": int(sample.get("motor_speed") or -1),
        "sample_rate": int(sr),
    }

    # --- global_data ---------------------------------------------------------
    global_data: dict = {
        "mic_positions": mic_pos.astype(np.float64),
        "rotor_positions": rotor_pos.astype(np.float64),
    }

    return TimeFrame.from_tracks(
        tracks,
        t_start=t0,
        tags=tags,
        global_data=global_data,
    )


def load_dregon_timeframes(
    data_dir: Path | str,
    *,
    splits: list[str] | None = None,
    target_sr: int | None = None,
    download: bool = True,
) -> list[TimeFrame]:
    """Load all DREGON recordings in *splits* as a flat ``list[TimeFrame]``.

    Parameters
    ----------
    data_dir : Path
        Root data directory (contains ``DREGON/`` subdirectory).
    splits : list[str] | None
        Which splits to load (e.g. ``["in_flight_noise"]``).  ``None`` = all.
    target_sr : int | None
        Resample audio to this rate.
    download : bool
        Download missing data if ``True``.
    """
    data_dir = Path(data_dir)
    dregon_dir = data_dir / "DREGON"

    if download:
        download_dregon_dataset(data_dir)

    geometry = get_geometry(dregon_dir)
    all_samples = discover_recordings(dregon_dir)

    if splits is not None:
        split_set = set(splits)
        all_samples = [s for s in all_samples if s["split"] in split_set]

    frames: list[TimeFrame] = []
    for s in all_samples:
        try:
            tf = load_timeframe(s, geometry=geometry, target_sr=target_sr)
            frames.append(tf)
        except Exception as e:
            print(f"Warning: skipping {s.get('recording_id', '?')}: {e}")
    return frames


# =============================================================================
# Command-spike cleaning (preserved — used by downstream training scripts)
# =============================================================================


def _find_step_artifact_length(command: np.ndarray) -> int:
    """Detect initial constant-value logging artifact.

    Returns number of leading samples to discard, or 0 if no artifact.
    """
    n_samples = command.shape[-1]
    if n_samples < 2:
        return 0
    init_val = command[0, 0]
    if init_val < 30.0:
        return 0
    n_same = 0
    for j in range(n_samples):
        if command[0, j] == init_val:
            n_same += 1
        else:
            break
    return n_same if n_same >= 100 else 0


def clean_command_spikes(
    command: np.ndarray,
    kernel: int = 21,
) -> np.ndarray:
    """Clean DREGON commanded rotor speeds by removing known artifacts.

    1. **Step artifact** — leading block of identical high command values.
    2. **Takeoff spikes** — median filter removes impulsive outliers.

    Args:
        command: (4, n_samples) array of commanded rotor speeds (RPS, Hz).
            Time is the LAST axis (matches EventSeries values convention).
        kernel: Median filter kernel size along time axis (odd int, default 21).

    Returns:
        (4, n_samples) cleaned command array, same shape and dtype.
    """
    cleaned = command.copy()
    n_step = _find_step_artifact_length(command)
    if n_step > 0:
        cleaned[:, :n_step] = 0.0
    cleaned = median_filter(cleaned, size=(1, kernel), mode="reflect")
    return cleaned
