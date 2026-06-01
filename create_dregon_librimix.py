# coding: utf-8
"""
DREGON-LibriMix Dataset Synthesis Script (v2)

Creates synthetic mixtures from:
- Speech samples from LibriSpeech (train-clean-100)
- Drone noise from DREGON dataset with real command RPS data

Key improvements over v1:
- Uses all 8 microphone channels (each as independent noise source)
- 3-second samples (exactly 48k samples at 16 kHz)
- Command speeds (not measured) cleaned via clean_command_spikes
- Recording-level train/valid split (no overlap)
- Synthetic steady-state motor combinations (~20% of train)
- Optional low-level white noise mixed with speech
"""

import argparse
import json
import random
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from data_processing.dregon import (
    DREGONRecord,
    clean_command_spikes,
    discover_recordings,
    get_geometry,
    load_dregon_dataset,
    load_record_from_sample,
)

# =============================================================================
# Constants
# =============================================================================

SAMPLE_RATE = 16000
SAMPLE_DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_DURATION * SAMPLE_RATE)  # 48000
MOTOR_SAMPLE_RATE = 929.0  # Hz — default DREGON motor logging rate
NUM_ROTORS = 4

# Train: all in_flight_noise recordings
TRAIN_NOISE_SPLITS = ["in_flight_noise"]
# Valid: specific in_flight_source recordings
VALID_NOISE_RECORDING_IDS = [
    "free-flight_whitenoise-low_room1",
    "free-flight_speech-low_room1",
]


# =============================================================================
# Audio I/O helpers
# =============================================================================


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE, mono: bool = True) -> np.ndarray:
    """Load audio file, resample to target sample rate, convert to mono."""
    audio, sr = sf.read(path)

    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range based on peak."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def adjust_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or randomly crop audio to exact target length."""
    current_length = len(audio)

    if current_length > target_length:
        start = np.random.randint(0, current_length - target_length + 1)
        audio = audio[start : start + target_length]
    elif current_length < target_length:
        pad_length = target_length - current_length
        audio = np.pad(audio, (0, pad_length), mode="constant", constant_values=0)

    return audio


def calculate_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    speech_power = np.sum(speech ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(speech_power / max(noise_power, 1e-10))


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix speech and noise at a target SNR.

    Returns:
        Tuple of (mixture, scaled_speech, scaled_noise)
    """
    speech_power = np.sum(speech ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power > 0 and speech_power > 0:
        target_noise_power = speech_power / (10 ** (target_snr / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        noise = noise * scale

    mixture = speech + noise

    # Anti-clipping: if mixture peaks above 0.95, scale everything down
    max_val = np.abs(mixture).max()
    if max_val > 0.95:
        scale_factor = 0.95 / max_val
        mixture = mixture * scale_factor
        speech = speech * scale_factor
        noise = noise * scale_factor

    return mixture.astype(np.float32), speech.astype(np.float32), noise.astype(np.float32)


def generate_white_noise(length: int, snr_db: float, speech: np.ndarray) -> np.ndarray:
    """
    Generate white noise and mix with speech at specified SNR (noise below speech).

    Args:
        length: Number of samples
        snr_db: SNR of white noise relative to speech (positive = noise quieter)
        speech: Reference speech signal for power calculation

    Returns:
        speech + white_noise mixed at target SNR
    """
    speech_power = np.sum(speech ** 2)
    noise = np.random.randn(length).astype(np.float32)
    noise_power = np.sum(noise ** 2)

    if noise_power > 0 and speech_power > 0:
        target_noise_power = speech_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        noise = noise * scale

    return (speech + noise).astype(np.float32)


# =============================================================================
# DREGON record loading
# =============================================================================


def load_dregon_noise_records(
    dregon_dir: Path,
    recording_ids: list[str] | None = None,
    splits: list[str] | None = None,
    cache_dir: Path | None = None,
) -> list[DREGONRecord]:
    """
    Load DREGON records, expanding all 8 channels into separate records.

    Args:
        dregon_dir: Path to DREGON data directory
        recording_ids: Specific recording IDs to load (overrides splits)
        splits: Split names to load (used if recording_ids is None)
        cache_dir: Directory for caching resampled audio

    Returns:
        List of DREGONRecord objects (one per channel per recording)
    """
    if recording_ids is not None:
        # Load specific recordings by ID
        dataset = load_dregon_dataset(
            dregon_dir.parent,
            splits=["in_flight_source", "in_flight_noise"],
            download=False,
        )
        # Collect samples matching requested recording_ids
        all_samples = []
        for split_data in dataset.values():
            for sample in split_data:
                if sample["recording_id"] in recording_ids:
                    all_samples.append(sample)
    else:
        dataset = load_dregon_dataset(
            dregon_dir.parent,
            splits=splits or ["in_flight_noise"],
            download=False,
        )
        all_samples = []
        for split_data in dataset.values():
            all_samples.extend(list(split_data))

    geometry = get_geometry(dregon_dir)

    if cache_dir is None:
        cache_dir = dregon_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for sample in tqdm(all_samples, desc="Loading DREGON records"):
        record = load_record_from_sample(sample, geometry=geometry)

        # Resample audio if needed
        if SAMPLE_RATE != record.sample_rate:
            record = record.resample_audio(SAMPLE_RATE, cache_dir=cache_dir)

        # Skip records without motor data
        if record.motors is None:
            print(f"Warning: {record.recording_id} has no motor data, skipping")
            continue

        # Expand each channel into a separate record
        n_channels = record.audio.shape[1] if record.audio.ndim > 1 else 1
        for ch in range(n_channels):
            # Create a single-channel copy
            ch_audio = record.audio[:, ch : ch + 1]  # keep 2D shape
            ch_record = DREGONRecord(
                recording_id=f"{record.recording_id}_ch{ch}",
                split=record.split,
                mic_positions=record.mic_positions,
                rotor_positions=record.rotor_positions,
                audio=ch_audio,
                audio_timestamps=record.audio_timestamps,
                flight_type=record.flight_type,
                source_type=record.source_type,
                source_level=record.source_level,
                room=record.room,
                sample_rate=record.sample_rate,
                imu=record.imu,
                motors=record.motors,
                source_position=record.source_position,
            )
            records.append(ch_record)

    return records


# =============================================================================
# Noise chunk extraction with command RPS
# =============================================================================


def extract_noise_chunk_with_command_rps(
    record: DREGONRecord,
    duration_sec: float = SAMPLE_DURATION,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract a random noise chunk with aligned command RPS (cleaned) from a DREGON record.

    Uses command speeds (not measured), cleaned via clean_command_spikes.

    Args:
        record: DREGON record with audio and motor data
        duration_sec: Duration of chunk in seconds
        channel: Microphone channel index (0-based) within this record's audio

    Returns:
        Tuple of (audio, rps, metadata)
        - audio: (n_samples,) mono audio at record's sample rate
        - rps: (4, n_motor_samples) cleaned command RPS at native motor rate
        - metadata: dict with recording_id, start_time, motor_sample_rate
    """
    # Calculate valid time range where both audio and motor data exist
    audio_start_time = float(record.audio_timestamps[0])
    audio_end_time = float(record.audio_timestamps[-1])
    motor_start_time = float(record.motors.timestamps[0])
    motor_end_time = float(record.motors.timestamps[-1])

    # Valid range = intersection of audio and motor time ranges
    valid_start = max(audio_start_time, motor_start_time)
    valid_end = min(audio_end_time, motor_end_time)
    valid_duration = valid_end - valid_start

    if valid_duration < duration_sec:
        raise ValueError(
            f"Record {record.recording_id} has insufficient overlap: "
            f"{valid_duration:.1f}s < {duration_sec}s"
        )

    # Random start time within valid range (relative to audio start)
    rel_valid_start = valid_start - audio_start_time
    rel_valid_end = valid_end - audio_start_time - duration_sec
    start_sec = np.random.uniform(rel_valid_start, rel_valid_end)

    # Slice record
    sliced = record.slice_by_time(start_sec, start_sec + duration_sec)

    # Extract mono audio from specified channel
    audio = sliced.audio[:, channel] if sliced.audio.ndim > 1 else sliced.audio

    # Use COMMAND speeds (not measured), cleaned
    command = sliced.motors.command.copy()  # (n_motor_samples, 4)
    command_cleaned = clean_command_spikes(command)  # (n_motor_samples, 4)
    rps = command_cleaned.T.astype(np.float32)  # (4, n_motor_samples)

    # Estimate motor sample rate
    if len(sliced.motors.timestamps) > 1:
        motor_sr = 1.0 / np.median(np.diff(sliced.motors.timestamps.astype(float)))
    else:
        motor_sr = MOTOR_SAMPLE_RATE

    metadata = {
        "recording_id": record.recording_id,
        "start_time": start_sec,
        "duration": duration_sec,
        "motor_sample_rate": float(motor_sr),
        "channel": channel,
    }

    return audio.astype(np.float32), rps, metadata


# =============================================================================
# Synthetic motor combinations
# =============================================================================


# Cached per-channel SPL targets (computed once, then reused)
_SPL_TARGETS_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _load_all_motor_wavs(motors_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """
    Load all individual motor WAV files (keep all 8 channels).

    Returns:
        Dict mapping (motor_id, speed) -> audio array (n_samples, 8), float32, 44100 Hz
    """
    motor_wavs = {}
    for wav_path in motors_dir.rglob("Motor*_*.wav"):
        stem = wav_path.stem  # e.g., "Motor1_70"
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        motor_id = int(parts[0].replace("Motor", ""))
        speed = int(parts[1])
        audio, _ = sf.read(wav_path)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]  # Ensure 2D
        motor_wavs[(motor_id, speed)] = audio.astype(np.float32)
    return motor_wavs


def _compute_per_channel_spl_targets(
    motor_wavs: dict[tuple[int, int], np.ndarray],
    motors_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel SPL targets for motor combo normalization.

    Returns:
        Tuple of (mean_single_rms_per_channel, allmotors_70_rms_per_channel)
        Each is shape (8,) float64.
    """
    global _SPL_TARGETS_CACHE
    if _SPL_TARGETS_CACHE is not None:
        return _SPL_TARGETS_CACHE

    n_channels = 8

    # --- allMotors_70 RMS per channel ---
    allm_paths = list(motors_dir.rglob("allMotors_70.wav"))
    if allm_paths:
        allm, _ = sf.read(allm_paths[0])
        if allm.ndim == 1:
            allm = allm[:, np.newaxis]
        allm_rms = np.sqrt(np.mean(allm.astype(np.float64) ** 2, axis=0))  # (8,)
    else:
        allm_rms = np.full(n_channels, 0.0348, dtype=np.float64)

    # --- Mean RMS per channel across all single-motor recordings ---
    single_rms_per_ch = np.zeros((len(motor_wavs), n_channels), dtype=np.float64)
    for i, (key, audio) in enumerate(sorted(motor_wavs.items())):
        single_rms_per_ch[i] = np.sqrt(np.mean(audio.astype(np.float64) ** 2, axis=0))

    mean_single_rms = single_rms_per_ch.mean(axis=0)  # (8,)

    _SPL_TARGETS_CACHE = (mean_single_rms, allm_rms)
    return _SPL_TARGETS_CACHE


def create_motor_combo_sample(
    motor_wavs: dict[tuple[int, int], np.ndarray],
    motors_dir: Path,
    duration_sec: float = SAMPLE_DURATION,
    target_sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a synthetic steady-state rotor combination.

    Picks one microphone channel, then sums the SAME channel across different
    motors (with random speeds). Normalizes SPL to match the expected level
    for that number of rotors.

    Args:
        motor_wavs: Dict of (motor_id, speed) -> audio array (n_samples, 8)
        motors_dir: Path to motor recordings directory (for allMotors_70 reference)
        duration_sec: Target duration
        target_sr: Target sample rate

    Returns:
        Tuple of (audio, rps, metadata)
        - audio: (n_samples,) mono audio at target_sr (resampled if needed)
        - rps: (4, n_motor_samples) constant RPS, zeros for inactive rotors
        - metadata: dict with combo info
    """
    mean_single_rms_per_ch, allm_rms_per_ch = _compute_per_channel_spl_targets(
        motor_wavs, motors_dir
    )

    motor_sr = 44100  # Native sample rate of individual motor WAVs
    motor_target_length = int(duration_sec * motor_sr)  # e.g., 132300 for 3s
    target_length = int(duration_sec * target_sr)  # e.g., 48000 for 3s@16kHz
    n_motor_samples = int(duration_sec * MOTOR_SAMPLE_RATE)
    n_channels = mean_single_rms_per_ch.shape[0]

    # Pick a random microphone channel
    channel = np.random.randint(0, n_channels)

    # Randomly choose number of motors (1–4)
    num_motors = np.random.randint(1, NUM_ROTORS + 1)

    # Randomly choose which motor IDs to use (1-4), and random speeds (50-90)
    available_motors = list(range(1, NUM_ROTORS + 1))
    chosen_motors = random.sample(available_motors, num_motors)
    available_speeds = [50, 60, 70, 80, 90]

    combo_specs = []  # list of (motor_id, speed)
    for motor_id in chosen_motors:
        speed = random.choice(available_speeds)
        combo_specs.append((motor_id, speed))

    # Sum the SAME channel across different motors at native sample rate
    summed_audio = np.zeros(motor_target_length, dtype=np.float64)
    for motor_id, speed in combo_specs:
        key = (motor_id, speed)
        if key not in motor_wavs:
            raise KeyError(f"Motor {motor_id} at {speed} rps not found")
        wav = motor_wavs[key]  # (n_total, 8) at 44100 Hz
        ch_audio = wav[:, channel]  # (n_total,) at 44100 Hz

        # Randomly crop a segment at native rate
        if len(ch_audio) > motor_target_length:
            start = np.random.randint(0, len(ch_audio) - motor_target_length + 1)
            segment = ch_audio[start : start + motor_target_length]
        else:
            segment = np.pad(ch_audio, (0, motor_target_length - len(ch_audio)), mode="constant")
        summed_audio += segment

    # Normalize SPL per channel (at native rate)
    target_rms = mean_single_rms_per_ch[channel] + (
        allm_rms_per_ch[channel] - mean_single_rms_per_ch[channel]
    ) * (num_motors - 1) / 3.0
    actual_rms = float(np.sqrt(np.mean(summed_audio ** 2)))
    if actual_rms > 1e-10:
        summed_audio = summed_audio * (target_rms / actual_rms)

    audio = summed_audio.astype(np.float32)

    # Resample from motor_sr to target_sr
    if target_sr != motor_sr:
        audio = librosa.resample(audio, orig_sr=motor_sr, target_sr=target_sr)
        # Trim/pad to exact target length after resampling
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

    # Build RPS array: constant speeds for active motors, 0 for inactive
    rps = np.zeros((NUM_ROTORS, n_motor_samples), dtype=np.float32)
    for motor_id, speed in combo_specs:
        rps[motor_id - 1, :] = float(speed)

    metadata = {
        "num_motors": num_motors,
        "combo": [(int(m), int(s)) for m, s in combo_specs],
        "channel": int(channel),
        "motor_sample_rate": MOTOR_SAMPLE_RATE,
        "target_rms": float(target_rms),
        "actual_rms_before_norm": float(actual_rms),
    }

    return audio, rps, metadata


# =============================================================================
# Main dataset creation
# =============================================================================


def create_dregon_librimix(
    speech_dir: Path,
    dregon_dir: Path,
    output_dir: Path,
    num_samples: int,
    sample_duration: float = SAMPLE_DURATION,
    sample_rate: int = SAMPLE_RATE,
    snr_range: tuple[float, float] = (-30.0, 0.0),
    split: str = "train",
    seed: int = 42,
    motor_combo_fraction: float = 0.0,
    white_noise_prob: float = 0.0,
    white_noise_snr: float = 30.0,
):
    """
    Create the DREGON-LibriMix dataset with command RPS data.

    Args:
        speech_dir: Directory containing LibriSpeech audio files
        dregon_dir: Path to DREGON dataset directory
        output_dir: Output directory for the dataset
        num_samples: Number of samples to generate
        sample_duration: Duration of each sample in seconds (default 3.0)
        sample_rate: Target sample rate
        snr_range: Range of target SNRs in dB
        split: 'train' or 'valid'
        seed: Random seed
        motor_combo_fraction: Fraction of samples from synthetic motor combos (0.0–1.0)
        white_noise_prob: Probability of adding low-level white noise to speech
        white_noise_snr: SNR of white noise relative to speech (dB, positive = quieter)
    """
    random.seed(seed)
    np.random.seed(seed)

    target_length = int(sample_duration * sample_rate)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # --- Find speech files ---
    speech_files = []
    for ext in ["*.wav", "*.flac"]:
        speech_files.extend(glob(str(speech_dir / "**" / ext), recursive=True))

    if len(speech_files) == 0:
        raise ValueError(f"No speech files found in {speech_dir}")

    print(f"Found {len(speech_files)} speech files")

    # --- Load noise records ---
    cache_dir = dregon_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        noise_records = load_dregon_noise_records(
            dregon_dir,
            splits=TRAIN_NOISE_SPLITS,
            cache_dir=cache_dir,
        )
    else:
        noise_records = load_dregon_noise_records(
            dregon_dir,
            recording_ids=VALID_NOISE_RECORDING_IDS,
            cache_dir=cache_dir,
        )

    if len(noise_records) == 0:
        raise ValueError("No valid DREGON records with motor data found")

    total_noise_duration = sum(r.duration for r in noise_records)
    print(f"Loaded {len(noise_records)} noise records ({total_noise_duration:.1f}s total)")

    # Identify unique base recordings (strip _chN suffix) for reporting
    base_recordings = set(r.recording_id.rsplit("_ch", 1)[0] for r in noise_records)
    print(f"  Unique base recordings: {sorted(base_recordings)}")

    # --- Load motor WAVs for synthetic combos (train only) ---
    motor_wavs = None
    motors_dir = None
    num_motor_combo_samples = 0

    if motor_combo_fraction > 0 and split == "train":
        motors_dir = dregon_dir / "DREGON_individual_motors_recordings"
        motor_wavs = _load_all_motor_wavs(motors_dir)

        if len(motor_wavs) == 0:
            print("Warning: No motor WAVs found, disabling motor combos")
            motor_combo_fraction = 0.0
        else:
            num_motor_combo_samples = int(num_samples * motor_combo_fraction)
            # Pre-compute SPL targets
            ms_rms, am_rms = _compute_per_channel_spl_targets(motor_wavs, motors_dir)
            print(
                f"Motor combos: {num_motor_combo_samples} samples ({motor_combo_fraction:.0%}), "
                f"{len(motor_wavs)} motor WAVs loaded"
            )
            # Report SPL targets averaged across channels
            print(
                f"  SPL targets (mean across channels): "
                f"1-motor RMS={ms_rms.mean():.6f}, "
                f"4-motor RMS={am_rms.mean():.6f}"
            )

    num_inflight_samples = num_samples - num_motor_combo_samples

    # --- Generate samples ---
    metadata_list = []

    for idx in tqdm(range(num_samples), desc=f"Creating {split} samples"):
        sample_id = f"sample_{idx:05d}"
        sample_dir = split_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        # Decide sample type: motor combo or in-flight noise
        is_motor_combo = (idx < num_motor_combo_samples)

        if is_motor_combo:
            # --- Synthetic motor combo ---
            noise, rps, noise_meta = create_motor_combo_sample(
                motor_wavs,
                motors_dir=motors_dir,
                duration_sec=sample_duration,
                target_sr=sample_rate,
            )
            noise = adjust_length(noise, target_length)
            noise = normalize_audio(noise)
            noise_source_id = f"motor_combo_{noise_meta['combo']}"
        else:
            # --- In-flight noise ---
            record = random.choice(noise_records)

            # Records are already single-channel; always use channel 0
            ch = 0

            # Try to extract a valid chunk
            noise = None
            for attempt in range(20):
                try:
                    noise, rps, noise_meta = extract_noise_chunk_with_command_rps(
                        record, duration_sec=sample_duration, channel=ch
                    )
                    break
                except ValueError:
                    record = random.choice(noise_records)
            else:
                raise ValueError("Could not find a valid noise chunk after 20 attempts")

            noise = adjust_length(noise, target_length)
            noise = normalize_audio(noise)
            noise_source_id = noise_meta["recording_id"]

        # --- Speech ---
        speech_path = random.choice(speech_files)
        speech = load_audio(speech_path, target_sr=sample_rate, mono=True)
        speech = adjust_length(speech, target_length)
        speech = normalize_audio(speech)

        # --- Optional white noise mixed into speech ---
        if white_noise_prob > 0 and random.random() < white_noise_prob:
            speech = generate_white_noise(target_length, white_noise_snr, speech)
            speech = normalize_audio(speech)

        # --- Random target SNR and mix ---
        target_snr = np.random.uniform(snr_range[0], snr_range[1])
        mixture, speech_scaled, noise_scaled = mix_at_snr(speech, noise, target_snr)
        actual_snr = calculate_snr(speech_scaled, noise_scaled)

        # --- Save files ---
        sf.write(sample_dir / "vocals.wav", speech_scaled, sample_rate)
        sf.write(sample_dir / "noise.wav", noise_scaled, sample_rate)
        sf.write(sample_dir / "mixture.wav", mixture, sample_rate)
        np.save(sample_dir / "rps.npy", rps)

        # --- Record metadata ---
        metadata_list.append(
            {
                "id": sample_id,
                "input_snr": float(actual_snr),
                "target_snr": float(target_snr),
                "speech_source": str(Path(speech_path).name),
                "noise_source": noise_source_id,
                "noise_start_time": noise_meta.get("start_time", 0.0),
                "motor_sample_rate": noise_meta.get(
                    "motor_sample_rate", MOTOR_SAMPLE_RATE
                ),
                "rps_shape": list(rps.shape),
                "is_motor_combo": is_motor_combo,
            }
        )

    # --- Save metadata ---
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}

    all_metadata[split] = metadata_list

    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Created {num_samples} {split} samples in {split_dir}")
    print(f"Metadata saved to {metadata_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Create DREGON-LibriMix dataset v2 with command RPS data"
    )
    parser.add_argument(
        "--speech_dir",
        type=Path,
        default=Path("data/librispeech/LibriSpeech/train-clean-100"),
        help="Path to LibriSpeech directory",
    )
    parser.add_argument(
        "--dregon_dir",
        type=Path,
        default=Path("data/DREGON"),
        help="Path to DREGON dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/DREGON-LM"),
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=6000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num_valid",
        type=int,
        default=600,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=SAMPLE_DURATION,
        help="Duration of each sample in seconds",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=SAMPLE_RATE,
        help="Target sample rate",
    )
    parser.add_argument(
        "--snr_min",
        type=float,
        default=-30.0,
        help="Minimum SNR in dB",
    )
    parser.add_argument(
        "--snr_max",
        type=float,
        default=0.0,
        help="Maximum SNR in dB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--motor_combo_fraction",
        type=float,
        default=0.2,
        help="Fraction of train samples from synthetic motor combos (0.0-1.0)",
    )
    parser.add_argument(
        "--white_noise_prob",
        type=float,
        default=0.0,
        help="Probability of adding white noise to speech",
    )
    parser.add_argument(
        "--white_noise_snr",
        type=float,
        default=30.0,
        help="SNR of additive white noise relative to speech (dB, positive=quieter)",
    )

    args = parser.parse_args()

    # --- Create training set ---
    print("=" * 60)
    print("Creating TRAINING set...")
    print("=" * 60)
    create_dregon_librimix(
        speech_dir=args.speech_dir,
        dregon_dir=args.dregon_dir,
        output_dir=args.output_dir,
        num_samples=args.num_train,
        sample_duration=args.duration,
        sample_rate=args.sample_rate,
        snr_range=(args.snr_min, args.snr_max),
        split="train",
        seed=args.seed,
        motor_combo_fraction=args.motor_combo_fraction,
        white_noise_prob=args.white_noise_prob,
        white_noise_snr=args.white_noise_snr,
    )

    # --- Create validation set ---
    print()
    print("=" * 60)
    print("Creating VALIDATION set...")
    print("=" * 60)
    create_dregon_librimix(
        speech_dir=args.speech_dir,
        dregon_dir=args.dregon_dir,
        output_dir=args.output_dir,
        num_samples=args.num_valid,
        sample_duration=args.duration,
        sample_rate=args.sample_rate,
        snr_range=(args.snr_min, args.snr_max),
        split="valid",
        seed=args.seed + 1,  # Different seed for validation
        motor_combo_fraction=0.0,  # No motor combos in validation
        white_noise_prob=args.white_noise_prob,
        white_noise_snr=args.white_noise_snr,
    )

    print()
    print("=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
