# coding: utf-8
"""
DREGON-LibriMix Dataset Synthesis Script

Creates synthetic mixtures from:
- Speech samples from LibriSpeech
- Drone noise from DREGON dataset (with real RPS/rotor speed data)

Unlike DN-LM, this dataset includes real rotor speed (RPS) data aligned with
each audio chunk, enabling training of RPS-informed speech enhancement models.
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
    NoiseSegment,
    discover_recordings,
    get_geometry,
    load_dregon_dataset,
    load_record_from_sample,
)


def load_audio(path: str | Path, target_sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(path)

    if mono and len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def adjust_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Adjust audio length by padding or random cropping."""
    current_length = len(audio)

    if current_length > target_length:
        start = np.random.randint(0, current_length - target_length + 1)
        audio = audio[start:start + target_length]
    elif current_length < target_length:
        pad_length = target_length - current_length
        audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    return audio


def calculate_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    speech_power = np.sum(speech ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(speech_power / noise_power)


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix speech and noise at a target SNR.

    Returns:
        Tuple of (mixture, speech, scaled_noise)
    """
    speech_power = np.sum(speech ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power > 0 and speech_power > 0:
        target_noise_power = speech_power / (10 ** (target_snr / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        noise = noise * scale

    mixture = speech + noise

    # Prevent clipping
    max_val = np.abs(mixture).max()
    if max_val > 0.95:
        scale_factor = 0.95 / max_val
        mixture = mixture * scale_factor
        speech = speech * scale_factor
        noise = noise * scale_factor

    return mixture, speech, noise


def load_dregon_noise_records(
    dregon_dir: Path,
    target_sr: int,
    cache_dir: Path | None = None,
) -> list[DREGONRecord]:
    """
    Load DREGON in_flight_noise recordings resampled to target sample rate.

    Returns list of DREGONRecord objects with motor data.
    """
    dataset = load_dregon_dataset(
        dregon_dir.parent,
        splits=["in_flight_noise"],
        download=True,
    )
    geometry = get_geometry(dregon_dir)

    records = []
    for sample in tqdm(dataset["in_flight_noise"], desc="Loading DREGON records"):
        record = load_record_from_sample(sample, geometry=geometry)

        # Resample audio if needed
        if target_sr != record.sample_rate:
            record = record.resample_audio(target_sr, cache_dir=cache_dir)

        if record.motors is not None:
            records.append(record)
        else:
            print(f"Warning: {record.recording_id} has no motor data, skipping")

    return records


def extract_noise_chunk_with_rps(
    record: DREGONRecord,
    duration_sec: float,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract a random noise chunk with aligned RPS from a DREGON record.

    Args:
        record: DREGON record with audio and motor data
        duration_sec: Duration of chunk in seconds
        channel: Microphone channel to use (0-7)

    Returns:
        Tuple of (audio, rps, metadata)
        - audio: (n_samples,) mono audio at record's sample rate
        - rps: (4, n_motor_samples) rotor speeds at native rate
        - metadata: dict with recording_id, start_time, etc.
    """
    # Calculate valid time range where both audio and motor data exist
    audio_start_time = record.audio_timestamps[0]
    audio_end_time = record.audio_timestamps[-1]
    motor_start_time = record.motors.timestamps[0]
    motor_end_time = record.motors.timestamps[-1]

    # Valid range is intersection of audio and motor time ranges
    valid_start = max(audio_start_time, motor_start_time)
    valid_end = min(audio_end_time, motor_end_time)
    valid_duration = valid_end - valid_start

    if valid_duration < duration_sec:
        raise ValueError(
            f"Record {record.recording_id} has insufficient overlap between audio and motor data: "
            f"{valid_duration:.1f}s < {duration_sec}s"
        )

    # Random start time within valid range (relative to audio start)
    rel_valid_start = valid_start - audio_start_time
    rel_valid_end = valid_end - audio_start_time - duration_sec
    start_sec = np.random.uniform(rel_valid_start, rel_valid_end)

    # Slice record
    sliced = record.slice_by_time(start_sec, start_sec + duration_sec)

    # Extract mono audio
    audio = sliced.audio[:, channel] if sliced.audio.ndim > 1 else sliced.audio

    # Extract RPS at native rate
    rps = sliced.motors.measured.T  # (4, n_motor_samples)

    # Estimate motor sample rate for metadata
    if len(sliced.motors.timestamps) > 1:
        motor_sr = 1.0 / np.median(np.diff(sliced.motors.timestamps))
    else:
        motor_sr = 929.0

    metadata = {
        "recording_id": record.recording_id,
        "start_time": start_sec,
        "duration": duration_sec,
        "motor_sample_rate": motor_sr,
        "channel": channel,
    }

    return audio.astype(np.float32), rps.astype(np.float32), metadata


def create_dregon_librimix(
    speech_dir: Path,
    dregon_dir: Path,
    output_dir: Path,
    num_samples: int,
    sample_duration: float = 1.0,
    sample_rate: int = 16000,
    snr_range: tuple[float, float] = (-30.0, 0.0),
    split: str = "train",
    seed: int = 42,
    channel: int = 0,
):
    """
    Create the DREGON-LibriMix dataset with RPS data.

    Args:
        speech_dir: Directory containing LibriSpeech audio files
        dregon_dir: Path to DREGON dataset directory
        output_dir: Output directory for the dataset
        num_samples: Number of samples to generate
        sample_duration: Duration of each sample in seconds
        sample_rate: Target sample rate
        snr_range: Range of target SNRs in dB
        split: 'train' or 'valid'
        seed: Random seed
        channel: DREGON microphone channel to use (0-7)
    """
    random.seed(seed)
    np.random.seed(seed)

    target_length = int(sample_duration * sample_rate)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Find speech files
    speech_files = []
    for ext in ['*.wav', '*.flac']:
        speech_files.extend(glob(str(speech_dir / '**' / ext), recursive=True))

    if len(speech_files) == 0:
        raise ValueError(f"No speech files found in {speech_dir}")

    print(f"Found {len(speech_files)} speech files")

    # Load DREGON records
    cache_dir = dregon_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = load_dregon_noise_records(dregon_dir, sample_rate, cache_dir)

    if len(records) == 0:
        raise ValueError(f"No valid DREGON records with motor data found")

    total_noise_duration = sum(r.duration for r in records)
    print(f"Loaded {len(records)} DREGON records ({total_noise_duration:.1f}s total)")

    # Generate samples
    metadata_list = []

    for idx in tqdm(range(num_samples), desc=f"Creating {split} samples"):
        sample_id = f"sample_{idx:05d}"
        sample_dir = split_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        # Random speech
        speech_path = random.choice(speech_files)
        speech = load_audio(speech_path, target_sr=sample_rate, mono=True)
        speech = adjust_length(speech, target_length)
        speech = normalize_audio(speech)

        # Random noise chunk with RPS
        record = random.choice(records)
        try:
            noise, rps, noise_meta = extract_noise_chunk_with_rps(
                record, sample_duration, channel=channel
            )
        except ValueError:
            # Record too short, try another
            for _ in range(10):
                record = random.choice(records)
                try:
                    noise, rps, noise_meta = extract_noise_chunk_with_rps(
                        record, sample_duration, channel=channel
                    )
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Could not find a valid noise chunk after 10 attempts")

        noise = adjust_length(noise, target_length)
        noise = normalize_audio(noise)

        # Random target SNR
        target_snr = np.random.uniform(snr_range[0], snr_range[1])

        # Mix
        mixture, speech_scaled, noise_scaled = mix_at_snr(speech, noise, target_snr)

        # Calculate actual SNR
        actual_snr = calculate_snr(speech_scaled, noise_scaled)

        # Save audio files
        sf.write(sample_dir / "vocals.wav", speech_scaled, sample_rate)
        sf.write(sample_dir / "noise.wav", noise_scaled, sample_rate)
        sf.write(sample_dir / "mixture.wav", mixture, sample_rate)

        # Save RPS data
        np.save(sample_dir / "rps.npy", rps)

        # Record metadata
        metadata_list.append({
            "id": sample_id,
            "input_snr": float(actual_snr),
            "target_snr": float(target_snr),
            "speech_source": str(Path(speech_path).name),
            "noise_source": noise_meta["recording_id"],
            "noise_start_time": noise_meta["start_time"],
            "motor_sample_rate": noise_meta["motor_sample_rate"],
            "rps_shape": list(rps.shape),
        })

    # Save metadata
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


def main():
    parser = argparse.ArgumentParser(
        description="Create DREGON-LibriMix dataset with RPS data"
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
        default=1.0,
        help="Duration of each sample in seconds",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
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
        "--channel",
        type=int,
        default=0,
        help="DREGON microphone channel (0-7)",
    )

    args = parser.parse_args()

    # Create training set
    print("=" * 60)
    print("Creating training set...")
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
        channel=args.channel,
    )

    # Create validation set
    print("=" * 60)
    print("Creating validation set...")
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
        channel=args.channel,
    )

    print("=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
