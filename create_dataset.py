# coding: utf-8
"""
DroneNoise-LibriMix (DN-LM) Dataset Synthesis Script

Based on the paper: "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement"

This script creates the DN-LM dataset by mixing:
- Speech samples from LibriSpeech
- Drone noise samples from a local directory or a Hugging Face dataset

Reference paper section 3.5 for methodology.
"""

import os
import json
import random
import argparse
import numpy as np
import soundfile as sf
import librosa
from glob import glob
from tqdm import tqdm
from pathlib import Path


HF_DATASET_PREFIX = "hf:"
HF_LOCAL_PREFIX = "hf-local:"
HF_DRONE_LABEL = 1


def load_audio(path, target_sr=16000, mono=True):
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(path)

    # Convert to mono if needed
    if mono and len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def load_audio_from_hf_item(item, target_sr=16000, mono=True):
    """Load audio from a Hugging Face dataset item."""
    audio = item.get("audio")
    if not isinstance(audio, dict) or "array" not in audio:
        raise ValueError("Hugging Face item does not contain audio array")
    data = np.asarray(audio["array"], dtype=np.float32)
    sr = audio.get("sampling_rate", target_sr)

    # Convert to mono if needed
    if mono and len(data.shape) > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data


def normalize_audio(audio):
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def _is_hf_dataset_source(noise_dir):
    return isinstance(noise_dir, str) and noise_dir.startswith(HF_DATASET_PREFIX)


def _is_hf_local_source(noise_dir):
    return isinstance(noise_dir, str) and noise_dir.startswith(HF_LOCAL_PREFIX)


def _parse_hf_dataset_name(noise_dir):
    return noise_dir[len(HF_DATASET_PREFIX):].strip()


def _parse_hf_local_path(noise_dir):
    return noise_dir[len(HF_LOCAL_PREFIX):].strip()


def _select_hf_split(dataset_dict, split_name):
    if not hasattr(dataset_dict, "keys"):
        return dataset_dict, "train"
    desired = split_name.lower()
    if desired in ("valid", "validation"):
        candidates = ["validation", "valid", "test", "train"]
    elif desired == "test":
        candidates = ["test", "validation", "valid", "train"]
    else:
        candidates = ["train", "training", "all", "test"]
    for name in candidates:
        if name in dataset_dict:
            return dataset_dict[name], name
    first = next(iter(dataset_dict.keys()))
    return dataset_dict[first], first


def _load_hf_drone_dataset(dataset_name, sample_rate, split_name):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ValueError(
            "datasets is required for Hugging Face noise sources. "
            "Install it with: uv add datasets"
        ) from exc

    dataset_dict = load_dataset(dataset_name)
    dataset, used_split = _select_hf_split(dataset_dict, split_name)
    if "label" not in dataset.features:
        raise ValueError(f"Hugging Face dataset '{dataset_name}' has no 'label' column")
    dataset = dataset.filter(lambda item: item["label"] == HF_DRONE_LABEL)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

    if len(dataset) == 0:
        raise ValueError(f"No samples with label {HF_DRONE_LABEL} in '{dataset_name}'")

    return dataset, used_split


def _load_hf_dataset_from_disk(dataset_path, sample_rate, split_name):
    try:
        from datasets import Audio, load_from_disk
    except ImportError as exc:
        raise ValueError(
            "datasets is required for Hugging Face noise sources. "
            "Install it with: uv add datasets"
        ) from exc

    dataset_dict = load_from_disk(dataset_path)
    if hasattr(dataset_dict, "keys"):  # DatasetDict
        dataset, used_split = _select_hf_split(dataset_dict, split_name)
    else:
        dataset = dataset_dict
        used_split = "train"
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    return dataset, used_split


def _build_hf_dataset_from_files(noise_files, sample_rate):
    try:
        from datasets import Audio, Dataset
    except ImportError as exc:
        raise ValueError(
            "datasets is required for Hugging Face noise sources. "
            "Install it with: uv add datasets"
        ) from exc
    dataset = Dataset.from_dict({"audio": [str(path) for path in noise_files]})
    return dataset.cast_column("audio", Audio(sampling_rate=sample_rate))


def _normalize_noise_sources(noise_dir):
    if isinstance(noise_dir, (list, tuple)):
        return list(noise_dir)
    return [noise_dir]


def adjust_length(audio, target_length):
    """Adjust audio length by padding or cropping."""
    current_length = len(audio)

    if current_length > target_length:
        # Random crop
        start = np.random.randint(0, current_length - target_length + 1)
        audio = audio[start:start + target_length]
    elif current_length < target_length:
        # Zero padding
        pad_length = target_length - current_length
        audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    return audio


def apply_distance_attenuation(audio, distance):
    """
    Apply distance attenuation based on free-field sound propagation model.
    α = 1/d (inverse distance law)
    """
    attenuation = 1.0 / distance
    return audio * attenuation


def calculate_snr(speech, noise):
    """Calculate Signal-to-Noise Ratio in dB."""
    speech_power = np.sum(speech ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(speech_power / noise_power)
    return snr


def mix_audio(speech, noise, target_snr=None):
    """
    Mix speech and noise signals.
    If target_snr is provided, adjust noise level to achieve it.
    """
    if target_snr is not None:
        # Calculate current powers
        speech_power = np.sum(speech ** 2)
        noise_power = np.sum(noise ** 2)

        if noise_power > 0 and speech_power > 0:
            # Calculate required noise scale
            # SNR = 10 * log10(speech_power / noise_power)
            # noise_power_new = speech_power / 10^(SNR/10)
            target_noise_power = speech_power / (10 ** (target_snr / 10))
            scale = np.sqrt(target_noise_power / noise_power)
            noise = noise * scale

    mixture = speech + noise
    return mixture, speech, noise


def create_dataset(
    speech_dir,
    noise_dir,
    output_dir,
    num_samples,
    sample_duration=1.0,
    sample_rate=16000,
    speech_distance_range=(5, 20),
    noise_distance=0.5,
    target_snr_range=(-30, 0),
    split='train',
    seed=42
):
    """
    Create the DN-LM dataset.

    Args:
        speech_dir: Directory containing LibriSpeech audio files
        noise_dir: Directory containing drone audio files or hf:<dataset_name>
        output_dir: Output directory for the dataset
        num_samples: Number of samples to generate
        sample_duration: Duration of each sample in seconds
        sample_rate: Target sample rate
        speech_distance_range: Range of distances for speech source (meters)
        noise_distance: Fixed distance for noise source (meters)
        target_snr_range: Range of target SNRs in dB (None for natural mixing)
        split: 'train' or 'valid'
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    target_length = int(sample_duration * sample_rate)

    hf_noise_dataset = None
    hf_dataset_names = []
    noise_sources = _normalize_noise_sources(noise_dir)

    # Find all audio files
    speech_files = []
    for ext in ['*.wav', '*.flac']:
        speech_files.extend(glob(os.path.join(speech_dir, '**', ext), recursive=True))

    noise_files = []
    hf_datasets = []
    for source in noise_sources:
        if _is_hf_dataset_source(source):
            dataset_name = _parse_hf_dataset_name(source)
            dataset, used_split = _load_hf_drone_dataset(dataset_name, sample_rate, split)
            hf_dataset_names.append(f"{dataset_name}:{used_split}")
            hf_datasets.append(dataset)
        elif _is_hf_local_source(source):
            dataset_path = _parse_hf_local_path(source)
            dataset, used_split = _load_hf_dataset_from_disk(dataset_path, sample_rate, split)
            hf_dataset_names.append(f"{dataset_path}:{used_split}")
            hf_datasets.append(dataset)
        else:
            for ext in ['*.wav', '*.WAV', '*.flac', '*.FLAC', '*.mp3', '*.MP3', '*.ogg', '*.OGG']:
                noise_files.extend(glob(os.path.join(source, '**', ext), recursive=True))

    if len(speech_files) == 0:
        raise ValueError(f"No speech files found in {speech_dir}")
    if not hf_datasets and len(noise_files) == 0:
        raise ValueError(f"No noise files found in {noise_dir}")

    if hf_datasets:
        if noise_files:
            hf_datasets.append(_build_hf_dataset_from_files(noise_files, sample_rate))
        from datasets import concatenate_datasets

        hf_noise_dataset = concatenate_datasets(hf_datasets)

    if hf_noise_dataset is None:
        print(f"Found {len(speech_files)} speech files and {len(noise_files)} noise files")
    else:
        print(
            f"Found {len(speech_files)} speech files and "
            f"{len(hf_noise_dataset)} drone samples in {', '.join(hf_dataset_names)}"
        )

    # Create output directory
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    metadata = []

    for i in tqdm(range(num_samples), desc=f"Creating {split} samples"):
        sample_id = f"sample_{i:05d}"
        sample_dir = os.path.join(split_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Load random speech and noise
        speech_path = random.choice(speech_files)
        noise_path = None
        noise_item = None
        noise_index = None
        if hf_noise_dataset is None:
            noise_path = random.choice(noise_files)
        else:
            noise_index = random.randrange(len(hf_noise_dataset))
            noise_item = hf_noise_dataset[noise_index]

        try:
            speech = load_audio(speech_path, target_sr=sample_rate)
            if hf_noise_dataset is None:
                noise = load_audio(noise_path, target_sr=sample_rate)
            else:
                noise = load_audio_from_hf_item(noise_item, target_sr=sample_rate)
        except Exception as e:
            print(f"Error loading audio: {e}")
            continue

        # Adjust length
        speech = adjust_length(speech, target_length)
        noise = adjust_length(noise, target_length)

        # Normalize
        speech = normalize_audio(speech)
        noise = normalize_audio(noise)

        # Apply distance attenuation
        speech_distance = random.uniform(*speech_distance_range)
        speech_attenuated = apply_distance_attenuation(speech, speech_distance)
        noise_attenuated = apply_distance_attenuation(noise, noise_distance)

        # Mix with target SNR or natural
        if target_snr_range:
            target_snr = random.uniform(*target_snr_range)
            mixture, speech_final, noise_final = mix_audio(
                speech_attenuated, noise_attenuated, target_snr=target_snr
            )
        else:
            mixture, speech_final, noise_final = mix_audio(
                speech_attenuated, noise_attenuated, target_snr=None
            )

        # Calculate actual SNR
        actual_snr = calculate_snr(speech_final, noise_final)

        # Normalize mixture to prevent clipping
        max_val = max(np.abs(mixture).max(), np.abs(speech_final).max(), np.abs(noise_final).max())
        if max_val > 1.0:
            scale = 0.95 / max_val
            mixture *= scale
            speech_final *= scale
            noise_final *= scale

        # Save audio files
        sf.write(os.path.join(sample_dir, 'vocals.wav'), speech_final, sample_rate)
        sf.write(os.path.join(sample_dir, 'noise.wav'), noise_final, sample_rate)
        sf.write(os.path.join(sample_dir, 'mixture.wav'), mixture, sample_rate)

        noise_source = (
            os.path.basename(noise_path)
            if noise_path
            else f"hf#idx={noise_index}"
        )
        metadata.append({
            'id': sample_id,
            'input_snr': float(actual_snr),
            'speech_source': os.path.basename(speech_path),
            'noise_source': noise_source,
            'speech_distance': speech_distance,
        })

    # Save metadata
    metadata_file = os.path.join(split_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({split: metadata}, f, indent=2)

    print(f"Created {len(metadata)} samples in {split_dir}")
    print(f"Metadata saved to {metadata_file}")

    # Print SNR statistics
    snrs = [m['input_snr'] for m in metadata if not np.isinf(m['input_snr'])]
    print(f"SNR statistics: mean={np.mean(snrs):.2f} dB, std={np.std(snrs):.2f} dB")
    print(f"SNR range: [{min(snrs):.2f}, {max(snrs):.2f}] dB")


def main():
    parser = argparse.ArgumentParser(description='Create DroneNoise-LibriMix Dataset')

    parser.add_argument('--speech_dir', type=str, required=True,
                        help='Directory containing LibriSpeech audio files')
    parser.add_argument('--noise_dir', type=str, required=True,
                        help='Directory containing drone audio files or hf:<dataset_name> or hf-local:<path>')
    parser.add_argument('--output_dir', type=str, default='./datasets/DN-LM',
                        help='Output directory for the dataset')
    parser.add_argument('--train_samples', type=int, default=6480,
                        help='Number of training samples (90% of 2 hours at 1s each)')
    parser.add_argument('--valid_samples', type=int, default=720,
                        help='Number of validation samples (10% of 2 hours at 1s each)')
    parser.add_argument('--sample_duration', type=float, default=1.0,
                        help='Duration of each sample in seconds')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Target sample rate')
    parser.add_argument('--snr_min', type=float, default=-30,
                        help='Minimum SNR in dB')
    parser.add_argument('--snr_max', type=float, default=0,
                        help='Maximum SNR in dB')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create training set
    print("\n=== Creating Training Set ===")
    create_dataset(
        speech_dir=args.speech_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        num_samples=args.train_samples,
        sample_duration=args.sample_duration,
        sample_rate=args.sample_rate,
        target_snr_range=(args.snr_min, args.snr_max),
        split='train',
        seed=args.seed
    )

    # Create validation set
    print("\n=== Creating Validation Set ===")
    create_dataset(
        speech_dir=args.speech_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        num_samples=args.valid_samples,
        sample_duration=args.sample_duration,
        sample_rate=args.sample_rate,
        target_snr_range=(args.snr_min, args.snr_max),
        split='valid',
        seed=args.seed + 1  # Different seed for validation
    )

    print("\n=== Dataset Creation Complete ===")
    print(f"Dataset saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
