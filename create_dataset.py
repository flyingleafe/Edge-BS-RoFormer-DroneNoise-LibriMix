# coding: utf-8
"""
DroneNoise-LibriMix (DN-LM) Dataset Synthesis Script

Based on the paper: "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement"

This script creates the DN-LM dataset by mixing:
- Speech samples from LibriSpeech
- Noise samples from DroneAudioDataset

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


def normalize_audio(audio):
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


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
        noise_dir: Directory containing DroneAudioDataset files
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

    # Find all audio files
    speech_files = []
    for ext in ['*.wav', '*.flac']:
        speech_files.extend(glob(os.path.join(speech_dir, '**', ext), recursive=True))

    noise_files = []
    for ext in ['*.wav', '*.WAV', '*.flac', '*.FLAC', '*.mp3', '*.MP3', '*.ogg', '*.OGG']:
        noise_files.extend(glob(os.path.join(noise_dir, '**', ext), recursive=True))

    if len(speech_files) == 0:
        raise ValueError(f"No speech files found in {speech_dir}")
    if len(noise_files) == 0:
        raise ValueError(f"No noise files found in {noise_dir}")

    print(f"Found {len(speech_files)} speech files and {len(noise_files)} noise files")

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
        noise_path = random.choice(noise_files)

        try:
            speech = load_audio(speech_path, target_sr=sample_rate)
            noise = load_audio(noise_path, target_sr=sample_rate)
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

        metadata.append({
            'id': sample_id,
            'input_snr': float(actual_snr),
            'speech_source': os.path.basename(speech_path),
            'noise_source': os.path.basename(noise_path),
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
                        help='Directory containing DroneAudioDataset files')
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
