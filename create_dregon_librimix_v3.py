#!/usr/bin/env python3
"""
DREGON-LM-V3: Simpler version of DREGON-LM-V2.

- 1-second segments (like old DREGON-LM)
- All 8 mic channels as independent noise sources
- LibriSpeech mixing (same as old: mono speech + single-channel noise)
- Validation: free-flight_nosource_room1 (all channels)
- Training: all other in_flight_noise recordings (all channels)
- No motor combos, no whitenoise augmentation

Reuses create_dregon_librimix.py's load_dregon_noise_records for cached loading.
"""

import argparse
import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from create_dregon_librimix import (
    MOTOR_SAMPLE_RATE,
    SAMPLE_RATE,
    load_audio,
    load_dregon_noise_records,
)

# Override duration for V3
SAMPLE_DURATION = 1.0
TARGET_LENGTH = int(SAMPLE_DURATION * SAMPLE_RATE)


def adjust_length(audio, target_length):
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio[:target_length]


def normalize_audio(audio):
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def mix_at_snr(speech, noise, snr_db):
    sp_rms = np.sqrt(np.mean(speech**2)) + 1e-10
    ns_rms = np.sqrt(np.mean(noise**2)) + 1e-10
    target_ns_rms = sp_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_ns_rms / ns_rms)
    mixture = speech + noise_scaled
    return mixture, speech, noise_scaled


def calculate_snr(signal, noise):
    sp = np.mean(signal**2) + 1e-10
    ns = np.mean(noise**2) + 1e-10
    return 10 * np.log10(sp / ns)


def resample_rps(rps_motor, n_target, motor_rate=MOTOR_SAMPLE_RATE):
    """Resample RPS from motor rate to target frame count using linear interp.
    rps_motor: (num_motors, num_motor_samples)"""
    n_src = rps_motor.shape[1]
    src_idx = np.linspace(0, n_src - 1, n_src)
    tgt_idx = np.linspace(0, n_src - 1, n_target)
    return np.array(
        [
            np.interp(tgt_idx, src_idx, rps_motor[i]).astype(np.float32)
            for i in range(rps_motor.shape[0])
        ]
    )


def get_random_chunk(tf, duration_sec=SAMPLE_DURATION):
    """Extract random audio chunk + motor RPS from a TimeFrame.

    Args:
        tf: TimeFrame with "audio" and "motors_command" (or "motors_measured") tracks,
            and a ``_cleaned_command`` attribute (pre-computed numpy array).
    Returns:
        (audio_chunk, rps_chunk) where audio is (target_samples,) and rps is (4, motor_target).
    """
    audio = tf["audio"].samples.squeeze()  # (n_samples,)
    command = tf._cleaned_command  # (num_motors, num_motor_samples) — time-last

    target_samples = int(duration_sec * SAMPLE_RATE)
    motor_target = int(duration_sec * MOTOR_SAMPLE_RATE)

    if len(audio) <= target_samples:
        chunk = audio.copy()
        rps_chunk = command.copy()  # (num_motors, num_motor_samples)
    else:
        start_sample = random.randint(0, len(audio) - target_samples)
        chunk = audio[start_sample : start_sample + target_samples].copy()
        motor_start = int(start_sample / SAMPLE_RATE * MOTOR_SAMPLE_RATE)
        motor_end = motor_start + motor_target
        motor_end = min(motor_end, command.shape[-1])
        motor_start = max(0, motor_end - motor_target)
        rps_chunk = command[:, motor_start:motor_end].copy()  # (num_motors, motor_target)
    return chunk, rps_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_dir", required=True)
    parser.add_argument("--dregon_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train", type=int, default=6000)
    parser.add_argument("--num_valid", type=int, default=600)
    parser.add_argument("--snr_min", type=float, default=-30)
    parser.add_argument("--snr_max", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dregon_dir = Path(args.dregon_dir) / "DREGON"
    cache_dir = dregon_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Find speech files
    speech_dir = Path(args.speech_dir)
    speech_files = []
    for ext in ["*.wav", "*.flac"]:
        speech_files.extend(glob(str(speech_dir / "**" / ext), recursive=True))
    print(f"Found {len(speech_files)} speech files (on-demand loading)")

    # ── Validation ──
    VALID_ID = "free-flight_nosource_room1"
    print(f"\nLoading validation: {VALID_ID}")
    valid_records = load_dregon_noise_records(
        dregon_dir, recording_ids=[VALID_ID], cache_dir=cache_dir
    )
    print(f"  {len(valid_records)} channel-records")

    # ── Training ──
    TRAIN_IDS = [
        "free-flight_nosource_room2",
        "hovering_nosource_room2",
        "updown_nosource_room2",
        "rectangle_nosource_room2",
        "spinning_nosource_room2",
    ]
    print(f"\nLoading training: {TRAIN_IDS}")
    train_records = load_dregon_noise_records(
        dregon_dir, recording_ids=TRAIN_IDS, cache_dir=cache_dir
    )
    print(f"  {len(train_records)} channel-records")

    # Pre-clean command spikes for all records (avoid repeated work)
    from data_processing.dregon import clean_command_spikes

    for tf in tqdm(valid_records + train_records, desc="Cleaning commands"):
        motor_key = "motors_command" if "motors_command" in tf else "motors_measured"
        motor_es = tf[motor_key]
        if motor_es.values is not None:
            tf._cleaned_command = clean_command_spikes(motor_es.values.copy())  # (4, M)
        else:
            tf._cleaned_command = np.zeros((4, 0), dtype=np.float32)

    for split, records, num_samples in [
        ("train", train_records, args.num_train),
        ("valid", valid_records, args.num_valid),
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_list = []

        total_dur = sum(tf["audio"].duration for tf in records)
        print(f"\nGenerating {split} ({num_samples}, from {total_dur:.0f}s audio)")

        for idx in tqdm(range(num_samples)):
            sample_id = f"sample_{idx:05d}"
            sample_dir = split_dir / sample_id
            sample_dir.mkdir(exist_ok=True)

            tf = random.choice(records)
            noise_chunk, rps_motor = get_random_chunk(tf)

            n_frames = TARGET_LENGTH // 512 + 1
            rps = resample_rps(rps_motor, n_frames)

            speech = load_audio(random.choice(speech_files), target_sr=SAMPLE_RATE, mono=True)
            speech = adjust_length(speech, TARGET_LENGTH)
            speech = normalize_audio(speech)

            noise_chunk = adjust_length(noise_chunk, TARGET_LENGTH)
            noise_chunk = normalize_audio(noise_chunk)

            snr = np.random.uniform(args.snr_min, args.snr_max)
            mixture, speech_scaled, noise_scaled = mix_at_snr(speech, noise_chunk, snr)
            actual_snr = calculate_snr(speech_scaled, noise_scaled)

            sf.write(sample_dir / "vocals.wav", speech_scaled, SAMPLE_RATE)
            sf.write(sample_dir / "noise.wav", noise_scaled, SAMPLE_RATE)
            sf.write(sample_dir / "mixture.wav", mixture, SAMPLE_RATE)
            np.save(sample_dir / "rps.npy", rps)

            metadata_list.append(
                {
                    "id": sample_id,
                    "input_snr": float(actual_snr),
                    "target_snr": float(snr),
                    "speech_source": Path(random.choice(speech_files)).name,
                    "noise_source": tf.tags["recording_id"],
                }
            )

        meta_path = output_dir / "metadata.json"
        all_meta = json.load(open(meta_path)) if meta_path.exists() else {}
        all_meta[split] = metadata_list
        json.dump(all_meta, open(meta_path, "w"), indent=2)

        print(f"Created {len(metadata_list)} {split} samples")

    print(f"\nDone! Dataset at {output_dir}")


if __name__ == "__main__":
    main()
