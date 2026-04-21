#!/usr/bin/env python3
"""
Extract longer (~8 second) samples from DREGON-LM validation data for RPS evaluation.
"""

import os
import random
import numpy as np
import torch
import torchaudio
from pathlib import Path

# Configuration
DATASET_DIR = Path("datasets/DREGON-LM/valid")
OUTPUT_DIR = Path("datasets/DREGON-LM/rps_eval_long_samples")
NUM_SAMPLES = 5
SAMPLE_DURATION = 8.0  # seconds
SEED = 42

def extract_sample(sample_dir, output_dir, start_time, duration):
    """Extract a segment from a sample."""
    sample_name = sample_dir.name
    
    # Load audio files
    mixture, sr = torchaudio.load(sample_dir / "mixture.wav")
    vocals, _ = torchaudio.load(sample_dir / "vocals.wav")
    noise, _ = torchaudio.load(sample_dir / "noise.wav")
    
    # Load RPS
    rps = np.load(sample_dir / "rps.npy")  # (4, n_motor_samples)
    
    # Calculate start and end samples for audio
    start_sample = int(start_time * sr)
    end_sample = start_sample + int(duration * sr)
    
    # Calculate corresponding RPS indices (RPS is at ~929 Hz, audio at 16 kHz)
    rps_sr = rps.shape[1] / (mixture.shape[1] / sr)  # RPS sampling rate
    start_rps = int(start_time * rps_sr)
    end_rps = start_rps + int(duration * rps_sr)
    
    # Extract segments
    mixture_seg = mixture[:, start_sample:end_sample]
    vocals_seg = vocals[:, start_sample:end_sample]
    noise_seg = noise[:, start_sample:end_sample]
    rps_seg = rps[:, start_rps:end_rps]
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_dir / "mixture.wav", mixture_seg, sr)
    torchaudio.save(output_dir / "vocals.wav", vocals_seg, sr)
    torchaudio.save(output_dir / "noise.wav", noise_seg, sr)
    np.save(output_dir / "rps.npy", rps_seg)
    
    # Save metadata
    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Source: {sample_dir}\n")
        f.write(f"Start time: {start_time}s\n")
        f.write(f"Duration: {duration}s\n")
        f.write(f"Audio samples: {mixture_seg.shape[1]}\n")
        f.write(f"RPS samples: {rps_seg.shape[1]}\n")
    
    return output_dir

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Get all validation samples
    valid_samples = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(valid_samples)} validation samples")
    
    # Select random samples
    selected = random.sample(valid_samples, NUM_SAMPLES)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, sample_dir in enumerate(selected):
        # Random start time (leave room for 8 seconds at the end)
        mixture, sr = torchaudio.load(sample_dir / "mixture.wav")
        total_duration = mixture.shape[1] / sr
        
        max_start = total_duration - SAMPLE_DURATION
        start_time = random.uniform(0, max_start)
        
        output_subdir = OUTPUT_DIR / f"sample_{i:03d}"
        extract_sample(sample_dir, output_subdir, start_time, SAMPLE_DURATION)
        
        print(f"Extracted {sample_dir.name} → {output_subdir}")
        print(f"  Start: {start_time:.2f}s, Duration: {SAMPLE_DURATION}s")
        print(f"  Source: {sample_dir}")
    
    print(f"\nExtracted {NUM_SAMPLES} samples to {OUTPUT_DIR}")
    
    # Save list of samples
    with open(OUTPUT_DIR / "sample_list.txt", "w") as f:
        for i in range(NUM_SAMPLES):
            f.write(f"sample_{i:03d}\n")

if __name__ == "__main__":
    main()
