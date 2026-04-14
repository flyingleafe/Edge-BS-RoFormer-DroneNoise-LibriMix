#!/usr/bin/env python3
"""
Extract specific samples from DREGON-LM validation data for RPS evaluation.
Samples: 00000, 00149, 00299, 00449, 00599
"""

import os
import numpy as np
import torch
import torchaudio
from pathlib import Path

# Configuration
DATASET_DIR = Path("datasets/DREGON-LM/valid")
OUTPUT_DIR = Path("datasets/DREGON-LM/rps_eval_specific_samples")
SAMPLE_IDS = ["00000", "00149", "00299", "00449", "00599"]

def extract_sample(source_dir, output_dir):
    """Extract a sample (copy full 8.224s duration)."""
    sample_name = source_dir.name
    
    # Load audio files
    mixture, sr = torchaudio.load(source_dir / "mixture.wav")
    vocals, _ = torchaudio.load(source_dir / "vocals.wav")
    noise, _ = torchaudio.load(source_dir / "noise.wav")
    
    # Load RPS
    rps = np.load(source_dir / "rps.npy")  # (4, n_motor_samples)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_dir / "mixture.wav", mixture, sr)
    torchaudio.save(output_dir / "vocals.wav", vocals, sr)
    torchaudio.save(output_dir / "noise.wav", noise, sr)
    np.save(output_dir / "rps.npy", rps)
    
    # Save metadata
    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Source: {source_dir}\n")
        f.write(f"Duration: {mixture.shape[1] / sr:.3f}s\n")
        f.write(f"Audio samples: {mixture.shape[1]}\n")
        f.write(f"RPS samples: {rps.shape[1]}\n")
    
    return output_dir

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for sample_id in SAMPLE_IDS:
        # Find the sample directory (format: sample_XXXXX)
        sample_dir = DATASET_DIR / f"sample_{sample_id}"
        
        if not sample_dir.exists():
            print(f"WARNING: {sample_dir} not found, skipping")
            continue
        
        output_subdir = OUTPUT_DIR / f"sample_{sample_id}"
        extract_sample(sample_dir, output_subdir)
        
        print(f"Extracted {sample_dir.name} → {output_subdir}")
    
    print(f"\nExtracted samples to {OUTPUT_DIR}")
    
    # Save list of samples
    with open(OUTPUT_DIR / "sample_list.txt", "w") as f:
        for sample_id in SAMPLE_IDS:
            f.write(f"sample_{sample_id}\n")

if __name__ == "__main__":
    main()
