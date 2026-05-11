#!/usr/bin/env python3
"""
Analyze RPS predictor performance on high-SNR samples from DREGON speech-high recordings.

This script extracts 8-second chunks from DREGON recordings where speech was played 
alongside drone flight (speech-high), which naturally have higher SNR than the 
synthetically mixed DREGON-LM samples.

Comparison targets:
- Current eval samples: mixed SNR (-30 to 0 dB range, mean ~-15 dB)
- High-SNR samples: real recordings with speech + drone, speech is dominant
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_rps_predictor import SimpleConv, DCUNetEncRPS, DCCRNEncRPS

# Constants
DREGON_DIR = Path("data/DREGON")
AUDIO_SR = 44100
TARGET_SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_DURATION = 8.224  # Match DREGON-LM samples
SAMPLE_LENGTH = int(SAMPLE_DURATION * TARGET_SR)  # 131584 samples

# High-SNR recordings from DREGON (speech source + drone noise)
SPEECH_HIGH_RECORDINGS = [
    "DREGON_free-flight_speech-high_room1",
]


def load_speech_high_chunk(recording_id: str, start_time: float, duration: float = SAMPLE_DURATION) -> tuple:
    """
    Load a chunk from speech-high DREGON recording with motor data.
    Returns (audio_16k, rps, motor_sr)
    """
    import scipy.io
    
    rec_dir = DREGON_DIR / recording_id
    
    # Load audio
    audio_path = rec_dir / f"{recording_id}.wav"
    audio, sr = torchaudio.load(str(audio_path))
    
    # Convert start time to samples
    start_sample = int(start_time * sr)
    end_sample = start_sample + int(duration * sr)
    
    # Resample audio to 16kHz
    audio_chunk = audio[:, start_sample:end_sample]
    
    # Resample to 16kHz
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio_16k = resampler(audio_chunk.mean(dim=0, keepdim=True))  # Mix to mono
    else:
        audio_16k = audio_chunk.mean(dim=0, keepdim=True)
    
    # Load motor data
    motor_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_motors.mat")
    motor_data = motor_mat['motor'][0, 0]
    measured = motor_data['measured']  # (n_motor, 4)
    motor_ts = motor_data['timestamps'].flatten()
    motor_sr = len(motor_ts) / (motor_ts[-1] - motor_ts[0])
    
    # Extract motor measurements for the chunk
    # Motor timestamps are Unix timestamps
    motor_start_idx = np.searchsorted(motor_ts, start_time)
    motor_end_idx = np.searchsorted(motor_ts, start_time + duration)
    rps_chunk = measured[motor_start_idx:motor_end_idx].T  # (4, n_motor_frames)
    
    return audio_16k, rps_chunk, motor_sr


def resample_rps_to_stft(rps: np.ndarray, motor_sr: float, n_frames: int) -> np.ndarray:
    """Resample motor RPS to STFT frame rate."""
    n_motor = rps.shape[1]
    
    # Motor timestamps to STFT frame timestamps
    # STFT hop = HOP_LENGTH / TARGET_SR seconds per frame
    stft_times = np.arange(n_frames) * (HOP_LENGTH / TARGET_SR)
    
    # Motor times relative to start
    t_start = motor_ts[0] if 'motor_ts' in dir() else 0  # Will be set in main
    motor_times = np.arange(n_motor) / motor_sr
    
    # Linear interpolation
    rps_stft = np.zeros((4, n_frames))
    for rotor in range(4):
        rps_stft[rotor] = np.interp(stft_times, motor_times, rps[rotor])
    
    return rps_stft


def evaluate_model(model, audio: torch.Tensor, rps_gt: torch.Tensor, device: str) -> dict:
    """Evaluate a model on a sample."""
    model.eval()
    audio = audio.to(device)
    rps_gt = rps_gt.to(device)
    
    with torch.no_grad():
        rps_pred = model(audio)  # (B, 4, T_pred)
        
        # Make sure rps_gt has correct shape (B, 4, T)
        if rps_gt.dim() == 4:
            rps_gt = rps_gt.squeeze(1)  # (B, 4, T)
        
        # Interpolate GT to match prediction length
        rps_gt_interp = F.interpolate(
            rps_gt, size=rps_pred.shape[-1], 
            mode="linear", align_corners=False
        )
        
        # Compute metrics
        mse = F.mse_loss(rps_pred, rps_gt_interp).item()
        mae = torch.abs(rps_pred - rps_gt_interp).mean().item()
        
        # R² score
        pred = rps_pred.cpu().numpy().flatten()
        target = rps_gt_interp.cpu().numpy().flatten()
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2
        }


def load_model(checkpoint_path: str, model_type: str, device: str):
    """Load a trained model."""
    if model_type == "simple_conv":
        model = SimpleConv(n_fft=N_FFT, hop_length=HOP_LENGTH)
    elif model_type == "dcunet":
        model = DCUNetEncRPS(n_fft=N_FFT, hop_length=HOP_LENGTH)
    elif model_type == "dccrn":
        model = DCCRNEncRPS(n_fft=N_FFT, hop_length=HOP_LENGTH)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    return model.to(device)


def extract_samples_from_speech_high(recording_id: str, num_samples: int = 10) -> list:
    """Extract high-SNR samples from speech-high recording."""
    import scipy.io
    
    rec_dir = DREGON_DIR / recording_id
    
    # Load audio and audio timestamps
    audio_path = rec_dir / f"{recording_id}.wav"
    audio_full, sr = torchaudio.load(str(audio_path))
    audio_ts_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_audiots.mat")
    audio_ts = audio_ts_mat['audio_timestamps'].flatten()
    
    # Load motor data
    motor_mat = scipy.io.loadmat(rec_dir / f"{recording_id}_motors.mat")
    motor_data = motor_mat['motor'][0, 0]
    measured = motor_data['measured']
    motor_ts = motor_data['timestamps'].flatten()
    motor_sr = len(motor_ts) / (motor_ts[-1] - motor_ts[0])
    
    recording_start = motor_ts[0]  # Motor timestamps are the reference
    recording_duration = motor_ts[-1] - motor_ts[0]
    print(f"Recording: {recording_duration:.1f}s, SR={sr}, Motor SR={motor_sr:.1f}")
    
    # Resampler for 44.1kHz -> 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
    
    # Extract evenly spaced samples
    samples = []
    start_offset = 5.0  # Skip first few seconds
    usable_duration = recording_duration - start_offset - SAMPLE_DURATION
    step = usable_duration / max(num_samples - 1, 1)
    
    for i in range(num_samples):
        rel_time = start_offset + i * step  # Relative to recording start
        
        # Convert to Unix time for audio timestamp lookup
        unix_time = recording_start + rel_time
        
        # Find audio sample index
        audio_start_idx = int((unix_time - audio_ts[0]) * sr)
        
        # Ensure we don't exceed audio length
        end_sample = min(audio_start_idx + int(SAMPLE_DURATION * sr), audio_full.shape[1])
        audio_start_idx = min(audio_start_idx, audio_full.shape[1] - int(SAMPLE_DURATION * sr))
        
        # Extract chunk
        audio_chunk = audio_full[:, audio_start_idx:end_sample]
        
        # Mix to mono and resample to 16kHz
        audio_mono = audio_chunk.mean(dim=0)  # (samples,)
        audio_16k = resampler(audio_mono.unsqueeze(0))  # (1, 16k_samples)
        
        # Normalize audio to similar level as DREGON-LM training samples
        # DREGON-LM samples are normalized to [-0.95, 0.95] range
        audio_max = audio_16k.abs().max()
        if audio_max > 0:
            audio_16k = audio_16k / audio_max * 0.9  # Normalize to ~0.9 peak
        
        # Get motor data for chunk
        motor_rel_start = rel_time
        motor_start_idx = np.searchsorted(motor_ts, recording_start + motor_rel_start) - 1
        motor_end_idx = np.searchsorted(motor_ts, recording_start + motor_rel_start + SAMPLE_DURATION) + 1
        rps_chunk = measured[motor_start_idx:motor_end_idx].T  # (4, n_motor_frames)
        
        samples.append({
            "audio": audio_16k,
            "rps": torch.from_numpy(rps_chunk.astype(np.float32)),
            "motor_sr": motor_sr,
            "rel_time": rel_time,
        })
        print(f"  Sample {i}: t={rel_time:.1f}s, audio={audio_16k.shape}, rps={rps_chunk.shape}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Analyze RPS predictor on high-SNR samples")
    parser.add_argument("--recording", default="DREGON_free-flight_speech-high_room1", 
                       help="Recording to use")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to extract")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", default="results/rps_high_snr_analysis.json")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model checkpoints
    models = {
        "simple_conv": "results/rps_predictor/best.pt",
        "dcunet": "results/rps_predictor_dcunet/best_dcunet_enc_rps.pt",
        "dccrn": "results/rps_predictor_dccrn/best_dccrn_enc_rps.pt",
    }
    
    print(f"\n=== Extracting {args.num_samples} samples from {args.recording} ===")
    samples = extract_samples_from_speech_high(args.recording, args.num_samples)
    print(f"Extracted {len(samples)} samples")
    
    # Load models
    loaded_models = {}
    for name, path in models.items():
        if Path(path).exists():
            print(f"Loading {name} from {path}...")
            loaded_models[name] = load_model(path, name, device)
        else:
            print(f"WARNING: {name} not found at {path}")
    
    # Evaluate each model on each sample
    results = {
        "recording": args.recording,
        "num_samples": len(samples),
        "sample_duration": SAMPLE_DURATION,
        "results": [],
    }
    
    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        audio = sample["audio"].unsqueeze(0)  # (1, 1, samples)
        rps_gt = sample["rps"]
        motor_sr = sample["motor_sr"]
        
        # Compute ground truth RPS at STFT frame rate
        n_frames = audio.shape[-1] // HOP_LENGTH + 1
        rps_stft = np.zeros((4, n_frames))
        for rotor in range(4):
            motor_times = np.arange(rps_gt.shape[1]) / motor_sr
            stft_times = np.arange(n_frames) * (HOP_LENGTH / TARGET_SR)
            rps_stft[rotor] = np.interp(stft_times, motor_times, rps_gt[rotor].numpy())
        rps_gt_stft = torch.from_numpy(rps_stft).float()
        
        # Evaluate each model
        sample_results = {"sample_id": i, "rel_time": sample["rel_time"]}
        for model_name, model in loaded_models.items():
            metrics = evaluate_model(model, audio, rps_gt_stft.unsqueeze(0), device)
            sample_results[model_name] = metrics
        
        results["results"].append(sample_results)
    
    # Aggregate metrics
    for model_name in loaded_models.keys():
        msns = [r[model_name]["mse"] for r in results["results"]]
        maes = [r[model_name]["mae"] for r in results["results"]]
        r2s = [r[model_name]["r2"] for r in results["results"]]
        
        results[f"{model_name}_avg"] = {
            "mse": np.mean(msns),
            "rmse": np.sqrt(np.mean(msns)),
            "mae": np.mean(maes),
            "r2": np.mean(r2s),
        }
    
    # Save results - convert numpy types to native Python
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(args.output, "w") as f:
        json.dump(convert_to_native(results), f, indent=2)
    
    print(f"\n=== Results saved to {args.output} ===")
    print("\nSummary:")
    for model_name in loaded_models.keys():
        avg = results[f"{model_name}_avg"]
        print(f"  {model_name}: MSE={avg['mse']:.2f}, MAE={avg['mae']:.2f}, R²={avg['r2']:.4f}")
    
    # Compare with low-SNR results
    print("\n=== Comparison with Low-SNR Results ===")
    low_snr_results = {
        "simple_conv": {"mse": 2.0, "mae": 1.3, "r2": 0.7},  # Approximate from existing eval
        "dcunet": {"mse": 1.5, "mae": 1.2, "r2": 0.8},
        "dccrn": {"mse": 1.4, "mae": 1.1, "r2": 0.8},
    }
    
    print("\nModel     | High-SNR MSE | Low-SNR MSE | Difference")
    print("-" * 55)
    for model_name in loaded_models.keys():
        if model_name in low_snr_results:
            high = results[f"{model_name}_avg"]["mse"]
            low = low_snr_results[model_name]["mse"]
            diff = ((high - low) / low) * 100
            print(f"{model_name:<10} | {high:>12.2f} | {low:>11.2f} | {diff:>+7.1f}%")


if __name__ == "__main__":
    main()