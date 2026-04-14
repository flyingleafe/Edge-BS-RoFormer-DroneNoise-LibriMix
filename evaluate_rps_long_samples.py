#!/usr/bin/env python3
"""
Evaluate RPS predictors on longer samples (~8 seconds).
Run on vast-server.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

# Import models from train_rps_predictor
sys.path.insert(0, str(Path(__file__).parent))
from train_rps_predictor import SimpleConv, DCUNetEncRPS, DCCRNEncRPS, DCCRNLiteEncRPS

# Configuration
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512

def load_model(model_path, model_type, device):
    """Load a trained RPS predictor model."""
    if model_type == "simple_conv":
        model = SimpleConv()
    elif model_type == "dcunet":
        model = DCUNetEncRPS()
    elif model_type == "dccrn":
        model = DCCRNEncRPS()
    elif model_type == "dccrn_lite":
        model = DCCRNLiteEncRPS()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def predict_rps(model, audio, device):
    """Predict RPS from audio."""
    with torch.no_grad():
        audio = audio.to(device)
        pred = model(audio)
        return pred.cpu()

def compute_metrics(pred, target):
    """Compute RMSE, MAE, R² metrics."""
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))
    
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return rmse, mae, r2

def process_sample(sample_dir, models, device):
    """Process a single sample with all models."""
    # Load audio and RPS
    mixture, sr = torchaudio.load(sample_dir / "mixture.wav")
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} Hz"
    
    rps_gt = np.load(sample_dir / "rps.npy")  # (4, T)
    
    # Predict with each model
    predictions = {}
    metrics = {}
    
    for name, model in models.items():
        pred = predict_rps(model, mixture, device)  # (1, 4, T)
        pred_np = pred.squeeze(0).numpy()  # (4, T)
        
        # Match lengths (RPS might be slightly different length)
        min_len = min(pred_np.shape[1], rps_gt.shape[1])
        pred_np = pred_np[:, :min_len]
        rps_gt_matched = rps_gt[:, :min_len]
        
        predictions[name] = pred_np
        
        # Compute metrics
        rmse, mae, r2 = compute_metrics(pred_np, rps_gt_matched)
        metrics[name] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    
    return {
        "audio": mixture,
        "rps_gt": rps_gt,
        "predictions": predictions,
        "metrics": metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/DREGON-LM/rps_eval_long_samples")
    parser.add_argument("--output_dir", type=str, default="results/rps_eval_long_samples")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    model_paths = {
        "simple_conv": "results/rps_predictor/best.pt",
        "dcunet": "results/rps_predictor_dcunet/best_dcunet_enc_rps.pt",
        "dccrn": "results/rps_predictor_dccrn/best_dccrn_enc_rps.pt"
    }
    
    print("\nLoading models...")
    models = {}
    for name, path in model_paths.items():
        if Path(path).exists():
            models[name] = load_model(path, name, device)
            print(f"  Loaded {name} from {path}")
        else:
            print(f"  WARNING: {path} not found, skipping {name}")
    
    if not models:
        print("No models loaded! Exiting.")
        return
    
    # Get samples
    data_dir = Path(args.data_dir)
    samples = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    print(f"\nFound {len(samples)} samples in {data_dir}")
    
    # Process each sample
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for sample_dir in tqdm(samples, desc="Processing samples"):
        sample_id = sample_dir.name
        print(f"\nProcessing {sample_id}...")
        
        result = process_sample(sample_dir, models, device)
        
        # Save results
        sample_out_dir = output_dir / sample_id
        sample_out_dir.mkdir(exist_ok=True)
        
        # Save audio
        torchaudio.save(sample_out_dir / "mixture.wav", result["audio"], SAMPLE_RATE)
        
        # Save RPS
        np.save(sample_out_dir / "ground_truth_rps.npy", result["rps_gt"])
        for model_name, pred in result["predictions"].items():
            np.save(sample_out_dir / f"{model_name}_rps.npy", pred)
        
        # Save metrics
        all_results.append({
            "sample_id": sample_id,
            "metrics": result["metrics"]
        })
        
        print(f"  Saved to {sample_out_dir}")
        for model_name, m in result["metrics"].items():
            print(f"    {model_name}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
    
    # Save summary
    summary = {
        "num_samples": len(samples),
        "sample_ids": [r["sample_id"] for r in all_results],
        "results": all_results
    }
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Compute aggregate metrics
    for model_name in models.keys():
        rmses = [r["metrics"][model_name]["rmse"] for r in all_results]
        maes = [r["metrics"][model_name]["mae"] for r in all_results]
        r2s = [r["metrics"][model_name]["r2"] for r in all_results]
        
        print(f"{model_name}:")
        print(f"  RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"  MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        print(f"  R²:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
