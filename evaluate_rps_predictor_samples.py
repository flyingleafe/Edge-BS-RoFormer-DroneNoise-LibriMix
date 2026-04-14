#!/usr/bin/env python3
"""
Evaluate RPS prediction models on random validation samples and generate plots.

Usage:
    python evaluate_rps_predictor_samples.py --output_dir results/rps_eval_samples

This script:
1. Loads 3 models (simple_conv, dcunet_enc_rps, dccrn_enc_rps)
2. Selects ~5 random samples from validation dataset
3. Runs inference to get predicted RPS
4. Generates plots: audio spectrogram on top, RPS time series (4 lines) on bottom
5. Saves audio files and RPS predictions

Output structure:
    results/rps_eval_samples/
    ├── sample_XXXXX/
    │   ├── audio.wav
    │   ├── ground_truth_rps.npy
    │   ├── simple_conv_rps.npy
    │   ├── dcunet_rps.npy
    │   ├── dccrn_rps.npy
    │   └── plot.pdf
    └── ...
"""

import argparse
import glob
import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train_rps_predictor import (
    DREGONRPSDataset,
    SimpleConv,
    DCUNetEncRPS,
    DCCRNEncRPS,
    MODEL_REGISTRY,
)


def load_models(device, n_fft=2048, hop_length=512, num_rotors=4):
    """Load all three RPS prediction models."""
    models = {}
    
    # Simple Conv
    simple_conv_path = "/root/harmonic-noise-suppression/results/rps_predictor/best.pt"
    if os.path.exists(simple_conv_path):
        models["simple_conv"] = SimpleConv(n_fft, hop_length, num_rotors).to(device)
        models["simple_conv"].load_state_dict(torch.load(simple_conv_path, map_location=device, weights_only=True))
        models["simple_conv"].eval()
        print(f"Loaded simple_conv from {simple_conv_path}")
    
    # DCUNet Enc RPS
    dcunet_path = "/root/harmonic-noise-suppression/results/rps_predictor_dcunet/best_dcunet_enc_rps.pt"
    if os.path.exists(dcunet_path):
        models["dcunet"] = DCUNetEncRPS(n_fft, hop_length, num_rotors).to(device)
        models["dcunet"].load_state_dict(torch.load(dcunet_path, map_location=device, weights_only=True))
        models["dcunet"].eval()
        print(f"Loaded dcunet from {dcunet_path}")
    
    # DCCRN Enc RPS
    dccrn_path = "/root/harmonic-noise-suppression/results/rps_predictor_dccrn/best_dccrn_enc_rps.pt"
    if os.path.exists(dccrn_path):
        models["dccrn"] = DCCRNEncRPS(n_fft, hop_length, num_rotors, lite=False).to(device)
        models["dccrn"].load_state_dict(torch.load(dccrn_path, map_location=device, weights_only=True))
        models["dccrn"].eval()
        print(f"Loaded dccrn from {dccrn_path}")
    
    return models


def predict_rps(models, audio, device):
    """Run all models on audio and return predictions."""
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # Add batch dim
    
    audio = audio.to(device)
    predictions = {}
    
    with torch.no_grad():
        for name, model in models.items():
            pred = model(audio)
            predictions[name] = pred.squeeze(0).cpu().numpy()  # (4, T)
    
    return predictions


def plot_sample(audio, ground_truth_rps, predictions, sample_id, output_path):
    """
    Create a plot with:
    - Top: Audio spectrogram
    - Bottom: RPS time series (4 lines, one per rotor)
    
    Landscape A4 format: 11.69 x 8.27 inches
    """
    fig = plt.figure(figsize=(11.69, 8.27))  # Landscape A4
    
    # Use GridSpec for flexible layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
    
    # Top: Spectrogram
    ax1 = fig.add_subplot(gs[0])
    audio_np = audio.squeeze().numpy()
    
    # Compute spectrogram
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft)
    X = torch.stft(audio_np, n_fft=n_fft, hop_length=hop_length, 
                   window=window, return_complex=True, normalized=True)
    Sxx = torch.abs(X).numpy()
    
    # Plot spectrogram (log scale)
    im = ax1.imshow(20 * np.log10(Sxx + 1e-8), aspect='auto', origin='lower',
                    cmap='magma', extent=[0, Sxx.shape[1], 0, Sxx.shape[0]])
    ax1.set_ylabel('Frequency (bins)', fontsize=10)
    ax1.set_title(f'Sample {sample_id} — Audio Spectrogram (top) and RPS Time Series (bottom)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time frames', fontsize=10)
    plt.colorbar(im, ax=ax1, label='Magnitude (dB)', shrink=0.8)
    
    # Bottom: RPS time series
    ax2 = fig.add_subplot(gs[1])
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # Distinct colors for 4 rotors
    rotor_names = ['Rotor 1', 'Rotor 2', 'Rotor 3', 'Rotor 4']
    
    T = ground_truth_rps.shape[0]
    time_axis = np.arange(T)
    
    # Plot ground truth (thicker, darker)
    for i in range(4):
        ax2.plot(time_axis, ground_truth_rps[:, i], color=colors[i], 
                 linewidth=2.5, alpha=0.9, label=f'{rotor_names[i]} (GT)')
    
    # Plot predictions (dashed lines)
    linestyles = {'simple_conv': '--', 'dcunet': '-.', 'dccrn': ':'}
    
    for model_name, pred in predictions.items():
        if pred is not None:
            for i in range(4):
                ax2.plot(time_axis, pred[:, i], color=colors[i],
                        linewidth=1.5, alpha=0.7, linestyle=linestyles.get(model_name, '-'),
                        label=f'{rotor_names[i]} ({model_name})' if i == 0 else None)
    
    ax2.set_ylabel('RPS (revolutions/s)', fontsize=10)
    ax2.set_xlabel('Time frames', fontsize=10)
    ax2.legend(loc='upper right', ncol=4, fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, T])
    
    # Add model comparison text
    mse_text = "Model MSE vs Ground Truth:\n"
    for name, pred in predictions.items():
        if pred is not None:
            mse = np.mean((pred - ground_truth_rps) ** 2)
            mse_text += f"  {name}: {mse:.2f}\n"
    
    ax2.text(0.02, 0.98, mse_text, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save as PDF (vector format, good for publications)
    pdf_path = os.path.join(output_path, 'plot.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    
    # Also save as PNG for quick viewing
    png_path = os.path.join(output_path, 'plot.png')
    plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate RPS predictors on sample plots")
    parser.add_argument("--data_root", default="/root/harmonic-noise-suppression/datasets/DREGON-LM",
                       help="Path to DREGON-LM dataset")
    parser.add_argument("--output_dir", default="/root/harmonic-noise-suppression/results/rps_eval_samples",
                       help="Output directory for plots and predictions")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of random samples to evaluate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    models = load_models(device)
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Load validation dataset
    valid_dir = os.path.join(args.data_root, "valid")
    dataset = DREGONRPSDataset(valid_dir)
    print(f"\nValidation dataset: {len(dataset)} samples")
    
    # Select random samples
    num_samples = min(args.num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    print(f"Selected {num_samples} random samples: {sample_indices}")
    
    # Store results
    results = {
        "samples": [],
        "models": list(models.keys())
    }
    
    # Process each sample
    for idx in sample_indices:
        audio, rps_gt = dataset[idx]
        sample_id = os.path.basename(dataset.samples[idx])
        
        print(f"\nProcessing {sample_id}...")
        
        # Create output directory for this sample
        sample_dir = os.path.join(args.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save audio
        audio_path = os.path.join(sample_dir, 'audio.wav')
        torchaudio.save(audio_path, audio.unsqueeze(0), 16000)
        
        # Save ground truth RPS
        gt_rps_path = os.path.join(sample_dir, 'ground_truth_rps.npy')
        np.save(gt_rps_path, rps_gt.numpy())
        
        # Get predictions
        predictions = predict_rps(models, audio, device)
        
        # Save predictions
        for name, pred in predictions.items():
            pred_path = os.path.join(sample_dir, f'{name}_rps.npy')
            np.save(pred_path, pred)
        
        # Create plot
        plot_sample(audio, rps_gt.numpy(), predictions, sample_id, sample_dir)
        
        print(f"  Saved to {sample_dir}")
        
        # Compute metrics
        sample_result = {
            "sample_id": sample_id,
            "sample_path": sample_dir,
            "metrics": {}
        }
        
        for name, pred in predictions.items():
            mse = float(np.mean((pred - rps_gt.numpy()) ** 2))
            mae = float(np.mean(np.abs(pred - rps_gt.numpy())))
            sample_result["metrics"][name] = {"mse": mse, "mae": mae}
            print(f"  {name}: MSE={mse:.4f}, MAE={mae:.4f}")
        
        results["samples"].append(sample_result)
    
    # Save results summary
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output_dir}")
    print(f"Results summary: {results_path}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Avg MSE':>12} {'Avg MAE':>12}")
    print("-"*50)
    
    for model_name in models.keys():
        ms = []
        mas = []
        for s in results["samples"]:
            if model_name in s["metrics"]:
                ms.append(s["metrics"][model_name]["mse"])
                mas.append(s["metrics"][model_name]["mae"])
        if ms:
            print(f"{model_name:<20} {np.mean(ms):>12.4f} {np.mean(mas):>12.4f}")


if __name__ == "__main__":
    main()
