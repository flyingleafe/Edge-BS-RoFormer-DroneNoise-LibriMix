#!/usr/bin/env python3
"""
Generate RPS comparison plots with spectrogram at top.
Format:
- Row 1: Spectrogram (same time axis as GT subplot)
- Row 2: Ground Truth (all 4 rotors)
- Row 3-5: SimpleConv, DCUNet, DCCRN predictions with dotted GT overlay
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchaudio

def plot_sample_comparison(sample_dir, output_dir):
    """Create comparison plot with spectrogram for a single sample."""
    sample_id = sample_dir.name
    
    # Load audio and RPS
    mixture, sr = torchaudio.load(sample_dir / "mixture.wav")
    mixture_np = mixture.squeeze().numpy()
    
    gt_rps = np.load(sample_dir / "ground_truth_rps.npy")  # (4, T)
    simple_conv_pred = np.load(sample_dir / "simple_conv_rps.npy")
    dcunet_pred = np.load(sample_dir / "dcunet_rps.npy")
    dccrn_pred = np.load(sample_dir / "dccrn_rps.npy")
    
    # Match lengths
    min_len = min(gt_rps.shape[1], simple_conv_pred.shape[1], dcunet_pred.shape[1], dccrn_pred.shape[1])
    gt_rps = gt_rps[:, :min_len]
    simple_conv_pred = simple_conv_pred[:, :min_len]
    dcunet_pred = dcunet_pred[:, :min_len]
    dccrn_pred = dccrn_pred[:, :min_len]
    
    # Time axis - 8.224 seconds for full DREGON-LM samples
    duration = mixture.shape[1] / sr  # Actual duration
    t = np.linspace(0, duration, min_len)
    
    # Colors for rotors (matching reference)
    rotor_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create figure with 5 subplots (spectrogram + GT + 3 models)
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.2, 1, 1, 1, 1], hspace=0.3)
    
    # Row 1: Spectrogram
    ax_spec = fig.add_subplot(gs[0])
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft)
    X = torch.stft(torch.from_numpy(mixture_np).float(), n_fft=n_fft, hop_length=hop_length,
                   window=window, return_complex=True, normalized=True)
    Sxx = torch.abs(X).numpy()
    
    # Create time axis for spectrogram
    spec_times = np.linspace(0, duration, Sxx.shape[1])
    freqs = np.linspace(0, sr/2, Sxx.shape[0])
    
    im = ax_spec.pcolormesh(spec_times, freqs, 20 * np.log10(Sxx + 1e-8), 
                            shading='auto', cmap='magma')
    ax_spec.set_ylabel('Frequency (Hz)', fontsize=11)
    ax_spec.set_title(f'{sample_id} — Input Spectrogram', fontsize=13, fontweight='bold')
    ax_spec.set_ylim(0, 4000)  # Limit to 4kHz for clarity
    ax_spec.set_xlim(0, duration)
    plt.colorbar(im, ax=ax_spec, label='Magnitude (dB)', pad=0.01)
    
    # Row 2: Ground Truth (all 4 rotors)
    ax_gt = fig.add_subplot(gs[1], sharex=ax_spec)
    for i, color in enumerate(rotor_colors):
        ax_gt.plot(t, gt_rps[i], label=f'Rotor {i+1}', color=color, linewidth=2)
    ax_gt.set_ylabel('RPS', fontsize=11)
    ax_gt.set_title('Ground Truth Motor Speeds', fontsize=13, fontweight='bold')
    ax_gt.legend(loc='upper right', ncol=4, fontsize=9)
    ax_gt.grid(True, alpha=0.3)
    ax_gt.set_xlim(0, duration)
    
    # Row 3: SimpleConv (prediction full, GT dotted)
    ax_simple = fig.add_subplot(gs[2], sharex=ax_spec)
    for i, color in enumerate(rotor_colors):
        # Ground truth as dotted translucent
        ax_simple.plot(t, gt_rps[i], color=color, linewidth=1.5, linestyle=':', alpha=0.4)
        # Prediction as full line
        ax_simple.plot(t, simple_conv_pred[i], color=color, linewidth=2, label=f'Rotor {i+1}' if i == 0 else "")
    ax_simple.set_ylabel('RPS', fontsize=11)
    mae = np.mean(np.abs(simple_conv_pred - gt_rps))
    ax_simple.set_title(f'SimpleConv Prediction (MAE={mae:.2f}) — Full=Prediction, Dotted=Target', 
                 fontsize=13, fontweight='bold')
    ax_simple.grid(True, alpha=0.3)
    ax_simple.set_xlim(0, duration)
    
    # Row 4: DCUNet (prediction full, GT dotted)
    ax_dcunet = fig.add_subplot(gs[3], sharex=ax_spec)
    for i, color in enumerate(rotor_colors):
        # Ground truth as dotted translucent
        ax_dcunet.plot(t, gt_rps[i], color=color, linewidth=1.5, linestyle=':', alpha=0.4)
        # Prediction as full line
        ax_dcunet.plot(t, dcunet_pred[i], color=color, linewidth=2, label=f'Rotor {i+1}' if i == 0 else "")
    ax_dcunet.set_ylabel('RPS', fontsize=11)
    mae = np.mean(np.abs(dcunet_pred - gt_rps))
    ax_dcunet.set_title(f'DCUNet Prediction (MAE={mae:.2f}) — Full=Prediction, Dotted=Target', 
                 fontsize=13, fontweight='bold')
    ax_dcunet.grid(True, alpha=0.3)
    ax_dcunet.set_xlim(0, duration)
    
    # Row 5: DCCRN (prediction full, GT dotted)
    ax_dccrn = fig.add_subplot(gs[4], sharex=ax_spec)
    for i, color in enumerate(rotor_colors):
        # Ground truth as dotted translucent
        ax_dccrn.plot(t, gt_rps[i], color=color, linewidth=1.5, linestyle=':', alpha=0.4)
        # Prediction as full line
        ax_dccrn.plot(t, dccrn_pred[i], color=color, linewidth=2, label=f'Rotor {i+1}' if i == 0 else "")
    ax_dccrn.set_ylabel('RPS', fontsize=11)
    mae = np.mean(np.abs(dccrn_pred - gt_rps))
    ax_dccrn.set_title(f'DCCRN Prediction (MAE={mae:.2f}) — Full=Prediction, Dotted=Target', 
                 fontsize=13, fontweight='bold')
    ax_dccrn.set_xlabel('Time (s)', fontsize=11)
    ax_dccrn.grid(True, alpha=0.3)
    ax_dccrn.set_xlim(0, duration)
    
    # Hide x tick labels for upper subplots
    for ax in [ax_spec, ax_gt, ax_simple, ax_dcunet]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{sample_id}_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f"{sample_id}_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"  Saved {sample_id}_comparison.png/pdf")

def plot_summary_metrics(results_dir, output_dir):
    """Create summary bar chart of metrics."""
    with open(results_dir / "evaluation_results.json") as f:
        results = json.load(f)
    
    # Collect metrics
    models = ["simple_conv", "dcunet", "dccrn"]
    model_labels = ["SimpleConv", "DCUNet", "DCCRN"]
    
    rmse_data = {m: [] for m in models}
    mae_data = {m: [] for m in models}
    r2_data = {m: [] for m in models}
    
    for sample in results["results"]:
        for model in models:
            rmse_data[model].append(sample["metrics"][model]["rmse"])
            mae_data[model].append(sample["metrics"][model]["mae"])
            r2_data[model].append(sample["metrics"][model]["r2"])
    
    # Compute means and stds
    rmse_means = [np.mean(rmse_data[m]) for m in models]
    rmse_stds = [np.std(rmse_data[m]) for m in models]
    mae_means = [np.mean(mae_data[m]) for m in models]
    mae_stds = [np.std(mae_data[m]) for m in models]
    r2_means = [np.mean(r2_data[m]) for m in models]
    r2_stds = [np.std(r2_data[m]) for m in models]
    
    # Create bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    x = np.arange(len(models))
    width = 0.6
    
    # RMSE
    bars = axes[0].bar(x, rmse_means, width, yerr=rmse_stds, capsize=5, color=colors, alpha=0.8)
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_title('Root Mean Square Error\n(Lower is Better)', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_labels)
    axes[0].set_ylim(0, max(rmse_means) * 1.3)
    for bar, mean in zip(bars, rmse_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=11)
    
    # MAE
    bars = axes[1].bar(x, mae_means, width, yerr=mae_stds, capsize=5, color=colors, alpha=0.8)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error\n(Lower is Better)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels)
    axes[1].set_ylim(0, max(mae_means) * 1.3)
    for bar, mean in zip(bars, mae_means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=11)
    
    # R²
    bars = axes[2].bar(x, r2_means, width, yerr=r2_stds, capsize=5, color=colors, alpha=0.8)
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].set_title('Coefficient of Determination\n(Higher is Better)', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_labels)
    axes[2].set_ylim(min(0, min(r2_means) - 0.2), 1.1)
    for bar, mean in zip(bars, r2_means):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "summary_metrics.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary_metrics.png/pdf")

def main():
    results_dir = Path("results/rps_eval_specific_samples")
    output_dir = Path("slides/2026-04-14/assets/rps_specific_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating RPS comparison plots with spectrograms...")
    
    # Get samples
    samples = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    for sample_dir in samples:
        plot_sample_comparison(sample_dir, output_dir)
    
    # Generate summary
    print("\nGenerating summary metrics plot...")
    plot_summary_metrics(results_dir, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
