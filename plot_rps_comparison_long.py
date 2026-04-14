#!/usr/bin/env python3
"""
Generate RPS comparison plots with specific format:
- Ground truth on top
- 3 variants one under another
- Prediction is full line, GT is half-translucent dotted on variant plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_sample_comparison(sample_dir, output_dir):
    """Create comparison plot for a single sample."""
    sample_id = sample_dir.name
    
    # Load data
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
    
    # Time axis (assuming ~929 Hz RPS sampling rate)
    t = np.arange(min_len) / 929.0  # seconds
    
    # Colors for rotors
    rotor_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create figure with 4 subplots (GT + 3 models)
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    
    # Plot 1: Ground Truth (all 4 rotors)
    ax = axes[0]
    for i, color in enumerate(rotor_colors):
        ax.plot(t, gt_rps[i], label=f'Rotor {i+1}', color=color, linewidth=2)
    ax.set_ylabel('RPS', fontsize=12)
    ax.set_title('Ground Truth Motor Speeds', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=4, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(gt_rps.min() - 2, gt_rps.max() + 2)
    
    # Plot 2: SimpleConv (prediction full, GT dotted)
    ax = axes[1]
    for i, color in enumerate(rotor_colors):
        # Ground truth as dotted translucent
        ax.plot(t, gt_rps[i], color=color, linewidth=1.5, linestyle=':', alpha=0.4)
        # Prediction as full line
        ax.plot(t, simple_conv_pred[i], color=color, linewidth=2, label=f'Rotor {i+1}' if i == 0 else "")
    ax.set_ylabel('RPS', fontsize=12)
    mae = np.mean(np.abs(simple_conv_pred - gt_rps))
    ax.set_title(f'SimpleConv Prediction (MAE={mae:.2f}) — Full=Prediction, Dotted=Target', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(gt_rps.min() - 2, gt_rps.max() + 2)
    
    # Plot 3: DCUNet (prediction full, GT dotted)
    ax = axes[2]
    for i, color in enumerate(rotor_colors):
        # Ground truth as dotted translucent
        ax.plot(t, gt_rps[i], color=color, linewidth=1.5, linestyle=':', alpha=0.4)
        # Prediction as full line
        ax.plot(t, dcunet_pred[i], color=color, linewidth=2, label=f'Rotor {i+1}' if i == 0 else "")
    ax.set_ylabel('RPS', fontsize=12)
    mae = np.mean(np.abs(dcunet_pred - gt_rps))
    ax.set_title(f'DCUNet Prediction (MAE={mae:.2f}) — Full=Prediction, Dotted=Target', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(gt_rps.min() - 2, gt_rps.max() + 2)
    
    # Plot 4: DCCRN (prediction full, GT dotted)
    ax = axes[3]
    for i, color in enumerate(rotor_colors):
        # Ground truth as dotted translucent
        ax.plot(t, gt_rps[i], color=color, linewidth=1.5, linestyle=':', alpha=0.4)
        # Prediction as full line
        ax.plot(t, dccrn_pred[i], color=color, linewidth=2, label=f'Rotor {i+1}' if i == 0 else "")
    ax.set_ylabel('RPS', fontsize=12)
    mae = np.mean(np.abs(dccrn_pred - gt_rps))
    ax.set_title(f'DCCRN Prediction (MAE={mae:.2f}) — Full=Prediction, Dotted=Target', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(gt_rps.min() - 2, gt_rps.max() + 2)
    
    # Add overall title
    fig.suptitle(f'{sample_id} — RPS Prediction Comparison (8 seconds)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
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
    axes[2].set_ylim(0, 1.1)
    for bar, mean in zip(bars, r2_means):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "summary_metrics.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary_metrics.png/pdf")

def main():
    results_dir = Path("results/rps_eval_long_samples")
    output_dir = Path("slides/2026-04-14/assets/rps_long_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating RPS comparison plots for long samples...")
    
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
