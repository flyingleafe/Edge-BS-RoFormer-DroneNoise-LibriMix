#!/usr/bin/env python3
"""
Generate RPS prediction comparison plots for slides.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_dir = Path("results/rps_eval_samples")
    
    with open(results_dir / "comparison_summary.json") as f:
        summary = json.load(f)
    
    models = list(summary["models"].keys())
    
    # Create summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE comparison
    rmse_means = [summary["models"][m]["rmse_mean"] for m in models]
    rmse_stds = [summary["models"][m]["rmse_std"] for m in models]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = axes[0].bar(models, rmse_means, yerr=rmse_stds, capsize=5, color=colors, alpha=0.8)
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_title('Root Mean Square Error\n(Lower is Better)', fontsize=12)
    axes[0].set_ylim(0, max(rmse_means) * 1.3)
    for bar, mean in zip(bars, rmse_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=11)
    
    # MAE comparison
    mae_means = [summary["models"][m]["mae_mean"] for m in models]
    mae_stds = [summary["models"][m]["mae_std"] for m in models]
    
    bars = axes[1].bar(models, mae_means, yerr=mae_stds, capsize=5, color=colors, alpha=0.8)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error\n(Lower is Better)', fontsize=12)
    axes[1].set_ylim(0, max(mae_means) * 1.3)
    for bar, mean in zip(bars, mae_means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=11)
    
    # R² comparison
    r2_means = [summary["models"][m]["r2_mean"] for m in models]
    r2_stds = [summary["models"][m]["r2_std"] for m in models]
    
    bars = axes[2].bar(models, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.8)
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].set_title('Coefficient of Determination\n(Higher is Better)', fontsize=12)
    axes[2].set_ylim(0, 1.1)
    for bar, mean in zip(bars, r2_means):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('slides/2026-04-14/assets/rps_comparison/summary_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create sample visualization with predictions vs ground truth
    with open(results_dir / "evaluation_results.json") as f:
        eval_results = json.load(f)
    
    sample = eval_results["samples"][0]  # First sample for visualization
    sample_id = sample["sample_id"]
    sample_dir = results_dir / sample_id
    
    gt_rps = np.load(sample_dir / "ground_truth_rps.npy")
    simple_conv_pred = np.load(sample_dir / "simple_conv_rps.npy")
    dcunet_pred = np.load(sample_dir / "dcunet_rps.npy")
    dccrn_pred = np.load(sample_dir / "dccrn_rps.npy")
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    rotor_labels = ['Rotor 1', 'Rotor 2', 'Rotor 3', 'Rotor 4']
    colors_gt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (ax, label, color) in enumerate(zip(axes, rotor_labels, colors_gt)):
        t = np.arange(gt_rps.shape[1])
        ax.plot(t, gt_rps[i], label='Ground Truth', color=color, linewidth=2, alpha=0.9)
        ax.plot(t, simple_conv_pred[i], label='SimpleConv', color=color, linewidth=1.5, linestyle='--', alpha=0.7)
        ax.plot(t, dcunet_pred[i], label='DCUNet', color=color, linewidth=1.5, linestyle=':', alpha=0.7)
        ax.plot(t, dccrn_pred[i], label='DCCRN', color=color, linewidth=1.5, linestyle='-.', alpha=0.7)
        ax.set_ylabel(f'{label}\n(RPS)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Frames', fontsize=12)
    fig.suptitle(f'RPS Prediction Comparison - {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('slides/2026-04-14/assets/rps_comparison/rps_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generated slides assets:")
    print("  - summary_metrics.png")
    print("  - rps_timeseries.png")
    
    # Copy sample plots
    import shutil
    for sample in eval_results["samples"]:
        sample_id = sample["sample_id"]
        src = results_dir / sample_id / "plot.png"
        dst = Path(f'slides/2026-04-14/assets/rps_comparison/{sample_id}_plot.png')
        shutil.copy(src, dst)
        print(f"  - {sample_id}_plot.png")

if __name__ == "__main__":
    main()
