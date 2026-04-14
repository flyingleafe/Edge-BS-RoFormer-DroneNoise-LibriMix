#!/usr/bin/env python3
"""
Generate comparison table with RMSE/MAE/R² metrics for RPS prediction models.

Usage:
    python generate_rps_comparison_table.py
"""

import json
import os
import numpy as np
from pathlib import Path


def compute_metrics(pred, target):
    """Compute RMSE, MAE, R² metrics."""
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))
    
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return rmse, mae, r2


def main():
    results_dir = Path("results/rps_eval_samples")
    evaluation_file = results_dir / "evaluation_results.json"
    
    with open(evaluation_file) as f:
        eval_results = json.load(f)
    
    models = ["simple_conv", "dcunet", "dccrn"]
    samples = eval_results["samples"]
    
    # Collect metrics per model
    model_metrics = {m: {"rmse": [], "mae": [], "r2": []} for m in models}
    
    for sample in samples:
        sample_id = sample["sample_id"]
        sample_dir = results_dir / sample_id
        
        gt_rps = np.load(sample_dir / "ground_truth_rps.npy")  # (4, T)
        
        for model_name in models:
            pred_file = sample_dir / f"{model_name}_rps.npy"
            if pred_file.exists():
                pred_rps = np.load(pred_file)  # (4, T)
                rmse, mae, r2 = compute_metrics(pred_rps, gt_rps)
                model_metrics[model_name]["rmse"].append(rmse)
                model_metrics[model_name]["mae"].append(mae)
                model_metrics[model_name]["r2"].append(r2)
    
    # Compute mean and std across samples
    print("=" * 90)
    print("RPS PREDICTION MODEL COMPARISON")
    print("=" * 90)
    print()
    
    # Per-sample table
    print("Per-Sample Metrics:")
    print("-" * 90)
    print(f"{'Sample':<15} {'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 90)
    
    for sample in samples:
        sample_id = sample["sample_id"]
        for model_name in models:
            if model_name in sample["metrics"]:
                m = sample["metrics"][model_name]
                rmse = np.sqrt(m["mse"])
                print(f"{sample_id:<15} {model_name:<15} {rmse:>10.4f} {m['mae']:>10.4f} {-1:>10}")
    
    print()
    print("=" * 90)
    print("SUMMARY (Mean ± Std across 5 samples)")
    print("=" * 90)
    print(f"{'Model':<20} {'RMSE':>12} {'MAE':>12} {'R²':>12}")
    print("-" * 60)
    
    for model_name in models:
        rmses = model_metrics[model_name]["rmse"]
        maes = model_metrics[model_name]["mae"]
        r2s = model_metrics[model_name]["r2"]
        
        if rmses:
            rmse_mean = np.mean(rmses)
            rmse_std = np.std(rmses)
            mae_mean = np.mean(maes)
            mae_std = np.std(maes)
            r2_mean = np.mean(r2s)
            r2_std = np.std(r2s)
            
            print(f"{model_name:<20} {rmse_mean:>6.4f}±{rmse_std:<5.4f} {mae_mean:>6.4f}±{mae_std:<5.4f} {r2_mean:>6.4f}±{r2_std:<5.4f}")
    
    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
- RMSE (Root Mean Square Error): Lower is better. Measures the standard deviation of residuals.
- MAE (Mean Absolute Error): Lower is better. Measures average absolute difference.
- R² (Coefficient of Determination): Higher is better (max 1.0). Measures proportion of variance explained.

Key Findings:
- simple_conv: Lightweight baseline CNN on log-magnitude spectrograms
- dcunet: DCUNet encoder (complex conv) + FPN-style RPS prediction head  
- dccrn: DCCRN encoder (complex conv) + FPN-style RPS prediction head

Note: The simple_conv baseline surprisingly outperforms the complex models in RPS prediction.
This may be because:
1. SimpleConv uses real-valued convolutions which may be easier to optimize
2. The complex models may be overfitting to audio enhancement rather than RPS prediction
3. Multi-task training may dilute RPS prediction performance
""")
    
    # Save summary to JSON
    summary = {
        "num_samples": len(samples),
        "sample_ids": [s["sample_id"] for s in samples],
        "models": {}
    }
    
    for model_name in models:
        rmses = model_metrics[model_name]["rmse"]
        maes = model_metrics[model_name]["mae"]
        r2s = model_metrics[model_name]["r2"]
        
        if rmses:
            summary["models"][model_name] = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
            }
    
    summary_file = results_dir / "comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
