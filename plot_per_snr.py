#!/usr/bin/env python3
"""Plot per-SNR metric comparison across models."""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNR_LEVELS = [-30, -25, -20, -15, -10, -5, 0]
METRICS = ["si_sdr", "estoi", "pesq"]
METRIC_LABELS = {"si_sdr": "SI-SDR (dB)", "estoi": "ESTOI", "pesq": "PESQ"}


def nearest_snr_level(snr):
    return min(SNR_LEVELS, key=lambda l: abs(snr - l))


def load_metadata(dataset_path="datasets/DREGON-LM"):
    meta_path = os.path.join(dataset_path, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    if "valid" in meta:
        return {e["id"]: e for e in meta["valid"]}
    return meta


def parse_eval_log(log_path):
    """Parse per-sample metrics from final_valid.py stdout log."""
    with open(log_path) as f:
        text = f.read()

    results = []
    song_pattern = re.compile(r"Song: .+/(sample_\d+)\s")
    metric_pattern = re.compile(r"Metric (\w+)\s+value:\s+([\d.eE+-]+)")

    current_sample = None
    current_metrics = {}

    for line in text.split("\n"):
        m = song_pattern.search(line)
        if m:
            if current_sample and current_metrics:
                results.append((current_sample, dict(current_metrics)))
            current_sample = m.group(1)
            current_metrics = {}
        m = metric_pattern.search(line)
        if m:
            current_metrics[m.group(1)] = float(m.group(2))

    if current_sample and current_metrics:
        results.append((current_sample, dict(current_metrics)))

    return results


def compute_per_snr(results, metadata):
    """Group results by nearest SNR level and compute means."""
    bins = defaultdict(lambda: {m: [] for m in METRICS})

    for sample_id, metrics in results:
        if sample_id not in metadata:
            continue
        snr = metadata[sample_id].get("input_snr")
        if snr is None:
            continue
        level = nearest_snr_level(snr)
        for m in METRICS:
            if m in metrics and not np.isnan(metrics[m]):
                bins[level][m].append(metrics[m])

    per_snr = {}
    for level in SNR_LEVELS:
        per_snr[level] = {}
        for m in METRICS:
            vals = bins[level][m]
            per_snr[level][m] = np.mean(vals) if vals else float("nan")
            per_snr[level][f"{m}_std"] = np.std(vals) if vals else float("nan")
            per_snr[level]["n"] = len(bins[level][METRICS[0]])

    return per_snr


def find_eval_excel(eval_dir):
    """Find the per-sample Excel file in a job results directory."""
    import glob
    patterns = [
        os.path.join(eval_dir, "eval", "samples", "*_validation.xlsx"),
        os.path.join(eval_dir, "eval", "*_validation.xlsx"),
        os.path.join(eval_dir, "*_validation.xlsx"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def compute_per_snr_from_excel(excel_path):
    """Compute per-SNR stats directly from the per-sample Excel file."""
    import pandas as pd
    df = pd.read_excel(excel_path)
    df["snr_level"] = df["Input_SNR"].apply(nearest_snr_level)

    per_snr = {}
    for level in SNR_LEVELS:
        subset = df[df["snr_level"] == level]
        per_snr[level] = {"n": len(subset)}
        for m in METRICS:
            vals = subset[m].dropna().values
            per_snr[level][m] = np.mean(vals) if len(vals) > 0 else float("nan")
            per_snr[level][f"{m}_std"] = np.std(vals) if len(vals) > 0 else float("nan")
    return per_snr


def find_eval_log(eval_dir):
    """Find the eval stdout log in a job results directory."""
    candidates = [
        os.path.join(eval_dir, "eval", "logs", "stdout.txt"),
        os.path.join(eval_dir, "eval_logs", "stdout.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def plot_comparison(model_data, output_path):
    """Create 3-panel comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]

        for model_idx, (model_name, per_snr) in enumerate(model_data.items()):
            x = [l for l in SNR_LEVELS if not np.isnan(per_snr[l][metric])]
            y = [per_snr[l][metric] for l in x]
            yerr = [per_snr[l][f"{metric}_std"] for l in x]

            color = colors[model_idx % len(colors)]
            marker = markers[model_idx % len(markers)]

            ax.errorbar(x, y, yerr=yerr, label=model_name,
                       marker=marker, color=color, capsize=3,
                       linewidth=1.5, markersize=6)

        ax.set_xlabel("Mixture SNR (dB)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(SNR_LEVELS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot per-SNR metric comparison")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Model display names")
    parser.add_argument("--eval-dirs", nargs="+", required=True,
                       help="Evaluation result directories (same order as --models)")
    parser.add_argument("--dataset", default="datasets/DREGON-LM",
                       help="Dataset path for metadata")
    parser.add_argument("--output", default="results/evaluation/per_snr_comparison.png",
                       help="Output plot path")
    args = parser.parse_args()

    if len(args.models) != len(args.eval_dirs):
        parser.error("--models and --eval-dirs must have the same number of arguments")

    metadata = None  # loaded lazily if needed

    model_data = {}
    for name, eval_dir in zip(args.models, args.eval_dirs):
        excel_path = find_eval_excel(eval_dir)
        if excel_path:
            print(f"{name}: reading per-sample data from {excel_path}")
            per_snr = compute_per_snr_from_excel(excel_path)
        else:
            log_path = find_eval_log(eval_dir)
            if log_path is None:
                raise FileNotFoundError(f"No Excel or log found in {eval_dir}")
            if metadata is None:
                metadata = load_metadata(args.dataset)
                print(f"Loaded metadata for {len(metadata)} samples")
            results = parse_eval_log(log_path)
            print(f"{name}: parsed {len(results)} samples from {log_path}")
            per_snr = compute_per_snr(results, metadata)
        model_data[name] = per_snr

    plot_comparison(model_data, args.output)


if __name__ == "__main__":
    main()
