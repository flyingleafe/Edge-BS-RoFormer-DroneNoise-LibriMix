#!/usr/bin/env python3
"""
Compute per-SNR stratified RPS evaluation metrics for the paper.

Uses:
- results/rps_predictor_comparison/dregon_lm_metadata.json  (SNR per sample)
- results/rps_predictor_comparison/val_inference/per_sample_metrics.json  (metrics per sample)

Outputs:
- results/rps_predictor_comparison/per_snr_metrics.json
- results/rps_predictor_comparison/per_snr_metrics.csv
- papers/rps-from-drone-sound/figures/rps_per_snr_table.tex
"""

import json
import numpy as np
import os

# Paths
META_PATH = "results/rps_predictor_comparison/dregon_lm_metadata.json"
METRICS_PATH = "results/rps_predictor/val_inference/per_sample_metrics.json"
OUT_DIR = "results/rps_predictor"
PAPER_FIG_DIR = "papers/rps-from-drone-sound/figures"


def load_data():
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return meta, metrics


def stratify_by_snr(meta, metrics, snr_edges):
    """
    Group samples into SNR bins and compute mean metrics per bin.

    Args:
        meta: dict with 'valid' list containing {'id', 'input_snr', ...}
        metrics: list of dicts with {'sample', 'mse', 'mae_frame', 'mae_clip', 'r2', ...}
        snr_edges: list of bin edges, e.g. [-30, -20, -10, 0]

    Returns:
        list of dicts with bin statistics
    """
    # Build lookup: sample_id -> input_snr
    snr_by_id = {s["id"]: s["input_snr"] for s in meta["valid"]}

    # Attach SNR to each metric entry
    entries = []
    for m in metrics:
        sid = m["sample"]
        snr = snr_by_id.get(sid, None)
        if snr is None:
            raise ValueError(f"Sample {sid} not found in metadata")
        entries.append({
            "sample": sid,
            "snr": snr,
            "mse": m["mse"],
            "mae_frame": m["mae_frame"],
            "mae_clip": m["mae_clip"],
            "r2": m["r2"],
        })

    bins = []
    for i in range(len(snr_edges) - 1):
        lo, hi = snr_edges[i], snr_edges[i + 1]
        label = f"[{lo:.0f}, {hi:.0f})"
        bin_entries = [e for e in entries if lo <= e["snr"] < hi]
        if not bin_entries:
            bins.append({
                "snr_range": label,
                "n": 0,
                "mse_mean": None,
                "mse_std": None,
                "mae_frame_mean": None,
                "mae_clip_mean": None,
                "r2_mean": None,
                "r2_std": None,
            })
            continue

        mses = [e["mse"] for e in bin_entries]
        maes_frame = [e["mae_frame"] for e in bin_entries]
        maes_clip = [e["mae_clip"] for e in bin_entries]
        r2s = [e["r2"] for e in bin_entries if e["r2"] is not None]

        bins.append({
            "snr_range": label,
            "n": len(bin_entries),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "mae_frame_mean": float(np.mean(maes_frame)),
            "mae_clip_mean": float(np.mean(maes_clip)),
            "r2_mean": float(np.mean(r2s)) if r2s else None,
            "r2_std": float(np.std(r2s)) if r2s else None,
        })

    # Overall
    mses = [e["mse"] for e in entries]
    maes_frame = [e["mae_frame"] for e in entries]
    maes_clip = [e["mae_clip"] for e in entries]
    r2s = [e["r2"] for e in entries if e["r2"] is not None]

    bins.append({
        "snr_range": "Overall",
        "n": len(entries),
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "mae_frame_mean": float(np.mean(maes_frame)),
        "mae_clip_mean": float(np.mean(maes_clip)),
        "r2_mean": float(np.mean(r2s)) if r2s else None,
        "r2_std": float(np.std(r2s)) if r2s else None,
    })

    return bins, entries


def print_table(bins):
    print("\nPer-SNR RPS estimation metrics (SimpleConv)")
    print("=" * 90)
    print(f"{'SNR (dB)':<12} {'N':>6} {'MSE':>10} {'RMSE':>8} {'MAE/frame':>10} {'MAE/clip':>10} {'R²':>8} {'R² std':>8}")
    print("-" * 90)
    for b in bins:
        if b["n"] == 0:
            print(f"{b['snr_range']:<12} {0:>6} {'—':>10} {'—':>8} {'—':>10} {'—':>10} {'—':>8} {'—':>8}")
            continue
        rmse = b["mse_mean"] ** 0.5
        print(f"{b['snr_range']:<12} {b['n']:>6} {b['mse_mean']:>10.2f} {rmse:>8.2f} {b['mae_frame_mean']:>10.2f} {b['mae_clip_mean']:>10.2f} {b['r2_mean']:>8.4f} {b['r2_std']:>8.4f}")
    print("=" * 90)


def write_csv(bins, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_range", "n", "mse_mean", "mse_std", "rmse_mean", "mae_frame_mean", "mae_clip_mean", "r2_mean", "r2_std"])
        for b in bins:
            rmse = b["mse_mean"] ** 0.5 if b["mse_mean"] is not None else None
            writer.writerow([
                b["snr_range"],
                b["n"],
                round(b["mse_mean"], 4) if b["mse_mean"] is not None else None,
                round(b["mse_std"], 4) if b["mse_std"] is not None else None,
                round(rmse, 4) if rmse is not None else None,
                round(b["mae_frame_mean"], 4) if b["mae_frame_mean"] is not None else None,
                round(b["mae_clip_mean"], 4) if b["mae_clip_mean"] is not None else None,
                round(b["r2_mean"], 4) if b["r2_mean"] is not None else None,
                round(b["r2_std"], 4) if b["r2_std"] is not None else None,
            ])
    print(f"CSV saved to {path}")


def write_latex_table(bins, path):
    """Generate a publication-ready LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{RPS estimation performance stratified by mixture SNR.}",
        r"  \label{tab:per-snr}",
        r"  \begin{tabular}{lcccccc}",
        r"    \toprule",
        r"    SNR (dB) & $N$ & MSE & RMSE & MAE/frame & MAE/clip & $R^{2}$ \\",
        r"    \midrule",
    ]

    for b in bins:
        if b["n"] == 0:
            lines.append(f"    {b['snr_range']} & 0 & — & — & — & — & — \\\\")
            continue
        rmse = b["mse_mean"] ** 0.5
        lines.append(
            f"    {b['snr_range']} & {b['n']} & "
            f"{b['mse_mean']:.2f} & {rmse:.2f} & "
            f"{b['mae_frame_mean']:.2f} & {b['mae_clip_mean']:.2f} & "
            f"{b['r2_mean']:.4f} \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {path}")


def write_figure(bins, path):
    """Generate a publication-ready grouped bar plot with MSE and R² vs SNR."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Skip "Overall" bin for the figure
    plot_bins = [b for b in bins if b["snr_range"] != "Overall"]
    x_labels = [b["snr_range"] for b in plot_bins]
    x = np.arange(len(x_labels))
    bar_width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4))

    color_mse = "#e41a1c"
    color_r2 = "#377eb8"

    # MSE bars on left axis
    mse_vals = [b["mse_mean"] for b in plot_bins]
    ax1.bar(x - bar_width/2, mse_vals, bar_width,
            color=color_mse, alpha=0.7, label="MSE")
    ax1.set_xlabel("SNR (dB)", fontsize=11)
    ax1.set_ylabel("MSE $(\\mathrm{rev}/\\mathrm{s})^{2}$", color=color_mse, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_mse)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=15, ha="right")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, axis="y")

    # R² bars on right axis
    ax2 = ax1.twinx()
    r2_vals = [b["r2_mean"] for b in plot_bins]
    ax2.bar(x + bar_width/2, r2_vals, bar_width,
            color=color_r2, alpha=0.7, label=r"$R^{2}$")
    ax2.set_ylabel(r"$R^{2}$", color=color_r2, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_r2)
    ax2.set_ylim([0, 1])

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left", framealpha=0.9)

    plt.title("RPS estimation performance vs. mixture SNR", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)

    meta, metrics = load_data()

    # Use the bins the supervisor suggested: [-30, -20], [-20, -10], [-10, 0]
    # We also add finer granularity with [-30, -25, -20, -15, -10, -5, 0]
    # to give the author flexibility.
    snr_edges = [-30, -25, -20, -15, -10, -5, 0]

    bins, entries = stratify_by_snr(meta, metrics, snr_edges)

    print_table(bins)

    # Save JSON
    json_path = os.path.join(OUT_DIR, "per_snr_metrics.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": "RPSPredictor-Standalone (simple_conv)",
            "n_total": len(entries),
            "bins": bins,
        }, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "per_snr_metrics.csv")
    write_csv(bins, csv_path)

    # Save LaTeX table
    tex_path = os.path.join(PAPER_FIG_DIR, "rps_per_snr_table.tex")
    write_latex_table(bins, tex_path)

    # Save figure
    fig_path = os.path.join(PAPER_FIG_DIR, "rps_per_snr.pdf")
    write_figure(bins, fig_path)


if __name__ == "__main__":
    main()
