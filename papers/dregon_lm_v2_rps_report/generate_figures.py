#!/usr/bin/env python3
"""Generate figures for the DREGON-LM-V2 cross-evaluation report."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def parse_log(log_path):
    """Parse training log to extract epoch-wise metrics."""
    epochs, train_mse, val_mse, val_pit_mse = [], [], [], []
    with open(log_path) as f:
        for line in f:
            # Look for lines with Train MSE and Val MSE
            m = re.match(
                r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\deE+-]+)",
                line
            )
            if m:
                epochs.append(int(m.group(1)))
                train_mse.append(float(m.group(2)))
                val_mse.append(float(m.group(3)))
    return {
        "epochs": np.array(epochs),
        "train_mse": np.array(train_mse),
        "val_mse": np.array(val_mse),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

with open(RESULTS_DIR / "rps_cross_eval" / "validation_metrics.json") as f:
    cross_eval = json.load(f)

v3_sc_log = parse_log(RESULTS_DIR / "rps_predictor_v3" / "simple_conv.log")
v3_bv_log = parse_log(RESULTS_DIR / "rps_predictor_v3" / "simple_conv_bigru_v2.log")

# ---------------------------------------------------------------------------
# Figure 1: Cross-evaluation — all 4 models × 2 datasets
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

models = ["OLD SC", "OLD BiGRU", "V3 SC", "V3 BiGRU"]
old_valid = [
    cross_eval["old_simple_conv__OLD_valid"]["pit_mse"],
    cross_eval["old_bigru_v2__OLD_valid"]["pit_mse"],
    cross_eval["v3_simple_conv__OLD_valid"]["pit_mse"],
    cross_eval["v3_bigru_v2__OLD_valid"]["pit_mse"],
]
v2_valid = [
    cross_eval["old_simple_conv__V2_valid"]["pit_mse"],
    cross_eval["old_bigru_v2__V2_valid"]["pit_mse"],
    cross_eval["v3_simple_conv__V2_valid"]["pit_mse"],
    cross_eval["v3_bigru_v2__V2_valid"]["pit_mse"],
]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, old_valid, width, label="OLD valid (1 s, overlap)", color="#1f77b4", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, v2_valid, width, label="V2 valid (3 s, no overlap)", color="#ff7f0e", edgecolor="black", linewidth=0.5)

ax.set_ylabel("PIT-MSE [(rev/s)$^2$]")
ax.set_title("Cross-evaluation: old checkpoints vs V3 models")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc="upper left")
ax.set_ylim(0, 360)
ax.grid(True, alpha=0.3, axis="y")

# Annotate bars
for bar in bars1:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_cross_eval.pdf")
plt.savefig(FIG_DIR / "fig_cross_eval.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Degradation factor (V2 / OLD)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.5))

degradation = [v2_valid[i] / old_valid[i] for i in range(4)]
colors = ["#d62728" if d > 10 else "#2ca02c" for d in degradation]

bars = ax.bar(models, degradation, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=1, color="black", linestyle="--", linewidth=0.8, label="No degradation")
ax.set_ylabel("Degradation factor (V2 MSE / OLD MSE)")
ax.set_title("How much harder is V2?")
ax.legend()
ax.set_ylim(0, max(degradation) * 1.15)
ax.grid(True, alpha=0.3, axis="y")

for bar, d in zip(bars, degradation):
    ax.annotate(f"{d:.1f}$\\times$", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_degradation.pdf")
plt.savefig(FIG_DIR / "fig_degradation.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: PIT vs Std MSE gap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.5))

models_gap = ["OLD SC", "OLD BiGRU", "V3 SC", "V3 BiGRU"]
pit_vals = [
    cross_eval["old_simple_conv__V2_valid"]["pit_mse"],
    cross_eval["old_bigru_v2__V2_valid"]["pit_mse"],
    cross_eval["v3_simple_conv__V2_valid"]["pit_mse"],
    cross_eval["v3_bigru_v2__V2_valid"]["pit_mse"],
]
std_vals = [
    cross_eval["old_simple_conv__V2_valid"]["std_mse"],
    cross_eval["old_bigru_v2__V2_valid"]["std_mse"],
    cross_eval["v3_simple_conv__V2_valid"]["std_mse"],
    cross_eval["v3_bigru_v2__V2_valid"]["std_mse"],
]

gaps = [(s - p) / p * 100 for p, s in zip(pit_vals, std_vals)]

bars = ax.bar(models_gap, gaps, color=["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"], edgecolor="black", linewidth=0.5)
ax.set_ylabel("Std MSE excess over PIT-MSE (%)")
ax.set_title("Rotor-order ambiguity on V2 valid")
ax.grid(True, alpha=0.3, axis="y")

for bar, g in zip(bars, gaps):
    ax.annotate(f"{g:.0f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_pit_std_gap.pdf")
plt.savefig(FIG_DIR / "fig_pit_std_gap.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: V3 training curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, data, title in zip(axes, [v3_sc_log, v3_bv_log],
                            ["V3 SimpleConv (538K params)", "V3 BiGRU-v2 (1.44M params)"]):
    ax.plot(data["epochs"], data["train_mse"], "o-", label="Train MSE", markersize=3, linewidth=1.2, color="#1f77b4")
    ax.plot(data["epochs"], data["val_mse"], "s-", label="Val MSE", markersize=3, linewidth=1.2, color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE [(rev/s)$^2$]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    # Clip y to see the converged region; first-epoch spike is ~4500
    ax.set_ylim(0, max(data["val_mse"][5:].max(), data["train_mse"][5:].max()) * 1.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_v3_training_curves.pdf")
plt.savefig(FIG_DIR / "fig_v3_training_curves.png")
plt.close()

# ---------------------------------------------------------------------------
# Table: LaTeX cross-evaluation summary
# ---------------------------------------------------------------------------
table_tex = r"""\begin{tabular}{lcccc}
\toprule
Model & OLD valid & V2 valid & Degradation & V2 MAE \\
\midrule
OLD SimpleConv & 5.2 & 331.9 & 63$\times$ & 7.68 \\
OLD BiGRU-v2 & 2.7 & 327.3 & 123$\times$ & 6.29 \\
V3 SimpleConv & 66.8 & 148.1 & 2.2$\times$ & 10.57 \\
V3 BiGRU-v2 & 15.3 & 71.1 & 4.7$\times$ & 4.34 \\
\bottomrule
\end{tabular}
"""
with open(FIG_DIR / "table_cross_eval.tex", "w") as f:
    f.write(table_tex)

# Table: per-channel
table_ch = r"""\begin{tabular}{lccc}
\toprule
Model & ch0 & ch1--7 & ch1--7/ch0 \\
\midrule
OLD SimpleConv & 293.6 & 338.0 & 1.15$\times$ \\
OLD BiGRU-v2 & 247.2 & 340.1 & 1.38$\times$ \\
V3 SimpleConv & 126.9 & 151.5 & 1.19$\times$ \\
V3 BiGRU-v2 & 66.2 & 71.9 & 1.09$\times$ \\
\bottomrule
\end{tabular}
"""
with open(FIG_DIR / "table_per_channel.tex", "w") as f:
    f.write(table_ch)

# Table: in-flight (median PIT MSE)
table_if = r"""\begin{tabular}{lcc}
\toprule
Model & speech-high & whitenoise-high \\
\midrule
OLD SimpleConv & 499 & 542 \\
OLD BiGRU-v2 & 504 & 542 \\
V3 SimpleConv & 639 & 509 \\
V3 BiGRU-v2 & 481 & 532 \\
\bottomrule
\end{tabular}
"""
with open(FIG_DIR / "table_inflight.tex", "w") as f:
    f.write(table_if)

print(f"Figures saved to {FIG_DIR}")
