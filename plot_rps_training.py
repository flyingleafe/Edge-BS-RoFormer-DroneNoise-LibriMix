#!/usr/bin/env python3
"""Plot RPS predictor training curves: train/val MSE loss and validation R²."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results/rps_predictor/training_log.csv"
OUT_PATH = "results/rps_predictor/training_curves.png"

# Naive baseline: predict training-set mean → R² = 0 by definition on train,
# and on val it depends on distribution shift.  For MSE, naive = 365.14.
NAIVE_VAL_MSE = 365.1363
# Naive R² on validation set (computed from val targets): predicting the
# per-element mean of the *training* set on validation data.
# R² = 1 - MSE_naive / Var(targets).  We estimated MSE_naive = 365.14
# and from the training log epoch-1 we can see total variance ≈ MSE at R²=0
# → Var ≈ 101.44 (from 1 - 184.98/Var = -0.8239 → Var = 184.98/1.8239 ≈ 101.4).
# More directly: R²_naive = 1 - 365.14 / Var.  But since our R² is computed as
# 1 - SS_res/SS_tot with SS_tot centered on the *validation* mean, the naive
# baseline predicting the *train* mean will have R² < 0 if means differ.
# We can back-compute: at epoch 1, model MSE=184.98 and R²=-0.8239
# → SS_tot_per_elem = 184.98 / (1+0.8239) ≈ 101.4
# → naive R² = 1 - 365.14/101.4 ≈ -2.60
VAR_PER_ELEM = 101.4  # approximate validation SS_tot per element
NAIVE_R2 = 1 - NAIVE_VAL_MSE / VAR_PER_ELEM

df = pd.read_csv(CSV_PATH)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Panel 1: MSE loss (log scale, clipped) ---
ax = axes[0]
ax.plot(df["epoch"], df["train_mse"], "o-", ms=3, label="Train MSE", color="#2196F3")
ax.plot(df["epoch"], df["val_mse"], "o-", ms=3, label="Val MSE", color="#F44336")
ax.axhline(NAIVE_VAL_MSE, ls="--", color="gray", lw=1, label=f"Naive baseline ({NAIVE_VAL_MSE:.1f})")

# Mark best epoch
best_idx = df["val_mse"].idxmin()
best_row = df.iloc[best_idx]
ax.plot(best_row["epoch"], best_row["val_mse"], "*", ms=14, color="gold",
        markeredgecolor="k", zorder=5,
        label=f"Best (ep {int(best_row['epoch'])}, MSE={best_row['val_mse']:.2f})")

ax.set_yscale("log")
ax.set_ylim(bottom=3, top=max(df["val_mse"].max() * 1.1, NAIVE_VAL_MSE * 1.3))
ax.set_ylabel("MSE Loss (log)")
ax.set_title("RPS Predictor — Training Curves")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 2: Validation R² ---
ax = axes[1]
ax.plot(df["epoch"], df["r2"], "o-", ms=3, label="Val R²", color="#4CAF50")
ax.axhline(NAIVE_R2, ls="--", color="gray", lw=1,
           label=f"Naive mean baseline R² ({NAIVE_R2:.2f})")
ax.axhline(0, ls=":", color="black", lw=0.5)

ax.plot(best_row["epoch"], best_row["r2"], "*", ms=14, color="gold",
        markeredgecolor="k", zorder=5,
        label=f"Best (ep {int(best_row['epoch'])}, R²={best_row['r2']:.3f})")

ax.set_ylim(-1.5, 1.02)
ax.set_xlabel("Epoch")
ax.set_ylabel("R²")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
plt.close()
