#!/usr/bin/env python3
"""Aggregate extra-evaluation results into summary tables."""

import json
from pathlib import Path
import numpy as np

VARIANTS = [
    "simple_conv",
    "simple_conv_bigru",
    "simple_conv_bigru_v2",
    "simple_conv_v2",
    "simple_conv_tcn",
    "simple_conv_magphase_bigru",
    "simple_conv_attn_pool",
    "simple_conv_se_next",
    "simple_conv_multiscale",
    "simple_conv_wide",
]

print("=" * 90)
print("EXTRA EVALUATION RESULTS — SimpleConv Variants")
print("=" * 90)

# ── 1. Full-sequence (speech-high room1) ────────────────────────────────────
print("\n1. FULL-SEQUENCE EVALUATION: DREGON free-flight speech-high room1")
print("-" * 90)
print(f"{'Model':<28} {'Global MSE':>10} {'Global MAE':>10} {'Global R²':>10} {'In-flight MSE':>14} {'In-flight MAE':>14}")
print("-" * 90)

full_results = {}
for v in VARIANTS:
    p = Path(f"results/rps_eval_full_sequence/{v}/metrics.json")
    if p.exists():
        with open(p) as f:
            m = json.load(f)
        full_results[v] = m
        inflight_mse = m.get("mse_inflight")
        inflight_mae = m.get("mae_inflight")
        print(f"{v:<28} {m['mse']:>10.2f} {m['mae']:>10.2f} {m['r2']:>10.4f} "
              f"{inflight_mse if inflight_mse is not None else 'N/A':>14} "
              f"{inflight_mae if inflight_mae is not None else 'N/A':>14}")
    else:
        print(f"{v:<28} MISSING")

# ── 2. Single-rotor summary ────────────────────────────────────────────────
print("\n2. SINGLE-ROTOR EVALUATION: Clean individual motor recordings")
print("-" * 90)
print(f"{'Model':<28} {'Best MSE':>10} {'Best MAE':>10} {'Avg MSE':>10} {'Avg MAE':>10} {'allMotors MSE':>14} {'allMotors MAE':>14}")
print("-" * 90)

single_results = {}
for v in VARIANTS:
    p = Path(f"results/rps_eval_single_rotor/{v}/metrics.json")
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        single_results[v] = data
        # Aggregate across all single-rotor files (excluding allMotors)
        best_mses = []
        best_maes = []
        avg_mses = []
        avg_maes = []
        allmotors = None
        for r in data["results"]:
            if r["motor_id"] == "all":
                allmotors = r
            else:
                best_mses.append(r["best_rotor"]["mse"])
                best_maes.append(r["best_rotor"]["mae"])
                avg_mses.append(r["avg"]["mse"])
                avg_maes.append(r["avg"]["mae"])

        print(f"{v:<28} {np.mean(best_mses):>10.1f} {np.mean(best_maes):>10.1f} "
              f"{np.mean(avg_mses):>10.1f} {np.mean(avg_maes):>10.1f} "
              f"{allmotors['best_rotor']['mse']:>14.2f} {allmotors['best_rotor']['mae']:>14.2f}" if allmotors else "N/A")
    else:
        print(f"{v:<28} MISSING")

# ── 3. allMotors_70 per-variant detail ─────────────────────────────────────
print("\n3. allMotors_70 DETAIL (best rotor / avg over 4)")
print("-" * 70)
print(f"{'Model':<28} {'Best MSE':>10} {'Best MAE':>10} {'Avg MSE':>10} {'Avg MAE':>10}")
print("-" * 70)
for v in VARIANTS:
    if v in single_results:
        for r in single_results[v]["results"]:
            if r["motor_id"] == "all":
                print(f"{v:<28} {r['best_rotor']['mse']:>10.2f} {r['best_rotor']['mae']:>10.2f} "
                      f"{r['avg']['mse']:>10.2f} {r['avg']['mae']:>10.2f}")
                break

# ── Save aggregate JSON ───────────────────────────────────────────────────
aggregate = {
    "full_sequence": full_results,
    "single_rotor": single_results,
}
out_path = Path("results/rps_extra_evals_aggregate.json")
with open(out_path, "w") as f:
    json.dump(aggregate, f, indent=2)
print(f"\nSaved aggregate results to {out_path}")
