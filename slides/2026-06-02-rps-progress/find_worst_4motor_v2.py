#!/usr/bin/env python3
"""Find the worst V2 sample for old models where all 4 motors are active (not synthetic partial)."""

import json
import os
import numpy as np
from pathlib import Path

BASE = Path("results/rps_cross_eval/samples")

samples = sorted(BASE.glob("v2_sample_*"))

results = []
for sample_dir in samples:
    sid = sample_dir.name
    
    # Load metrics
    with open(sample_dir / "metrics_old_simple_conv.json") as f:
        m_sc = json.load(f)["pit_mse"]
    with open(sample_dir / "metrics_old_bigru_v2.json") as f:
        m_bg = json.load(f)["pit_mse"]
    
    # Load RPS target to check if all 4 motors are active
    rps = np.load(sample_dir / "rps_target.npy")  # (4, T)
    # A motor is "active" if its mean RPS > 5 Hz (not zero/idle)
    active = [np.mean(rps[i]) > 5.0 for i in range(4)]
    n_active = sum(active)
    
    # Also check if it's a synthetic motor combo by looking for constant RPS
    # (std < 0.1 means constant speed)
    is_constant = [np.std(rps[i]) < 0.1 for i in range(4)]
    n_constant = sum(is_constant)
    
    results.append({
        "sid": sid,
        "sc_mse": m_sc,
        "bg_mse": m_bg,
        "mean_mse": (m_sc + m_bg) / 2,
        "n_active": n_active,
        "n_constant": n_constant,
        "mean_rps": [float(np.mean(rps[i])) for i in range(4)],
    })

print("All V2 samples:")
for r in results:
    flag = "CONSTANT" if r["n_constant"] > 0 else ""
    print(f"  {r['sid']}: active={r['n_active']}, constant={r['n_constant']}, "
          f"SC={r['sc_mse']:.1f}, BG={r['bg_mse']:.1f}, mean={r['mean_mse']:.1f} {flag}")

# Filter to 4 active motors, not synthetic constant
filtered = [r for r in results if r["n_active"] == 4 and r["n_constant"] == 0]
print(f"\nSamples with 4 active motors (non-constant): {len(filtered)}")
for r in sorted(filtered, key=lambda x: x["mean_mse"], reverse=True):
    print(f"  {r['sid']}: SC={r['sc_mse']:.1f}, BG={r['bg_mse']:.1f}, mean={r['mean_mse']:.1f}")
