#!/usr/bin/env python3
"""Batch-run full-sequence and single-rotor evaluations for all SimpleConv variants."""

import subprocess
from pathlib import Path

VARIANTS = [
    ("simple_conv",              "results/rps_exp_simple_conv/best_simple_conv.pt",
     "results/rps_eval_full_sequence/simple_conv",
     "results/rps_eval_single_rotor/simple_conv"),
    ("simple_conv_bigru",        "results/rps_exp_simple_conv_bigru/best_simple_conv_bigru.pt",
     "results/rps_eval_full_sequence/simple_conv_bigru",
     "results/rps_eval_single_rotor/simple_conv_bigru"),
    ("simple_conv_bigru_v2",     "results/rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt",
     "results/rps_eval_full_sequence/simple_conv_bigru_v2",
     "results/rps_eval_single_rotor/simple_conv_bigru_v2"),
    ("simple_conv_v2",           "results/rps_exp_v2/best_simple_conv_v2.pt",
     "results/rps_eval_full_sequence/simple_conv_v2",
     "results/rps_eval_single_rotor/simple_conv_v2"),
    ("simple_conv_tcn",          "results/rps_exp_tcn/best_simple_conv_tcn.pt",
     "results/rps_eval_full_sequence/simple_conv_tcn",
     "results/rps_eval_single_rotor/simple_conv_tcn"),
    ("simple_conv_magphase_bigru", "results/rps_exp_magphase_bigru/best_simple_conv_magphase_bigru.pt",
     "results/rps_eval_full_sequence/simple_conv_magphase_bigru",
     "results/rps_eval_single_rotor/simple_conv_magphase_bigru"),
    ("simple_conv_attn_pool",    "results/rps_exp_attn_pool/best_simple_conv_attn_pool.pt",
     "results/rps_eval_full_sequence/simple_conv_attn_pool",
     "results/rps_eval_single_rotor/simple_conv_attn_pool"),
    ("simple_conv_se_next",      "results/rps_exp_se_next/best_simple_conv_se_next.pt",
     "results/rps_eval_full_sequence/simple_conv_se_next",
     "results/rps_eval_single_rotor/simple_conv_se_next"),
    ("simple_conv_multiscale",   "results/rps_exp_multiscale/best_simple_conv_multiscale.pt",
     "results/rps_eval_full_sequence/simple_conv_multiscale",
     "results/rps_eval_single_rotor/simple_conv_multiscale"),
    ("simple_conv_wide",         "results/rps_exp_wide/best_simple_conv_wide.pt",
     "results/rps_eval_full_sequence/simple_conv_wide",
     "results/rps_eval_single_rotor/simple_conv_wide"),
]

DEVICE = "cpu"

for model, ckpt, out_full, out_single in VARIANTS:
    print(f"\n{'='*60}")
    print(f"Evaluating: {model}")
    print(f"{'='*60}")

    # Full sequence
    cmd = [
        "python", "scripts/eval_rps_full_sequence.py",
        "--model", model,
        "--checkpoint", ckpt,
        "--device", DEVICE,
        "--out_dir", out_full,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

    # Single rotor
    cmd = [
        "python", "scripts/eval_rps_single_rotor.py",
        "--model", model,
        "--checkpoint", ckpt,
        "--device", DEVICE,
        "--out_dir", out_single,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

print("\nAll evaluations complete.")
