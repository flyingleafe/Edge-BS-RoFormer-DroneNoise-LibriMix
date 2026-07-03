#!/usr/bin/env python3
"""Check: do online-mix mixtures with generated noise + augmentation produce NaN?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_processing.online_mixing import OnlineMixIterableDataset


def sample_batches(ds, n_batches, skip_first, label):
    it = iter(ds)
    for i in range(skip_first):
        try:
            next(it)
        except StopIteration:
            pass
        if i % 500 == 0 and i > 0:
            print(f"  {label}: skipped {i}/{skip_first}...")

    mixtures, rps_vals = [], []
    has_nan, has_inf = False, False
    extremes = []
    for i in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        mix = batch[0].numpy()
        rps = batch[1].numpy()
        if np.any(np.isnan(mix)):
            has_nan = True
            print(f"  {label} batch {i}: NaN in mixture!")
        if np.any(np.isinf(mix)):
            has_inf = True
            print(f"  {label} batch {i}: Inf in mixture!")
        if np.any(np.isnan(rps)):
            has_nan = True
            print(f"  {label} batch {i}: NaN in RPS!")
        mixtures.append(mix)
        rps_vals.append(rps)
        extremes.append({
            "mix_max": float(np.max(np.abs(mix))),
            "mix_rms": float(np.sqrt(np.mean(mix**2))),
            "mix_min": float(np.min(mix)),
            "rps_max": float(np.max(rps)),
            "rps_min": float(np.min(rps)),
        })
    if not mixtures:
        print(f"  {label}: NO batches!")
        return None
    all_mix = np.concatenate([m.flatten() for m in mixtures])
    all_rps = np.concatenate([r.flatten() for r in rps_vals])
    mix_maxes = [e["mix_max"] for e in extremes]
    mix_rmses = [e["mix_rms"] for e in extremes]
    print(f"\n  {label} ({len(mixtures)} batches):")
    print(f"    mixture shape: {mixtures[0].shape}")
    print(f"    mixture |max|: mean={np.mean(mix_maxes):.4f}, max={np.max(mix_maxes):.4f}, p99={np.percentile(mix_maxes,99):.4f}")
    print(f"    mixture RMS:   mean={np.mean(mix_rmses):.4f}, std={np.std(mix_rmses):.4f}, min={np.min(mix_rmses):.4f}")
    print(f"    mixture values: mean={all_mix.mean():.4f}, std={all_mix.std():.4f}, min={all_mix.min():.4f}, max={all_mix.max():.4f}")
    print(f"    NaN: {has_nan}, Inf: {has_inf}")
    print(f"    RPS: mean={all_rps.mean():.2f}, std={all_rps.std():.2f}, min={all_rps.min():.2f}, max={all_rps.max():.2f}")
    return {"has_nan": has_nan, "has_inf": has_inf, "n_batches": len(mixtures)}


def main():
    CONFIG = REPO / "configs/online_mix_generated_augment_gpfs.yaml"
    cfg = OmegaConf.load(CONFIG)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print("=== Config loaded ===")
    for src in cfg["sources"]["noise"]:
        print(f"  noise: kind={src.get('kind','?')}, weight={src.get('weight',1.0)}")
    ds = OnlineMixIterableDataset.from_config(cfg)

    print("\n=== BEFORE augmentation (first 100 batches) ===")
    before = sample_batches(ds, 100, skip_first=0, label="BEFORE")

    # 50k samples / batch_size=16 = 3125 batches to cross aug threshold
    print("\n=== AFTER augmentation (skip 3500 batches, sample 100) ===")
    after = sample_batches(ds, 100, skip_first=3500, label="AFTER")

    print("\n=== SUMMARY ===")
    for label, res in [("BEFORE aug", before), ("AFTER aug", after)]:
        if res:
            print(f"  {label}: NaN={res['has_nan']}, Inf={res['has_inf']}, batches={res['n_batches']}")
    print("Done.")


if __name__ == "__main__":
    main()
