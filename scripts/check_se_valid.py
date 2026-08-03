#!/usr/bin/env python
"""Integrity check for published SE validation sets.

For each clip of a ``tdframe-v1`` SE valid set (``{mixture, target, meta}``)
this script measures mean power of the mixture, the target, and the implied
noise (``mixture - target``), then reports:

- all-zero / silent MIXTURE clips (the F1 zeroing bug's signature),
- all-zero / silent TARGET clips,
- silent-NOISE clips (a silent noise draw that entered a mix),
- per-category counts.

Exit code is nonzero when any silent mixture or target is found, so the
script can gate a republish. Silence threshold = ``mixing.MIN_DRAW_POWER``.

Usage::

    python scripts/check_se_valid.py SE-valid-drone SE-valid-harmonic
    python scripts/check_se_valid.py SE-valid-drone@<version>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from data_processing.mixing import MIN_DRAW_POWER
from data_processing.streams import iter_published_frames


def check_dataset(spec: str) -> dict:
    name, _, version = spec.partition("@")
    n = 0
    bad_mix: list[str] = []
    bad_target: list[str] = []
    silent_noise: list[str] = []
    per_category: Counter[str] = Counter()
    bad_per_category: Counter[str] = Counter()
    for frame in iter_published_frames(name, version or None):
        n += 1
        meta = frame["meta"]
        clip_id = str(meta["id"]) if "id" in meta else f"clip_{n:05d}"
        category = str(meta["category"]) if "category" in meta else "?"
        per_category[category] += 1
        mix = np.asarray(frame["mixture"].data, dtype=np.float64)
        tgt = np.asarray(frame["target"].data, dtype=np.float64)
        noise = mix - tgt
        is_bad = False
        if float(np.mean(mix**2)) <= MIN_DRAW_POWER:
            bad_mix.append(clip_id)
            is_bad = True
        if float(np.mean(tgt**2)) <= MIN_DRAW_POWER:
            bad_target.append(clip_id)
            is_bad = True
        if float(np.mean(noise**2)) <= MIN_DRAW_POWER:
            silent_noise.append(clip_id)
            is_bad = True
        if is_bad:
            bad_per_category[category] += 1
    return {
        "dataset": spec,
        "n_clips": n,
        "bad_mixture": bad_mix,
        "bad_target": bad_target,
        "silent_noise": silent_noise,
        "per_category": dict(per_category),
        "bad_per_category": dict(bad_per_category),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+", help="dataset name, optionally NAME@version")
    args = ap.parse_args()

    failed = False
    for spec in args.datasets:
        r = check_dataset(spec)
        n_bad = len(set(r["bad_mixture"]) | set(r["bad_target"]) | set(r["silent_noise"]))
        print(f"\n=== {r['dataset']}: {r['n_clips']} clips, {n_bad} defective ===")
        for key in ("bad_mixture", "bad_target", "silent_noise"):
            ids = r[key]
            frac = len(ids) / max(r["n_clips"], 1)
            print(f"  {key}: {len(ids)} ({frac:.2%})")
            for clip_id in ids[:20]:
                print(f"    - {clip_id}")
        if r["bad_per_category"]:
            print(f"  defective per category: {r['bad_per_category']}")
        if r["bad_mixture"] or r["bad_target"] or r["silent_noise"]:
            failed = True
    print("\nRESULT:", "FAIL (defective clips found)" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
