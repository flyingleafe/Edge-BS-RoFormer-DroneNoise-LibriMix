"""Score checkpoints on each PART of a matched-mixture validation set separately.

A `MixtureMatchedValidDataset` concatenates the whole frozen real split with one
synthetic part per non-real source of the training policy, sized by that policy's
own weights. One averaged number over the concatenation is what training
monitors -- but it cannot say whether an arm improved on REAL audio or only on
the synthetic parts that share the training generator.

This CLI reports the metrics per part. The real part is the whole frozen split
and is not subsampled, so its row is directly comparable with every real-only
row in the project.

    python scripts/eval_matched_val_parts.py \
        --experiments m3mixv2_unigru128,m3mixv2_transformer,m3mixv2_scv2

`--skip-kinds generated` drops the parts whose producer needs a GPU; the parts
that remain are seeded exactly as they are in the whole set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import zoo  # noqa: E402
from data_processing.frame_datasets import MixtureMatchedValidDataset  # noqa: E402
from metrics import RPSMetric  # noqa: E402

STATS = ("mse", "rmse", "mae_frame", "mae_clip")


def build_valid(
    data_yaml: str, skip_kinds: tuple[str, ...], policy: str | None = None
) -> MixtureMatchedValidDataset:
    cfg = OmegaConf.load(data_yaml)
    params: dict[str, Any] = dict(
        cast(dict, OmegaConf.to_container(cfg.valid.params, resolve=True))
    )
    params["skip_kinds"] = list(skip_kinds)
    if policy:
        # A COPY of the training policy, used only to move a producer off the
        # GPU. The source ORDER must be unchanged: each part is seeded
        # `base_seed + i` from its index in the full source list.
        params["policy_path"] = policy
    return MixtureMatchedValidDataset(**params)


def part_slices(ds: MixtureMatchedValidDataset) -> list[tuple[str, int, int]]:
    """``[(name, start, stop)]`` over the concatenation, in construction order."""
    out, off = [], 0
    for name, n in ds.counts.items():
        if n <= 0:
            continue
        out.append((name, off, off + n))
        off += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default="conf/data/m3mix_matchedval.yaml")
    ap.add_argument("--experiments", required=True, help="comma-separated experiment names")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--skip-kinds", default="", help="comma-separated source kinds to drop")
    ap.add_argument(
        "--policy",
        default=None,
        help="override the policy path (a copy with the same source ORDER)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="frames per part (0 = all)")
    ap.add_argument("--out", default="results/matched_val_parts.json")
    a = ap.parse_args()

    skip = tuple(k for k in a.skip_kinds.split(",") if k)
    ds = build_valid(a.data, skip, a.policy)
    parts = part_slices(ds)
    print(f"{len(ds)} frames: " + ", ".join(f"{n} {b - s}" for n, s, b in parts), flush=True)

    metrics = {s: RPSMetric(s, rate=(16000, 512)) for s in STATS}
    rows: dict[str, dict[str, dict[str, float]]] = {}

    for exp in a.experiments.split(","):
        exp = exp.strip()
        fm = zoo.load(exp, ckpt=a.ckpt, device=a.device)
        rows[exp] = {}
        for name, start, stop in parts:
            idx = range(start, stop if not a.limit else min(stop, start + a.limit))
            acc: dict[str, list[float]] = {s: [] for s in STATS}
            for i in idx:
                frame = ds[i]
                pred = fm(frame)
                for s, m in metrics.items():
                    acc[s].append(float(m(pred, frame)))
            rows[exp][name] = {s: float(np.mean(v)) for s, v in acc.items()}
            print(
                f"  {exp:24s} {name:16s} "
                + "  ".join(f"{s}={np.mean(v):8.4f}" for s, v in acc.items()),
                flush=True,
            )

    print(f"\n{'experiment':>24s} {'part':>16s} " + " ".join(f"{s:>10s}" for s in STATS))
    for exp, per in rows.items():
        for name, vals in per.items():
            print(f"{exp:>24s} {name:>16s} " + " ".join(f"{vals[s]:10.4f}" for s in STATS))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
