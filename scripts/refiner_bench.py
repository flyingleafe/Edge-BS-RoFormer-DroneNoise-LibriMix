"""Score one conditional RPS refiner on the frozen real split (P4's M1-M5).

Thin CLI over :func:`experiments.refiner_bench.run` — the probe that measured
HG-CKLA v1 and put candidate C2 on the shortlist "conditional on three fixes".
Writes ``results.json`` and ``REPORT.md`` into ``--out``.

Usage::

    python scripts/refiner_bench.py --experiment hb_hgckla_ref_v2 \
        --out results/refiner_bench/hb_hgckla_ref_v2

    # add a seed track of your own as a conditioning (an .npy of shape
    # (296, 4, 251) in rev/s, in the frames' order)
    python scripts/refiner_bench.py --experiment hb_hgckla_ref_v2 \
        --extra-cond slotcomb=results/slot_real/seed.npy --out results/rb

    # wiring smoke: the first two clips only
    python scripts/refiner_bench.py --experiment hb_hgckla_ref --n-frames 16 \
        --out /tmp/rb_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.refiner_bench import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, help="experiment name zoo.load resolves")
    ap.add_argument(
        "--extra-cond",
        action="append",
        default=[],
        metavar="NAME=PATH.npy",
        help="extra M2/M5 conditioning, (N, 4, 251) rev/s in the frames' order; repeatable",
    )
    ap.add_argument(
        "--cond-experiment",
        action="append",
        default=None,
        metavar="NAME",
        help="zoo experiment used as an M2/M5 conditioning (dump if present, else "
        "regenerated); repeatable; default r4hb_scv2",
    )
    ap.add_argument("--passes", type=int, default=3, help="M5 iteration depth")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument(
        "--n-frames", type=int, default=None, help="truncate the set (SMOKE, not a result)"
    )
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--batch", type=int, default=8, help="frames per forward pass")
    args = ap.parse_args()

    extra: dict[str, np.ndarray] = {}
    for item in args.extra_cond:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--extra-cond wants NAME=PATH.npy, got {item!r}")
        extra[name] = np.load(path)

    results = run(
        args.experiment,
        extra_conds=extra or None,
        cond_experiments=tuple(args.cond_experiment or ["r4hb_scv2"]),
        passes=args.passes,
        out_dir=args.out,
        n_frames=args.n_frames,
        device=args.device,
        batch=args.batch,
    )
    m3 = results["M3"].get("cruise") or results["M3"].get("all")
    print(f"wrote {args.out}/REPORT.md ({results['inference_seconds']} s of inference)")
    if m3:
        print(f"M3 oracle floor: {m3['refined']:.4f} rev/s PIT MAE")


if __name__ == "__main__":
    main()
