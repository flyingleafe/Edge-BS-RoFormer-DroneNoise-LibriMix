"""What fraction of each arm's TRAINING stream is zero, ramp and cruise?

Sixteen arms sit on a ramp-against-cruise frontier at Spearman -0.57, and the
two arms built to combine the cell winners' ingredients (W and X) landed on the
frontier rather than above it. So the trade-off is not a knob in the stream
description. The obvious remaining suspect is not WHAT the stream contains but
HOW MUCH of each regime it contains: a stream that is mostly cruise should teach
cruise, and one full of ramps should teach ramps, and a single scalar of that
kind would trace exactly the curve the campaign keeps landing on.

This measures that scalar per policy, using the evaluation's own regime rule on
the chunk's rotor track (max rotor < 1 is zero, mean < 45 is ramp, else cruise).
Pair it with scripts/transfer_board.py's cells: if an arm's ramp cell tracks its
stream's ramp share, the frontier is a sampling problem and the fix is
regime-balanced sampling or a per-regime loss weight — neither of which is a
property of the noise model at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SR = 16000
WIN = 1.0


def regime_shares(policy: str, n_chunks: int) -> dict:
    import yaml

    from data_processing.online_mixing import build_noise_stream

    cfg = yaml.safe_load(Path(policy).read_text())
    specs = cfg["sources"]["noise"]
    pipeline, _ = build_noise_stream(
        specs, sample_rate=SR, window_s=WIN, seed=int(cfg.get("base_seed", 0))
    )
    counts = {"zero": 0, "low": 0, "flight": 0}
    frames = 0
    for i, chunk in enumerate(pipeline):
        if i >= n_chunks:
            break
        rps = np.asarray(chunk["rps"].data, dtype=np.float64)
        if rps.ndim == 1:
            rps = rps[None]
        lab = np.full(rps.shape[1], "low", dtype=object)
        lab[rps.max(axis=0) < 1.0] = "zero"
        lab[rps.mean(axis=0) >= 45.0] = "flight"
        for k in counts:
            counts[k] += int((lab == k).sum())
        frames += lab.size
    frames = max(frames, 1)
    return {k: counts[k] / frames for k in counts} | {"frames": frames}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", nargs="+", required=True)
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    print(f"{'policy':34s} {'zero':>8s} {'ramp':>8s} {'cruise':>8s}")
    print("-" * 62)
    for policy in args.policy:
        try:
            sh = regime_shares(policy, args.chunks)
        except Exception as exc:  # noqa: BLE001
            print(f"{Path(policy).stem:34s} FAILED ({exc!r})", flush=True)
            continue
        rows.append({"policy": Path(policy).stem, **sh})
        print(
            f"{Path(policy).stem:34s} {100 * sh['zero']:7.1f}% {100 * sh['low']:7.1f}% "
            f"{100 * sh['flight']:7.1f}%",
            flush=True,
        )
    # The frozen split, for reference: 12.7% zero, 13.5% ramp, 73.8% cruise.
    print(f"\n{'FROZEN VALIDATION SPLIT':34s} {12.7:7.1f}% {13.5:7.1f}% {73.8:7.1f}%")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
