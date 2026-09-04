#!/usr/bin/env python
"""Decode the frozen real split with a trained (or untrained) C1 arm and dump it.

Writes, under ``--out``:

* ``pred_clips.npy``  ``(37, 4, 251)`` rev/s, one row per clip (eight mics
  decoded jointly), rotor rows sorted ascending as ``SlotCombNet.decode`` emits
  them;
* ``pred_frames.npy`` ``(296, 4, 251)``: the same track repeated for the eight
  mono frames of each clip, in ``rps_bench.part("real")`` order — the shape
  ``experiments.refiner_bench`` takes as an ``extra_conds`` entry, so the C2
  refiner can be run behind C1;
* ``table.json``: the P1c table of ``experiments.slot_real.score_real``.

``--best`` is the trainer's ``best.pt`` (trainable parameters only); omit it
for the zero-parameter corner (``--parts none``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", default="none", help="comma list, or 'none' for the corner")
    ap.add_argument("--best", default="", help="best.pt of the trained arm (optional)")
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--floor-hz", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="arm")
    ap.add_argument(
        "--octave",
        action="store_true",
        help="turn the decoder's coverage-judged octave move ON (the P1c protocol "
        "keeps it off: at eight mics it cost the untrained FLY124 22.4 -> 31.1)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    from train_slot_real import build_model

    from experiments import slot_real as sr

    net, parts = build_model(args, args.device)
    if args.best:
        missing = net.load_state_dict(torch.load(args.best, map_location=args.device), strict=False)
        print(f"loaded {args.best}: unexpected {missing.unexpected_keys}", flush=True)
    clips = sr.real_clips()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net.eval()
    preds = []
    rows = []
    kw = {"subgrid": True, "octave": bool(args.octave), "relocate": True}  # P1c: octave off
    with torch.no_grad():
        for clip in clips:
            t0 = time.time()
            au = torch.as_tensor(clip["audio"], device=args.device)
            pred = net.decode(au, **kw)[0].cpu().numpy().astype(np.float64)
            mae, gt = sr._align(pred, clip["rps"])
            preds.append(pred)
            rows.append(
                {
                    "clip": clip["clip"],
                    "phase": clip["phase"],
                    "rig": clip["rig"],
                    "mae": mae,
                    "wall": time.time() - t0,
                    "pred": pred,
                    "gt": gt,
                }
            )
    res = sr.table(rows, name=args.name, floor_bins=int(net.floor_bins))
    res["rows"] = [{k: v for k, v in r.items() if k not in ("pred", "gt")} for r in rows]
    pc = np.stack(preds)  # (37, 4, T)
    if pc.shape[-1] != 251:
        from experiments.rps_bench import resample_like_metric

        pc = np.stack([resample_like_metric(p, 251) for p in pc])
    pf = np.repeat(pc, 8, axis=0)  # frame i = clip i // 8
    np.save(out / "pred_clips.npy", pc)
    np.save(out / "pred_frames.npy", pf)
    res["parts"] = list(parts)
    res["best"] = args.best
    (out / "table.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out}/pred_clips.npy {pc.shape}, pred_frames.npy {pf.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
