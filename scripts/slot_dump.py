#!/usr/bin/env python
"""Decode a benchmark part with a trained (or untrained) C1 arm and dump it.

``--part real8`` (the default) is the eight-microphone read of the frozen real
split. It writes, under ``--out``:

* ``pred_clips.npy``  ``(37, 4, 251)`` rev/s, one row per clip (eight mics
  decoded jointly), rotor rows sorted ascending as ``SlotCombNet.decode`` emits
  them;
* ``pred_frames.npy`` ``(296, 4, 251)``: the same track repeated for the eight
  mono frames of each clip, in ``rps_bench.part("real")`` order — the shape
  ``experiments.refiner_bench`` takes as an ``extra_conds`` entry, so the C2
  refiner can be run behind C1;
* ``table.json``: the P1c table of ``experiments.slot_real.score_real``.

The other three parts write the format `scripts/rps_dump.py` writes, so a C1
arm reads on the same tables and in the same notebook as every neural model:
``<out>/<set>/<name>.npz`` with ``pred`` ``(N, 4, T)`` NaN-padded, ``n_t`` and
the per-frame PIT MAE ``metric``. ``--out`` defaults to ``results/rps_dump``
there, and the set's ``_gt.npz`` / ``_meta.json`` are left alone.

* ``real_mono`` — the 296 mono frames of ``rps_bench.part("real")``, one
  microphone at a time; the set is ``real``. This is the FAIR comparison: the
  neural models never see the eight-microphone average.
* ``comb`` / ``stoch`` — the 256 mono 8 s frames of each synthetic part.

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
    ap.add_argument(
        "--part",
        default="real8",
        choices=("real8", "real_mono", "comb", "stoch"),
        help="real8 = the eight-microphone frozen split (pred_clips/pred_frames/table); "
        "the others write one rps_dump-format npz per set",
    )
    ap.add_argument(
        "--out",
        default="",
        help="real8: the output directory (required). Otherwise the dump ROOT, "
        "default results/rps_dump, written as <out>/<set>/<name>.npz",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="frames per set (0 = all); the rps_dump-format parts only, for smokes",
    )
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

    from rps_dump import pad_stack
    from train_slot_real import build_model

    from experiments import rps_bench as rb
    from experiments import slot_real as sr
    from metrics._common import get_array

    net, parts = build_model(args, args.device)
    if args.best:
        missing = net.load_state_dict(torch.load(args.best, map_location=args.device), strict=False)
        print(f"loaded {args.best}: unexpected {missing.unexpected_keys}", flush=True)
    net.eval()
    kw = {"subgrid": True, "octave": bool(args.octave), "relocate": True}  # P1c: octave off

    if args.part != "real8":
        # The neural models' own format and their own frames, so a C1 arm joins
        # `results/rps_dump` and every reader of it without a second layout.
        name = "real" if args.part == "real_mono" else args.part
        if args.limit and args.part in ("comb", "stoch"):
            # A synthetic part is SYNTHESIZED on demand, so a smoke asks for the
            # frames it wants and never pays for the other 253.
            frames = rb.part(name, n=args.limit)
        else:
            frames = rb.part(name)
            if args.limit:
                frames = frames[: args.limit]
        d = Path(args.out or "results/rps_dump") / name
        d.mkdir(parents=True, exist_ok=True)
        preds, metrics = [], []
        with torch.no_grad():
            for f in frames:
                # `(1, N)`: one microphone with its channel axis kept, so the
                # decoder reads one item of one channel.
                au = torch.as_tensor(
                    np.asarray(f["mixture"].data, dtype=np.float32).ravel()[None],
                    device=args.device,
                )
                pred = net.decode(au, **kw)[0].cpu().numpy().astype(np.float64)
                gt = np.asarray(get_array(f, "rps"), dtype=np.float64)
                preds.append(pred.astype(np.float32))
                metrics.append(rb.pit_mae(pred, gt))
        pred, n_t = pad_stack(preds)
        metric = np.asarray(metrics, dtype=np.float64)
        np.savez(d / f"{args.name}.npz", pred=pred, n_t=n_t, metric=metric)
        print(
            f"wrote {d / args.name}.npz pred={pred.shape} n_t={n_t.shape} "
            f"metric={metric.shape}  mean={metric.mean():.4f} "
            f"median={np.median(metric):.4f}",
            flush=True,
        )
        return 0

    if not args.out:
        raise SystemExit("--part real8 needs --out")
    clips = sr.real_clips()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    preds = []
    rows = []
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
