"""Evaluate the conditional RPS refiner (``ckla_refiner``) on the 37-clip
VK-parity protocol.

Loads a trained ``simple_conv_v2_ckla_phaseonly_cond`` checkpoint and a
*coarse* track source, feeds each 8 s protocol clip as
``refiner(audio, coarse)``, and reports per-clip PIT-aligned MAE (rev/s)
BEFORE (the coarse track itself) and AFTER refinement, plus the pooled cruise
numbers of ``scripts/rps_predictor_vk_eval.py`` (same clips, same
``align_rps_to_gt`` alignment, same pools).

Coarse sources (``--coarse``):

* ``neural:<model_key>`` — a model from ``rps_predictor_vk_eval.MODELS``
  (e.g. ``e12_transformer_best``); its independent per-clip ch0 prediction
  (the baseline ``none`` arm) is the conditioning.
* ``npz:<path>`` — a ``results/vk_blind_sweep/`` run NPZ (or a directory of
  them, one per recording): the chosen stage trajectory (``--stage``,
  default the last = refined stage) is interpolated from the sweep's
  ``seg_lo + ft`` recording-time grid onto each clip's frame grid
  (``start_s + i·0.032``; clips not fully covered by the sweep window are
  skipped). Rotor order of a VK track is arbitrary — fine as conditioning;
  the before/after metric is PIT-aligned per clip.

``--rounds N`` feeds the refiner's output back as conditioning N times
(default 1).

Run::

    python scripts/rps_refiner_eval.py --ckpt results/ckla_refiner/best.ckpt \
        --coarse neural:e12_transformer_best
    python scripts/rps_refiner_eval.py --ckpt r2://.../best.ckpt \
        --coarse npz:results/vk_blind_sweep --stage last --rounds 2

Outputs (``results/rps_refiner_eval/`` unless ``--out``): ``per_clip.csv``,
``report.json``, and a pooled table on stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
# Same src pin as rps_predictor_vk_eval (editable install may point elsewhere
# on omnirun worktrees); also pin scripts/ so the protocol module imports.
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import rps_predictor_vk_eval as vke  # noqa: E402  (protocol: CLIPS/POOLS/loaders)

SR = vke.SR
HOP = vke.HOP
FRAME_S = HOP / SR
CLIP_FRAMES = vke.CLIP_FRAMES
N_ROTORS = vke.N_ROTORS

#: vk_blind_sweep recording ids -> the valid-set recording ids of CLIPS.
_NPZ_RID_MAP = {"FLY124-cruise": "michaels_FLY124"}


def _npz_recording_id(path: Path) -> str:
    """Recording id from a ``<rid>[@Ns]__<ladder>__<arm>[...].npz`` filename."""
    head = path.name.split("__", 1)[0]
    head = re.sub(r"@[0-9.]+s$", "", head)
    return _NPZ_RID_MAP.get(head, head)


def load_npz_coarse(spec_path: str, stage: str) -> dict[str, np.ndarray]:
    """clip_id -> (4, 251) coarse track from vk_blind_sweep NPZ file(s).

    ``stage`` is a ``stage_labels`` entry, or ``"last"`` (the final/refined
    stage). Interpolation grid: the sweep stores the trajectory on
    ``seg_lo + ft`` in recording time (FRAME_HOP_S = 0.032 s — the same hop
    as the clip grid); each protocol clip covers ``start_s + i*0.032``.
    Clips not fully inside the sweep window are skipped.
    """
    p = Path(spec_path)
    files = sorted(p.glob("*.npz")) if p.is_dir() else [p]
    files = [f for f in files if "__" in f.name]
    if not files:
        raise SystemExit(f"no vk_blind_sweep NPZ files at {spec_path}")

    coarse: dict[str, np.ndarray] = {}
    for f in files:
        rid = _npz_recording_id(f)
        clips = [c for c in vke.CLIPS if c[1] == rid]
        if not clips:
            print(f"[npz] {f.name}: recording {rid!r} not in the 37-clip protocol; skipping")
            continue
        with np.load(f, allow_pickle=False) as z:
            if "stage_snaps" not in z:
                print(f"[npz] {f.name}: no stage_snaps (not a run NPZ); skipping")
                continue
            labels = [str(v) for v in z["stage_labels"]]
            snaps = np.asarray(z["stage_snaps"], dtype=np.float64)  # (S, 4, F)
            ft = np.asarray(z["ft"], dtype=np.float64)
            seg_lo = float(np.asarray(z["seg_bounds"])[0])
        if stage == "last":
            idx = len(labels) - 1
        elif stage in labels:
            idx = labels.index(stage)
        else:
            raise SystemExit(f"{f.name}: stage {stage!r} not in {labels}")
        traj = snaps[idx]  # (4, F) on seg_lo + ft
        t_abs = seg_lo + ft
        n_kept = 0
        for c in clips:
            cid, start_s = c[0], float(c[2])
            t_clip = start_s + np.arange(CLIP_FRAMES) * FRAME_S
            if t_clip[0] < t_abs[0] - FRAME_S / 2 or t_clip[-1] > t_abs[-1] + FRAME_S / 2:
                continue  # clip not covered by the sweep window
            coarse[cid] = np.stack(
                [np.interp(t_clip, t_abs, traj[r]) for r in range(N_ROTORS)]
            ).astype(np.float32)
            n_kept += 1
        print(f"[npz] {f.name}: stage {labels[idx]!r} -> {n_kept}/{len(clips)} clips of {rid}")
    if not coarse:
        raise SystemExit("npz coarse source covered no protocol clips")
    return coarse


def load_neural_coarse(
    model_key: str, audio: dict[str, np.ndarray], device: str, batch: int
) -> dict[str, np.ndarray]:
    """clip_id -> (4, 251): independent per-clip ch0 predictions (none arm)."""
    if model_key not in vke.MODELS:
        raise SystemExit(f"unknown neural coarse model {model_key!r}; one of {sorted(vke.MODELS)}")
    experiment, ckpt_uri, _ = vke.MODELS[model_key]
    model = vke.load_model(experiment, ckpt_uri, device)
    ids = [c[0] for c in vke.CLIPS if c[0] in audio]
    wins = np.stack([audio[cid][0] for cid in ids])
    preds = vke.batched_forward(model, wins, device, batch, CLIP_FRAMES)
    del model
    return {cid: preds[k] for k, cid in enumerate(ids)}


def refine_clips(
    refiner: Any,
    audio: dict[str, np.ndarray],
    coarse: dict[str, np.ndarray],
    *,
    rounds: int,
    device: str,
    batch: int,
) -> dict[int, dict[str, np.ndarray]]:
    """round (1-based) -> clip_id -> (4, 251) refined track (ch0 audio)."""
    import torch

    ids = sorted(coarse)
    wins = torch.from_numpy(np.stack([audio[cid][0] for cid in ids])).to(device)
    cond = torch.from_numpy(np.stack([coarse[cid] for cid in ids])).to(device)
    out: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for r in range(1, rounds + 1):
            refined_batches = []
            for i in range(0, len(ids), batch):
                refined_batches.append(refiner(wins[i : i + batch], cond[i : i + batch]))
            cond = torch.cat(refined_batches, dim=0)
            arr = cond.float().cpu().numpy()
            out[r] = {cid: arr[k] for k, cid in enumerate(ids)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--ckpt", required=True, help="refiner checkpoint (path or r2:// URI)")
    ap.add_argument(
        "--experiment",
        default="ckla_refiner",
        help="experiment config that builds the refiner model",
    )
    ap.add_argument(
        "--coarse",
        required=True,
        help="coarse track source: neural:<model_key> or npz:<file-or-dir>",
    )
    ap.add_argument(
        "--stage",
        default="last",
        help="vk_blind_sweep stage label for npz: sources (default: last stage)",
    )
    ap.add_argument("--rounds", type=int, default=1, help="iterative refinement rounds")
    ap.add_argument("--data", default=None, help="dataset dir or dload: URI")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results/rps_refiner_eval")
    args = ap.parse_args()

    import torch

    from tasks.rps_prediction import align_rps_to_gt

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data
    if data_path is None:
        local = _ROOT / "datasets" / "DREGON-LM-V4-michaels-full" / "valid"
        data_path = str(local) if local.is_dir() else "dload:DREGON-LM-V4-michaels-valid-full"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[refiner_eval] data: {data_path} | device: {device}", flush=True)

    audio, gt = vke.load_clip_data(data_path)

    kind, _, src = args.coarse.partition(":")
    if kind == "neural" and src:
        coarse = load_neural_coarse(src, audio, device, args.batch)
    elif kind == "npz" and src:
        coarse = load_npz_coarse(src, args.stage)
    else:
        raise SystemExit(f"--coarse must be neural:<model_key> or npz:<path>, got {args.coarse!r}")

    refiner = vke.load_model(args.experiment, args.ckpt, device)
    refined = refine_clips(
        refiner, audio, coarse, rounds=max(1, args.rounds), device=device, batch=args.batch
    )

    clip_info = {c[0]: c for c in vke.CLIPS}
    rows: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, float]] = {}
    for cid in sorted(coarse):
        g = gt[cid]
        info = clip_info[cid]
        arms: dict[str, np.ndarray] = {"coarse": coarse[cid]}
        for r, tracks in refined.items():
            arms[f"refined_r{r}"] = tracks[cid]
        for arm, track in arms.items():
            d = align_rps_to_gt(track, g) - g
            mae = float(np.mean(np.abs(d)))
            rows.append(
                {
                    "arm": arm,
                    "clip": cid,
                    "recording": info[1],
                    "regime": info[3],
                    "regime_mean_rps": info[4],
                    "mae": mae,
                    "mse": float(np.mean(d**2)),
                }
            )
            by_arm.setdefault(arm, {})[cid] = mae

    pooled: dict[str, dict[str, float]] = {}
    for arm, by_clip in by_arm.items():
        pooled[arm] = {}
        for pool_name, sel in vke.POOLS.items():
            vals = [by_clip[c[0]] for c in vke.CLIPS if c[0] in by_clip and sel(c)]
            if vals:
                pooled[arm][pool_name] = float(np.mean(vals))

    arm_order = ["coarse"] + [f"refined_r{r}" for r in sorted(refined)]
    header = f"{'arm':<14}" + "".join(f"{p:>15}" for p in vke.POOLS) + f"{'n_clips':>10}"
    lines = [
        f"Refiner {args.experiment} ({args.ckpt}) | coarse = {args.coarse} "
        f"| rounds = {max(1, args.rounds)}",
        "Pooled PIT-MAE (rev/s), mean of per-clip MAE — before (coarse) vs after (refined)",
        header,
        "-" * len(header),
    ]
    for arm in arm_order:
        cells = pooled.get(arm, {})
        row = f"{arm:<14}"
        for p in vke.POOLS:
            row += f"{cells.get(p, float('nan')):>15.3f}"
        row += f"{len(by_arm.get(arm, {})):>10d}"
        lines.append(row)
    table = "\n".join(lines)
    print("\n" + table, flush=True)

    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "report.json", "w") as f:
        json.dump(
            {
                "protocol": "37-clip per-clip PIT-MAE (align_rps_to_gt once per clip)",
                "experiment": args.experiment,
                "checkpoint": args.ckpt,
                "coarse": args.coarse,
                "stage": args.stage,
                "rounds": max(1, args.rounds),
                "data": data_path,
                "n_clips_covered": len(coarse),
                "pooled": pooled,
            },
            f,
            indent=2,
        )
    print(f"\n[refiner_eval] wrote {out_dir}/report.json, per_clip.csv", flush=True)


if __name__ == "__main__":
    main()
