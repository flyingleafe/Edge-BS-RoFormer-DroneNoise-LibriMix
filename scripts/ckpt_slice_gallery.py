"""Render per-slice RPS predictions for two checkpoints of one experiment.

Built to answer one question visually: a synthetic-only model's `last`
checkpoint fits the synthetic distribution far better than its `best`
checkpoint (8.63 -> 3.70 all-MAE) while doing much worse on real recordings
(7.40 -> 14.19). WHERE does each happen? Aggregate numbers cannot say.

Emits one PNG per slice showing ground truth against both checkpoints'
predictions, plus a JSON of per-slice PIT MAE. Predictions are PIT-aligned to
the ground truth with `losses.pit.align_rps_to_gt`, the same matching the
metrics use — without it rotors appear swapped even when a prediction is good.

    python scripts/ckpt_slice_gallery.py --exp stoch_s1id_scv2 --out-dir /tmp/gal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from valid_regime_eval import VALID, clip_rigs, frame_regimes, pit_abs_error  # noqa: E402

CKPTS = ("best", "last")


def slice_regime(target: np.ndarray) -> str:
    """One label for a whole slice: the regime holding most of its frames."""
    labels = frame_regimes(target)
    vals, counts = np.unique(labels.astype(str), return_counts=True)
    return str(vals[int(np.argmax(counts))])


def real_slices(want_per_cell: int = 1) -> list[dict]:
    """A spread of real clips: every rig x regime cell that the split contains."""
    from data_processing.frame_datasets import DregonLMFrameDataset

    rigs = clip_rigs()
    ds = DregonLMFrameDataset(
        data_dir=VALID, n_fft=2048, hop_length=512, sample_rate=16000, channel=0
    )
    picked: dict[tuple[str, str], list[dict]] = {}
    for i in range(len(ds)):
        frame = ds[i]
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        cell = (rigs[i] if i < len(rigs) else "dregon", slice_regime(target))
        rows = picked.setdefault(cell, [])
        if len(rows) < want_per_cell:
            rows.append({"frame": frame, "target": target, "rig": cell[0], "regime": cell[1], "idx": i})
    out = []
    for cell in sorted(picked):
        out.extend(picked[cell])
    return out


def synth_slices(policy: str, base_seed: int, n_per_regime: int = 3) -> list[dict]:
    """A spread of held-out synthetic clips, one group per regime."""
    from omegaconf import OmegaConf

    from data_processing.frame_datasets import OnlineMixFrameDataset

    cfg = OmegaConf.load(policy)
    cfg.base_seed = int(base_seed)
    cfg.duration_s = 8.0
    for stage in cfg.policy.stages:
        for key in ("augmentations", "noise_augmentations", "noise_time_warp"):
            if key in stage:
                del stage[key]
    stream = OnlineMixFrameDataset.from_config(cfg, flatten_channels=True)

    picked: dict[str, list[dict]] = {}
    seen = 0
    for frame in stream:
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        regime = slice_regime(target)
        rows = picked.setdefault(regime, [])
        if len(rows) < n_per_regime:
            rows.append({"frame": frame, "target": target, "rig": "synthetic", "regime": regime, "idx": seen})
        seen += 1
        if seen > 400 or sum(len(v) for v in picked.values()) >= n_per_regime * 3:
            break
    out = []
    for regime in ("zero", "low", "flight"):
        out.extend(picked.get(regime, []))
    return out


def render(rows: list[dict], models: dict, out_dir: Path, tag: str) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from losses.pit import align_rps_to_gt

    meta = []
    for k, row in enumerate(rows):
        target = row["target"]
        fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=120)
        hop_s = 512 / 16000.0
        t = np.arange(target.shape[1]) * hop_s
        for r in range(target.shape[0]):
            ax.plot(t, target[r], color="#111", lw=2.0, alpha=0.85,
                    label="ground truth" if r == 0 else None, zorder=3)
        colors = {"best": "#1f77b4", "last": "#d62728"}
        maes = {}
        for name, model in models.items():
            pred = np.asarray(model(row["frame"])["rps_pred"].data, dtype=np.float64)
            width = min(pred.shape[1], target.shape[1])
            err = pit_abs_error(pred[:, :width], target[:, :width])
            maes[name] = float(err.mean())
            aligned = align_rps_to_gt(pred[:, :width], target[:, :width])
            tp = np.arange(width) * hop_s
            for r in range(aligned.shape[0]):
                ax.plot(tp, aligned[r], color=colors[name], lw=1.1, alpha=0.8,
                        label=f"{name} (MAE {maes[name]:.1f})" if r == 0 else None, zorder=4)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("rev/s", fontsize=8)
        ax.set_title(f"{row['rig']} · {row['regime']}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        name = f"{tag}_{k:02d}_{row['rig']}_{row['regime']}.png"
        fig.savefig(out_dir / name, bbox_inches="tight")
        plt.close(fig)
        meta.append({"file": name, "rig": row["rig"], "regime": row["regime"],
                     "idx": row["idx"], "mae": maes})
        print(f"  {name}  " + "  ".join(f"{n}={v:.2f}" for n, v in maes.items()), flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="stoch_s1id_scv2")
    ap.add_argument("--policy", default="conf/online_mix/stoch_s1id_dload.yaml")
    ap.add_argument("--base-seed", type=int, default=99000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import zoo

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {c: zoo.load(args.exp, ckpt=c, device="cpu") for c in CKPTS}

    print("real slices:", flush=True)
    meta_real = render(real_slices(want_per_cell=2), models, out_dir, "real")
    print("synthetic slices:", flush=True)
    meta_syn = render(synth_slices(args.policy, args.base_seed), models, out_dir, "synth")

    (out_dir / "meta.json").write_text(json.dumps(
        {"experiment": args.exp, "real": meta_real, "synthetic": meta_syn}, indent=1))
    print(f"wrote {out_dir/'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
