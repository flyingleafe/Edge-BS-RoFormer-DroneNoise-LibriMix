"""Per-regime MAE + prediction-vs-GT overlay figures for the 3 full-flight
conditions (07-12 report), transformer arch only (budget: CPU inference,
one arch). Downloads the 3 R2 checkpoints to /tmp, runs CPU inference over
DREGON-LM-V4-michaels-valid-full (same valid set for all three, streamed via
dload:), computes per-regime MAE (rev/s, PIT-aligned via align_rps_to_gt),
and saves one representative pred/GT overlay per (condition, regime).

Never invents numbers: if a checkpoint or the dataset is unreachable, the
condition is skipped and reported as missing.
"""

import sys
from pathlib import Path

import boto3
import matplotlib
import numpy as np
import torch
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv(ROOT / ".env")

import os

from data_processing.collate import batch_size as fbatch_size
from data_processing.collate import frame_collate, slice_sample
from data_processing.frames import get_meta
from metrics._common import get_array
from tasks.rps_prediction import align_rps_to_gt
from training.config import build_dataset, build_task_and_codec, instantiate_model

CACHE = Path("/tmp/claude-1000/regime_eval_ckpts")
CACHE.mkdir(parents=True, exist_ok=True)

# 3 conditions x 3 archs = 9 checkpoints. R2 key naming is inconsistent
# across experiment batches (e5 uses "uni_gru128", e11/e12 use "unigru128") —
# preserved here as-is, verified to exist via `s3 list_objects_v2`.
ARCHS = [
    ("transformer", "Transformer", "simple_conv_v2_transformer"),
    ("uni_gru128", "Uni-GRU-128", "simple_conv_v2_uni_gru128"),
    ("scv2", "SimpleConv-v2", "simple_conv_v2"),
]

CONDITIONS = [
    (
        "baseline",
        "real-only, cruise-trained",
        {
            "transformer": "artifacts/e5_baseline_transformer/checkpoints/best.ckpt",
            "uni_gru128": "artifacts/e5_baseline_uni_gru128/checkpoints/best.ckpt",
            "scv2": "artifacts/e5_baseline_scv2/checkpoints/best.ckpt",
        },
    ),
    (
        "curriculum",
        "sim full-flight curriculum",
        {
            "transformer": "artifacts/e11_full_ft_warp_transformer/checkpoints/best.ckpt",
            "uni_gru128": "artifacts/e11_full_ft_warp_unigru128/checkpoints/best.ckpt",
            "scv2": "artifacts/e11_full_ft_warp_scv2/checkpoints/best.ckpt",
        },
    ),
    (
        "real_fullflight",
        "real full-flight (min_rps=0)",
        {
            "transformer": "artifacts/e12_real_fullflight_transformer/checkpoints/best.ckpt",
            "uni_gru128": "artifacts/e12_real_fullflight_unigru128/checkpoints/best.ckpt",
            "scv2": "artifacts/e12_real_fullflight_scv2/checkpoints/best.ckpt",
        },
    ),
]


def model_cfg(arch_name: str) -> dict:
    return {
        "task": "rps_prediction",
        "task_params": {"n_channels": None, "sr": [16000, 1], "frame_rate": [16000, 512]},
        "_target_": "models.registry.build_model",
        "params": {
            "name": arch_name,
            "n_fft": 2048,
            "hop_length": 512,
            "num_rotors": 4,
        },
    }


MODEL_CFG = model_cfg(
    "simple_conv_v2_transformer"
)  # used for codec/dataset build (front-end identical across archs)

DATA_CFG = {
    "_target_": "data_processing.frame_datasets.DregonLMFrameDataset",
    "params": {
        "data_dir": "dload:DREGON-LM-V4-michaels-valid-full",
        "n_fft": 2048,
        "hop_length": 512,
        "sample_rate": 16000,
        "flatten_channels": True,
    },
}


def download_ckpt(key: str) -> Path | None:
    dst = CACHE / key.replace("/", "_")
    if dst.exists():
        return dst
    acc = os.environ.get("R2_ACCOUNT_ID")
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not (acc and ak and sk):
        print(f"  MISSING R2 creds, cannot fetch {key}")
        return None
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{acc}.r2.cloudflarestorage.com",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
    )
    try:
        s3.download_file("ml-data", key, str(dst))
        return dst
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED to fetch {key}: {e}")
        return None


def regime_of(gt: np.ndarray) -> str:
    m = float(gt.mean())
    if m < 5:
        return "ground"
    if m < 45:
        return "warmup"
    return "cruise"


def main():
    out_marker = HERE / "assets" / "regime_mae_table.typ"
    if out_marker.exists() and not os.environ.get("FORCE_REGIME_EVAL"):
        print(
            f"skip: {out_marker} already exists (set FORCE_REGIME_EVAL=1 to rerun; "
            "this step streams ~300 val windows + 3x CPU inference, a few minutes)"
        )
        return
    device = torch.device("cpu")
    _task, codec = build_task_and_codec(MODEL_CFG)

    print("Loading validation set (dload:DREGON-LM-V4-michaels-valid-full)...")
    valid_ds = build_dataset(DATA_CFG)
    loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=frame_collate
    )

    # Cache GT once (same for every condition).
    all_gt = []
    all_meta = []
    all_channel = []
    for batch in loader:
        for i in range(fbatch_size(batch)):
            s = slice_sample(batch, i)
            all_gt.append(get_array(s, "rps"))
            all_meta.append(get_meta(s, "recording_id", None))
            all_channel.append(get_meta(s, "channel", None))
    print(f"  {len(all_gt)} validation clips loaded")
    regimes = [regime_of(g) for g in all_gt]
    from collections import Counter

    print("  regime counts:", Counter(regimes))

    # Pick one representative window per regime for the overlay figures: the
    # most STABLE window of that regime (lowest GT std), so e.g. "cruise" isn't
    # accidentally a window whose mean crosses the threshold mid-ramp.
    picked_clips = {}
    for reg in ("cruise", "warmup", "ground"):
        cand = [i for i, r in enumerate(regimes) if r == reg]
        if cand:
            picked_clips[reg] = min(cand, key=lambda i: float(all_gt[i].std()))

    # results[cond_key][arch_key] = {...}; also keep a transformer-only view
    # (`results["<cond_key>"]` legacy shape) for the overlay figures.
    results_by_arch: dict[str, dict[str, dict | None]] = {}

    for arch_key, arch_label, arch_model_name in ARCHS:
        mcfg = model_cfg(arch_model_name)
        results_by_arch[arch_key] = {}
        for cond_key, cond_label, r2keys in CONDITIONS:
            print(f"\n=== {cond_key} / {arch_key} ({cond_label}, {arch_label}) ===")
            ckpt_path = download_ckpt(r2keys[arch_key])
            if ckpt_path is None:
                results_by_arch[arch_key][cond_key] = None
                continue
            model = instantiate_model(mcfg).to(device)
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(sd)
            model.eval()

            preds = []
            with torch.no_grad():
                for batch in loader:
                    inputs = codec.to_inputs(batch)
                    outputs = codec.call_model(model, inputs)
                    pred_frame = codec.to_frame(outputs, batch)
                    for i in range(fbatch_size(pred_frame)):
                        p = slice_sample(pred_frame, i)
                        preds.append(get_array(p, "rps_pred"))

            assert len(preds) == len(all_gt), (len(preds), len(all_gt))

            mae_by_regime = {"cruise": [], "warmup": [], "ground": []}
            all_abs_err = []
            for p, g, reg in zip(preds, all_gt, regimes, strict=True):
                p_aligned = align_rps_to_gt(p, g)
                err = float(np.mean(np.abs(p_aligned - g)))
                mae_by_regime[reg].append(err)
                all_abs_err.append(err)

            results_by_arch[arch_key][cond_key] = {
                "mae": {r: (float(np.mean(v)) if v else None) for r, v in mae_by_regime.items()},
                "mae_all": float(np.mean(all_abs_err)),
                "n": {r: len(v) for r, v in mae_by_regime.items()},
                "preds": preds,
            }
            entry = results_by_arch[arch_key][cond_key]
            assert entry is not None
            for r in ("cruise", "warmup", "ground"):
                m = entry["mae"][r]
                print(
                    f"  {r}: MAE={m:.2f} rev/s (n={entry['n'][r]})"
                    if m is not None
                    else f"  {r}: n/a"
                )

    # transformer-only view, kept for the overlay figures below (unchanged shape)
    results = results_by_arch["transformer"]

    # ── overlay figures: one clip per regime, all available conditions (transformer) ──
    for reg, ci in picked_clips.items():
        gt = all_gt[ci]
        n_rows = sum(1 for k, _, _ in CONDITIONS if results.get(k))
        fig, axs = plt.subplots(n_rows, 1, figsize=(7.4, 1.55 * n_rows), sharex=True)
        if n_rows == 1:
            axs = [axs]
        row = 0
        for key, label, _ in CONDITIONS:
            rkey = results.get(key)
            if rkey is None:
                continue
            ax = axs[row]
            p = rkey["preds"][ci]
            p_aligned = align_rps_to_gt(p, gt)
            t = np.arange(gt.shape[1]) * 512 / 16000
            for r in range(gt.shape[0]):
                ax.plot(t, gt[r], ":", color="grey", lw=1.1, label="GT" if r == 0 else None)
                ax.plot(t, p_aligned[r], "-", lw=1.1, label="pred" if r == 0 else None)
            m = rkey["mae"][reg]
            ax.set_ylabel("rev/s", fontsize=8)
            ax.set_title(f"{label} — MAE {m:.1f} rev/s" if m is not None else label, fontsize=9)
            if row == 0:
                # Legend outside the axes (upper-right corner, above the plot)
                # to avoid colliding with the rotor-speed curves it used to
                # sit on top of.
                ax.legend(
                    frameon=False,
                    fontsize=7,
                    loc="lower left",
                    bbox_to_anchor=(1.0, 0.0),
                    borderaxespad=0.0,
                )
            row += 1
        axs[-1].set_xlabel("time (s)", fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        outp = HERE / "assets" / f"regime_overlay_{reg}.png"
        fig.savefig(outp, dpi=140)
        plt.close(fig)
        print("saved", outp)

    # ── full-flight overlay (all 3 conditions, transformer, whole valid set) ──
    # NOTE: `recording_id` here is a *per-window* id ("sample_00000", ...), not
    # a shared physical-recording id -- the axis that repeats across a window
    # is `channel` (8 mic copies of the *same* window, identical RPS). Grouping
    # by recording_id (as an earlier version of this script did) therefore
    # picked the 8 channel-copies of a single window and concatenated them as
    # if they were 8 consecutive time windows -- if that window happened to be
    # a ground (RPS=0) window, the "whole recording" plot was 8x the same
    # all-zero GT. Fixed: take one channel (0) and walk the dataset's natural
    # iteration order, which *is* time order (verified: RPS means step
    # ground->warmup->cruise consistently across consecutive indices).
    idxs = [i for i, c in enumerate(all_channel) if c == 0]
    if idxs:
        gt_cat = np.concatenate([all_gt[i] for i in idxs], axis=1)
        n_rows = sum(1 for k, _, _ in CONDITIONS if results.get(k))
        fig, axs = plt.subplots(n_rows, 1, figsize=(9.0, 1.5 * n_rows), sharex=True)
        if n_rows == 1:
            axs = [axs]
        row = 0
        for key, label, _ in CONDITIONS:
            rkey = results.get(key)
            if rkey is None:
                continue
            ax = axs[row]
            p_cat = np.concatenate(
                [align_rps_to_gt(rkey["preds"][i], all_gt[i]) for i in idxs], axis=1
            )
            t = np.arange(gt_cat.shape[1]) * 512 / 16000
            for r in range(gt_cat.shape[0]):
                ax.plot(t, gt_cat[r], ":", color="grey", lw=0.9, label="GT" if r == 0 else None)
                ax.plot(t, p_cat[r], "-", lw=0.9, label="pred" if r == 0 else None)
            ax.set_ylabel("rev/s", fontsize=8)
            ax.set_title(label, fontsize=9)
            if row == 0:
                ax.legend(
                    frameon=False,
                    fontsize=7,
                    loc="lower left",
                    bbox_to_anchor=(1.0, 0.0),
                    borderaxespad=0.0,
                )
            row += 1
        axs[-1].set_xlabel("time (s)", fontsize=8)
        fig.suptitle(
            f"Whole valid-full set, channel 0, time order ({len(idxs)} windows concatenated)",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 0.9, 0.95))
        outp = HERE / "assets" / "regime_overlay_fullflight.png"
        fig.savefig(outp, dpi=140)
        plt.close(fig)
        print("saved", outp)
    else:
        print("  no channel metadata available, skipping full-flight overlay")

    # ── MAE table (typst): TALL layout, one row per (recipe, regime), one
    # column per architecture -- easier to read at slide size than the old
    # wide 5-col x 9-row layout (round-4 critique: "table small, make vert
    # cols" -- fewer, wider columns; more, shorter rows; bigger font).
    arch_labels = [lab for _, lab, _ in ARCHS]
    align_str = "left, " + ", ".join(["right"] * len(ARCHS))
    lines = [
        "#figure(",
        "  table(",
        f"    columns: {1 + len(ARCHS)}, align: ({align_str}), stroke: 0.5pt,",
        "    table.header([Recipe / regime], " + ", ".join(f"[{a}]" for a in arch_labels) + "),",
    ]
    rlabel = {k: lab for k, lab, _ in CONDITIONS}
    for cond_key, _, _ in CONDITIONS:
        lines.append(
            f"    table.cell(colspan: {1 + len(ARCHS)})[" + f"#emph[{rlabel[cond_key]}]" + "],"
        )
        for reg, reg_label in (
            ("cruise", "Cruise"),
            ("warmup", "Warm-up"),
            ("ground", "Ground"),
            ("all", "All"),
        ):
            cells = []
            for arch_key, _, _ in ARCHS:
                r = results_by_arch[arch_key].get(cond_key)
                if r is None:
                    cells.append("n/a")
                elif reg == "all":
                    cells.append(f"{r['mae_all']:.1f}")
                else:
                    m = r["mae"][reg]
                    cells.append(f"{m:.1f}" if m is not None else "n/a")
            lines.append(f"    [{reg_label}], " + ", ".join(f"[{c}]" for c in cells) + ",")
    lines.append("  ),")
    ok = next(
        (r for a in results_by_arch.values() for r in a.values() if r is not None),
        None,
    )
    n = ok["n"] if ok else {}
    n_str = f"{n.get('cruise', '?')} cruise / {n.get('warmup', '?')} warm-up / {n.get('ground', '?')} ground"
    lines.append(
        "  caption: [Per-regime MAE (rev/s, PIT-aligned via "
        "`align_rps_to_gt`), all 3 architectures x 3 recipes, full-flight "
        f"validation set (michaels-valid-full, {n_str} STFT-frame windows; "
        'regime classified per-window by mean ground-truth RPS; "All" is '
        "the mean over all windows, not a regime average.],"
    )
    lines.append(")")
    (HERE / "assets" / "regime_mae_table.typ").write_text("\n".join(lines) + "\n")
    print("saved", HERE / "assets" / "regime_mae_table.typ")


if __name__ == "__main__":
    main()
