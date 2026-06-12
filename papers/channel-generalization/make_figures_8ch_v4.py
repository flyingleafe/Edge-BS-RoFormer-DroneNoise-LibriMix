#!/usr/bin/env python3
"""Generate figures for the 8ch-trained model evaluation (no conclusions)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))  # project root

from tasks.rps_prediction import (
    HOP,
    N_FFT,
    SR_AUDIO,
    load_input_set,
    load_predictor,
)
from train_rps_predictor import _ROTOR_PERMS, pit_mse_loss
from utils.plots.rps_prediction.sample_comparison import plot_sample_comparison

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parents[2]
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATASET = PROJECT / "datasets" / "DREGON-LM-V4" / "valid"
EVAL_NO_PIT = PROJECT / "results" / "dregon_v4_eval" / "eval_8ch_v4.json"
EVAL_PIT = PROJECT / "results" / "dregon_v4_eval" / "eval_8ch_v4_pit.json"

MODELS = [
    ("SimpleConv (8ch)", "simple_conv@results/rps_8ch_v4_simple_conv/best_simple_conv.pt"),
    (
        "SimpleConvV2 (8ch)",
        "simple_conv_v2@results/rps_8ch_v4_simple_conv_v2/best_simple_conv_v2.pt",
    ),
]

RECORDINGS = [
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
]

RECORDING_LABELS = {
    "free-flight_nosource_room1": "nosource",
    "free-flight_speech-low_room1": "speech-low",
    "free-flight_whitenoise-low_room1": "whitenoise-low",
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. MSE barplots (regular and PIT)
# ═══════════════════════════════════════════════════════════════════════════


def _load_eval(path: Path):
    with open(path) as f:
        data = json.load(f)
    return data["per_sample"]


def _agg_mse(rows: list[dict]) -> dict[tuple[str, int], float]:
    from collections import defaultdict

    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["recording_id"], r["channel"])
        groups[key].append(r["mse"])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def plot_mse_bars(eval_path: Path, out_name: str, title_suffix: str = ""):
    all_rows = _load_eval(eval_path)
    fig, axes = plt.subplots(2, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(f"Per-channel MSE by recording and model (8ch-trained){title_suffix}", fontsize=14)

    for mi, (mname, _) in enumerate(MODELS):
        mse_by_key = _agg_mse(all_rows[mi])
        for ri, rec in enumerate(RECORDINGS):
            ax = axes[mi, ri]
            ch_mses = [mse_by_key.get((rec, ch), 0.0) for ch in range(8)]
            colors = ["#2ca02c" if ch == 0 else "#d62728" for ch in range(8)]
            ax.bar(range(8), ch_mses, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(8))
            ax.set_xlabel("Channel")
            if ri == 0:
                ax.set_ylabel(f"{mname}\nMSE")
            if mi == 0:
                ax.set_title(RECORDING_LABELS[rec])
            ax.grid(axis="y", alpha=0.3)
            worst_ch = int(np.argmax(ch_mses))
            worst_val = ch_mses[worst_ch]
            ax.annotate(
                f"ch{worst_ch}\n{worst_val:.1f}",
                xy=(worst_ch, worst_val),
                xytext=(worst_ch, worst_val + max(ch_mses) * 0.05),
                ha="center",
                fontsize=7,
                color="red",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sample comparison figures (PIT-permuted)
# ═══════════════════════════════════════════════════════════════════════════


def _invert_perm(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[j] = i
    return inv


def _get_pit_perm(pred: np.ndarray, gt: np.ndarray) -> list[int]:
    p_t = torch.from_numpy(np.asarray(pred, dtype=np.float32)).unsqueeze(0)
    g_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0)
    _, best_idx = pit_mse_loss(p_t, g_t, perms=_ROTOR_PERMS, return_indices=True)
    best_perm = _ROTOR_PERMS[best_idx[0]].tolist()
    return best_perm


def _permute_pred_for_plot(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    best_perm = _get_pit_perm(pred, gt)
    inv_perm = _invert_perm(best_perm)
    return pred[inv_perm]


def _predict_all_channels(model, sample) -> dict[int, np.ndarray]:
    audio = np.asarray(sample["audio"].samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    preds = {}
    for ch in range(audio.shape[0]):
        pred = model.predict(audio[ch], sr=SR_AUDIO)
        preds[ch] = pred
    return preds


def _get_gt_on_frame_grid(sample):
    rps_es = sample["rps"]
    audio = np.asarray(sample["audio"].samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    dur = audio.shape[1] / SR_AUDIO
    n_frames = audio.shape[1] // HOP + 1
    frame_times = np.arange(n_frames) * HOP / SR_AUDIO + rps_es.t_start + N_FFT / SR_AUDIO / 2
    gt = rps_es.interpolate(frame_times)
    return gt, frame_times


def make_sample_comparison_figure(
    sample_id: str,
    sample_path: Path,
    model_name: str,
    model_spec: str,
    out_name: str,
):
    abs_model_spec = model_spec
    if "@" in model_spec:
        typ, path = model_spec.split("@", 1)
        abs_path = str(PROJECT / path)
        abs_model_spec = f"{typ}@{abs_path}"
    predictor = load_predictor(abs_model_spec)

    sample = next(s for s in load_input_set(str(sample_path)) if s.tags.get("id") == sample_id)
    preds_by_ch = _predict_all_channels(predictor, sample)
    gt, frame_times = _get_gt_on_frame_grid(sample)

    preds_dict = {}
    for ch in sorted(preds_by_ch.keys()):
        pred = preds_by_ch[ch]
        F = min(pred.shape[1], gt.shape[1])
        pred = pred[:, :F]
        gt_ch = gt[:, :F]
        pred_aligned = _permute_pred_for_plot(pred, gt_ch)
        preds_dict[f"ch{ch}"] = pred_aligned

    fig = plot_sample_comparison(
        sample=sample,
        channel="all",
        preds=preds_dict,
        figsize=(20, 28),
        show_separate_gt=True,
    )

    fig.suptitle(
        f"{model_name} — {sample_id} ({sample.tags.get('recording_id', '')})\n"
        "Predictions are PIT-permuted per channel",
        fontsize=14,
        y=1.01,
    )

    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 1. Barplots ===")
    plot_mse_bars(EVAL_NO_PIT, "mse_bars_8ch_v4.png")
    plot_mse_bars(EVAL_PIT, "mse_bars_8ch_v4_pit.png", " (PIT)")

    print("\n=== 2. Sample comparison figures ===")
    make_sample_comparison_figure(
        "sample_00014",
        DATASET,
        "SimpleConv (8ch)",
        MODELS[0][1],
        "sample_nosource_8ch_v4_simpleconv.png",
    )
    make_sample_comparison_figure(
        "sample_00002",
        DATASET,
        "SimpleConv (8ch)",
        MODELS[0][1],
        "sample_speech_8ch_v4_simpleconv.png",
    )
    make_sample_comparison_figure(
        "sample_00014",
        DATASET,
        "SimpleConvV2 (8ch)",
        MODELS[1][1],
        "sample_nosource_8ch_v4_simpleconv_v2.png",
    )
    make_sample_comparison_figure(
        "sample_00002",
        DATASET,
        "SimpleConvV2 (8ch)",
        MODELS[1][1],
        "sample_speech_8ch_v4_simpleconv_v2.png",
    )

    print("\nAll figures generated.")
