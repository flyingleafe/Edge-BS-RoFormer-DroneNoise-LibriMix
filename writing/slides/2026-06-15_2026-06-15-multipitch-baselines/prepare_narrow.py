#!/usr/bin/env python3
"""Append narrow-band + super-resolution salience figures to the multipitch slide deck.

Generates a parallel set of assets (suffixed ``_narrow_sr``) for the two new
checkpoints (``multif0_salience_narrow_sr``, ``basic_pitch_narrow_sr``), without
touching the existing assets. Mirrors the figure style of ``prepare.py``.
"""

# Imports follow os.chdir() into the project root (matches sibling prepare.py).
# ruff: noqa: E402

import json
import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tdseries as td
import torch

PROJECT_ROOT = pathlib.Path("../../../").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from typing import cast

from plots.rps_prediction.salience_comparison import (
    model_rps_prediction,
    model_salience_series,
    select_channel,
)
from plots.rps_prediction.sample_comparison import _load_sample
from plots.timeframe.registry import TrackContext
from plots.timeframe.renderers import make_spectrogram_series, render_salience
from tasks.rps_prediction import align_rps_to_gt
from train_rps_predictor import get_model

SLIDE_DIR = pathlib.Path(__file__).parent.resolve()
ASSETS = SLIDE_DIR / "assets"
ASSETS.mkdir(exist_ok=True)
RESULTS = PROJECT_ROOT / "results" / "dregon_v4_eval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MULTIF0_CFG = dict(
    n_octaves=1,
    over_sample=10,
    harmonics=[1, 2, 3, 4],
    superres_out=True,
    out_fmin=55.0,
    out_fmax=110.0,
    out_bins=360,
)
BP_CFG = dict(
    bp_fmin=55.0,
    bins_per_semitone=4,
    n_contour_semitones=12,
    superres_out=True,
    out_fmin=55.0,
    out_fmax=110.0,
    out_bins=360,
)

# ── 1. load metrics (regression + original salience + narrow-SR) ────────────
with open(RESULTS / "salience_baselines_final_valid.json") as f:
    salience_data = json.load(f)["results"]
with open(RESULTS / "simpleconv_8ch_v4_full_valid.json") as f:
    conv_data = json.load(f)["results"]
with open(RESULTS / "salience_narrow_sr_final_valid.json") as f:
    narrow_data = json.load(f)["results"]

MODELS = {
    "SimpleConvV2 (8ch)": {**conv_data["simple_conv_v2"], "color": "#ff7f0e"},
    "SimpleConv (8ch)": {**conv_data["simple_conv"], "color": "#1f77b4"},
    "multif0_salience": {**salience_data["multif0_salience"], "color": "#2ca02c"},
    "multif0_salience_fastest": {**salience_data["multif0_salience_fastest"], "color": "#d62728"},
    "basic_pitch_salience": {**salience_data["basic_pitch"], "color": "#9467bd"},
    "multif0_salience_narrow_sr": {**narrow_data["multif0_salience_narrow_sr"], "color": "#8c564b"},
    "basic_pitch_narrow_sr": {**narrow_data["basic_pitch_narrow_sr"], "color": "#e377c2"},
}

names = list(MODELS)
rmse_vals = [MODELS[n]["rmse"] for n in names]
r2_vals = [MODELS[n]["r2"] for n in names]
colors = [MODELS[n]["color"] for n in names]

# ── 2. leaderboard (all models incl. narrow-SR) ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
bars1 = ax1.bar(range(len(names)), rmse_vals, color=colors, width=0.6)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax1.set_ylabel("RMSE (Hz)", fontsize=11)
ax1.set_title("RPS prediction error", fontsize=13)
ax1.grid(axis="y", alpha=0.3)
for bar, val in zip(bars1, rmse_vals):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

bars2 = ax2.bar(range(len(names)), r2_vals, color=colors, width=0.6)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("R²", fontsize=11)
ax2.set_title("Coefficient of determination", fontsize=13)
ax2.grid(axis="y", alpha=0.3)
ax2.axhline(0, color="black", linewidth=0.5)
for bar, val in zip(bars2, r2_vals):
    offset = 0.3 if val >= 0 else -0.8
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        f"{val:.2f}",
        ha="center",
        va="bottom" if val >= 0 else "top",
        fontsize=8,
    )

plt.tight_layout()
fig.savefig(ASSETS / "leaderboard_metrics_narrow_sr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: leaderboard_metrics_narrow_sr.png")

# ── 3. per-rotor MAE (all salience models incl. narrow-SR) ───────────────────
sal = {
    "LateDeep": salience_data["multif0_salience"]["mae_per_rotor"],
    "LateDeep-fast": salience_data["multif0_salience_fastest"]["mae_per_rotor"],
    "Basic Pitch": salience_data["basic_pitch"]["mae_per_rotor"],
    "LateDeep narrow-SR": narrow_data["multif0_salience_narrow_sr"]["mae_per_rotor"],
    "Basic Pitch narrow-SR": narrow_data["basic_pitch_narrow_sr"]["mae_per_rotor"],
}
sal_colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
x = np.arange(4)
width = 0.16
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (label, vals) in enumerate(sal.items()):
    ax.bar(x + i * width, vals, width, label=label, color=sal_colors[i])
ax.set_xlabel("Rotor index", fontsize=11)
ax.set_ylabel("MAE (Hz)", fontsize=11)
ax.set_title("Per-rotor frame MAE", fontsize=13)
ax.set_xticks(x + 2 * width)
ax.set_xticklabels(["0", "1", "2", "3"])
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(ASSETS / "per_rotor_mae_narrow_sr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: per_rotor_mae_narrow_sr.png")

# ── 4. load narrow-SR salience models + SimpleConvV2 ─────────────────────────
print(f"Device: {DEVICE}")
narrow_models = {
    "multif0_salience_narrow_sr": get_model(
        "multif0_salience", hcqt_fmin=55.0, salience_cfg=MULTIF0_CFG
    ),
    "basic_pitch_narrow_sr": get_model("basic_pitch_salience", salience_cfg=BP_CFG),
}
ckpts = {
    "multif0_salience_narrow_sr": PROJECT_ROOT
    / "results/rps_baselines_v4/multif0_salience_narrow_sr/best_multif0_salience.pt",
    "basic_pitch_narrow_sr": PROJECT_ROOT
    / "results/rps_baselines_v4/basic_pitch_narrow_sr/best_basic_pitch_salience.pt",
}
for k, m in narrow_models.items():
    m.load_state_dict(torch.load(ckpts[k], map_location=DEVICE, weights_only=True), strict=True)
    m.eval().to(DEVICE)
print("Loaded narrow-SR models")

# ── 5. per-model 3-pane figures (spectrogram + salience + RPS) ───────────────
SAMPLE_ID = "sample_00026"
CHANNEL = 0
DATASET = PROJECT_ROOT / "datasets" / "DREGON-LM-V4" / "valid"
SALIENCE_VMAX = "auto"

sample = _load_sample(str(DATASET / SAMPLE_ID))
mono = select_channel(cast(td.Series, sample["audio"]), CHANNEL)
gt_rps_series = cast(td.Series, sample["rps"])
gt_rps_data = np.asarray(gt_rps_series.data, dtype=np.float32)
gt_rps_timestamps = np.asarray(cast(td.StampIndex, gt_rps_series.tindex).abs_stamps)

spec_track = make_spectrogram_series(mono, fmax=4000)
S = np.asarray(spec_track.series.data)
t_spec = cast(td.GridIndex, spec_track.series.tindex).sample_times()
freqs = np.linspace(0, 4000, S.shape[0])
colors_rps = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for model_name, model in narrow_models.items():
    print(f"3-pane: {model_name}")
    salience_track = model_salience_series(model, mono, device=DEVICE)
    pred = model_rps_prediction(model, mono, device=DEVICE, track_threshold=0.3)
    if pred.shape[1] != gt_rps_data.shape[1]:
        from scipy import signal

        resampled = np.zeros((4, gt_rps_data.shape[1]))
        for i in range(4):
            resampled[i] = cast(np.ndarray, signal.resample(pred[i], gt_rps_data.shape[1]))
        pred = resampled
    pred = align_rps_to_gt(pred, gt_rps_data)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1]}
    )
    ax = axes[0]
    ax.pcolormesh(t_spec, freqs, S, shading="gouraud", cmap="magma")
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title(f"Audio spectrogram — {SAMPLE_ID} (channel {CHANNEL})", fontsize=12)
    ax.set_ylim(0, 4000)

    ax = axes[1]
    t_start = salience_track.series.t_start
    t_end = t_start + salience_track.series.duration
    context = TrackContext(
        ax=ax,
        name=f"{model_name} salience",
        t_start=t_start,
        t_end=t_end,
        style={
            "salience_vmax": SALIENCE_VMAX,
            "salience_colorbar": False,
            "_frame": sample,
            "_hints": salience_track.hints,
        },
    )
    render_salience(salience_track.series, context)
    ax.set_title(f"{model_name} salience", fontsize=12)

    ax = axes[2]
    for i in range(4):
        ax.plot(
            gt_rps_timestamps,
            gt_rps_data[i],
            ":",
            color=colors_rps[i],
            linewidth=1.5,
            label=f"GT R{i + 1}",
        )
        ax.plot(
            gt_rps_timestamps,
            pred[i],
            "-",
            color=colors_rps[i],
            linewidth=1.5,
            label=f"pred R{i + 1}",
        )
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("RPS", fontsize=11)
    ax.set_title(f"RPS — {model_name} (solid) vs GT (dotted)", fontsize=12)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / f"{SAMPLE_ID}_{model_name}_3pane.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {SAMPLE_ID}_{model_name}_3pane.pdf")

# ── 6. SimpleConvV2 vs each narrow-SR model ─────────────────────────────────
from models.rps_predictor import SimpleConvV2

simpleconv_v2 = SimpleConvV2(num_rotors=4)
simpleconv_v2.load_state_dict(
    torch.load(
        PROJECT_ROOT / "results/rps_8ch_v4_simple_conv_v2/best_simple_conv_v2.pt",
        map_location=DEVICE,
    ),
    strict=True,
)
simpleconv_v2.eval().to(DEVICE)

wav = torch.as_tensor(np.asarray(mono.data, dtype=np.float32), device=DEVICE).unsqueeze(0)
with torch.no_grad():
    pred_v2 = simpleconv_v2(wav)[0].cpu().numpy()
if pred_v2.shape[1] != gt_rps_data.shape[1]:
    from scipy import signal

    resampled_v2 = np.zeros((4, gt_rps_data.shape[1]))
    for i in range(4):
        resampled_v2[i] = cast(np.ndarray, signal.resample(pred_v2[i], gt_rps_data.shape[1]))
    pred_v2 = resampled_v2
pred_v2 = align_rps_to_gt(pred_v2, gt_rps_data)

t = gt_rps_timestamps
for model_name, model in narrow_models.items():
    print(f"compare: SimpleConvV2 vs {model_name}")
    pred = model_rps_prediction(model, mono, device=DEVICE, track_threshold=0.3)
    if pred.shape[1] != gt_rps_data.shape[1]:
        from scipy import signal

        resampled = np.zeros((4, gt_rps_data.shape[1]))
        for i in range(4):
            resampled[i] = cast(np.ndarray, signal.resample(pred[i], gt_rps_data.shape[1]))
        pred = resampled
    pred = align_rps_to_gt(pred, gt_rps_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for i in range(4):
        ax1.plot(t, gt_rps_data[i], ":", color=colors_rps[i], linewidth=1.5)
        ax1.plot(t, pred_v2[i], "-", color=colors_rps[i], linewidth=1.5)
    ax1.set_ylabel("RPS", fontsize=11)
    ax1.set_title("SimpleConvV2 (8ch)", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(["GT", "pred"], loc="upper right", fontsize=8)
    for i in range(4):
        ax2.plot(t, gt_rps_data[i], ":", color=colors_rps[i], linewidth=1.5)
        ax2.plot(t, pred[i], "-", color=colors_rps[i], linewidth=1.5)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("RPS", fontsize=11)
    ax2.set_title(model_name, fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(["GT", "pred"], loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(
        ASSETS / f"{SAMPLE_ID}_compare_simpleconvv2_vs_{model_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved comparison")

print("\nAll narrow-SR assets generated.")
