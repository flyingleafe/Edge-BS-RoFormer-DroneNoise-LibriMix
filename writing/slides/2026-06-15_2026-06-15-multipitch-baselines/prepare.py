#!/usr/bin/env python3
"""Generate figures and tables for the multipitch baselines slide deck.

Regenerates all plots from scratch using matplotlib + JSON metrics.
Copies clean paper illustrations from bibliography/.
Generates per-model salience+RPS plots via project inference code.
"""

import json
import os
import pathlib
import shutil
import sys
from typing import cast

# ── bootstrap project paths ─────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path("../../../").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tdseries as td
import torch

from models.salience_rps import BasicPitchSalience, LateDeepSalience
from plots.rps_prediction.salience_comparison import (
    model_rps_prediction,
    model_salience_series,
    plot_salience_comparison,
    select_channel,
)
from plots.rps_prediction.sample_comparison import _load_sample
from tasks.rps_prediction import align_rps_to_gt

# ── paths ───────────────────────────────────────────────────────────────
SLIDE_DIR = pathlib.Path(__file__).parent.resolve()
ASSETS = SLIDE_DIR / "assets"
ASSETS.mkdir(exist_ok=True)
BIB = PROJECT_ROOT / "bibliography"
RESULTS = PROJECT_ROOT / "results" / "dregon_v4_eval"

# ── 1. copy clean paper illustrations ──────────────────────────────────
for src_name in [
    "basic-pitch-illustration.png",
    "multif0-illustration-1.png",
    "multif0-illustration-2.png",
]:
    src = BIB / src_name
    dst = ASSETS / src_name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")
    else:
        print(f"MISSING: {src}")

# ── 2. load metrics ─────────────────────────────────────────────────────
with open(RESULTS / "salience_baselines_final_valid.json") as f:
    salience_data = json.load(f)
with open(RESULTS / "simpleconv_8ch_v4_full_valid.json") as f:
    conv_data = json.load(f)

# unified model order
MODELS = {
    "SimpleConvV2 (8ch)": {
        "rmse": conv_data["results"]["simple_conv_v2"]["rmse"],
        "r2": conv_data["results"]["simple_conv_v2"]["r2"],
        "mae_frame": conv_data["results"]["simple_conv_v2"]["mae_frame"],
        "mae_clip": conv_data["results"]["simple_conv_v2"]["mae_clip"],
        "color": "#ff7f0e",
    },
    "SimpleConv (8ch)": {
        "rmse": conv_data["results"]["simple_conv"]["rmse"],
        "r2": conv_data["results"]["simple_conv"]["r2"],
        "mae_frame": conv_data["results"]["simple_conv"]["mae_frame"],
        "mae_clip": conv_data["results"]["simple_conv"]["mae_clip"],
        "color": "#1f77b4",
    },
    "multif0_salience": {
        "rmse": salience_data["results"]["multif0_salience"]["rmse"],
        "r2": salience_data["results"]["multif0_salience"]["r2"],
        "mae_frame": salience_data["results"]["multif0_salience"]["mae_frame"],
        "mae_clip": salience_data["results"]["multif0_salience"]["mae_clip"],
        "color": "#2ca02c",
    },
    "multif0_salience_fastest": {
        "rmse": salience_data["results"]["multif0_salience_fastest"]["rmse"],
        "r2": salience_data["results"]["multif0_salience_fastest"]["r2"],
        "mae_frame": salience_data["results"]["multif0_salience_fastest"]["mae_frame"],
        "mae_clip": salience_data["results"]["multif0_salience_fastest"]["mae_clip"],
        "color": "#d62728",
    },
    "basic_pitch_salience": {
        "rmse": salience_data["results"]["basic_pitch"]["rmse"],
        "r2": salience_data["results"]["basic_pitch"]["r2"],
        "mae_frame": salience_data["results"]["basic_pitch"]["mae_frame"],
        "mae_clip": salience_data["results"]["basic_pitch"]["mae_clip"],
        "color": "#9467bd",
    },
}

names = list(MODELS.keys())
rmse_vals = [MODELS[n]["rmse"] for n in names]
r2_vals = [MODELS[n]["r2"] for n in names]
colors = [MODELS[n]["color"] for n in names]

# ── 3. leaderboard: RMSE + R² ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

bars1 = ax1.bar(range(len(names)), rmse_vals, color=colors, width=0.6)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
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
        fontsize=9,
    )

bars2 = ax2.bar(range(len(names)), r2_vals, color=colors, width=0.6)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
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
        fontsize=9,
    )

plt.tight_layout()
fig.savefig(ASSETS / "leaderboard_metrics.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: leaderboard_metrics.png")

# ── 4. per-rotor MAE (salience models only) ──────────────────────────────
salience_names = ["multif0_salience", "multif0_salience_fastest", "basic_pitch_salience"]
salience_labels = ["LateDeep", "LateDeep-fast", "Basic Pitch"]
salience_colors = [MODELS[n]["color"] for n in salience_names]


def _result_key(name):
    if name == "basic_pitch_salience":
        return "basic_pitch"
    return name


per_rotor = {n: salience_data["results"][_result_key(n)]["mae_per_rotor"] for n in salience_names}

x = np.arange(4)
width = 0.25
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (name, label, color) in enumerate(zip(salience_names, salience_labels, salience_colors)):
    ax.bar(x + i * width, per_rotor[name], width, label=label, color=color)

ax.set_xlabel("Rotor index", fontsize=11)
ax.set_ylabel("MAE (Hz)", fontsize=11)
ax.set_title("Per-rotor frame MAE", fontsize=13)
ax.set_xticks(x + width)
ax.set_xticklabels(["0", "1", "2", "3"])
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(ASSETS / "per_rotor_mae.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: per_rotor_mae.png")

# ── 5. load salience models for per-sample inference ─────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def load_salience_models(device=DEVICE):
    """Load the three salience models with correct construction configs."""
    models = {}
    ckpt_root = PROJECT_ROOT / "results" / "rps_baselines_v4"

    # multif0_salience (LateDeep, fmin=32.7, 3 harmonics)
    m = LateDeepSalience(n_fft=2048, hop_length=512, num_rotors=4, fmin=32.7)
    ckpt = torch.load(
        ckpt_root / "multif0_salience" / "best_multif0_salience.pt", map_location=device
    )
    m.load_state_dict(ckpt, strict=True)
    m.eval().to(device)
    models["multif0_salience"] = m

    # multif0_salience_fastest (stacked, fused, fmin=27.5, 4 harmonics)
    m = LateDeepSalience(
        n_fft=2048, hop_length=512, num_rotors=4, fmin=27.5, stacked=True, fused_branches=True
    )
    ckpt = torch.load(
        ckpt_root / "multif0_salience_fastest" / "best_multif0_salience.pt", map_location=device
    )
    m.load_state_dict(ckpt, strict=True)
    m.eval().to(device)
    models["multif0_salience_fastest"] = m

    # basic_pitch
    m = BasicPitchSalience(n_fft=2048, hop_length=512, num_rotors=4)
    ckpt = torch.load(
        ckpt_root / "basic_pitch" / "best_basic_pitch_salience.pt", map_location=device
    )
    m.load_state_dict(ckpt, strict=True)
    m.eval().to(device)
    models["basic_pitch_salience"] = m

    return models


print("Loading salience models...")
salience_models = load_salience_models()
print(f"Loaded: {list(salience_models.keys())}")

# ── 6. generate per-model salience + RPS plots ───────────────────────────
SAMPLE_ID = "sample_00026"
CHANNEL = 0
DATASET = PROJECT_ROOT / "datasets" / "DREGON-LM-V4" / "valid"
SALIENCE_VMAX = "auto"

sample = _load_sample(str(DATASET / SAMPLE_ID))
print(f"Loaded sample: {SAMPLE_ID}, channel {CHANNEL}")

# Generate the full comparison figure (all models on one page)
fig = plot_salience_comparison(
    sample, salience_models, channel=CHANNEL, device=DEVICE, salience_vmax=SALIENCE_VMAX
)
fig.suptitle(f"{SAMPLE_ID}  (channel {CHANNEL})", y=1.005, fontsize=13, fontweight="bold")
fig.savefig(ASSETS / f"{SAMPLE_ID}_salience_all_models.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SAMPLE_ID}_salience_all_models.png")

# Generate individual per-model figures: 3-pane (spectrogram + salience + RPS)
from plots.timeframe.registry import TrackContext
from plots.timeframe.renderers import make_spectrogram_series, render_salience

audio_us = cast(td.Series, sample["audio"])
mono = select_channel(audio_us, CHANNEL)

# Get GT RPS data
gt_rps_series = cast(td.Series, sample["rps"])
gt_rps_data = np.asarray(gt_rps_series.data, dtype=np.float32)
gt_rps_timestamps = np.asarray(cast(td.StampIndex, gt_rps_series.tindex).abs_stamps)

# ── Precompute shared spectrogram data ──
print("Precomputing spectrogram data...")
spec_track = make_spectrogram_series(mono, fmax=4000)
S = np.asarray(spec_track.series.data)
t_spec = cast(td.GridIndex, spec_track.series.tindex).sample_times()
freqs = np.linspace(0, 4000, S.shape[0])

for model_name, model in salience_models.items():
    print(f"Generating 3-pane figure for {model_name}...")

    # Get salience series and RPS prediction
    salience_track = model_salience_series(model, mono, device=DEVICE)
    rps_pred = model_rps_prediction(model, mono, device=DEVICE, track_threshold=0.3)

    # Resample pred to match GT length if needed
    pred = rps_pred
    if pred.shape[1] != gt_rps_data.shape[1]:
        from scipy import signal

        pred_resampled = np.zeros((4, gt_rps_data.shape[1]))
        for i in range(4):
            pred_resampled[i] = np.asarray(signal.resample(pred[i], gt_rps_data.shape[1]))
        pred = pred_resampled

    # Align rotor order to GT (PIT match) so the per-rotor colours are consistent.
    pred = align_rps_to_gt(pred, gt_rps_data)

    # ── 3-pane figure: spectrogram + salience + RPS ──
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1]}
    )

    # Pane 1: spectrogram (no colorbar)
    ax = axes[0]
    im = ax.pcolormesh(t_spec, freqs, S, shading="gouraud", cmap="magma")
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title(f"Audio spectrogram — {SAMPLE_ID} (channel {CHANNEL})", fontsize=12)
    ax.set_ylim(0, 4000)

    # Pane 2: salience map (no colorbar)
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

    # Pane 3: RPS trajectories
    ax = axes[2]
    t = gt_rps_timestamps
    colors_rps = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(4):
        ax.plot(t, gt_rps_data[i], ":", color=colors_rps[i], linewidth=1.5, label=f"GT R{i + 1}")
        ax.plot(t, pred[i], "-", color=colors_rps[i], linewidth=1.5, label=f"pred R{i + 1}")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("RPS", fontsize=11)
    ax.set_title(f"RPS — {model_name} (solid) vs GT (dotted)", fontsize=12)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / f"{SAMPLE_ID}_{model_name}_3pane.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {SAMPLE_ID}_{model_name}_3pane.pdf")

# ── 7. SimpleConvV2 RPS plot + comparisons ───────────────────────────────
print("\nGenerating SimpleConvV2 RPS plot...")
from models.rps_predictor import SimpleConvV2

simpleconv_v2 = SimpleConvV2(num_rotors=4)
ckpt_v2 = torch.load(
    PROJECT_ROOT / "results" / "rps_8ch_v4_simple_conv_v2" / "best_simple_conv_v2.pt",
    map_location=DEVICE,
)
simpleconv_v2.load_state_dict(ckpt_v2, strict=True)
simpleconv_v2.eval().to(DEVICE)

wav = torch.as_tensor(np.asarray(mono.data, dtype=np.float32), device=DEVICE).unsqueeze(0)
with torch.no_grad():
    pred_v2 = simpleconv_v2(wav)[0].cpu().numpy()  # (4, T)

# SimpleConvV2 RPS vs GT
fig, ax = plt.subplots(figsize=(14, 3.5))
t = gt_rps_timestamps
# Resample pred_v2 to match GT length
if pred_v2.shape[1] != gt_rps_data.shape[1]:
    from scipy import signal

    pred_v2_resampled = np.zeros((4, gt_rps_data.shape[1]))
    for i in range(4):
        pred_v2_resampled[i] = np.asarray(signal.resample(pred_v2[i], gt_rps_data.shape[1]))
    pred_v2 = pred_v2_resampled

# Align rotor order to GT (PIT match) before plotting.
pred_v2 = align_rps_to_gt(pred_v2, gt_rps_data)

colors_rps = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i in range(4):
    ax.plot(t, gt_rps_data[i], ":", color=colors_rps[i], linewidth=1.5, label=f"GT R{i + 1}")
    ax.plot(t, pred_v2[i], "-", color=colors_rps[i], linewidth=1.5, label=f"pred R{i + 1}")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("RPS", fontsize=11)
ax.set_title("RPS — SimpleConvV2 (8ch) (solid) vs GT (dotted)", fontsize=12)
ax.legend(loc="upper right", ncol=2, fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(ASSETS / f"{SAMPLE_ID}_simpleconvv2_rps.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: SimpleConvV2 RPS")

# Comparison plots: SimpleConvV2 vs each salience model
for model_name, model in salience_models.items():
    print(f"Generating comparison: SimpleConvV2 vs {model_name}...")

    rps_pred = model_rps_prediction(model, mono, device=DEVICE, track_threshold=0.3)
    pred = rps_pred
    if pred.shape[1] != gt_rps_data.shape[1]:
        from scipy import signal

        pred_resampled = np.zeros((4, gt_rps_data.shape[1]))
        for i in range(4):
            pred_resampled[i] = np.asarray(signal.resample(pred[i], gt_rps_data.shape[1]))
        pred = pred_resampled

    pred = align_rps_to_gt(pred, gt_rps_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # SimpleConvV2
    for i in range(4):
        ax1.plot(t, gt_rps_data[i], ":", color=colors_rps[i], linewidth=1.5)
        ax1.plot(t, pred_v2[i], "-", color=colors_rps[i], linewidth=1.5)
    ax1.set_ylabel("RPS", fontsize=11)
    ax1.set_title("SimpleConvV2 (8ch)", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(["GT", "pred"], loc="upper right", fontsize=8)

    # Salience model
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

print("\nAll assets generated.")
