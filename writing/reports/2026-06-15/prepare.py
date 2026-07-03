#!/usr/bin/env python3
"""Generate figures and tables for the salience-baseline RPS report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tdseries as td
import torch

# Ensure project src is importable (prepare.py runs inside writing/reports/2026-06-15/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from models.registry import get_rps_model as get_model
from models.salience_rps import BasicPitchSalience, LateDeepSalience, SalienceRPSPredictor
from plots.rps_prediction.salience_comparison import (
    build_salience_tracks,
    model_rps_prediction,
    model_salience_series,
    select_channel,
)
from plots.rps_prediction.sample_comparison import _load_sample
from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.renderers import ROTOR_COLORS, make_spectrogram_series
from tasks.rps_prediction import align_rps_to_gt

ASSETS = Path("assets")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = PROJECT_ROOT / "datasets" / "DREGON-LM-V4" / "valid"

# Model specifications for this report.
MODELS = {
    "simple_conv": {
        "ctor": lambda: get_model("simple_conv"),
        "ckpt": PROJECT_ROOT / "results" / "rps_8ch_v4_simple_conv" / "best_simple_conv.pt",
        "display": "SimpleConv (8ch)",
        "salience": False,
        "color": "#1f77b4",
    },
    "simple_conv_v2": {
        "ctor": lambda: get_model("simple_conv_v2"),
        "ckpt": PROJECT_ROOT / "results" / "rps_8ch_v4_simple_conv_v2" / "best_simple_conv_v2.pt",
        "display": "SimpleConvV2 (8ch)",
        "salience": False,
        "color": "#ff7f0e",
    },
    "multif0_salience": {
        "ctor": lambda: LateDeepSalience(n_fft=2048, hop_length=512, num_rotors=4, fmin=32.7),
        "ckpt": PROJECT_ROOT
        / "results"
        / "rps_baselines_v4"
        / "multif0_salience"
        / "best_multif0_salience.pt",
        "display": "multif0_salience",
        "salience": True,
        "color": "#2ca02c",
    },
    "multif0_salience_fastest": {
        "ctor": lambda: LateDeepSalience(
            n_fft=2048, hop_length=512, num_rotors=4, fmin=27.5, stacked=True, fused_branches=True
        ),
        "ckpt": PROJECT_ROOT
        / "results"
        / "rps_baselines_v4"
        / "multif0_salience_fastest"
        / "best_multif0_salience.pt",
        "display": "multif0_salience_fastest",
        "salience": True,
        "color": "#d62728",
    },
    "basic_pitch": {
        "ctor": lambda: BasicPitchSalience(n_fft=2048, hop_length=512, num_rotors=4),
        "ckpt": PROJECT_ROOT
        / "results"
        / "rps_baselines_v4"
        / "basic_pitch"
        / "best_basic_pitch_salience.pt",
        "display": "basic_pitch_salience",
        "salience": True,
        "color": "#9467bd",
    },
}

# Evaluation JSONs (must already exist).
SALIENCE_EVAL_JSON = (
    PROJECT_ROOT / "results" / "dregon_v4_eval" / "salience_baselines_final_valid.json"
)
REGRESSION_EVAL_JSON = (
    PROJECT_ROOT / "results" / "dregon_v4_eval" / "simpleconv_8ch_v4_full_valid.json"
)

SAMPLE_IDS = ["sample_00026", "sample_00020", "sample_00000"]
CHANNEL = 0
TRACK_THRESHOLD = 0.3


def load_eval_metrics() -> pd.DataFrame:
    """Load and merge evaluation metrics for all five models."""
    rows = []

    with open(SALIENCE_EVAL_JSON) as f:
        salience_data = json.load(f)
    for key in ["multif0_salience", "multif0_salience_fastest", "basic_pitch"]:
        r = salience_data["results"][key]
        rows.append(
            {
                "model": MODELS[key]["display"],
                "key": key,
                "rmse": r["rmse"],
                "mae_frame": r["mae_frame"],
                "mae_clip": r["mae_clip"],
                "r2": r["r2"],
                "r2_median": r["r2_median"],
                "eval_seconds": r["eval_seconds"],
            }
        )

    with open(REGRESSION_EVAL_JSON) as f:
        reg_data = json.load(f)
    for key in ["simple_conv", "simple_conv_v2"]:
        r = reg_data["results"][key]
        rows.append(
            {
                "model": MODELS[key]["display"],
                "key": key,
                "rmse": r["rmse"],
                "mae_frame": r["mae_frame"],
                "mae_clip": r["mae_clip"],
                "r2": r["r2"],
                "r2_median": r["r2_median"],
                "eval_seconds": r.get("eval_seconds", np.nan),
            }
        )

    df = pd.DataFrame(rows)
    # Order: regression models first, then salience baselines.
    order = [
        "simple_conv_v2",
        "simple_conv",
        "multif0_salience",
        "multif0_salience_fastest",
        "basic_pitch",
    ]
    df["sort_key"] = pd.Categorical(df["key"], categories=order, ordered=True)
    df = df.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    return df


def generate_metrics_table(df: pd.DataFrame) -> str:
    """Return Typst table source for the leaderboard."""
    lines = [
        "#figure(",
        "  placement: none,",
        "  table(",
        "    columns: (2fr, auto, auto, auto, auto, auto),",
        "    inset: 6pt,",
        "    align: (left + horizon, center + horizon, center + horizon, center + horizon, center + horizon, center + horizon),",
        "    table.header([*Model*], [*RMSE (Hz)*], [*MAE frame (Hz)*], [*MAE clip (Hz)*], [*$R^2$*], [*$R^2$ median*]),",
        "    table.hline(),",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"    [{row['model']}], [{row['rmse']:.2f}], [{row['mae_frame']:.2f}], "
            f"[{row['mae_clip']:.2f}], [{row['r2']:.3f}], [{row['r2_median']:.3f}],"
        )
    lines.extend(
        [
            "  ),",
            "  caption: [RPS prediction leaderboard on DREGON-LM-V4/valid (30 clips × 8 channels, PIT eval).],",
            ") <tab:leaderboard>",
        ]
    )
    return "\n".join(lines)


def plot_leaderboard(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Bar chart of RMSE and R² for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    models = df["model"].tolist()
    colors = [MODELS[k]["color"] for k in df["key"]]
    x = np.arange(len(models))
    width = 0.6

    ax = axes[0]
    bars = ax.bar(x, df["rmse"], width, color=colors, alpha=0.85)
    ax.set_ylabel("RMSE (Hz)")
    ax.set_title("RPS prediction error")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, df["rmse"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax = axes[1]
    bars = ax.bar(x, df["r2"], width, color=colors, alpha=0.85)
    ax.set_ylabel("$R^2$")
    ax.set_title("Coefficient of determination")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(min(df["r2"].min() - 1, -1), 1.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, df["r2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    return fig


def plot_per_rotor_mae() -> matplotlib.figure.Figure:
    """Per-rotor MAE for the three salience models (regression models lack per-rotor breakdown)."""
    with open(SALIENCE_EVAL_JSON) as f:
        data = json.load(f)

    models = ["multif0_salience", "multif0_salience_fastest", "basic_pitch"]
    labels = [MODELS[m]["display"] for m in models]
    per_rotor = [data["results"][m]["mae_per_rotor"] for m in models]
    per_rotor = np.array(per_rotor)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    width = 0.25
    for i, (model, label) in enumerate(zip(models, labels)):
        ax.bar(
            x + i * width,
            per_rotor[i],
            width,
            label=label,
            color=MODELS[model]["color"],
            alpha=0.85,
        )

    ax.set_xlabel("Rotor index")
    ax.set_ylabel("MAE (Hz)")
    ax.set_title("Per-rotor frame MAE")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def load_all_models() -> dict[str, torch.nn.Module]:
    """Load all five models onto DEVICE."""
    loaded: dict[str, torch.nn.Module] = {}
    for key, spec in MODELS.items():
        print(f"Loading {key}...")
        m = spec["ctor"]().to(DEVICE)
        m.load_state_dict(
            torch.load(spec["ckpt"], map_location=DEVICE, weights_only=True), strict=True
        )
        m.eval()
        loaded[key] = m
    return loaded


def regression_rps_prediction(model: torch.nn.Module, audio: td.Series) -> np.ndarray:
    """Run a regression model on mono audio Series and return (4, T_stft) numpy."""
    wav = torch.as_tensor(np.asarray(audio.data, dtype=np.float32), device=DEVICE)
    if wav.ndim != 1:
        wav = wav.mean(dim=0) if wav.ndim == 2 else wav.reshape(-1)
    with torch.no_grad():
        pred = model(wav.unsqueeze(0))[0]
    return pred.detach().cpu().numpy()


def plot_full_comparison(
    sample_id: str, models: dict[str, torch.nn.Module]
) -> matplotlib.figure.Figure:
    """Spectrogram + salience rows + RPS row overlaying all five models."""
    sample_path = DATASET / sample_id
    sample = _load_sample(str(sample_path))
    audio_us = cast(td.Series, sample["audio"])
    mono = select_channel(audio_us, CHANNEL)

    salience_models = {
        k: cast(SalienceRPSPredictor, v) for k, v in models.items() if MODELS[k]["salience"]
    }

    # Build the plot tracks: spectrogram, GT RPS, and one salience heatmap per model.
    tracks_map = build_salience_tracks(
        sample,
        salience_models,
        channel=CHANNEL,
        device=DEVICE,
        fmax=4000.0,
        track_threshold=TRACK_THRESHOLD,
    )

    plot_tracks: list[PlotTrack | td.Series] = [tracks_map["spectrogram"]] + [
        tracks_map[f"salience_{name}"] for name in salience_models
    ]
    height_ratios = [1.0] + [2.0] * len(salience_models)
    if "rps" in tracks_map:
        plot_tracks.append(tracks_map["rps"])
        height_ratios.append(1.0)

    fig = plot_timeframe(
        sample,
        tracks=plot_tracks,
        figsize=(
            15,
            3.0 * sum(height_ratios) / max(len(plot_tracks), 1) + 2.0 * len(plot_tracks),
        ),
        height_ratios=height_ratios,
        salience_vmax="auto",
    )

    # Overlay all model predictions on the final RPS axis.
    if "rps" in tracks_map:
        gt_track = cast(td.Series, tracks_map["rps"])
        ax = fig.axes[-1]
        dur = mono.duration
        for key, model in models.items():
            if MODELS[key]["salience"]:
                pred = model_rps_prediction(
                    cast(SalienceRPSPredictor, model),
                    mono,
                    device=DEVICE,
                    track_threshold=TRACK_THRESHOLD,
                )
            else:
                pred = regression_rps_prediction(model, mono)
            pred_times = np.linspace(mono.t_start, mono.t_start + dur, pred.shape[-1])
            pred = align_rps_to_gt(pred, np.asarray(gt_track.interpolate(pred_times)))
            style = "-" if MODELS[key]["salience"] else "--"
            alpha = 0.85 if MODELS[key]["salience"] else 1.0
            for r in range(min(pred.shape[0], len(ROTOR_COLORS))):
                ax.plot(
                    pred_times,
                    pred[r],
                    color=ROTOR_COLORS[r],
                    linewidth=1.4,
                    linestyle=style,
                    alpha=alpha,
                )
        ax.set_title(
            "RPS — GT (solid thin) vs salience models (solid) vs regression models (dashed)"
        )

    fig.suptitle(f"{sample_id}  (channel {CHANNEL})", y=1.005, fontsize=13, fontweight="bold")
    return fig


def plot_separate_rps_comparison(
    sample_id: str,
    models: dict[str, torch.nn.Module],
    *,
    channel: int = 0,
    track_threshold: float = 0.3,
) -> matplotlib.figure.Figure:
    """7-pane: spectrogram + 3 salience maps + 3 per-model RPS panels."""
    sample_path = DATASET / sample_id
    sample = _load_sample(str(sample_path))
    audio_us = cast(td.Series, sample["audio"])
    mono = select_channel(audio_us, channel)

    salience_models = {
        k: cast(SalienceRPSPredictor, v) for k, v in models.items() if MODELS[k]["salience"]
    }
    rps_track = cast(td.Series, sample["rps"]) if "rps" in sample else None
    dur = mono.duration

    # Compute global y-limits for RPS panels
    all_rps_values = []
    if rps_track is not None and rps_track.data is not None:
        all_rps_values.append(rps_track.data)
    for key, model in models.items():
        if MODELS[key]["salience"]:
            pred = model_rps_prediction(
                cast(SalienceRPSPredictor, model),
                mono,
                device=DEVICE,
                track_threshold=track_threshold,
            )
        else:
            pred = regression_rps_prediction(model, mono)
        all_rps_values.append(pred)
    if all_rps_values:
        all_rps_arr = np.concatenate([np.asarray(v) for v in all_rps_values], axis=1)
        rps_ymin = float(np.min(all_rps_arr)) - 5
        rps_ymax = float(np.max(all_rps_arr)) + 5
    else:
        rps_ymin, rps_ymax = 0, 150

    salience_names = list(salience_models.keys())
    n_rows = 1 + 2 * len(salience_names)
    height_ratios = [1.0]
    for _ in range(len(salience_names)):
        height_ratios.extend([2.0, 1.5])

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

    # Row 0: spectrogram
    ax = fig.add_subplot(gs[0])
    spec_track = make_spectrogram_series(mono, fmax=4000.0)
    S = spec_track.series.data
    times = cast(td.GridIndex, spec_track.series.tindex).sample_times()
    freq_max_hz = spec_track.hints.get(
        "freq_max_hz", float(cast(td.GridIndex, mono.tindex).rate) / 2.0
    )
    freqs = np.linspace(0, freq_max_hz, S.shape[0])
    ax.pcolormesh(times, freqs, S, shading="auto", cmap="magma")
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("spectrogram")
    ax.set_xlim(mono.t_start, mono.t_start + dur)

    row_idx = 1
    for key in salience_names:
        model = salience_models[key]

        # Salience row
        ax_sal = fig.add_subplot(gs[row_idx])
        sal_track = model_salience_series(
            model,
            mono,
            device=DEVICE,
            with_prediction=False,
            title=f"{MODELS[key]['display']} salience",
        )
        S = sal_track.series.data
        times = cast(td.GridIndex, sal_track.series.tindex).sample_times()
        freqs = sal_track.hints["freqs"]
        vmax = float(np.percentile(S, 99.5)) or 1.0
        mesh = ax_sal.pcolormesh(times, freqs, S, shading="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ax_sal.set_yscale("log")
        ax_sal.set_ylim(freqs[0], freqs[-1])
        ax_sal.set_ylabel("Freq (Hz)")
        ax_sal.set_title(sal_track.hints.get("title", f"{MODELS[key]['display']} salience"))
        fig.colorbar(mesh, ax=ax_sal, pad=0.01, fraction=0.025, label="salience")
        ax_sal.set_xlim(mono.t_start, mono.t_start + dur)

        row_idx += 1

        # RPS row
        ax_rps = fig.add_subplot(gs[row_idx])
        pred = model_rps_prediction(model, mono, device=DEVICE, track_threshold=track_threshold)
        pred_times = np.linspace(mono.t_start, mono.t_start + dur, pred.shape[-1])

        if rps_track is not None and rps_track.data is not None:
            gt = rps_track.interpolate(pred_times)
            pred = align_rps_to_gt(pred, np.asarray(gt))
            for r in range(min(gt.shape[0], len(ROTOR_COLORS))):
                ax_rps.plot(
                    pred_times,
                    gt[r],
                    color=ROTOR_COLORS[r],
                    linewidth=1.8,
                    linestyle=":",
                    alpha=0.9,
                    label=f"GT R{r + 1}",
                )

        for r in range(min(pred.shape[0], len(ROTOR_COLORS))):
            ax_rps.plot(
                pred_times,
                pred[r],
                color=ROTOR_COLORS[r],
                linewidth=1.6,
                linestyle="-",
                alpha=0.95,
                label=f"pred R{r + 1}",
            )

        ax_rps.set_ylabel("RPS")
        ax_rps.set_ylim(rps_ymin, rps_ymax)
        ax_rps.set_xlim(mono.t_start, mono.t_start + dur)
        ax_rps.set_title(f"RPS — {MODELS[key]['display']} (solid) vs GT (dotted)")
        ax_rps.legend(loc="upper right", ncol=2, fontsize=6)
        ax_rps.grid(True, alpha=0.3)

        row_idx += 1

    # Only bottom axis shows x label
    for ax in fig.axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.axes[-1].set_xlabel("Time (s)")

    gs.tight_layout(fig)
    fig.suptitle(f"{sample_id}  (channel {channel})", y=1.005, fontsize=13, fontweight="bold")
    return fig


def plot_all_models_rps(
    sample_id: str,
    models: dict[str, torch.nn.Module],
    *,
    channel: int = 0,
    track_threshold: float = 0.3,
) -> matplotlib.figure.Figure:
    """5-pane figure: one pane per model, each showing GT (dotted) and model predictions (solid)."""
    sample_path = DATASET / sample_id
    sample = _load_sample(str(sample_path))
    audio_us = cast(td.Series, sample["audio"])
    mono = select_channel(audio_us, channel)

    rps_track = cast(td.Series, sample["rps"]) if "rps" in sample else None
    dur = mono.duration

    # Determine model order
    order = [
        "simple_conv_v2",
        "simple_conv",
        "multif0_salience",
        "multif0_salience_fastest",
        "basic_pitch",
    ]
    model_keys = [k for k in order if k in models]
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, 1, figsize=(15, 3.0 * n_models + 1.0), sharex=True)
    if n_models == 1:
        axes = [axes]

    # Common time grid from first model
    first_key = model_keys[0]
    first_model = models[first_key]
    if MODELS[first_key]["salience"]:
        dummy_pred = model_rps_prediction(
            cast(SalienceRPSPredictor, first_model),
            mono,
            device=DEVICE,
            track_threshold=track_threshold,
        )
    else:
        dummy_pred = regression_rps_prediction(first_model, mono)
    pred_times = np.linspace(mono.t_start, mono.t_start + dur, dummy_pred.shape[-1])

    # Global y-limits across all models
    all_rps_values = []
    if rps_track is not None and rps_track.data is not None:
        gt = rps_track.interpolate(pred_times)
        all_rps_values.append(gt)
    for key in model_keys:
        model = models[key]
        if MODELS[key]["salience"]:
            pred = model_rps_prediction(
                cast(SalienceRPSPredictor, model),
                mono,
                device=DEVICE,
                track_threshold=track_threshold,
            )
        else:
            pred = regression_rps_prediction(model, mono)
        all_rps_values.append(pred)
    if all_rps_values:
        all_rps_arr = np.concatenate([np.asarray(v) for v in all_rps_values], axis=1)
        rps_ymin = float(np.min(all_rps_arr)) - 5
        rps_ymax = float(np.max(all_rps_arr)) + 5
    else:
        rps_ymin, rps_ymax = 0, 150

    for ax, key in zip(axes, model_keys):
        model = models[key]
        if MODELS[key]["salience"]:
            pred = model_rps_prediction(
                cast(SalienceRPSPredictor, model),
                mono,
                device=DEVICE,
                track_threshold=track_threshold,
            )
        else:
            pred = regression_rps_prediction(model, mono)

        # GT (dotted)
        if rps_track is not None and rps_track.data is not None:
            gt = rps_track.interpolate(pred_times)
            pred = align_rps_to_gt(pred, np.asarray(gt))
            for r in range(min(gt.shape[0], len(ROTOR_COLORS))):
                ax.plot(
                    pred_times,
                    gt[r],
                    color=ROTOR_COLORS[r],
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.9,
                    label=f"GT R{r + 1}",
                )

        # Model predictions (solid)
        for r in range(min(pred.shape[0], len(ROTOR_COLORS))):
            ax.plot(
                pred_times,
                pred[r],
                color=ROTOR_COLORS[r],
                linewidth=1.6,
                linestyle="-",
                alpha=0.95,
                label=f"pred R{r + 1}",
            )

        ax.set_ylabel("RPS")
        ax.set_ylim(rps_ymin, rps_ymax)
        ax.set_xlim(mono.t_start, mono.t_start + dur)
        ax.set_title(MODELS[key]["display"])
        ax.legend(loc="upper right", ncol=2, fontsize=6)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"RPS trajectories — all models vs GT  ({sample_id}, channel {channel})",
        y=1.005,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def main():
    ASSETS.mkdir(exist_ok=True)

    print("Loading metrics...")
    df = load_eval_metrics()
    df.to_csv(ASSETS / "metrics.csv", index=False)

    table_src = generate_metrics_table(df)
    (ASSETS / "metrics_table.typ").write_text(table_src)
    print("Wrote assets/metrics_table.typ")

    print("Generating leaderboard plot...")
    fig = plot_leaderboard(df)
    fig.savefig(ASSETS / "leaderboard_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/leaderboard_metrics.png")

    print("Generating per-rotor MAE plot...")
    fig = plot_per_rotor_mae()
    fig.savefig(ASSETS / "per_rotor_mae.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/per_rotor_mae.png")

    print("Loading models for sample comparisons...")
    models = load_all_models()

    print("Generating separate-RPS comparison figure for sample_00026...")
    fig = plot_separate_rps_comparison(
        "sample_00026", models, channel=CHANNEL, track_threshold=TRACK_THRESHOLD
    )
    fig.savefig(ASSETS / "sample_00026_separate_rps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/sample_00026_separate_rps.png")

    print("Generating all-models RPS comparison figure for sample_00026...")
    fig = plot_all_models_rps(
        "sample_00026", models, channel=CHANNEL, track_threshold=TRACK_THRESHOLD
    )
    fig.savefig(ASSETS / "sample_00026_all_models_rps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/sample_00026_all_models_rps.png")

    print("Done.")


if __name__ == "__main__":
    main()
