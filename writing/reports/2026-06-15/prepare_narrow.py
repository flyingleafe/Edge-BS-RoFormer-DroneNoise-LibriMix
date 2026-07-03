#!/usr/bin/env python3
"""Append narrow-band + super-resolution salience results to the 2026-06-15 report.

Generates a parallel set of figures/tables (suffixed ``_narrow_sr``) for the two
new checkpoints (``multif0_salience_narrow_sr``, ``basic_pitch_narrow_sr``),
without touching the existing assets. Reuses the per-sample plotting helpers from
the sibling ``prepare.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402  (sibling report prepare.py)

from train_rps_predictor import get_model  # noqa: E402

ASSETS = P.ASSETS
DEVICE = P.DEVICE
PROJECT_ROOT = P.PROJECT_ROOT

# ── narrow-input + super-resolution-output configs (experiment run config) ──
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

# Register the two narrow-SR models in P.MODELS so the reused per-sample plotters
# (which read MODELS[key] for display/colour/salience flag) find them.
P.MODELS["multif0_salience_narrow_sr"] = {
    "ctor": lambda: get_model("multif0_salience", hcqt_fmin=55.0, salience_cfg=MULTIF0_CFG),
    "ckpt": PROJECT_ROOT
    / "results"
    / "rps_baselines_v4"
    / "multif0_salience_narrow_sr"
    / "best_multif0_salience.pt",
    "display": "multif0_salience_narrow_sr",
    "salience": True,
    "color": "#8c564b",
}
P.MODELS["basic_pitch_narrow_sr"] = {
    "ctor": lambda: get_model("basic_pitch_salience", salience_cfg=BP_CFG),
    "ckpt": PROJECT_ROOT
    / "results"
    / "rps_baselines_v4"
    / "basic_pitch_narrow_sr"
    / "best_basic_pitch_salience.pt",
    "display": "basic_pitch_narrow_sr",
    "salience": True,
    "color": "#e377c2",
}

NARROW_EVAL_JSON = (
    PROJECT_ROOT / "results" / "dregon_v4_eval" / "salience_narrow_sr_final_valid.json"
)

# Full model order for the combined leaderboard (originals + narrow-SR).
ORDER = [
    "simple_conv_v2",
    "simple_conv",
    "multif0_salience",
    "multif0_salience_fastest",
    "basic_pitch",
    "multif0_salience_narrow_sr",
    "basic_pitch_narrow_sr",
]


def load_all_metrics() -> pd.DataFrame:
    """Merge regression + original-salience + narrow-SR metrics into one frame."""
    df = P.load_eval_metrics()  # 5 original models
    with open(NARROW_EVAL_JSON) as f:
        narrow = json.load(f)["results"]
    rows = []
    for key in ["multif0_salience_narrow_sr", "basic_pitch_narrow_sr"]:
        r = narrow[key]
        rows.append(
            {
                "model": P.MODELS[key]["display"],
                "key": key,
                "rmse": r["rmse"],
                "mae_frame": r["mae_frame"],
                "mae_clip": r["mae_clip"],
                "r2": r["r2"],
                "r2_median": r["r2_median"],
                "eval_seconds": r.get("eval_seconds", np.nan),
            }
        )
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df["sort_key"] = pd.Categorical(df["key"], categories=ORDER, ordered=True)
    return df.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def generate_metrics_table(df: pd.DataFrame) -> str:
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
            "  caption: [RPS prediction leaderboard on DREGON-LM-V4/valid including the "
            "narrow-band super-resolution salience models (last two rows).],",
            ") <tab:leaderboard-narrow>",
        ]
    )
    return "\n".join(lines)


def plot_leaderboard(df: pd.DataFrame) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    models = df["model"].tolist()
    colors = [P.MODELS[k]["color"] for k in df["key"]]
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
    """Per-rotor MAE for all five salience models (originals + narrow-SR)."""
    with open(P.SALIENCE_EVAL_JSON) as f:
        orig = json.load(f)["results"]
    with open(NARROW_EVAL_JSON) as f:
        narrow = json.load(f)["results"]
    series = {
        "multif0_salience": orig["multif0_salience"]["mae_per_rotor"],
        "multif0_salience_fastest": orig["multif0_salience_fastest"]["mae_per_rotor"],
        "basic_pitch": orig["basic_pitch"]["mae_per_rotor"],
        "multif0_salience_narrow_sr": narrow["multif0_salience_narrow_sr"]["mae_per_rotor"],
        "basic_pitch_narrow_sr": narrow["basic_pitch_narrow_sr"]["mae_per_rotor"],
    }
    labels = {
        "basic_pitch": "basic_pitch_salience",
    }
    names = list(series)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(4)
    width = 0.16
    for i, name in enumerate(names):
        color = P.MODELS[name if name != "basic_pitch" else "basic_pitch"]["color"]
        ax.bar(
            x + i * width,
            series[name],
            width,
            label=labels.get(name, P.MODELS[name]["display"]),
            color=color,
            alpha=0.85,
        )
    ax.set_xlabel("Rotor index")
    ax.set_ylabel("MAE (Hz)")
    ax.set_title("Per-rotor frame MAE (salience models)")
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    ASSETS.mkdir(exist_ok=True)

    print("Loading metrics (incl. narrow-SR)...")
    df = load_all_metrics()
    df.to_csv(ASSETS / "metrics_narrow_sr.csv", index=False)
    (ASSETS / "metrics_table_narrow_sr.typ").write_text(generate_metrics_table(df))
    print("Wrote assets/metrics_table_narrow_sr.typ")

    fig = plot_leaderboard(df)
    fig.savefig(ASSETS / "leaderboard_metrics_narrow_sr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/leaderboard_metrics_narrow_sr.png")

    fig = plot_per_rotor_mae()
    fig.savefig(ASSETS / "per_rotor_mae_narrow_sr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/per_rotor_mae_narrow_sr.png")

    # Per-sample figures: load regression + narrow-SR models only.
    keys = ["simple_conv_v2", "simple_conv", "multif0_salience_narrow_sr", "basic_pitch_narrow_sr"]
    models = {}
    for key in keys:
        spec = P.MODELS[key]
        m = spec["ctor"]().to(DEVICE)
        m.load_state_dict(
            torch.load(spec["ckpt"], map_location=DEVICE, weights_only=True), strict=True
        )
        m.eval()
        models[key] = m

    narrow_only = {k: models[k] for k in ["multif0_salience_narrow_sr", "basic_pitch_narrow_sr"]}

    print("Generating separate-RPS comparison for sample_00026 (narrow-SR)...")
    fig = P.plot_separate_rps_comparison(
        "sample_00026", narrow_only, channel=P.CHANNEL, track_threshold=P.TRACK_THRESHOLD
    )
    fig.savefig(ASSETS / "sample_00026_narrow_sr_separate_rps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/sample_00026_narrow_sr_separate_rps.png")

    print("Generating all-models RPS comparison for sample_00026 (narrow-SR vs regression)...")
    fig = plot_all_models_rps_narrow("sample_00026", models, keys)
    fig.savefig(ASSETS / "sample_00026_narrow_sr_all_models_rps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/sample_00026_narrow_sr_all_models_rps.png")

    print("Done.")


def plot_all_models_rps_narrow(sample_id, models, order):
    """Like P.plot_all_models_rps but with an explicit model order (incl. narrow keys)."""
    from typing import cast

    import tdseries as td

    from plots.rps_prediction.salience_comparison import model_rps_prediction, select_channel
    from plots.rps_prediction.sample_comparison import _load_sample
    from plots.timeframe.renderers import ROTOR_COLORS
    from tasks.rps_prediction import align_rps_to_gt

    sample = _load_sample(str(P.DATASET / sample_id))
    mono = select_channel(cast(td.Series, sample["audio"]), P.CHANNEL)
    rps_track = cast(td.Series, sample["rps"]) if "rps" in sample else None
    dur = mono.duration

    def predict(key, model):
        if P.MODELS[key]["salience"]:
            return model_rps_prediction(
                model, mono, device=DEVICE, track_threshold=P.TRACK_THRESHOLD
            )
        return P.regression_rps_prediction(model, mono)

    keys = [k for k in order if k in models]
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.0 * n + 1.0), sharex=True)
    if n == 1:
        axes = [axes]

    pred0 = predict(keys[0], models[keys[0]])
    pred_times = np.linspace(mono.t_start, mono.t_start + dur, pred0.shape[-1])
    all_vals = []
    if rps_track is not None and rps_track.data is not None:
        all_vals.append(rps_track.interpolate(pred_times))
    for key in keys:
        all_vals.append(predict(key, models[key]))
    arr = np.concatenate([np.asarray(v) for v in all_vals], axis=1)
    ymin, ymax = float(arr.min()) - 5, float(arr.max()) + 5

    for ax, key in zip(axes, keys):
        pred = predict(key, models[key])
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
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(mono.t_start, mono.t_start + dur)
        ax.set_title(P.MODELS[key]["display"])
        ax.legend(loc="upper right", ncol=2, fontsize=6)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"RPS trajectories — regression vs narrow-SR salience  ({sample_id}, channel {P.CHANNEL})",
        y=1.005,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
