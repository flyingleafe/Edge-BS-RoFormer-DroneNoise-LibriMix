#!/usr/bin/env python3
"""
Generate publication-quality figures for the classical-baselines report.
Run from the project root:
    python papers/classical_baselines_report/generate_figures.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

from classical_rps_predictors import (
    pyin_single_f0,
    cepstral_tracker,
    hps_tracker,
    matched_filter_tracker,
    nmf_tracker,
    evaluate_predictions,
)
from train_rps_predictor import SimpleConv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SR = 16_000
N_FFT = 2048
HOP_LENGTH = 512
DATA_DIR = PROJECT_ROOT / "datasets/DREGON-LM-test/valid"
DEVICE = torch.device("cpu")

METHODS = {
    "pyin": pyin_single_f0,
    "cepstral": cepstral_tracker,
    "hps": hps_tracker,
    "matched_filter": matched_filter_tracker,
    "nmf": nmf_tracker,
}

METHOD_LABELS = {
    "simple_conv": "SimpleConv",
    "pyin": "PYIN",
    "cepstral": "Cepstral",
    "hps": "HPS",
    "matched_filter": "Matched Filter",
    "nmf": "NMF",
}

METHOD_COLORS = {
    "gt": "#333333",
    "pyin": "#ff7f0e",
    "cepstral": "#2ca02c",
    "hps": "#d62728",
    "matched_filter": "#9467bd",
    "nmf": "#8c564b",
    "simple_conv": "#1f77b4",
}

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "pdf.compression": 9,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample(sample_dir: Path):
    audio, sr = sf.read(sample_dir / "mixture.wav")
    gt_raw = np.load(sample_dir / "rps.npy")
    n_frames = len(audio) // HOP_LENGTH + 1
    gt = np.zeros((4, n_frames), dtype=np.float32)
    x_old = np.linspace(0, 1, gt_raw.shape[1])
    x_new = np.linspace(0, 1, n_frames)
    for r in range(4):
        gt[r] = np.interp(x_new, x_old, gt_raw[r])
    return audio.astype(np.float32), sr, gt


def load_simpleconv():
    model = SimpleConv(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    ckpt = torch.load(
        PROJECT_ROOT / "results/rps_predictor/best.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def simpleconv_predict(model, audio):
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x)
    return pred.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
def fig_bar_mse(df_summary):
    methods = ["simple_conv", "pyin", "cepstral", "hps", "matched_filter", "nmf"]
    labels = ["SimpleConv", "PYIN", "Cepstral", "HPS", "Matched\nFilter", "NMF"]
    means = [df_summary.loc[m, ("mse", "mean")] for m in methods]
    stds = [df_summary.loc[m, ("mse", "std")] for m in methods]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=[METHOD_COLORS[m] for m in methods],
                  edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_ylabel("Mean squared error [(rev/s)$^2$]")
    ax.set_title("Aggregate MSE on DREGON-LM test set (10 samples)")
    ax.set_yscale("log")
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, m in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f"{m:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_bar_mse.pdf")
    plt.close(fig)
    print("Saved fig_bar_mse.pdf")


# ---------------------------------------------------------------------------
# Big per-sample page: spectrogram + 5 per-rotor method panels
# ---------------------------------------------------------------------------
def plot_sample_page(audio, gt, preds_dict, sid):
    """
    One big figure per sample.
    Top row: spectrogram.
    Rows 1-6: per-rotor traces for each method [SC, PYIN, Cepstral, HPS, MF, NMF].
    """
    duration = len(audio) / SR
    method_keys = ["simple_conv", "pyin", "cepstral", "hps", "matched_filter", "nmf"]

    fig, axes = plt.subplots(7, 1, figsize=(7.2, 10.5),
                             gridspec_kw={"height_ratios": [1.2, 1, 1, 1, 1, 1, 1],
                                          "hspace": 0.22})

    # --- Spectrogram (shared) ---
    ax = axes[0]
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max
    )
    ax.imshow(D, origin="lower", aspect="auto",
              extent=[0, duration, 0, SR / 2 / 1000], cmap="magma",
              vmin=-80, vmax=0)
    ax.set_ylim(0, 4)
    ax.set_ylabel("Freq [kHz]")
    ax.set_title(f"{sid}")
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Per-rotor panels for each method ---
    for idx, method_key in enumerate(method_keys):
        ax = axes[idx + 1]
        pred = preds_dict[method_key]
        min_len = min(gt.shape[1], pred.shape[1])
        t_gt = np.linspace(0, duration, gt.shape[1])[:min_len]
        t_pred = np.linspace(0, duration, pred.shape[1])[:min_len]

        for r in range(4):
            ax.plot(t_gt, gt[r, :min_len], ":", color=ROTOR_COLORS[r],
                    lw=0.7, alpha=0.45)
            ax.plot(t_pred, pred[r, :min_len], "-", color=ROTOR_COLORS[r],
                    lw=0.9, alpha=0.85)

        ax.set_xlim(0, duration)
        ax.set_ylabel("RPS")
        ax.set_title(METHOD_LABELS[method_key], loc="left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.25)

        if idx == len(method_keys) - 1:
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xticklabels([])

    # Add a single legend at the bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=ROTOR_COLORS[r], lw=1.5, label=f"R{r+1}")
        for r in range(4)
    ] + [Line2D([0], [0], color="#333333", linestyle=":", lw=1.2, label="GT")]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out_path = FIG_DIR / f"fig_page_{sid}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model = load_simpleconv()
    print(f"SimpleConv loaded on {DEVICE}")

    sample_dirs = sorted(DATA_DIR.glob("sample_*"))[:10]
    sample_ids = [s.name for s in sample_dirs]
    print(f"Selected {len(sample_ids)} samples")

    results = []
    all_preds = {}

    for sample_dir in sample_dirs:
        sid = sample_dir.name
        audio, sr, gt = load_sample(sample_dir)
        sample_result = {"sample_id": sid}
        preds_for_sample = {}

        sc_pred = simpleconv_predict(model, audio)
        preds_for_sample["simple_conv"] = sc_pred
        sample_result["simple_conv"] = evaluate_predictions(sc_pred, gt)

        for name, fn in METHODS.items():
            try:
                pred = fn(audio, sr)
                preds_for_sample[name] = pred
                sample_result[name] = evaluate_predictions(pred, gt)
            except Exception as e:
                print(f"  {sid} {name} FAILED: {e}")
                preds_for_sample[name] = np.zeros_like(gt)
                sample_result[name] = {"mse": np.nan, "mae": np.nan, "r2": np.nan}

        results.append(sample_result)
        all_preds[sid] = {"gt": gt, **preds_for_sample}
        print(f"Done {sid}")

    # Build DataFrame
    rows = []
    for sid_data in results:
        sid = sid_data["sample_id"]
        for method in ["simple_conv", "pyin", "cepstral", "hps", "matched_filter", "nmf"]:
            m = sid_data[method]
            rows.append({
                "sample": sid,
                "method": method,
                "mse": m["mse"],
                "mae": m["mae"],
                "r2": m["r2"],
            })
    df = pd.DataFrame(rows)

    # Summary table
    summary = df.groupby("method")[["mse", "mae", "r2"]].agg(["mean", "std"]).round(2)
    summary = summary.reindex(["simple_conv", "pyin", "cepstral", "hps", "matched_filter", "nmf"])
    print("\nSummary:")
    print(summary)

    summary_latex = summary.to_latex(
        float_format="%.2f",
        multicolumn_format="c",
        escape=True,
        column_format="lcccccc",
    )
    with open(FIG_DIR / "table_summary.tex", "w") as f:
        f.write(summary_latex)
    print("Saved table_summary.tex")

    pivot_mse = df.pivot(index="sample", columns="method", values="mse")
    pivot_mse = pivot_mse[["simple_conv", "pyin", "cepstral", "hps", "matched_filter", "nmf"]]
    pivot_latex = pivot_mse.round(1).to_latex(
        float_format="%.1f",
        escape=True,
        column_format="lcccccc",
    )
    with open(FIG_DIR / "table_per_sample.tex", "w") as f:
        f.write(pivot_latex)
    print("Saved table_per_sample.tex")

    # Save metrics JSON
    metrics = {
        "summary": {k: v for k, v in summary.to_dict().items()},
        "per_sample_mse": pivot_mse.to_dict(),
    }
    def flatten_keys(d):
        out = {}
        for k, v in d.items():
            if isinstance(k, tuple):
                k = "_".join(str(x) for x in k)
            if isinstance(v, dict):
                out[k] = flatten_keys(v)
            else:
                out[k] = v
        return out
    with open(FIG_DIR / "metrics.json", "w") as f:
        json.dump(flatten_keys(metrics), f, indent=2)

    # --- Generate figures ---
    fig_bar_mse(summary)

    # Big per-sample pages for a few representative samples
    for sid in ["sample_00000", "sample_00002", "sample_00005"]:
        if sid in all_preds:
            audio, sr, gt = load_sample(DATA_DIR / sid)
            preds_dict = {k: v for k, v in all_preds[sid].items() if k != "gt"}
            plot_sample_page(audio, gt, preds_dict, sid)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
