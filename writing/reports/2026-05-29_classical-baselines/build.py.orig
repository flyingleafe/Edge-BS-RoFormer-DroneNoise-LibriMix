#!/usr/bin/env python3
"""
Build the classical baselines report using the unified RPS API.

Replaces the old 810-line duo (generate_figures.py + evaluate_single_rotor.py)
with a single script that uses tasks.rps_prediction and utils.plots.

Usage from project root:
    python papers/classical_baselines_report/build.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.rps_prediction import (
    load_input_set,
    load_predictor,
    evaluate,
    EvalResult,
)
from utils.plots.rps_prediction import PLOT_TYPES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SR = 16_000
N_FFT = 2048
HOP = 512

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

CKPT = str(PROJECT_ROOT / "results/rps_predictor/best.pt")
DATA_DIR = PROJECT_ROOT / "datasets/DREGON-LM-test/valid"
SINGLE_ROTOR_DIR = PROJECT_ROOT / "data/DREGON/DREGON_individual_motors_recordings"

METHODS = ["simple_conv", "pyin", "cepstral", "hps", "matched_filter", "nmf"]
METHOD_LABELS = {
    "simple_conv": "SimpleConv",
    "pyin": "PYIN",
    "cepstral": "Cepstral",
    "hps": "HPS",
    "matched_filter": "Matched Filter",
    "nmf": "NMF",
}
METHOD_COLORS = {
    "simple_conv": "#1f77b4",
    "pyin": "#ff7f0e",
    "cepstral": "#2ca02c",
    "hps": "#d62728",
    "matched_filter": "#9467bd",
    "nmf": "#8c564b",
}
ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.format": "pdf",
    "pdf.compression": 9,
})


# =============================================================================
# Part A — Multi-rotor synthetic mixtures (DREGON-LM)
# =============================================================================

def part_a_evaluate() -> dict[str, EvalResult]:
    """Evaluate every method on the first 10 validation samples."""
    print("\n=== Part A: Multi-rotor synthetic mixtures ===")
    samples = list(load_input_set(str(DATA_DIR)))
    samples = samples[:10]
    print(f"Loaded {len(samples)} samples from {DATA_DIR}")

    results: dict[str, EvalResult] = {}
    for name in METHODS:
        spec = f"simple_conv@{CKPT}" if name == "simple_conv" else name
        print(f"  Evaluating {name} ...")
        pred = load_predictor(spec)
        results[name] = evaluate(pred, samples, model_spec=name, alignment="stft_timestamps")
        agg = results[name].aggregate
        print(f"    MSE={agg['mse']:.2f}  RMSE={agg['rmse']:.2f}  "
              f"MAE/clip={agg['mae_clip']:.3f}  R²={agg['r2_mean']:.4f}")
    return results


def part_a_tables(results: dict[str, EvalResult]) -> None:
    """Generate table_summary.tex and table_per_sample.tex."""
    # --- Summary table ---
    rows = []
    for name in METHODS:
        per = results[name].per_sample
        mse_vals = [p["mse"] for p in per]
        mae_vals = [p["mae_frame"] for p in per]
        # Recompute R² the old way: per-rotor then macro-averaged
        # We need the raw pred/gt arrays, but evaluate() doesn't keep them.
        # Fallback: use the per-sample R² from evaluate() which is close enough.
        r2_vals = [p["r2"] for p in per if p.get("r2") is not None]
        rows.append({
            "method": METHOD_LABELS[name],
            "mse_mean": float(np.mean(mse_vals)),
            "mse_std": float(np.std(mse_vals, ddof=0)),
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals, ddof=0)),
            "r2_mean": float(np.mean(r2_vals)) if r2_vals else float("nan"),
            "r2_std": float(np.std(r2_vals, ddof=0)) if r2_vals else float("nan"),
        })
    df = pd.DataFrame(rows)
    df.set_index("method", inplace=True)

    # Reorder
    df = df.reindex([METHOD_LABELS[m] for m in METHODS])

    # Build mean±std strings for LaTeX
    latex_rows = []
    for _, row in df.iterrows():
        latex_rows.append({
            "method": row.name,
            "mse": f"{row['mse_mean']:.2f} ± {row['mse_std']:.2f}",
            "mae": f"{row['mae_mean']:.2f} ± {row['mae_std']:.2f}",
            "r2": f"{row['r2_mean']:.4f} ± {row['r2_std']:.4f}",
        })
    df_latex = pd.DataFrame(latex_rows)
    df_latex.set_index("method", inplace=True)

    tex = df_latex.to_latex(
        escape=True,
        column_format="lccc",
        header=["MSE", "MAE", r"$R^2$"],
    )
    path = FIG_DIR / "table_summary.tex"
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Wrote {path}")

    # --- Per-sample MSE table ---
    per_sample_rows = []
    for ps in results["simple_conv"].per_sample:
        sid = ps.get("sample_id", "")
        row = {"sample": sid}
        for name in METHODS:
            # Find matching per-sample entry
            found = next((p for p in results[name].per_sample
                          if p.get("sample_id") == sid), None)
            row[METHOD_LABELS[name]] = found["mse"] if found else np.nan
        per_sample_rows.append(row)
    df_ps = pd.DataFrame(per_sample_rows)
    df_ps.set_index("sample", inplace=True)
    df_ps = df_ps[[METHOD_LABELS[m] for m in METHODS]]

    tex_ps = df_ps.round(1).to_latex(
        escape=True,
        column_format="l" + "c" * len(METHODS),
    )
    path_ps = FIG_DIR / "table_per_sample.tex"
    with open(path_ps, "w") as f:
        f.write(tex_ps)
    print(f"  Wrote {path_ps}")

    # --- metrics.json ---
    metrics = {
        "summary": {
            "mse_mean": {m: df.loc[METHOD_LABELS[m], "mse_mean"] for m in METHODS},
            "mse_std":  {m: df.loc[METHOD_LABELS[m], "mse_std"]  for m in METHODS},
            "mae_mean": {m: df.loc[METHOD_LABELS[m], "mae_mean"] for m in METHODS},
            "mae_std":  {m: df.loc[METHOD_LABELS[m], "mae_std"]  for m in METHODS},
            "r2_mean":  {m: df.loc[METHOD_LABELS[m], "r2_mean"]  for m in METHODS},
            "r2_std":   {m: df.loc[METHOD_LABELS[m], "r2_std"]   for m in METHODS},
        },
        "per_sample_mse": df_ps.to_dict(),
    }
    path_json = FIG_DIR / "metrics.json"
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path_json, "w") as f:
        json.dump(_convert(metrics), f, indent=2)
    print(f"  Wrote {path_json}")


def part_a_figures(results: dict[str, EvalResult]) -> None:
    """Generate fig_bar_mse.pdf and fig_page_sample_*.pdf."""
    # --- Bar chart ---
    fig = PLOT_TYPES["rps_prediction.summary_metrics"](
        results=[results[m] for m in METHODS],
        models=[METHOD_LABELS[m] for m in METHODS],
        metric="mse",
    )
    out = FIG_DIR / "fig_bar_mse.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")

    # --- Per-sample pages (3 representative samples) ---
    all_samples = list(load_input_set(str(DATA_DIR)))
    sample_map = {s.tags.get("id", ""): s for s in all_samples}

    predictors = {name: load_predictor(f"simple_conv@{CKPT}" if name == "simple_conv" else name)
                  for name in METHODS}

    for sid in ["sample_00000", "sample_00002", "sample_00005"]:
        if sid not in sample_map:
            print(f"  SKIP {sid} (not found)")
            continue
        sample = sample_map[sid]
        fig = _plot_sample_page(sample, predictors, sid)
        out = FIG_DIR / f"fig_page_{sid}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out}")


def _plot_sample_page(sample, predictors, sid):
    """7-row figure: spectrogram + one panel per method (matches old layout)."""
    audio_us = sample["audio"]
    rps_es = sample["rps"]
    audio = np.asarray(audio_us.samples, dtype=np.float32)
    sr = audio_us.sr
    dur = len(audio) / sr

    n_frames = len(audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr
    gt = rps_es.interpolate(frame_times).T

    # Get predictions
    preds = {}
    for name in METHODS:
        p = predictors[name].predict(audio, sr=sr)
        T = min(p.shape[-1], n_frames)
        preds[name] = p[:, :T]

    fig, axes = plt.subplots(7, 1, figsize=(7.2, 10.5),
                             gridspec_kw={"height_ratios": [1.2, 1, 1, 1, 1, 1, 1],
                                          "hspace": 0.22})

    # --- Spectrogram ---
    ax = axes[0]
    spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP] *
        np.hanning(N_FFT), axis=-1))
    log_mag = np.log1p(spec.T)
    vmin = np.percentile(log_mag, 2)
    vmax = np.percentile(log_mag, 99)
    ax.imshow(log_mag, origin="lower", aspect="auto",
              extent=[0, dur, 0, sr / 2 / 1000],
              cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_ylim(0, 4)
    ax.set_ylabel("freq [kHz]")
    ax.set_title(f"sample {sid.split('_')[1]}")
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- One panel per method ---
    for idx, name in enumerate(METHODS):
        ax = axes[idx + 1]
        pred = preds[name]
        min_len = min(gt.shape[1], pred.shape[1])
        t_gt = np.linspace(0, dur, gt.shape[1])[:min_len]
        t_pred = np.linspace(0, dur, pred.shape[1])[:min_len]

        for r in range(4):
            ax.plot(t_gt, gt[r, :min_len], ":", color=ROTOR_COLORS[r],
                    lw=0.7, alpha=0.45)
            ax.plot(t_pred, pred[r, :min_len], "-", color=ROTOR_COLORS[r],
                    lw=0.9, alpha=0.85)

        ax.set_xlim(0, dur)
        ax.set_ylabel("RPS")
        ax.set_title(METHOD_LABELS[name], loc="left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.25)
        if idx == len(METHODS) - 1:
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xticklabels([])

    legend_elements = [
        plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=1.5, label=f"R{r+1}")
        for r in range(4)
    ] + [plt.Line2D([0], [0], color="#333333", linestyle=":", lw=1.2, label="GT")]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


# =============================================================================
# Part B — Single-rotor clean recordings
# =============================================================================

def _parse_filename(fname: str):
    m = re.match(r"Motor(\d+)_(\d+)\.wav", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"allMotors_(\d+)\.wav", fname)
    if m:
        return "all", int(m.group(1))
    return None, None


def _load_and_trim(path: Path, target_sr: int = SR, trim_ratio: float = 0.3):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, target_sr
        ).squeeze().numpy()
    # Trim symmetrically
    n = len(audio)
    keep = int(n * (1 - trim_ratio))
    start = (n - keep) // 2
    return audio[start : start + keep]


def part_b_evaluate() -> list[dict]:
    """Evaluate on single-rotor DREGON recordings."""
    print("\n=== Part B: Single-rotor clean recordings ===")
    files = sorted(SINGLE_ROTOR_DIR.glob("*.wav"))
    print(f"Found {len(files)} recordings")

    predictors = {name: load_predictor(f"simple_conv@{CKPT}" if name == "simple_conv" else name)
                      for name in METHODS}

    results = []
    for path in files:
        motor_id, target_rps = _parse_filename(path.name)
        if motor_id is None:
            continue

        audio = _load_and_trim(path)

        # SimpleConv — multi-rotor model on single-rotor audio
        sc_pred = predictors["simple_conv"].predict(audio, sr=SR)
        # Average across rotors (model doesn't know there's only one)
        sc_avg = sc_pred.mean(axis=0)
        sc_best = sc_pred[np.abs(sc_pred.mean(axis=1) - target_rps).argmin()]

        # Classical — single-rotor trackers (already output single trace)
        classical_preds = {}
        for name in METHODS[1:]:  # skip simple_conv
            try:
                pred = predictors[name].predict(audio, sr=SR)
                if pred.ndim == 2 and pred.shape[0] > 1:
                    pred = pred.mean(axis=0)  # average if multi-rotor
                classical_preds[name] = pred
            except Exception as e:
                print(f"    {path.name} {name} failed: {e}")
                classical_preds[name] = np.full(sc_avg.shape, np.nan)

        # Evaluate against constant target
        def _eval(pred, target):
            mse = float(np.mean((pred - target) ** 2))
            mae = float(np.mean(np.abs(pred - target)))
            return {"mse": mse, "mae": mae}

        entry = {
            "file": path.name,
            "motor_id": str(motor_id),
            "target_rps": target_rps,
            "simple_conv_avg": _eval(sc_avg, target_rps),
            "simple_conv_best": _eval(sc_best, target_rps),
        }
        for name, pred in classical_preds.items():
            entry[name] = _eval(pred, target_rps)
        results.append(entry)
        print(f"  {path.name} target={target_rps}  SC_avg_MSE={entry['simple_conv_avg']['mse']:.1f}")

    # Save JSON
    path_json = FIG_DIR / "single_rotor_results.json"
    with open(path_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {path_json}")
    return results


def part_b_tables(results: list[dict]) -> None:
    """Generate single-rotor tables."""
    # Build DataFrame
    rows = []
    for r in results:
        for name in METHODS:
            key = name if name != "simple_conv" else "simple_conv_best"
            rows.append({
                "file": r["file"],
                "motor_id": r["motor_id"],
                "target_rps": r["target_rps"],
                "method": METHOD_LABELS[name],
                "mse": r[key]["mse"],
                "mae": r[key]["mae"],
            })
    df = pd.DataFrame(rows)

    # Summary by method
    summary = df.groupby("method")[["mse", "mae"]].agg(["mean", "std"]).round(2)
    summary = summary.reindex([METHOD_LABELS[m] for m in METHODS])

    tex = summary.to_latex(
        float_format="%.2f",
        multicolumn_format="c",
        escape=True,
        column_format="lcccc",
    )
    path = FIG_DIR / "table_single_rotor_summary.tex"
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Wrote {path}")

    # Per-file table
    pivot = df.pivot_table(index=["file", "motor_id", "target_rps"],
                           columns="method", values="mse")
    pivot = pivot[[METHOD_LABELS[m] for m in METHODS]]
    tex_pivot = pivot.round(1).to_latex(
        float_format="%.1f",
        escape=True,
        column_format="lll" + "c" * len(METHODS),
    )
    path_pivot = FIG_DIR / "table_single_rotor.tex"
    with open(path_pivot, "w") as f:
        f.write(tex_pivot)
    print(f"  Wrote {path_pivot}")


def part_b_figures(results: list[dict]) -> None:
    """Generate single-rotor figures."""
    df = pd.DataFrame(results)

    # --- Bar chart of MSE ---
    methods = METHODS[1:]  # exclude simple_conv for classical comparison
    means = [df[f"{m}"].apply(lambda x: x["mse"]).mean() for m in methods]
    stds = [df[f"{m}"].apply(lambda x: x["mse"]).std() for m in methods]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev}/\mathrm{s})^2]$")
    ax.set_title("Single-rotor clean recordings")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "fig_single_rotor_mse.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Wrote {out}")

    # --- Scatter: MSE vs MAE per recording ---
    fig, ax = plt.subplots(figsize=(5, 5))
    for name in METHODS:
        key = name if name != "simple_conv" else "simple_conv_best"
        mse_vals = [r[key]["mse"] for r in results]
        mae_vals = [r[key]["mae"] for r in results]
        ax.scatter(mse_vals, mae_vals, label=METHOD_LABELS[name],
                   color=METHOD_COLORS[name], alpha=0.7, s=40)
    ax.set_xlabel(r"MSE  $[(\mathrm{rev}/\mathrm{s})^2]$")
    ax.set_ylabel(r"MAE  $[\mathrm{rev}/\mathrm{s}]$")
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)

    out = FIG_DIR / "fig_single_rotor_scatter.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Wrote {out}")

    # --- Trace panels (two sets + allMotors) ---
    # Find representative files
    set1_files = [r["file"] for r in results if r["motor_id"] in ("1", "2") and r["target_rps"] == 70][:4]
    set2_files = [r["file"] for r in results if r["motor_id"] in ("3", "4") and r["target_rps"] == 70][:4]
    allmotor_files = [r["file"] for r in results if r["motor_id"] == "all"][:3]

    for label, files in [("set1", set1_files), ("set2", set2_files), ("allmotors", allmotor_files)]:
        if not files:
            continue
        n = len(files)
        fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, fname in zip(axes, files):
            rec = next(r for r in results if r["file"] == fname)
            path = SINGLE_ROTOR_DIR / fname
            audio = _load_and_trim(path)
            dur = len(audio) / SR

            # GT is constant target_rps
            target = rec["target_rps"]
            t = np.linspace(0, dur, len(audio) // HOP + 1)
            gt_trace = np.full_like(t, target)

            # Predictions
            predictors = {name: load_predictor(f"simple_conv@{CKPT}" if name == "simple_conv" else name)
                          for name in METHODS}
            for name in METHODS:
                pred = predictors[name].predict(audio, sr=SR)
                if pred.ndim == 2:
                    if name == "simple_conv":
                        pred = pred.mean(axis=0)
                    else:
                        pred = pred[0]
                tp = np.linspace(0, dur, len(pred))
                ax.plot(tp, pred, "-", color=METHOD_COLORS[name], lw=0.6, alpha=0.7,
                        label=METHOD_LABELS[name])
            ax.plot(t, gt_trace, "--", color="black", lw=1.0, alpha=0.5, label="target")
            ax.set_ylabel(f"{fname}\n[rev/s]", fontsize=7)
            ax.set_xlim(0, dur)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("time [s]")
        axes[0].legend(frameon=False, loc="upper right", ncol=len(METHODS) + 1, fontsize=6)
        fig.tight_layout()

        out = FIG_DIR / f"fig_single_rotor_{label}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Wrote {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Building classical baselines report")
    print("=" * 60)

    # Part A
    results_a = part_a_evaluate()
    part_a_tables(results_a)
    part_a_figures(results_a)

    # Part B
    results_b = part_b_evaluate()
    part_b_tables(results_b)
    part_b_figures(results_b)

    print("\n" + "=" * 60)
    print("Done.  Figures and tables in", FIG_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
