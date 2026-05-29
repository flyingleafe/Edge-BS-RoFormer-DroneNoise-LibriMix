#!/usr/bin/env python3
"""
Evaluate classical baselines and SimpleConv on clean DREGON individual motor
recordings (constant-RPS, single-rotor, no speech).  Uses the *single-pitch*
version of each classical method (no greedy multi-rotor hack).

Run from project root:
    python papers/classical_baselines_report/evaluate_single_rotor.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import re
import json
import numpy as np
import soundfile as sf
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classical_rps_predictors import (
    pyin_single_f0,
    cepstral_tracker,
    hps_tracker,
    matched_filter_tracker,
    nmf_tracker,
    evaluate_predictions,
    _stft_frame_count,
    _cepstral_rps_estimate,
    _hps_rps_estimate,
    _matched_filter_rps_estimate,
    _frame_spectra,
)
from train_rps_predictor import SimpleConv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SR = 16_000
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = torch.device("cpu")
DATA_DIR = PROJECT_ROOT / "data/DREGON/DREGON_individual_motors_recordings"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "pdf.compression": 9,
})


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


def parse_filename(fname: str):
    """Return (motor_id, rps). motor_id is 'all' for allMotors."""
    m = re.match(r"(?:Motor(\d)|allMotors)_(\d+)\.wav", fname)
    if not m:
        return None, None
    motor_id = m.group(1) if m.group(1) is not None else "all"
    return motor_id, int(m.group(2))


def load_and_trim(path: Path, target_sr: int = 16_000, trim_ratio: float = 0.3):
    """
    Load audio, resample to target_sr, return the middle (1-2*trim_ratio) portion.
    Uses channel 0 (first microphone).
    """
    audio, sr = sf.read(path)
    # DREGON recordings have 8 channels; take first
    if audio.ndim > 1:
        audio = audio[:, 0]
    # Resample to 16 kHz
    if sr != target_sr:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),
            orig_freq=sr,
            new_freq=target_sr,
        ).numpy()[0]
    # Trim start and end transients
    n_samples = len(audio)
    start = int(n_samples * trim_ratio)
    end = int(n_samples * (1 - trim_ratio))
    return audio[start:end]


# ---------------------------------------------------------------------------
# Single-pitch classical methods (no greedy hack)
# ---------------------------------------------------------------------------
def pyin_single(audio, sr=SR):
    """Direct PYIN call, no multi-rotor replication."""
    # pyin_single_f0 already returns (4, T) by replicating; we want the raw trace
    preds_4 = pyin_single_f0(audio, sr)
    return preds_4[0]  # just one rotor trace


def cepstral_single(audio, sr=SR):
    """Cepstral estimate per frame, no greedy suppression."""
    specs, _ = _frame_spectra(audio)
    n_frames = specs.shape[0]
    preds = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        preds[t] = _cepstral_rps_estimate(specs[t])
    return preds


def hps_single(audio, sr=SR):
    """HPS estimate per frame, no greedy suppression."""
    specs, _ = _frame_spectra(audio)
    n_frames = specs.shape[0]
    preds = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        preds[t] = _hps_rps_estimate(specs[t])
    return preds


def matched_filter_single(audio, sr=SR):
    """Matched filter estimate per frame, no greedy suppression."""
    specs, _ = _frame_spectra(audio)
    n_frames = specs.shape[0]
    preds = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        preds[t] = _matched_filter_rps_estimate(specs[t])
    return preds


def nmf_single(audio, sr=SR):
    """NMF with n_rotors=1 — returns the single strongest activation per frame."""
    preds = nmf_tracker(audio, sr, n_rotors=1)
    return preds[0]


METHODS = {
    "pyin": pyin_single,
    "cepstral": cepstral_single,
    "hps": hps_single,
    "matched_filter": matched_filter_single,
    "nmf": nmf_single,
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
    "pyin": "#ff7f0e",
    "cepstral": "#2ca02c",
    "hps": "#d62728",
    "matched_filter": "#9467bd",
    "nmf": "#8c564b",
    "simple_conv": "#1f77b4",
}


def evaluate_single_rotor(pred, target_rps):
    """Scalar metrics for a single 1-D trace against constant target."""
    pred = np.asarray(pred)
    target = np.full_like(pred, fill_value=target_rps, dtype=np.float32)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    # R^2 relative to constant mean = target_rps
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-6 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2, "mean_pred": float(pred.mean()), "std_pred": float(pred.std())}


def main():
    model = load_simpleconv()
    print(f"SimpleConv loaded on {DEVICE}")

    files = sorted(DATA_DIR.glob("*.wav"))
    results = []

    for fpath in files:
        fname = fpath.name
        motor_id, rps = parse_filename(fname)
        if motor_id is None:
            continue

        print(f"\n--- {fname} (motor={motor_id}, target RPS={rps}) ---")
        audio = load_and_trim(fpath)
        duration = len(audio) / SR
        print(f"  Trimmed duration: {duration:.2f}s")

        sample_result = {"file": fname, "motor_id": motor_id, "target_rps": rps, "preds": {}}

        # SimpleConv (4 channels) — evaluate all 4 outputs against the same target
        sc_pred = simpleconv_predict(model, audio)
        # sc_pred shape: (4, T)
        sc_metrics_per_rotor = []
        for r in range(4):
            m = evaluate_single_rotor(sc_pred[r], rps)
            sc_metrics_per_rotor.append(m)
        # Take best rotor (closest mean) or average? Let's report average and best
        avg_mse = np.mean([m["mse"] for m in sc_metrics_per_rotor])
        avg_mae = np.mean([m["mae"] for m in sc_metrics_per_rotor])
        best_idx = int(np.argmin([m["mse"] for m in sc_metrics_per_rotor]))
        sample_result["simple_conv_avg"] = {"mse": avg_mse, "mae": avg_mae}
        sample_result["simple_conv_best"] = sc_metrics_per_rotor[best_idx]
        sample_result["preds"]["simple_conv"] = sc_pred
        sample_result["preds"]["simple_conv_best_idx"] = best_idx
        print(f"  SimpleConv best rotor (R{best_idx+1}): MSE={sc_metrics_per_rotor[best_idx]['mse']:.2f}, MAE={sc_metrics_per_rotor[best_idx]['mae']:.2f}, mean={sc_metrics_per_rotor[best_idx]['mean_pred']:.1f}, std={sc_metrics_per_rotor[best_idx]['std_pred']:.2f}")
        print(f"  SimpleConv avg across 4: MSE={avg_mse:.2f}, MAE={avg_mae:.2f}")

        # Classical methods (single-pitch, no greedy)
        for name, fn in METHODS.items():
            pred = fn(audio)
            m = evaluate_single_rotor(pred, rps)
            sample_result[name] = m
            sample_result["preds"][name] = pred
            print(f"  {name}: MSE={m['mse']:.2f}, MAE={m['mae']:.2f}, mean={m['mean_pred']:.1f}, std={m['std_pred']:.2f}")

        results.append(sample_result)

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY — Single-pitch classical methods on clean single-rotor recordings")
    print("="*70)

    # Aggregate by method across all files
    methods = ["pyin", "cepstral", "hps", "matched_filter", "nmf", "simple_conv_best", "simple_conv_avg"]
    for method in methods:
        mses = [r[method]["mse"] for r in results if method in r]
        maes = [r[method]["mae"] for r in results if method in r]
        print(f"{method:20s}  MSE={np.mean(mses):7.2f}±{np.std(mses):5.2f}   MAE={np.mean(maes):5.2f}±{np.std(maes):4.2f}")

    # Save JSON (strip raw prediction arrays to keep it small and serialisable)
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != "preds"}
        json_results.append(jr)
    with open(FIG_DIR / "single_rotor_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print("\nSaved single_rotor_results.json")

    # Save full per-file LaTeX table
    rows = []
    for r in results:
        rows.append({
            "file": r["file"].replace(".wav", ""),
            "target": r["target_rps"],
            "PYIN": f"{r['pyin']['mse']:.1f} / {r['pyin']['mae']:.1f}",
            "Cepstral": f"{r['cepstral']['mse']:.1f} / {r['cepstral']['mae']:.1f}",
            "HPS": f"{r['hps']['mse']:.1f} / {r['hps']['mae']:.1f}",
            "MatchedF": f"{r['matched_filter']['mse']:.1f} / {r['matched_filter']['mae']:.1f}",
            "NMF": f"{r['nmf']['mse']:.1f} / {r['nmf']['mae']:.1f}",
            "SC_best": f"{r['simple_conv_best']['mse']:.1f} / {r['simple_conv_best']['mae']:.1f}",
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    latex = df.to_latex(index=False, escape=True, column_format="lcccccc")
    with open(FIG_DIR / "table_single_rotor.tex", "w") as f:
        f.write(latex)
    print("Saved table_single_rotor.tex")

    # Save aggregate summary LaTeX table (matches hardcoded format in main.tex)
    summary_rows = []
    method_names = [
        ("PYIN", "pyin"),
        ("Cepstral", "cepstral"),
        ("HPS", "hps"),
        ("Matched Filter", "matched_filter"),
        ("NMF", "nmf"),
        ("SimpleConv (best rotor)", "simple_conv_best"),
        ("SimpleConv (avg over 4)", "simple_conv_avg"),
    ]
    for label, key in method_names:
        mses = [r[key]["mse"] for r in results if key in r]
        maes = [r[key]["mae"] for r in results if key in r]
        summary_rows.append({
            "Method": label,
            "Mean MSE": f"{np.mean(mses):.1f}",
            "Mean MAE": f"{np.mean(maes):.1f}",
        })
    df_sum = pd.DataFrame(summary_rows)
    latex_sum = df_sum.to_latex(index=False, escape=True, column_format="lcc")
    with open(FIG_DIR / "table_single_rotor_summary.tex", "w") as f:
        f.write(latex_sum)
    print("Saved table_single_rotor_summary.tex")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    # One big summary figure: bar chart of MSE by method, grouped by file
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(results))
    width = 0.13
    method_keys = ["pyin", "cepstral", "hps", "matched_filter", "nmf", "simple_conv_best"]
    labels = ["PYIN", "Cepstral", "HPS", "Matched Filter", "NMF", "SimpleConv (best rotor)"]
    for i, (mk, lab) in enumerate(zip(method_keys, labels)):
        vals = [r[mk]["mse"] for r in results]
        ax.bar(x + (i - 2) * width, vals, width, label=lab, color=METHOD_COLORS.get(mk, "#333333"))

    ax.set_xticks(x)
    ax.set_xticklabels([r["file"].replace(".wav", "") for r in results], rotation=45, ha="right")
    ax.set_ylabel("MSE [(rev/s)$^2$]")
    ax.set_title("MSE on clean single-rotor DREGON recordings (single-pitch methods, no greedy hack)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_single_rotor_mse.pdf")
    plt.close(fig)
    print("Saved fig_single_rotor_mse.pdf")

    # Scatter plot: predicted mean vs target RPS for each method
    fig, ax = plt.subplots(figsize=(6, 5.5))
    targets = [r["target_rps"] for r in results]
    for mk, lab in zip(method_keys, labels):
        preds = [r[mk]["mean_pred"] for r in results]
        ax.scatter(targets, preds, label=lab, alpha=0.7, s=50, color=METHOD_COLORS.get(mk, "#333333"))

    # Perfect prediction line
    t_range = np.array([45, 95])
    ax.plot(t_range, t_range, "k--", lw=1, alpha=0.4, label="Perfect")
    ax.set_xlabel("Target RPS [rev/s]")
    ax.set_ylabel("Predicted mean RPS [rev/s]")
    ax.set_title("Mean predicted RPS vs ground truth (clean single-rotor recordings)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls="--", alpha=0.3)
    ax.set_xlim(45, 95)
    ax.set_ylim(45, 95)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_single_rotor_scatter.pdf")
    plt.close(fig)
    print("Saved fig_single_rotor_scatter.pdf")

    # -----------------------------------------------------------------------
    # Trace plots for selected recordings
    # -----------------------------------------------------------------------
    def _plot_trace_panel(ax, result, show_sc_channels=True):
        """Plot one panel with GT + all classical methods + SimpleConv."""
        target = result["target_rps"]
        duration = len(result["preds"]["pyin"]) * HOP_LENGTH / SR
        t = np.linspace(0, duration, len(result["preds"]["pyin"]))

        # GT constant line
        ax.axhline(target, color="#333333", linestyle="--", lw=1.2, alpha=0.6, label="GT")

        # Classical methods
        for mk in ["pyin", "cepstral", "hps", "matched_filter", "nmf"]:
            if mk in result["preds"]:
                ax.plot(t, result["preds"][mk], "-", color=METHOD_COLORS[mk],
                        lw=0.9, alpha=0.85, label=METHOD_LABELS[mk])

        # SimpleConv
        sc = result["preds"].get("simple_conv")
        if sc is not None:
            best_idx = result["preds"].get("simple_conv_best_idx", 0)
            sc_mean = sc.mean(axis=0)
            for r in range(4):
                ax.plot(t, sc[r], ":", color=METHOD_COLORS["simple_conv"],
                        lw=0.8, alpha=0.8)
            ax.plot(t, sc_mean, "-", color=METHOD_COLORS["simple_conv"],
                    lw=1.0, alpha=0.9, label="SimpleConv (mean)")
            ax.plot(t, sc[best_idx], ":", color=METHOD_COLORS["simple_conv"],
                    lw=1.8, alpha=0.95, label="SimpleConv (best)")

        ax.set_ylabel("RPS [rev/s]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.25)

    # Find specific recordings
    by_file = {r["file"]: r for r in results}

    # --- Set 1: Motor1 at 50, 70, 90 ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.3), sharey=True)
    for ax, fname in zip(axes, ["Motor1_50.wav", "Motor1_70.wav", "Motor1_90.wav"]):
        if fname in by_file:
            _plot_trace_panel(ax, by_file[fname])
            ax.set_title(f"{fname.replace('.wav', '')}  (target={by_file[fname]['target_rps']} rev/s)")
            ax.set_xlabel("Time [s]")
    axes[0].set_ylim(0, 120)
    axes[0].legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_single_rotor_set1.pdf")
    plt.close(fig)
    print("Saved fig_single_rotor_set1.pdf")

    # --- Set 2: Motor4 at 50, 70, 90 ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.3), sharey=True)
    for ax, fname in zip(axes, ["Motor4_50.wav", "Motor4_70.wav", "Motor4_90.wav"]):
        if fname in by_file:
            _plot_trace_panel(ax, by_file[fname])
            ax.set_title(f"{fname.replace('.wav', '')}  (target={by_file[fname]['target_rps']} rev/s)")
            ax.set_xlabel("Time [s]")
    axes[0].set_ylim(0, 120)
    axes[0].legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_single_rotor_set2.pdf")
    plt.close(fig)
    print("Saved fig_single_rotor_set2.pdf")

    # --- allMotors_70 ---
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    fname = "allMotors_70.wav"
    if fname in by_file:
        _plot_trace_panel(ax, by_file[fname])
        ax.set_title(f"{fname.replace('.wav', '')}  (target={by_file[fname]['target_rps']} rev/s)")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(0, 120)
        ax.legend(frameon=False, loc="upper left", fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_single_rotor_allmotors.pdf")
    plt.close(fig)
    print("Saved fig_single_rotor_allmotors.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
