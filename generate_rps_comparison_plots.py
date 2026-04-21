#!/usr/bin/env python3
"""
Regenerate RPS comparison plots for slides (rps_comparison assets).

Mirrors the `compare_all_models` function from
`notebooks/rps_evaluation_interactive.ipynb`: motor RPS ground truth is
linearly interpolated to the prediction's STFT frame count instead of
cropping by min length.

Inputs (local):
    results/rps_eval_specific_samples/
        evaluation_results.json
        sample_XXXXX/{mixture.wav,ground_truth_rps.npy,simple_conv_rps.npy,dcunet_rps.npy,dccrn_rps.npy}

Outputs:
    slides/2026-04-14/assets/rps_comparison/
        sample_XXXXX_plot.png       # per-sample compare_all_models figure
        summary_metrics.png         # bar chart
        rps_timeseries.png          # time-series overlay from first sample
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


DATA_DIR = Path("results/rps_eval_specific_samples")
OUT_DIR = Path("slides/2026-04-14/assets/rps_comparison")
MODEL_NAMES = ["simple_conv", "dcunet", "dccrn"]
MODEL_LABELS = {"simple_conv": "SimpleConv", "dcunet": "DCUNet", "dccrn": "DCCRN"}
ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512


def resample_rps_to_length(rps_gt: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation along time axis to match a given target length.

    Same logic as `DREGONRPSDataset.__getitem__` in `train_rps_predictor.py`
    and `resample_rps_to_length` in the evaluation notebook.
    """
    x = torch.from_numpy(np.asarray(rps_gt, dtype=np.float32)).unsqueeze(0)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.squeeze(0).numpy()


def load_sample(sample_id: str):
    sdir = DATA_DIR / sample_id
    mixture, sr = torchaudio.load(sdir / "mixture.wav")
    assert sr == SAMPLE_RATE, f"Unexpected sample rate: {sr}"
    gt_raw = np.load(sdir / "ground_truth_rps.npy")
    preds = {}
    for m in MODEL_NAMES:
        path = sdir / f"{m}_rps.npy"
        if path.exists():
            preds[m] = np.load(path)
    return mixture.squeeze(0).numpy(), gt_raw, preds


def plot_sample_comparison(sample_id: str, out_path: Path) -> dict:
    """Generate the compare_all_models-style figure and return per-model MAE."""
    mixture_np, gt_raw, preds = load_sample(sample_id)
    duration = mixture_np.shape[0] / SAMPLE_RATE
    n_frames_stft = mixture_np.shape[0] // HOP_LENGTH + 1

    # Ground truth on the STFT grid (matches training targets)
    gt_stft = resample_rps_to_length(gt_raw, n_frames_stft)
    t_stft = np.linspace(0, duration, n_frames_stft)

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.2, 1, 1, 1, 1], hspace=0.3)

    ax_spec = fig.add_subplot(gs[0])
    window = torch.hann_window(N_FFT)
    X = torch.stft(
        torch.from_numpy(mixture_np).float(),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
        normalized=True,
    )
    Sxx = torch.abs(X).numpy()
    spec_times = np.linspace(0, duration, Sxx.shape[1])
    freqs = np.linspace(0, SAMPLE_RATE / 2, Sxx.shape[0])
    im = ax_spec.pcolormesh(
        spec_times, freqs, 20 * np.log10(Sxx + 1e-8), shading="auto", cmap="magma"
    )
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=11)
    ax_spec.set_title(
        f"{sample_id} — Input Spectrogram", fontsize=13, fontweight="bold"
    )
    ax_spec.set_ylim(0, 4000)
    plt.colorbar(im, ax=ax_spec, label="Magnitude (dB)", pad=0.01)

    ax_gt = fig.add_subplot(gs[1], sharex=ax_spec)
    for i, color in enumerate(ROTOR_COLORS):
        ax_gt.plot(t_stft, gt_stft[i], label=f"Rotor {i+1}", color=color, linewidth=2)
    ax_gt.set_ylabel("RPS", fontsize=11)
    ax_gt.set_title("Ground Truth Motor Speeds", fontsize=13, fontweight="bold")
    ax_gt.legend(loc="upper right", ncol=4, fontsize=9)
    ax_gt.grid(True, alpha=0.3)

    mae_per_model: dict[str, float] = {}
    for idx, model_name in enumerate(MODEL_NAMES):
        ax = fig.add_subplot(gs[2 + idx], sharex=ax_spec)
        label = MODEL_LABELS[model_name]

        if model_name in preds:
            pred = preds[model_name]
            gt_matched = resample_rps_to_length(gt_raw, pred.shape[1])
            t_pred = np.linspace(0, duration, pred.shape[1])
            for i, color in enumerate(ROTOR_COLORS):
                ax.plot(
                    t_pred,
                    gt_matched[i],
                    color=color,
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.4,
                )
                ax.plot(t_pred, pred[i], color=color, linewidth=2)
            mae = float(np.mean(np.abs(pred - gt_matched)))
            mae_per_model[model_name] = mae
            ax.set_title(
                f"{label} Prediction (MAE={mae:.2f})",
                fontsize=13,
                fontweight="bold",
            )
        else:
            ax.set_title(f"{label} — Not Available", fontsize=13, fontweight="bold")

        ax.set_ylabel("RPS", fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx == len(MODEL_NAMES) - 1:
            ax.set_xlabel("Time (s)", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mae_per_model


def compute_metrics_interpolated(sample_ids: list[str]) -> dict:
    """Recompute RMSE/MAE/R² per sample with GT linearly interpolated to pred length.

    The original `evaluation_results.json` cropped GT to `min_len` samples, which
    truncated the 8.2 s motor trace down to ~0.28 s. This function computes the
    metrics over the full clip (as training does).
    """
    per_sample: dict[str, dict[str, dict[str, float]]] = {}
    for sid in sample_ids:
        sdir = DATA_DIR / sid
        gt_raw = np.load(sdir / "ground_truth_rps.npy")
        metrics: dict[str, dict[str, float]] = {}
        for m in MODEL_NAMES:
            pred_path = sdir / f"{m}_rps.npy"
            if not pred_path.exists():
                continue
            pred = np.load(pred_path)
            gt = resample_rps_to_length(gt_raw, pred.shape[1])
            diff = pred - gt
            mse = float(np.mean(diff**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(diff)))
            ss_res = float(np.sum(diff**2))
            ss_tot = float(np.sum((gt - gt.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics[m] = {"rmse": rmse, "mae": mae, "r2": r2}
        per_sample[sid] = metrics
    return per_sample


def plot_summary_metrics(per_sample_metrics: dict, out_path: Path) -> dict:
    """Bar chart of average RMSE/MAE/R² across samples for each model.

    Returns a dict of aggregated {model: {rmse_mean, rmse_std, ...}} for reuse.
    """
    models = MODEL_NAMES
    sample_ids = list(per_sample_metrics.keys())

    rmse = {m: [per_sample_metrics[s][m]["rmse"] for s in sample_ids] for m in models}
    mae = {m: [per_sample_metrics[s][m]["mae"] for s in sample_ids] for m in models}
    r2 = {m: [per_sample_metrics[s][m]["r2"] for s in sample_ids] for m in models}

    def stats(xs: list[float]) -> tuple[float, float]:
        return float(np.mean(xs)), float(np.std(xs))

    labels = [MODEL_LABELS[m] for m in models]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name, data, ylabel, title in [
        (axes[0], "RMSE", rmse, "RMSE", "Root Mean Square Error\n(Lower is Better)"),
        (axes[1], "MAE", mae, "MAE", "Mean Absolute Error\n(Lower is Better)"),
        (axes[2], "R2", r2, "R²", "Coefficient of Determination\n(Higher is Better)"),
    ]:
        means = [stats(data[m])[0] for m in models]
        stds = [stats(data[m])[1] for m in models]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(means) + max(stds)) * 0.02,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        m: {
            "rmse_mean": float(np.mean(rmse[m])),
            "rmse_std": float(np.std(rmse[m])),
            "mae_mean": float(np.mean(mae[m])),
            "mae_std": float(np.std(mae[m])),
            "r2_mean": float(np.mean(r2[m])),
            "r2_std": float(np.std(r2[m])),
        }
        for m in models
    }


def plot_rps_timeseries(sample_id: str, out_path: Path) -> None:
    """Per-rotor overlay plot comparing GT and all model predictions."""
    mixture_np, gt_raw, preds = load_sample(sample_id)
    duration = mixture_np.shape[0] / SAMPLE_RATE

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    linestyles = {"simple_conv": "--", "dcunet": "-.", "dccrn": ":"}

    for i, (ax, color) in enumerate(zip(axes, ROTOR_COLORS)):
        ref_len = next(iter(preds.values())).shape[1] if preds else gt_raw.shape[1]
        t = np.linspace(0, duration, ref_len)
        gt_matched = resample_rps_to_length(gt_raw, ref_len)
        ax.plot(
            t,
            gt_matched[i],
            label="Ground Truth",
            color=color,
            linewidth=2.5,
            alpha=0.9,
        )
        for m in MODEL_NAMES:
            if m not in preds:
                continue
            pred = preds[m]
            if pred.shape[1] != ref_len:
                # Align predictions in the (rare) case of shape mismatch.
                pred = resample_rps_to_length(pred, ref_len)
            ax.plot(
                t,
                pred[i],
                label=MODEL_LABELS[m],
                color=color,
                linewidth=1.5,
                alpha=0.7,
                linestyle=linestyles[m],
            )
        ax.set_ylabel(f"Rotor {i+1}\n(RPS)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(
        f"RPS Prediction Comparison — {sample_id}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "evaluation_results.json") as f:
        eval_results = json.load(f)

    sample_ids = eval_results["sample_ids"]

    print("=== Per-sample plots (compare_all_models style) ===")
    per_sample_maes: dict[str, dict[str, float]] = {}
    for sid in sample_ids:
        out = OUT_DIR / f"{sid}_plot.png"
        maes = plot_sample_comparison(sid, out)
        per_sample_maes[sid] = maes
        print(f"  {sid}: {out.name}  MAE={maes}")

    print("\n=== Recomputing summary metrics with interpolation ===")
    per_sample_metrics = compute_metrics_interpolated(sample_ids)
    (DATA_DIR / "evaluation_results_interpolated.json").write_text(
        json.dumps(
            {
                "num_samples": len(sample_ids),
                "sample_ids": sample_ids,
                "method": "linear interpolation of GT to prediction length",
                "results": [
                    {"sample_id": sid, "metrics": per_sample_metrics[sid]}
                    for sid in sample_ids
                ],
            },
            indent=2,
        )
    )

    print("\n=== Summary bar chart ===")
    aggregated = plot_summary_metrics(
        per_sample_metrics, OUT_DIR / "summary_metrics.png"
    )
    for m, stats in aggregated.items():
        print(
            f"  {MODEL_LABELS[m]:<12} RMSE={stats['rmse_mean']:.2f}±{stats['rmse_std']:.2f}  "
            f"MAE={stats['mae_mean']:.2f}±{stats['mae_std']:.2f}  "
            f"R²={stats['r2_mean']:.2f}±{stats['r2_std']:.2f}"
        )
    print(f"  wrote {OUT_DIR / 'summary_metrics.png'}")

    print("\n=== Time-series overlay ===")
    plot_rps_timeseries(sample_ids[0], OUT_DIR / "rps_timeseries.png")
    print(f"  wrote {OUT_DIR / 'rps_timeseries.png'}")


if __name__ == "__main__":
    main()
