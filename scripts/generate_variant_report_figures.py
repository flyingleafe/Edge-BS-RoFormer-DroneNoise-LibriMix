#!/usr/bin/env python3
"""Generate all figures for the SimpleConv variants report."""

import json
import numpy as np
from pathlib import Path
import scipy.io
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Config ─────────────────────────────────────────────────────────────────
FIG_DIR = Path("papers/simpleconv_variants_report/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROTOR_LABELS = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]

# ─── Data ───────────────────────────────────────────────────────────────────
# Validation metrics from the sweep
VAL_DATA = {
    "simple_conv_v2":            {"mse": 2.61, "rmse": 1.61, "mae": 0.76, "r2": 0.951, "params": 1.50e6},
    "simple_conv_bigru_v2":      {"mse": 2.67, "rmse": 1.64, "mae": 0.78, "r2": 0.948, "params": 1.44e6},
    "simple_conv_bigru":         {"mse": 2.74, "rmse": 1.66, "mae": 0.80, "r2": 0.945, "params": 0.663e6},
    "simple_conv_tcn":           {"mse": 3.09, "rmse": 1.76, "mae": 0.83, "r2": 0.936, "params": 1.38e6},
    "simple_conv_magphase_bigru":{"mse": 3.16, "rmse": 1.78, "mae": 0.96, "r2": 0.917, "params": 0.666e6},
    "simple_conv_attn_pool":     {"mse": 4.87, "rmse": 2.21, "mae": 1.25, "r2": 0.860, "params": 0.563e6},
    "simple_conv_wide":          {"mse": 5.04, "rmse": 2.24, "mae": 1.32, "r2": 0.847, "params": 3.94e6},
    "simple_conv_multiscale":    {"mse": 5.15, "rmse": 2.27, "mae": 1.31, "r2": 0.840, "params": 1.36e6},
    "simple_conv":               {"mse": 5.21, "rmse": 2.28, "mae": 1.36, "r2": 0.837, "params": 0.538e6},
    "simple_conv_se_next":       {"mse": 7.30, "rmse": 2.70, "mae": 1.86, "r2": 0.688, "params": 1.41e6},
}

SHORT_NAMES = {
    "simple_conv": "Baseline",
    "simple_conv_bigru": "BiGRU",
    "simple_conv_bigru_v2": "BiGRU-v2",
    "simple_conv_v2": "v2 (SE+Attn)",
    "simple_conv_tcn": "TCN",
    "simple_conv_magphase_bigru": "MagPhase",
    "simple_conv_attn_pool": "AttnPool",
    "simple_conv_wide": "Wide",
    "simple_conv_multiscale": "MultiScale",
    "simple_conv_se_next": "SE-Next",
}

# Directory names (without "baseline" suffix)
DIR_NAMES = {
    "simple_conv": "simple_conv",
    "simple_conv_bigru": "simple_conv_bigru",
    "simple_conv_bigru_v2": "simple_conv_bigru_v2",
    "simple_conv_v2": "simple_conv_v2",
    "simple_conv_tcn": "simple_conv_tcn",
    "simple_conv_magphase_bigru": "simple_conv_magphase_bigru",
    "simple_conv_attn_pool": "simple_conv_attn_pool",
    "simple_conv_wide": "simple_conv_wide",
    "simple_conv_multiscale": "simple_conv_multiscale",
    "simple_conv_se_next": "simple_conv_se_next",
}

ORDER = [
    "simple_conv_v2",
    "simple_conv_bigru_v2",
    "simple_conv_bigru",
    "simple_conv_tcn",
    "simple_conv_magphase_bigru",
    "simple_conv_attn_pool",
    "simple_conv_wide",
    "simple_conv_multiscale",
    "simple_conv",
    "simple_conv_se_next",
]

# ─── Fig 1: Validation leaderboard ──────────────────────────────────────────
def fig_validation_leaderboard():
    models = [SHORT_NAMES[m] for m in ORDER]
    mses = [VAL_DATA[m]["mse"] for m in ORDER]
    r2s = [VAL_DATA[m]["r2"] for m in ORDER]
    params = [VAL_DATA[m]["params"] / 1e6 for m in ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))

    # MSE
    ax = axes[0]
    bars = ax.barh(range(len(models)), mses, color=colors, edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Validation MSE [(rev/s)²] (lower is better)")
    ax.set_title("(a) MSE on held-out synthetic mixtures")
    ax.set_xlim(0, max(mses) * 1.15)
    for i, (bar, v) in enumerate(zip(bars, mses)):
        ax.text(v + 0.15, bar.get_y() + bar.get_height()/2, f"{v:.2f}",
                va="center", ha="left", fontsize=8)

    # R²
    ax = axes[1]
    bars = ax.barh(range(len(models)), r2s, color=colors, edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Validation R² (higher is better)")
    ax.set_title("(b) Coefficient of determination")
    ax.set_xlim(0.5, 1.0)
    for i, (bar, v) in enumerate(zip(bars, r2s)):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                va="center", ha="left", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_leaderboard_validation.pdf")
    fig.savefig(FIG_DIR / "fig_leaderboard_validation.png", dpi=200)
    plt.close(fig)
    print("Saved fig_leaderboard_validation")


# ─── Fig 2: Pareto plot (params vs R²) ──────────────────────────────────────
def fig_pareto():
    fig, ax = plt.subplots(figsize=(6, 4.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ORDER)))
    for i, m in enumerate(ORDER):
        ax.scatter(VAL_DATA[m]["params"]/1e6, VAL_DATA[m]["r2"],
                   s=120, c=[colors[i]], edgecolors="k", linewidths=0.5, zorder=5)
        ax.annotate(SHORT_NAMES[m], (VAL_DATA[m]["params"]/1e6, VAL_DATA[m]["r2"]),
                    textcoords="offset points", xytext=(6, 0), fontsize=7.5,
                    ha="left", va="center")

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Validation R²")
    ax.set_title("Pareto frontier: parameter count vs. tracking accuracy")
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_pareto_params_r2.pdf")
    fig.savefig(FIG_DIR / "fig_pareto_params_r2.png", dpi=200)
    plt.close(fig)
    print("Saved fig_pareto_params_r2")


# ─── Fig 3: Full-sequence comparison (top 5 variants) ─────────────────────
def fig_fullsequence_comparison():
    """Load full-sequence predictions for top variants and plot comparison."""
    variants = ["simple_conv", "simple_conv_bigru", "simple_conv_bigru_v2", "simple_conv_v2", "simple_conv_tcn"]
    labels = ["Baseline", "BiGRU", "BiGRU-v2", "v2 (SE+Attn)", "TCN"]

    # Load ground truth from one variant (same for all)
    gt = np.load("results/rps_eval_full_sequence/simple_conv/rps_gt_stft.npy")
    n_frames = gt.shape[1]
    duration = n_frames * 512 / 16000
    t = np.linspace(0, duration, n_frames)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.35})

    # Panel 1: GT + all predictions overlay
    ax = axes[0]
    for r in range(4):
        ax.plot(t, gt[r], ":", color=ROTOR_COLORS[r], lw=1.0, alpha=0.4)

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for v, lab, ls in zip(variants, labels, linestyles):
        pred = np.load(f"results/rps_eval_full_sequence/{DIR_NAMES[v]}/rps_pred.npy")
        # Plot mean across rotors for clarity
        pred_mean = pred.mean(axis=0)
        ax.plot(t, pred_mean, linestyle=ls, lw=1.2, alpha=0.85, label=lab)

    # Re-add GT lines for legend
    for r in range(4):
        ax.plot([], [], ":", color=ROTOR_COLORS[r], lw=1.0, alpha=0.6, label=f"GT R{r+1}")

    ax.legend(loc="lower center", ncol=5, frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_ylabel("Mean rotor speed [rev/s]")
    ax.set_xlim(0, duration)
    ax.set_xticklabels([])
    ax.set_title("Full-sequence predictions on DREGON speech-high room1 (~47 s)")

    # Panel 2: Per-frame MSE traces
    ax = axes[1]
    for v, lab, ls in zip(variants, labels, linestyles):
        pred = np.load(f"results/rps_eval_full_sequence/{DIR_NAMES[v]}/rps_pred.npy")
        mse_frame = ((pred - gt[:, :pred.shape[1]]) ** 2).mean(axis=0)
        # 1-s smoothing
        w = max(1, int(1.0 / (512/16000)))
        kernel = np.ones(w) / w
        mse_smooth = np.convolve(mse_frame, kernel, mode="same")
        ax.plot(t, mse_smooth, linestyle=ls, lw=1.0, alpha=0.85, label=lab)

    ax.axhline(5.15, color="#444", ls="--", lw=0.8, alpha=0.6, label="Synthetic val MSE = 5.15")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Per-frame MSE [(rev/s)²]")
    ax.set_xlim(0, duration)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, 1.02))
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_fullsequence_comparison.pdf")
    fig.savefig(FIG_DIR / "fig_fullsequence_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved fig_fullsequence_comparison")


# ─── Fig 4: Full-sequence per-variant 3-panel plots (top 3) ────────────────
def fig_fullsequence_3panel(variant: str, label: str):
    """Reproduce the paper's 3-panel figure for a single variant."""
    pred = np.load(f"results/rps_eval_full_sequence/{DIR_NAMES[variant]}/rps_pred.npy")
    gt = np.load(f"results/rps_eval_full_sequence/{DIR_NAMES[variant]}/rps_gt_stft.npy")
    mse_frame = np.load(f"results/rps_eval_full_sequence/{DIR_NAMES[variant]}/mse_per_frame.npy")

    # Load audio from simple_conv (same audio for all)
    # Load audio from cached .npz
    cache = np.load("data/DREGON/.cache/free-flight_speech-high_room1_16000hz_f11caf951813.npz")
    audio_full = cache["audio"][:, 0].astype(np.float32)
    # Match audio timestamps to motor range
    audio_ts = scipy.io.loadmat("data/DREGON/DREGON_free-flight_speech-high_room1/DREGON_free-flight_speech-high_room1_audiots.mat")["audio_timestamps"].flatten()
    motor_mat = scipy.io.loadmat("data/DREGON/DREGON_free-flight_speech-high_room1/DREGON_free-flight_speech-high_room1_motors.mat")
    motor_ts = motor_mat["motor"][0, 0]["timestamps"].flatten()
    t0, t1 = motor_ts[0], motor_ts[-1]
    sr = 16000
    audio_start = int((t0 - audio_ts[0]) * sr)
    audio_end = int((t1 - audio_ts[0]) * sr)
    audio_np = audio_full[audio_start:audio_end]

    n_frames = pred.shape[1]
    duration = len(audio_np) / sr
    t_stft = np.linspace(0, duration, n_frames)

    # Identify low-RPS regions
    low_rps = np.all(gt < 50, axis=0)
    transitions = np.diff(low_rps.astype(int))
    low_starts = np.where(transitions == 1)[0] + 1
    low_ends = np.where(transitions == -1)[0]
    if low_rps[0]:
        low_starts = np.r_[0, low_starts]
    if low_rps[-1]:
        low_ends = np.r_[low_ends, len(low_rps) - 1]

    fig, axes = plt.subplots(
        3, 1, figsize=(7.5, 6.2),
        gridspec_kw={"height_ratios": [1.2, 1.0, 0.8], "hspace": 0.35}
    )

    # Panel 1: spectrogram
    ax = axes[0]
    n_fft, hop = 2048, 512
    spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(audio_np, n_fft)[::hop] *
        np.hanning(n_fft), axis=-1))
    log_mag = np.log1p(spec.T)
    vmin = np.percentile(log_mag, 2)
    vmax = np.percentile(log_mag, 99)
    ax.imshow(
        log_mag, origin="lower", aspect="auto",
        extent=[0, duration, 0, 8],
        cmap="hot", vmin=vmin, vmax=vmax,
    )
    ax.set_ylim(0, 4)
    ax.set_ylabel("freq [kHz]")
    ax.set_title(f"{label} — DREGON free-flight speech-high room1")
    ax.set_xticklabels([])
    ax.grid(False)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)

    # Panel 2: rotor speeds
    ax = axes[1]
    for r in range(4):
        ax.plot(t_stft, gt[r], ":", color=ROTOR_COLORS[r], lw=0.6, alpha=0.5)
        ax.plot(t_stft, pred[r], "-", color=ROTOR_COLORS[r], lw=0.5, alpha=0.75)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)
    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=0.5, ls=":", alpha=0.5, label="GT"),
        plt.Line2D([0], [0], color="black", lw=0.5, ls="-", alpha=0.75, label="Pred"),
    ] + [plt.Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.0, label=ROTOR_LABELS[r]) for r in range(4)]
    ax.legend(handles=legend_handles, loc="lower center", frameon=False,
              fontsize=6.5, ncol=3, columnspacing=0.7,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_ylabel("rotor speed [rev/s]")
    ax.set_xlim(0, duration)
    ax.set_xticklabels([])

    # Panel 3: per-frame MSE
    ax = axes[2]
    mse_smooth = smooth(mse_frame, window_sec=1.0)
    for s, e in zip(low_starts, low_ends):
        ax.axvspan(t_stft[s], t_stft[e], color="gray", alpha=0.12, lw=0)
    ax.plot(t_stft, mse_smooth, "-", color="#d62728", lw=0.8)
    ax.fill_between(t_stft, mse_smooth, alpha=0.15, color="#d62728")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"MSE  $[(\mathrm{rev/s})^2]$")
    ax.set_xlim(0, duration)
    ax.axhline(5.15, ls="--", lw=0.8, color="#444", alpha=0.7,
               label="synthetic val = 5.15")
    ax.legend(frameon=False, loc="upper center", fontsize=7,
              bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_fullsequence_{variant}.pdf")
    fig.savefig(FIG_DIR / f"fig_fullsequence_{variant}.png", dpi=200)
    plt.close(fig)
    print(f"Saved fig_fullsequence_{variant}")


import scipy.io

def smooth(x, window_sec=1.0):
    frame_dur = 512 / 16000
    w = max(1, int(window_sec / frame_dur))
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


# ─── Fig 5: allMotors_70 trace comparison (classical_baselines style) ───────
def fig_allmotors_traces():
    variants = ["simple_conv", "simple_conv_bigru", "simple_conv_bigru_v2", "simple_conv_v2", "simple_conv_multiscale"]
    labels = ["Baseline", "BiGRU", "BiGRU-v2", "v2", "MultiScale"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 5), sharey=True)
    axes = axes.flatten()

    for idx, (v, lab, c) in enumerate(zip(variants, labels, colors)):
        ax = axes[idx]
        with open(f"results/rps_eval_single_rotor/{DIR_NAMES[v]}/metrics.json") as f:
            data = json.load(f)

        # Find allMotors_70 result
        allmotors = None
        for r in data["results"]:
            if r["motor_id"] == "all":
                allmotors = r
                break
        if allmotors is None:
            continue

        # We don't have the raw traces saved for single-rotor eval.
        # Load the prediction from the model directly for allMotors
        import torch, torchaudio, soundfile as sf
        from train_rps_predictor import get_model
        model = get_model(v, n_fft=2048, hop_length=512, num_rotors=4)
        ckpt_path = f"results/rps_exp_{DIR_NAMES[v].replace('simple_conv_', '') if v != 'simple_conv' else 'simple_conv'}/best_{DIR_NAMES[v]}.pt"
        # Adjust path for baseline
        if v == "simple_conv":
            ckpt_path = "results/rps_exp_simple_conv/best_simple_conv.pt"
        elif v == "simple_conv_bigru":
            ckpt_path = "results/rps_exp_simple_conv_bigru/best_simple_conv_bigru.pt"
        elif v == "simple_conv_bigru_v2":
            ckpt_path = "results/rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt"
        elif v == "simple_conv_v2":
            ckpt_path = "results/rps_exp_v2/best_simple_conv_v2.pt"
        elif v == "simple_conv_multiscale":
            ckpt_path = "results/rps_exp_multiscale/best_simple_conv_multiscale.pt"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        # Load audio
        audio, sr = sf.read("data/DREGON/DREGON_individual_motors_recordings/allMotors_70.wav")
        audio = audio[:, 0]  # ch0
        if sr != 16000:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),
                orig_freq=sr, new_freq=16000,
            ).numpy()[0]
        # Trim middle 40%
        n = len(audio)
        audio = audio[int(n*0.3):int(n*0.7)]
        x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]  # (4, T)

        t = np.arange(pred.shape[1]) * 512 / 16000
        target = 70.0

        ax.axhline(target, color="#333333", ls="--", lw=1.2, alpha=0.6, label="GT")
        sc_mean = pred.mean(axis=0)
        best_idx = allmotors["best_rotor"]["rotor_idx"]
        for r in range(4):
            ax.plot(t, pred[r], ":", color=c, lw=0.8, alpha=0.7)
        ax.plot(t, sc_mean, "-", color=c, lw=1.0, alpha=0.9, label=f"{lab} (mean)")
        ax.plot(t, pred[best_idx], ":", color=c, lw=1.8, alpha=0.95, label=f"{lab} (best)")

        ax.set_ylabel("RPS [rev/s]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"{lab} — allMotors_70 (best MSE={allmotors['best_rotor']['mse']:.1f})")
        ax.set_ylim(0, 120)
        ax.legend(frameon=False, loc="upper left", fontsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.25)

    # Hide last subplot
    axes[5].axis("off")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_single_rotor_allmotors_comparison.pdf")
    fig.savefig(FIG_DIR / "fig_single_rotor_allmotors_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved fig_single_rotor_allmotors_comparison")


# ─── Fig 6: Bar chart — allMotors_70 MSE comparison ─────────────────────────
def fig_allmotors_bar():
    variants = list(SHORT_NAMES.keys())
    labels = [SHORT_NAMES[v] for v in variants]

    best_mses = []
    avg_mses = []
    for v in variants:
        with open(f"results/rps_eval_single_rotor/{DIR_NAMES[v]}/metrics.json") as f:
            data = json.load(f)
        for r in data["results"]:
            if r["motor_id"] == "all":
                best_mses.append(r["best_rotor"]["mse"])
                avg_mses.append(r["avg"]["mse"])
                break

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(variants))
    w = 0.35
    bars1 = ax.bar(x - w/2, best_mses, w, label="Best channel", color="#1f77b4", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, avg_mses, w, label="Mean over 4 channels", color="#ff7f0e", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MSE [(rev/s)²]")
    ax.set_title("allMotors_70 — synchronized 4-rotor recording at 70 rev/s")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_ylim(0, max(avg_mses) * 1.15)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_allmotors_mse_bar.pdf")
    fig.savefig(FIG_DIR / "fig_allmotors_mse_bar.png", dpi=200)
    plt.close(fig)
    print("Saved fig_allmotors_mse_bar")


# ─── Fig 7: Full-sequence metrics bar chart ─────────────────────────────────
def fig_fullsequence_bar():
    variants = list(SHORT_NAMES.keys())
    labels = [SHORT_NAMES[v] for v in variants]

    inflight_mses = []
    for v in variants:
        with open(f"results/rps_eval_full_sequence/{DIR_NAMES[v]}/metrics.json") as f:
            m = json.load(f)
        inflight_mses.append(m.get("mse_inflight", 0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(variants)))
    bars = ax.bar(range(len(variants)), inflight_mses, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("In-flight MSE [(rev/s)²]")
    ax.set_title("Full-sequence evaluation: in-flight MSE (DREGON speech-high room1)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_ylim(0, max(inflight_mses) * 1.15)
    for bar, v in zip(bars, inflight_mses):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_fullsequence_inflight_mse_bar.pdf")
    fig.savefig(FIG_DIR / "fig_fullsequence_inflight_mse_bar.png", dpi=200)
    plt.close(fig)
    print("Saved fig_fullsequence_inflight_mse_bar")


# ─── Fig 8: Individual motor bar chart (classical style) ────────────────────
def fig_individual_motor_bar():
    """Bar chart of MSE on individual motor recordings, grouped by motor/speed."""
    variants = ["simple_conv", "simple_conv_bigru", "simple_conv_bigru_v2", "simple_conv_v2", "simple_conv_tcn"]
    labels = ["Baseline", "BiGRU", "BiGRU-v2", "v2", "TCN"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Collect best-rotor MSE per file for each variant
    file_names = []
    all_data = {v: [] for v in variants}

    for v in variants:
        with open(f"results/rps_eval_single_rotor/{DIR_NAMES[v]}/metrics.json") as f:
            data = json.load(f)
        files = []
        vals = []
        for r in data["results"]:
            if r["motor_id"] != "all":
                files.append(r["file"].replace(".wav", ""))
                vals.append(r["best_rotor"]["mse"])
        all_data[v] = (files, vals)
        if not file_names:
            file_names = files

    fig, ax = plt.subplots(figsize=(14, 4.5))
    x = np.arange(len(file_names))
    w = 0.15
    for i, (v, lab, c) in enumerate(zip(variants, labels, colors)):
        _, vals = all_data[v]
        ax.bar(x + (i - 2) * w, vals, w, label=lab, color=c, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(file_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Best-channel MSE [(rev/s)²]")
    ax.set_title("Individual motor recordings — best channel MSE (all variants fail, as expected)")
    ax.legend(frameon=False, ncol=5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_ylim(0, 9000)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_individual_motor_mse_bar.pdf")
    fig.savefig(FIG_DIR / "fig_individual_motor_mse_bar.png", dpi=200)
    plt.close(fig)
    print("Saved fig_individual_motor_mse_bar")


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_validation_leaderboard()
    fig_pareto()
    fig_fullsequence_comparison()
    for v, lab in [("simple_conv", "Baseline"), ("simple_conv_bigru", "BiGRU"),
                   ("simple_conv_bigru_v2", "BiGRU-v2")]:
        fig_fullsequence_3panel(v, lab)
    fig_allmotors_traces()
    fig_allmotors_bar()
    fig_fullsequence_bar()
    fig_individual_motor_bar()
    print("\nAll figures generated.")
