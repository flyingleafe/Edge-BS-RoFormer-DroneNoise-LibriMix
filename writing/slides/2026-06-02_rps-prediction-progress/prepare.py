#!/usr/bin/env python3
"""Regenerate the sample-comparison figures for the RPS-prediction-progress deck.

Consolidates the deck's original figure-generation scripts into a single entry
point:

    generate_sample_comparison.py     -> assets/sample_comparison.png
    generate_sample_comparison_v3.py  -> assets/sample_comparison_v3.png
    generate_v2_sample_comparison.py  -> assets/sample_comparison_v2_old.png
                                         assets/sample_comparison_v2_v3.png
    generate_v4_sample_comparison.py  -> assets/sample_comparison_v4.png
    find_worst_4motor_v2.py           -> diagnostic printout (no figure)

DATA DEPENDENCY: these figures read model checkpoints from ``results/`` and audio
from ``datasets/``. Sync those first, e.g.::

    ./scripts/sync_results.sh        # legacy rsync fallback
    # or: dload pull <name>          # datasets (see dload.lock for names)

Run from this deck directory with the repo root on PYTHONPATH::

    PYTHONPATH=<repo-root> python3 prepare.py

NOTE: six figures referenced by the slides had no committed generator script and
are NOT produced here: classical_vs_neural.png, leaderboard.png, pareto.png,
cross_eval.png, degradation.png, pit_gap.png.
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from models.rps_predictor import SimpleConv, SimpleConvBiGRUV2
from utils.paths import get_datasets_path, get_results_path

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FFT = 2048
HOP_LENGTH = 512
NUM_ROTORS = 4
SR = 16000
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


# ─── Shared helpers ─────────────────────────────────────────────────────────
def _load_models(sc_ckpt, bg_ckpt):
    """Instantiate SimpleConv + BiGRU-v2 and load their state dicts."""
    models = {
        "sc": SimpleConv(N_FFT, HOP_LENGTH, NUM_ROTORS).to(DEVICE),
        "bg": SimpleConvBiGRUV2(N_FFT, HOP_LENGTH, NUM_ROTORS).to(DEVICE),
    }
    for key, ckpt_path in (("sc", sc_ckpt), ("bg", bg_ckpt)):
        state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
        models[key].load_state_dict(state)
        models[key].eval()
        print(f"Loaded {key} from {ckpt_path}")
    return models


def _infer(model, audio):
    """Run a model on a mono audio tensor, returning (4, T) numpy predictions."""
    with torch.no_grad():
        pred = model(audio.unsqueeze(0).to(DEVICE))
    return pred.squeeze(0).cpu().numpy()


def _plot_3panel(audio, rps_gt, time, pred_sc, pred_bg, sc_title, bg_title, out_path):
    """Spectrogram + SimpleConv RPS + BiGRU-v2 RPS, each vs ground truth."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("white")

    # -- Panel 1: Spectrogram --
    window = torch.hann_window(N_FFT)
    X = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
        normalized=True,
    )
    Sxx = torch.abs(X).numpy()

    ax1 = axes[0]
    ax1.imshow(
        20 * np.log10(Sxx + 1e-10),
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[time[0], time[-1], 0, SR // 2],
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Noisy Mixture Spectrogram")
    ax1.set_ylim(0, 4000)

    # -- Panel 2: SimpleConv RPS vs GT --
    ax2 = axes[1]
    for r in range(NUM_ROTORS):
        ax2.plot(
            time,
            rps_gt[r].numpy(),
            color=COLORS[r],
            linestyle="--",
            alpha=0.7,
            label=f"GT R{r + 1}" if r == 0 else "",
        )
        ax2.plot(
            time,
            pred_sc[r],
            color=COLORS[r],
            linestyle="-",
            label=f"Pred R{r + 1}" if r == 0 else "",
        )
    ax2.set_ylabel("RPS (Hz)")
    ax2.set_title(sc_title)
    ax2.legend(loc="upper right", ncol=2, fontsize=8)

    # -- Panel 3: BiGRU-v2 RPS vs GT --
    ax3 = axes[2]
    for r in range(NUM_ROTORS):
        ax3.plot(time, rps_gt[r].numpy(), color=COLORS[r], linestyle="--", alpha=0.7)
        ax3.plot(time, pred_bg[r], color=COLORS[r], linestyle="-")
    ax3.set_ylabel("RPS (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title(bg_title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def _load_valid_sample(sample_dir):
    """Load a DREGON-LM/valid sample: mono audio + RPS resampled to STFT frames."""
    audio, sr = torchaudio.load(str(sample_dir / "mixture.wav"))
    audio = audio[0]  # mono
    rps_gt = torch.from_numpy(np.load(sample_dir / "rps.npy")).float()  # (4, rps_T)
    n_frames = audio.shape[0] // HOP_LENGTH + 1
    rps_gt = F.interpolate(
        rps_gt.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
    ).squeeze(0)  # (4, n_frames)
    time = np.arange(n_frames) * HOP_LENGTH / sr
    return audio, rps_gt, time


# ─── Figure generators ──────────────────────────────────────────────────────
def sample_comparison(assets):
    """Old checkpoints on DREGON-LM/valid/sample_00004 (slide 3)."""
    sample_dir = get_datasets_path("DREGON-LM/valid") / "sample_00004"
    print(f"Using sample: {sample_dir}")
    audio, rps_gt, time = _load_valid_sample(sample_dir)
    models = _load_models(
        get_results_path("rps_exp_simple_conv/best_simple_conv.pt"),
        get_results_path("rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt"),
    )
    _plot_3panel(
        audio,
        rps_gt,
        time,
        _infer(models["sc"], audio),
        _infer(models["bg"], audio),
        "SimpleConv Predictions vs Ground Truth",
        "BiGRU-v2 Predictions vs Ground Truth",
        assets / "sample_comparison.png",
    )


def sample_comparison_v3(assets):
    """V3 checkpoints on DREGON-LM/valid/sample_00114 (slide 5, right)."""
    sample_dir = get_datasets_path("DREGON-LM/valid") / "sample_00114"
    print(f"Using sample: {sample_dir}")
    audio, rps_gt, time = _load_valid_sample(sample_dir)
    models = _load_models(
        get_results_path("rps_predictor_v3/simple_conv/best_simple_conv.pt"),
        get_results_path("rps_predictor_v3/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt"),
    )
    _plot_3panel(
        audio,
        rps_gt,
        time,
        _infer(models["sc"], audio),
        _infer(models["bg"], audio),
        "V3 SimpleConv Predictions vs Ground Truth",
        "V3 BiGRU-v2 Predictions vs Ground Truth",
        assets / "sample_comparison_v3.png",
    )


def v2_sample_comparison(assets):
    """Pre-computed old + V3 predictions on the V2 sample v2_sample_00558 (slide 6)."""
    sample_dir = get_results_path("rps_cross_eval/samples/v2_sample_00558")
    print(f"Using sample: {sample_dir}")
    audio, _ = torchaudio.load(str(sample_dir / "mixture.wav"))
    audio = audio[0]  # mono
    rps_gt = torch.from_numpy(np.load(sample_dir / "rps_target.npy"))  # (4, T)
    n_frames = rps_gt.shape[1]
    time = np.arange(n_frames) * HOP_LENGTH / SR

    preds = {
        key: np.load(sample_dir / f"rps_pred_{key}.npy")
        for key in ("old_simple_conv", "old_bigru_v2", "v3_simple_conv", "v3_bigru_v2")
    }

    _plot_3panel(
        audio,
        rps_gt,
        time,
        preds["old_simple_conv"],
        preds["old_bigru_v2"],
        "Old SimpleConv Predictions vs Ground Truth",
        "Old BiGRU-v2 Predictions vs Ground Truth",
        assets / "sample_comparison_v2_old.png",
    )
    _plot_3panel(
        audio,
        rps_gt,
        time,
        preds["v3_simple_conv"],
        preds["v3_bigru_v2"],
        "V3 SimpleConv Predictions vs Ground Truth",
        "V3 BiGRU-v2 Predictions vs Ground Truth",
        assets / "sample_comparison_v2_v3.png",
    )


def v4_sample_comparison(assets):
    """V4 (2.5% synth) checkpoints on v2_sample_00558.

    Not referenced by the current slides, but preserved from the original
    generate_v4_sample_comparison.py.
    """
    sample_dir = get_results_path("rps_cross_eval/samples/v2_sample_00558")
    print(f"Using sample: {sample_dir}")
    audio, _ = torchaudio.load(str(sample_dir / "mixture.wav"))
    audio = audio[0]
    rps_gt = torch.from_numpy(np.load(sample_dir / "rps_target.npy")).float()
    n_frames = rps_gt.shape[1]
    time = np.arange(n_frames) * HOP_LENGTH / SR

    models = _load_models(
        get_results_path("rps_predictor_v4_2.5pct/simple_conv/best_simple_conv.pt"),
        get_results_path(
            "rps_predictor_v4_2.5pct/simple_conv_bigru_v2/best_simple_conv_bigru_v2.pt"
        ),
    )
    _plot_3panel(
        audio,
        rps_gt,
        time,
        _infer(models["sc"], audio),
        _infer(models["bg"], audio),
        "V4 (2.5% synth) SimpleConv Predictions vs Ground Truth",
        "V4 (2.5% synth) BiGRU-v2 Predictions vs Ground Truth",
        assets / "sample_comparison_v4.png",
    )


def find_worst_4motor():
    """Diagnostic: rank V2 cross-eval samples where all 4 motors are active.

    Prints only (no figure); ported from find_worst_4motor_v2.py to help pick a
    representative sample. Not invoked by main().
    """
    base = get_results_path("rps_cross_eval/samples")
    results = []
    for sample_dir in sorted(base.glob("v2_sample_*")):
        with open(sample_dir / "metrics_old_simple_conv.json") as f:
            m_sc = json.load(f)["pit_mse"]
        with open(sample_dir / "metrics_old_bigru_v2.json") as f:
            m_bg = json.load(f)["pit_mse"]
        rps = np.load(sample_dir / "rps_target.npy")  # (4, T)
        n_active = sum(np.mean(rps[i]) > 5.0 for i in range(NUM_ROTORS))
        n_constant = sum(np.std(rps[i]) < 0.1 for i in range(NUM_ROTORS))
        results.append(
            {
                "sid": sample_dir.name,
                "sc_mse": m_sc,
                "bg_mse": m_bg,
                "mean_mse": (m_sc + m_bg) / 2,
                "n_active": n_active,
                "n_constant": n_constant,
            }
        )

    print("All V2 samples:")
    for r in results:
        flag = "CONSTANT" if r["n_constant"] > 0 else ""
        print(
            f"  {r['sid']}: active={r['n_active']}, constant={r['n_constant']}, "
            f"SC={r['sc_mse']:.1f}, BG={r['bg_mse']:.1f}, mean={r['mean_mse']:.1f} {flag}"
        )

    filtered = [r for r in results if r["n_active"] == 4 and r["n_constant"] == 0]
    print(f"\nSamples with 4 active motors (non-constant): {len(filtered)}")
    for r in sorted(filtered, key=lambda x: x["mean_mse"], reverse=True):
        print(f"  {r['sid']}: SC={r['sc_mse']:.1f}, BG={r['bg_mse']:.1f}, mean={r['mean_mse']:.1f}")


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    assets = pathlib.Path("assets")
    assets.mkdir(exist_ok=True)
    for generator in (
        sample_comparison,
        sample_comparison_v3,
        v2_sample_comparison,
        v4_sample_comparison,
    ):
        try:
            generator(assets)
        except Exception as exc:  # data-dependent; keep going for the rest
            print(f"WARNING: {generator.__name__} failed: {exc}")


if __name__ == "__main__":
    main()
