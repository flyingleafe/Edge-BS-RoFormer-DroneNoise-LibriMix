#!/usr/bin/env python3
"""Generate all figures for the Channel Generalization presentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from tasks.rps_prediction import HOP, N_FFT, SR_AUDIO, load_input_set, load_predictor
from train_rps_predictor import _ROTOR_PERMS, pit_mse_loss
from utils.plots.rps_prediction.sample_comparison import plot_sample_comparison

FIG_DIR = Path(__file__).resolve().parent / "public"
FIG_DIR.mkdir(exist_ok=True)


def _pit_permute(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Apply PIT permutation to align pred with gt."""
    F = min(pred.shape[1], gt.shape[1])
    pred = pred[:, :F]
    gt_ch = gt[:, :F]
    p_t = torch.from_numpy(np.asarray(pred, dtype=np.float32)).unsqueeze(0)
    g_t = torch.from_numpy(np.asarray(gt_ch, dtype=np.float32)).unsqueeze(0)
    _, best_idx = pit_mse_loss(p_t, g_t, perms=_ROTOR_PERMS, return_indices=True)
    best_perm = _ROTOR_PERMS[best_idx[0]].tolist()
    inv = [0] * 4
    for i, j in enumerate(best_perm):
        inv[j] = i
    return pred[inv]


def get_gt(sample) -> np.ndarray:
    """Extract GT RPS on STFT frame grid."""
    rps_es = sample["rps"]
    audio = np.asarray(sample["audio"].samples, dtype=np.float32)
    if audio.ndim == 1:
        n_frames = len(audio) // HOP + 1
    else:
        n_frames = audio.shape[1] // HOP + 1
    frame_times = np.arange(n_frames) * HOP / SR_AUDIO + rps_es.t_start + N_FFT / SR_AUDIO / 2
    return rps_es.interpolate(frame_times)


def make_ch0only_sc() -> None:
    """ch0-only SimpleConv on sample_00014 (nosource)."""
    predictor = load_predictor("simple_conv@results/rps_exp_simple_conv/best_simple_conv.pt")
    sample = next(
        s
        for s in load_input_set("datasets/DREGON-LM-V4/valid")
        if s.tags.get("id") == "sample_00014"
    )
    audio = np.asarray(sample["audio"].samples, dtype=np.float32)
    gt = get_gt(sample)

    preds = {}
    for ch in [0, 1]:
        pred = predictor.predict(audio[ch], sr=16000.0)
        preds[f"ch{ch}"] = _pit_permute(pred, gt)

    fig = plot_sample_comparison(
        sample=sample,
        channel=[0, 1],
        preds=preds,
        two_columns=True,
        figsize=(16, 6),
        show_separate_gt=False,
    )
    fig.suptitle("SimpleConv (ch0 only, PIT) — sample_00014", fontsize=12, y=1.02)
    out = FIG_DIR / "slide_ch0only_sc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def make_ch0only_v2() -> None:
    """ch0-only SimpleConvV2 on sample_00014 (nosource)."""
    predictor = load_predictor("simple_conv_v2@results/rps_exp_v2/best_simple_conv_v2.pt")
    sample = next(
        s
        for s in load_input_set("datasets/DREGON-LM-V4/valid")
        if s.tags.get("id") == "sample_00014"
    )
    audio = np.asarray(sample["audio"].samples, dtype=np.float32)
    gt = get_gt(sample)

    preds = {}
    for ch in [0, 3]:
        pred = predictor.predict(audio[ch], sr=16000.0)
        preds[f"ch{ch}"] = _pit_permute(pred, gt)

    fig = plot_sample_comparison(
        sample=sample,
        channel=[0, 3],
        preds=preds,
        two_columns=True,
        figsize=(16, 6),
        show_separate_gt=False,
    )
    fig.suptitle("SimpleConvV2 (ch0 only, PIT) — sample_00014", fontsize=12, y=1.02)
    out = FIG_DIR / "slide_ch0only_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def make_dynamic_8ch_v2() -> None:
    """8ch SimpleConvV2 with PIT on sample_00012 (dynamic, all 8 channels)."""
    predictor = load_predictor(
        "simple_conv_v2@results/rps_8ch_v4_simple_conv_v2/best_simple_conv_v2.pt"
    )
    sample = next(
        s
        for s in load_input_set("datasets/DREGON-LM-V4/valid")
        if s.tags.get("id") == "sample_00012"
    )
    audio = np.asarray(sample["audio"].samples, dtype=np.float32)
    gt = get_gt(sample)

    preds = {}
    for ch in range(8):
        pred = predictor.predict(audio[ch], sr=16000.0)
        preds[f"ch{ch}"] = _pit_permute(pred, gt)

    fig = plot_sample_comparison(
        sample=sample,
        channel=list(range(4)),
        preds=preds,
        two_columns=True,
        figsize=(16, 7),
        show_separate_gt=False,
    )
    fig.suptitle(
        "SimpleConvV2 (8ch, PIT) — sample_00012 (dynamic, 30→80 RPS)",
        fontsize=12,
        y=1.01,
    )
    out = FIG_DIR / "slide_dynamic_8ch_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def make_dynamic_8ch_sc() -> None:
    """8ch SimpleConv with PIT on sample_00012 (dynamic, all 8 channels)."""
    predictor = load_predictor("simple_conv@results/rps_8ch_v4_simple_conv/best_simple_conv.pt")
    sample = next(
        s
        for s in load_input_set("datasets/DREGON-LM-V4/valid")
        if s.tags.get("id") == "sample_00012"
    )
    audio = np.asarray(sample["audio"].samples, dtype=np.float32)
    gt = get_gt(sample)

    preds = {}
    for ch in range(8):
        pred = predictor.predict(audio[ch], sr=16000.0)
        preds[f"ch{ch}"] = _pit_permute(pred, gt)

    fig = plot_sample_comparison(
        sample=sample,
        channel=list(range(4)),
        preds=preds,
        two_columns=True,
        figsize=(16, 7),
        show_separate_gt=False,
    )
    fig.suptitle(
        "SimpleConv (8ch, PIT) — sample_00012 (dynamic, 30→80 RPS)",
        fontsize=12,
        y=1.01,
    )
    out = FIG_DIR / "slide_dynamic_8ch_sc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def make_tikz_images() -> None:
    """Compile TiKZ standalone figures from the LaTeX report."""
    report_dir = Path(__file__).resolve().parents[2] / "papers" / "simpleconv_variants_report"
    tex_files = [
        ("tikz_simpleconv_standalone.tex", "simpleconv_tikz.png"),
        ("tikz_simpleconv_v2_standalone.tex", "simpleconv_v2_tikz.png"),
    ]
    for tex_name, png_name in tex_files:
        tex_path = report_dir / tex_name
        if not tex_path.exists():
            print(f"  WARNING: {tex_path} not found, skipping")
            continue
        # Compile to PDF
        pdf_path = report_dir / tex_name.replace(".tex", ".pdf")
        if not pdf_path.exists():
            import subprocess

            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", str(tex_name)],
                cwd=report_dir,
                capture_output=True,
            )
        # Convert to PNG
        png_path = FIG_DIR / png_name
        import subprocess

        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                "300",
                "-singlefile",
                str(tex_name.replace(".tex", ".pdf")),
                str(png_path.with_suffix("")),
            ],
            cwd=report_dir,
            capture_output=True,
        )
        # pdftoppm outputs to report_dir; move to FIG_DIR
        src = report_dir / (tex_name.replace(".tex", ".png"))
        if src.exists():
            src.rename(png_path)
        print(f"  → {png_path}")


def main():
    print("Generating presentation figures...")
    make_tikz_images()
    make_ch0only_sc()
    make_ch0only_v2()
    make_dynamic_8ch_v2()
    make_dynamic_8ch_sc()
    print("Done.")


if __name__ == "__main__":
    main()
