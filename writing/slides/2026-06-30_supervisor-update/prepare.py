#!/usr/bin/env python3
"""Generate / collect figures for the supervisor-update slide deck.

Three sections:
  1. RPS-prediction experiments (arch sweep + online/offline + causal) — reuse the
     figures from writing/reports/2026-06-19_rps-arch-sweep-v4-michaels/assets/.
  2. Drone-noise generative model — a schematic block diagram (drawn here) plus
     real-vs-generated spectrogram comparisons rendered from the trained
     PositionalHarmonicNoiseGen checkpoint (ports notebooks/noise_gen_real_vs_generated).
  3. Realistic RPS-trajectory synthesis — reuse the figures from
     writing/reports/2026-06-30_synthetic-rps-trajectories/assets/.

Run via `make figures` (sets PYTHONPATH to the repo root).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── bootstrap project paths (run from the repo root) ───────────────────────
SLIDE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SLIDE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ASSETS = SLIDE_DIR / "assets"
REPORTS = PROJECT_ROOT / "writing" / "reports"
ARCH_SWEEP = REPORTS / "2026-06-19_rps-arch-sweep-v4-michaels" / "assets"
RPS_TRAJ = REPORTS / "2026-06-30_synthetic-rps-trajectories" / "assets"

SR = 16000
DEVICE = "cpu"
CKPT = PROJECT_ROOT / "results/noise_gen_dregon_michaels_swapped/best_positional_harmonic_gen.pt"


# ===========================================================================
# Section 1 + 3 — reuse existing report figures
# ===========================================================================
def copy_reused() -> None:
    pairs = [
        # section 1 — RPS prediction / overfitting
        (ARCH_SWEEP / "fig_offline_leaderboard.png", "s1_offline_leaderboard.png"),
        (ARCH_SWEEP / "fig_online_leaderboard.png", "s1_online_leaderboard.png"),
        (ARCH_SWEEP / "fig_offline_vs_online.png", "s1_offline_vs_online.png"),
        # section 3 — realistic RPS trajectories
        (RPS_TRAJ / "rc_sticks.png", "s3_rc_sticks.png"),
        (RPS_TRAJ / "model_comparison.png", "s3_model_comparison.png"),
        (RPS_TRAJ / "intermittent_agg.png", "s3_intermittent_agg.png"),
        (RPS_TRAJ / "drone_profile_sweep.png", "s3_drone_profile_sweep.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, ASSETS / dst)
            print(f"  copied {src.name} -> {dst}")
        else:
            print(f"  WARNING missing {src} (run that report's `make figures` first)")


# ===========================================================================
# Section 2a — model architecture diagram (faithful recreation of the Stage-2
# report's Figure 3.1: "Diagram of the proposed drone noise synthesis model")
# ===========================================================================
# Colour palette matching the report figure.
_GREEN = "#cde6cd"  # Conv1d encoder blocks
_BLUE = "#cfe0f3"  # Linear + sigmoid heads
_YELLOW = "#fdebc0"  # upsampling
_PURPLE = "#dcdcf0"  # synthesiser / filter
_PINK = "#fbc4ab"  # loss parallelograms
_WAVE = "#1f5fbf"  # waveform plots


def _demo_motor_speeds(n: int = 80000, rng=None):
    """4 wandering rotor-speed curves with a mid-flight dip (illustration only)."""
    rng = np.random.default_rng(0) if rng is None else rng
    t = np.linspace(0, 1, n)
    base = np.array([80.0, 75.0, 82.0, 78.0])
    dip = 12.0 * np.exp(-((t - 0.5) ** 2) / (2 * 0.10**2))  # throttle-down in the middle
    out = []
    for b in base:
        wander = np.cumsum(rng.standard_normal(n)) * 0.06
        wander -= np.linspace(wander[0], wander[-1], n)
        out.append(b - dip + wander + rng.standard_normal(n) * 0.3)
    return t, np.array(out)


def _demo_noise_wave(n: int = 4000, rng=None):
    env = 0.6 + 0.4 * np.sin(np.linspace(0, 9, n)) ** 2
    rng = np.random.default_rng(1) if rng is None else rng
    return env * rng.standard_normal(n)


def fig_model_diagram() -> None:
    import matplotlib.patches as mp

    fig, ax = plt.subplots(figsize=(11.5, 14.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 20)
    ax.axis("off")
    ax.set_aspect("auto")

    def rbox(cx, cy, w, h, text, fc, fs=10.5):
        ax.add_patch(
            mp.FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                fc=fc,
                ec="#444",
                lw=1.2,
            )
        )
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def pbox(cx, cy, w, h, text, fc=_PINK, fs=9.5):
        sk = 0.32
        poly = np.array(
            [
                [cx - w / 2 + sk, cy - h / 2],
                [cx + w / 2 + sk, cy - h / 2],
                [cx + w / 2 - sk, cy + h / 2],
                [cx - w / 2 - sk, cy + h / 2],
            ]
        )
        ax.add_patch(mp.Polygon(poly, closed=True, fc=fc, ec="#a33", lw=1.1))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, dashed=False, color="#333", lw=1.6):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=lw,
                color=color,
                linestyle="--" if dashed else "-",
                shrinkA=0,
                shrinkB=0,
            ),
        )

    def label(x, y, text, fs=9.5):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#222")

    xc = 4.3  # encoder centre column
    xL, xR = 2.7, 6.2  # harmonic (left) / diffuse (right) branches

    # --- top: motor-speed plot --------------------------------------------
    label(
        xc,
        19.5,
        "Motor speeds (4 channels, linearly interpolated to\naudio sample rate (16 kHz))",
        10,
    )
    ax_top = ax.inset_axes((xc - 1.7, 17.7, 3.4, 1.45), transform=ax.transData)
    _t, ms = _demo_motor_speeds()
    for i, row in enumerate(ms):
        ax_top.plot(
            np.linspace(0, len(row), len(row)),
            row,
            lw=0.6,
            label=f"ch{i}",
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i],
        )
    ax_top.legend(fontsize=5, loc="upper right", ncol=1, handlelength=1, framealpha=0.6)
    ax_top.tick_params(labelsize=5)

    # --- encoder stack (green) --------------------------------------------
    enc = [
        "Causal Conv1d (32 channels, size 256, stride 32)\n+ BatchNorm + ReLU",
        "Causal Conv1d (64 channels, size 3, dilation 2)\n+ BatchNorm + ReLU",
        "Causal Conv1d (128 channels, size 3, dilation 4)\n+ BatchNorm + ReLU",
        "Causal Conv1d (256 channels, size 3, dilation 8)\n+ BatchNorm + ReLU",
        "Causal Conv1d (512 channels, size 3, dilation 8)\n+ BatchNorm + ReLU",
    ]
    ys = [16.2, 15.1, 14.0, 12.9, 11.8]
    arrow(xc, 17.6, xc, ys[0] + 0.5)
    for i, (txt, y) in enumerate(zip(enc, ys)):
        rbox(xc, y, 4.6, 0.85, txt, _GREEN, 9.3)
        if i:
            arrow(xc, ys[i - 1] - 0.45, xc, y + 0.45)

    # --- split into two Linear heads (blue) -------------------------------
    arrow(xc, ys[-1] - 0.45, xL, 10.45)
    arrow(xc, ys[-1] - 0.45, xR, 10.45)
    rbox(xL, 10.0, 2.7, 0.85, "Linear (size 400)\n+ sigmoid", _BLUE, 9.5)
    rbox(xR, 10.0, 2.7, 0.85, "Linear (size 60)\n+ sigmoid", _BLUE, 9.5)

    # --- harmonic (left) branch -------------------------------------------
    arrow(xL, 9.55, xL, 8.95)
    rbox(xL, 8.5, 2.9, 1.0, "Upsampling to 16kHz\nvia overlap-add with\nHann window", _YELLOW, 9.0)
    arrow(xL, 8.0, xL, 7.4)
    label(xL, 7.15, "4x harmonic distribution\namplitudes A$_i$(t, k)", 9)
    arrow(xL, 6.7, xL, 6.15)
    rbox(xL, 5.7, 2.5, 0.8, "Sinusoidal synthesizer", _PURPLE, 9.3)
    arrow(xL, 5.3, xL, 4.85)
    label(xL, 4.6, "4x harmonic\nnoise components", 9)

    # --- diffuse (right) branch -------------------------------------------
    arrow(xR, 9.55, xR, 7.6)
    label(xR, 7.3, "Diffuse noise\nspectral shape", 9)
    arrow(xR, 6.95, xR, 6.15)
    rbox(xR, 5.7, 2.6, 0.8, "Linear time-variant filter", _PURPLE, 9.0)
    rbox(9.6, 6.7, 2.4, 0.85, "White noise\n(randomly generated)", "#ffffff", 8.8)
    arrow(9.6, 6.27, xR + 0.7, 5.95)  # white noise -> filter
    arrow(xR, 5.3, xR, 4.85)
    label(xR, 4.6, "Diffuse noise\ncomponent", 9)

    # --- bypass: motor speeds -> sinusoidal synthesizer (fundamentals) ----
    ax.plot([0.7, 0.7], [5.7, 18.4], color="#333", lw=1.6)
    arrow(0.7, 18.4, xc - 1.75, 18.4)  # tap off the top plot
    ax.plot([0.7, 0.7], [18.4, 18.4], color="#333", lw=1.6)
    arrow(0.7, 5.7, xL - 1.27, 5.7)  # into synthesizer (left side)

    # --- sum and generated waveform ---------------------------------------
    sx, sy = (xL + xR) / 2, 3.7
    ax.add_patch(mp.Circle((sx, sy), 0.28, fc="white", ec="#333", lw=1.4))
    ax.text(sx, sy, "+", ha="center", va="center", fontsize=15)
    arrow(xL, 4.3, sx - 0.2, sy + 0.2)
    arrow(xR, 4.3, sx + 0.2, sy + 0.2)
    arrow(sx, sy - 0.3, sx, 3.0)
    ax_gen = ax.inset_axes((sx - 1.6, 1.5, 3.2, 1.4), transform=ax.transData)
    ax_gen.plot(_demo_noise_wave(), lw=0.4, color=_WAVE)
    ax_gen.axis("off")
    label(sx, 1.2, "Generated noise waveform", 9.5)

    # --- real noise recording (bottom right) ------------------------------
    ax_real = ax.inset_axes((11.0 - 1.6, 1.5, 3.2, 1.4), transform=ax.transData)
    ax_real.plot(_demo_noise_wave(rng=np.random.default_rng(7)), lw=0.4, color=_WAVE)
    ax_real.axis("off")
    label(11.0, 1.2, "Real noise recording", 9.5)

    # --- loss (pink) ------------------------------------------------------
    # Only the multi-resolution spectral loss is used by the current training
    # script (train_noise_generation.py: build_loss -> MultiScaleSTFT only). The
    # Stage-2 report's smoothness-loss terms are not part of the current run.
    pbox(7.6, 2.2, 3.2, 0.9, "Multi-resolution\nspectral loss")
    arrow(6.1, 2.2, sx + 1.65, 2.2, dashed=True, color="#a33", lw=1.3)  # -> generated wave
    arrow(9.1, 2.2, 11.0 - 1.65, 2.2, dashed=True, color="#a33", lw=1.3)  # -> real wave

    fig.savefig(ASSETS / "s2_model_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  drew s2_model_diagram.png (Figure-3.1 recreation)")


# ===========================================================================
# Section 2b — real vs generated spectrograms (from the trained checkpoint)
# ===========================================================================
def _logspec(x, n_fft=1024, hop=256):
    import torch

    X = torch.stft(
        torch.from_numpy(x).float(),
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    return 20 * np.log10(np.abs(X.numpy()) + 1e-6)


def render_spectrograms() -> None:
    try:
        import torch

        from data_processing.dregon import load_dregon_timeframes
        from data_processing.michaels import load_michaels_timeframes
        from data_processing.online_mixing import (
            _extract_audio_array,
            interpolate_rps_to_stft_grid,
        )
        from tasks.noise_generation import DroneCodebook, geometry_to_rel_pos
        from train_noise_generation import build_loss, get_model
    except Exception as e:  # pragma: no cover
        print(f"  WARNING: cannot import noise-gen stack ({e}); skipping spectrograms.")
        return
    if not CKPT.exists():
        print(f"  WARNING: checkpoint missing at {CKPT}; skipping spectrograms.")
        return

    bundle = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = get_model(
        "positional_harmonic_gen", sample_rate=SR, n_harmonics=100, cond_dim=bundle["cond_dim"]
    )
    model.load_state_dict(bundle["model"])
    model.to(DEVICE).eval()
    codebook = DroneCodebook(bundle["cond_dim"], names=bundle["drone_names"]).to(DEVICE)
    codebook.load_state_dict(bundle["codebook"])
    loss_fn = build_loss(
        argparse.Namespace(stft_sizes=[2048, 1024, 512, 256, 128], log_weight=1.0, loss_type="L1")
    ).to(DEVICE)

    recordings = {}
    for tf in load_dregon_timeframes(
        PROJECT_ROOT / "data", splits=["in_flight_noise"], target_sr=SR, download=False
    ):
        recordings[f"dregon:{tf.tags['recording_id']}"] = (tf, "dregon")
    for tf in load_michaels_timeframes(data_root=PROJECT_ROOT / "data", sr=SR):
        recordings[f"michaels:FLY{tf.tags['recording_id']}"] = (tf, "michaels")

    def render(rec_id, start_s, dur_s, mic):
        tf, drone = recordings[rec_id]
        t0 = tf["audio"].t_start
        sl = tf.slice(t0 + start_s, t0 + start_s + dur_s)
        n = int(round(dur_s * SR))
        target = _extract_audio_array(sl, target_len=n)
        rps = interpolate_rps_to_stft_grid(sl, n_frames=n, hop_length=1)
        gd = sl.global_data
        rel = geometry_to_rel_pos(gd["mic_positions"], gd["rotor_positions"])[: target.shape[0]]
        z = codebook([drone])
        with torch.no_grad():
            pred = (
                model(
                    torch.from_numpy(rps)[None].to(DEVICE),
                    torch.from_numpy(rel)[None].to(DEVICE),
                    z,
                )[0]
                .cpu()
                .numpy()
            )
            loss = loss_fn(
                torch.from_numpy(pred).to(DEVICE), torch.from_numpy(target).to(DEVICE)
            ).item()
        return target[mic], pred[mic], rps, loss

    def save_compare(rec_id, start_s, dur_s, mic, out, title):
        tgt, prd, _rps, loss = render(rec_id, start_s, dur_s, mic)
        St, Sp = _logspec(tgt), _logspec(prd)
        vmax = max(St.max(), Sp.max())
        vmin = vmax - 80
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for a, S, ttl in [(ax[0], St, "REAL"), (ax[1], Sp, "GENERATED")]:
            im = a.imshow(
                S,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap="magma",
                extent=[0, dur_s, 0, SR / 2000.0],
            )
            a.set_title(ttl, fontsize=12)
            a.set_xlabel("time (s)")
            a.set_ylabel("kHz")
            fig.colorbar(im, ax=a, format="%+0.0f dB", fraction=0.046)
        fig.suptitle(f"{title}   |   multi-scale STFT loss = {loss:.3f}", fontsize=13)
        fig.tight_layout()
        fig.savefig(ASSETS / out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  rendered {out}  (loss={loss:.3f})")

    # Michael's (DJI M100) — the model fits this drone better.
    save_compare(
        "michaels:FLY124",
        start_s=30.0,
        dur_s=4.0,
        mic=5,
        out="s2_spec_michaels.png",
        title="Michael's DJI Matrice 100 (FLY124, mic 5)",
    )
    # DREGON — confused by wind noise; weaker mid-frequency harmonics.
    save_compare(
        "dregon:free-flight_nosource_room1",
        start_s=20.0,
        dur_s=4.0,
        mic=0,
        out="s2_spec_dregon.png",
        title="DREGON (free-flight, mic 0)",
    )


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    os.chdir(PROJECT_ROOT)  # model/data loaders assume repo root as cwd
    print("Section 1 + 3: reusing report figures")
    copy_reused()
    print("Section 2a: model diagram")
    fig_model_diagram()
    print("Section 2b: real vs generated spectrograms")
    render_spectrograms()
    print("Done ->", ASSETS)


if __name__ == "__main__":
    main()
