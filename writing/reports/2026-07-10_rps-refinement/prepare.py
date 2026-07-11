#!/usr/bin/env python3
"""Generate figures for the RPS-refinement report.

Method figures are synthesized here; results figures are rendered from the
experiment artifacts under results/rps_refinement/{validation,spcup,robustness}/
produced by scripts/rps_refinement_{validation,spcup,robustness}.py.
Run from the repo root (Makefile sets PYTHONPATH) so `src/` imports resolve.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
DATA = HERE / "data"
RESULTS = ROOT / "results" / "rps_refinement"

# Consistent trajectory colours across the results figures.
C_MEASURED = "black"  # ground truth
C_COMMAND = "0.55"  # telemetry initialisation (gray)
C_BC = "#d62728"  # stage B+C (magnitude)
C_D = "#1f77b4"  # stage D (coherent)


def fig_method_comb() -> None:
    """Hero figure: mistuned comb misses the ridges; refinement locks it."""
    from data_processing.rps_refinement import RefineConfig, compute_logmag, refine_trajectories

    sr, dur = 16000, 6.0
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(7)
    r_true = 75.0 + 3.0 * np.sin(2 * np.pi * 0.25 * t) + 1.5 * np.sin(2 * np.pi * 0.07 * t + 1.0)
    phase = 2 * np.pi * np.cumsum(r_true) / sr
    x = sum((0.5 / k) * np.cos(k * phase + rng.uniform(0, 2 * np.pi)) for k in range(1, 41))
    x = np.asarray(x) + 0.03 * rng.standard_normal(len(t))

    cfg = RefineConfig()
    spec = compute_logmag(x.astype(np.float32), cfg)
    r_frames = np.interp(spec.frame_times, t, r_true)[None, :]
    r_init = r_frames + 0.45 + 0.25 * np.sin(2 * np.pi * 0.15 * spec.frame_times)[None, :]
    res = refine_trajectories(spec, r_init, cfg)

    logmag = spec.logmag[0].numpy()
    # Zoom to high harmonics where a 0.5 rev/s error is a visible fraction of
    # the comb spacing (k≈26: displacement ~13 Hz vs 75 Hz spacing).
    f_lo, f_hi, t_hi = 1700.0, 2500.0, 3.0
    b0, b1 = int(f_lo / spec.bin_hz), int(f_hi / spec.bin_hz)
    n_t = int(np.searchsorted(spec.frame_times, t_hi))
    extent = (0.0, t_hi, f_lo, f_hi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, r, title in (
        (axes[0], r_init[0], "initial labels (error ≈ 0.5 rev/s): tracks miss the ridges"),
        (axes[1], res.r_refined[0], "after refinement: tracks lock on"),
    ):
        crop = logmag[b0:b1, :n_t]
        ax.imshow(
            crop,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=float(np.percentile(crop, 30)),
        )
        first = True
        for k in range(int(np.ceil(f_lo / 75.0)), int(f_hi / 70.0) + 1):
            track = k * r[:n_t]
            if track.min() > f_hi or track.max() < f_lo:
                continue
            ax.plot(
                spec.frame_times[:n_t],
                track,
                color="cyan",
                lw=1.4,
                ls=(0, (4, 3)),
                alpha=0.95,
                label="labelled harmonic tracks k·r̂" if first else None,
            )
            first = False
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time [s]")
        ax.set_ylim(f_lo, f_hi)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("frequency [Hz]")
    err_i = float(np.abs(r_init - r_frames).mean())
    err_r = float(np.abs(res.r_refined - r_frames).mean())
    fig.suptitle(
        f"Comb alignment: mean |error| {err_i:.2f} → {err_r:.3f} rev/s (synthetic, 40 harmonics)"
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "method_comb_alignment.png", dpi=150)
    plt.close(fig)


def fig_displacement() -> None:
    """The k·delta displacement picture: same label error across harmonics."""
    delta = 0.3
    bin_hz = 16000 / 2048
    k = np.arange(1, 61)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(k, k * delta, lw=2, label=f"displacement of harmonic k (δ = {delta} rev/s)")
    ax.axhline(
        bin_hz / 2, color="gray", ls="--", lw=1, label="half a spectrogram bin (2048-pt STFT)"
    )
    ax.axhline(2 * bin_hz, color="firebrick", ls=":", lw=1.5, label="≈ peak width (2 bins)")
    ax.set_xlabel("harmonic index k")
    ax.set_ylabel("frequency displacement [Hz]")
    ax.set_title("A constant label error displaces the k-th harmonic k× further")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "method_displacement.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Results figures (rendered from experiment artifacts).                        #
# --------------------------------------------------------------------------- #

_VAL_NPZ = RESULTS / "validation" / "dregon_free-flight_nosource_room1.npz"


def fig_validation_overlay() -> None:
    """Validation on DREGON: stage B+C drifts down, stage D and command do not.

    Left  — one rotor, 12 s zoom, all four trajectories overlaid.
    Right — per-rotor SIGNED mean error against measured truth.
    """
    d = np.load(_VAL_NPZ, allow_pickle=True)
    ft = d["frame_times"]
    cmd, meas, bc, dd = d["command"], d["measured"], d["refined"], d["refined_coherent"]

    # Rotor 0 over [48.6, 60.6] s is where the B+C downward bias is largest
    # (window/rotor chosen as the minimum of the windowed mean of B+C − measured).
    rotor, t0, t1 = 0, 48.6, 60.6
    m = (ft >= t0) & (ft <= t1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=(1.55, 1.0))

    axL.plot(ft[m], meas[rotor, m], color=C_MEASURED, lw=1.0, label="measured (truth)")
    axL.plot(
        ft[m], cmd[rotor, m], color=C_COMMAND, lw=1.4, ls="--", label="command (telemetry init)"
    )
    axL.plot(ft[m], bc[rotor, m], color=C_BC, lw=1.6, label="stage B+C (magnitude)")
    axL.plot(ft[m], dd[rotor, m], color=C_D, lw=1.6, label="stage D (coherent)")
    axL.set_xlabel("time [s]")
    axL.set_ylabel("rotor speed [rev/s]")
    axL.set_title(f"Rotor {rotor + 1}, 12 s zoom: stage B+C sits below the truth", fontsize=10)
    axL.legend(loc="upper right", fontsize=8, ncol=1)
    axL.margins(x=0.01)

    # Right: per-rotor signed mean error (estimate − measured), rev/s.
    labels = ["command", "stage B+C", "stage D"]
    colors = [C_COMMAND, C_BC, C_D]
    arrs = [cmd, bc, dd]
    n_rot = meas.shape[0]
    x = np.arange(n_rot)
    w = 0.26
    for j, (lab, col, arr) in enumerate(zip(labels, colors, arrs)):
        err = (arr - meas).mean(axis=1)
        axR.bar(x + (j - 1) * w, err, w, color=col, label=lab, edgecolor="black", linewidth=0.4)
    axR.axhline(0, color="black", lw=0.8)
    bc_bias = float((bc - meas).mean())
    axR.axhline(bc_bias, color=C_BC, lw=1.0, ls=":")
    # Empty band above 0 (all bars are negative) holds the legend cleanly.
    axR.set_ylim(-0.63, 0.16)
    axR.text(
        0.985,
        0.24,
        f"B+C bias {bc_bias:.2f} rev/s",
        transform=axR.transAxes,
        fontsize=8,
        color=C_BC,
        va="center",
        ha="right",
    )
    axR.set_xticks(x)
    axR.set_xticklabels([f"R{i + 1}" for i in range(n_rot)])
    axR.set_ylabel("signed mean error vs measured [rev/s]")
    axR.set_title("Per-rotor bias: B+C is systematic; command & D ≈ 0", fontsize=10)
    axR.legend(loc="upper center", ncol=3, fontsize=8, columnspacing=1.0, handletextpad=0.4)

    fig.suptitle(
        "DREGON free-flight (no source): refining command → measured, without ever seeing measured",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "val_overlay.png", dpi=150)
    plt.close(fig)


def fig_validation_jitter() -> None:
    """The irreducible jitter: fast measured wiggle that no track can follow."""
    d = np.load(_VAL_NPZ, allow_pickle=True)
    ft = d["frame_times"]
    meas, cmd, dd = d["measured"], d["command"], d["refined_coherent"]

    # Rotor 1 over [23.4, 27.4] s: nearly flat mean, strong fast fluctuation.
    rotor, t0, t1 = 1, 23.4, 27.4
    m = (ft >= t0) & (ft <= t1)

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.plot(ft[m], meas[rotor, m], color=C_MEASURED, lw=1.3, label="measured (truth) — jitters")
    ax.plot(ft[m], cmd[rotor, m], color=C_COMMAND, lw=1.6, ls="--", label="command (telemetry)")
    ax.plot(ft[m], dd[rotor, m], color=C_D, lw=1.6, label="stage D (coherent, refined)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rotor speed [rev/s]")
    ax.set_title(
        f"Rotor {rotor + 1}, 4 s ultra-zoom: the measured speed jitters faster than\n"
        "either the smooth telemetry or the refined track can follow",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(ASSETS / "val_jitter.png", dpi=150)
    plt.close(fig)


def fig_basin() -> None:
    """Capture basin equals the coarse grid range, exactly."""
    import pandas as pd

    df = pd.read_csv(DATA / "basin_summary.csv")
    # Average over the sign patterns (same/opposite offsets across rotors).
    agg = df.groupby(["config", "offset"])["success_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    specs = [
        ("default_dmax3", 3.0, "#1f77b4", "grid δ_max = 3 rev/s (default)"),
        ("dmax6", 6.0, "#d62728", "grid δ_max = 6 rev/s (wide)"),
    ]
    ymax = 1.06
    for cfg, dmax, col, lab in specs:
        sub = agg[agg.config == cfg]
        off = np.asarray(sub["offset"], dtype=float)
        sr = np.asarray(sub["success_rate"], dtype=float)
        order = np.argsort(off)
        ax.plot(off[order], sr[order], "-o", color=col, lw=1.8, ms=4, label=lab)
        ax.axvline(dmax, color=col, ls=":", lw=1.2)
        ax.axvspan(dmax, agg.offset.max(), color=col, alpha=0.07)

    ax.text(
        4.5,
        0.5,
        "beyond δ_max:\ntotal failure",
        fontsize=8,
        ha="center",
        va="center",
        color="0.35",
    )
    ax.annotate(
        "basin edge = coarse grid range δ_max",
        xy=(3.0, 0.83),
        xytext=(3.4, 0.30),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9),
    )
    ax.set_xlabel("initial label error (constant offset) [rev/s]")
    ax.set_ylabel("recovery success rate")
    ax.set_ylim(-0.04, ymax)
    ax.set_title(
        "Capture basin = the coarse grid range, exactly\n"
        "(the gradient stage adds precision, not range)",
        fontsize=11,
    )
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "basin.png", dpi=150)
    plt.close(fig)


def fig_noise_floor() -> None:
    """Error vs harmonic SNR, per noise type, with the 0.15 rev/s tolerance."""
    import pandas as pd

    df = pd.read_csv(DATA / "robustness_summary.csv")
    styles = {
        "white": ("#1f77b4", "o", "white noise"),
        "pink": ("#d62728", "s", "pink noise"),
        "speech": ("#2ca02c", "^", "speech-shaped noise"),
    }
    tol = 0.15
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for nt, (col, mk, lab) in styles.items():
        sub = df[df.noise_type == nt]
        snr = np.asarray(sub["snr_db"], dtype=float)
        err = np.asarray(sub["mean_error"], dtype=float)
        order = np.argsort(snr)
        ax.plot(snr[order], err[order], "-", marker=mk, color=col, lw=1.8, ms=5, label=lab)
    ax.axhline(tol, color="0.35", ls="--", lw=1.2)
    ax.text(
        df.snr_db.min(), tol * 1.08, f"tolerance {tol} rev/s", fontsize=8, color="0.35", va="bottom"
    )
    ax.set_yscale("log")
    ax.set_xlabel("harmonic SNR [dB]")
    ax.set_ylabel("mean RPS error [rev/s]")
    ax.set_title("Noise floor: white/pink tolerated to ≈ 0 dB; speech-shaped noise bites earlier")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "noise_floor.png", dpi=150)
    plt.close(fig)


def fig_confidence_gate() -> None:
    """Confidence vs error: the gate works, except for identity capture."""
    import pandas as pd

    df = pd.read_csv(DATA / "confidence_scatter.csv")
    thr = 0.171
    sweeps = {
        "basin": ("#1f77b4", "initialisation sweep"),
        "robust": ("#d62728", "noise sweep"),
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for sw, (col, lab) in sweeps.items():
        sub = df[df.sweep == sw]
        ax.scatter(
            sub.confidence, sub.error, s=18, color=col, alpha=0.6, edgecolor="none", label=lab
        )
    ax.set_yscale("log")
    ax.axvline(thr, color="black", lw=1.3, ls="--")
    ax.text(thr * 1.03, ax.get_ylim()[1] * 0.6, f"Youden gate\nconf > {thr}", fontsize=8, va="top")

    # Identity-capture stripe: high confidence, moderate-but-wrong error.
    stripe = df[(df.confidence > thr) & (df.error > 0.15) & (df.error < 0.3)]
    if len(stripe):
        ax.axhspan(
            stripe.error.min() * 0.85,
            stripe.error.max() * 1.15,
            xmin=0,
            xmax=1,
            color="orange",
            alpha=0.12,
        )
        ax.annotate(
            'high confidence but WRONG:\n"identity capture" (locked on\nthe wrong rotor)',
            xy=(stripe.confidence.median(), stripe.error.median()),
            xytext=(0.20, 0.55),
            textcoords="axes fraction",
            fontsize=8,
            color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.0),
        )
    ax.set_xlabel("comb confidence (contrast)")
    ax.set_ylabel("RPS error [rev/s]")
    ax.set_title("Confidence gate rejects noise-induced failures, but not wrong-rotor capture")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(ASSETS / "confidence_gate.png", dpi=150)
    plt.close(fig)


def fig_spcup() -> None:
    """Blind SPCup annotation: a clean two-comb lock vs an honest refusal."""
    from data_processing.rps_refinement import RefineConfig, compute_logmag

    spc = RESULTS / "spcup"
    ku_name = "KU_Leuven__SPCUP19_KU_Leuven_Team_1_recording"
    idea_name = "Idea_ssu__free_flight_1"
    ku = np.load(spc / f"{ku_name}.npz", allow_pickle=True)
    idea = np.load(spc / f"{idea_name}.npz", allow_pickle=True)

    fig, (axS, axSp) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=(1.0, 1.15))

    # --- Left: base-speed scan curves. -----------------------------------
    axS.plot(
        ku["grid"], ku["scores"], color="#1f77b4", lw=1.5, label="KU Leuven (locks, conf 0.51)"
    )
    axS.plot(
        idea["grid"], idea["scores"], color="#d62728", lw=1.5, label="Idea_ssu (refuses, conf 0.02)"
    )
    kb = float(ku["base"])
    axS.plot(kb, ku["scores"][int(np.argmin(np.abs(ku["grid"] - kb)))], "v", color="#1f77b4", ms=9)
    axS.annotate(
        f"peak → base {kb:.1f} rev/s\n(resolves 42.5 + 46.2)",
        xy=(kb, ku["scores"].max()),
        xytext=(kb + 12, ku["scores"].max() - 0.02),
        fontsize=8,
        color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9),
    )
    axS.text(
        95,
        idea["scores"].max() - 0.12,
        "no sharp peak →\nno comb → rejected",
        fontsize=8,
        color="#d62728",
        ha="center",
    )
    axS.set_xlabel("candidate base speed r₀ [rev/s]")
    axS.set_ylabel("comb score S(r₀)")
    axS.set_title("Blind base-speed scan: a real comb makes a sharp peak", fontsize=10)
    axS.legend(loc="lower right", fontsize=8)

    # --- Right: KU Leuven spectrogram, zoomed to a few high harmonics so the
    # mismatch is visible: the ridges bend (maneuvers), the refined tracks do
    # not follow — the refiner recovers a mean operating point only.
    seg = np.load(spc / "segments" / f"{ku_name}.npz", allow_pickle=True)
    audio = seg["audio"].astype(np.float32)
    cfg = RefineConfig()
    spec = compute_logmag(audio, cfg)
    logmag = spec.logmag.mean(dim=0).numpy()  # average over mics
    f_lo, f_hi = 780.0, 1280.0  # harmonics k≈18..30 of ~43 rev/s
    b0, b1 = int(f_lo / spec.bin_hz), int(f_hi / spec.bin_hz)
    crop = logmag[b0:b1, :]
    ft = spec.frame_times
    extent = (float(ft[0]), float(ft[-1]), f_lo, f_hi)
    axSp.imshow(
        crop,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=float(np.percentile(crop, 40)),
        vmax=float(np.percentile(crop, 99.7)),
    )
    r_ref = ku["r_refined"]
    rt = ku["frame_times"]
    comb_cols = ["#00e5ff", "#ffe600"]
    speeds = ku["base_speeds"]
    for i in range(r_ref.shape[0]):
        first = True
        for k in range(1, cfg.k_max + 1):
            track = k * r_ref[i]
            if track.max() < f_lo or track.min() > f_hi:
                continue
            axSp.plot(
                rt,
                track,
                color=comb_cols[i % len(comb_cols)],
                lw=1.5,
                ls=(0, (5, 3)),
                alpha=0.95,
                label=f"comb {i + 1}: {speeds[i]:.1f} rev/s (near-flat)" if first else None,
            )
            first = False
    axSp.annotate(
        "ridges bend (maneuver)\ntracks do not follow",
        xy=(7.3, 1130.0),
        xytext=(9.6, 1215.0),
        fontsize=9,
        color="white",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
    )
    axSp.set_ylim(f_lo, f_hi)
    axSp.set_xlabel("time [s]")
    axSp.set_ylabel("frequency [Hz]")
    axSp.set_title("KU Leuven: mean speed recovered; maneuvers not tracked", fontsize=10)
    axSp.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "SPCup blind analysis: comb detection and refusal work; trajectory tracking does not",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "spcup.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    fig_displacement()
    fig_method_comb()
    fig_validation_overlay()
    fig_validation_jitter()
    fig_basin()
    fig_noise_floor()
    fig_confidence_gate()
    fig_spcup()


if __name__ == "__main__":
    main()
