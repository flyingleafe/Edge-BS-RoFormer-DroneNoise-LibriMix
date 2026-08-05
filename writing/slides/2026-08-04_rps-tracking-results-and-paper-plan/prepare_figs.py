#!/usr/bin/env python3
"""Figures for the 2026-08-04 RPS-tracking / paper-plan deck.

Real data, three sources:
  1. ``results/vk_phase_validation_decomp/rows.csv`` -- the lock ladder
     (S3 single motor, S3b static bench, S3c hover, S4 free flight).
  2. ``results/beamform_lock_probe/lock_table.csv`` -- "more mics?" bars.
  3. ``results/beatvk_vk_arms_pre_recalib_268c7660/runs/*.npz`` (VK) +
     ``rps_predictor_vk_eval`` / ``beatvk_eval`` loaders (GT + CKLA), for the
     two DREGON/FLY124 cruise-window overlay panels.

Run: PYTHONPATH=<repo root> python3 prepare_figs.py
"""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ROOT = HERE.resolve().parents[2]
RESULTS = ROOT / "results"

ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
INK = "#222222"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def ensure_dirs() -> None:
    ASSETS.mkdir(exist_ok=True)


# ─── Money slide: the lock ladder ──────────────────────────────────────────


def fig_lock_ladder() -> None:
    """Bar chart: per-harmonic phase lock (k=1, k=2) across the realism
    ladder. Data: ``vk_phase_validation_decomp`` rows.csv, method=iter_warp
    (the captured/converged pass, not the raw init offset)."""
    with open(RESULTS / "vk_phase_validation_decomp" / "rows.csv") as fh:
        rows = list(csv.DictReader(fh))

    def stage_lock(stage: str) -> tuple[float, float]:
        """Best (max) lock achieved over all rotors/init offsets at this
        stage -- the ceiling a converged tracker reaches, so the comparison
        is fair to each stage's own best case, not penalised by a capture
        failure at a bad init offset."""
        sub = [r for r in rows if r["stage"] == stage and r["method"] == "iter_warp"]
        l1 = np.array([float(r["lock1"]) for r in sub])
        l2 = np.array([float(r["lock2"]) for r in sub])
        return float(l1.max()), float(l2.max())

    stages = [
        ("single motor\n(static, 1 rotor)", "S3"),
        ("4 motors\n(static bench)", "S3b"),
        ("hover", "S3c"),
        ("free flight", "S4"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(stages))
    w = 0.32
    mid1, mid2 = [], []
    for _, tag in stages:
        a, b = stage_lock(tag)
        mid1.append(a)
        mid2.append(b)
    mid1, mid2 = map(np.array, (mid1, mid2))
    ax.bar(x - w / 2, mid1, w, label="k = 1 (best case)", color="#1f77b4")
    ax.bar(x + w / 2, mid2, w, label="k = 2 (best case)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in stages], fontsize=11)
    ax.set_ylabel("phase lock (resultant length, 0-1)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.7, color="#2ca02c", ls="--", lw=1.6)
    ax.text(
        0.34,
        0.72,
        "needed for lock (~0.7)",
        color="#2ca02c",
        fontsize=10,
        ha="center",
        transform=ax.get_yaxis_transform(),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Harmonic phase coherence collapses as soon as more than one motor runs")
    fig.tight_layout()
    fig.savefig(ASSETS / "lock_ladder.png", dpi=150)
    plt.close(fig)
    print(
        "lock_ladder: single-motor k1/k2",
        (mid1[0], mid2[0]),
        "| static-4motor k1/k2",
        (mid1[1], mid2[1]),
        "| free-flight k1/k2",
        (mid1[3], mid2[3]),
    )


# ─── "Use more microphones?" ────────────────────────────────────────────────


def fig_beamform() -> None:
    with open(RESULTS / "beamform_lock_probe" / "lock_table.csv") as fh:
        rows = list(csv.DictReader(fh))
    treatments = ["ch0", "das", "best_mic", "self_steer"]
    labels = {
        "ch0": "single mic",
        "das": "delay-\nand-sum",
        "best_mic": "best single\nmic (oracle)",
        "self_steer": "self-steered\ncombiner\n(oracle)",
    }
    ks = ["1", "2"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(treatments))
    w = 0.32
    for i, k in enumerate(ks):
        vals = []
        for t in treatments:
            sub = [float(r["lock"]) for r in rows if r["treatment"] == t and r["k"] == k]
            vals.append(float(np.mean(sub)) if sub else float("nan"))
        ax.bar(x + (i - 0.5) * w, vals, w, label=f"k = {k}", color=ROTOR_COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in treatments], fontsize=10.5)
    ax.axhline(0.7, color="#2ca02c", ls="--", lw=1.6)
    ax.text(
        0.05,
        0.72,
        "needed for lock (~0.7)",
        color="#2ca02c",
        fontsize=10,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_ylabel("phase lock (resultant length, 0-1)")
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("The incoherence arrives identically at every microphone")
    fig.tight_layout()
    fig.savefig(ASSETS / "beamform.png", dpi=150)
    plt.close(fig)


# ─── Output comparison panels: DREGON and FLY124 cruise windows ───────────


def _run_comparison(rec_id: str, window_idx: int, vk_arm_dir: str, out_name: str, title: str):
    """One panel: GT (dotted) vs blind-VK (from cached npz) vs CKLA-4s
    (fresh forward pass) on one cruise window of one recording."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import rps_predictor_vk_eval as vkev  # type: ignore
    import torch

    from data_processing import streams  # type: ignore
    from data_processing.frames import meta_dict, resample_audio_series  # type: ignore
    from tasks.rps_prediction import align_rps_to_gt  # type: ignore

    SR = 16000
    HOP = 512
    FRAME_S = HOP / SR

    # GT + native audio for the recording, from the frozen protocol dataset.
    frame = None
    for f in streams.iter_published_frames("beatvk-valid-raw", None):
        meta = meta_dict(f)
        if str(meta["recording_id"]) == rec_id:
            frame = f
            meta_ = meta
            break
    assert frame is not None, f"recording {rec_id} not found in beatvk-valid-raw"
    win = meta_["windows"][window_idx]
    start_s, end_s = float(win["start_s"]), float(win["end_s"])

    rps_series = frame["rps_raw"]
    ts = np.asarray(rps_series.tindex.abs_stamps, dtype=np.float64)
    vals = np.asarray(rps_series.data, dtype=np.float64)
    grid = np.arange(int(round(start_s / FRAME_S)), int(round(end_s / FRAME_S))) * FRAME_S
    gt = np.stack([np.interp(grid, ts, vals[r]) for r in range(4)])

    # Cached blind-VK trajectory for this window (frozen-protocol run).
    vk_npz = RESULTS / "beatvk_vk_arms_pre_recalib_268c7660" / "runs" / vk_arm_dir
    d = np.load(vk_npz, allow_pickle=True)
    vk_ft = np.asarray(d["ft"], dtype=np.float64) + start_s
    vk_traj = np.asarray(d["traj"], dtype=np.float64)
    vk_grid = np.stack([np.interp(grid, vk_ft, vk_traj[r]) for r in range(4)])
    vk_grid = align_rps_to_gt(vk_grid, gt)
    vk_mae = float(np.abs(vk_grid - gt).mean())

    # Fresh CKLA-4s forward pass on the same window's native audio.
    experiment, ckpt_uri, _ = vkev.MODELS["ckla_phaseonly_best"]
    model = vkev.load_model(experiment, ckpt_uri, "cpu")
    audio16 = resample_audio_series(frame["audio"], SR)
    audio = np.atleast_2d(np.asarray(audio16.data, dtype=np.float32))
    win_frames, slide_frames = 251, 32
    f_total = audio.shape[-1] // HOP + 1
    starts = vkev.window_starts(f_total, win_frames, slide_frames)
    with torch.no_grad():
        preds = vkev.predict_windows(model, audio, starts, "chmean", "cpu", 8, win_frames)
    stack = vkev.stitch_stack(preds, starts, f_total, win_frames)
    pred = np.nanmean(stack, axis=0).astype(np.float64)  # (4, f_total)
    t_pred = np.arange(pred.shape[-1], dtype=np.float64) * FRAME_S
    ckla_grid = np.stack([np.interp(grid, t_pred, pred[r]) for r in range(4)])
    ckla_grid = align_rps_to_gt(ckla_grid, gt)
    ckla_mae = float(np.abs(ckla_grid - gt).mean())

    t = grid - start_s
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)
    for ax, pred_grid, mae, name in (
        (axes[0], vk_grid, vk_mae, "blind VK chain"),
        (axes[1], ckla_grid, ckla_mae, "CKLA phase-only 4 s"),
    ):
        for r in range(4):
            ax.plot(t, gt[r], color=ROTOR_COLORS[r], lw=1.4, ls=":", alpha=0.75)
            ax.plot(t, pred_grid[r], color=ROTOR_COLORS[r], lw=2.0)
        ax.set_ylabel("rev/s")
        ax.text(
            0.02,
            0.90,
            f"{name}: PIT-MAE {mae:.2f} rev/s",
            transform=ax.transAxes,
            fontsize=12,
            color=INK,
            va="top",
        )
    axes[1].set_xlabel("s (window-relative)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(ASSETS / out_name, dpi=150)
    plt.close(fig)
    print(f"{out_name}: VK {vk_mae:.2f} | CKLA {ckla_mae:.2f}")
    return vk_mae, ckla_mae


def fig_output_comparisons() -> None:
    try:
        _run_comparison(
            rec_id="free-flight_nosource_room1",
            window_idx=1,
            vk_arm_dir="free-flight_nosource_room1__w01__blind_fullrange.npz",
            out_name="compare_dregon.png",
            title="DREGON cruise window",
        )
    except Exception as exc:  # pragma: no cover - keep the build alive
        print(f"[WARN] DREGON comparison figure failed: {exc}")
    try:
        _run_comparison(
            rec_id="FLY124",
            window_idx=4,
            vk_arm_dir="FLY124__w04__blind_fullrange.npz",
            out_name="compare_fly124.png",
            title="FLY124 cruise window",
        )
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] FLY124 comparison figure failed: {exc}")


# ─── VK chain stepper: same window, 5 stage snapshots ──────────────────────

STAGE_ORDER = ["coarse_init", "viterbi_c", "vit2dsp", "refine", "pi_kalman"]
STAGE_TITLE = {
    "coarse_init": "step 1: input & global stage",
    "viterbi_c": "step 2: ramp handling",
    "vit2dsp": "step 3: per-rotor decoupling",
    "refine": "step 4: coupled envelope solve",
    "pi_kalman": "step 5: phase-increment refine",
}
STEPPER_K = 8  # harmonic overlaid on the spectrogram, fixed across steps

# Flagship alternation trace (2026-08-04) -- blind init + peeled pi_kalman,
# dumped by the flagship runner for one FLY124 cruise window.
ALT_TRACE = pathlib.Path(
    "/tmp/claude-1000/-home-flyingleafe-Research-PhD-projects-harmonic-noise-suppression/"
    "44eb8c52-54f5-46b4-b6d6-d8f94442050e/scratchpad/pikalman_iter/blind_fly124_w03.json"
)


def _run_stepper_chain():
    """Run the blind baseline chain once on one FLY124 cruise window, keeping
    a copy of the raw (4, N) trajectory array at every named stage (the
    scoreboard ``Recorder`` only keeps metrics, not the array itself)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import rps_refine_lab as rl  # type: ignore

    from tasks.rps_prediction import align_rps_to_gt  # type: ignore

    prep, weights, meta = rl.real_window("FLY124", 4)

    captured: dict[str, np.ndarray] = {}
    orig_add = rl.Recorder.add

    def patched_add(self, stage, traj):  # noqa: ANN001
        captured[stage] = np.asarray(traj, dtype=np.float64).copy()
        return orig_add(self, stage, traj)

    rl.Recorder.add = patched_add
    try:
        rl.run_chain(
            "fly124_stepper",
            prep,
            weights,
            meta,
            "baseline",
            dict(rl.DEFAULT_PK),
            1,
            False,
            True,
            False,
            0,
        )
    finally:
        rl.Recorder.add = orig_add

    gt = prep.r_meas
    aligned = {
        name: align_rps_to_gt(captured[name], gt) for name in STAGE_ORDER if name in captured
    }
    return prep, gt, aligned


def fig_stepper() -> None:
    """Five two-panel (spectrogram-with-combs / RPS-vs-GT) figures, one per
    VK chain stage, same FLY124 cruise window and layout throughout so the
    slides read as a slider."""
    from scipy.signal import spectrogram as sp_spectrogram

    prep, gt, aligned = _run_stepper_chain()
    SR = 16000
    f, t_spec, S = sp_spectrogram(prep.audio[0], fs=SR, nperseg=1024, noverlap=768)
    S_db = 10 * np.log10(S + 1e-12)
    fmax = 900.0
    fmask = f <= fmax

    for stage in STAGE_ORDER:
        if stage not in aligned:
            print(f"[WARN] stepper: stage {stage!r} missing, skipping")
            continue
        traj = aligned[stage]
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]}
        )
        ax_top.pcolormesh(
            t_spec, f[fmask], S_db[fmask], shading="auto", cmap="bone_r", vmin=-90, vmax=-20
        )
        for r in range(4):
            ax_top.plot(prep.ft, STEPPER_K * traj[r], color=ROTOR_COLORS[r], lw=1.8)
        ax_top.set_ylim(0, fmax)
        ax_top.set_ylabel("Hz")
        ax_top.set_title(f"harmonic k={STEPPER_K} comb overlay", fontsize=11)

        for r in range(4):
            ax_bot.plot(prep.ft, gt[r], color=ROTOR_COLORS[r], lw=1.4, ls=":", alpha=0.8)
            ax_bot.plot(prep.ft, traj[r], color=ROTOR_COLORS[r], lw=2.0)
        ax_bot.set_ylabel("rev/s")
        ax_bot.set_xlabel("s")
        ax_bot.set_xlim(prep.ft[0], prep.ft[-1])

        fig.suptitle(STAGE_TITLE[stage], fontsize=12)
        fig.tight_layout()
        fig.savefig(ASSETS / f"stepper_{stage}.png", dpi=150)
        plt.close(fig)
        print(f"stepper_{stage}: mean pooled |err| {float(np.abs(traj - gt).mean()):.2f}")


# ─── Flagship alternation loop: peel guard + iterate-to-plateau ───────────


def fig_peel_guard() -> None:
    """Step 4 panel: what the peel removes per rotor (application 1, FLY124
    w03) and the energy-removed guard that gates it."""
    import json

    d = json.loads(ALT_TRACE.read_text())
    app1 = d["arms"]["peeled"]["snapshots"][1]["extras"]["peel"]
    labels = [f"rotor {r + 1}" for r in range(4)]
    removed = [p["e_removed_frac"] for p in app1["per_rotor"]]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = np.arange(4)
    ax.bar(x, removed, 0.55, color=ROTOR_COLORS)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("fraction of that rotor's comb energy removed by peel")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Peel application 1: each rotor's audio minus the OTHER\nrotors' reconstructed combs",
        fontsize=11,
    )
    ok = app1["energy_ok"]
    ax.text(
        0.5,
        0.92,
        f"peel-energy guard: {'PASS -> keep peel' if ok else 'FAIL -> fall back to init'}",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#2ca02c" if ok else "#d62728",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "stepper_refine.png", dpi=150)
    plt.close(fig)
    print("stepper_refine (peel guard): removed", [round(v, 2) for v in removed], "ok", ok)


def fig_alternation_loop() -> None:
    """Step 5 panel: pooled PIT-MAE across alternation applications, naive
    re-application vs peeled -- shows peeled converges to a plateau while
    naive stalls higher, on the same FLY124 w03 window."""
    import json

    d = json.loads(ALT_TRACE.read_text())
    apps = list(range(6))  # iter_0 (init) .. iter_5
    naive = [s["extras"]["pit_mae"]["mean"] for s in d["arms"]["naive"]["snapshots"]]
    peeled = [s["extras"]["pit_mae"]["mean"] for s in d["arms"]["peeled"]["snapshots"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(apps, naive, "o--", color="#888888", lw=1.8, label="naive re-application")
    ax.plot(apps, peeled, "o-", color="#1f77b4", lw=2.2, label="peeled (flagship)")
    ax.axvspan(2, 4, color="#1f77b4", alpha=0.08)
    ax.annotate(
        "plateau (2-4 applications)",
        xy=(3, peeled[3]),
        xytext=(4.6, peeled[0] * 0.68),
        fontsize=10,
        color="#1f77b4",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    )
    ax.set_xlabel("alternation application (0 = blind init)")
    ax.set_ylabel("PIT-MAE (rev/s), FLY124 w03")
    ax.set_xticks(apps)
    ax.set_title("Peeled iteration converges; naive iteration stalls higher", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "stepper_pi_kalman.png", dpi=150)
    plt.close(fig)
    print(
        f"alternation loop: init {peeled[0]:.3f} -> peeled x4 {peeled[4]:.3f} | naive x4 {naive[4]:.3f}"
    )


if __name__ == "__main__":
    ensure_dirs()
    fig_lock_ladder()
    fig_beamform()
    fig_output_comparisons()
    fig_stepper()
    fig_peel_guard()
    fig_alternation_loop()
    print("done.")
