#!/usr/bin/env python3
"""Figures for the annotation-bottleneck deck (2026-08-11 restructure).

Two families:
  * NEW_FIGURES (default) -- the paper-narrative figures, built from
    ``results/gen_label_sensitivity/per_k.csv``, ``results/telemetry_report_6d.json``
    and closed-form landscape facts.
  * LEGACY_FIGURES (``python3 prepare_figs.py --all``) -- the tracking figures of
    the previous deck; they need cached run artifacts, a dload stream and a model
    forward pass, so they are rebuilt only on request.

Legacy sources:

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
import shutil
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


# ══════════════════════════════════════════════════════════════════════════
# Paper-narrative figures (2026-08-11 restructure)
# ══════════════════════════════════════════════════════════════════════════

FITNESS_REPORT = RESULTS / "telemetry_report_6d.json"
GEN_LABEL_CSV = RESULTS / "gen_label_sensitivity" / "per_k.csv"


def _rolling_median(y: np.ndarray, w: int = 5) -> np.ndarray:
    pad = w // 2
    ext = np.pad(y, pad, mode="edge")
    return np.array([np.median(ext[i : i + w]) for i in range(len(y))])


def fig_gen_label_bias() -> None:
    """Section 1a: the generator's high harmonics collapse under a CONSTANT
    label bias, and train flat under exact labels. Data:
    ``results/gen_label_sensitivity/per_k.csv`` (phase-7 label A/B)."""
    with open(GEN_LABEL_CSV) as fh:
        rows = list(csv.DictReader(fh))
    k = np.array([int(r["k"]) for r in rows])
    arms = {
        "exact labels": ("exact_delta_db", "#1f77b4"),
        "staircase only": ("tach_pure_delta_db", "#7f7f7f"),
        "constant 0.54 % bias": ("scale_delta_db", "#d62728"),
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    band = (k >= 50) & (k <= 80)
    notes = []
    for label, (col, color) in arms.items():
        y = np.array([float(r[col]) for r in rows])
        ax.plot(k, y, color=color, lw=0.9, alpha=0.30)
        ax.plot(k, _rolling_median(y), color=color, lw=2.4, label=label)
        notes.append((label, float(np.mean(y[band])), color))
    ax.axhline(0.0, color="#333333", ls="--", lw=1.0)
    ax.axvspan(50, 80, color="#d62728", alpha=0.06)
    ax.set_xlabel("harmonic index k")
    ax.set_ylabel("learned line power $-$ true (dB)")
    ax.set_xlim(1, 80)
    ax.set_ylim(-16, 3)
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    txt = "\n".join(f"{lab}: {val:+.1f} dB" for lab, val, _ in notes)
    ax.text(
        0.985,
        0.05,
        f"mean over k = 50–80\n{txt}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="#bbbbbb", alpha=0.92, pad=4),
    )
    ax.set_title("A constant label bias, alone, collapses the high harmonics", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "gen_label_bias.png", dpi=150)
    plt.close(fig)
    print("gen_label_bias:", [(lab, round(v, 2)) for lab, v, _ in notes])


def fig_sim2real() -> None:
    """Section 1b: RPS predictors trained on generated noise do not transfer.
    Numbers: docs/experiments/e8-static-comb.md (E7 neural-gen arm vs the
    real-data reference, same architecture, same real validation set)."""
    labels = ["trained on\ngenerated noise", "trained on\nreal noise"]
    vals = [222.3, 7.33]
    colors = ["#d62728", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(np.arange(2), vals, 0.5, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel("PIT-MSE on the real validation set (log)")
    ax.set_ylim(1, 900)
    for x, v, note in zip(np.arange(2.0), vals, ["R² = −10.5", "R² > 0"], strict=False):
        ax.text(x, v * 1.25, f"{v:g}\n{note}", ha="center", fontsize=11.5)
    ax.set_title("Generator-trained predictors are worse than predicting the mean", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "sim2real.png", dpi=150)
    plt.close(fig)
    print("sim2real:", vals)


def _fitness_report() -> dict:
    import json

    with open(FITNESS_REPORT) as fh:
        return json.load(fh)


def _ridge(rep: dict, key: str) -> float:
    return float(rep["controls"][key]["ridge"])


def _ridge_bars(rows, out_name: str, title: str, figsize=(9.0, 4.6)) -> None:
    """Shared renderer: on-comb ridge bar + its paired off-comb null bar."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(rows))
    w = 0.36
    on = [r[1] for r in rows]
    null = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    ax.bar(x - w / 2, on, w, color=colors, label="candidate (on the comb)")
    ax.bar(
        x + w / 2,
        null,
        w,
        color="white",
        edgecolor="#777777",
        hatch="///",
        label="its off-comb null",
    )
    ax.axhline(0.0, color="#2ca02c", ls="--", lw=1.4, label="0 dB = pure noise (calibrated)")
    for xi, v in zip(x, on, strict=False):
        ax.text(xi - w / 2, max(v, 0.0) + 0.14, f"{v:+.2f}", ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=11)
    ax.set_ylabel("ridge (dB, higher = better lock)")
    ax.set_ylim(min(-1.2, min(null) - 0.4), max(on) + 0.9)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / out_name, dpi=150)
    plt.close(fig)
    print(f"{out_name}:", [(r[0].replace("\n", " "), round(r[1], 2)) for r in rows])


def fig_ridge_telemetry() -> None:
    """Section 1c: the two rigs' raw telemetry, same instrument, same settings."""
    rep = _fitness_report()
    rows = [
        (
            "DREGON\ntelemetry",
            _ridge(rep, "dregon|telemetry|on"),
            _ridge(rep, "dregon|telemetry|offcomb"),
            "#d62728",
        ),
        (
            "FLY124\ntelemetry (recalibrated)",
            _ridge(rep, "fly124-cruise|telemetry|on"),
            _ridge(rep, "fly124-cruise|telemetry|offcomb"),
            "#1f77b4",
        ),
    ]
    _ridge_bars(
        rows,
        "ridge_telemetry.png",
        "One rig's labels sit on the comb; the other's do not",
        figsize=(6.8, 4.4),
    )


def fig_ridge_candidates() -> None:
    """Section 2: the verdict table as a figure — telemetry, the best constant,
    and the fitted trajectory, each against its own off-comb null."""
    rep = _fitness_report()
    rows = [
        (
            "DREGON\ntelemetry",
            _ridge(rep, "dregon|telemetry|on"),
            _ridge(rep, "dregon|telemetry|offcomb"),
            "#d62728",
        ),
        (
            "DREGON\nbest constant scale",
            _ridge(rep, "dregon|scale:0.99458|on"),
            _ridge(rep, "dregon|scale:0.99458|offcomb"),
            "#ff7f0e",
        ),
        (
            "DREGON\nfitted trajectory",
            _ridge(rep, "dregon|fit:main|on"),
            _ridge(rep, "dregon|fit:main|offcomb"),
            "#2ca02c",
        ),
        (
            "FLY124\ntelemetry",
            _ridge(rep, "fly124-cruise|telemetry|on"),
            _ridge(rep, "fly124-cruise|telemetry|offcomb"),
            "#1f77b4",
        ),
        (
            "FLY124\nfitted trajectory",
            _ridge(rep, "fly124-cruise|fit:main|on"),
            _ridge(rep, "fly124-cruise|fit:main|offcomb"),
            "#9467bd",
        ),
    ]
    _ridge_bars(
        rows,
        "ridge_candidates.png",
        "Fitting the trajectory buys 2.47 dB on DREGON and 0.26 dB on FLY124",
    )


def fig_scale_profile() -> None:
    """Section 2: the one-parameter scale profile and its off-comb null."""
    rep = _fitness_report()
    cur_on = rep["profile"]["curves"]["dregon|on|none"]
    cur_off = rep["profile"]["curves"]["dregon|offcomb|none"]
    m = rep["profile"]["minima"]["dregon|on|none"]
    s_on = (np.array(cur_on["s"], dtype=float) - 1.0) * 100.0
    s_off = (np.array(cur_off["s"], dtype=float) - 1.0) * 100.0
    y_on = np.array(cur_on["mean"], dtype=float)
    y_off = np.array(cur_off["mean"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(s_on, y_on, "o-", color="#1f77b4", lw=2.2, ms=4, label="DREGON, on the comb")
    ax.plot(s_off, y_off, "o--", color="#888888", lw=1.8, ms=3, label="off-comb null")
    vertex = float(m["scale_pct"])
    lo, hi = float(m["ci"]["lo"]), float(m["ci"]["hi"])
    ax.axvspan(lo, hi, color="#1f77b4", alpha=0.12)
    ax.axvline(vertex, color="#1f77b4", ls=":", lw=1.8)
    ax.axvline(0.0, color="#333333", lw=1.0)
    ax.annotate(
        f"{vertex:.3f} %  [{lo:.3f}, {hi:.3f}]",
        xy=(vertex, float(np.max(y_on))),
        xytext=(vertex + 0.30, float(np.max(y_on)) - 0.06),
        fontsize=12,
        color="#1f77b4",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    )
    ax.text(
        0.015,
        0.06,
        f"basin depth {float(m['depth']):.2f} dB\nnull depth "
        f"{float(rep['profile']['minima']['dregon|offcomb|none']['depth']):.2f} dB",
        transform=ax.transAxes,
        ha="left",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#bbbbbb", alpha=0.92, pad=4),
    )
    ax.set_xlabel("constant rate scale applied to the labels (%)")
    ax.set_ylabel("ridge (dB)")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.set_title(
        "One free parameter, four hold-out families, an 8x deeper basin than the null", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "scale_profile.png", dpi=150)
    plt.close(fig)
    print(f"scale_profile: vertex {vertex} ci [{lo}, {hi}] depth {m['depth']}")


def fig_ridge_instrument() -> None:
    """Section 2: schematic of what the instrument reads on one cell — a fixed
    line band against a local floor annulus, on the demodulated envelope
    spectrum. Illustrative synthetic data, drawn to scale in rev/s."""
    rng = np.random.default_rng(7)
    f = np.linspace(-1.2, 1.2, 601)
    dc = 0.10

    def envelope(offset: float) -> np.ndarray:
        line = 12.0 * np.exp(-0.5 * ((f - offset) / 0.035) ** 2)
        floor = rng.exponential(0.25, size=f.size)
        return 10 * np.log10(line + floor + 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    for ax, off, name in (
        (axes[0], 0.0, "carrier on the comb"),
        (axes[1], 0.35, "carrier 0.35 rev/s off"),
    ):
        ax.plot(f, envelope(off), color="#333333", lw=1.0)
        ax.axvspan(-dc, dc, color="#1f77b4", alpha=0.18)
        ax.axvspan(-1.2, -dc - 0.12, color="#ff7f0e", alpha=0.10)
        ax.axvspan(dc + 0.12, 1.2, color="#ff7f0e", alpha=0.10)
        ax.set_xlabel("demodulated frequency (rev/s)")
        ax.set_title(name, fontsize=11.5)
        ax.set_xlim(-1.2, 1.2)
    axes[0].set_ylabel("power (dB)")
    axes[0].text(0.0, -26.0, "line band", ha="center", color="#1f77b4", fontsize=11)
    axes[0].text(0.75, -26.0, "floor annulus", ha="center", color="#c26a00", fontsize=11)
    axes[0].text(0.02, 0.93, "ridge high", transform=axes[0].transAxes, fontsize=12)
    axes[1].text(0.02, 0.93, "ridge ≈ 0 dB", transform=axes[1].transAxes, fontsize=12)
    fig.suptitle("ridge = 10 log10 (power in the fixed band / local floor density)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "ridge_instrument.png", dpi=150)
    plt.close(fig)
    print("ridge_instrument: schematic written")


def fig_basin_law() -> None:
    """Section 3: the basin law 1/(K*T) against DREGON's measured label error."""
    K = np.arange(2, 121)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for T, color, style in ((0.25, "#9467bd", ":"), (1.0, "#1f77b4", "-"), (4.0, "#2ca02c", "--")):
        ax.loglog(K, 1.0 / (K * T), style, color=color, lw=2.2, label=f"T = {T:g} s")
    err_lo, err_hi = 0.0035 * 60.0, 0.0085 * 60.0
    ax.axhspan(err_lo, err_hi, color="#d62728", alpha=0.14)
    ax.text(
        26.0,
        (err_lo * err_hi) ** 0.5,
        "DREGON label error, 0.35–0.85 % of ~60 Hz",
        color="#d62728",
        fontsize=11,
        va="center",
    )
    ax.plot([80], [1.0 / 80.0], "o", color="#1f77b4", ms=9)
    ax.annotate(
        "K = 80, T = 1 s → 0.0125 Hz\n(the label error is 16–40 basins out)",
        xy=(80, 1.0 / 80.0),
        xytext=(9, 0.0035),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    )
    ax.set_xlabel("harmonics summed, K")
    ax.set_ylabel("basin width Δf0 (Hz)")
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax.grid(True, which="both", color="#dddddd", lw=0.6)
    ax.set_title("Precision and basin width are the same phenomenon: Δf0 ≈ 1/(K·T)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "basin_law.png", dpi=150)
    plt.close(fig)
    print("basin_law: written")


def fig_alias_lattice() -> None:
    """Section 3: the sub-multiple comb contains the true comb (nesting)."""
    f0 = 80.0
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    for m in range(1, 17):
        ax.vlines(m * f0 / 2, 0, 1.0, color="#bbbbbb", lw=6.0)
    for k in range(1, 9):
        ax.vlines(k * f0, 0, 1.0, color="#1f77b4", lw=2.4)
    ax.set_ylim(0, 1.6)
    ax.set_xlim(0, 9 * f0)
    ax.set_yticks([])
    ax.set_xlabel("frequency (Hz)")
    ax.plot([], [], color="#1f77b4", lw=3.0, label="true comb, rate f0")
    ax.plot([], [], color="#999999", lw=2.0, label="candidate comb, rate f0/2")
    ax.legend(frameon=False, loc="upper right", fontsize=11.5, ncols=2)
    ax.text(
        0.3 * f0,
        1.22,
        "every true line is also a line of the f0/2 comb: the fit is exactly degenerate",
        fontsize=11.5,
        color="#333333",
    )
    ax.annotate(
        "the extra lines sit on empty spectrum:\nonly an order penalty charges for them",
        xy=(2.5 * f0, 1.0),
        xytext=(2.1 * f0, 0.30),
        fontsize=11,
        color="#666666",
        arrowprops=dict(arrowstyle="->", color="#888888"),
    )
    ax.set_title("Sub-multiples are nested, not nearby: no smoothing removes them", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "alias_lattice.png", dpi=150)
    plt.close(fig)
    print("alias_lattice: written")


def copy_campaign_figures() -> None:
    """Steps 1, 3, 4 and 5 are figures produced by the campaign drivers
    themselves (``scripts``-side, under ``results/``). Nothing is redrawn here:
    the PNGs are copied into ``assets/`` so the deck stays self-contained and
    ``make`` still refreshes them from the newest run."""
    copies = {
        "results/fvk_telemetry/fig_oracle_sanity.png": "oracle_sanity.png",
        "results/fvk_arms/fig_step4_arms.png": "step4_arms.png",
        "results/fvk_arms/fig_step5_pareto.png": "step5_pareto.png",
        "results/fvk_bench/fig_gra.png": "bench_gra.png",
    }
    for src_rel, dst_name in copies.items():
        src = ROOT / src_rel
        if not src.exists():
            print(f"copy: MISSING {src_rel} (kept the asset already in place)")
            continue
        shutil.copyfile(src, ASSETS / dst_name)
        print(f"copy: {src_rel} -> assets/{dst_name}")


NEW_FIGURES = (
    copy_campaign_figures,
    fig_gen_label_bias,
    fig_sim2real,
    fig_ridge_telemetry,
    fig_ridge_candidates,
    fig_scale_profile,
    fig_ridge_instrument,
    fig_basin_law,
    fig_alias_lattice,
)

LEGACY_FIGURES = (
    fig_lock_ladder,
    fig_beamform,
    fig_output_comparisons,
    fig_stepper,
    fig_peel_guard,
    fig_alternation_loop,
)


if __name__ == "__main__":
    ensure_dirs()
    # The legacy figures need cached run artifacts and a model forward pass;
    # their PNGs are already in assets/, so they are rebuilt only on request.
    figures = NEW_FIGURES + (LEGACY_FIGURES if "--all" in sys.argv else ())
    for fn in figures:
        fn()
    print("done.")
