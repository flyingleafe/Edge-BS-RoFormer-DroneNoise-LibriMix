#!/usr/bin/env python3
"""Slide figures for the DCUNet-generalization deck.

Same per-clip CSVs as the report
(``writing/reports/2026-07-26_dcunet-generalization``), re-plotted for
projection: larger type, fewer series per panel, one idea per figure.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
ROOT = pathlib.Path(__file__).resolve().parents[3]
F2 = ROOT / "results" / "f2_perclip"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Part 2 (CKLA campaign) figures: copied verbatim from the source report —
# no regeneration needed, same figures serve slides at a larger crop. Only
# the block diagram is still used at full-report emphasis; the rest of Part 2
# is regenerated (new augmentation-family + freq-shift-failure + results
# visuals) per the 2026-07-27 review spec.
CKLA_REPORT_ASSETS = ROOT / "writing" / "reports" / "2026-07-27_ckla-campaign" / "assets"
CKLA_FIGURES = ("ckla_block_diagram.png",)


def copy_ckla_figures() -> None:
    for name in CKLA_FIGURES:
        shutil.copyfile(CKLA_REPORT_ASSETS / name, ASSETS / name)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2A — augmentation-tuple figures (base set, expanded set, freq-shift
# failure). Source clip: local DREGON-LM-V4-michaels-full/valid, loaded the
# same way scripts/rps_predictor_vk_eval.py::load_clip_data does (via
# DregonLMFrameDataset, channel=None). All spectrogram/RPS grids share axis
# ranges within a figure so absolute changes are visible at a glance.
# ═══════════════════════════════════════════════════════════════════════════

RPS_SR = 16000
RPS_HOP = 512
RPS_NFFT = 2048
RPS_FRAME_HZ = RPS_SR / RPS_HOP  # ~31.25 Hz, the label rate used below
LOCAL_VALID = ROOT / "datasets" / "DREGON-LM-V4-michaels-full" / "valid"


def load_sample(clip_idx: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """One local valid clip -> (audio ch0 (T,) float32, rps (4, F) float32)."""
    from data_processing.frame_datasets import DregonLMFrameDataset

    ds = DregonLMFrameDataset(
        str(LOCAL_VALID), n_fft=RPS_NFFT, hop_length=RPS_HOP, sample_rate=RPS_SR, channel=None
    )
    fr = ds[clip_idx]
    audio = np.atleast_2d(np.asarray(fr["mixture"].data, dtype=np.float32))
    gt = np.asarray(fr["rps"].data, dtype=np.float32)
    return audio[0].copy(), gt


def _spec_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import scipy.signal as sps

    f, t, z = sps.stft(x, fs=RPS_SR, nperseg=RPS_NFFT, noverlap=RPS_NFFT - RPS_HOP)
    return f, t, 20.0 * np.log10(np.abs(z) + 1e-8)


ROTOR_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def tuple_grid(
    cases: list[tuple[str, np.ndarray, np.ndarray]],
    out_name: str,
    *,
    freq_max: float = 1200.0,
    highlight: int | None = None,
    vmin: float = -70.0,
    vmax: float = 10.0,
    t_max: float | None = None,
    rps_max: float | None = None,
    rps_min: float | None = None,
) -> None:
    """Row 1: spectrograms (shared colour scale, shared f/t axes).
    Row 2: the 4 rotor-speed curves (shared y-axis).
    ``cases`` = [(title, audio_ch0, rps (4,F)), ...]; all panels share x/y
    ranges so absolute shifts (esp. in RPS) are directly comparable.

    No colour bar: it stole width from the top row only, so the spectrogram
    panel and the RPS panel underneath no longer lined up. The dB scale is
    fixed and identical everywhere, so the bar carried no information the
    caption cannot state."""
    n = len(cases)
    fig, axes = plt.subplots(
        2, n, figsize=(3.4 * n, 5.4), sharex="col", gridspec_kw={"height_ratios": (2.0, 1.3)}
    )
    if n == 1:
        axes = axes.reshape(2, 1)
    if t_max is None:
        t_max = max(len(a) / RPS_SR for _, a, _ in cases)
    if rps_max is None:
        rps_max = max(float(np.max(r)) for _, _, r in cases) * 1.08
    if rps_min is None:
        # ignore the zero-filled tail freq_scale leaves behind when it shortens
        # a clip, otherwise one padded panel drags the shared floor to 0.
        rps_min = min(float(np.min(r[r > 0])) for _, _, r in cases) * 0.88
    im = None
    for i, (title, audio, rps) in enumerate(cases):
        f, t, s_db = _spec_db(audio)
        im = axes[0, i].pcolormesh(t, f, s_db, vmin=vmin, vmax=vmax, cmap="magma", shading="auto")
        axes[0, i].set_ylim(0, freq_max)
        axes[0, i].set_xlim(0, t_max)
        axes[0, i].set_title(title, fontsize=13)
        if i == 0:
            axes[0, i].set_ylabel("Hz")
        t_label = np.arange(rps.shape[-1]) / RPS_FRAME_HZ
        for r in range(rps.shape[0]):
            axes[1, i].plot(t_label, rps[r], color=ROTOR_COLORS[r], lw=1.8)
        axes[1, i].set_ylim(rps_min, rps_max)
        axes[1, i].set_xlim(0, t_max)
        axes[1, i].set_xticks(np.arange(0, t_max, 2.0))
        axes[1, i].set_xlabel("s")
        if i == 0:
            axes[1, i].set_ylabel("rev/s")
        else:
            # identical ranges across panels -> tick labels only on the left
            # column (keeps neighbouring panels from clipping each other).
            axes[0, i].tick_params(labelleft=False)
            axes[1, i].tick_params(labelleft=False)
        if highlight is not None and i == highlight:
            for ax in (axes[0, i], axes[1, i]):
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d62728")
                    spine.set_linewidth(3.0)
    del im
    fig.subplots_adjust(wspace=0.11, hspace=0.14, left=0.06, right=0.995, top=0.93, bottom=0.12)
    fig.savefig(ASSETS / out_name, dpi=150)
    plt.close(fig)


def _freq_shift_numpy(
    audio: np.ndarray, label: np.ndarray, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Same convention as noise_augmentations._freq_scale (rescale audio+RPS
    together by ``ratio``, crop/pad to original length), fixed ratio (no
    RNG draw) — used for the deterministic 2%/10% probe."""
    import soxr

    from data_processing.noise_augmentations import _fit_len

    T = audio.shape[-1]
    y = soxr.resample(audio.astype(np.float64), float(RPS_SR), RPS_SR / ratio)
    out = _fit_len(np.asarray(y, dtype=np.float32), T)
    F = label.shape[-1]
    t_label = np.arange(F, dtype=np.float64) / RPS_FRAME_HZ
    dur = T / float(RPS_SR)
    src_t = ratio * t_label
    in_range = src_t <= dur + 1e-9
    new_label = np.empty_like(label)
    for r in range(label.shape[0]):
        new_label[r] = np.where(
            in_range, ratio * np.interp(src_t, t_label, label[r].astype(np.float64)), 0.0
        ).astype(np.float32)
    return out, new_label


def _warp_numpy(
    audio: np.ndarray, label: np.ndarray, seed: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Visualization-only reimplementation of noise_time_warp's
    alpha(t)=c+a*sin(2*pi*f*t+phi) resampling, applied directly to the
    already-extracted (audio, frame-rate-label) pair instead of a td.Frame
    (the production apply_time_warp() needs extra source margin beyond one
    clip's audio + a StampIndex motor track; this keeps the same closed-form
    tau(t) and interpolation, just skips the Frame plumbing — logged in
    creator-log.md)."""
    rng = np.random.default_rng(seed)
    c = float(rng.uniform(1 - 0.08, 1 + 0.08))
    a = float(rng.uniform(0, 0.04))
    f = float(rng.uniform(0.1, 1.0))
    phi = float(rng.uniform(0, 2 * np.pi))

    def tau(t: np.ndarray) -> np.ndarray:
        return c * t + (a / (2 * np.pi * f)) * (np.cos(phi) - np.cos(2 * np.pi * f * t + phi))

    T = audio.shape[-1]
    t_tgt = np.arange(T, dtype=np.float64) / RPS_SR
    src_pos = tau(t_tgt) * RPS_SR
    src_pos = np.clip(src_pos, 0, T - 1)
    warped = np.interp(src_pos, np.arange(T, dtype=np.float64), audio.astype(np.float64))

    F = label.shape[-1]
    t_label = np.arange(F, dtype=np.float64) / RPS_FRAME_HZ
    tau_l = np.clip(tau(t_label), 0, (T - 1) / RPS_SR)
    alpha_l = c + a * np.sin(2 * np.pi * f * t_label + phi)
    new_label = np.empty_like(label)
    for r in range(label.shape[0]):
        new_label[r] = alpha_l * np.interp(tau_l, t_label, label[r].astype(np.float64))
    return warped.astype(np.float32), new_label


def fig_base_augs() -> None:
    from data_processing.online_mixing import _apply_one_augmentation

    audio, label = load_sample()
    rng = np.random.default_rng(0)
    gain = _apply_one_augmentation(
        audio[None, :],
        {"probability": 1.0, "choices": [{"random_gain": {"min_db": 6, "max_db": 6}}]},
        rng,
    )[0]
    polarity = _apply_one_augmentation(
        audio[None, :], {"probability": 1.0, "choices": ["random_polarity"]}, rng
    )[0]
    # Deterministic drop (the real transform picks the channel at random; we
    # force it here so the illustration always shows the visible effect —
    # the dropped channel going silent — rather than a coin flip).
    stereo = np.stack([audio, np.roll(audio, 37)])
    dropped = stereo.copy()
    dropped[1, :] = 0.0
    warped_audio, warped_label = _warp_numpy(audio, label, seed=3)

    tuple_grid(
        [
            ("clean", audio, label),
            ("random_gain (+6 dB)", gain, label),
            ("random_polarity", polarity, label),
            ("channel_drop (other mic zeroed)", dropped[1], label),
            ("noise_time_warp", warped_audio, warped_label),
        ],
        "aug_base_grid.png",
    )


def fig_expanded_augs() -> None:
    from data_processing.noise_augmentations import (
        _floor_inject,
        _freq_scale,
        _random_reverb,
        _spec_mask,
        _spectral_recolor,
        _tooth_dropout,
    )

    audio, label = load_sample()
    a2 = audio[None, :]
    sr = RPS_SR
    lr = RPS_FRAME_HZ
    rng = np.random.default_rng(1)
    fs_a, fs_l = _freq_scale(
        a2, label, {"alpha_low": 1.2, "alpha_high": 1.2}, rng, sample_rate=sr, label_rate_hz=lr
    )
    rc_a, _ = _spectral_recolor(a2, label, {}, rng, sample_rate=sr, label_rate_hz=lr)
    rv_a, _ = _random_reverb(a2, label, {}, rng, sample_rate=sr, label_rate_hz=lr)
    td_a, _ = _tooth_dropout(a2, label, {}, rng, sample_rate=sr, label_rate_hz=lr)
    sm_a, _ = _spec_mask(a2, label, {}, rng, sample_rate=sr, label_rate_hz=lr)
    fi_a, _ = _floor_inject(a2, label, {}, rng, sample_rate=sr, label_rate_hz=lr)

    grid1 = [
        ("clean", audio, label),
        ("freq_scale (α=1.2)", fs_a[0], fs_l),
        ("spectral_recolor", rc_a[0], label),
        ("random_reverb", rv_a[0], label),
    ]
    grid2 = [
        ("clean", audio, label),
        ("tooth_dropout", td_a[0], label),
        ("spec_mask", sm_a[0], label),
        ("floor_inject", fi_a[0], label),
    ]
    # Both halves of the expanded set must be directly comparable: same
    # clean reference panel first, and identical t/RPS ranges across figures.
    shared_t = max(len(a) / RPS_SR for _, a, _ in grid1 + grid2)
    shared_rps = max(float(np.max(r)) for _, _, r in grid1 + grid2) * 1.08
    shared_rps_lo = min(float(np.min(r[r > 0])) for _, _, r in grid1 + grid2) * 0.88
    tuple_grid(
        grid1,
        "aug_expanded_grid1.png",
        highlight=1,
        t_max=shared_t,
        rps_max=shared_rps,
        rps_min=shared_rps_lo,
    )
    tuple_grid(
        grid2,
        "aug_expanded_grid2.png",
        freq_max=1200.0,
        t_max=shared_t,
        rps_max=shared_rps,
        rps_min=shared_rps_lo,
    )


def fig_freqshift_predictions() -> None:
    """g2_if_transformer (NO freq-shift augmentation) predictions on a
    regular clip vs the same clip uniformly frequency-shifted 2% / 10%."""
    import rps_predictor_vk_eval as vk
    import torch

    audio, label = load_sample()
    model = vk.load_model(
        "g2_if_transformer", "r2://ml-data/artifacts/g2_if_transformer/checkpoints/best.ckpt", "cpu"
    )
    win = audio.shape[-1]

    def predict(a: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(a[None, :]))
        return out[0].numpy()

    cases = []
    for name, ratio in (("no shift", 1.0), ("2% shift", 1.02), ("10% shift", 1.10)):
        a, lab = (audio, label) if ratio == 1.0 else _freq_shift_numpy(audio, label, ratio)
        pred = predict(a)
        pred = vk.perm_align(pred.astype(np.float64), lab.astype(np.float64))
        cases.append((name, a, lab, pred))

    fig, axes = plt.subplots(2, len(cases), figsize=(3.8 * len(cases), 5.6), sharex="col")
    # Rescaling by `ratio` leaves the last `1 - 1/ratio` of the shifted clip
    # without a valid label (it is zero-filled), which shows up as an
    # end-of-clip plunge. Crop every panel to the duration that is valid in
    # all of them so only the real failure mode is on screen.
    t_max = win / RPS_SR / 1.10
    n_valid = int(np.floor(t_max * RPS_FRAME_HZ))
    cases = [(name, a, lab[:, :n_valid], pred[:, :n_valid]) for name, a, lab, pred in cases]
    rps_max = max(float(max(lab.max(), pred.max())) for _, _, lab, pred in cases) * 1.1
    for i, (name, a, lab, pred) in enumerate(cases):
        f, t, s_db = _spec_db(a)
        axes[0, i].pcolormesh(t, f, s_db, vmin=-70, vmax=10, cmap="magma", shading="auto")
        axes[0, i].set_ylim(0, 1200)
        axes[0, i].set_xlim(0, t_max)
        axes[0, i].set_title(name, fontsize=13)
        if i == 0:
            axes[0, i].set_ylabel("Hz")
        t_l = np.arange(lab.shape[-1]) / RPS_FRAME_HZ
        for r in range(4):
            axes[1, i].plot(t_l, lab[r], color=ROTOR_COLORS[r], lw=1.4, ls="--", alpha=0.7)
            axes[1, i].plot(t_l, pred[r], color=ROTOR_COLORS[r], lw=2.2)
        axes[1, i].set_ylim(0, rps_max)
        axes[1, i].set_xlim(0, t_max)
        axes[1, i].set_xlabel("s")
        if i == 0:
            axes[1, i].set_ylabel("rev/s")
        else:
            axes[0, i].tick_params(labelleft=False)
            axes[1, i].tick_params(labelleft=False)
    fig.savefig(ASSETS / "freqshift_predictions.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


OLD_CKPT = "r2://ml-data/artifacts/g2_if_transformer/checkpoints/best.ckpt"
NEW_CKPT = "r2://ml-data/artifacts/g2_if_freqscale_v2/checkpoints/best.ckpt"
NEW_EXP = "g2_if_freqscale_v2"


def fig_freqshift_both() -> None:
    """Same probe, both regimes: old (no freq_scale) vs new (freq_scale
    firing). Row 1 = spectrogram of the clip, row 2 = old-regime prediction,
    row 3 = new-regime prediction. Dashed = ground truth."""
    import rps_predictor_vk_eval as vk
    import torch

    audio, label = load_sample()
    models = [
        ("old regime — no freq_scale", vk.load_model("g2_if_transformer", OLD_CKPT, "cpu")),
        ("uniform freq_scale (v2)", vk.load_model(NEW_EXP, NEW_CKPT, "cpu")),
    ]
    win = audio.shape[-1]
    t_max = win / RPS_SR / 1.10
    n_valid = int(np.floor(t_max * RPS_FRAME_HZ))

    cases = []
    for name, ratio in (("no shift", 1.0), ("2% shift", 1.02), ("10% shift", 1.10)):
        a, lab = (audio, label) if ratio == 1.0 else _freq_shift_numpy(audio, label, ratio)
        preds = []
        for _, m in models:
            with torch.no_grad():
                p = m(torch.from_numpy(a[None, :]))[0].numpy()
            preds.append(vk.perm_align(p.astype(np.float64), lab.astype(np.float64))[:, :n_valid])
        cases.append((name, a, lab[:, :n_valid], preds))

    rps_max = (
        max(max(float(lab.max()), *(float(p.max()) for p in ps)) for _, _, lab, ps in cases) * 1.06
    )
    rps_min = (
        min(min(float(lab.min()), *(float(p.min()) for p in ps)) for _, _, lab, ps in cases) * 0.94
    )
    fig, axes = plt.subplots(
        3,
        len(cases),
        figsize=(4.0 * len(cases), 7.4),
        sharex="col",
        gridspec_kw={"height_ratios": (1.5, 1.0, 1.0)},
    )
    for i, (name, a, lab, preds) in enumerate(cases):
        f, t, s_db = _spec_db(a)
        axes[0, i].pcolormesh(t, f, s_db, vmin=-70, vmax=10, cmap="magma", shading="auto")
        axes[0, i].set_ylim(0, 1200)
        axes[0, i].set_title(name, fontsize=15)
        axes[0, i].grid(False)
        t_l = np.arange(lab.shape[-1]) / RPS_FRAME_HZ
        for row, (mname, _) in enumerate(models, start=1):
            ax = axes[row, i]
            for r in range(4):
                ax.plot(t_l, lab[r], color=ROTOR_COLORS[r], lw=1.4, ls="--", alpha=0.7)
                ax.plot(t_l, preds[row - 1][r], color=ROTOR_COLORS[r], lw=2.2)
            ax.set_ylim(rps_min, rps_max)
            ax.set_xlim(0, t_max)
            m_gt = float(np.mean(lab))
            m_pr = float(np.mean(preds[row - 1]))
            ax.text(
                0.02,
                0.06,
                f"mean truth {m_gt:.1f} · mean pred {m_pr:.1f} rev/s",
                transform=ax.transAxes,
                fontsize=11,
                color=INK,
            )
            print(f"  {name:>10} | {mname:<32} GT {m_gt:6.2f}  pred {m_pr:6.2f}")
            if i == 0:
                ax.set_ylabel(mname.split(" — ")[0] + "\nrev/s", fontsize=13)
        axes[2, i].set_xlabel("s")
        axes[0, i].set_xlim(0, t_max)
        if i == 0:
            axes[0, i].set_ylabel("Hz")
        else:
            for row in range(3):
                axes[row, i].tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.06, hspace=0.16, left=0.09, right=0.995, top=0.95, bottom=0.07)
    fig.savefig(ASSETS / "freqshift_both.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2A' — "where did the augmentation actually help": per-regime PIT-MAE
# for both transformer regimes over the whole 37-clip valid-full set.
# Loader/alignment follow scripts/rps_predictor_vk_eval.py exactly (same
# DregonLMFrameDataset read, same Hungarian per-clip alignment); clip -> source
# comes from that script's embedded CLIPS table. Regimes are per FRAME, on the
# rotor-mean ground truth: zero (<1 rev/s), warmup (1-50), free-flight (>=50).
# ═══════════════════════════════════════════════════════════════════════════

VALID_FULL = ROOT / "datasets" / "DREGON-LM-V4-michaels-full" / "valid"
REGIME_CSV = ASSETS / "regime_mae.csv"


def _regime_masks(gt: np.ndarray) -> dict[str, np.ndarray]:
    m = gt.mean(axis=0)  # (F,) rotor-mean GT per frame
    return {"zero": m < 1.0, "warmup": (m >= 1.0) & (m < 50.0), "free": m >= 50.0}


def table_regime_mae() -> None:
    import rps_predictor_vk_eval as vk
    import torch

    torch.set_num_threads(4)
    audio, gt = vk.load_clip_data(str(VALID_FULL))
    source = {c[0]: ("FLY124" if c[1] == "michaels_FLY124" else "DREGON") for c in vk.CLIPS}

    arms = [("old", "g2_if_transformer", OLD_CKPT), ("new", NEW_EXP, NEW_CKPT)]
    rows = []
    for arm, exp, ckpt in arms:
        model = vk.load_model(exp, ckpt, "cpu")
        # (source, regime) -> [sum abs err, n frames*rotors]
        acc: dict[tuple[str, str], list[float]] = {}
        tot = [0.0, 0.0]
        for clip, a in audio.items():
            g = gt[clip].astype(np.float64)
            with torch.no_grad():
                p = model(torch.from_numpy(a[0][None, :]))[0].numpy().astype(np.float64)
            p = vk.perm_align(p, g)
            err = np.abs(p - g)  # (4, F)
            tot[0] += float(err.sum())
            tot[1] += err.size
            for reg, mask in _regime_masks(g).items():
                if not mask.any():
                    continue
                k = (source[clip], reg)
                s = acc.setdefault(k, [0.0, 0.0])
                s[0] += float(err[:, mask].sum())
                s[1] += float(err[:, mask].size)
        for (src, reg), (s, n) in sorted(acc.items()):
            rows.append({"arm": arm, "source": src, "regime": reg, "mae": s / n, "frames": n / 4})
        rows.append(
            {"arm": arm, "source": "ALL", "regime": "all", "mae": tot[0] / tot[1], "frames": ""}
        )
        del model
    df = pd.DataFrame(rows)
    df.to_csv(REGIME_CSV, index=False)
    print(df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# Part 2B — CKLA results (WIP) prediction overlay from the current best
# checkpoint.
# ═══════════════════════════════════════════════════════════════════════════


CKLA_CKPT = "r2://ml-data/artifacts/ckla_pnoise_fs_v2/checkpoints/best.ckpt"
CKLA_EXP = "ckla_pnoise_fs_v2"


def fig_ckla_freqshift() -> None:
    """ITEM 3 probe: does the current best CKLA model follow a frequency
    shift? Same machinery/axes as ``fig_freqshift_both``, one model row.
    Whatever it does is what gets drawn — the scale ratio is printed and
    annotated on the panel."""
    import rps_predictor_vk_eval as vk
    import torch

    audio, label = load_sample()
    model = vk.load_model(CKLA_EXP, CKLA_CKPT, "cpu")
    win = audio.shape[-1]
    t_max = win / RPS_SR / 1.10
    n_valid = int(np.floor(t_max * RPS_FRAME_HZ))

    cases = []
    for name, ratio in (("no shift", 1.0), ("2% shift", 1.02), ("10% shift", 1.10)):
        a, lab = (audio, label) if ratio == 1.0 else _freq_shift_numpy(audio, label, ratio)
        with torch.no_grad():
            p = model(torch.from_numpy(a[None, :]))[0].numpy()
        p = vk.perm_align(p.astype(np.float64), lab.astype(np.float64))[:, :n_valid]
        cases.append((name, ratio, a, lab[:, :n_valid], p))

    base_pred = float(np.mean(cases[0][4]))
    rps_max = max(max(float(lab.max()), float(p.max())) for _, _, _, lab, p in cases) * 1.06
    rps_min = min(min(float(lab.min()), float(p.min())) for _, _, _, lab, p in cases) * 0.94
    fig, axes = plt.subplots(
        2,
        len(cases),
        figsize=(4.0 * len(cases), 5.4),
        sharex="col",
        gridspec_kw={"height_ratios": (1.4, 1.0)},
    )
    for i, (name, ratio, a, lab, p) in enumerate(cases):
        f, t, s_db = _spec_db(a)
        axes[0, i].pcolormesh(t, f, s_db, vmin=-70, vmax=10, cmap="magma", shading="auto")
        axes[0, i].set_ylim(0, 1200)
        axes[0, i].set_title(name, fontsize=15)
        axes[0, i].set_xlim(0, t_max)
        axes[0, i].grid(False)
        ax = axes[1, i]
        t_l = np.arange(lab.shape[-1]) / RPS_FRAME_HZ
        for r in range(4):
            ax.plot(t_l, lab[r], color=ROTOR_COLORS[r], lw=1.4, ls="--", alpha=0.7)
            ax.plot(t_l, p[r], color=ROTOR_COLORS[r], lw=2.2)
        ax.set_ylim(rps_min, rps_max)
        ax.set_xlim(0, t_max)
        ax.set_xlabel("s")
        m_gt, m_pr = float(np.mean(lab)), float(np.mean(p))
        ratio_obs = m_pr / base_pred
        ax.text(
            0.02,
            0.06,
            f"truth {m_gt:.1f} · pred {m_pr:.1f} rev/s\npred ×{ratio_obs:.3f} (ideal ×{ratio:.2f})",
            transform=ax.transAxes,
            fontsize=11,
            color=INK,
        )
        print(f"  CKLA {name:>10}: GT {m_gt:6.2f} pred {m_pr:6.2f} ratio {ratio_obs:.4f}")
        if i == 0:
            axes[0, i].set_ylabel("Hz")
            ax.set_ylabel("CKLA\nrev/s", fontsize=13)
        else:
            axes[0, i].tick_params(labelleft=False)
            ax.tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.06, hspace=0.16, left=0.09, right=0.995, top=0.93, bottom=0.10)
    fig.savefig(ASSETS / "ckla_freqshift.png", dpi=150)
    plt.close(fig)


def fig_ckla_prediction_overlay() -> None:
    import rps_predictor_vk_eval as vk
    import torch

    audio, label = load_sample(clip_idx=1)
    model = vk.load_model(
        "ckla_p1_pnoise", "r2://ml-data/artifacts/ckla_p1_pnoise_pb/checkpoints/best.ckpt", "cpu"
    )
    with torch.no_grad():
        pred = model(torch.from_numpy(audio[None, :]))[0].numpy()
    pred = vk.perm_align(pred.astype(np.float64), label.astype(np.float64))

    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw={"height_ratios": (1.6, 1.0)}
    )
    f, t, s_db = _spec_db(audio)
    axes[0].pcolormesh(t, f, s_db, vmin=-70, vmax=10, cmap="magma", shading="auto")
    axes[0].set_ylim(0, 1200)
    axes[0].set_ylabel("Hz")
    axes[0].set_title("CKLA, current run (WIP) — one DREGON cruise clip")
    t_l = np.arange(label.shape[-1]) / RPS_FRAME_HZ
    for r in range(4):
        axes[1].plot(
            t_l,
            label[r],
            color=ROTOR_COLORS[r],
            lw=1.6,
            ls="--",
            alpha=0.7,
            label=f"GT r{r}" if r == 0 else None,
        )
        axes[1].plot(t_l, pred[r], color=ROTOR_COLORS[r], lw=2.4, label="pred" if r == 0 else None)
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("rev/s")
    axes[1].legend(loc="upper right", fontsize=10)
    fig.savefig(ASSETS / "ckla_prediction_overlay.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


SEEN = "#1f77b4"
UNSEEN = "#d62728"
CTRL = "#7f7f7f"
MPSE = "#2ca02c"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "figure.dpi": 170,
    }
)


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(F2 / name)
    return pd.DataFrame(d[d.groupby("clip_id")["si_sdr"].transform("max") > -70])


def bysnr(d: pd.DataFrame, cat: str | None = None) -> pd.DataFrame:
    if cat is not None:
        d = pd.DataFrame(d[d["category"] == cat])
    cols = [c for c in ("si_sdr", "estoi", "pesq", "pesq_nb", "gain_db", "corr") if c in d.columns]
    return pd.DataFrame(d.groupby("input_snr")[cols].mean()).sort_index()


def fig_seen_unseen() -> None:
    """The money slide: one model, two halves, only training exposure differs."""
    probe = load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv")
    noisy = load("noisy__SE-valid-avq-split.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for ax, metric, ylab in zip(
        axes, ("si_sdr", "estoi"), ("output SI-SDR (dB)", "eSTOI"), strict=True
    ):
        for cat, colour, name in [
            ("avq_ego_s1", SEEN, "noise it TRAINED on"),
            ("avq_ego_s2", UNSEEN, "noise it NEVER heard"),
        ]:
            g = bysnr(probe, cat)
            ax.plot(g.index, g[metric], color=colour, marker="o", ms=8, lw=3, label=name)
        g = bysnr(noisy, "avq_ego_s2")
        ax.plot(g.index, g[metric], color="black", lw=1.6, alpha=0.6, ls="--", label="do nothing")
        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="lower right", framealpha=0.95)
    axes[0].annotate(
        "12.9 dB",
        xy=(-14.3, -1.6),
        ha="left",
        fontsize=17,
        color=UNSEEN,
        weight="bold",
    )
    axes[0].annotate(
        "", xy=(-15, 3.6), xytext=(-15, -9.3), arrowprops=dict(arrowstyle="<->", color=UNSEEN, lw=2)
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "seen_unseen.png", bbox_inches="tight")
    plt.close(fig)


def fig_control() -> None:
    """Same picture for the model that trained on ALL five recordings."""
    ctrl = load("f2_dcunet_avq_survey__SE-valid-avq-split.csv")
    probe = load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for ax, d, title in zip(
        axes,
        (ctrl, probe),
        ("trained on ALL 5 recordings", "trained on session 1 only"),
        strict=True,
    ):
        for cat, colour, name in [
            ("avq_ego_s1", SEEN, "session 1"),
            ("avq_ego_s2", UNSEEN, "session 2"),
        ]:
            g = bysnr(d, cat)
            ax.plot(g.index, g["si_sdr"], color=colour, marker="o", ms=8, lw=3, label=name)
        ax.set_title(title)
        ax.set_xlabel("input SNR (dB)")
    axes[0].set_ylabel("output SI-SDR (dB)")
    axes[0].legend(loc="upper left")
    for ax, label, colour in (
        (axes[0], "gap 0.3 dB", CTRL),
        (axes[1], "gap 12.9 dB", UNSEEN),
    ):
        ax.text(
            0.5,
            0.05,
            label,
            transform=ax.transAxes,
            ha="center",
            color=colour,
            fontsize=15,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "control.png", bbox_inches="tight")
    plt.close(fig)


def fig_ladder() -> None:
    """Widening the training pool destroys intelligibility, not energy."""
    noisy = bysnr(load("noisy__SE-valid-avq-survey.csv"))
    arms = [
        ("AVQ only (100%)", "f2_dcunet_avq_survey__SE-valid-avq-survey.csv", "#1f77b4"),
        ("+ all drone (14%)", "f2_dcunet_alldrone__SE-valid-avq-survey.csv", "#ff7f0e"),
        ("+ all harmonic (2%)", "f2_dcunet_allharmonic__SE-valid-avq-survey.csv", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for label, csv_name, colour in arms:
        g = bysnr(load(csv_name))
        axes[0].plot(
            g.index, g["estoi"] - noisy["estoi"], marker="o", ms=8, color=colour, lw=3, label=label
        )
        axes[1].plot(
            g.index,
            g["si_sdr"] - noisy["si_sdr"],
            marker="o",
            ms=8,
            color=colour,
            lw=3,
            label=label,
        )
    axes[0].axhline(0, color="black", lw=1.5)
    axes[0].set_ylabel("ΔeSTOI  (speech recovered)")
    axes[0].set_title("intelligibility gain collapses")
    axes[1].set_ylabel("ΔSI-SDR (dB)  (energy removed)")
    axes[1].set_title("…energy gain survives")
    for ax in axes:
        ax.set_xlabel("input SNR (dB)")
    axes[0].legend(title="AVQ share of training", fontsize=11, title_fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "ladder.png", bbox_inches="tight")
    plt.close(fig)


def fig_mpsenet() -> None:
    """The architecture control, on unseen noise only."""
    noisy = bysnr(load("noisy__SE-valid-avq-survey.csv"))
    noisy_s2 = bysnr(load("noisy__SE-valid-avq-split.csv"), "avq_ego_s2")
    mp = bysnr(load("f1_mpsenet_a__SE-valid-avq-survey.csv"))
    probe = bysnr(load("f2_dcunet_avq_heldout__SE-valid-avq-split.csv"), "avq_ego_s2")

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(
        mp.index,
        mp["estoi"] - noisy["estoi"],
        marker="*",
        ms=16,
        color=MPSE,
        lw=3,
        label="MP-SENet — never heard this drone",
    )
    ax.plot(
        probe.index,
        probe["estoi"] - noisy_s2["estoi"],
        marker="D",
        ms=9,
        color=UNSEEN,
        lw=3,
        label="DCUNet — same drone, unseen session",
    )
    ax.axhline(0, color="black", lw=1.5)
    ax.set_xlabel("input SNR (dB)")
    ax.set_ylabel("ΔeSTOI vs doing nothing")
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "mpsenet.png", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — F1 blind-baseline per-SNR curves on SE-valid-drone, and output
# spectrograms. Source: results/f1_perclip/*.csv (one row per clip, all 350
# clips of SE-valid-drone) + the local best.ckpt of each F1 Pass-A arm.
# Model colours are fixed here and used by EVERY Part-1 figure.
# ═══════════════════════════════════════════════════════════════════════════

F1 = ROOT / "results" / "f1_perclip"

# name -> (colour, marker); shared across all Part-1 model figures.
MODEL_STYLE = {
    "MP-SENet": ("#2ca02c", "*"),
    "TF-GridNet": ("#9467bd", "s"),
    "Edge-BS-RoFormer": ("#ff7f0e", "^"),
    "DCUNet": ("#d62728", "D"),
    "SGMSE+": ("#8c564b", "v"),
    "Wiener": ("#7f7f7f", "x"),
}
ARCH_SLUG = {
    "MP-SENet": "mpsenet",
    "TF-GridNet": "tfgridnet",
    "Edge-BS-RoFormer": "edge_bs_rof",
    "DCUNet": "dcunet",
    "SGMSE+": "sgmse",
}


def _f1_bad_clips() -> list[str]:
    """The 5 digitally-silent SE-valid-drone clips (empty target -> SI-SDR
    pinned at the -80 dB floor). Identified from the noisy anchor and dropped
    from every method so all curves are over the same 345 clips."""
    d = pd.read_csv(F1 / "noisy__SE-valid-drone.csv")
    return [str(c) for c in d.loc[d["si_sdr"] < -70, "clip_id"]]


def _f1_bysnr(fname: str, bad: list[str]) -> pd.DataFrame:
    d = pd.read_csv(F1 / fname)
    d = pd.DataFrame(d[~d["clip_id"].isin(bad)])
    return pd.DataFrame(d.groupby("input_snr")[["si_sdr", "estoi"]].mean()).sort_index()


def fig_f1_persnr(pass_: str) -> None:
    """All F1 models of one training pass on SE-valid-drone: SI-SDR | eSTOI."""
    bad = _f1_bad_clips()
    noisy = _f1_bysnr("noisy__SE-valid-drone.csv", bad)
    wiener = _f1_bysnr("wiener__SE-valid-drone.csv", bad)

    series = []
    for name in ("MP-SENet", "TF-GridNet", "Edge-BS-RoFormer", "DCUNet", "SGMSE+"):
        f = F1 / f"f1_{ARCH_SLUG[name]}_{pass_}__SE-valid-drone.csv"
        if f.exists():
            series.append((name, _f1_bysnr(f.name, bad)))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, metric, ylab in zip(
        axes, ("si_sdr", "estoi"), ("output SI-SDR (dB)", "output eSTOI"), strict=True
    ):
        for name, g in series:
            colour, marker = MODEL_STYLE[name]
            ax.plot(g.index, g[metric], color=colour, marker=marker, ms=9, lw=2.6, label=name)
        ax.plot(
            wiener.index,
            wiener[metric],
            color=MODEL_STYLE["Wiener"][0],
            marker="x",
            ms=7,
            lw=1.6,
            ls=":",
            label="Wiener",
        )
        ax.plot(
            noisy.index,
            noisy[metric],
            color="black",
            lw=2.0,
            ls="--",
            label="do nothing (input)",
        )
        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel(ylab)
    axes[0].set_ylim(-36, 15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=4,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, -0.13),
    )
    fig.tight_layout()
    fig.savefig(ASSETS / f"f1_persnr_{pass_}.png", bbox_inches="tight")
    plt.close(fig)


SPEC_CLIPS = ((-15.0, "−15 dB input"), (-5.0, "−5 dB input"))
SPEC_DURATION_S = 8.0


class _Arr:
    """Minimal stand-in for a tdseries audio series: only ``.data`` is used
    by ``eval_se_perclip._estimates_model``."""

    def __init__(self, x: np.ndarray) -> None:
        self.data = x


def _long_drone_clips(duration_s: float = SPEC_DURATION_S) -> list[dict]:
    """Build one `duration_s`-long SE-valid-drone-style clip per SNR in
    SPEC_CLIPS.

    The published `SE-valid-drone` set stores 2 s clips only, so an 8 s
    example cannot be read off disk. We re-run the *builder's own* mixing
    code (`scripts/build_se_valid.py` pools + `_scale_source_to_snr`) at
    `duration_s`, with the same held-out speakers, the same drone noise pool
    and the same silence rejection — i.e. a clip drawn from exactly the
    valid-set distribution, just longer. Nothing is written to disk outside
    `assets/`."""
    import build_se_valid as bsv

    from data_processing.online_mixing import (
        _extract_audio_array,
        _is_silent,
        _scale_source_to_snr,
        build_noise_pool,
    )

    sr = 16000
    target_len = int(round(duration_s * sr))
    noise_pool = build_noise_pool(
        bsv.CATEGORY_NOISE["drone"], duration_s=duration_s, sample_rate=sr
    )
    speech_pool = bsv._heldout_speech_pool(duration_s, sr)
    clips = []
    for snr, _ in SPEC_CLIPS:
        rng = np.random.default_rng(20260727 + int(snr))
        for _ in range(40):
            noise_tf = noise_pool.sample_timeframe(rng, duration_s)
            noise = bsv._mono(_extract_audio_array(noise_tf, target_len=target_len), rng)
            speech = speech_pool.sample_mono(rng)
            if not _is_silent(noise) and not _is_silent(speech):
                break
        scaled = _scale_source_to_snr(speech[None, :], noise[None, :], float(snr))[0]
        mixture = (noise + scaled).astype(np.float32)
        clips.append({"mixture": _Arr(mixture), "target": _Arr(scaled.astype(np.float32))})
    return clips


def fig_f1_spectrograms() -> None:
    """noisy | each Pass-A model output | clean target, one row per clip.

    Runs the four discriminative F1 Pass-A checkpoints on CPU over two
    8-second clips built from the SE-valid-drone recipe. Every panel of a row
    shares one colour scale and one frequency axis, so panels are directly
    comparable."""
    import torch

    ds = _long_drone_clips()
    idxs = list(range(len(ds)))
    names = ("DCUNet", "Edge-BS-RoFormer", "MP-SENet", "TF-GridNet")

    from eval_se_perclip import (
        ARCH_MODEL,  # type: ignore[import-not-found]
        _estimates_model,  # type: ignore[import-not-found]
    )

    ests: dict[str, list[np.ndarray]] = {}
    for name in names:
        exp = f"f1_{ARCH_SLUG[name]}_a"
        # absolute paths: prepare.py runs from the slide dir, the eval helper
        # resolves conf/ and results/ relative to the repo root.
        ests[name] = _estimates_model(
            exp,
            ds,
            idxs,
            1,
            torch.device("cpu"),
            model_cfg_path=str(ROOT / "conf" / "model" / f"{ARCH_MODEL[ARCH_SLUG[name]]}.yaml"),
            ckpt_path=str(ROOT / "results" / exp / "best.ckpt"),
        )

    for row, (i, (_snr, label)) in enumerate(zip(idxs, SPEC_CLIPS, strict=True)):
        mix = np.asarray(ds[i]["mixture"].data, np.float32).reshape(-1)
        tgt = np.asarray(ds[i]["target"].data, np.float32).reshape(-1)
        panels = [("noisy input", mix)]
        panels += [(n, np.asarray(ests[n][row], np.float32).reshape(-1)) for n in names]
        panels.append(("clean target", tgt))

        # Chunked inference leaves the final partial chunk of some models
        # (DCUNet) as exact zeros. That is a harness artefact, not a model
        # property, so every panel is cropped to the region all models
        # actually produced.
        keep = min(
            int(np.max(np.nonzero(np.abs(x) > 0)[0]) + 1) if np.any(np.abs(x) > 0) else len(x)
            for _, x in panels
        )
        panels = [(t_, x[:keep]) for t_, x in panels]

        # 2 x 3 grid, not one long row: fills a 16:9 slide at legible size.
        fig, axgrid = plt.subplots(2, 3, figsize=(11.4, 6.0), sharey=True, sharex=True)
        axes = list(axgrid.reshape(-1))
        ref = 20.0 * np.log10(np.abs(mix).max() + 1e-8)
        for ax, (title, x) in zip(axes, panels, strict=True):
            f, t, s_db = _spec_db(x / (np.abs(x).max() + 1e-8))
            ax.pcolormesh(t, f / 1000.0, s_db, vmin=-80, vmax=0, cmap="magma", shading="auto")
            ax.set_title(title, fontsize=15)
            ax.grid(False)
        for ax in axes[3:]:
            ax.set_xlabel("s")
        for ax in (axes[0], axes[3]):
            ax.set_ylabel("kHz")
        fig.suptitle(f"{label} — each panel peak-normalised, same dB scale", fontsize=15)
        fig.tight_layout()
        fig.savefig(ASSETS / f"f1_spec_{row + 1}.png", bbox_inches="tight")
        plt.close(fig)
        del ref


# ═══════════════════════════════════════════════════════════════════════════
# Part 2C — explanatory schematics for the CKLA story. These are drawings,
# not measurements: no number on them comes from a run, and none is claimed
# to. They exist so each conference-audience slide has one picture.
# ═══════════════════════════════════════════════════════════════════════════

INK = "#22303c"
ACC = "#d62728"
BLUE = "#1f77b4"


def _box(ax, x, y, w, h, text, *, fc="#eef3f8", ec=INK, fs=13, weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            fc=fc,
            ec=ec,
            lw=1.8,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight
    )


def _arrow(ax, xy_from, xy_to, *, color=INK, rad=0.0, lw=2.0, text=None, dx=0.0, dy=0.06):
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )
    if text:
        ax.text(
            (xy_from[0] + xy_to[0]) / 2 + dx,
            (xy_from[1] + xy_to[1]) / 2 + dy,
            text,
            ha="center",
            va="bottom",
            fontsize=12,
            color=color,
        )


def _blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)


def _dia_comb() -> None:
    """A rotor's sound is a comb; the spacing between teeth IS the rotor
    speed. Idealised sketch, not measured data."""
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    t = np.linspace(0, 8, 800)
    f0 = 42 + 4.0 * np.sin(2 * np.pi * 0.12 * t) + 1.2 * np.sin(2 * np.pi * 0.5 * t)
    ax.set_facecolor("#12070f")
    for k in range(1, 13):
        ax.plot(t, k * f0, color="#f7c873", lw=2.2, alpha=0.85 if k > 2 else 0.5)
    ax.plot(t, 5 * f0, color=ACC, lw=3.4)
    ax.annotate(
        "one harmonic track:\n5 × rotor speed",
        xy=(6.4, 5 * f0[640]),
        xytext=(3.9, 500),
        color=ACC,
        fontsize=14,
        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2),
    )
    xb = 1.4
    j = int(xb / 8 * 800)
    ax.annotate(
        "",
        xy=(xb, 8 * f0[j]),
        xytext=(xb, 9 * f0[j]),
        arrowprops=dict(arrowstyle="<|-|>", color="white", lw=2.0, mutation_scale=10),
    )
    ax.text(
        xb + 0.15,
        8.4 * f0[j],
        "gap between teeth\n= the rotor speed",
        color="white",
        fontsize=13,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_ylim(0, 560)
    ax.set_xlim(0, 8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_comb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dia_vk_loop() -> None:
    """Vold–Kalman order tracking as an alternating two-step loop."""
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    _blank(ax)
    _box(
        ax,
        0.05,
        0.52,
        0.38,
        0.30,
        "Fix the frequency track\n→ solve the envelopes\n(one big least squares)",
    )
    ax.set_xlim(-0.01, 1.01)
    _box(ax, 0.62, 0.52, 0.36, 0.30, "Fix the envelopes\n→ re-estimate the\nfrequency track")
    _arrow(ax, (0.44, 0.74), (0.61, 0.74), rad=-0.30, text="residual", dy=0.09)
    _arrow(ax, (0.61, 0.60), (0.44, 0.60), rad=-0.30)
    ax.text(0.5, 0.40, "repeat until it stops moving", ha="center", fontsize=13, color=INK)
    _box(
        ax,
        0.05,
        0.06,
        0.90,
        0.24,
        "Accurate — but it needs a pitch guess to start, sees the whole clip at once,\n"
        "and is far too slow to run inside a network.",
        fc="#fdf0ef",
        ec=ACC,
        fs=13,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_vk_loop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dia_kla_pipeline() -> None:
    """A Kalman filter drawn as a network layer."""
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    _blank(ax)
    y, h = 0.45, 0.30
    _box(ax, 0.02, y, 0.16, h, "token\nstream", fc="#e8eef5")
    _box(ax, 0.24, y, 0.20, h, "learned evidence\ndecay $\\bar{a}$,  $\\varphi$,  $\\kappa$")
    _box(ax, 0.50, y, 0.20, h, "belief update\nprecision $\\lambda$,  info $\\eta$", fc="#dfeaf6")
    _box(ax, 0.76, y, 0.22, h, "readout\n$\\mu=\\eta/\\lambda$  →  output")
    for a, b in ((0.18, 0.24), (0.44, 0.50), (0.70, 0.76)):
        _arrow(ax, (a, y + h / 2), (b, y + h / 2))
    _arrow(ax, (0.60, y), (0.60, 0.20), color=BLUE)
    ax.text(0.615, 0.17, "carries its own confidence", fontsize=13, color=BLUE)
    ax.text(
        0.5,
        0.90,
        "one filter per state slot, run in parallel — a drop-in for an attention layer",
        ha="center",
        fontsize=14,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_kla_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dia_phasor() -> None:
    """Real decay vs complex decay-and-rotate memory."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    n = 26
    for ax, (gamma, omega, title) in zip(
        axes,
        (
            (0.09, 0.0, "plain KLA: memory only fades"),
            (0.09, 0.42, "CKLA: memory fades and turns"),
        ),
        strict=True,
    ):
        z = np.exp((-gamma + 1j * omega) * np.arange(n))
        zr, zi = np.real(z), np.imag(z)
        ax.plot(zr, zi, color=BLUE, lw=2.0, alpha=0.6)
        ax.scatter(zr, zi, s=26, color=BLUE, zorder=3)
        for k in (0, 6, 14):
            ax.annotate(
                "",
                xy=(float(zr[k]), float(zi[k])),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ACC if k == 0 else INK, lw=2.2),
            )
        circ = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
        ax.plot(np.real(circ), np.imag(circ), color="#ccd4dd", lw=1.4, zorder=0)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.35, 1.25)
        ax.set_aspect("equal")
        ax.axhline(0, color="#aab", lw=1)
        ax.axvline(0, color="#aab", lw=1)
        ax.set_title(title.replace("*", ""), fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1].annotate(
        "one step = shrink by $e^{-\\gamma\\Delta t}$,\nturn by $\\omega\\Delta t$",
        xy=(np.exp(-0.09 * 3) * np.cos(0.42 * 3), np.exp(-0.09 * 3) * np.sin(0.42 * 3)),
        xytext=(-1.15, -1.1),
        fontsize=13,
        color=INK,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6),
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_phasor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dia_phase_error() -> None:
    """A slightly wrong turning rate de-coheres; confidence keeps climbing."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    t = np.linspace(0, 8, 400)
    true_w, assumed_w = 2.0, 1.55
    ax = axes[0]
    ax.plot(t, true_w * t, color=BLUE, lw=3, label="true phase")
    ax.plot(t, assumed_w * t, color=ACC, lw=3, ls="--", label="filter's assumed phase")
    ax.fill_between(t, assumed_w * t, true_w * t, color=ACC, alpha=0.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("phase (rad)")
    ax.set_title("a slightly wrong turning rate", fontsize=14)
    ax.legend(loc="upper left", fontsize=12)

    ax = axes[1]
    err = (true_w - assumed_w) * t
    ax.plot(t, err, color=ACC, lw=3, label="phase error")
    ax.axhline(np.pi, color=INK, lw=2, ls=":")
    ax.text(0.2, np.pi + 0.12, "half a cycle — now tracking noise", fontsize=12, color=INK)
    ax.set_ylim(0, 4.4)
    ax2 = ax.twinx()
    ax2.plot(t, 1 - np.exp(-0.9 * t), color=BLUE, lw=3, label="filter's confidence")
    ax2.set_ylabel("confidence $\\lambda$", color=BLUE)
    ax2.set_ylim(0, 1.35)
    ax2.grid(False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("phase error (rad)", color=ACC)
    ax.set_title("the error grows, the confidence grows too", fontsize=14)
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_phase_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dia_kf_vs_vk() -> None:
    """Left: a Kalman filter is a recursive chain (predict-update per sample).
    Right: Vold-Kalman is one global smooth fit over the whole record.
    Drawing, not a measurement."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
    ax = axes[0]
    _blank(ax)
    for i in range(5):
        x = 0.03 + i * 0.195
        _box(ax, x, 0.42, 0.14, 0.26, f"$x_{i}$", fc="#e8eef5", fs=13)
        if i:
            _arrow(ax, (x - 0.055, 0.55), (x - 0.002, 0.55), lw=1.8)
        ax.annotate(
            "",
            xy=(x + 0.07, 0.40),
            xytext=(x + 0.07, 0.18),
            arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8),
        )
    ax.text(0.5, 0.08, "one measurement per step", ha="center", fontsize=12, color=ACC)
    ax.set_title("1993: recursive chain, step by step", fontsize=14)

    ax = axes[1]
    t = np.linspace(0, 8, 400)
    rng = np.random.default_rng(4)
    env = 1.0 + 0.35 * np.sin(2 * np.pi * 0.16 * t)
    obs = env + rng.normal(0, 0.18, t.size)
    ax.plot(t, obs, color="#9fb0c0", lw=1.0, label="whole recording")
    ax.plot(t, env, color=BLUE, lw=3.2, label="one smooth envelope, fitted at once")
    ax.set_title("today: one global batch fit", fontsize=14)
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_kf_vs_vk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dia_block_swap() -> None:
    """Transformer block vs KLA block: identical scaffolding, mixer swapped."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, (title, mixer, mix_fc) in zip(
        axes,
        (
            ("transformer block", "self-attention\n$O(T^2)$ pairwise", "#e8eef5"),
            ("KLA block", "bank of Kalman filters\nassociative scan", "#fdf0ef"),
        ),
        strict=True,
    ):
        _blank(ax)
        _box(ax, 0.18, 0.06, 0.64, 0.12, "input tokens", fc="#eef3f8", fs=12)
        _box(ax, 0.18, 0.26, 0.64, 0.12, "LayerNorm", fc="#eef3f8", fs=12)
        _box(ax, 0.14, 0.44, 0.72, 0.18, mixer, fc=mix_fc, ec=ACC if mix_fc != "#e8eef5" else INK)
        _box(ax, 0.18, 0.70, 0.64, 0.12, "LayerNorm + MLP", fc="#eef3f8", fs=12)
        _box(ax, 0.18, 0.88, 0.64, 0.10, "output tokens", fc="#eef3f8", fs=12)
        for a, b in ((0.18, 0.26), (0.38, 0.44), (0.62, 0.70), (0.82, 0.88)):
            _arrow(ax, (0.5, a), (0.5, b), lw=1.6)
        _arrow(ax, (0.10, 0.12), (0.10, 0.76), rad=0.0, lw=1.6, color=BLUE)
        ax.text(0.02, 0.44, "residual", rotation=90, fontsize=11, color=BLUE)
        ax.set_title(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_block_swap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dia_ou() -> None:
    """Left: a real Ornstein-Uhlenbeck sample path, fading toward zero.
    Right: a complex OU sample path, a phasor that fades AND turns -- the
    model behind a tone with linewidth gamma. Drawing, not a measurement."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    rng = np.random.default_rng(11)

    # --- left: real OU, clearly nonzero start, low noise, visible fade ---
    ax = axes[0]
    n = 260
    dt = 0.03
    gamma = 0.55
    x0 = 1.0
    x = np.zeros(n)
    x[0] = x0
    for k in range(1, n):
        x[k] = np.exp(-gamma * dt) * x[k - 1] + rng.normal(0, 0.03)
    t = np.arange(n) * dt
    env = x0 * np.exp(-gamma * t)
    ax.plot(t, env, color="#aab", lw=1.6, ls="--")
    ax.plot(t, -env, color="#aab", lw=1.6, ls="--")
    ax.plot(t, x, color=BLUE, lw=2.0)
    ax.axhline(0, color="#ccd4dd", lw=1)
    ax.annotate(
        r"$e^{-\gamma\Delta t}$",
        xy=(3.2, env[int(3.2 / dt)]),
        xytext=(5.4, 0.75),
        fontsize=14,
        color=INK,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6),
    )
    ax.set_title("real OU: fades", fontsize=15, pad=10)
    ax.set_xlabel("time (s)")
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.grid(False)

    # --- right: complex OU, omega >> gamma, low noise, clean spiral ---
    ax = axes[1]
    n2 = 700
    dt2 = 0.01
    gamma2, omega2 = 0.15, 2 * np.pi * 0.6
    z = np.zeros(n2, dtype=complex)
    z[0] = 1.0 + 0.0j
    for k in range(1, n2):
        z[k] = np.exp((-gamma2 + 1j * omega2) * dt2) * z[k - 1] + (
            rng.normal(0, 0.003) + 1j * rng.normal(0, 0.003)
        )
    zr, zi = np.real(z), np.imag(z)
    ax.plot(zr, zi, color=BLUE, lw=1.8)
    ax.scatter([zr[0]], [zi[0]], s=46, color=ACC, zorder=3)
    ax.scatter([zr[-1]], [zi[-1]], s=46, color=INK, zorder=3)
    ax.annotate(
        "start",
        xy=(zr[0], zi[0]),
        xytext=(zr[0] + 0.12, zi[0] + 0.12),
        fontsize=12,
        color=ACC,
    )
    ax.annotate(
        "end",
        xy=(zr[-1], zi[-1]),
        xytext=(zr[-1] + 0.12, zi[-1] - 0.05),
        fontsize=12,
        color=INK,
    )
    ax.axhline(0, color="#ccd4dd", lw=1)
    ax.axvline(0, color="#ccd4dd", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("complex OU: fades and turns", fontsize=15, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    fig.suptitle("same recursion, real vs. complex noise", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(ASSETS / "dia_ou.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ckla_diagrams() -> None:
    _dia_kf_vs_vk()
    _dia_block_swap()
    _dia_comb()
    _dia_vk_loop()
    _dia_kla_pipeline()
    _dia_phasor()
    _dia_phase_error()
    _dia_ou()


def main() -> None:
    import os

    ASSETS.mkdir(exist_ok=True)
    # Optional: `python prepare.py <name> [...]` regenerates only the named
    # figures (handy during review rounds — the model-running ones cost
    # minutes on CPU).
    selected = sys.argv[1:]
    steps: dict[str, object] = {
        "seen_unseen": fig_seen_unseen,
        "control": fig_control,
        "ladder": fig_ladder,
        "mpsenet": fig_mpsenet,
        "persnr_a": lambda: fig_f1_persnr("a"),
        "persnr_b": lambda: fig_f1_persnr("b"),
        "ckla_copy": copy_ckla_figures,
        "base_augs": fig_base_augs,
        "expanded_augs": fig_expanded_augs,
        "diagrams": fig_ckla_diagrams,
    }
    model_steps: dict[str, object] = {
        "spectrograms": fig_f1_spectrograms,  # ~4 min on CPU: 4 ckpts x 2 x 8 s
        "freqshift": fig_freqshift_predictions,
        "freqshift_both": fig_freqshift_both,
        "overlay": fig_ckla_prediction_overlay,
        "regime_table": table_regime_mae,
        "ckla_freqshift": fig_ckla_freqshift,
    }
    allsteps = {**steps, **model_steps}
    if selected:
        for name in selected:
            allsteps[name]()  # type: ignore[operator]
        print("regenerated", ", ".join(selected))
        return
    for fn in steps.values():
        fn()  # type: ignore[operator]
    if os.environ.get("SKIP_MODEL_FIGS") != "1":
        for fn in model_steps.values():
            fn()  # type: ignore[operator]
    print("slide figures written to", ASSETS)


if __name__ == "__main__":
    main()
