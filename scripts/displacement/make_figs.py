#!/usr/bin/env python3
"""Figures for the per-harmonic displacement measurement (F1 / F2 / F3)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import measure_displacement as M  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"
SPECS = OUT / "specs"
DATA = json.loads((OUT / "displacement.json").read_text())

K_STRIPS = (2, 5, 8, 13, 16, 22, 30)
LOW_K = tuple(DATA["protocol"]["low_k_set"])
HIGH_K = tuple(DATA["protocol"]["high_k_set"])
JITTER = DATA["protocol"]["jitter_floor_rev_s"]

PANELS = [
    ("free-flight_nosource_room1", 1, "DREGON free-flight nosource, w01 (cruise)"),
    ("free-flight_speech-low_room1", 1, "DREGON free-flight speech-low, w01 (cruise)"),
    ("FLY124", 3, "FLY124 w03 (cruise)"),
]


DISP_SEG_S = 2.0  # uniform display segment for every k
DISP_HOP_S = 0.25


def display_strip(z_k, band_hz, k, clean_ft, ft):
    """(t, rev_axis, snr_db (F,T), profile_db (F,)) for one harmonic."""
    n_seg = int(DISP_SEG_S * M.FS_ENV)
    hop = int(DISP_HOP_S * M.FS_ENV)
    n_env = z_k.shape[-1]
    starts = list(range(0, n_env - n_seg + 1, hop))
    win = np.hanning(n_seg)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_seg, d=1.0 / M.FS_ENV))
    keep = np.abs(freqs) <= band_hz
    rev = freqs[keep] / k
    spec = np.empty((len(starts), int(keep.sum())))
    for a, st in enumerate(starts):
        seg = z_k[:, st : st + n_seg] * win
        spec[a] = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=-1), axes=-1)) ** 2).mean(0)[keep]
    t = (np.array(starts) + n_seg / 2.0) / M.FS_ENV
    snr_db = 10.0 * np.log10(spec / np.median(spec, axis=1, keepdims=True) + 1e-300)
    ok = np.interp(t, ft, clean_ft.astype(float)) > 0.999
    prof_src = spec[ok] if ok.sum() >= 3 else spec
    prof = prof_src.mean(axis=0)
    prof_db = 10.0 * np.log10(prof / np.median(prof) + 1e-300)
    # ridge = peak of the profile inside the search window, after smoothing to
    # ~0.05 rev/s so a single noisy bin cannot claim the line
    step = float(rev[1] - rev[0])
    n_sm = max(3, int(round(0.05 / step)) | 1)
    kern = np.hanning(n_sm)
    prof_sm = np.convolve(prof_db, kern / kern.sum(), mode="same")
    sw = np.abs(rev) <= M.search_hz(k) / k
    j = int(np.argmax(prof_sm[sw]))
    peak = float(rev[sw][j])
    strength = float(prof_sm[sw][j])
    return t, rev, snr_db.T, prof_db, prof_sm, peak, strength, bool(ok.sum() >= 3)


def fig1() -> Path:
    """Demodulated envelope spectrogram strips, rev/s offset from k*telemetry."""
    n_rows = len(K_STRIPS)
    fig = plt.figure(figsize=(5.3 * len(PANELS), 1.7 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        3 * len(PANELS),
        width_ratios=[3.4, 1.0, 0.45] * len(PANELS),
        hspace=0.22,
        wspace=0.06,
    )
    for c, (rid, widx, title) in enumerate(PANELS):
        key = f"{rid}__w{widx:02d}"
        win = DATA["windows"][key]
        with np.load(M.PREP / f"{key}.npz") as z:
            audio = np.asarray(z["audio"], np.float64)
            ft = np.asarray(z["ft"], np.float64)
            r_ft = np.asarray(z["r_meas"], np.float64)
        # the rotor whose k = 8 harmonic has the strongest surviving ridge
        rot = max(range(4), key=lambda r: win["rotors"][str(r)]["8"][1])
        ks = list(range(1, M.K_MAX + 1))
        z_on, band_hz_k = M.envelope_bank(audio, r_ft, ft, rot, ks)
        clean = ~M.collision_mask(r_ft, rot, ks)
        for r, k in enumerate(K_STRIPS):
            ax = fig.add_subplot(gs[r, 3 * c])
            axp = fig.add_subplot(gs[r, 3 * c + 1], sharey=ax)
            t, rev, snr_db, prof_db, prof_sm, peak, strength, gated = display_strip(
                z_on[:, k - 1], float(band_hz_k[k - 1]), k, clean[k - 1], ft
            )
            ax.pcolormesh(
                t,
                rev,
                snr_db,
                cmap="magma",
                vmin=1.0,
                vmax=13.0,
                shading="nearest",
                rasterized=True,
            )
            ax.axhline(0.0, color="#00e5ff", lw=1.5, ls="--")
            sh = M.search_hz(k) / k
            for a in (ax, axp):
                a.axhline(peak, color="#39ff14", lw=1.3, alpha=0.95)
                a.set_ylim(-1.05, 1.05)
            axp.plot(prof_db, rev, color="0.65", lw=0.7)
            axp.plot(prof_sm, rev, color="0.1", lw=1.5)
            axp.axhline(0.0, color="#0090a8", lw=1.4, ls="--")
            axp.axhspan(-sh, sh, color="#ffd27f", alpha=0.3)
            axp.set_xlim(-1.0, max(6.0, float(prof_db.max()) * 1.15))
            axp.tick_params(labelleft=False, labelsize=6.5)
            axp.grid(alpha=0.3)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.tick_params(labelsize=7)
            ax.text(
                0.012,
                0.93,
                f"k = {k}   $\\delta$ = {peak:+.2f} rev/s   ({strength:.1f} dB ridge)"
                + ("" if gated else "  ungated"),
                transform=ax.transAxes,
                color="w" if strength >= 2.0 else "#ff8a8a",
                fontsize=8.5,
                va="top",
                fontweight="bold",
            )
            if c == 0:
                ax.set_ylabel("rev/s", fontsize=8)
            if r == 0:
                ax.set_title(f"{title} — rotor {rot}", fontsize=9.5, loc="left", pad=16)
                axp.set_title("time avg\n(dB over floor)", fontsize=6.5, pad=3)
            if r == n_rows - 1:
                ax.set_xlabel("time in window (s)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
                axp.tick_params(labelbottom=False)
    fig.suptitle(
        "F1 — demodulated envelope of harmonic $k$, frequency axis rescaled to shaft-rate "
        "offset $(f - k g)/k$ rev/s.  Color = dB over the in-band noise floor.\n"
        "Cyan dashed = telemetry ($\\delta$ = 0);  green = measured ridge offset;  shaded = "
        "peak-search window.\n"
        "On DREGON the ridge sits BELOW telemetry for low $k$ and returns to it by "
        "$k \\geq 16$;  FLY124 stays on the line throughout.\n"
        'Profiles average the uncollided frames ("ungated" = none survived); a red label '
        "marks a ridge under 2 dB, where the single-window offset is not meaningful.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.005, 0, 0.995, 0.935))
    p = FIGS / "F1_demod_strips.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig2() -> Path:
    """delta_k vs k, three classes, with a demod-quality sub-panel."""
    fig, (ax, axq) = plt.subplots(2, 1, figsize=(11.0, 7.4), sharex=True, height_ratios=[3.0, 1.0])
    # steady (w01/w02) and ramp (w00) are the two disjoint halves of DREGON cruise
    series = [
        ("dregon_steady", "DREGON steady control, w01/w02 (6 windows)", "#ff9896", "s", 2),
        ("dregon_ramp", "DREGON ramp control, w00 (3 windows)", "#9467bd", "D", 2),
        ("fly124_cruise", "FLY124 cruise (4 windows)", "#1f77b4", "^", 3),
        ("dregon_cruise", "DREGON cruise, all 9 windows", "#d62728", "o", 4),
    ]
    ax.axhspan(-0.5, -0.3, color="#ffd27f", alpha=0.35, zorder=0)
    ax.text(
        40.5, -0.4, "0.3-0.5 rev/s\n(the reported band)", fontsize=8, va="center", color="#8a6d1a"
    )
    ax.axvspan(13.5, 15.5, color="0.85", alpha=0.7, zorder=0)
    ax.text(14.5, 0.66, "k = 13 | 16\nboundary", fontsize=8, ha="center", color="0.35")
    ax.axhline(0.0, color="k", lw=1.1)
    lo, hi = -0.72, 0.72
    for klass, label, color, marker, zo in series:
        prof = DATA["pooled"][klass]["per_k"]
        ks = np.array(sorted(int(k) for k in prof))
        m = np.array([prof[str(k)]["mean_offset_rev_s"] for k in ks])
        e = np.array([prof[str(k)]["sem_rev_s"] for k in ks])
        ax.errorbar(
            ks,
            np.clip(m, lo, hi),
            yerr=e,
            color=color,
            marker=marker,
            ms=5.0 if zo == 4 else 3.6,
            lw=2.0 if zo == 4 else 1.1,
            alpha=1.0 if zo >= 3 else 0.75,
            capsize=2.5,
            label=label,
            zorder=zo,
        )
        off = (m < lo) | (m > hi)
        if off.any():
            ax.plot(
                ks[off],
                np.clip(m[off], lo, hi),
                marker="v",
                ms=9,
                ls="none",
                color=color,
                mec="k",
                mew=0.6,
                zorder=6,
            )
        axq.plot(
            ks,
            10 * np.log10([prof[str(k)]["median_snr"] for k in ks]),
            color=color,
            marker=marker,
            ms=3.2,
            lw=1.3,
            zorder=zo,
            alpha=1.0 if zo >= 3 else 0.75,
        )
    for klass, color in (("dregon_cruise", "#d62728"), ("fly124_cruise", "#1f77b4")):
        pl = DATA["pooled"][klass]
        ax.hlines(pl["low_k_mean_offset_rev_s"], 2, 13, color=color, ls=":", lw=2.4)
        ax.hlines(pl["high_k_mean_offset_rev_s"], 16, 40, color=color, ls=":", lw=2.4)
    ax.set_ylabel("acoustic shaft-rate offset $\\delta_k$ from telemetry (rev/s)")
    ax.set_title(
        "F2 — per-harmonic displacement of the rotor comb from telemetry\n"
        "SNR-weighted over windows x rotors, frames with a neighbouring-rotor line in the "
        "search window gated out;\nerror bars = weighted SEM. Dotted = low-$k$ / high-$k$ "
        "band means; triangles = points off scale.\n"
        "The steady and ramp controls are the two disjoint halves of DREGON cruise.",
        fontsize=11,
    )
    ax.set_ylim(lo - 0.03, hi + 0.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    axq.axhline(3.0, color="k", ls="--", lw=1.0)
    axq.text(40.6, 3.0, "3 dB", fontsize=7.5, va="center")
    axq.set_xlabel("harmonic order $k$")
    axq.set_ylabel("median demod\nSNR (dB)", fontsize=9)
    axq.grid(alpha=0.3)
    axq.set_xlim(0.5, 41)
    fig.tight_layout()
    p = FIGS / "F2_delta_vs_k.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig3() -> Path:
    """Three-way error decomposition per window + pooled."""
    dec = DATA["error_decomposition"]
    keys = [k for k in dec["per_window"] if dec["per_window"][k]["klass"] != "fly124_warmup"]
    keys.sort(key=lambda k: (dec["per_window"][k]["klass"], k))
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    x = np.arange(len(keys) + 2, dtype=float)
    x[len(keys) :] += 0.8  # gap before the pooled bars
    fields = [
        ("low_k_comb_mae", f"low-$k$ comb (k={LOW_K[0]}..{LOW_K[-1]})", "#e07b39"),
        ("high_k_comb_mae", f"high-$k$ comb (k={HIGH_K[0]}..{HIGH_K[-1]})", "#2c7fb8"),
        ("flagship_track_mae", "flagship blind track (peeled x3)", "#7a4fa3"),
    ]
    rows = [dec["per_window"][k] for k in keys] + [
        dec["pooled"]["dregon_cruise"],
        dec["pooled"]["fly124_cruise"],
    ]
    w = 0.27
    for i, (f, label, color) in enumerate(fields):
        vals = [r.get(f) or np.nan for r in rows]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v * 1.06,
                    f"{v:.2f}",
                    ha="center",
                    fontsize=6.3,
                    rotation=90,
                )
    ax.axhline(JITTER, color="k", ls="--", lw=1.4)
    ax.text(x[-1] + 0.6, JITTER * 1.06, f"{JITTER} rev/s\njitter floor", fontsize=8, va="bottom")
    labels = [
        k.replace("free-flight_", "").replace("_room1", "").replace("__", " ") for k in keys
    ] + ["POOLED\nDREGON cruise", "POOLED\nFLY124 cruise"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(0.035, 14)
    ax.set_ylabel("MAE against telemetry GT (rev/s, log scale)")
    ax.set_title(
        "F3 — error decomposition: where does the blind tracker's error come from?\n"
        "The high-$k$ acoustic comb sits at/below the label-jitter floor, an order of "
        "magnitude under the flagship track — the residual is ESTIMATOR error, not physics.",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    p = FIGS / "F3_error_decomposition.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3):
        print(f())
