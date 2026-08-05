#!/usr/bin/env python3
"""Figures F1 (rebuilt), F4 (null controls), F5 (prominence map), F6 (wiggle)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

import measure_displacement as M  # noqa: E402
import nullcontrol as NC  # noqa: E402

FIGS = OUT / "figs"
NULL = json.loads((OUT / "nullcontrol.json").read_text())
WINS = NULL["windows"]

I_OFF, I_SNR, I_PROM, I_PP, I_COH, I_SEARCH = 0, 1, 4, 6, 7, 9
K_ALL = list(range(1, M.K_MAX + 1))
LOW_K, HIGH_K = M.LOW_K, M.HIGH_K
PROM_BAR = 6.0  # dB over the in-band floor: the usability threshold

DREGON_CRUISE = [k for k in WINS if k.split("__")[0] in M.DREGON_RECS]
FLY_CRUISE = [k for k in WINS if k.startswith("FLY124") and WINS[k]["regime"] == "cruise"]
CLASSES = {"DREGON cruise": DREGON_CRUISE, "FLY124 cruise": FLY_CRUISE}
REC_TITLE = {
    "free-flight_nosource_room1": "DREGON nosource",
    "free-flight_speech-low_room1": "DREGON speech-low",
    "free-flight_whitenoise-low_room1": "DREGON whitenoise-low",
    "FLY124": "FLY124 (Michael's DJI M100)",
}


def entry(key: str, rot: int, variant: str, k: int) -> list[Any]:
    return WINS[key]["rotors"][str(rot)][variant]["per_k"][str(k)]


def collect(keys: list[str], variant: str, idx: int, k: int, absolute: bool = False) -> np.ndarray:
    v = []
    for key in keys:
        for rot in range(M.N_ROTORS):
            e = entry(key, rot, variant, k)[idx]
            if e is None:
                continue
            v.append(abs(e) if absolute else e)
    return np.asarray(v, dtype=float)


# ───────────────────────────── prominence.json ──────────────────────────────


def write_prominence() -> Path:
    out: dict[str, Any] = {
        "definition": (
            "prominence_db = peak (inside the peak-search window) of the "
            "time-averaged demod power profile of harmonic k, in dB over that "
            "profile's own in-band median, after smoothing to ~0.05 rev/s. "
            "Frames with a neighbouring-rotor line in the search window are "
            "excluded from the average. 'null' repeats it at the off-comb rate "
            "(k + 0.5) * g(t), where no rotor line exists."
        ),
        "usability_threshold_db": PROM_BAR,
        "k_max": M.K_MAX,
        "windows": {},
        "pooled": {},
    }
    for key, w in WINS.items():
        out["windows"][key] = {
            "regime": w["regime"],
            "rotor_mean_rev_s": w["rotor_mean_rev_s"],
            "rotors": {
                str(r): {
                    "on": [entry(key, r, "on", k)[I_PROM] for k in K_ALL],
                    "off": [entry(key, r, "off", k)[I_PROM] for k in K_ALL],
                }
                for r in range(M.N_ROTORS)
            },
        }
    for name, keys in CLASSES.items():
        prof = {}
        for k in K_ALL:
            on = collect(keys, "on", I_PROM, k)
            off = collect(keys, "off", I_PROM, k)
            prof[str(k)] = {
                "median_on_db": round(float(np.median(on)), 3),
                "median_off_db": round(float(np.median(off)), 3),
                "excess_db": round(float(np.median(on) - np.median(off)), 3),
                "frac_units_over_bar": round(float(np.mean(on >= PROM_BAR)), 3),
            }
        out["pooled"][name] = prof
    p = OUT / "prominence.json"
    p.write_text(json.dumps(out, indent=1))
    return p


# ───────────────────────────────── F4 ───────────────────────────────────────


def band_stat(keys: list[str], variant: str, field: str) -> np.ndarray:
    return np.asarray(
        [WINS[key]["rotors"][str(r)][variant][field] for key in keys for r in range(M.N_ROTORS)],
        dtype=float,
    )


def fig4() -> Path:
    fig = plt.figure(figsize=(15.0, 10.4))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.19, height_ratios=[1.15, 1.0, 1.0])

    # ── (a) the headline statistic: combined-series MAE, measured vs nulls ──
    axb = fig.add_subplot(gs[0, :])
    groups, meas, offn, misn, anal = [], [], [], [], []
    for cname, keys in CLASSES.items():
        for bname, field, kset in (
            (f"low $k$ ({LOW_K[0]}-{LOW_K[-1]})", "low_k_series_mae", LOW_K),
            (f"high $k$ ({HIGH_K[0]}-{HIGH_K[-1]})", "high_k_series_mae", HIGH_K),
        ):
            groups.append(f"{cname}\n{bname}")
            meas.append(float(np.mean(band_stat(keys, "on", field))))
            offn.append(float(np.mean(band_stat(keys, "off", field))))
            misn.append(float(np.mean(band_stat(keys, "mis", field))))
            hw = np.array([np.mean(collect(keys, "on", I_SEARCH, k)) for k in kset])
            anal.append(float(np.mean(hw) / 2.0))
    x = np.arange(len(groups))
    w = 0.2
    for i, (v, lab, c) in enumerate(
        (
            (meas, "MEASURED (on-comb, $k\\,g_r$)", "#2c7fb8"),
            (offn, "NULL off-comb $(k{+}0.5)\\,g_r$", "#e07b39"),
            (misn, "NULL mismatched telemetry", "#c94c4c"),
            (anal, "analytic: uniform peak in window", "0.45"),
        )
    ):
        bars = axb.bar(x + (i - 1.5) * w, v, w, color=c, label=lab)
        for b, vv in zip(bars, v):
            axb.text(
                b.get_x() + b.get_width() / 2, vv * 1.04, f"{vv:.3f}", ha="center", fontsize=7.5
            )
    axb.set_xticks(x)
    axb.set_xticklabels(groups, fontsize=9)
    axb.set_yscale("log")
    axb.set_ylabel("MAE of the combined\noffset series (rev/s)")
    axb.grid(axis="y", alpha=0.3)
    axb.legend(fontsize=8.5, ncol=2, loc="upper left", framealpha=0.95)
    axb.set_title(
        "(a) the headline statistic run through the IDENTICAL pipeline on the nulls — "
        "a null as small as the measurement means the number is window-limited, not signal-limited.\n"
        "MAE is two-sided, so a genuinely DISPLACED comb also scores high: for the low-$k$ bars "
        'read the signed mean and the prominence (F5) instead.  "Measurement = null" refutes '
        "only a claim of AGREEMENT, which is exactly what the high-$k$ bars asserted.",
        fontsize=10,
        loc="left",
    )
    top = max(meas + offn + misn + anal)
    axb.set_ylim(top=top * 4.0)
    axb.set_xticklabels(
        [f"{g}\nmeasured / off-comb null = {m0 / o0:.2f}" for g, m0, o0 in zip(groups, meas, offn)],
        fontsize=9,
    )
    for xi, (m0, o0) in enumerate(zip(meas, offn)):
        dead = 0.8 < m0 / o0 < 1.25
        axb.text(
            xi,
            top * 1.55,
            "INDISTINGUISHABLE\nFROM THE NULL" if dead else "clear of the null",
            ha="center",
            va="center",
            fontsize=10,
            color="#8b0000" if dead else "#0a5c0a",
            fontweight="bold",
        )

    # ── (b) per-k |offset| vs the nulls ────────────────────────────────────
    for c, (cname, keys) in enumerate(CLASSES.items()):
        ax = fig.add_subplot(gs[1, c])
        ks = np.array(K_ALL[1:])  # k >= 2
        for variant, lab, col in (
            ("on", "MEASURED on-comb", "#2c7fb8"),
            ("off", "NULL off-comb", "#e07b39"),
            ("mis", "NULL mismatched", "#c94c4c"),
        ):
            m = np.array([np.mean(collect(keys, variant, I_OFF, k, absolute=True)) for k in ks])
            ax.plot(ks, m, marker="o", ms=3.2, lw=1.5, color=col, label=lab)
        hw = np.array([np.mean(collect(keys, "on", I_SEARCH, k)) for k in ks])
        ax.plot(ks, hw / 2.0, ls="--", lw=1.8, color="0.35", label="analytic (uniform peak)")
        ax.plot(ks, hw, ls=":", lw=1.4, color="0.6", label="search half-width")
        ax.set_title(f"(b{c + 1}) {cname}: per-$k$ mean $|\\delta_k|$ vs the nulls", fontsize=10)
        ax.set_xlabel("harmonic order $k$")
        if c == 0:
            ax.set_ylabel("mean $|\\delta_k|$ over windows\n$\\times$ rotors (rev/s)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.axvspan(13.5, 15.5, color="0.88", zorder=0)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    # ── (c) window-independent estimator + demod quality ───────────────────
    for c, (cname, keys) in enumerate(CLASSES.items()):
        ax = fig.add_subplot(gs[2, c])
        ks = np.array(K_ALL[1:])
        for variant, lab, col in (
            ("on", "MEASURED on-comb", "#2c7fb8"),
            ("off", "NULL off-comb", "#e07b39"),
        ):
            m = np.array([np.mean(collect(keys, variant, I_PP, k, absolute=True)) for k in ks])
            ax.plot(ks, m, marker="o", ms=3.2, lw=1.5, color=col, label=f"{lab}: $|$pulse-pair$|$")
        ax.set_ylabel("$|$phase-increment offset$|$\n(rev/s)" if c == 0 else "")
        ax.set_yscale("log")
        ax.set_xlabel("harmonic order $k$")
        ax.grid(alpha=0.3)
        axq = ax.twinx()
        for variant, col in (("on", "#2c7fb8"), ("off", "#e07b39")):
            snr = np.array([10 * np.log10(np.median(collect(keys, variant, I_SNR, k))) for k in ks])
            axq.plot(ks, snr, lw=1.0, ls="--", color=col, alpha=0.8)
        axq.set_ylabel("median demod SNR (dB, dashed)", fontsize=8)
        axq.set_ylim(-1, 22)
        ax.set_title(
            f"(c{c + 1}) {cname}: window-INDEPENDENT coherent estimator (solid)\n"
            "and demod SNR (dashed).  Pulse-pair is band-limited, not search-limited.",
            fontsize=9.5,
        )
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "F4 — null controls for the displaced-comb measurement\n"
        "Every null runs the SAME pipeline (same demod band, same "
        "$\\pm\\min(1.5k, 8)$ Hz search, same collision gate, same SNR-weighted "
        "combination over $k$) with only the carrier changed:\n"
        "off-comb rides $(k{+}0.5)\\,g_r(t)$, where no rotor line exists; mismatched rides "
        "$k\\,g_r(t)$ taken from a DIFFERENT window.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.004, 0.0, 0.996, 0.93))
    p = FIGS / "F4_nullcontrol.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ───────────────────────────────── F5 ───────────────────────────────────────


def fig5() -> Path:
    recs = list(REC_TITLE)
    fig, axes = plt.subplots(2, len(recs), figsize=(4.6 * len(recs), 8.2))
    ks = np.array(K_ALL[1:])
    rot_cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for c, rec in enumerate(recs):
        keys = [k for k in WINS if k.split("__")[0] == rec and WINS[k]["regime"] == "cruise"]
        ax = axes[0, c]
        null_all = np.array(
            [[entry(key, r, "off", k)[I_PROM] for k in ks] for key in keys for r in range(4)]
        )
        ax.fill_between(
            ks,
            np.percentile(null_all, 10, axis=0),
            np.percentile(null_all, 90, axis=0),
            color="0.65",
            alpha=0.45,
            label="off-comb null (10-90%)",
            zorder=1,
        )
        ax.plot(ks, np.median(null_all, axis=0), color="0.3", lw=1.4, ls="--", zorder=2)
        for r in range(4):
            v = np.array([[entry(key, r, "on", k)[I_PROM] for k in ks] for key in keys])
            rate = float(np.mean([WINS[key]["rotor_mean_rev_s"][r] for key in keys]))
            ax.plot(
                ks,
                np.median(v, axis=0),
                color=rot_cols[r],
                lw=1.7,
                marker="o",
                ms=3.0,
                label=f"rotor {r} ({rate:.1f} rev/s)",
                zorder=4,
            )
        ax.axhline(PROM_BAR, color="k", lw=1.5, ls=":")
        ax.text(40.4, PROM_BAR + 0.4, f"{PROM_BAR:.0f} dB bar", fontsize=8, va="bottom", ha="right")
        ax.axvspan(13.5, 15.5, color="0.9", zorder=0)
        ax.set_title(f"{REC_TITLE[rec]}\n({len(keys)} cruise windows)", fontsize=10)
        ax.set_xlabel("harmonic order $k$")
        if c == 0:
            ax.set_ylabel("ridge prominence over the\nin-band floor (dB)")
        ax.set_ylim(-1.0, 36.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.0, loc="upper right")

        axh = axes[1, c]
        rows, labels = [], []
        for key in keys:
            for r in range(4):
                rows.append([entry(key, r, "on", k)[I_PROM] for k in ks])
                labels.append(f"{key.split('__')[1]} r{r}")
        im = axh.pcolormesh(
            ks,
            np.arange(len(rows)),
            np.asarray(rows),
            cmap="viridis",
            vmin=0.0,
            vmax=14.0,
            shading="nearest",
        )
        axh.contour(
            ks,
            np.arange(len(rows)),
            np.asarray(rows),
            levels=[PROM_BAR],
            colors="w",
            linewidths=1.1,
        )
        axh.set_yticks(np.arange(len(rows)))
        axh.set_yticklabels(labels, fontsize=6.0)
        axh.set_xlabel("harmonic order $k$")
        if c == len(recs) - 1:
            fig.colorbar(im, ax=axh, label="prominence (dB)", pad=0.02)
    fig.suptitle(
        "F5 — harmonic prominence map: how far above its own in-band noise floor each harmonic's "
        "demod ridge actually sits.\nTop: median over that recording's cruise windows, one line "
        "per rotor; grey = the off-comb null (what a rate with NO line scores).  "
        "Bottom: every (window, rotor) unit, white contour = the 6 dB bar.\n"
        'A harmonic below the grey band carries no measurable line, and its "offset" is a '
        "peak-pick of noise.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.004, 0.0, 0.996, 0.9))
    p = FIGS / "F5_prominence.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ───────────────────────────────── F1 ───────────────────────────────────────

DISP_HOP_S = 0.25


def strip(z_k: np.ndarray, band_hz: float, k: int, clean_k: np.ndarray, ft: np.ndarray):
    """(t, rev, snr_db (F,T), prof_sm_db, peak, prominence, n_kept)."""
    n_seg = M.seg_len_env(k)
    hop = int(DISP_HOP_S * M.FS_ENV)
    n_env = z_k.shape[-1]
    n_seg = min(n_seg, n_env)
    starts = np.arange(0, n_env - n_seg + 1, hop)
    win = np.hanning(n_seg)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_seg, d=1.0 / M.FS_ENV))
    keep = np.abs(freqs) <= band_hz
    rev = freqs[keep] / k
    spec = np.empty((len(starts), int(keep.sum())))
    for a, s in enumerate(starts):
        seg = z_k[:, s : s + n_seg] * win
        spec[a] = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=-1), axes=-1)) ** 2).mean(0)[keep]
    t = (starts + n_seg / 2.0) / M.FS_ENV
    snr_db = 10.0 * np.log10(spec / np.median(spec, axis=1, keepdims=True) + 1e-300)
    ok = np.interp(t, ft, clean_k.astype(float)) > 0.999
    prof = (spec[ok] if ok.sum() >= 3 else spec).mean(axis=0)
    prof_db = 10.0 * np.log10(prof / np.median(prof) + 1e-300)
    step = float(rev[1] - rev[0])
    n_sm = max(3, int(round(0.05 / step)) | 1)
    kern = np.hanning(n_sm)
    prof_sm = np.convolve(prof_db, kern / kern.sum(), mode="same")
    sw = np.abs(rev) <= NC.eff_search_revs(k, band_hz)
    j = int(np.argmax(prof_sm[sw]))
    return t, rev, snr_db.T, prof_sm, float(rev[sw][j]), float(prof_sm[sw][j]), int(ok.sum())


PANELS = [
    ("free-flight_nosource_room1__w01", "DREGON nosource w01 (cruise)"),
    ("free-flight_speech-low_room1__w01", "DREGON speech-low w01 (cruise)"),
    ("FLY124__w03", "FLY124 w03 (cruise)"),
]
N_STRIPS = 7


def select_ks(key: str, rot: int) -> tuple[list[int], int]:
    """Top-N most prominent harmonics, forcing >= 2 high-$k$ ones when any of
    them clears the bar.  Returns ``(ks, n_highk_over_bar)``."""
    prom = {k: entry(key, rot, "on", k)[I_PROM] or -99.0 for k in K_ALL if k >= 2}
    order = sorted(prom, key=lambda k: -prom[k])
    high_over = [k for k in order if k >= HIGH_K[0] and prom[k] >= PROM_BAR]
    sel = order[:N_STRIPS]
    forced = [k for k in order if k >= HIGH_K[0]][:2]
    for f in forced:  # keep the low-k/high-k contrast visible
        if f not in sel:
            sel = sel[:-1] + [f]
    return sorted(sel), len(high_over)


def fig1() -> tuple[Path, dict[str, Any]]:
    info: dict[str, Any] = {}
    fig = plt.figure(figsize=(5.6 * len(PANELS), 1.72 * N_STRIPS))
    gs = fig.add_gridspec(
        N_STRIPS,
        3 * len(PANELS),
        width_ratios=[3.3, 1.0, 0.42] * len(PANELS),
        hspace=0.24,
        wspace=0.06,
    )
    for c, (key, title) in enumerate(PANELS):
        w = NC.load_window(key)
        rot = max(
            range(4),
            key=lambda r: float(
                np.median([entry(key, r, "on", k)[I_PROM] or -99.0 for k in range(2, M.K_MAX + 1)])
            ),
        )
        ks_sel, n_high = select_ks(key, rot)
        info[key] = {
            "rotor": rot,
            "ks": ks_sel,
            "n_highk_over_bar": n_high,
            "prom": {k: entry(key, rot, "on", k)[I_PROM] for k in ks_sel},
            "null_prom": {k: entry(key, rot, "off", k)[I_PROM] for k in ks_sel},
            "offset": {k: entry(key, rot, "on", k)[I_OFF] for k in ks_sel},
        }
        z_on, band_hz_k = NC.bank(w["audio"], w["r_ft"][rot], w["ft"], K_ALL, half=False)
        clean = ~M.collision_mask(w["r_ft"], rot, K_ALL)
        for r, k in enumerate(ks_sel):
            ax = fig.add_subplot(gs[r, 3 * c])
            axp = fig.add_subplot(gs[r, 3 * c + 1], sharey=ax)
            t, rev, snr_db, prof_sm, peak, prom, n_ok = strip(
                z_on[:, k - 1], float(band_hz_k[k - 1]), k, clean[k - 1], w["ft"]
            )
            # per-strip robust limits: 60th..99.5th percentile of THIS strip
            vmin, vmax = np.percentile(snr_db, [70.0, 99.5])
            vmax = max(vmax, vmin + 2.0)
            ax.pcolormesh(
                t,
                rev,
                snr_db,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                rasterized=True,
            )
            ax.axhline(0.0, color="#00e5ff", lw=1.6, ls="--")
            ax.axhline(peak, color="#39ff14", lw=1.2, alpha=0.9)
            # quote the SAME prominence statistic F5 and the pooled claim use
            prom = entry(key, rot, "on", k)[I_PROM] or -99.0
            null_p = entry(key, rot, "off", k)[I_PROM]
            over = prom >= PROM_BAR
            ax.text(
                0.011,
                0.945,
                f"k={k}  |  {prom:.1f} dB (null {null_p:.1f})  |  "
                f"$\\delta$={peak:+.2f} rev/s" + ("" if over else "  *"),
                transform=ax.transAxes,
                color="w" if over else "#ff9d9d",
                fontsize=8.0,
                va="top",
                fontweight="bold",
            )
            axp.plot(prof_sm, rev, color="0.1", lw=1.4)
            axp.axhline(0.0, color="#0090a8", lw=1.4, ls="--")
            axp.axhline(peak, color="#22aa22", lw=1.1)
            axp.axvline(PROM_BAR, color="#c05000", lw=1.0, ls=":")
            sh = NC.eff_search_revs(k, float(band_hz_k[k - 1]))
            axp.axhspan(-sh, sh, color="#ffd27f", alpha=0.28)
            axp.set_xlim(-1.5, max(8.0, prom * 1.2))
            axp.tick_params(labelleft=False, labelsize=6.5)
            axp.grid(alpha=0.3)
            for a in (ax, axp):
                a.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.tick_params(labelsize=7.5)
            if c == 0:
                ax.set_ylabel("rev/s", fontsize=8)
            if r == 0:
                ax.set_title(
                    f"{title}\nrotor {rot}, {w['r_ft'][rot].mean():.1f} rev/s",
                    fontsize=10,
                    loc="left",
                    pad=8,
                )
                axp.set_title("time avg\n(dB over floor)", fontsize=6.5, pad=3)
            if r == N_STRIPS - 1:
                ax.set_xlabel("time in window (s)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
                axp.tick_params(labelbottom=False)

    # the honest high-k statement is a POOLED one, not a two-panel anecdote
    def over_bar(keys: list[str], ks: tuple[int, ...], variant: str) -> tuple[int, int]:
        v = [
            entry(key, r, variant, k)[I_PROM] for key in keys for r in range(M.N_ROTORS) for k in ks
        ]
        v = [x for x in v if x is not None]
        return sum(1 for x in v if x >= PROM_BAR), len(v)

    d_on, d_tot = over_bar(DREGON_CRUISE, HIGH_K, "on")
    d_null, _ = over_bar(DREGON_CRUISE, HIGH_K, "off")
    f_on, f_tot = over_bar(FLY_CRUISE, HIGH_K, "on")
    note = (
        f"Pooled over ALL cruise windows $\\times$ rotors: on DREGON only {d_on} of {d_tot} "
        f"harmonics with $k \\geq 16$ clear the 6 dB bar — FEWER than the off-comb null's "
        f"{d_null}.  There is no usable DREGON high-$k$ line;\nthe high-$k$ strips shown are "
        f"the best available and are still noise.  On FLY124 {f_on} of {f_tot} clear it, which "
        "is why its $k$ = 16 and 18 strips still show a ridge."
    )
    fig.suptitle(
        "F1 — demodulated envelope of harmonic $k$, frequency axis rescaled to the shaft-rate "
        "offset $(f - kg)/k$ rev/s.  Cyan dashed = telemetry ($\\delta = 0$), green = measured "
        "ridge, shaded = peak-search window.\n"
        "Each panel shows the rotor with the strongest harmonic set and its 7 most prominent "
        "harmonics, with the two most prominent $k \\geq 16$ forced in.  Colour limits are "
        "per strip (70th-99.5th percentile of its own dB-over-floor values).\n"
        "Labels: the gated measurement prominence (the F5 statistic), the same statistic "
        'measured off-comb at $(k{+}0.5)g$ where no line exists ("null"), and the ridge of '
        "the UNgated displayed strip.  * = under the 6 dB bar.\n" + note,
        fontsize=10.5,
        y=0.995,
        va="top",
    )
    fig.tight_layout(rect=(0.004, 0.0, 0.996, 0.885))
    p = FIGS / "F1_demod_strips.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p, info


# ───────────────────────────────── F6 ───────────────────────────────────────

WIG_PANELS = [
    ("free-flight_nosource_room1__w01", "DREGON nosource w01 (cruise)", "#d62728"),
    ("FLY124__w03", "FLY124 w03 (cruise)", "#1f77b4"),
]
WIG_ALL = [
    "free-flight_nosource_room1__w01",
    "free-flight_speech-low_room1__w01",
    "free-flight_whitenoise-low_room1__w01",
    "FLY124__w03",
    "FLY124__w04",
]
N_TRACE_K = 5


def corr_matrix(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Pairwise correlation of the detrended rows + the mean off-diagonal."""
    n = x.shape[0]
    xm = x - np.nanmean(x, axis=1, keepdims=True)
    c = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            m = np.isfinite(xm[i]) & np.isfinite(xm[j])
            if m.sum() > 8 and np.std(xm[i][m]) > 0 and np.std(xm[j][m]) > 0:
                c[i, j] = float(np.corrcoef(xm[i][m], xm[j][m])[0, 1])
    off = c[~np.eye(n, dtype=bool)]
    return c, float(np.nanmean(off)) if np.isfinite(off).any() else float("nan")


def wiggle_stats(key: str) -> dict[str, Any]:
    """delta_k(t) diagnostics for one window's strongest rotor.

    Traces are UNGATED on purpose: on DREGON the k = 2 harmonic of a twin pair
    is collided at every frame, so the gate would delete exactly the trace in
    question. The twin's position is instead drawn on the same axis, where an
    interference-driven ridge would be visible as the ridge following it.
    """
    d = np.load(OUT / "traces" / f"{key}.npz")
    rot = int(d["rotor"])
    t, ks = d["on__t"], d["on__k"]
    prom = np.array([entry(key, rot, "on", int(k))[I_PROM] or -99.0 for k in ks])
    cand = [i for i in range(len(ks)) if ks[i] >= 2]
    sel = sorted(sorted(cand, key=lambda i: -prom[i])[:N_TRACE_K], key=lambda i: ks[i])
    over = [i for i in sel if prom[i] >= PROM_BAR]
    out: dict[str, Any] = {
        "key": key,
        "rotor": rot,
        "t": t,
        "ks": ks[sel],
        "prom": prom[sel],
        "over_bar": np.array([prom[i] >= PROM_BAR for i in sel]),
        "d_peak": d["on__d_peak"][sel],
        "d_pp": d["on__d_pp"][sel],
        "d_peak_off": d["off__d_peak"][sel],
        "search_half": np.array(
            [entry(key, rot, "on", int(ks[i]))[I_SEARCH] for i in sel], dtype=float
        ),
        "keep": d["on__keep"][sel],
        "r_ft": d["r_ft"],
        "ft": d["ft"],
    }
    # every other rotor's rate difference: where its harmonic k sits on the SAME
    # rescaled axis, for EVERY k (the offset r_j - r_rot is k-independent)
    out["twin_tracks"] = [
        (j, np.interp(t, d["ft"], d["r_ft"][j] - d["r_ft"][rot]))
        for j in range(d["r_ft"].shape[0])
        if j != rot
    ]
    for name in ("d_peak", "d_pp", "d_peak_off"):
        c, mr = corr_matrix(np.asarray(out[name], dtype=float))
        out[f"corr_{name}"], out[f"meanr_{name}"] = c, mr
    # r(delta_2, delta_k) for EVERY k, alongside that k's prominence: the
    # non-cherry-picked version of "does the wiggle appear at all harmonics?"
    i2 = int(np.where(np.asarray(ks) == 2)[0][0])
    base = np.asarray(d["on__d_peak"][i2], dtype=float)
    r_vs_k, r_vs_k_null = [], []
    b2 = np.asarray(d["off__d_peak"][i2], dtype=float)
    for a in range(len(ks)):
        for src, dst in ((d["on__d_peak"][a], r_vs_k), (d["off__d_peak"][a], r_vs_k_null)):
            ref = base if dst is r_vs_k else b2
            y = np.asarray(src, dtype=float)
            m = np.isfinite(y) & np.isfinite(ref)
            dst.append(
                float(np.corrcoef(y[m], ref[m])[0, 1])
                if m.sum() > 8 and np.std(y[m]) > 0 and np.std(ref[m]) > 0
                else np.nan
            )
    out["r_vs_k"] = np.array(r_vs_k)
    out["r_vs_k_null"] = np.array(r_vs_k_null)
    if len(over) >= 2:
        _, mr_over = corr_matrix(d["on__d_peak"][over])
        _, mr_over_null = corr_matrix(d["off__d_peak"][over])
    else:
        mr_over = mr_over_null = float("nan")
    out["meanr_over_bar"] = mr_over
    out["meanr_over_bar_null"] = mr_over_null
    out["ks_over_bar"] = [int(ks[i]) for i in over]
    # per-k mean and spread — FLAT in k for a real shaft-rate offset, ~1/k for a
    # fixed-frequency artefact
    x = np.asarray(out["d_peak"], dtype=float)
    out["mean_per_k"] = np.nanmean(x, axis=1)
    out["std_per_k"] = np.nanstd(x, axis=1)
    # the shape test uses EVERY harmonic, not just the five plotted as traces
    xa = np.asarray(d["on__d_peak"], dtype=float)
    out["all_k"] = np.asarray(ks, dtype=float)
    out["all_mean"] = np.nanmean(xa, axis=1)
    out["all_std"] = np.nanstd(xa, axis=1)
    out["all_prom"] = prom
    # slope between the two most prominent harmonics: 1.0 = shared rate
    # deviation, k_a / k_b = shared fixed FREQUENCY
    o2 = sorted(sel, key=lambda i: -prom[i])[:2]
    if len(o2) == 2:
        a, b = np.asarray(d["on__d_peak"][o2[0]]), np.asarray(d["on__d_peak"][o2[1]])
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        r = float(np.corrcoef(a, b)[0, 1])
        out["pair"] = (int(ks[o2[0]]), int(ks[o2[1]]))
        out["pair_r"] = r
        out["pair_slope_lo"] = r * float(np.std(b) / np.std(a)) if np.std(a) > 0 else np.nan
        out["pair_slope_hi"] = float(np.std(b) / np.std(a)) / r if r != 0 else np.nan
        out["pair_freq_pred"] = float(ks[o2[0]]) / float(ks[o2[1]])
    # prominence-weighted mean trace over the harmonics that clear the bar
    src = over if over else sel
    xx = np.asarray(d["on__d_peak"][src], dtype=float)
    wgt = np.maximum(prom[src], 0.0)[:, None] * np.isfinite(xx)
    den = wgt.sum(axis=0)
    out["d_bar"] = np.where(
        den > 0,
        np.nansum(np.where(np.isfinite(xx), xx, 0.0) * wgt, axis=0) / np.maximum(den, 1e-9),
        np.nan,
    )
    # telemetry-lag test: a late telemetry track would make delta = tau * dg/dt
    g = np.interp(t, out["ft"], out["r_ft"][rot])
    dg = np.gradient(g, t)
    db = out["d_bar"] - np.nanmean(out["d_bar"])
    lags = np.arange(-8, 9)
    rr = []
    for lag in lags:
        a = db if lag == 0 else (db[lag:] if lag > 0 else db[: lag or None])
        b = dg - dg.mean()
        b = b[: len(a)] if lag > 0 else b[-len(a) :]
        m = np.isfinite(a) & np.isfinite(b)
        rr.append(
            float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 8 and np.std(b[m]) > 0 else np.nan
        )
    out["lags_s"] = lags * NC.DISP_HOP_S
    out["lag_r"] = np.array(rr)
    out["dg_std"] = float(np.std(dg))
    out["dbar_std"] = float(np.nanstd(db))
    out["implied_tau_s"] = out["dbar_std"] / out["dg_std"] if out["dg_std"] > 0 else np.nan
    m = np.isfinite(db)
    out["r_with_g"] = float(np.corrcoef(db[m], g[m] - g[m].mean())[0, 1]) if m.sum() > 8 else np.nan
    x2 = np.where(np.isfinite(db), db, 0.0)
    sp = np.abs(np.fft.rfft(x2 * np.hanning(len(x2)))) ** 2
    fr = np.fft.rfftfreq(len(x2), d=NC.DISP_HOP_S)
    ok = fr > 0.05
    out["wiggle_hz"] = float(fr[ok][int(np.argmax(sp[ok]))])
    means = np.array([float(np.mean(out["r_ft"][r])) for r in range(4)])
    out["rotor_means"] = means
    out["twin_split_stats"] = [
        [
            j,
            round(float(np.mean(trk)), 3),
            round(float(np.std(trk)), 3),
            round(float(np.min(trk)), 3),
            round(float(np.max(trk)), 3),
        ]
        for j, trk in out["twin_tracks"]
    ]
    return out


def _argbest(v: np.ndarray) -> int:
    a = np.abs(np.asarray(v, dtype=float))
    return int(np.nanargmax(a)) if np.isfinite(a).any() else 0


def fig6() -> tuple[Path, dict[str, Any]]:
    S = {k: wiggle_stats(k) for k in WIG_ALL}
    fig, axes = plt.subplots(4, 2, figsize=(15.4, 17.0))
    cmap = plt.get_cmap("viridis")
    for c, (key, title, _col) in enumerate(WIG_PANELS):
        s = S[key]
        ax = axes[0, c]
        n = len(s["ks"])
        for i in range(n):
            ax.plot(
                s["t"],
                s["d_peak"][i],
                color=cmap(0.08 + 0.8 * i / max(n - 1, 1)),
                lw=2.2 if s["over_bar"][i] else 0.9,
                alpha=1.0 if s["over_bar"][i] else 0.5,
                label=f"k={int(s['ks'][i])} ({s['prom'][i]:.1f} dB)"
                + ("" if s["over_bar"][i] else ", under bar"),
            )
        first = True
        for j, trk in s["twin_tracks"]:
            if np.median(np.abs(trk)) > 4.0:
                continue  # never enters the strip
            ax.plot(
                s["t"],
                trk,
                color="#b03060",
                lw=1.6,
                ls=(0, (5, 2)),
                label="rotor {} rate - rotor {} rate (where ITS line sits)".format(j, s["rotor"])
                if first
                else None,
            )
            first = False
        ax.axhline(0.0, color="#0090a8", lw=1.6, ls="--", label="telemetry")
        ax.set_ylim(-1.7, 1.7)
        ax.set_title(
            f"(a{c + 1}) {title} — rotor {s['rotor']}\n"
            f"$\\delta_k(t)$; thick = ridge over the 6 dB bar.  Mean pairwise cross-$k$ "
            f"r = {s['meanr_d_peak']:+.2f} (off-comb null {s['meanr_d_peak_off']:+.2f})",
            fontsize=10,
        )
        ax.set_xlabel("time in window (s)")
        ax.set_ylabel("offset from telemetry (rev/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.0, ncol=2, loc="lower left", framealpha=0.9)

        axc = axes[1, c]
        cm = s["corr_d_peak"]
        im = axc.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
        labs = [f"k={int(v)}" + ("" if o else "*") for v, o in zip(s["ks"], s["over_bar"])]
        axc.set_xticks(range(n))
        axc.set_xticklabels(labs, fontsize=8, rotation=45)
        axc.set_yticks(range(n))
        axc.set_yticklabels(labs, fontsize=8)
        for i in range(n):
            for j in range(n):
                if np.isfinite(cm[i, j]):
                    axc.text(
                        j,
                        i,
                        f"{cm[i, j]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="k" if abs(cm[i, j]) < 0.6 else "w",
                    )
        axc.set_title(
            f"(b{c + 1}) cross-$k$ correlation of $\\delta_k(t)$   (* = ridge under the bar)\n"
            "a shared deviation must correlate; independent peak-pick noise cannot",
            fontsize=9.5,
        )
        fig.colorbar(im, ax=axc, fraction=0.046, pad=0.02)

    for c, (key, title, col) in enumerate(WIG_PANELS):
        s = S[key]
        axr = axes[2, c]
        kk = s["all_k"]
        axr.axhline(0.0, color="k", lw=1.0)
        axr.plot(
            kk,
            s["r_vs_k"],
            color=col,
            marker="o",
            ms=4,
            lw=1.6,
            label="r($\\delta_2$, $\\delta_k$) — measured",
        )
        axr.plot(
            kk,
            s["r_vs_k_null"],
            color="0.55",
            marker=".",
            ms=3,
            lw=1.0,
            ls="--",
            label="the same on the off-comb null",
        )
        axr.set_ylim(-1.05, 1.05)
        axr.set_xlabel("harmonic order $k$")
        axr.set_ylabel("correlation with $\\delta_2(t)$")
        axr.grid(alpha=0.3)
        axp2 = axr.twinx()
        axp2.fill_between(kk, 0, s["all_prom"], color=col, alpha=0.16, zorder=0)
        axp2.axhline(PROM_BAR, color=col, ls=":", lw=1.2)
        axp2.set_ylabel("prominence (dB, shaded)", fontsize=8)
        axp2.set_ylim(0, 36)
        axr.set_title(
            f"(c{c + 1}) {title.split(' ')[0]}: does the wiggle appear at EVERY harmonic, or only "
            "where a line exists?\ncorrelation of every $\\delta_k(t)$ with $\\delta_2(t)$, "
            "against that harmonic's prominence",
            fontsize=9.5,
        )
        axr.legend(fontsize=7.5, loc="upper right")

    axm = axes[3, 0]
    for key, title, col in WIG_PANELS:
        s = S[key]
        kk, mm, pp = s["all_k"], s["all_mean"], s["all_prom"]
        keep = kk >= 2
        strong = keep & (pp >= PROM_BAR)
        axm.plot(kk[keep], mm[keep], color=col, lw=1.0, alpha=0.45)
        axm.scatter(
            kk[keep],
            mm[keep],
            s=6 + 4.0 * np.clip(pp[keep], 0, 30),
            color=col,
            alpha=0.35,
            edgecolors="none",
        )
        axm.scatter(
            kk[strong],
            mm[strong],
            s=6 + 4.0 * np.clip(pp[strong], 0, 30),
            color=col,
            edgecolors="k",
            linewidths=0.8,
            zorder=5,
            label=f"{title.split(' ')[0]}: $\\delta_k$ (marker size = prominence; "
            "black edge = over the 6 dB bar)",
        )
        if strong.any():
            ref = float(np.mean(mm[strong & (kk <= 13)])) if (strong & (kk <= 13)).any() else 0.0
            kref = float(kk[strong][0])
            axm.axhline(ref, ls=":", lw=1.6, color=col, alpha=0.8)
            axm.plot(
                kk[keep],
                ref * kref / kk[keep],
                ls="--",
                lw=1.3,
                color=col,
                alpha=0.7,
                label=f"{title.split(' ')[0]}: $1/k$ prediction from $k$ = {kref:.0f}",
            )
    axm.axhline(0.0, color="k", lw=1.2)
    axm.set_xlabel("harmonic order $k$")
    axm.set_ylabel("time-mean $\\delta_k$ (rev/s)")
    axm.set_ylim(-0.95, 0.55)
    axm.grid(alpha=0.3)
    axm.set_title(
        "(d) the shape test, every harmonic of the same rotor: a real shaft-rate offset is "
        "FLAT in $k$ (dotted),\na fixed-FREQUENCY artefact falls as $1/k$ (dashed).  "
        "Small faint markers are harmonics with no measurable line.",
        fontsize=9.5,
    )
    axm.legend(fontsize=7.0, loc="lower right")

    axl = axes[3, 1]
    for key, title, col in WIG_PANELS:
        s = S[key]
        axl.plot(s["lags_s"], s["lag_r"], marker="o", color=col, lw=1.8, label=title)
    axl.axhline(0.0, color="k", lw=1.0)
    axl.set_xlabel("lag applied to $\\bar{\\delta}(t)$ (s)")
    axl.set_ylabel("correlation with $dg/dt$")
    axl.set_ylim(-1, 1)
    axl.grid(alpha=0.3)
    d0 = S[WIG_PANELS[0][0]]
    axl.set_title(
        "(e) telemetry-lag test: a merely late telemetry track gives "
        "$\\delta = \\tau\\,dg/dt$\nand would peak sharply here.  DREGON needs "
        f"$\\tau$ = {d0['implied_tau_s']:.2f} s to explain the wiggle amplitude",
        fontsize=10,
    )
    axl.legend(fontsize=8)
    fig.suptitle(
        "F6 — is the DREGON low-$k$ wiggle a real deviation of the shaft rate from telemetry, "
        "or interference?\nTraces: peak-pick offsets on a common 2 s / 0.25 s base, UNGATED "
        "(the twin-collision gate would delete the DREGON $k = 2$ trace entirely).",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0.004, 0.0, 0.996, 0.945))
    p = FIGS / "F6_wiggle.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    summary = {
        key: {
            "rotor": int(s["rotor"]),
            "ks": [int(v) for v in s["ks"]],
            "prominence_db": [round(float(v), 2) for v in s["prom"]],
            "ks_over_6dB_bar": s["ks_over_bar"],
            "mean_delta_per_k_rev_s": [round(float(v), 3) for v in s["mean_per_k"]],
            "std_delta_per_k_rev_s": [round(float(v), 3) for v in s["std_per_k"]],
            "mean_pairwise_r_all": round(s["meanr_d_peak"], 3),
            "mean_pairwise_r_over_bar": round(s["meanr_over_bar"], 3),
            "mean_pairwise_r_over_bar_offcomb_null": round(s["meanr_over_bar_null"], 3),
            "mean_pairwise_r_pulsepair": round(s["meanr_d_pp"], 3),
            "r_delta2_vs_deltak": {
                int(kv): round(float(rv), 3)
                for kv, rv in zip(s["all_k"], s["r_vs_k"])
                if np.isfinite(rv)
            },
            "prominence_all_k_db": {
                int(kv): round(float(pv), 2) for kv, pv in zip(s["all_k"], s["all_prom"])
            },
            "strongest_pair": list(s.get("pair", [])),
            "pair_r": round(float(s.get("pair_r", np.nan)), 3),
            "pair_slope_bracket": [
                round(float(s.get("pair_slope_lo", np.nan)), 3),
                round(float(s.get("pair_slope_hi", np.nan)), 3),
            ],
            "pair_slope_if_fixed_frequency": round(float(s.get("pair_freq_pred", np.nan)), 3),
            "best_lag_s": round(float(s["lags_s"][_argbest(s["lag_r"])]), 3),
            "best_lag_r": round(float(s["lag_r"][_argbest(s["lag_r"])]), 3),
            "implied_telemetry_lag_s": round(float(s["implied_tau_s"]), 3),
            "r_with_telemetry_level": round(float(s["r_with_g"]), 3),
            "dominant_wiggle_hz": round(float(s["wiggle_hz"]), 3),
            "rotor_mean_rev_s": [round(float(v), 3) for v in s["rotor_means"]],
            "other_rotor_offset_mean_std_min_max": s["twin_split_stats"],
            "frac_twin_collided": round(1.0 - float(np.mean(s["keep"])), 3),
        }
        for key, s in S.items()
    }
    return p, summary


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    print(write_prominence())
    print(fig4())
    print(fig5())
    p1, info1 = fig1()
    print(p1)
    (OUT / "f1_selection.json").write_text(json.dumps(info1, indent=1, default=float))
    p6, sum6 = fig6()
    print(p6)
    (OUT / "wiggle_stats.json").write_text(json.dumps(sum6, indent=1, default=float))
    print(json.dumps(sum6, indent=1, default=float))
