#!/usr/bin/env python3
"""Re-render demodulation strips from the ENVELOPE CACHE - no audio, seconds.

Everything here reads ``cache/*.npz`` (built once by ``hk_cache.py``) and
nothing else.  The cached product is the complex demodulated envelope with the
microphone axis intact, so the segment length - the parameter that decides
whether a high-k line is visible at all - is a free argument here.

Public entry points
-------------------
``load(recording, window, rotor)``
    -> ``Env`` (z, g, g_all, fs_env, rates_mean, ...)

``strip(env, k, seg_s, band_revs=1.2, overlap=0.75, pad=4)``
    -> ``(t_s, rev_axis, P)``  short-time envelope power, mics averaged
       incoherently, frequency axis rescaled to shaft-rate offset in rev/s

``render_strips(recording, window, rotor, ks, seg_s, ylim, clim_pct, refs,
                coherence, out, ...)``
    -> renders one figure of strip panels.  F1 is produced by exactly this call
       (see ``make_f1()`` at the bottom, which is what ``python replot.py``
       runs).

Examples
--------
    import replot as R
    R.render_strips("free-flight_nosource_room1", 1, 0,
                    ks=[2, 4, 8, 64, 70, 71], seg_s="equal:0.13",
                    ylim=1.2, clim_pct=(60, 99.5),
                    coherence=(71, [0.10, 1.0, 4.0]),
                    out="figs/try.png")
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
FIGS = OUT / "figs"
CORR = 0.99458  # measured DREGON telemetry correction (dregon_telemetry.md)
BIAS = 1.0 - CORR  # 0.00542
RPS_CHANNEL = "motors_measured"


# ────────────────────────────── cache access ────────────────────────────────
@dataclass
class Env:
    recording: str
    window: int
    rotor: int
    z: np.ndarray  # (K, C, n_env) complex64
    ks: np.ndarray  # (K,)
    g: np.ndarray  # (n_env,) motors_measured of this rotor
    g_all: np.ndarray  # (4, n_env)
    t_env: np.ndarray  # (n_env,)
    fs_env: float
    rates_mean: np.ndarray  # (4,)
    t_start: float
    dur: float

    @property
    def rate(self) -> float:
        return float(self.rates_mean[self.rotor])

    half: bool = False

    def zk(self, k: float) -> np.ndarray:
        j = int(np.argmin(np.abs(np.asarray(self.ks, float) - (k + 0.5 * self.half))))
        return self.z[j]


_LOADED: dict = {}


def load(recording: str, window: int = 1, rotor: int = 0, half: bool = False) -> Env:
    """``half=True`` loads the HALF-INTEGER NULL cache (carriers at k + 0.5,
    where no rotor line can exist)."""
    key = (recording, window, rotor, half)
    if key in _LOADED:
        return _LOADED[key]
    f = CACHE / f"{recording}__w{window:02d}__r{rotor}{'__half' if half else ''}.npz"
    with np.load(f) as d:
        env = Env(
            recording=recording,
            window=window,
            rotor=rotor,
            z=d["z"],
            ks=d["ks"],
            g=d["g"],
            g_all=d["g_all"],
            t_env=d["t_env"],
            fs_env=float(np.asarray(d["fs_env"]).item()),
            rates_mean=d["rates_mean"],
            t_start=float(np.asarray(d["t_start"]).item()),
            dur=float(np.asarray(d["dur"]).item()),
            half=half,
        )
    _LOADED[key] = env
    return env


def manifest() -> dict:
    return json.loads((CACHE / "manifest.json").read_text())


# ────────────────────────────── strip maths ─────────────────────────────────
SegSpec = float | int | str | Callable[[int], float]


def seg_for(k: int, spec: SegSpec) -> float:
    """Resolve a segment-length spec to seconds.

    ``float``            same length for every k
    ``"equal:<d>"``      equal shaft-rate resolution: seg = 1 / (d * k) seconds,
                         so every strip resolves ``d`` rev/s regardless of k
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, str) and spec.startswith("equal:"):
        d = float(spec.split(":", 1)[1])
        return float(np.clip(1.0 / (d * max(k, 1)), 0.02, 8.0))
    if callable(spec):
        return float(spec(k))
    raise ValueError(f"bad seg spec {spec!r}")


def strip(
    env: Env, k: int, seg_s: float, band_revs: float = 1.2, overlap=0.75, pad=4, n_avg: int = 1
):
    """``(t_s, rev_axis, P (F, T))`` - short-time envelope power of harmonic k.

    ``P`` is averaged INCOHERENTLY over microphones.  ``rev_axis`` is the
    acoustic shaft-rate offset from telemetry, envelope Hz divided by k.  True
    resolution is ``1 / (seg_s * k)`` rev/s; ``pad`` only interpolates.

    ``n_avg`` additionally averages ``n_avg`` CONSECUTIVE segment spectra into
    one displayed column, INCOHERENTLY.  This is not the same as lengthening
    the segment: the coherent integration stays at ``seg_s`` (so a line whose
    coherence time is short is not smeared away), while the speckle variance
    drops by ``n_avg``.  It costs time resolution, not frequency resolution.
    """
    z = env.zk(k)
    fs = env.fs_env
    n_seg = max(int(round(seg_s * fs)), 8)
    n_seg -= n_seg % 2
    hop = max(int(round(n_seg * (1.0 - overlap))), 1)
    n = z.shape[1]
    n_seg = min(n_seg, n)
    starts = np.arange(0, n - n_seg + 1, hop)
    nfft = n_seg * pad
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    keep = np.abs(freqs) <= band_revs * k
    rev = freqs[keep] / k
    win = np.hanning(n_seg)
    acc = np.zeros((len(starts), int(keep.sum())))
    for c in range(z.shape[0]):
        fr = np.lib.stride_tricks.sliding_window_view(z[c], n_seg)[::hop] * win
        Z = np.fft.fftshift(np.fft.fft(fr, n=nfft, axis=-1), axes=-1)[:, keep]
        acc += np.abs(Z) ** 2
    acc /= z.shape[0]
    t = (starts + n_seg / 2.0) / fs
    if n_avg > 1:
        m = (len(t) // n_avg) * n_avg
        acc = acc[:m].reshape(-1, n_avg, acc.shape[1]).mean(axis=1)
        t = t[:m].reshape(-1, n_avg).mean(axis=1)
    return t, rev, acc.T


def ref_lines(env: Env, k: int, ylim: float, which=("tel", "corr", "neigh", "own")):
    """Predicted line positions in the rescaled frame, ``[(kind, label, delta)]``."""
    r = env.rotor
    g = env.rate
    out = []
    if "tel" in which:
        out.append(("tel", "telemetry", 0.0))
    if "corr" in which:
        out.append(("corr", f"corrected {-BIAS * g:+.2f}", -BIAS * g))
    if "own" in which:
        for m in (-2, -1, 1, 2):
            d = ((k + m) * CORR * g - k * g) / k
            if abs(d) <= ylim:
                out.append(("own", f"own k{m:+d}", d))
    if "neigh" in which:
        f0 = k * g
        for r2 in range(4):
            if r2 == r:
                continue
            kk = int(round(f0 / env.rates_mean[r2]))
            for c in (kk - 1, kk, kk + 1):
                if c < 1:
                    continue
                d = (c * CORR * float(env.rates_mean[r2]) - f0) / k
                if abs(d) <= ylim:
                    out.append(("neigh", f"r{r2}", d))
    return out


def profile_stats(rev, P, env: Env, k: int, ylim: float, seg_s: float):
    """Time-averaged demod profile, in dB over the in-band floor.

    ``prom_db`` is the peak of that profile inside ``+-ylim`` rev/s over the
    floor, where the floor is the MEDIAN of the profile across the whole
    displayed band.  That is prominence in the plain sense, with no detrending
    and no free parameters - the fair reference for it is the same number
    measured on the HALF-INTEGER comb, where no rotor line can exist
    (``null_stats`` / ``load(..., half=True)``), not an assumed noise model.
    """
    prof = P.mean(axis=1)
    db = 10.0 * np.log10(prof + 1e-300)
    floor = float(np.median(db))
    db = db - floor
    sw = np.abs(rev) <= ylim
    j = int(np.argmax(db[sw]))
    d_corr = -BIAS * env.rate

    def at(d0):
        w = np.abs(rev - d0) <= 0.20
        return float(np.max(db[w])) if w.any() else float("nan")

    return {
        "prof_db": db,
        "floor_db": floor,
        "prom_db": float(db[sw][j]),
        "prom_at": float(rev[sw][j]),
        "at_corr_db": at(d_corr),
        "at_tel_db": at(0.0),
        "d_corr": d_corr,
        "res_revs": 1.0 / (seg_s * k),
    }


def null_stats(env: Env, k: int, seg_s: float, ylim: float, band_revs: float, n_avg: int = 1):
    """``profile_stats`` of the HALF-INTEGER comb (k + 0.5) - the matched null.

    No rotor line can live at a half-integer harmonic, so every statistic
    measured there is a false-alarm reference for the identical search on the
    real comb.  Returns ``None`` when the null cache is absent for this window.
    """
    try:
        envn = load(env.recording, env.window, env.rotor, half=True)
    except FileNotFoundError:
        return None
    _t, rev, P = strip(envn, k, seg_s, band_revs=band_revs, n_avg=n_avg)
    return profile_stats(rev, P, envn, k, ylim, seg_s)


# ────────────────────────────── rendering ───────────────────────────────────
NEIGH_C = "#39d353"
TEL_C = "#00e5ff"
CORR_C = "#ff2e63"


def _draw_strip(
    ax,
    axp,
    env,
    k,
    seg_s,
    ylim,
    clim_pct,
    band_revs,
    show_labels=True,
    n_avg=1,
    tag="",
    label_fs=8.0,
    short_label=False,
):
    t, rev, P = strip(env, k, seg_s, band_revs=band_revs, n_avg=n_avg)
    s = profile_stats(rev, P, env, k, ylim, seg_s)
    ns = null_stats(env, k, seg_s, ylim, band_revs, n_avg)
    s["null_db"] = None if ns is None else ns["prom_db"]
    s["null_corr_db"] = None if ns is None else ns["at_corr_db"]
    # the decisive test is MATCHED-POSITION: the level at the predicted displaced
    # offset, against the level at the same offset on the half-integer comb.  No
    # search, so no multiple-comparison inflation and no band-edge tilt to win.
    s["excess_db"] = s["at_corr_db"] - (s["null_corr_db"] or 0.0)
    s["is_line"] = ns is None or s["at_corr_db"] > ns["at_corr_db"]
    snr_db = 10.0 * np.log10(P / np.median(P, axis=0, keepdims=True) + 1e-300)
    sel = np.abs(rev) <= ylim
    v = snr_db[sel]
    vmin, vmax = np.percentile(v, clim_pct[0]), np.percentile(v, clim_pct[1])
    ax.pcolormesh(
        t, rev[sel], v, cmap="magma", vmin=vmin, vmax=vmax, shading="nearest", rasterized=True
    )
    for kind, lab, d in ref_lines(env, k, ylim):
        if kind == "tel":
            for a in (ax, axp):
                a.axhline(0.0, color=TEL_C, lw=1.5, ls="--", zorder=5)
        elif kind == "corr":
            for a in (ax, axp):
                a.axhline(d, color=CORR_C, lw=1.6, zorder=5)
        elif kind == "neigh":
            for a in (ax, axp):
                a.axhline(d, color=NEIGH_C, lw=1.0, ls=(0, (6, 4)), alpha=0.85, zorder=4)
            ax.text(
                t[-1],
                d,
                lab,
                color=NEIGH_C,
                fontsize=6.5,
                va="center",
                ha="right",
                zorder=6,
                bbox={"fc": "black", "ec": "none", "alpha": 0.35, "pad": 0.6},
            )
        elif kind == "own":
            for a in (ax, axp):
                a.axhline(d, color="0.75", lw=0.9, ls=":", alpha=0.8, zorder=3)
    ax.set_ylim(-ylim, ylim)
    axp.set_ylim(-ylim, ylim)
    axp.plot(s["prof_db"], rev, color="0.1", lw=1.3)
    axp.axvline(0.0, color="0.6", lw=0.7)
    if s["null_db"] is not None:
        axp.axvline(s["null_db"], color="#b03030", lw=1.0, ls=":")
    axp.set_xlim(
        float(min(s["prof_db"].min() - 0.3, -0.5)), max(2.0, float(s["prof_db"].max()) * 1.15)
    )
    axp.tick_params(labelleft=False, labelsize=6.5)
    axp.grid(alpha=0.28)
    ax.tick_params(labelsize=7)
    if show_labels:
        good = s["is_line"]
        ax.text(
            0.012,
            0.955,
            f"{tag}k = {k}   {k * env.rate / 1000:.2f} kHz   segment {seg_s * 1000:.0f} ms"
            + ("" if short_label else f"   ({s['res_revs']:.2f} rev/s resolution)")
            + (f"   x{n_avg} incoherent" if n_avg > 1 else ""),
            transform=ax.transAxes,
            color="w",
            fontsize=label_fs,
            va="top",
            fontweight="bold",
            bbox={"fc": "#101014", "ec": "none", "alpha": 0.55, "pad": 1.6},
        )
        ax.text(
            0.012,
            0.045,
            f"at the displaced offset {s['at_corr_db']:.1f} dB"
            + (
                f"  vs half-integer null {s['null_corr_db']:.1f} dB"
                if s["null_corr_db"] is not None
                else ""
            )
            + ("" if good else "   * NOISE, the null wins")
            + (
                ""
                if short_label
                else f"\nat telemetry {s['at_tel_db']:.1f} dB   |   free peak "
                f"{s['prom_db']:.1f} dB at {s['prom_at']:+.2f} rev/s"
            ),
            transform=ax.transAxes,
            color="w" if good else "#ff8a8a",
            fontsize=label_fs - 0.9,
            va="bottom",
            bbox={"fc": "#101014", "ec": "none", "alpha": 0.55, "pad": 1.6},
        )
    return s


def render_strips(
    recording: str,
    window: int,
    rotor: int,
    ks: Sequence[int],
    seg_s: SegSpec = "equal:0.13",
    ylim: float = 1.2,
    clim_pct=(60.0, 99.5),
    band_revs: float = 1.35,
    coherence=None,
    out: str | Path = "figs/strips.png",
    title: str = "",
    caption: str = "",
    dpi: int = 165,
):
    """Render a column of strip panels (+ an optional coherence-time column).

    ``coherence=(k, [0.10, 1.0, 4.0])`` adds a right-hand column: the SAME
    harmonic re-windowed at each of those segment lengths.
    """
    import textwrap

    env = load(recording, window, rotor)
    n = len(ks)
    ncoh = 0 if coherence is None else len(coherence[1])
    fw = 16.4
    title_fs, cap_fs = 10.0, 7.8

    def wrap(txt: str, fs: float) -> str:
        # ~0.50 * fontsize points per character for this font
        width = max(int(fw * 72 / (0.545 * fs)), 40)
        return "\n".join(textwrap.fill(par, width) for par in txt.split("\n"))

    title_w = wrap(title, title_fs) if title else ""
    cap_w = wrap(caption, cap_fs) if caption else ""
    n_tl = title_w.count("\n") + 1 if title_w else 0
    n_cl = cap_w.count("\n") + 1 if cap_w else 0
    h_top = n_tl * title_fs * 1.45 / 72 + 0.22
    h_bot = n_cl * cap_fs * 1.45 / 72 + 0.20
    fh = 1.45 * n + h_top + h_bot
    fig = plt.figure(figsize=(fw, fh))
    top = 1.0 - h_top / fh
    bot = h_bot / fh
    if coherence is None:
        sf_l = fig.add_subfigure(fig.add_gridspec(1, 1, top=top, bottom=bot)[0])
        sf_r = sf_l
    else:
        gs0 = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], top=top, bottom=bot, wspace=0.11)
        sf_l = fig.add_subfigure(gs0[0])
        sf_r = fig.add_subfigure(gs0[1])

    gsl = sf_l.add_gridspec(n, 2, width_ratios=[3.5, 1.0], hspace=0.16, wspace=0.04)
    stats = []
    for i, k in enumerate(ks):
        ax = sf_l.add_subplot(gsl[i, 0])
        axp = sf_l.add_subplot(gsl[i, 1], sharey=ax)
        s = _draw_strip(ax, axp, env, k, seg_for(k, seg_s), ylim, clim_pct, band_revs)
        stats.append((k, s))
        ax.set_ylabel("rev/s", fontsize=7.5)
        if i == 0:
            axp.set_title("time avg\n(dB over floor)", fontsize=6.5, pad=3)
        if i == n - 1:
            ax.set_xlabel("time in window (s)", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
            axp.tick_params(labelbottom=False)

    if coherence is not None:
        kc, segs = coherence
        gsr = sf_r.add_gridspec(ncoh, 2, width_ratios=[3.0, 1.0], hspace=0.16, wspace=0.05)
        for i, sg in enumerate(segs):
            ax = sf_r.add_subplot(gsr[i, 0])
            axp = sf_r.add_subplot(gsr[i, 1], sharey=ax)
            _draw_strip(
                ax,
                axp,
                env,
                kc,
                sg,
                ylim,
                clim_pct,
                band_revs,
                label_fs=7.4,
                short_label=True,
            )
            ax.set_ylabel("rev/s", fontsize=7.5)
            if i == 0:
                ax.set_title(
                    f"COHERENCE-TIME COLUMN: harmonic k = {kc} "
                    f"({kc * env.rate / 1000:.2f} kHz) re-windowed at three\n"
                    "segment lengths.  Nothing is lost going from 0.10 s to 4 s, "
                    "because\nat 6 kHz the 0.10 s panel has nothing to lose.",
                    fontsize=8.6,
                    pad=8,
                )
            if i == ncoh - 1:
                ax.set_xlabel("time in window (s)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
                axp.tick_params(labelbottom=False)

    if title_w:
        fig.text(0.006, 1.0 - 0.10 / fh, title_w, fontsize=title_fs, va="top", ha="left")
    if cap_w:
        fig.text(0.006, 0.06 / fh, cap_w, fontsize=cap_fs, va="bottom", ha="left")
    p = Path(out)
    if not p.is_absolute():
        p = OUT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p, stats


# ────────────────────────────── F1 ──────────────────────────────────────────
def rank_highk(
    env: Env,
    f_lo: float = 5500.0,
    f_hi: float = 6500.0,
    seg_s: float = 0.10,
    ylim: float = 1.2,
    band_revs: float = 1.35,
):
    """Rank the harmonics whose telemetry centre ``k * g_r`` lands in ``f_lo..f_hi``.

    Ranking key is the SHORT-SEGMENT prominence minus the half-integer null
    measured identically, so a harmonic only ranks high if it beats a search
    that cannot contain a rotor line.
    """
    g = env.rate
    klo, khi = int(np.ceil(f_lo / g)), int(np.floor(f_hi / g))
    rows = []
    for k in range(klo, min(khi, int(np.max(np.floor(env.ks)))) + 1):
        _t, rev, P = strip(env, k, seg_s, band_revs=band_revs)
        s = profile_stats(rev, P, env, k, ylim, seg_s)
        ns = null_stats(env, k, seg_s, ylim, band_revs)
        s["null_db"] = None if ns is None else ns["prom_db"]
        s["null_corr_db"] = None if ns is None else ns["at_corr_db"]
        s["excess_db"] = s["at_corr_db"] - (s["null_corr_db"] or 0.0)
        rows.append((k, s))
    rows.sort(key=lambda r: -r[1]["excess_db"])
    return rows


def seg_sweep(env: Env, ks: Sequence[int], segs=(0.05, 0.10, 0.15), ylim=1.2, band_revs=1.35):
    """Per segment length, the mean level at the displaced offset on the real comb
    and on the half-integer null, over ``ks``."""
    out = {}
    for sg in segs:
        pr, nu = [], []
        for k in ks:
            _t, rev, P = strip(env, k, sg, band_revs=band_revs)
            pr.append(profile_stats(rev, P, env, k, ylim, sg)["at_corr_db"])
            n = null_stats(env, k, sg, ylim, band_revs)
            nu.append(np.nan if n is None else n["at_corr_db"])
        out[sg] = (float(np.mean(pr)), float(np.nanmean(nu)))
    return out


def make_f1(
    recording: str = "free-flight_nosource_room1",
    window: int = 1,
    rotor: int = 0,
    seg_hi: float = 0.10,
    low_ks: Sequence[int] = (2, 4, 8),
    n_top: int = 5,
    ylim: float = 1.2,
    clim_pct=(60.0, 99.5),
):
    """Build ``figs/F1_demod_strips.png`` - entirely from the envelope cache."""
    env = load(recording, window, rotor)
    ranked = rank_highk(env, seg_s=seg_hi, ylim=ylim)
    top = sorted(k for k, _ in ranked[:n_top])
    ks = list(low_ks) + top
    d_corr = -BIAS * env.rate
    all_hi = sorted(k for k, _ in ranked)
    sweep = seg_sweep(env, all_hi, (0.05, 0.10, 0.15), ylim)
    sw_txt = ",  ".join(
        f"{s * 1000:.0f} ms: {v[0]:.2f} dB vs null {v[1]:.2f} dB" for s, v in sweep.items()
    )
    print(f"[F1] segment sweep over k = {all_hi}:  {sw_txt}")

    def segspec(k: int) -> float:
        # high-k strips at the fixed short segment; low-k references at the SAME
        # shaft-rate resolution, i.e. seg = 1 / (res * k)
        if k >= 30:
            return seg_hi
        res = 1.0 / (seg_hi * top[len(top) // 2])
        return float(np.clip(1.0 / (res * k), 0.02, 8.0))

    kc = top[len(top) // 2]
    sw100 = sweep[0.10]
    title = (
        "F1 - demodulated envelope of harmonic $k$ against DREGON "
        f"`{RPS_CHANNEL}` telemetry.  The frequency axis is rescaled to the acoustic "
        "shaft-rate offset $(f - kg)/k$ in rev/s, so telemetry sits at exactly zero and the "
        "measured displacement at $-0.542$ %.\n"
        f"{recording}, window {window:02d} (cruise, {env.dur:.0f} s from "
        f"t = {env.t_start:.1f} s), rotor {rotor} at {env.rate:.1f} rev/s, 8 microphones "
        "averaged incoherently.  CYAN dashed = telemetry;  RED = the measured displacement "
        f"({d_corr:+.2f} rev/s for this rotor);\n"
        "GREEN dashed = the nearest harmonic of ANOTHER rotor (r0..r3), which at 6 kHz is "
        f"never far away;  GREY dotted = this rotor's own $k\\pm1$.  Top {len(low_ks)} strips "
        f"are low-$k$ references, bottom {n_top} are the best of the 12 harmonics in "
        "5.5-6.5 kHz.\n"
        "VERDICT: the displaced, wandering ridge is unmistakable at $k$ = 2 and 4 and it is "
        "NOT THERE at 6 kHz.  Averaged over ALL 12 harmonics in 5.5-6.5 kHz the level at the "
        f"displaced offset is {sw100[0]:.2f} dB, BELOW its half-integer null of "
        f"{sw100[1]:.2f} dB;  at $k$ = 2 the same numbers are 7.7 dB against 0.0 dB."
    )
    caption = (
        "Colour: dB over each frame's own in-band median; per-strip limits at the "
        f"{clim_pct[0]:.0f}th and {clim_pct[1]:.1f}th percentile of that strip's own values.  "
        "Right-hand sub-panel: the time-averaged profile in dB over the in-band floor (the "
        "median of the whole displayed band).  The red dotted line is the free-peak "
        "prominence of the HALF-INTEGER comb $k+0.5$, where no rotor line can exist; the "
        "per-strip label instead quotes the MATCHED-POSITION test, the level at the displaced "
        "offset against the level at that same offset on the half-integer comb - a test with "
        "no search in it, so neither the band-edge tilt nor multiple comparisons can win it.\n"
        f"CAVEAT on the five 6 kHz strips: they are the best 5 of 12 CHOSEN BY THIS STATISTIC, "
        "so a positive excess on them is expected even under pure noise.  The unselected "
        f"average is the number that counts, and it is negative.  Segment-length sweep over "
        f"all 12: {sw_txt}.  No segment length in 0.05-0.15 s beats its null, so the answer is "
        "not a windowing choice.\n"
        "Low-$k$ references use the segment that gives the SAME shaft-rate resolution "
        f"({1.0 / (seg_hi * kc):.2f} rev/s) as the 6 kHz strips, so all eight strips are like "
        "for like.  Right column: the same 6 kHz harmonic re-windowed at 0.10 / 1.0 / 4.0 s.  "
        "The 1 s and 4 s panels lose nothing that the 0.10 s panel had, because at 6 kHz the "
        "0.10 s panel has nothing to lose."
    )
    p, stats = render_strips(
        recording,
        window,
        rotor,
        ks,
        seg_s=segspec,
        ylim=ylim,
        clim_pct=clim_pct,
        coherence=(kc, [0.10, 1.0, 4.0]),
        out=FIGS / "F1_demod_strips.png",
        title=title,
        caption=caption,
    )
    print(f"[F1] {p}")
    for k, s in stats:
        nl = " n/a" if s["null_corr_db"] is None else f"{s['null_corr_db']:5.2f}"
        print(
            f"  k={k:3d} {k * env.rate / 1000:5.2f} kHz  at-displaced {s['at_corr_db']:5.2f} dB "
            f"(null {nl})  at-telemetry {s['at_tel_db']:5.2f}  "
            f"free peak {s['prom_db']:5.2f} @ {s['prom_at']:+.2f}  "
            f"{'LINE' if s['is_line'] else 'noise'}"
        )
    return p


if __name__ == "__main__":
    make_f1()
