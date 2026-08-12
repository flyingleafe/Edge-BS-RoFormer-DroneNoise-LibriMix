"""Figures for the residual-attribution campaign."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "plot_identifiability",
    "plot_synthetic_recovery",
    "plot_real_fit",
    "plot_controls",
    "plot_coherence",
]

_ROTOR_C = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
_CONTROL_C = {
    "true": "#2ca02c",
    "rot45": "#1f77b4",
    "mirror_z": "#ff7f0e",
    "random": "#7f7f7f",
    "centroid": "#d62728",
}


def _save(fig, path: str | Path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_identifiability(freqs, diag, alias_f, path) -> str:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for r in range(diag["vif"].shape[1]):
        ax[0].loglog(freqs, diag["vif"][:, r], color=_ROTOR_C[r], lw=1.2, label=f"rotor {r}")
    ax[0].axhline(10, color="k", ls=":", lw=1)
    ax[0].set(xlabel="Hz", ylabel="VIF", title="per-rotor variance inflation\n(1 = orthogonal)")
    ax[0].legend(fontsize=7)
    ax[1].semilogx(freqs, diag["max_cos"], color="k", lw=1.2)
    ax[1].axhline(0.9, color="r", ls=":", lw=1)
    ax[1].set(xlabel="Hz", ylabel="max |cos|", title="worst rotor-pair collinearity", ylim=(0, 1))
    ax[2].loglog(freqs, diag["cond"], color="k", lw=1.2, label="off-diagonal design")
    ax[2].loglog(freqs, diag["cond_with_diag"], color="gray", lw=1, ls="--", label="joint design")
    ax[2].set(xlabel="Hz", ylabel="cond", title="condition number")
    ax[2].legend(fontsize=7)
    for a in ax:
        a.axvline(alias_f, color="tab:orange", ls="-.", lw=1)
        a.grid(alpha=0.25, which="both")
    fig.suptitle(
        "Geometric identifiability of the 4-rotor + 8-diagonal model "
        f"(orange = classical aliasing frequency {alias_f:.0f} Hz)",
        y=1.05,
    )
    return _save(fig, path)


def plot_synthetic_recovery(cases, path) -> str:
    """``cases``: list of ``(label, freqs, p_true (F,R), p_hat (F,R))``.

    Band-averaged before plotting. Per-bin NNLS returns exact zeros for a rotor
    it deactivates, which on a log axis is a wall of dropouts that hides the
    thing being shown.
    """
    from .csd import band_average, band_edges

    n = len(cases)
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False, sharey=True)
    edges = band_edges(50.0, 8000.0, 40)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    lo = np.inf
    for i, (label, f, pt, ph) in enumerate(cases):
        a = ax[0, i]
        tb, hb = band_average(f, pt, edges), band_average(f, ph, edges)
        for r in range(pt.shape[1]):
            a.loglog(ctr, tb[:, r], color=_ROTOR_C[r], lw=1.6, alpha=0.9, label=f"rotor {r}")
            a.loglog(ctr, np.maximum(hb[:, r], 1e-30), color=_ROTOR_C[r], lw=1.1, ls=":")
        lo = min(lo, float(np.nanmin(tb)))
        a.set(xlabel="Hz", title=label, xlim=(50, 8000))
        a.grid(alpha=0.25, which="both")
    ax[0, 0].set(ylabel="PSD at 1 m", ylim=(lo * 0.05, None))
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Synthetic recovery (band-averaged) — solid = true per-rotor PSD, dotted = fitted", y=1.03
    )
    return _save(fig, path)


def plot_real_fit(freqs, att, edges, path, title="") -> str:
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    from .csd import band_average

    ctr = np.sqrt(edges[:-1] * edges[1:])
    for r in range(att.p_rotor.shape[1]):
        pb = band_average(freqs, att.p_rotor[:, r : r + 1], edges)[:, 0]
        ax[0].loglog(ctr, np.maximum(pb, 1e-30), color=_ROTOR_C[r], lw=1.5, label=f"rotor {r}")
    dm = band_average(freqs, att.d_mic, edges)
    ax[0].loglog(ctr, dm.mean(axis=1), color="k", lw=1.5, ls="--", label="per-mic diag (mean)")
    hi = float(np.nanmax(dm.mean(axis=1)))
    ax[0].set(xlabel="Hz", ylabel="PSD", title="fitted components", ylim=(hi * 1e-6, hi * 3))
    ax[0].legend(fontsize=7)

    sh = band_average(freqs, att.shares, edges)
    bottom = np.zeros(len(ctr))
    for r in range(att.p_rotor.shape[1]):
        ax[1].bar(ctr, sh[:, r], bottom=bottom, width=ctr * 0.5, color=_ROTOR_C[r], label=f"r{r}")
        bottom = bottom + sh[:, r]
    ax[1].bar(ctr, sh[:, -1], bottom=bottom, width=ctr * 0.5, color="0.75", label="diagonal")
    ax[1].set(xscale="log", xlabel="Hz", ylabel="share of residual power", title="energy shares")
    ax[1].legend(fontsize=7, ncol=2)

    oe = band_average(freqs, att.off_explained[:, None], edges)[:, 0]
    ax[2].semilogx(ctr, oe, "k-o", ms=3)
    ax[2].set(
        xlabel="Hz",
        ylabel="fraction",
        title="off-diagonal energy explained\nby the 4-rotor model",
        ylim=(0, 1),
    )
    for a in ax:
        a.grid(alpha=0.25, which="both")
    if title:
        fig.suptitle(title, y=1.04)
    return _save(fig, path)


def plot_controls(real, synth_rows, path) -> str:
    """``real``/``synth_rows``: ``{band_label: {control: score}}``."""
    keys = ["true", "rot45", "mirror_z", "random", "centroid"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for a, rows, name in (
        (ax[0], synth_rows, "SYNTHETIC (ideal free field)"),
        (ax[1], real, "REAL"),
    ):
        bands = list(rows)
        w = 0.16
        xs = np.arange(len(bands))
        for i, k in enumerate(keys):
            a.bar(
                xs + (i - 2) * w,
                [rows[b].get(k, np.nan) for b in bands],
                width=w,
                label=k,
                color=_CONTROL_C[k],
            )
            if k == "random":
                a.errorbar(
                    xs + (i - 2) * w,
                    [rows[b].get(k, np.nan) for b in bands],
                    yerr=[2 * rows[b].get("random_std", 0.0) for b in bands],
                    fmt="none",
                    ecolor="k",
                    lw=1,
                    capsize=2,
                )
        a.set_xticks(xs, bands, rotation=20, fontsize=8)
        a.set(title=name, ylim=(0, 1))
        a.grid(alpha=0.25, axis="y")
    ax[0].set_ylabel("off-diagonal energy explained")
    ax[0].legend(fontsize=7, ncol=3)
    fig.suptitle(
        "Null controls: the true rotor geometry must beat wrong geometries of the same dimension",
        y=1.05,
    )
    return _save(fig, path)


def plot_coherence(freqs, meas, model, path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogx(freqs, meas, "k-", lw=1.2, label="measured residual")
    ax.semilogx(freqs, model, "r--", lw=1.2, label="4-rotor + diagonal fit")
    ax.set(
        xlabel="Hz",
        ylabel=r"mean MSC over mic pairs",
        title="inter-microphone coherence of the broadband residual",
        ylim=(0, None),
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    return _save(fig, path)
