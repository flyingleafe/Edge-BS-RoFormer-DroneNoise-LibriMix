#!/usr/bin/env python3
"""F7 — the regime ladder: acoustic rate against its reference, per regime."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"

LAD = json.loads((OUT / "ladder.json").read_text())
BEN = json.loads((OUT / "bench_rate.json").read_text())


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.28)

    g = LAD["groups"]["translate_cruise"]
    u = g["units"]
    r = np.array([x["r"] for x in u])
    a = np.array([x["a"] for x in u])

    # (a) acoustic vs telemetry, free flight, measured tachometer
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(r, a, s=16, color="#1f4e79", alpha=0.8, label="bar-clearing units")
    xs = np.array([r.min() - 2, r.max() + 2])
    ax.plot(xs, xs, "k--", lw=1.0, label="acoustic = telemetry")
    ax.plot(xs, g["slope_prop"] * xs, color="crimson", lw=1.4, label=f"fit a={g['slope_prop']:.5f}")
    ax.set_xlabel("telemetry rate (motors_measured), rev/s")
    ax.set_ylabel("acoustic shaft rate, rev/s")
    ax.set_title(
        f"(a) DREGON free flight, cruise\n{g['n_units']} units / "
        f"{g['n_windows']} windows, R2={g['r2']:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")

    # (b) residual vs rate, with the two candidate mechanisms
    ax = fig.add_subplot(gs[0, 1])
    d = a - r
    ax.scatter(r, d, s=16, color="#1f4e79", alpha=0.8)
    xr = np.linspace(r.min() - 2, r.max() + 2, 200)
    ax.plot(
        xr,
        (g["slope_prop"] - 1) * xr,
        color="crimson",
        lw=1.4,
        label=f"multiplicative  {(g['slope_prop'] - 1) * 100:+.3f} %",
    )
    ax.plot(
        xr,
        -g["quad_c"] * xr**2,
        color="darkorange",
        lw=1.4,
        ls="--",
        label=f"fixed tick miscount  c={g['quad_c']:.2e}",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-0.2, color="0.6", lw=0.8, ls=":", label="0.2 rev/s jitter floor")
    ax.set_xlabel("telemetry rate, rev/s")
    ax.set_ylabel("acoustic - telemetry, rev/s")
    ax.set_title(
        "(b) the offset is proportional to rate\n"
        f"RSS prop {g['rss_prop']:.2f} vs quadratic {g['rss_quad']:.2f} "
        "(indistinguishable over 56-87)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")

    # (c) per-regime scale error with CIs
    ax = fig.add_subplot(gs[0, 2])
    order = [
        ("free flight / measured", "translate_cruise"),
        ("free flight / command", "room1_command_cruise"),
        ("all DREGON cruise", "all_dregon_cruise"),
    ]
    ys, labs = [], []
    for i, (lab, key) in enumerate(order):
        gg = LAD["groups"][key]
        if "scale_error_pct" not in gg:
            continue
        lo, hi = gg["scale_error_pct_ci"]
        ax.errorbar(
            gg["scale_error_pct"],
            i,
            xerr=[[gg["scale_error_pct"] - lo], [hi - gg["scale_error_pct"]]],
            fmt="o",
            color="#1f4e79",
            capsize=4,
        )
        ys.append(i)
        labs.append("")
        ax.text(
            -1.16,
            i - 0.30,
            f"{lab}   (n={gg['n_units']}, {gg['n_windows']} win)",
            fontsize=8,
            va="center",
            color="#1f4e79",
        )
    # regimes with too few bar-clearing units, shown as what they are
    for lab, key in (("hover (room2)", "hover_cruise"), ("maneuver (room2)", "maneuver_cruise")):
        gg = LAD["groups"][key]
        i = len(ys)
        ax.text(
            -1.16, i - 0.30, f"{lab}, command telemetry only", fontsize=8, va="center", color="0.35"
        )
        ax.text(
            -1.16,
            i + 0.06,
            f"    only {gg['n_units']} bar-clearing units - no fit",
            fontsize=8,
            va="center",
            color="0.35",
        )
        ys.append(i)
        labs.append("")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(len(ys) - 0.45, -0.75)
    ax.set_xlabel("acoustic / telemetry - 1  [%]   (95 % block bootstrap)")
    ax.set_title("(c) the regime ladder", fontsize=10)

    # (d) static bench: acoustic vs the file-name nominal
    ax = fig.add_subplot(gs[1, 0])
    rows = [
        r_
        for r_ in BEN["bench"]["rows"]
        if r_["motor"].startswith("Motor") and r_["f0_std_rev_s"] < 0.5
    ]
    dropped = [r_["file"] for r_ in BEN["bench"]["rows"] if r_["f0_std_rev_s"] >= 0.5]
    cols = {"Motor1": "#1f4e79", "Motor2": "#c0392b", "Motor3": "#27ae60", "Motor4": "#8e44ad"}
    for mot in sorted(cols):
        sel = sorted((x for x in rows if x["motor"] == mot), key=lambda x: x["nominal"])
        ax.plot(
            [x["nominal"] for x in sel],
            [x["f0_median_rev_s"] for x in sel],
            "o-",
            ms=4,
            color=cols[mot],
            label=mot,
            lw=1.0,
        )
    ax.plot([45, 95], [45, 95], "k--", lw=1.0, label="acoustic = nominal")
    ax.set_xlabel("file-name nominal (a COMMAND; no tachometer on the bench)")
    ax.set_ylabel("acoustic shaft rate, rev/s")
    ax.set_title(
        f"(d) static bench, single motors\n({len(dropped)} unstable file(s) dropped: {', '.join(dropped)})",
        fontsize=9,
    )
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    for mot in sorted(cols):
        sel = sorted((x for x in rows if x["motor"] == mot), key=lambda x: x["nominal"])
        ax.plot(
            [x["nominal"] for x in sel],
            [100 * (x["acoustic_over_nominal"] - 1) for x in sel],
            "o-",
            ms=4,
            color=cols[mot],
            label=mot,
            lw=1.0,
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-0.542, color="crimson", ls="--", lw=1.2, label="in-flight -0.542 %")
    ax.set_xlabel("nominal")
    ax.set_ylabel("acoustic / nominal - 1  [%]")
    ax.set_title(
        "(e) bench: per-motor spread is 2.8 pp — the nominal is an\n"
        "open-loop command, not a measurement",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    # (f) the telemetry channel itself
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    txt = (
        "DREGON motors.mat\n\n"
        "measured : v = 1 / (n x 42.0 us), n integer\n"
        "           reciprocal-period tachometer\n"
        "           updates at 49.7 Hz, logged at 1002 Hz\n"
        "           lattice fit residual 0.0006 of a step\n\n"
        "command  : continuous float, 94431 distinct values\n"
        "           measured / command = 1.0000 to 1.0009\n"
        "           per rotor (5 room1 recordings)\n\n"
        "audio clock : 44100.0 Hz, verified to +-0.025 %\n"
        "           against the emitted white-noise file\n"
        "           (3 recordings, 52-70 s spans)\n\n"
        "=> the reference is a MEASURED tachometer,\n"
        "   the audio time base is correct, and the\n"
        "   offset is a clean multiplicative -0.54 %."
    )
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle("F7 — DREGON acoustic shaft rate against its telemetry, by regime", fontsize=13)
    fig.savefig(FIGS / "F7_regime_ladder.png", dpi=130, bbox_inches="tight")
    print(f"wrote {FIGS / 'F7_regime_ladder.png'}")


if __name__ == "__main__":
    main()
