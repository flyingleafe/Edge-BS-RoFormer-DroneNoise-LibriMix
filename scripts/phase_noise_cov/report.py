#!/usr/bin/env python3
"""WP18 — turn ``summary.json`` into the markdown tables the doc quotes.

    python scripts/phase_noise_cov/report.py [--results results/phase_noise_covariance]

Five tables, in the order the question was asked:

1. **coverage** — how many of the 30 harmonics each band schedule actually
   leaves usable after the twin-collision gate.  On a real quadrotor this is
   not bookkeeping: it decides whether a K x K covariance can be fitted at all.
2. **the common term** — ``sigma_c^2`` against the high-pass cutoff, with its
   block-bootstrap standard error, so "is there an irreducible floor?" is
   answered with a significance rather than a number.
3. **the weight curve** — ``v_k`` and the fitted exponent ``alpha`` in
   ``1/v_k ∝ k^alpha P_k``, against the ``k^2`` the stage assumes.
4. **saturation** — ``k*`` and the share of the fused variance the floor
   accounts for over the stage's own ``k`` range.
5. **fit quality** — is rank-one-plus-diagonal the right shape, or are the
   off-diagonals structured?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _f(x: Any, d: int = 4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    if v != 0 and (abs(v) < 10**-d or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{d}f}"


def _table(rows: list[list[str]], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_noise_covariance")
    ap.add_argument("--arms", default="fixB1.5,fixB3,fixB6,fixB12,kscale0.1,kscale0.25,kscale0.5")
    ap.add_argument("--groups", default="")
    args = ap.parse_args()
    s = json.loads((Path(args.results) / "summary.json").read_text())
    groups = [g for g in s["groups"] if not args.groups or g in args.groups.split(",")]
    arms = args.arms.split(",")

    print("### Coverage — usable harmonics of 30, after the twin gate\n")
    rows = []
    for g in groups:
        gd = s["groups"][g]
        r = [g, str(gd["n_units"])]
        for a in arms:
            e = gd["arms"].get(a)
            r.append(
                f"{_f(e.get('n_keep_all'), 1)} (fit {e['n'] - e['n_units_failed'] if False else e['n']}/{e['n_units_attempted']})"
                if e
                else "—"
            )
        rows.append(r)
    print(_table(rows, ["group", "units", *arms]))

    print("\n### The common term — sigma_c^2 (rev^2/s^2) vs high-pass cutoff\n")
    for a in arms:
        any_row = False
        rows = []
        for g in groups:
            e = s["groups"][g]["arms"].get(a)
            if not e:
                continue
            cuts = e.get("cutoffs", {})
            if not cuts:
                continue
            any_row = True
            r = [g]
            for tag in ("0", "0.5", "1", "2", "3", "4", "6", "8"):
                c = cuts.get(tag)
                r.append(f"{_f(c['sigma_c2_mean'], 6)}±{_f(c['sigma_c2_se'], 6)}" if c else "—")
            r.append(_f(cuts.get("0", {}).get("sigma_c2_signif"), 2))
            rows.append(r)
        if any_row:
            print(f"\n**{a}**\n")
            print(
                _table(
                    rows,
                    [
                        "group",
                        *[f"fc={t}" for t in ("0", "0.5", "1", "2", "3", "4", "6", "8")],
                        "z(fc0)",
                    ],
                )
            )

    print("\n### The weight curve — 1/v_k ∝ k^alpha * P_k (the stage assumes alpha = 2)\n")
    rows = []
    for g in groups:
        gd = s["groups"][g]
        for a in arms:
            e = gd["arms"].get(a)
            if not e or "alpha_signal" not in e:
                continue
            rows.append(
                [
                    g,
                    a,
                    _f(e["alpha_signal"]["median"], 2),
                    _f(e["alpha_signal"].get("iqr"), 2),
                    _f(e["alpha_signal"].get("r2_median"), 2),
                    _f(e.get("alpha_raw", {}).get("median"), 2),
                    _f(e.get("alpha_snr", {}).get("median"), 2),
                    str(int(e["n"])),
                ]
            )
    print(_table(rows, ["group", "arm", "alpha", "IQR", "R²", "alpha_raw", "alpha_snr", "n"]))

    print("\n### Saturation — k* and the floor's share of the fused variance (k >= 6)\n")
    rows = []
    for g in groups:
        gd = s["groups"][g]
        for a in arms:
            e = gd["arms"].get(a)
            if not e:
                continue
            for tag in ("0", "2", "4"):
                c = e.get("cutoffs", {}).get(tag)
                if not c:
                    continue
                rows.append(
                    [
                        g,
                        a,
                        tag,
                        _f(c.get("k_star"), 1),
                        f"{_f(1 - (c.get('k_star_none_frac') or 0), 2)}",
                        _f(c.get("floor_share"), 3),
                        _f(c.get("inv_W_stage"), 6),
                    ]
                )
    print(_table(rows, ["group", "arm", "fc", "k*", "frac resolved", "floor share", "1/W(k>=6)"]))

    print("\n### Fit quality — is rank-one-plus-diagonal the right shape?\n")
    rows = []
    for g in groups:
        gd = s["groups"][g]
        for a in arms:
            e = gd["arms"].get(a)
            c = (e or {}).get("cutoffs", {}).get("0")
            if not c:
                continue
            rows.append(
                [
                    g,
                    a,
                    _f(c.get("offdiag_chi2"), 2),
                    _f(c.get("offdiag_excess_rel"), 2),
                    _f(c.get("rank1_energy_frac"), 2),
                    _f(c.get("loading_beta"), 2),
                    _f(c.get("loading_snr_corr"), 2),
                    _f(c.get("offdiag_corr_min"), 2),
                    _f(c.get("offdiag_corr_absdiff"), 2),
                    f"{c.get('n_significant')}/{c.get('n')}",
                ]
            )
    print(
        _table(
            rows,
            [
                "group",
                "arm",
                "chi²",
                "excess/σ",
                "rank1 frac",
                "beta",
                "corr(a,SNR)",
                "corr(C,min k)",
                "corr(C,|Δk|)",
                "resolved",
            ],
        )
    )

    print("\n### Channel coherence of the common term (1 = one shaft, 0 = per-mic)\n")
    rows = [
        [g, _f(s["groups"][g].get("chan_coherence"), 3), str(s["groups"][g]["n_units"])]
        for g in groups
    ]
    print(_table(rows, ["group", "mean pairwise corr", "units"]))

    ctl = {
        g: s["groups"][g]["arms"].get("fixB6", {}).get("control")
        for g in groups
        if g.startswith("synth")
    }
    if any(ctl.values()):
        print("\n### Synthetic control — measured vs injected (fixB6)\n")
        rows = []
        for g, c in ctl.items():
            if not c:
                continue
            for tag, d in c.items():
                rows.append([g, tag, _f(d["measured"], 6), _f(d["predicted"], 6)])
        print(_table(rows, ["group", "fc", "measured", "injected"]))

    if "indoor_outdoor" in s:
        print("\n### Indoor (DREGON) vs outdoor (Michael's)\n")
        print("```json")
        print(json.dumps(s["indoor_outdoor"], indent=2))
        print("```")


if __name__ == "__main__":
    main()
