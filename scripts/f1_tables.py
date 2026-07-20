#!/usr/bin/env python3
"""Assemble the F1 SE-baseline tables from the anchor + model eval CSVs.

Reads ``results/se_anchors/*.csv`` (noisy/Wiener) and
``results/f1_eval/f1_<arch>_<pass>__<valid>.csv`` (models), and writes markdown
tables under ``results/f1_tables/``:
  1. per-SNR floor table (one per valid × metric): rows = method, cols = SNR;
  2. diversity delta = Pass B − Pass A on SE-valid-drone (per arch, per SNR);
  3. per-category transfer on SE-valid-harmonic (per model × category, mean SI-SDR).
Noisy + Wiener anchors head every floor table (the plan's requirement).
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

ANCHOR_DIR = Path("results/se_anchors")
MODEL_DIR = Path("results/f1_eval")
OUT = Path("results/f1_tables")
SNRS = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0]
METHOD_ORDER = ["noisy", "wiener"]  # anchors first; models appended in discovery order


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _load(valid: str) -> dict[tuple[str, str, float], dict[str, float]]:
    """(method, category, snr) -> {metric: value}. Anchors + all model CSVs for `valid`."""
    rows: dict[tuple[str, str, float], dict[str, float]] = {}
    anchor = ANCHOR_DIR / f"{valid}.csv"
    srcs = [str(anchor)] if anchor.is_file() else []
    srcs += sorted(glob.glob(str(MODEL_DIR / f"*__{valid}.csv")))
    for src in srcs:
        for r in _read(src):
            key = (r["method"], r.get("category", "all"), float(r["input_snr"]))
            rows[key] = {
                m: (float(r[m]) if r[m] not in ("", "nan") else float("nan"))
                for m in ("si_sdr", "sdr", "pesq", "estoi")
            }
    return rows


def _methods(rows) -> list[str]:
    seen = [m for m in METHOD_ORDER if any(k[0] == m for k in rows)]
    for k in rows:
        if k[0] not in seen:
            seen.append(k[0])
    return seen


def floor_table(valid: str, metric: str, rows) -> str:
    methods = _methods(rows)
    out = [
        f"### {valid} — {metric} (per input SNR, dB)\n",
        "| method | " + " | ".join(f"{int(s)}" for s in SNRS) + " |",
        "|" + "---|" * (len(SNRS) + 1),
    ]
    for m in methods:
        cells = []
        for s in SNRS:
            v = rows.get((m, "all", s), {}).get(metric)
            cells.append(f"{v:.2f}" if isinstance(v, float) and v == v else "—")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def diversity_table(rows, metric="si_sdr") -> str:
    archs = sorted(
        {
            m[len("f1_") :].rsplit("_", 1)[0]
            for (mm, _c, _s) in rows
            for m in [mm]
            if mm.startswith("f1_")
        }
    )
    out = [
        f"### Diversity delta on SE-valid-drone: Pass B − Pass A ({metric}, dB)\n",
        "| arch | " + " | ".join(f"{int(s)}" for s in SNRS) + " |",
        "|" + "---|" * (len(SNRS) + 1),
    ]
    for a in archs:
        cells = []
        for s in SNRS:
            va = rows.get((f"f1_{a}_a", "all", s), {}).get(metric)
            vb = rows.get((f"f1_{a}_b", "all", s), {}).get(metric)
            cells.append(f"{vb - va:+.2f}" if (va == va and vb == vb) else "—")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def transfer_table(rows, metric="si_sdr") -> str:
    cats = sorted({k[1] for k in rows if k[1] != "all"})
    methods = _methods(rows)
    out = [
        f"### SE-valid-harmonic per-category transfer (mean {metric} over SNR, dB)\n",
        "| method | " + " | ".join(cats) + " |",
        "|" + "---|" * (len(cats) + 1),
    ]
    for m in methods:
        cells = []
        for c in cats:
            vals = [
                rows[(m, c, s)][metric]
                for s in SNRS
                if (m, c, s) in rows and rows[(m, c, s)][metric] == rows[(m, c, s)][metric]
            ]
            cells.append(f"{sum(vals) / len(vals):.2f}" if vals else "—")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    drone = _load("SE-valid-drone")
    harm = _load("SE-valid-harmonic")
    parts = ["# F1 SE blind-baseline tables\n"]
    for metric in ("si_sdr", "pesq", "estoi"):
        if drone:
            parts.append(floor_table("SE-valid-drone", metric, drone))
    if drone:
        parts.append(diversity_table(drone))
    if harm:
        parts.append(floor_table("SE-valid-harmonic", "si_sdr", harm))
        parts.append(transfer_table(harm))
    (OUT / "f1_tables.md").write_text("\n".join(parts))
    print(f"wrote {OUT / 'f1_tables.md'} (drone rows={len(drone)}, harmonic rows={len(harm)})")


if __name__ == "__main__":
    main()
