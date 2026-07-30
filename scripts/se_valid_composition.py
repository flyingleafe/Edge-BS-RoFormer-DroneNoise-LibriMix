#!/usr/bin/env python3
"""Tabulate the composition of the fixed SE valid sets (drone + harmonic).

Counts are read live from the published sets (``SEValidFrameDataset``); the
source-dataset mapping per category is the one baked into
``data_processing.derivations (se_valid generator)`` (``CATEGORY_NOISE``), reproduced here as
documentation. Writes a tidy CSV (one row per category) and prints a Markdown
table.

    python scripts/se_valid_composition.py --out results/f1_tables/valid_composition.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from data_processing.frame_datasets import SEValidFrameDataset
from data_processing.frames import get_meta

SR = 16000
CLIP_SECONDS = 2.0
# category -> (human noise family, source dload datasets) — mirrors
# derivations.SE_CATEGORY_NOISE. Held-out = last 2 whole shards of each dataset.
CATEGORY_SOURCES: dict[str, tuple[str, str]] = {
    "drone": (
        "rotating UAV noise",
        "DREGON-frames, michaels-frames, drone_audio, DroneAudioSet",
    ),
    "mimii": ("industrial machines (fan/pump/valve/slider)", "MIMII"),
    "mimii_dg": ("industrial machines, domain-shift", "MIMII-DG"),
    "aircraft": ("propeller-aircraft flyover", "AeroSonicDB"),
    "motors": ("rotating electric motors", "HUSTmotor, KAIST-rotating-acoustic"),
    "horns": ("tonal horns (non-rotating)", "HornBase"),
}
VALIDS = {
    "SE-valid-drone": ["drone"],
    "SE-valid-harmonic": ["drone", "mimii", "mimii_dg", "aircraft", "motors", "horns"],
}


def _counts(name: str) -> dict[str, Counter]:
    ds = SEValidFrameDataset(name, sample_rate=SR)
    per_cat: dict[str, Counter] = {}
    for i in range(len(ds)):
        cat = str(get_meta(ds[i], "category", "all"))
        snr = float(get_meta(ds[i], "input_snr"))
        per_cat.setdefault(cat, Counter())[snr] += 1
    return per_cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/f1_tables/valid_composition.csv")
    args = ap.parse_args()

    rows = []
    for valid, cats in VALIDS.items():
        counts = _counts(valid)
        for cat in cats:
            c = counts.get(cat, Counter())
            snrs = sorted(c)
            per_snr = c[snrs[0]] if snrs else 0
            family, sources = CATEGORY_SOURCES.get(cat, ("?", "?"))
            rows.append(
                {
                    "valid": valid,
                    "category": cat,
                    "family": family,
                    "sources": sources,
                    "n_snr": len(snrs),
                    "clips_per_snr": per_snr,
                    "total_clips": sum(c.values()),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)\n")

    # Markdown preview
    print(
        f"Clip format: {CLIP_SECONDS:.0f}s, {SR // 1000} kHz mono. SNR grid: "
        "-30,-25,-20,-15,-10,-5,0 dB.\n"
    )
    for valid in VALIDS:
        vr = [r for r in rows if r["valid"] == valid]
        tot = sum(r["total_clips"] for r in vr)
        print(f"### {valid} — {tot} clips")
        print("| category | family | source datasets | clips/SNR | total |")
        print("|---|---|---|---|---|")
        for r in vr:
            print(
                f"| {r['category']} | {r['family']} | {r['sources']} | "
                f"{r['clips_per_snr']} | {r['total_clips']} |"
            )
        print()


if __name__ == "__main__":
    main()
