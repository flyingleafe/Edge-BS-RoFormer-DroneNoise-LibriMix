#!/usr/bin/env python3
"""Generate figures and tables for the RPS Prediction Study slides.

This script copies the previously-generated figures from the old slide location.
"""

import pathlib
import shutil


def main():
    assets = pathlib.Path("assets")
    assets.mkdir(exist_ok=True)

    # Old figures location (relative to this slide directory)
    old_figures = pathlib.Path("../../../slides/2026-04-14/assets/rps_comparison")

    # Figures needed by this slide deck
    needed = [
        "summary_metrics.png",
        "sample_00000_plot.png",
        "sample_00149_plot.png",
        "sample_00449_plot.png",
        "sample_00599_plot.png",
    ]

    for fig in needed:
        src = old_figures / fig
        dst = assets / fig
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fig}")
        else:
            print(f"WARNING: {src} not found")

    print("Done.")


if __name__ == "__main__":
    main()
