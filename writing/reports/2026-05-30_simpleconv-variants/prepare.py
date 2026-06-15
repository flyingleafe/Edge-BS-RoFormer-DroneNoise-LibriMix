#!/usr/bin/env python3
"""Generate figures and tables for the SimpleConv Variants report.

This script copies the previously-generated figures from the old report location.
In a fresh recreation, the figures would be generated from raw evaluation data.
"""

import pathlib
import shutil


def main():
    assets = pathlib.Path("assets")
    assets.mkdir(exist_ok=True)

    # Old figures location (relative to this report directory)
    old_figures = pathlib.Path("../../../writing/papers/simpleconv_variants_report/figures")

    # Figures needed by this report
    needed = [
        "fig_leaderboard_validation.png",
        "fig_pareto_params_r2.png",
        "fig_fullsequence_comparison.png",
        "fig_fullsequence_inflight_mse_bar.png",
        "fig_individual_motor_mse_bar.png",
        "fig_single_rotor_allmotors_comparison.png",
        "fig_allmotors_mse_bar.png",
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
