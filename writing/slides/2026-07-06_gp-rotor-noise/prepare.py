#!/usr/bin/env python3
"""Copy figures from the report assets into the slide assets directory."""

import pathlib
import shutil

SLIDE_DIR = pathlib.Path(__file__).resolve().parent
ASSETS = SLIDE_DIR / "assets"
ASSETS.mkdir(exist_ok=True)
REPORT_ASSETS = SLIDE_DIR.parents[1] / "reports/2026-07-06_gp-rotor-noise/assets"

NAMES = [
    "noise_gen_diagram.png",
    "noise_gen_spec_dregon.png",
    "noise_gen_spec_michaels.png",
    "e3_smoothness_sweep.png",
    "e4_aug_degradation.png",
    "gp_qd2026_tiers.png",
    "gp_overview.png",
    "gp_v1_v2_rmse.png",
    "gp_faithful_spectrum.png",
]


def main():
    for name in NAMES:
        src = REPORT_ASSETS / name
        if src.exists():
            shutil.copy2(src, ASSETS / name)
        else:
            print(f"[warn] missing {src}")
    print("[prepare] copied", len(NAMES), "figures to", ASSETS)


if __name__ == "__main__":
    main()
