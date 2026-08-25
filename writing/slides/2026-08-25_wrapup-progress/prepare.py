#!/usr/bin/env python3
"""Assets for the wrap-up progress deck.

Copies the three qualitative panels already rendered for the wrap-up paper
(writing/papers/2026-08_wrapup/figures/qual_*.png) into assets/. No new
figures are generated — every number on these slides comes straight from
docs/experiments/unified-baseline-eval.md and the paper source, quoted in
Typst tables/text.
"""

import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets"
PAPER_FIGS = ROOT / "writing" / "papers" / "2026-08_wrapup" / "figures"


def main():
    ASSETS.mkdir(exist_ok=True)
    for name in ["qual_zero.png", "qual_transition.png", "qual_cruise.png"]:
        src = PAPER_FIGS / name
        if src.exists():
            shutil.copy(src, ASSETS / name)
        else:
            print(f"WARNING: missing {src}")


if __name__ == "__main__":
    main()
