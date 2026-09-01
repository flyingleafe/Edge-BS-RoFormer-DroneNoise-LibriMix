#!/usr/bin/env python3
"""Assets for the 2026-08-31 supervisor update.

Four of the five figure assets are generated here, by the three sibling
scripts in this directory:

* ``make_assets_spectra.py`` -> ``assets/families_row.pdf`` (slide 2) and
  ``assets/stoch_samples.pdf`` (slide 3)
* ``make_assets_fan.py``     -> ``assets/fan_panels.pdf`` (slide 6)
* ``make_assets_slots.py``   -> ``assets/slots_vs_regressor.pdf`` (slide 10)

``assets/freq_probe_nophase.pdf`` (slide 1) is the wrap-up paper's Fig. 3
re-rendered without the phase-increment curve; ``make_assets_freqprobe.py``
reproduces it from the paper's own response JSON, and refuses to overwrite the
file that is already there unless called with ``--force``.

The slide-7 and slide-8 schematics are drawn in Typst, not here.

Every asset script is slow (it loads checkpoints and streams the frozen
validation split), so an asset that already exists is left alone. Delete it
to force a rebuild, or run the script directly.
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"

JOBS = [
    ("make_assets_freqprobe.py", ["freq_probe_nophase.pdf"]),
    ("make_assets_spectra.py", ["families_row.pdf", "stoch_samples.pdf"]),
    ("make_assets_fan.py", ["fan_panels.pdf"]),
    ("make_assets_slots.py", ["slots_vs_regressor.pdf"]),
]


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for script, outputs in JOBS:
        path = HERE / script
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        if all((ASSETS / o).exists() for o in outputs):
            print(f"up to date: {', '.join(outputs)}")
            continue
        print(f"running {script} ...")
        subprocess.run([sys.executable, str(path)], check=True, cwd=HERE)
    for name in [o for _, outs in JOBS for o in outs]:
        if not (ASSETS / name).exists():
            print(f"WARNING: asset still missing: assets/{name}")


if __name__ == "__main__":
    main()
