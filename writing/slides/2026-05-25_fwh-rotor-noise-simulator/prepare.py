#!/usr/bin/env python3
"""Copy figures for the FWH Rotor Noise Simulator slides."""

import pathlib
import shutil


def main():
    assets = pathlib.Path("assets")
    assets.mkdir(exist_ok=True)

    old = pathlib.Path("../../../slides/2026-05-25/assets")
    for f in old.iterdir():
        if f.is_file():
            shutil.copy2(f, assets / f.name)
            print(f"Copied {f.name}")

    print("Done.")


if __name__ == "__main__":
    main()
