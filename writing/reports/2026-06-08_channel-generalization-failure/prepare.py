#!/usr/bin/env python3
"""Copy figures from the old report into this report's assets directory."""

import pathlib
import shutil

ASSETS = pathlib.Path("assets")
SOURCE = pathlib.Path("../../papers/channel-generalization/figures")

FIGURES = [
    "mic_array.png",
    "mse_bars.png",
    "mse_bars_pit.png",
    "mse_bars_8ch_v4.png",
    "mse_bars_8ch_v4_pit.png",
    "sample_nosource_simpleconv.png",
    "sample_nosource_simpleconv_v2.png",
    "sample_speech_simpleconv.png",
    "sample_speech_simpleconv_v2.png",
    "sample_nosource_8ch_v4_simpleconv.png",
    "sample_nosource_8ch_v4_simpleconv_v2.png",
    "sample_speech_8ch_v4_simpleconv.png",
    "sample_speech_8ch_v4_simpleconv_v2.png",
    "sample_nosource_varied_8ch_v4_simpleconv.png",
    "sample_nosource_varied_8ch_v4_simpleconv_v2.png",
]


def main():
    ASSETS.mkdir(exist_ok=True)
    for fig in FIGURES:
        src = SOURCE / fig
        dst = ASSETS / fig
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fig}")
        else:
            print(f"Missing source figure: {src}")


if __name__ == "__main__":
    main()
