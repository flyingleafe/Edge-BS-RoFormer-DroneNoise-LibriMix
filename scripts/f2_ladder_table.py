#!/usr/bin/env python3
"""Build the F2 ladder's per-SNR comparison tables from the per-clip CSVs.

    python scripts/f2_ladder_table.py [--dir results/f2_perclip] [--csv OUT.csv]

Two things the raw CSVs do not say on their own:

* **Whether an arm collapsed.** `gain_db` (output/target energy ratio) and
  `corr` (|<est,ref>|/||est|| ||ref||) are printed next to SI-SDR precisely
  because a low SI-SDR alone cannot distinguish a degenerate near-null output
  from an honest-but-distorted one. A collapsed arm shows corr -> 0 with `sdr`
  pinned near 0 dB regardless of input SNR.
* **Whether it beats doing nothing.** Every metric is shown as a delta against
  the `noisy` anchor on the SAME valid set, because "SI-SDR improves with SNR"
  is trivially true and the only interesting question is the margin over the
  unprocessed mixture.

Clips whose target is digitally silent are dropped: the F1 valid sets were built
before the silent-draw guard landed in `src/data_processing/online_mixing.py`,
and an all-zero reference sends SI-SDR to the -80 dB floor, dragging a whole
SNR group's mean by several dB (see docs/experiments/f1-se-blind-baselines.md).
They are reported, not silently discarded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["si_sdr", "sdr", "pesq", "estoi", "gain_db", "corr"]
# Mukhutdinov et al. 2023 (IEEE Access), DCUNet at -15 dB input SNR. Measured at
# 8 kHz on TIMIT and on a different drone (AS), so a reference point, not a bar.
PAPER_AT_MINUS15 = {"si_sdr": 3.7, "estoi": 0.4, "pesq": 1.9}


def load(directory: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(directory.glob("*.csv")):
        d = pd.read_csv(f)
        if "method" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        raise SystemExit(f"no per-clip CSVs in {directory}")
    return pd.concat(frames, ignore_index=True)


def drop_silent(df: pd.DataFrame) -> pd.DataFrame:
    """Drop clips that are all-zero for EVERY method (a silent reference)."""
    if "si_sdr" not in df.columns:
        return df
    # A silent reference pins si_sdr at the eps floor for EVERY method at once,
    # so a clip is silent iff no method manages to exceed the floor on it.
    best: dict[tuple[str, str], float] = {}
    for valid, clip, value in zip(
        df["valid"].astype(str), df["clip_id"].astype(str), df["si_sdr"].astype(float), strict=True
    ):
        key = (valid, clip)
        if value > best.get(key, float("-inf")):
            best[key] = value
    bad = {k for k, v in best.items() if v < -70.0}
    if not bad:
        return df
    print(f"dropping {len(bad)} silent-reference clip(s): {sorted(c for _, c in bad)[:8]}")
    keep = [
        (v, c) not in bad
        for v, c in zip(df["valid"].astype(str), df["clip_id"].astype(str), strict=True)
    ]
    return pd.DataFrame(df[keep])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/f2_perclip")
    ap.add_argument("--csv", default=None, help="also write the tidy table here")
    args = ap.parse_args()

    df = drop_silent(load(Path(args.dir)))
    have = [m for m in METRICS if m in df.columns]
    grouped = df.groupby(["valid", "method", "input_snr"])[have].mean().reset_index()

    pd.set_option("display.width", 220)
    for valid, g in grouped.groupby("valid"):
        print(f"\n{'=' * 100}\n{valid}\n{'=' * 100}")
        anchor = g[g["method"] == "noisy"].set_index("input_snr")
        for method, gm in g.groupby("method"):
            gm = gm.set_index("input_snr").sort_index()
            print(f"\n-- {method}")
            show = gm[have].round(3)
            if method != "noisy" and not anchor.empty:
                for m in ("si_sdr", "estoi", "pesq"):
                    if m in show.columns and m in anchor.columns:
                        show[f"d_{m}"] = (gm[m] - anchor[m]).round(3)
            print(show.to_string())
            if valid == "SE-valid-avq-survey" and -15.0 in gm.index and method != "noisy":
                row = gm.loc[-15.0]
                deltas = ", ".join(
                    f"{m} {row[m]:.3f} vs paper {v}"
                    for m, v in PAPER_AT_MINUS15.items()
                    if m in row
                )
                print(f"   @-15 dB vs the 2023 survey's DCUNet: {deltas}")

    if args.csv:
        grouped.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
