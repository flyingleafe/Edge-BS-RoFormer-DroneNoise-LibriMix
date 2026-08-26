"""The synthetic-only transfer leaderboard: real-trained target, then the best
synthetic-only models, one row per MODEL.

Every row is one checkpoint measured on every regime. Ranking is by all-regime
PIT mean absolute error, which is what the campaign's goal is stated in — never
a per-regime best-of across different models, which no single model achieves.

    python scripts/transfer_board.py            # top 3 synthetic-only
    python scripts/transfer_board.py --top 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("results/stoch_transfer")

#: Rows measured before the results directory existed, kept so the board is
#: complete. Each is one run of scripts/valid_regime_eval.py.
KNOWN: list[dict] = [
    {"experiment": "r4hb_scv2", "kind": "real", "aggregate_mse": 17.59, "all_mae": 2.67,
     "zero_mae": 2.87, "low_mae": 3.48, "flight_mae": 2.49},
    {"experiment": "hb_scv2_mag_nogate", "kind": "real", "aggregate_mse": 22.78, "all_mae": 2.72,
     "zero_mae": 3.36, "low_mae": 4.18, "flight_mae": 2.35},
    {"experiment": "m3abl_comb_unigru128_s1", "kind": "synthetic", "aggregate_mse": 190.62,
     "all_mae": 8.30, "zero_mae": 4.73, "low_mae": 24.24, "flight_mae": 6.00},
    {"experiment": "m3abl_comb_scv2_s1", "kind": "synthetic", "aggregate_mse": 218.30,
     "all_mae": 9.50, "zero_mae": 5.64, "low_mae": 26.32, "flight_mae": 7.09},
    {"experiment": "m3cur_scv2_s1", "kind": "synthetic", "aggregate_mse": 328.96,
     "all_mae": 10.41, "zero_mae": 20.30, "low_mae": 11.33, "flight_mae": 8.55},
    {"experiment": "stoch_s1e_scv2", "kind": "synthetic", "aggregate_mse": 280.66,
     "all_mae": 10.74, "zero_mae": 8.92, "low_mae": 27.55, "flight_mae": 7.98},
    {"experiment": "stoch_s1f_scv2", "kind": "synthetic", "aggregate_mse": 343.69,
     "all_mae": 11.35, "zero_mae": 34.69, "low_mae": 16.95, "flight_mae": 6.32},
    {"experiment": "stoch_s1g_scv2", "kind": "synthetic", "aggregate_mse": 176.34,
     "all_mae": 8.08, "zero_mae": 20.27, "low_mae": 16.20, "flight_mae": 4.50},
    {"experiment": "stoch_s1h_scv2", "kind": "synthetic", "aggregate_mse": 300.36,
     "all_mae": 9.07, "zero_mae": 27.98, "low_mae": 26.77, "flight_mae": 2.60},
    {"experiment": "comb_fixed_scv2", "kind": "synthetic", "aggregate_mse": 1389.03,
     "all_mae": 35.09, "zero_mae": 41.21, "low_mae": 10.42, "flight_mae": 38.55},
    {"experiment": "stoch_s1r_long", "kind": "synthetic", "aggregate_mse": 453.41,
     "all_mae": 18.20, "zero_mae": 16.86, "low_mae": 17.51, "flight_mae": 18.55},
    {"experiment": "stoch_s1q_gru", "kind": "synthetic", "aggregate_mse": 331.57,
     "all_mae": 15.18, "zero_mae": 17.09, "low_mae": 19.18, "flight_mae": 14.12},
    {"experiment": "stoch_s1s_both", "kind": "synthetic", "aggregate_mse": 296.42,
     "all_mae": 13.95, "zero_mae": 17.94, "low_mae": 8.94, "flight_mae": 14.19},
]

REAL_NAMES = {"r4hb_scv2", "hb_scv2_mag_nogate"}


def load() -> list[dict]:
    rows = {r["experiment"]: dict(r) for r in KNOWN}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            found = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for row in found:
            if row.get("rescale_rms") is not None or "all_mae" not in row:
                continue
            row = dict(row)
            row["kind"] = "real" if row["experiment"] in REAL_NAMES else "synthetic"
            rows[row["experiment"]] = row
    return list(rows.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    rows = load()
    real = sorted([r for r in rows if r["kind"] == "real"], key=lambda r: r["all_mae"])
    synth = sorted([r for r in rows if r["kind"] == "synthetic"], key=lambda r: r["all_mae"])
    target = real[0] if real else None

    head = f"{'model':26s} {'trained on':11s} {'all-MAE':>8s} {'zero':>7s} {'low':>7s} {'flight':>7s}"
    print(head)
    print("-" * len(head))
    for r in real[:1]:
        print(f"{r['experiment']:26s} {'real':11s} {r['all_mae']:8.2f} {r['zero_mae']:7.2f} "
              f"{r['low_mae']:7.2f} {r['flight_mae']:7.2f}   <- target")
    for r in synth[: args.top]:
        print(f"{r['experiment']:26s} {'synthetic':11s} {r['all_mae']:8.2f} {r['zero_mae']:7.2f} "
              f"{r['low_mae']:7.2f} {r['flight_mae']:7.2f}")
    if target and synth:
        # The best cell any single synthetic model holds, named — so the board
        # never reads as though one model held all three.
        print()
        print("best synthetic-only IN EACH CELL, and which model holds it:")
        for cell, key in (("zero", "zero_mae"), ("low", "low_mae"), ("flight", "flight_mae")):
            best = min(synth, key=lambda r: r[key])
            print(f"   {cell:7s} {best[key]:6.2f}  {best['experiment']:26s} "
                  f"({best[key] / target[key]:.2f}x target)")
        b = synth[0]
        print()
        print("distance from parity, best synthetic-only model (times the target):")
        for cell, key in (("zero", "zero_mae"), ("low", "low_mae"), ("flight", "flight_mae"),
                          ("all", "all_mae")):
            print(f"   {cell:7s} {b[key] / target[key]:5.2f}x   ({b[key]:.2f} against {target[key]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
