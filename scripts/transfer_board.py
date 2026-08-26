"""The synthetic-only transfer leaderboard: the fine-tuned target, then the best
synthetic-only models, one row per MODEL.

Every row is one checkpoint measured on every regime. Ranking is by all-regime
PIT mean absolute error, which is what the campaign's goal is stated in — never
a per-regime best-of across different models, which no single model achieves.

The frozen split is TWO aircraft, so every number also splits by rig. The two
halves are not the same measurement:

    DREGON    22 clips  zero 844  low  182  flight 4496   (5522 frames)
    Michael's 15 clips  zero 333  low 1071  flight 2361   (3765 frames)

so the ramp cell is 85% Michael's and the zero cell 72% DREGON, and an
all-regime average is 59% DREGON. It matters more than a split usually does,
because the target does not face the same task on both halves: it trains on
DREGON room2 and is scored on room1 (a room change), but it trains on Michael's
FLY125 and is scored on FLY124 — the same aircraft, the same array, an adjacent
flight. Its Michael's column is therefore close to an in-domain number and its
DREGON column is a transfer number. Compare synthetic-only against each on its
own terms.

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
    {"experiment": "stoch_s1t_ownsilence", "kind": "synthetic", "aggregate_mse": 454.68,
     "all_mae": 16.44, "zero_mae": 37.61, "low_mae": 20.24, "flight_mae": 12.11},
    {"experiment": "stoch_s1u_composite", "kind": "synthetic", "aggregate_mse": 464.24,
     "all_mae": 17.96, "zero_mae": 13.75, "low_mae": 14.24, "flight_mae": 19.36},
    {"experiment": "stoch_s1v_ground", "kind": "synthetic", "aggregate_mse": 430.72,
     "all_mae": 15.07, "zero_mae": 6.57, "low_mae": 13.87, "flight_mae": 16.74},
    {"experiment": "stoch_s1w_scv2", "kind": "synthetic", "aggregate_mse": 296.49,
     "all_mae": 12.63, "zero_mae": 16.19, "low_mae": 22.55, "flight_mae": 10.20},
    {"experiment": "stoch_s1x_scv2", "kind": "synthetic", "aggregate_mse": 262.42,
     "all_mae": 11.59, "zero_mae": 8.68, "low_mae": 11.88, "flight_mae": 12.04},
    # Cross-rig CONTROL, not a target: the target's recipe with Michael's
    # removed, so its Michael's column is what a real-trained model is worth on
    # a rig it never met.
    {"experiment": "xrig_dregon_only", "kind": "control", "aggregate_mse": 220.77,
     "all_mae": 9.72, "zero_mae": 9.23, "low_mae": 28.31, "flight_mae": 6.41},
    {"experiment": "xrig_michaels_only", "kind": "control", "aggregate_mse": 1571.81,
     "all_mae": 24.60, "zero_mae": 3.43, "low_mae": 11.52, "flight_mae": 30.63},
]

#: Per-rig cells, from job regime-rig-f123b7 (8 channels, all 37 clips).
#: Each tuple is (all, zero, low, flight) mean absolute error, rev/s.
RIG_CELLS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "r4hb_scv2": {
        "dregon": (3.0, 2.24, 7.21, 2.98),
        "michaels": (2.18, 4.48, 2.85, 1.55),
    },
    "stoch_s1g_scv2": {
        "dregon": (6.49, 14.64, 22.42, 4.31),
        "michaels": (10.4, 34.53, 15.14, 4.85),
    },
    "m3abl_comb_unigru128_s1": {
        "dregon": (5.75, 5.63, 14.52, 5.42),
        "michaels": (12.03, 2.44, 25.89, 7.1),
    },
    "stoch_s1h_scv2": {
        "dregon": (5.96, 22.23, 24.26, 2.16),
        "michaels": (13.64, 42.56, 27.19, 3.42),
    },
    "stoch_s1s_both": {
        "dregon": (15.43, 16.72, 13.63, 15.26),
        "michaels": (11.79, 21.05, 8.14, 12.14),
    },
    "stoch_s1v_ground": {
        "dregon": (19.14, 6.13, 9.34, 21.97),
        "michaels": (9.10, 7.69, 14.64, 6.78),
    },
    "stoch_s1w_scv2": {
        "dregon": (12.71, 17.22, 27.12, 11.28),
        "michaels": (12.50, 13.55, 21.78, 8.15),
    },
    "stoch_s1x_scv2": {
        "dregon": (14.10, 6.54, 10.75, 15.65),
        "michaels": (7.93, 14.12, 12.07, 5.17),
    },
    "xrig_dregon_only": {
        "dregon": (5.80, 12.17, 15.46, 4.21),
        "michaels": (15.47, 1.78, 30.50, 10.59),
    },
    "hb_scv2_mag_nogate": {
        "dregon": (3.01, 3.1, 10.56, 2.69),
        "michaels": (2.3, 4.01, 3.09, 1.7),
    },
    "m3abl_comb_scv2_s1": {
        "dregon": (8.22, 7.41, 19.23, 7.92),
        "michaels": (11.38, 1.14, 27.53, 5.5),
    },
    "m3cur_scv2_s1": {
        "dregon": (13.25, 24.8, 20.77, 10.78),
        "michaels": (6.26, 8.9, 9.72, 4.31),
    },
    "stoch_s1e_scv2": {
        "dregon": (8.51, 11.93, 9.77, 7.82),
        "michaels": (14.01, 1.29, 30.57, 8.28),
    },
    "stoch_s1f_scv2": {
        "dregon": (9.89, 24.78, 17.27, 6.79),
        "michaels": (13.49, 59.79, 16.89, 5.42),
    },
    "stoch_s1r_long": {
        "dregon": (17.75, 12.13, 9.57, 19.14),
        "michaels": (18.85, 28.86, 18.86, 17.44),
    },
    "stoch_s1q_gru": {
        "dregon": (15.11, 13.63, 16.85, 15.32),
        "michaels": (15.27, 25.86, 19.58, 11.83),
    },
    "stoch_s1t_ownsilence": {
        "dregon": (18.51, 37.62, 22.07, 14.77),
        "michaels": (13.41, 37.58, 19.93, 7.04),
    },
    "stoch_s1u_composite": {
        "dregon": (22.41, 12.24, 11.3, 24.76),
        "michaels": (11.43, 17.59, 14.73, 9.07),
    },
    "xrig_michaels_only": {
        "dregon": (37.66, 1.49, 15.35, 45.35),
        "michaels": (5.46, 8.35, 10.87, 2.59),
    },
}

#: Frames behind each cell, so a small cell is never read as a firm result.
FRAMES = {
    "dregon": {"all": 5522, "zero": 844, "low": 182, "flight": 4496},
    "michaels": {"all": 3765, "zero": 333, "low": 1071, "flight": 2361},
}

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
            row["kind"] = (
                "control"
                if row["experiment"].startswith("xrig_")
                else ("real" if row["experiment"] in REAL_NAMES else "synthetic")
            )
            for rig in ("dregon", "michaels"):
                if f"{rig}_all_mae" in row:
                    RIG_CELLS.setdefault(row["experiment"], {})[rig] = (
                        row[f"{rig}_all_mae"], row[f"{rig}_zero_mae"],
                        row[f"{rig}_low_mae"], row[f"{rig}_flight_mae"],
                    )
            rows[row["experiment"]] = row
    return list(rows.values())


def rig_grid(target: dict, shown: list[dict]) -> None:
    """The 2x3 rig-by-regime grid, and where each cell stands against the target."""
    have = [r for r in shown if r["experiment"] in RIG_CELLS]
    if target["experiment"] not in RIG_CELLS or not have:
        return
    cells = ("all", "zero", "low", "flight")
    print()
    print("BY RIG (the same checkpoints, split by which aircraft the noise came from)")
    head = f"{'model':26s} {'rig':9s} {'frames':>7s} " + " ".join(f"{c:>8s}" for c in cells)
    print(head)
    print("-" * len(head))
    for row in [target, *[r for r in have if r["experiment"] != target["experiment"]]]:
        if row["experiment"] not in RIG_CELLS:
            continue
        for rig in ("dregon", "michaels"):
            v = RIG_CELLS[row["experiment"]].get(rig)
            if v is None:
                continue
            mark = "   <- target" if row is target and rig == "michaels" else ""
            print(f"{row['experiment'] if rig == 'dregon' else '':26s} {rig:9s} "
                  f"{FRAMES[rig]['all'] if rig else 0:7d} "
                  + " ".join(f"{x:8.2f}" for x in v) + mark)

    print()
    print("best synthetic-only per RIG x REGIME cell, against the target in that cell:")
    tgt = RIG_CELLS[target["experiment"]]
    synth = [r for r in have if r["kind"] == "synthetic"]
    for rig in ("dregon", "michaels"):
        for j, cell in enumerate(cells):
            if cell == "all":
                continue
            best = min(synth, key=lambda r, j=j, rig=rig: RIG_CELLS[r["experiment"]][rig][j])
            got, want = RIG_CELLS[best["experiment"]][rig][j], tgt[rig][j]
            flag = "  <- AT OR BETTER THAN TARGET" if got <= want else ""
            print(f"   {rig:9s} {cell:7s} {got:6.2f} vs {want:5.2f}  ({got / want:.2f}x)  "
                  f"{best['experiment']:26s} [{FRAMES[rig][cell]:4d} frames]{flag}")


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
    if target:
        # every model with rig cells, not just the top-N of the regime view
        rig_grid(target, [target, *[r for r in synth if r["experiment"] in RIG_CELLS]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
