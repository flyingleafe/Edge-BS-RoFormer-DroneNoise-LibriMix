#!/usr/bin/env python3
"""Aggregate a :mod:`scripts.blind_corpus` run and draw the overlay figures.

Reads ``<run>/raw/*.json`` (the unit readings), ``<run>/traj/*.npz`` (the
annotated trajectories) and ``<run>/windows/*.npz`` (the cached audio), and
writes:

* ``<out>/units.csv`` — one row per unit, every headline reading.
* ``<out>/by_recording.csv`` — per-recording means plus the cross-window
  self-consistency of the annotation (the spread of a rotor's mean rate across
  windows, which on a hover is a direct precision proxy).
* ``<out>/overlay_<uid>.png`` — the whitened log-magnitude spectrogram with the
  annotated comb drawn on it, for the units named by ``--overlays`` (default:
  the three best and the three worst by ridge clearance).

Run:
  PYTHONPATH=src python scripts/blind_corpus_report.py --run results/blind_corpus/avq_mono
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

SR = 16000

#: The unit-level columns the report is read off. Order is the table's order.
COLUMNS = (
    "uid",
    "recording_id",
    "window",
    "arm",
    "t0_s",
    "dur_s",
    "n_channels",
    "ridge_clearance_db",
    "ridge_clearance_mismatch_db",
    "ridge_margin_half_db",
    "ridge_margin_double_db",
    "fvk_blind",
    "fvk_well_p03",
    "fvk_ratio_double",
    "fvk_ratio_half",
    "fvk_alias_ratio_half",
    "seed_octave",
    "coarse_halved",
    "coarse_mode",
    "spread_rev_s",
    "ref_mae_rev_s",
    "wall_ladder_s",
)


def load_rows(run: Path) -> list[dict[str, Any]]:
    raw = run / "raw"
    if not raw.is_dir():
        raise SystemExit(f"no unit JSONs under {raw}")
    rows = []
    for p in sorted(raw.glob("*.json")):
        d = json.loads(p.read_text())
        d["unit_uid"] = p.stem
        d["rps_mean_str"] = ", ".join(f"{x:.1f}" for x in d.get("rps_mean", []))
        d["rps_std_max"] = max(d.get("rps_std", [0.0]) or [0.0])
        rows.append(d)
    return rows


def write_units(rows: list[dict[str, Any]], out: Path) -> None:
    cols = [*COLUMNS, "rps_mean_str", "rps_std_max"]
    with (out / "units.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def by_recording(rows: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    """Per (recording, arm) aggregate + the cross-window self-consistency.

    ``rate_spread_across_windows`` is the standard deviation, over windows, of
    the window-mean rate averaged across rotors. On a hover the true rate is
    near constant, so this is what an annotation's repeatability looks like
    when no label exists to compare against.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r.get("recording_id")), str(r.get("arm"))), []).append(r)

    def _m(g: list[dict[str, Any]], key: str) -> float | None:
        v = [x[key] for x in g if isinstance(x.get(key), (int, float))]
        return round(float(np.mean(v)), 3) if v else None

    agg = []
    for (rid, arm), g in sorted(groups.items()):
        win_means = [float(np.mean(x["rps_mean"])) for x in g if x.get("rps_mean")]
        agg.append(
            {
                "recording_id": rid,
                "arm": arm,
                "n_windows": len(g),
                "ridge_clearance_db": _m(g, "ridge_clearance_db"),
                "ridge_margin_half_db": _m(g, "ridge_margin_half_db"),
                "fvk_well_p03": _m(g, "fvk_well_p03"),
                "fvk_ratio_double": _m(g, "fvk_ratio_double"),
                "fvk_ratio_half": _m(g, "fvk_ratio_half"),
                "within_window_std_rev_s": _m(g, "rps_std_max"),
                "rate_spread_across_windows": (
                    round(float(np.std(win_means)), 3) if len(win_means) > 1 else None
                ),
                "mean_rate_rev_s": round(float(np.mean(win_means)), 2) if win_means else None,
                "ref_mae_rev_s": _m(g, "ref_mae_rev_s"),
                "wall_ladder_s": _m(g, "wall_ladder_s"),
            }
        )
    with (out / "by_recording.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    return agg


def overlay(run: Path, uid: str, out: Path, k_draw: int = 8) -> Path | None:
    """Whitened spectrogram of the window + the annotated comb drawn over it.

    The frequency axis is cropped to the first ``k_draw`` teeth of the FASTEST
    rotor. A quadrotor comb has a tooth every ~40-120 Hz per rotor, so a full
    0-8 kHz view of four combs is 400 lines on top of each other and shows
    nothing; the first few teeth are where a reader can actually see whether a
    drawn line sits on a ridge or between two.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tracking.vk_blind_seeding import whitened_logmag

    win_uid = uid.rsplit("__", 1)[0]  # strip the arm suffix
    audio_p = run / "windows" / f"{win_uid}.npz"
    traj_p = run / "traj" / f"{uid}.npz"
    if not (audio_p.exists() and traj_p.exists()):
        return None
    audio = np.asarray(np.load(audio_p)["audio"], dtype=np.float64)
    tj = np.load(traj_p)
    r, ft = np.asarray(tj["rps"], dtype=np.float64), np.asarray(tj["ft"], dtype=np.float64)

    white, bin_hz, st = whitened_logmag(audio.astype(np.float32), float(SR))
    f_top = float(k_draw * r.mean(axis=1).max() * 1.15)
    f_hi_bin = int(np.clip(round(f_top / bin_hz), 16, white.shape[0]))
    band = white[:f_hi_bin]

    fig, ax = plt.subplots(figsize=(12, 6.0), constrained_layout=True)
    ax.imshow(
        band,
        origin="lower",
        aspect="auto",
        extent=(float(st[0]), float(st[-1]), 0.0, f_hi_bin * bin_hz),
        cmap="gray_r",
        interpolation="nearest",
        vmin=float(np.percentile(band, 55)),
        vmax=float(np.percentile(band, 99.0)),
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for i, row in enumerate(r):
        for k in range(1, k_draw + 1):
            line = k * np.interp(st, ft, row)
            if np.median(line) > f_hi_bin * bin_hz:
                break
            ax.plot(
                st,
                line,
                color=colors[i % len(colors)],
                lw=1.1,
                alpha=0.85,
                ls=(0, (5, 4)),
                label=f"rotor {i}: {row.mean():.1f} rev/s" if k == 1 else None,
            )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(
        f"{uid} — blind annotation (dashed) over the whitened spectrogram, first {k_draw} teeth"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=2)
    p = out / f"overlay_{uid}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--run", required=True, help="a scripts/blind_corpus.py output directory")
    ap.add_argument("--out", default=None, help="default: <run>/report")
    ap.add_argument(
        "--overlays", default=None, help="comma-separated unit uids (default: best/worst)"
    )
    ap.add_argument("--n-extremes", type=int, default=3)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "report"
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(run)
    write_units(rows, out)
    agg = by_recording(rows, out)

    ranked = [r for r in rows if isinstance(r.get("ridge_clearance_db"), (int, float))]
    ranked.sort(key=lambda r: r["ridge_clearance_db"])
    if args.overlays:
        uids = [u.strip() for u in args.overlays.split(",") if u.strip()]
    else:
        n = args.n_extremes
        uids = [r["unit_uid"] for r in ranked[:n]] + [r["unit_uid"] for r in ranked[-n:]]
    made = [p for u in uids if (p := overlay(run, u, out)) is not None]

    print(
        json.dumps(
            {"n_units": len(rows), "n_recordings": len(agg), "overlays": len(made)}, indent=1
        )
    )
    print(f"\nWorst {args.n_extremes} by ridge clearance:")
    for r in ranked[: args.n_extremes]:
        print(
            f"  {r['unit_uid']}: clearance {r['ridge_clearance_db']} dB, "
            f"half_margin {r.get('ridge_margin_half_db')} dB, "
            f"well_p03 {r.get('fvk_well_p03')}, rps [{r['rps_mean_str']}]"
        )
    print(f"\nBest {args.n_extremes} by ridge clearance:")
    for r in ranked[-args.n_extremes :]:
        print(
            f"  {r['unit_uid']}: clearance {r['ridge_clearance_db']} dB, "
            f"half_margin {r.get('ridge_margin_half_db')} dB, "
            f"well_p03 {r.get('fvk_well_p03')}, rps [{r['rps_mean_str']}]"
        )
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
