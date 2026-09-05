"""Profile the ERROR DISTRIBUTION of RPS predictors from a `scripts/rps_dump.py` dump.

The monitored number is a mean, and a mean cannot tell "imprecise on every
clip" from "exact on most clips, lost on a few". Those call for different
fixes (a better front end against a better decoder), so this CLI reads the
per-frame predictions the dump kept and reports, per (model, set):

* the percentiles of the per-frame PIT MAE and the share of the total error
  carried by the worst tenth of the frames (the skew);
* how much of the frame-to-frame variance is BETWEEN flights (all eight mics
  of one clip failing together) against within a flight;
* a classification of every failed rotor track by the SHAPE of its error --
  a phantom rotor on silence, a missed rotor, a harmonic alias, two tracks
  collapsed onto one rotor, a constant offset, bursts, or a wander.

    python scripts/rps_error_profile.py --dump results/rps_dump --out results/rps_profile

It also reads the FAN, the statistic of `scripts/spread_eval.py` measured on
the dump instead of on a live stream. The fan of a time-frame is the maximum
rotor speed minus the minimum one, and it is read on cruise time-frames only
(the true mean over the rotors is 45 rev/s or more). A model that tracks four
lines follows the true fan. A model that collapsed onto the mean of the comb
keeps a near-constant fan however far the rotors separate. `fan.csv` gives the
true and the predicted fan in each bucket of the true fan, `summary.csv` gives
the two means over all cruise time-frames plus `fan_slope`, the least-squares
slope of the predicted fan against the true one (1 for a tracker, 0 for a fixed
fan).

Outputs are tidy CSVs (`summary.csv`, `frames.csv`, `classes.csv`, `fan.csv`)
that `scripts/table.py` pivots, and the headline tables are printed.

PIT here is the MAE-optimal assignment over all 4! permutations, on the
label resampled to the prediction's frame grid the way the salience metric
resamples it. That is the salience family's monitored convention; the
regressors' monitored metric assigns on MSE, so the per-frame numbers here can
differ from `metric` in the dump by a few percent on a frame where the two
assignments disagree -- the dump's `metric` column is kept in `frames.csv` as
`metric_monitored` for that check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import permutations
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.rps_bench import resample_like_metric  # noqa: E402

# ─── Thresholds (rev/s) ───────────────────────────────────────────────────────
BAD_ROTOR = 1.0  # a rotor track with MAE above this is "failed"
STOPPED = 1.0  # a rotor whose mean label is below this is stopped
RUNNING = 5.0  # ... and above this is unambiguously running
ALIAS_TOL = 0.04  # relative tolerance for a rational-ratio alias
RATIOS = {
    "1/2": 0.5,
    "1/3": 1 / 3,
    "2/3": 2 / 3,
    "3/4": 0.75,
    "4/5": 0.8,
    "5/4": 1.25,
    "4/3": 4 / 3,
    "3/2": 1.5,
    "2": 2.0,
    "3": 3.0,
}
PERMS = list(permutations(range(4)))

# ─── Fan (rotor spread) ───────────────────────────────────────────────────────
# The definitions of `scripts/spread_eval.py`: a cruise time-frame has a true
# mean rotor speed of CRUISE_MIN or more, and the buckets are on the true fan.
CRUISE_MIN = 45.0
SPREAD_EDGES = [0.0, 2.0, 5.0, 10.0, 20.0, np.inf]


def parse_name(exp: str) -> dict[str, str]:
    """Experiment name -> (arch, train, speech, objective) columns for pivots."""
    m = re.match(r"salv2_(hf0|hppnet|scv2)_(comb|stoch)_(nomix|mix)(_crf)?$", exp)
    if m:
        arch, train, speech, crf = m.groups()
        obj = "crf" if crf else ("mse" if arch == "scv2" else "bce")
        return dict(arch=arch, train=train, speech=speech, objective=obj)
    m = re.match(r"m3mixv2_(scv2|unigru128|transformer)$", exp)
    if m:
        return dict(arch=m.group(1), train="mixed", speech="mix", objective="mse")
    m = re.match(r"r4hb_(scv2|gru|tr)$", exp)
    if m:
        arch = {"gru": "unigru128", "tr": "transformer"}.get(m.group(1), m.group(1))
        return dict(arch=arch, train="comb>real", speech="mix", objective="mse")
    m = re.match(r"(hppnet|hf0)_r4_l4$", exp)
    if m:
        return dict(arch=m.group(1), train="comb>real", speech="mix", objective="bce")
    return dict(arch=exp, train="?", speech="?", objective="?")


def pit_mae(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """MAE-optimal rotor assignment; returns gt permuted onto pred's rows."""
    cost = np.abs(pred[:, None] - gt[None, :]).mean(-1)  # (R_pred, R_gt)
    best = min(PERMS, key=lambda p: sum(cost[k, p[k]] for k in range(4)))
    return gt[list(best)], best


def acorr_time(x: np.ndarray, frame_s: float, max_lag: int = 120) -> float:
    x = x - x.mean()
    v = float((x * x).mean())
    if v <= 0:
        return 0.0
    for lag in range(1, min(max_lag, len(x) - 1)):
        if float((x[:-lag] * x[lag:]).mean()) / v < np.exp(-1.0):
            return lag * frame_s
    return max_lag * frame_s


def classify_rotor(
    pred: np.ndarray, gt: np.ndarray, gt_all: np.ndarray, k: int
) -> tuple[str, dict[str, float]]:
    """One failed track ``pred[k]`` against its matched label ``gt[k]``.

    Order matters: the structural cases (phantom, missed, dup, alias) are
    named before the residual-shape cases (offset, burst, wander), so a track
    that is both an alias and "wrong most of the time" is called an alias.
    """
    e = pred[k] - gt[k]
    gm, pm = float(gt[k].mean()), float(pred[k].mean())
    bias = float(e.mean())
    resid = float((e - bias).std())
    bad_frac = float((np.abs(e) > BAD_ROTOR).mean())
    feats = dict(bias=bias, resid=resid, bad_frac=bad_frac, gt_mean=gm, pred_mean=pm)
    if gm < STOPPED and pm > 2.0:
        return "phantom", feats
    if gm > RUNNING and pm < 2.0:
        return "missed", feats
    # collapsed onto another rotor: this track sits on some OTHER label
    others = [c for c in range(4) if c != k]
    if any(np.abs(pred[k] - gt_all[c]).mean() < BAD_ROTOR for c in others):
        return "dup", feats
    run = gt[k] > RUNNING
    if run.mean() > 0.5:
        ratio = float(np.median(pred[k][run] / gt[k][run]))
        for name, q in RATIOS.items():
            if abs(ratio / q - 1.0) < ALIAS_TOL:
                feats["ratio"] = ratio
                return f"alias {name}", feats
    if abs(bias) >= BAD_ROTOR and resid < 0.5 * abs(bias):
        return "offset", feats
    if bad_frac < 0.5:
        return "burst", feats
    return "wander", feats


def cruise_fan(pred: np.ndarray, gt_al: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The true and the predicted fan of the CRUISE time-frames of one frame.

    Both arrays have one value for each cruise time-frame, so the caller can
    concatenate them over the samples and bucket them by the true fan.
    """
    cruise = gt_al.mean(0) >= CRUISE_MIN
    true_sp = (gt_al.max(0) - gt_al.min(0))[cruise]
    pred_sp = (pred.max(0) - pred.min(0))[cruise]
    return true_sp, pred_sp


def fan_summary(true_sp: np.ndarray, pred_sp: np.ndarray) -> dict[str, float]:
    """The fan means and the fan slope over all cruise time-frames of a model.

    The slope is the least-squares slope of the predicted fan against the true
    fan. It is 1 for a model that follows the rotors and 0 for a fixed fan. It
    stays NaN below two time-frames, and also when every true fan is the same,
    because the slope is not defined there.
    """
    if true_sp.size == 0:
        return dict(fan_true=np.nan, fan_pred=np.nan, fan_slope=np.nan)
    slope = np.nan
    if true_sp.size >= 2:
        dt = true_sp - true_sp.mean()
        var = float((dt * dt).sum())
        if var > 0:
            slope = float((dt * (pred_sp - pred_sp.mean())).sum() / var)
    return dict(fan_true=float(true_sp.mean()), fan_pred=float(pred_sp.mean()), fan_slope=slope)


def fan_buckets(true_sp: np.ndarray, pred_sp: np.ndarray) -> list[dict[str, Any]]:
    """One row for each bucket of the true fan. An empty bucket gives NaN."""
    rows: list[dict[str, Any]] = []
    for b in range(len(SPREAD_EDGES) - 1):
        lo, hi = SPREAD_EDGES[b], SPREAD_EDGES[b + 1]
        m = (true_sp >= lo) & (true_sp < hi)
        rows.append(
            dict(
                spread_lo=lo,
                spread_hi=hi,
                n_frames=int(m.sum()),
                fan_true=float(true_sp[m].mean()) if m.any() else np.nan,
                fan_pred=float(pred_sp[m].mean()) if m.any() else np.nan,
            )
        )
    return rows


def profile_frame(
    pred: np.ndarray, gt_raw: np.ndarray, frame_s: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """One dump sample: the row of `frames.csv` plus its cruise fan arrays."""
    gt = resample_like_metric(gt_raw, pred.shape[-1])
    gt_al, _ = pit_mae(pred, gt)
    e = pred - gt_al
    per_rotor = np.abs(e).mean(-1)
    true_sp, pred_sp = cruise_fan(pred, gt_al)
    row: dict[str, Any] = dict(
        mae=float(per_rotor.mean()),
        mae_rotor_max=float(per_rotor.max()),
        n_bad_rotors=int((per_rotor >= BAD_ROTOR).sum()),
        frac_bad_frames=float((np.abs(e) > BAD_ROTOR).mean()),
        frac_bad_frames5=float((np.abs(e) > 5.0).mean()),
        bias_abs=float(np.abs(e.mean(-1)).mean()),
        resid_std=float((e - e.mean(-1, keepdims=True)).std(-1).mean()),
        tau_s=float(np.median([acorr_time(x, frame_s) for x in e - e.mean(-1, keepdims=True)])),
        gt_mean=float(gt_al.mean()),
        gt_zero_frac=float((gt_al < 0.5).mean()),
        gt_range=float((gt_al.max(-1) - gt_al.min(-1)).mean()),
        gt_spread=float(gt_al.mean(-1).max() - gt_al.mean(-1).min()),
        n_stopped=int((gt_al.mean(-1) < STOPPED).sum()),
        # the fan of this sample, on its cruise time-frames only
        fan_true=float(true_sp.mean()) if true_sp.size else np.nan,
        fan_pred=float(pred_sp.mean()) if pred_sp.size else np.nan,
        fan_n=int(true_sp.size),
    )
    # the class of the frame is the class of the rotor carrying most error
    if row["n_bad_rotors"]:
        k = int(np.argmax(per_rotor))
        cls, feats = classify_rotor(pred, gt_al, gt_al, k)
        row["cls"] = cls
        row["cls_bias"] = feats["bias"]
        row["cls_resid"] = feats["resid"]
        row["cls_ratio"] = feats.get("ratio", np.nan)
        row["cls_all"] = "|".join(
            classify_rotor(pred, gt_al, gt_al, r)[0] for r in range(4) if per_rotor[r] >= BAD_ROTOR
        )
    else:
        row["cls"] = "ok"
        row["cls_bias"] = row["cls_resid"] = row["cls_ratio"] = np.nan
        row["cls_all"] = ""
    return row, true_sp, pred_sp


def flight_id(meta: dict[str, Any]) -> str:
    return str(meta.get("recording_id", meta.get("sample_id", "?")))


def load_set(d: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    g = np.load(d / "_gt.npz")
    meta = json.loads((d / "_meta.json").read_text())
    return g["rps"], g["n_t"], meta


def summarize(fr: pd.DataFrame) -> dict[str, float]:
    m = fr["mae"].to_numpy()
    q = np.percentile(m, [10, 25, 50, 75, 90, 95])
    worst = np.sort(m)[::-1][: max(1, int(round(0.1 * len(m))))]
    by_flight = fr.groupby("flight")["mae"].mean()
    icc = float(by_flight.var() / m.var()) if len(by_flight) > 1 and m.var() > 0 else np.nan
    return dict(
        n=len(m),
        mean=float(m.mean()),
        monitored=float(fr["metric_monitored"].mean()),
        p10=q[0],
        p25=q[1],
        median=q[2],
        p75=q[3],
        p90=q[4],
        p95=q[5],
        max=float(m.max()),
        mean_over_median=float(m.mean() / q[2]) if q[2] > 0 else np.nan,
        worst10_share=float(worst.sum() / m.sum()) if m.sum() > 0 else np.nan,
        frac_gt_0p5=float((m > 0.5).mean()),
        frac_gt_1=float((m > 1.0).mean()),
        frac_gt_5=float((m > 5.0).mean()),
        flight_var_share=icc,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dump", default="results/rps_dump")
    ap.add_argument("--sets", default="", help="comma-separated set names (default: all)")
    ap.add_argument("--experiments", default="", help="comma-separated (default: all dumped)")
    ap.add_argument("--out", default="results/rps_profile")
    ap.add_argument("--frame-s", type=float, default=512 / 16000)
    a = ap.parse_args()

    dump = Path(a.dump)
    sets = [s for s in a.sets.split(",") if s] or sorted(
        p.name for p in dump.iterdir() if (p / "_gt.npz").exists()
    )
    want = {e for e in a.experiments.split(",") if e}

    frames: list[dict[str, Any]] = []
    # the cruise fans of every (exp, set), one value for each cruise
    # time-frame. The per-sample means cannot give the buckets, so the buckets
    # come from these arrays.
    fan_acc: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for s in sets:
        d = dump / s
        gt, gt_n_t, meta = load_set(d)
        for f in sorted(d.glob("*.npz")):
            exp = f.stem
            if exp.startswith("_") or (want and exp not in want):
                continue
            z = np.load(f)
            pred, n_t, metric = z["pred"], z["n_t"], z["metric"]
            cols = parse_name(exp)
            acc = fan_acc.setdefault((exp, s), ([], []))
            for i in range(pred.shape[0]):
                row, true_sp, pred_sp = profile_frame(
                    pred[i, :, : n_t[i]].astype(np.float64),
                    gt[i, :, : gt_n_t[i]].astype(np.float64),
                    a.frame_s,
                )
                acc[0].append(true_sp)
                acc[1].append(pred_sp)
                row.update(
                    exp=exp,
                    set=s,
                    frame=i,
                    flight=flight_id(meta[i]),
                    channel=meta[i].get("channel", -1),
                    metric_monitored=float(metric[i]),
                    **cols,
                )
                frames.append(row)
            print(f"  {s:12s} {exp:32s} {pred.shape[0]:4d} frames", flush=True)

    fr = pd.DataFrame(frames)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fr.to_csv(out / "frames.csv", index=False)

    # one pair of concatenated cruise-fan arrays for each (exp, set)
    fans = {
        k: (
            np.concatenate(v[0]) if v[0] else np.zeros(0),
            np.concatenate(v[1]) if v[1] else np.zeros(0),
        )
        for k, v in fan_acc.items()
    }
    fan_rows = []
    for (exp, s), (true_sp, pred_sp) in fans.items():
        for b in fan_buckets(true_sp, pred_sp):
            fan_rows.append(dict(exp=exp, set=s, **parse_name(exp), **b))
    fan = pd.DataFrame(fan_rows)
    fan.to_csv(out / "fan.csv", index=False)

    rows = []
    for key, g in fr.groupby(["exp", "set"], sort=False):
        exp, s = cast(tuple[str, str], key)
        ts, ps = fans.get((exp, s), (np.zeros(0), np.zeros(0)))
        rows.append(dict(exp=exp, set=s, **parse_name(exp), **summarize(g), **fan_summary(ts, ps)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    bad = fr[fr["cls"] != "ok"]
    cls_rows = []
    for key, g in fr.groupby(["exp", "set"], sort=False):
        exp, s = cast(tuple[str, str], key)
        tot = g["mae"].sum()
        gb = g[g["cls"] != "ok"]
        for cls, h in gb.groupby("cls"):
            cls_rows.append(
                dict(
                    exp=exp,
                    set=s,
                    cls=cls,
                    n=len(h),
                    frac_frames=len(h) / len(g),
                    error_share=float(h["mae"].sum() / tot) if tot > 0 else np.nan,
                    mae=float(h["mae"].mean()),
                )
            )
    classes = pd.DataFrame(cls_rows)
    classes.to_csv(out / "classes.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print("\n=== per-frame PIT MAE, rev/s (rows: model; blocks: set) ===")
    show = cast(
        pd.DataFrame,
        summary[
            [
                "exp",
                "set",
                "n",
                "monitored",
                "mean",
                "median",
                "p90",
                "max",
                "mean_over_median",
                "worst10_share",
                "frac_gt_1",
                "flight_var_share",
            ]
        ],
    ).round(3)
    for s in sets:
        print(f"\n--- {s}")
        print(show[show["set"] == s].drop(columns="set").to_string(index=False))

    print("\n=== failed frames by error class (share of the model's total error) ===")
    if len(classes):
        piv = classes.pivot_table(
            index=["set", "exp"], columns="cls", values="error_share", aggfunc="sum"
        ).fillna(0.0)
        print(piv.round(2).to_string())

    print("\n=== fan on cruise time-frames (slope 1 = tracks, 0 = fixed fan) ===")
    print(
        cast(pd.DataFrame, summary[["exp", "set", "fan_true", "fan_pred", "fan_slope"]])
        .round(3)
        .to_string(index=False)
    )
    print(
        f"\n{len(bad)} failed frames of {len(fr)}; "
        f"wrote {out}/summary.csv, frames.csv, classes.csv, fan.csv"
    )


if __name__ == "__main__":
    main()
