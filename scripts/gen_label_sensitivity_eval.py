#!/usr/bin/env python
"""Per-harmonic readout of the Phase-7 generator label-sensitivity arms.

The question is whether telemetry label noise (the tachometer staircase) makes
a conditioned noise generator underfit the *high* harmonics. Each arm trained on
the same frozen-profile comb and differs only in the RPS conditioning it was
shown (``conf/experiment/p7_labelsens_*.yaml``). This script reads the trained
generators, so the verdict is per ``k``, not one scalar.

Two readouts, deliberately different:

**1. Learned line profile** (``--mode line``, the primary). Condition each arm on
a *constant* rotor speed chosen so every harmonic lands on an exact FFT bin of
the analysis window, render, and compare each line's power against the profile's
**analytic** amplitude ``a_k * gain`` — no reference rendering, hence no
reference-side measurement noise. This measures the amplitude the arm *learned*,
with its response to a jittery input switched off.

Because an arm trained on ``scale``d labels may have absorbed the 0.542 % bias
into its own frequency mapping, a single global frequency scale is estimated
first, from the low harmonics only, and reported. Everything after that is read
in a narrow band at the scale-corrected location — a fixed region with a local
floor subtracted, never a peak search inside a window (that estimator returns
about W/2 on pure noise and has already withdrawn two claims in this project;
see docs/experiments/dregon-comb-displacement.md).

**2. Delivered fidelity** (``--mode track``, the complement). Condition each arm
on the labels it was *trained* on, over held-out OU trajectories, and score the
comb-masked mean ``|Delta log-mag|`` along the harmonic tracks of the TRUE
trajectory, in ``k`` bands — the E6 measure. Readout 1 can exonerate an arm that
learned "smear when the input jitters" rather than "attenuate"; readout 2 cannot.

Usage::

    python scripts/gen_label_sensitivity_eval.py --out results/gen_label_sensitivity
    python scripts/gen_label_sensitivity_eval.py --arms exact,tach --ckpt last
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ARMS = ("exact", "scale", "tach", "tach_presmooth")
#: Arms whose experiment name is not their dataset ``label_mode``. ``tach_pure``
#: is arm B trained with ``label_scale: 1.0`` — the staircase with no constant
#: bias — so its data is ``tach`` at unit scale, which is what ``_dataset``
#: builds anyway (``StaticCombGenDataset.label_scale`` defaults to 1.0, and the
#: model readouts deliberately never pass ``--label-scale``: the fitted global
#: frequency scale is the one nuisance parameter, estimated per arm).
ARM_LABEL_MODE: dict[str, str] = {
    "exact": "exact",
    "scale": "scale",
    "tach": "tach",
    "tach_presmooth": "tach_presmooth",
    "tach_pure": "tach",
}
#: k bands of the summary table. The loss's finest scale (n_fft 2048) has
#: 7.8 Hz bins, and the staircase displaces harmonic k by 0.106*k Hz, so the
#: displacement crosses half a bin near k = 37 and one bin near k = 74.
K_BANDS: dict[str, range] = {
    "k1-9": range(1, 10),
    "k10-24": range(10, 25),
    "k25-49": range(25, 50),
    "k50-80": range(50, 81),
}


def _dataset(label_mode: str, **kw: Any):
    from data_processing.frame_datasets import StaticCombGenDataset

    return StaticCombGenDataset(label_mode=label_mode, **kw)


# ---------------------------------------------------------------------------
# readout 1: the learned line profile at constant conditioning


def _spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular-window power spectrum, normalized so a sinusoid of amplitude
    ``A`` integrates to ``A**2 / 2`` over its band.

    No taper: the eval windows are chosen so every harmonic is periodic in them,
    which is worth more than a taper. The ``2 / N**2`` factor is what makes the
    measurement directly comparable to the profile's analytic ``a_k * gain``
    without a fitted gain in between.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    spec = np.fft.rfft(x)
    return (np.abs(spec) ** 2) * (2.0 / n**2), np.fft.rfftfreq(n, 1.0 / sr)


def _band_power(
    power: np.ndarray, freqs: np.ndarray, center: float, half_bw: float
) -> tuple[float, float]:
    """Floor-subtracted power in ``|f - center| <= half_bw``, and its rms spread.

    One line, because this measurement has ONE implementation:
    :func:`tracking.fitness.line_power`. It was written here and promoted there
    in phase 6d, where the tracking harness's ridge component needed the same
    reading (a fixed band against a local floor, never a peak search) on the
    demodulated envelope spectrum. The floor is the median power *density* of
    the annulus ``[3, 8] * half_bw`` on both sides.
    """
    from tracking.fitness import line_power

    lp = line_power(power, freqs, center, half_bw)
    if lp.n_bins == 0:
        return float("nan"), float("nan")
    return float(lp.total), float(lp.spread_hz)


def _frequency_scale(
    power: np.ndarray, freqs: np.ndarray, f0: float, k_max: int, tol: float
) -> float:
    """One global frequency scale, estimated from the low harmonics.

    An arm trained on biased labels can absorb the bias into its own mapping, so
    its comb may sit a constant fraction off. That is ONE nuisance parameter, and
    it is estimated once from the harmonics with the best line-to-floor ratio —
    never per ``k``, which would make every displacement unmeasurable by
    construction.
    """
    ratios = []
    for k in range(1, k_max + 1):
        center = k * f0
        off = freqs - center
        sel = np.abs(off) <= tol * center
        if not sel.any():
            continue
        p = np.clip(power[sel] - float(np.median(power[sel])), 0.0, None)
        if p.sum() <= 0.0:
            continue
        ratios.append(float((freqs[sel] * p).sum() / p.sum()) / center)
    return float(np.median(ratios)) if ratios else 1.0


def line_readout(
    fm: Any,
    ds: Any,
    *,
    f0_grid: tuple[float, ...],
    dur_s: float,
    scale_k_max: int,
    scale_tol: float,
) -> dict[str, Any]:
    """Per-``k`` line power of one arm against the analytic profile."""
    import torch

    sr = ds.sample_rate
    a_k = np.asarray(ds.profile.a_k, dtype=np.float64) * ds.gain
    k_max = a_k.shape[0]
    ref_power = (a_k**2) / 2.0  # power of a unit-amplitude sinusoid

    per_k: dict[int, list[float]] = {k: [] for k in range(1, k_max + 1)}
    per_k_spread: dict[int, list[float]] = {k: [] for k in range(1, k_max + 1)}
    scales: list[float] = []
    n_t = int(round(dur_s * sr))

    for f0 in f0_grid:
        label = f0 * (1.0 if ds.label_mode == "exact" else ds.label_scale)
        frame = td.Frame(
            {
                "audio": td.uniform(
                    np.zeros((1, n_t), np.float32), sr, dims=("mic", "time"), t_start=0.0
                ),
                "rps": td.uniform(
                    np.full((1, n_t), label, np.float32), sr, dims=("rotor", "time"), t_start=0.0
                ),
                "mic_pos": td.wrap(ds.MIC_POS, dims=("mic", None)),
                "rotor_pos": td.wrap(ds.ROTOR_POS, dims=("rotor", None)),
                "meta": td.Frame({"drone": "synth"}),
            }
        )
        with torch.no_grad():
            pred = np.asarray(fm(frame)["audio"].data, dtype=np.float64)
        pred = pred[0] if pred.ndim == 2 else pred
        power, freqs = _spectrum(pred, sr)
        df = float(freqs[1] - freqs[0])
        s = _frequency_scale(power, freqs, f0, scale_k_max, scale_tol)
        scales.append(s)
        for k in range(1, k_max + 1):
            center = s * k * f0
            if center >= sr / 2.0 * 0.98 or ref_power[k - 1] <= 0.0:
                continue
            half_bw = max(5.0 * df, 0.0015 * center)
            p, spread = _band_power(power, freqs, center, half_bw)
            if not np.isfinite(p):
                continue
            per_k[k].append(10.0 * np.log10(max(p, 1e-30) / ref_power[k - 1]))
            per_k_spread[k].append(spread)

    delta_db = {k: float(np.mean(v)) for k, v in per_k.items() if v}
    spread_hz = {
        k: float(np.nanmean(v)) for k, v in per_k_spread.items() if v and np.any(np.isfinite(v))
    }
    return {
        "freq_scale": float(np.median(scales)) if scales else float("nan"),
        "delta_db": delta_db,
        "spread_hz": spread_hz,
    }


# ---------------------------------------------------------------------------
# readout 2: delivered fidelity on the arm's own labels (the E6 measure)


def _logspec(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    win = np.hanning(n_fft)
    n_frames = 1 + (x.shape[-1] - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * win for i in range(n_frames)], axis=-1)
    return 20.0 * np.log10(np.abs(np.fft.rfft(frames, axis=0)) + 1e-6)


def track_readout(
    fm: Any, ds: Any, *, n_windows: int, n_fft: int = 2048, hop: int = 512
) -> dict[str, float]:
    """Comb-masked mean ``|Delta log-mag|`` per ``k`` band, arm's own labels."""
    import torch

    sr = ds.sample_rate
    k_max = ds.profile.a_k.shape[0]
    sums: dict[str, list[float]] = {name: [] for name in K_BANDS}
    for idx in range(n_windows):
        frame = ds[idx]
        ext, sl = ds._traj(idx)
        rps_true = ext[sl]
        target = np.asarray(frame["audio"].data, dtype=np.float64)[0]
        with torch.no_grad():
            pred = np.asarray(fm(frame)["audio"].data, dtype=np.float64)
        pred = pred[0] if pred.ndim == 2 else pred
        rms_t = float(np.sqrt(np.mean(target**2))) or 1.0
        rms_p = float(np.sqrt(np.mean(pred**2))) or 1.0
        pred = pred * (rms_t / rms_p)
        s_t, s_p = _logspec(target, n_fft, hop), _logspec(pred, n_fft, hop)
        # Frame f spans [f*hop, f*hop + n_fft), so its rate is the one at its
        # CENTRE. Point-sampling at f*hop instead is a 64 ms lead, which at
        # k = 80 displaces the read bin by more than the effect being measured.
        centres = np.arange(s_t.shape[1]) * hop + n_fft // 2
        rf = rps_true[np.clip(centres, 0, rps_true.shape[-1] - 1)]
        for name, ks in K_BANDS.items():
            diffs = []
            for k in ks:
                if k > k_max:
                    continue
                bins = np.rint(k * rf * n_fft / sr).astype(int)
                ok = (bins > 0) & (bins < s_t.shape[0])
                if not ok.any():
                    continue
                t_idx = np.arange(len(rf))[ok]
                diffs.append(np.abs(s_t[bins[ok], t_idx] - s_p[bins[ok], t_idx]))
            if diffs:
                sums[name].append(float(np.mean(np.concatenate(diffs))))
    return {name: float(np.mean(v)) if v else float("nan") for name, v in sums.items()}


# ---------------------------------------------------------------------------


def loss_pressure(
    *, n_windows: int, label_scale: float, split: str, compensate_scale: bool
) -> dict[str, dict[str, float]]:
    """What each arm's label costs a generator that keeps FULL-amplitude lines.

    No model: render the frozen comb twice from the same profile, once at the
    true trajectory and once at the arm's label, and score the comb-masked
    ``|Delta log-mag|`` between them. That is the mismatch the objective charges
    an arm for *not* attenuating, per ``k`` band — the pressure the trained
    arms are responding to, available before any training and independent of it.

    ``compensate_scale`` divides the label back by ``label_scale``. Both readings
    are needed: a constant gain is a much larger *raw* displacement than the
    staircase (0.542 % of 6.4 kHz is 35 Hz, 4.5 bins, against the staircase's
    8 Hz), but it is also the one the model can absorb. Uncompensated says what
    an arm faces if it never learns the bias; compensated isolates the staircase,
    which is what ``B - S`` and ``C - S`` measure.
    """
    out: dict[str, dict[str, float]] = {}
    for mode in ARMS:
        ds = _dataset(mode, n_samples=n_windows, split=split, label_scale=label_scale)
        k_max = ds.profile.a_k.shape[0]
        band: dict[str, list[float]] = {name: [] for name in K_BANDS}
        for i in range(n_windows):
            ext, sl = ds._traj(i)
            rps_true = ext[sl]
            label = ds.label_for(ext)[sl]
            if compensate_scale and mode != "exact":
                label = label / label_scale
            true = ds.render(rps_true, np.random.default_rng(100 + i))
            fake = ds.render(label, np.random.default_rng(200 + i))
            s_t, s_p = _logspec(true, 2048, 512), _logspec(fake, 2048, 512)
            centres = np.arange(s_t.shape[1]) * 512 + 1024
            rf = rps_true[np.clip(centres, 0, rps_true.shape[-1] - 1)]
            for name, ks in K_BANDS.items():
                diffs = []
                for k in ks:
                    if k > k_max:
                        continue
                    bins = np.rint(k * rf * 2048 / ds.sample_rate).astype(int)
                    ok = (bins > 0) & (bins < s_t.shape[0])
                    if ok.any():
                        t_idx = np.arange(len(rf))[ok]
                        diffs.append(np.abs(s_t[bins[ok], t_idx] - s_p[bins[ok], t_idx]))
                if diffs:
                    band[name].append(float(np.mean(np.concatenate(diffs))))
        out[mode] = {n: float(np.mean(v)) if v else float("nan") for n, v in band.items()}
    return out


def self_test(ds: Any, *, f0_grid: tuple[float, ...], dur_s: float, tol_db: float = 0.05) -> int:
    """Push the TRUE comb through readout 1 and require it to read back flat.

    The readout has to survive its own floor subtraction, its band width and its
    scale estimate before any arm's number means anything. Rendering the target
    itself and asking for 0 dB at every ``k`` is the cheapest way to know it
    does; it lands within 0.01 dB.
    """
    sr = ds.sample_rate
    a_k = np.asarray(ds.profile.a_k, dtype=np.float64) * ds.gain
    ref = (a_k**2) / 2.0
    rng = np.random.default_rng(0)
    per_k: dict[int, list[float]] = {}
    n_t = int(round(dur_s * sr))
    for f0 in f0_grid:
        power, freqs = _spectrum(ds.render(np.full(n_t, f0), rng), sr)
        df = float(freqs[1] - freqs[0])
        s = _frequency_scale(power, freqs, f0, 12, 0.015)
        for k in range(1, a_k.shape[0] + 1):
            center = s * k * f0
            if center >= sr / 2.0 * 0.98 or ref[k - 1] <= 0.0:
                continue
            p, _ = _band_power(power, freqs, center, max(5.0 * df, 0.0015 * center))
            per_k.setdefault(k, []).append(10.0 * np.log10(max(p, 1e-30) / ref[k - 1]))
    worst = max(abs(float(np.mean(v))) for v in per_k.values())
    print(f"self-test: worst per-k bias {worst:.4f} dB over k=1..{max(per_k)} (tol {tol_db})")
    return 0 if worst <= tol_db else 1


def _band_mean(delta_db: dict[int, float]) -> dict[str, float]:
    out = {}
    for name, ks in K_BANDS.items():
        vals = [delta_db[k] for k in ks if k in delta_db]
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--prefix", default="p7_labelsens_", help="experiment-name prefix")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/gen_label_sensitivity")
    ap.add_argument(
        "--f0-grid",
        default="76,80,84,88",
        help=(
            "constant rev/s for the line readout. Each must make k*f0*dur integral AND "
            "lie inside the training marginal (measured p5-p95 = 75.6-89.9 rev/s): a grid "
            "point outside it measures extrapolation, not the learned line amplitude."
        ),
    )
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--scale-k-max", type=int, default=12)
    ap.add_argument("--scale-tol", type=float, default=0.015)
    ap.add_argument("--track-windows", type=int, default=24)
    ap.add_argument("--split", default="eval", help="held-out trajectory stream")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="verify the readout reads the TRUE comb back flat, then exit",
    )
    ap.add_argument(
        "--pressure",
        action="store_true",
        help="print the model-free per-k loss pressure of each arm's label, then exit",
    )
    ap.add_argument("--label-scale", type=float, default=0.99458)
    args = ap.parse_args()

    f0_self = tuple(float(x) for x in args.f0_grid.split(","))
    if args.pressure:
        for tag, comp in (("raw", False), ("scale-compensated", True)):
            print(f"\nloss pressure, {tag} (mean |delta log-mag| dB on the harmonic tracks):")
            table = loss_pressure(
                n_windows=args.track_windows,
                label_scale=args.label_scale,
                split=args.split,
                compensate_scale=comp,
            )
            for mode, bands in table.items():
                print(f"  {mode:16s} " + "  ".join(f"{k}={v:6.2f}" for k, v in bands.items()))
        return 0
    if args.self_test:
        return self_test(
            _dataset("exact", n_samples=1, split=args.split), f0_grid=f0_self, dur_s=args.dur
        )

    import zoo

    arms = [a for a in args.arms.split(",") if a]
    f0_grid = tuple(float(x) for x in args.f0_grid.split(","))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for arm in arms:
        name = f"{args.prefix}{arm}"
        print(f"[{arm}] loading {name}@{args.ckpt} ...", flush=True)
        fm = zoo.load(name, ckpt=args.ckpt, device=args.device)
        ds = _dataset(ARM_LABEL_MODE[arm], n_samples=max(args.track_windows, 1), split=args.split)
        line = line_readout(
            fm,
            ds,
            f0_grid=f0_grid,
            dur_s=args.dur,
            scale_k_max=args.scale_k_max,
            scale_tol=args.scale_tol,
        )
        track = track_readout(fm, ds, n_windows=args.track_windows)
        results[arm] = {
            "experiment": name,
            "freq_scale": line["freq_scale"],
            "delta_db": {str(k): v for k, v in line["delta_db"].items()},
            "spread_hz": {str(k): v for k, v in line["spread_hz"].items()},
            "band_delta_db": _band_mean(line["delta_db"]),
            "band_track_db": track,
        }
        print(f"  freq_scale={line['freq_scale']:.6f}  bands={results[arm]['band_delta_db']}")
        print(f"  track     ={track}")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    ks = sorted({int(k) for r in results.values() for k in r["delta_db"]})
    header = ["k"] + [f"{a}_delta_db" for a in arms] + [f"{a}_spread_hz" for a in arms]
    rows = [",".join(header)]
    for k in ks:
        cells = [str(k)]
        cells += [f"{results[a]['delta_db'].get(str(k), float('nan')):.4f}" for a in arms]
        cells += [f"{results[a]['spread_hz'].get(str(k), float('nan')):.4f}" for a in arms]
        rows.append(",".join(cells))
    (out_dir / "per_k.csv").write_text("\n".join(rows) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arm in arms:
            d = results[arm]["delta_db"]
            xs = sorted(int(k) for k in d)
            ax.plot(xs, [d[str(k)] for k in xs], marker="o", ms=3, lw=1.2, label=arm)
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("harmonic index k")
        ax.set_ylabel(r"line power error $\Delta$ (dB, learned $-$ true)")
        ax.set_title("Generator label sensitivity: per-harmonic underfit")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "per_k.png", dpi=150)
    except Exception as exc:  # pragma: no cover - plotting is a convenience
        print(f"plot skipped: {exc}")

    print(f"wrote {out_dir}/summary.json, per_k.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
