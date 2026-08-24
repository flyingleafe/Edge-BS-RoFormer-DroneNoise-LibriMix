#!/usr/bin/env python3
"""Coarse audio-vs-flight-log alignment for the un-calibrated Michael's recordings.

FLY103 / FLY108 (raw tree ``new-drone-noises``) have a full DatCon telemetry
CSV but no alignment constants at all, and their logs run ~2x longer than the
audio, so the audio is an unknown sub-window of the log. The fine calibration
(``fit_new.py``, the VK reconstruction residual) has a capture range of a few
hundred milliseconds, so it needs a seed. This script produces it.

**The measure.** Resample the audio to 16 kHz mono and take one STFT. Inside
the analysis band, z-score each frame's log-magnitude over frequency, so the
score asks "how far above the local spectrum does this bin stand", not "how
loud is this frame". For a candidate log-clock offset ``X`` (the log time of
the audio's first sample) the telemetry predicts, for every audio frame and
every rotor, a comb at ``k * BLADES * rps_r``; the score is the mean z-score
over those bins, over the cruise frames only (mean rps > 45 rev/s). A wrong
offset puts the bins on spectrum floor and scores ~0; the right one puts them
on the comb teeth.

**Three passes.**

  1. ``coarse``  the whole feasible offset range (every offset for which the
     audio fits inside the log), 20 ms step, with a +-1 bin tolerance so a
     rev/s scale error of up to ~0.5 % cannot suppress the peak.
  2. ``fine``    a joint (offset, rev/s scale) grid around the coarse winner,
     exact bins. The scale axis is a free by-product: it is a first estimate
     of the ``MICHAELS_RPS_SCALE`` constant, ~30x coarser than the VK fit.
  3. ``thirds``  the fine offset scan repeated on each third of the recording.
     A clock DILATION shows up as a linear drift of the per-third optimum;
     the implied ``b`` (s/s) seeds ``time_dilation`` for the fine fit.

Outputs per recording, under ``--out`` (default ``results/michaels_coarse``):
``<rid>.json`` (curves + verdict) and ``<rid>.png`` (score vs offset, full
range + zoom + the per-third fits).

Cheap enough to run on a laptop: about 1 minute per recording.

    python scripts/michaels_calib/coarse_align.py
    python scripts/michaels_calib/coarse_align.py --rid FLY103 --step 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from data_processing.sources.michaels import (  # noqa: E402
    MICHAELS_TEST_FILES,
    read_motor_speeds,
    resolve_test_raw_root,
)

SR = 16000
N_FFT = 2048
HOP = 512
#: DJI Matrice 100 propellers: the comb's strongest lines are the blade-passing
#: orders ``k * BLADES * rps``.
BLADES = 2
N_HARMONICS = 10
BAND_HZ = (100.0, 4000.0)
#: frames below this mean rotor speed are ground/warm-up — no stable comb.
CRUISE_RPS = 45.0

#: recording id -> (wav_rel, csv_rel), from the sources registry (never retyped).
TEST_FILES: dict[str, tuple[str, str]] = {
    Path(csv_rel).stem: (wav_rel, csv_rel) for wav_rel, csv_rel, _off, _dil in MICHAELS_TEST_FILES
}


# ────────────────────────────────────────────────────────── the measure
def spectrum(wav_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """``(z (F_band, T), frame times (T,), bin width Hz)`` of one recording."""
    import librosa as lr

    y, _ = lr.load(str(wav_path), sr=SR, mono=True)
    spec = np.abs(lr.stft(np.asarray(y, dtype=np.float32), n_fft=N_FFT, hop_length=HOP))
    log_spec = np.log(spec + 1e-10)
    df = SR / N_FFT
    lo, hi = int(round(BAND_HZ[0] / df)), int(round(BAND_HZ[1] / df))
    band = log_spec[lo:hi]
    z = (band - band.mean(axis=0, keepdims=True)) / (band.std(axis=0, keepdims=True) + 1e-9)
    t_frames = np.arange(z.shape[1], dtype=np.float64) * HOP / SR
    return np.ascontiguousarray(z.astype(np.float32)), t_frames, df


def _rps_at(
    offsets: np.ndarray,
    t_frames: np.ndarray,
    t_log: np.ndarray,
    rps_log: np.ndarray,
    dilation: float,
) -> np.ndarray:
    """``(n_offsets, 4, T)`` telemetry interpolated onto the audio frame grid.

    The alignment model is ``t_log = time_offset + t_audio / time_dilation``,
    the inverse of the loader's ``(t_log - time_offset) * time_dilation``.
    """
    query = (offsets[:, None] + t_frames[None, :] / dilation).ravel()
    out = np.stack([np.interp(query, t_log, row) for row in rps_log])
    return out.reshape(rps_log.shape[0], len(offsets), len(t_frames)).transpose(1, 0, 2)


def score_offsets(
    z: np.ndarray,
    t_frames: np.ndarray,
    df: float,
    t_log: np.ndarray,
    rps_log: np.ndarray,
    offsets: np.ndarray,
    *,
    rps_scale: float = 1.0,
    dilation: float = 1.0,
    tolerance: int = 0,
    chunk: int = 64,
) -> np.ndarray:
    """Mean comb z-score for every candidate offset (``(n_offsets,)``).

    ``tolerance`` widens each predicted bin to a +-``tolerance``-bin maximum,
    which absorbs a small rev/s scale error at the cost of lag resolution.
    """
    n_bins, n_frames = z.shape
    ks = (np.arange(1, N_HARMONICS + 1) * BLADES).astype(np.float64)
    lo_bin = int(round(BAND_HZ[0] / df))
    frame_idx = np.arange(n_frames)
    out = np.empty(len(offsets), dtype=np.float64)
    for start in range(0, len(offsets), chunk):
        block = offsets[start : start + chunk]
        rps = _rps_at(block, t_frames, t_log, rps_log, dilation) * rps_scale  # (B, 4, T)
        cruise = rps.mean(axis=1) > CRUISE_RPS  # (B, T)
        freqs = rps[:, :, None, :] * ks[None, None, :, None]  # (B, 4, K, T)
        bins = np.rint(freqs / df).astype(np.int64) - lo_bin
        valid = (bins >= tolerance) & (bins < n_bins - tolerance)
        bins = np.clip(bins, tolerance, n_bins - 1 - tolerance)
        vals = z[bins, frame_idx]
        if tolerance:
            for shift in range(1, tolerance + 1):
                vals = np.maximum(
                    vals, np.maximum(z[bins - shift, frame_idx], z[bins + shift, frame_idx])
                )
        keep = valid & cruise[:, None, None, :]
        totals = np.where(keep, vals, 0.0).sum(axis=(1, 2, 3))
        counts = keep.sum(axis=(1, 2, 3))
        out[start : start + chunk] = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return out


def parabolic_peak(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Sub-grid maximum via the parabola through the 3 points around the max."""
    i = int(np.nanargmax(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), float(ys[i])
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    den = y0 - 2 * y1 + y2
    if den >= 0:
        return float(xs[i]), float(y1)
    h = float(xs[i] - xs[i - 1])
    return float(xs[i] - 0.5 * h * (y2 - y0) / den), float(y1 - 0.125 * (y2 - y0) ** 2 / den)


# ────────────────────────────────────────────────────────── verdict
def peak_verdict(offsets: np.ndarray, scores: np.ndarray, exclude_s: float = 2.0) -> dict[str, Any]:
    """Is the maximum unambiguous? Compare it with the best rival >= 2 s away."""
    best_i = int(np.nanargmax(scores))
    best_x, best_y = offsets[best_i], scores[best_i]
    far = np.abs(offsets - best_x) >= exclude_s
    rival_y = float(np.nanmax(scores[far])) if far.any() else float("nan")
    rival_x = float(offsets[far][int(np.nanargmax(scores[far]))]) if far.any() else float("nan")
    med = float(np.nanmedian(scores))
    mad = float(np.nanmedian(np.abs(scores - med))) or 1e-9
    return {
        "best_offset_s": round(float(best_x), 4),
        "best_score": round(float(best_y), 5),
        "runner_up_offset_s": round(rival_x, 4),
        "runner_up_score": round(rival_y, 5),
        "margin_over_runner_up": round(float(best_y) - rival_y, 5),
        "z_over_background": round((float(best_y) - med) / mad, 2),
        "background_median": round(med, 5),
        "unambiguous": bool(float(best_y) - rival_y > 0.25 * (float(best_y) - med)),
    }


def regime_split(t_log: np.ndarray, rps_log: np.ndarray, t0: float, t1: float) -> dict[str, Any]:
    """Fraction of the audio span the telemetry calls ground / warm-up / cruise."""
    inside = (t_log >= t0) & (t_log <= t1)
    mean_rps = rps_log[:, inside].mean(axis=0)
    total = max(len(mean_rps), 1)
    return {
        "n_telemetry_rows": int(total),
        "ground_frac": round(float((mean_rps < 5.0).sum()) / total, 4),
        "warmup_frac": round(float(((mean_rps >= 5.0) & (mean_rps < 45.0)).sum()) / total, 4),
        "cruise_frac": round(float((mean_rps >= 45.0).sum()) / total, 4),
        "cruise_seconds": round(
            float((mean_rps >= 45.0).sum()) * float(np.median(np.diff(t_log))), 2
        ),
        "mean_cruise_rps": (
            round(float(mean_rps[mean_rps >= 45.0].mean()), 3) if (mean_rps >= 45.0).any() else None
        ),
        "max_rps": round(float(mean_rps.max()), 3),
    }


# ────────────────────────────────────────────────────────── driver
def run_one(rid: str, root: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    wav_rel, csv_rel = TEST_FILES[rid]
    wav_path, csv_path = root / wav_rel, root / csv_rel
    t0 = time.time()
    z, t_frames, df = spectrum(wav_path)
    dur = float(t_frames[-1] + HOP / SR)
    t_log, rps_log = read_motor_speeds(csv_path, 1.0)
    good = np.isfinite(rps_log).all(axis=0)
    t_log, rps_log = t_log[good], rps_log[:, good]
    print(f"{rid}: audio {dur:.2f} s, log {t_log[0]:.2f}..{t_log[-1]:.2f} s, {z.shape[1]} frames")

    # ── pass 1: one global scan over the whole feasible offset range ────────
    lo = float(t_log[0]) if args.lo is None else args.lo
    hi = float(t_log[-1] - dur) if args.hi is None else args.hi
    coarse_x = np.round(np.arange(lo, hi + 1e-9, args.step), 6)
    coarse_y = score_offsets(z, t_frames, df, t_log, rps_log, coarse_x, tolerance=1)
    verdict = peak_verdict(coarse_x, coarse_y)
    print(
        f"  coarse: {len(coarse_x)} offsets in [{lo:.2f}, {hi:.2f}] -> "
        f"{verdict['best_offset_s']:+.3f} s  score {verdict['best_score']:.4f}  "
        f"(runner-up {verdict['runner_up_score']:.4f} @ {verdict['runner_up_offset_s']:+.2f} s, "
        f"{verdict['z_over_background']:.0f} MAD over background, "
        f"{'clear' if verdict['unambiguous'] else 'AMBIGUOUS'})"
    )

    # ── pass 2: per-segment peaks AROUND that anchor -> (offset, dilation) ──
    # A recording-wide scan at dilation 1 is a compromise smeared by whatever
    # clock error the pair has (both turn out to have a large one). Each
    # segment is short enough that the drift inside it stays inside a bin, so
    # each has its own sharp peak; the line through those peaks IS the
    # (offset, dilation) pair. The search is bounded to +-``--seg-span`` of the
    # global anchor, or a segment happily locks onto a distant rival.
    anchor = float(verdict["best_offset_s"])
    seg_x = np.round(np.arange(anchor - args.seg_span, anchor + args.seg_span + 1e-9, args.step), 6)
    edges = np.linspace(0.0, dur, args.segments + 1)
    seg_masks = [(t_frames >= edges[i]) & (t_frames < edges[i + 1]) for i in range(args.segments)]
    seg_rows: list[dict[str, Any]] = []
    for i, sel in enumerate(seg_masks):
        if sel.sum() < 50:
            continue
        y = score_offsets(z[:, sel], t_frames[sel], df, t_log, rps_log, seg_x, tolerance=1)
        if not np.isfinite(y).any():
            continue
        verdict_i = peak_verdict(seg_x, y, exclude_s=1.0)
        seg_rows.append(
            {
                "segment": i,
                "t_centre": round(0.5 * float(edges[i] + edges[i + 1]), 3),
                "coarse": verdict_i,
                "best_offset_s": verdict_i["best_offset_s"],
            }
        )
        print(
            f"  seg {i} @ {seg_rows[-1]['t_centre']:6.1f} s -> "
            f"{verdict_i['best_offset_s']:+8.3f} s  score {verdict_i['best_score']:.4f}  "
            f"({verdict_i['z_over_background']:5.1f} MAD, margin "
            f"{verdict_i['margin_over_runner_up']:+.4f}, "
            f"{'clear' if verdict_i['unambiguous'] else 'AMBIGUOUS'})"
        )
    if len(seg_rows) < 2:
        raise SystemExit(f"{rid}: only {len(seg_rows)} usable segments — cannot fit a line")

    def fit_line(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
        """``(time_offset, time_dilation, residual RMS ms)``, outlier-trimmed.

        One pass of trimming: a segment whose residual is more than 3x the
        median absolute residual is dropped and the line refitted, provided at
        least 3 points survive.
        """
        tc = np.array([r["t_centre"] for r in rows], dtype=float)
        ox = np.array([r["best_offset_s"] for r in rows], dtype=float)
        if len(rows) < 2:
            return float(ox.mean()), 1.0, 0.0
        slope, intercept = np.polyfit(tc, ox, 1)
        resid = ox - (slope * tc + intercept)
        mad = float(np.median(np.abs(resid)))
        keep = np.abs(resid) <= max(3.0 * mad, 0.02)
        if 3 <= keep.sum() < len(rows):
            for row, ok in zip(rows, keep, strict=True):
                row["outlier"] = not bool(ok)
            slope, intercept = np.polyfit(tc[keep], ox[keep], 1)
            resid = ox[keep] - (slope * tc[keep] + intercept)
        rms = float(np.sqrt(np.mean(resid**2))) * 1e3
        # X(t) = off + t * (1/dilation - 1)  ->  dilation = 1 / (1 + slope)
        return float(intercept), 1.0 / (1.0 + float(slope)), rms

    off, dil, rms = fit_line(seg_rows)
    print(f"  line 0: offset {off:+.4f} s  dilation {dil:.7f}  resid RMS {rms:.1f} ms")
    fine_scale = 1.0
    for it in range(args.iterations):
        # The dilation now rides inside the query, so the whole recording is
        # modelled by ONE offset: each segment's scan is centred on it and its
        # optimum is a residual. The residual is mapped back onto the
        # dilation-1 parameterisation (X_i = x + t_i * (1/dil - 1)) so the same
        # line fit applies as in pass 1.
        span = args.fine_span if it else max(args.fine_span, 4.0 * rms * 1e-3)
        xs = np.round(np.arange(off - span, off + span + 1e-9, args.fine_step), 6)
        for row in seg_rows:
            sel = seg_masks[int(row["segment"])]
            y = score_offsets(
                z[:, sel],
                t_frames[sel],
                df,
                t_log,
                rps_log,
                xs,
                rps_scale=fine_scale,
                dilation=dil,
            )
            if not np.isfinite(y).any():
                continue
            x_hat, y_hat = parabolic_peak(xs, y)
            row["resid_ms"] = round((x_hat - off) * 1e3, 2)
            row["best_score"] = round(y_hat, 5)
            row["best_offset_s"] = round(x_hat + float(row["t_centre"]) * (1.0 / dil - 1.0), 6)
        off, dil, rms = fit_line(seg_rows)
        print(
            f"  line {it + 1}: offset {off:+.4f} s  dilation {dil:.7f}  "
            f"resid RMS {rms:.1f} ms  (segment residuals "
            f"{[r['resid_ms'] for r in seg_rows]} ms)"
        )

    # ── pass 3: joint (offset, rev/s scale) at the fitted dilation ──────────
    fine_x = np.round(
        np.arange(off - args.fine_span, off + args.fine_span + 1e-9, args.fine_step), 6
    )
    scales = np.round(np.arange(args.scale_lo, args.scale_hi + 1e-9, args.scale_step), 6)
    grid = np.stack(
        [
            score_offsets(
                z, t_frames, df, t_log, rps_log, fine_x, rps_scale=s, dilation=dil, tolerance=0
            )
            for s in scales
        ]
    )  # (n_scales, n_offsets)
    si, oi = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
    fine_off, fine_score = parabolic_peak(fine_x, grid[si])
    fine_scale, _ = parabolic_peak(scales, grid[:, oi])
    # like-for-like with the pass-1 anchor: same +-1 bin tolerance, whole
    # recording, one number. This is the honest "did the dilation help" test.
    fit_score = float(
        score_offsets(
            z,
            t_frames,
            df,
            t_log,
            rps_log,
            np.array([fine_off]),
            rps_scale=fine_scale,
            dilation=dil,
            tolerance=1,
        )[0]
    )
    print(
        f"  fine:   offset {fine_off:+.4f} s  dilation {dil:.7f}  "
        f"rps_scale {fine_scale:.5f}  exact-bin score {fine_score:.4f}"
    )
    print(
        f"  gain:   whole-recording score at the fit {fit_score:.4f} vs "
        f"{verdict['best_score']:.4f} for the best single offset at dilation 1 "
        f"(+{100 * (fit_score / verdict['best_score'] - 1):.1f} %)"
    )
    dilation_seed = {
        "fit_score_tol1": round(fit_score, 5),
        "anchor_score_tol1": round(float(verdict["best_score"]), 5),
        "n_segments": len(seg_rows),
        "time_dilation": round(dil, 9),
        "resid_rms_ms": round(rms, 2),
        "segment_resid_ms": [r.get("resid_ms") for r in seg_rows],
    }

    regimes = regime_split(t_log, rps_log, fine_off, fine_off + dur / dil)
    print(
        f"  regimes over the audio span: ground {regimes['ground_frac']:.1%} / "
        f"warm-up {regimes['warmup_frac']:.1%} / cruise {regimes['cruise_frac']:.1%} "
        f"({regimes['cruise_seconds']:.0f} s, mean {regimes['mean_cruise_rps']} rev/s)"
    )

    result = {
        "rid": rid,
        "wav": str(wav_rel),
        "csv": str(csv_rel),
        "audio_duration_s": round(dur, 4),
        "log_span_s": [round(float(t_log[0]), 3), round(float(t_log[-1]), 3)],
        "stft": {
            "sr": SR,
            "n_fft": N_FFT,
            "hop": HOP,
            "band_hz": list(BAND_HZ),
            "df_hz": round(df, 4),
        },
        "harmonics": {"blades": BLADES, "n": N_HARMONICS, "cruise_rps": CRUISE_RPS},
        "coarse": {
            "lo": lo,
            "hi": hi,
            "step": args.step,
            "tolerance_bins": 1,
            "offsets": [round(float(v), 4) for v in coarse_x],
            "scores": [None if not np.isfinite(v) else round(float(v), 5) for v in coarse_y],
            **verdict,
        },
        "fine": {
            "offsets": [round(float(v), 5) for v in fine_x],
            "scales": [float(s) for s in scales],
            "scores": np.round(np.nan_to_num(grid, nan=-9.0), 5).tolist(),
            "best_offset_s": round(fine_off, 5),
            "best_rps_scale": round(fine_scale, 6),
            "best_score": round(fine_score, 5),
        },
        "segments": seg_rows,
        "dilation_seed": dilation_seed,
        "regimes": regimes,
        "proposed": {
            "time_offset": round(fine_off, 5),
            "time_dilation": round(dil, 9),
            "rps_scale_hint": round(fine_scale, 5),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{rid}.json").write_text(json.dumps(result, indent=1))
    plot(result, out_dir / f"{rid}.png")
    return result


def plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    cx = np.array(result["coarse"]["offsets"], dtype=float)
    cy = np.array([np.nan if v is None else v for v in result["coarse"]["scores"]], dtype=float)
    best = result["fine"]["best_offset_s"]
    axes[0].plot(cx, cy, lw=0.7)
    axes[0].axvline(best, color="crimson", lw=1.0, ls="--")
    axes[0].set_title(f"{result['rid']}: comb score vs log-clock offset")
    axes[0].set_xlabel("time_offset (s)")
    axes[0].set_ylabel("mean comb z-score")

    fx = np.array(result["fine"]["offsets"], dtype=float)
    grid = np.array(result["fine"]["scores"], dtype=float)
    scales = np.array(result["fine"]["scales"], dtype=float)
    im = axes[1].imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=(float(fx[0]), float(fx[-1]), float(scales[0]), float(scales[-1])),
    )
    axes[1].plot([best], [result["fine"]["best_rps_scale"]], "r+", ms=12)
    axes[1].set_title("fine joint (offset, rev/s scale)")
    axes[1].set_xlabel("time_offset (s)")
    axes[1].set_ylabel("rps scale")
    fig.colorbar(im, ax=axes[1])

    seg = result["segments"]
    dil = float(result["proposed"]["time_dilation"])
    if seg:
        tc = np.array([r["t_centre"] for r in seg], dtype=float)
        oy = np.array([r["best_offset_s"] for r in seg], dtype=float)
        axes[2].plot(tc, oy, "o", label="per-segment optimum")
        axes[2].plot(
            tc,
            result["proposed"]["time_offset"] + tc * (1.0 / dil - 1.0),
            "--",
            color="gray",
            label=f"fit, dilation {dil:.6f}",
        )
        axes[2].set_title(
            f"per-segment offset (resid RMS {result['dilation_seed']['resid_rms_ms']} ms)"
        )
        axes[2].legend(fontsize=8)
    axes[2].set_xlabel("segment centre (s)")
    axes[2].set_ylabel("equivalent time_offset at dilation 1 (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rid", action="append", choices=sorted(TEST_FILES), help="default: both")
    ap.add_argument("--root", default=None, help="raw root (default: the new-drone-noises pin)")
    ap.add_argument("--out", default=str(REPO / "results" / "michaels_coarse"))
    ap.add_argument("--step", type=float, default=0.02, help="coarse offset step (s)")
    ap.add_argument("--lo", type=float, default=None, help="coarse offset lower bound (s)")
    ap.add_argument("--hi", type=float, default=None)
    ap.add_argument("--fine-span", type=float, default=0.5, help="fine half-width (s)")
    ap.add_argument("--fine-step", type=float, default=0.005)
    ap.add_argument("--scale-lo", type=float, default=0.994)
    ap.add_argument("--scale-hi", type=float, default=1.012)
    ap.add_argument("--scale-step", type=float, default=0.001)
    ap.add_argument("--segments", type=int, default=5)
    ap.add_argument(
        "--seg-span",
        type=float,
        default=2.5,
        help="per-segment search half-width around the global anchor (s)",
    )
    ap.add_argument("--iterations", type=int, default=2, help="line-refit passes")
    args = ap.parse_args()

    root = resolve_test_raw_root(args.root)
    out_dir = Path(args.out)
    print(f"raw root: {root}")
    results = {rid: run_one(rid, root, out_dir, args) for rid in (args.rid or sorted(TEST_FILES))}
    summary = {
        rid: {
            "proposed": r["proposed"],
            "segments_unambiguous": [bool(s["coarse"]["unambiguous"]) for s in r["segments"]],
            "segment_z_over_background": [s["coarse"]["z_over_background"] for s in r["segments"]],
            "segment_offsets_s": [s["coarse"]["best_offset_s"] for s in r["segments"]],
            "line_resid_rms_ms": r["dilation_seed"]["resid_rms_ms"],
            "regimes": r["regimes"],
        }
        for rid, r in results.items()
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{json.dumps(summary, indent=1)}\n\nwrote {out_dir}")


if __name__ == "__main__":
    main()
