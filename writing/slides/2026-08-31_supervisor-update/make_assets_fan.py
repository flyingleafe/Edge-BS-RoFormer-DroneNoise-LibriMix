#!/usr/bin/env python
"""The fixed-fan figure: rotor-speed regressors emit an evenly-spaced fan.

Two panels side by side. Each panel draws the four PREDICTED rotor tracks
(solid) over the four TRUE tracks (dotted), one color for each rotor, after the
global PIT permutation that evaluation itself uses
(``losses.pit.align_rps_to_gt``). Under each panel a strip plots the per-frame
ROTOR SPREAD (max minus min of the four speeds) for the truth and for the
prediction, which is the claim of the figure stated as one line each.

A 0-4 kHz spectrogram strip was tried in this slot and dropped: at slide size
neither clip shows a legible comb, so the strip cost height and said nothing.

LEFT   ``stoch_s1id_scv2`` @ ``last`` on the stochastic-comb benchmark
       (``data_processing.comb_bench_stochastic.stoch_comb_clip``) at a LARGE
       rotor spread. This is the exact model whose fixed fan was measured in
       ``docs/experiments/synthetic-solvability-limits.md``.
RIGHT  ``r4hb_scv2`` @ ``best`` (the project's best real-trained regressor,
       2.67 rev/s PIT MAE) on one cruise clip of the frozen real split
       ``dload:DREGON-LM-V4-michaels-valid-full``, channel 0.

A second file, ``fan_counterexample.pdf``, carries the honest counter-example:
the SAME real model on a Michael's FLY124 cruise clip, where its predicted
spread does follow the true spread.

Layout and helpers follow ``writing/papers/2026-08_wrapup/make_figures.py``.

    PYTHONPATH=src python writing/slides/2026-08-31_supervisor-update/make_assets_fan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tdseries as td  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from data_processing.comb_bench_stochastic import stoch_comb_clip  # noqa: E402
from data_processing.frame_datasets import DregonLMFrameDataset  # noqa: E402
from data_processing.frames import audio_series, rps_series  # noqa: E402
from losses.pit import align_rps_to_gt  # noqa: E402
from plots.timeframe.renderers import ROTOR_COLORS  # noqa: E402

SR = 16_000
N_FFT = 2048
HOP = 512
REAL_DATA = "dload:DREGON-LM-V4-michaels-valid-full"

# The synthetic clip. `spread` is the rotor separation the benchmark builds in;
# 42.75 rev/s is the top spread bucket of the measurement in
# docs/experiments/synthetic-solvability-limits.md. The seed was chosen from a
# 16-seed scan as the one whose PREDICTED MEAN is closest to the true mean, so
# the panel isolates the fan and is not dominated by a centre offset — every
# seed in that scan gave a predicted spread of 9.4 to 12.8 rev/s.
SYNTH_SEED = 115
SYNTH_SPREAD = 42.75
SYNTH_DUR_S = 8.0

REAL_CLIP = 20  # DREGON free-flight_nosource_room1, 8 s of pure cruise
COUNTER_CLIP = 27  # michaels_FLY124 cruise, where the same model DOES track

STYLE = {
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "pdf.compression": 9,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def synthetic_frame(seed: int, spread: float, dur_s: float) -> td.Frame:
    """One stochastic-comb clip as the Frame an RPS model consumes."""
    audio, rps, _ft = stoch_comb_clip(seed=seed, spread=spread, dur_s=dur_s, sr=SR, hop=HOP)
    return td.Frame(
        {
            "mixture": audio_series(audio[None, :].astype(np.float32), SR),
            "rps": rps_series(np.asarray(rps, dtype=np.float32), sample_rate=SR, hop_length=HOP),
            "meta": td.Frame({"sample_id": int(seed), "task": "rps_prediction"}),
        }
    )


def real_frame(index: int, channel: int = 0) -> td.Frame:
    dataset = DregonLMFrameDataset(
        data_dir=REAL_DATA, n_fft=N_FFT, hop_length=HOP, sample_rate=SR, channel=channel
    )
    return dataset[index]


def audio_of(frame: td.Frame) -> td.Series:
    for name in ("mixture", "audio"):
        if name in frame:
            series = frame[name]
            if isinstance(series, td.Series):
                return series
    raise KeyError(f"no waveform in frame (have {list(frame.entries)})")


def predict(experiment: str, ckpt: str, frame: td.Frame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(aligned_prediction, truth)``, both ``(4, F)`` on one grid."""
    import zoo

    model = zoo.load(experiment, ckpt=ckpt, device="cpu")
    pred = np.asarray(model(frame)["rps_pred"].data, dtype=np.float64)
    truth = np.asarray(frame["rps"].data, dtype=np.float64)
    width = min(pred.shape[1], truth.shape[1])
    pred, truth = pred[:, :width], truth[:, :width]
    return align_rps_to_gt(pred, truth), truth


def spread_stats(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Per-frame spread = max minus min over the four rotors."""
    ts = truth.max(0) - truth.min(0)
    ps = pred.max(0) - pred.min(0)
    return {
        "true_spread_mean": float(ts.mean()),
        "true_spread_min": float(ts.min()),
        "true_spread_max": float(ts.max()),
        "true_spread_std": float(ts.std()),
        "pred_spread_mean": float(ps.mean()),
        "pred_spread_min": float(ps.min()),
        "pred_spread_max": float(ps.max()),
        "pred_spread_std": float(ps.std()),
        "spread_pearson_r": float(np.corrcoef(ts, ps)[0, 1]) if ts.std() > 0 else float("nan"),
        "pit_mae": float(np.abs(pred - truth).mean()),
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_tracks(
    ax: Any, t: np.ndarray, pred: np.ndarray, truth: np.ndarray, stats: dict[str, float]
) -> None:
    for r in range(truth.shape[0]):
        ax.plot(t, truth[r], ":", color=ROTOR_COLORS[r], lw=1.7, alpha=0.95)
    for r in range(pred.shape[0]):
        ax.plot(t, pred[r], "-", color=ROTOR_COLORS[r], lw=1.9, alpha=0.95)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotor speed (rev/s)")
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.015,
        0.975,
        f"true spread {stats['true_spread_min']:.1f}–{stats['true_spread_max']:.1f} rev/s\n"
        f"predicted spread {stats['pred_spread_min']:.1f}–"
        f"{stats['pred_spread_max']:.1f} rev/s    ·    PIT MAE {stats['pit_mae']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.95, "pad": 3.0},
    )


def panel_ylim(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    lo = min(float(pred.min()), float(truth.min()))
    hi = max(float(pred.max()), float(truth.max()))
    span = max(hi - lo, 1.0)
    # headroom for the statistics box, which must never cover a track
    return lo - 0.08 * span, hi + 0.40 * span


def draw_spread(ax, t: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> None:
    """The claim as one line each: per-frame spread, truth against prediction."""
    ax.plot(t, truth.max(0) - truth.min(0), ":", color="0.15", lw=2.0, label="true")
    ax.plot(t, pred.max(0) - pred.min(0), "-", color="0.15", lw=2.0, label="predicted")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spread\n(rev/s)", labelpad=2)
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lo = min(0.0, float((pred.max(0) - pred.min(0)).min()))
    hi = max(float((truth.max(0) - truth.min(0)).max()), float((pred.max(0) - pred.min(0)).max()))
    ax.set_ylim(lo - 0.08 * max(hi - lo, 1.0), hi + 0.18 * max(hi - lo, 1.0))


def build_figure(columns: list[dict[str, Any]], *, width: float = 11.0, height: float = 4.6):
    """One figure: a trajectory panel over a rotor-spread strip, for each column."""
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(
        2,
        len(columns),
        figure=fig,
        height_ratios=[2.35, 0.9],
        hspace=0.16,
        wspace=0.20,
        left=0.062,
        right=0.988,
        top=0.905,
        bottom=0.175,
    )
    for col, spec in enumerate(columns):
        pred, truth, audio = spec["pred"], spec["truth"], spec["audio"]
        t0 = float(audio.t_start)
        t1 = t0 + float(np.asarray(audio.data).shape[-1]) / SR
        t = np.linspace(t0, t1, pred.shape[1])

        ax = fig.add_subplot(gs[0, col])
        draw_tracks(ax, t, pred, truth, spec["stats"])
        ax.set_ylim(*panel_ylim(pred, truth))
        ax.set_title(spec["title"], pad=7)
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

        ax_sp = fig.add_subplot(gs[1, col], sharex=ax)
        draw_spread(ax_sp, t, pred, truth)

    handles = [
        Line2D([0], [0], color=ROTOR_COLORS[r], lw=2.2, label=f"rotor {r + 1}") for r in range(4)
    ]
    handles += [
        Line2D([0], [0], color="0.15", lw=2.2, ls="-", label="predicted"),
        Line2D([0], [0], color="0.15", lw=2.2, ls=":", label="true"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, -0.008),
    )
    return fig


def column(title: str, frame: td.Frame, experiment: str, ckpt: str) -> dict[str, Any]:
    pred, truth = predict(experiment, ckpt, frame)
    stats = spread_stats(pred, truth)
    return {
        "title": title,
        "pred": pred,
        "truth": truth,
        "audio": audio_of(frame),
        "stats": stats,
        "experiment": experiment,
        "ckpt": ckpt,
    }


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {out_dir}/{stem}.{{pdf,png}}")


def report(name: str, spec: dict[str, Any]) -> None:
    s = spec["stats"]
    print(
        f"  {name}: {spec['experiment']}@{spec['ckpt']}  PIT MAE {s['pit_mae']:.2f}\n"
        f"      true spread  {s['true_spread_min']:6.2f} to {s['true_spread_max']:6.2f} "
        f"(mean {s['true_spread_mean']:.2f}, sd {s['true_spread_std']:.2f})\n"
        f"      pred spread  {s['pred_spread_min']:6.2f} to {s['pred_spread_max']:6.2f} "
        f"(mean {s['pred_spread_mean']:.2f}, sd {s['pred_spread_std']:.2f})   "
        f"r(true, pred) = {s['spread_pearson_r']:+.3f}"
    )


def main() -> int:
    plt.rcParams.update(STYLE)
    out_dir = HERE / "assets"

    print("Rendering fan_panels ...")
    left = column(
        f"stoch_s1id_scv2 on synthetic (stochastic comb, spread {SYNTH_SPREAD:.0f} rev/s)",
        synthetic_frame(SYNTH_SEED, SYNTH_SPREAD, SYNTH_DUR_S),
        "stoch_s1id_scv2",
        "last",
    )
    right = column(
        f"r4hb_scv2 on real (DREGON valid, cruise clip {REAL_CLIP})",
        real_frame(REAL_CLIP),
        "r4hb_scv2",
        "best",
    )
    report("left  (synthetic)", left)
    report("right (real)", right)
    save(build_figure([left, right]), out_dir, "fan_panels")

    print("Rendering fan_counterexample ...")
    counter = column(
        f"r4hb_scv2 on real (Michael's FLY124, cruise clip {COUNTER_CLIP})",
        real_frame(COUNTER_CLIP),
        "r4hb_scv2",
        "best",
    )
    report("counter-example", counter)
    save(build_figure([right, counter]), out_dir, "fan_counterexample")

    record = {
        "fan_panels": {
            "left": {
                "experiment": left["experiment"],
                "ckpt": left["ckpt"],
                "data": "comb_bench_stochastic.stoch_comb_clip",
                "seed": SYNTH_SEED,
                "spread": SYNTH_SPREAD,
                "duration_s": SYNTH_DUR_S,
                **left["stats"],
            },
            "right": {
                "experiment": right["experiment"],
                "ckpt": right["ckpt"],
                "data": REAL_DATA,
                "clip": REAL_CLIP,
                "channel": 0,
                **right["stats"],
            },
        },
        "fan_counterexample_right": {
            "experiment": counter["experiment"],
            "ckpt": counter["ckpt"],
            "data": REAL_DATA,
            "clip": COUNTER_CLIP,
            "channel": 0,
            **counter["stats"],
        },
    }
    (out_dir / "fan_panels.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"  wrote {out_dir}/fan_panels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
