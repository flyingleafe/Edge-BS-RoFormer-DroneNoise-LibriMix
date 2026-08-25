#!/usr/bin/env python
"""Qualitative per-regime comparison figures for the 2026-08 wrap-up paper.

One figure for each validation clip: a 0-4 kHz spectrogram on top, then one
panel for each method. Each method panel draws the four predicted rotor
tracks as solid lines over the dotted ground truth, on a shared time axis
with one color for each rotor.

Layout and taste follow ``writing/reports/2026-05-29_classical-baselines``.
The spectrogram comes from the project renderer
(:func:`plots.timeframe.renderers.make_log_spectrogram_series` plus the
``"audio_spectrogram"`` renderer) — this file draws no time-frequency data
of its own.

Usage
-----
Default clips and methods::

    python writing/papers/2026-08_wrapup/make_figures.py

Other clips, other methods::

    python writing/papers/2026-08_wrapup/make_figures.py \
        --clip cruise:20 --clip zero:36 \
        --method "HB (ours)=zoo:hb_scv2_if" --method "NMF=classical:nmf" \
        --out-dir src/figures

Method sources
--------------
``zoo:<experiment>``     A trained checkpoint, through ``zoo.load``.
``classical:<name>``     A key of ``experiments.classical_rps.predictors.CLASSICAL_TRACKERS``.
``npz:<path>[#<key>]``   A precomputed ``(4, T)`` trajectory on the 2048/512 grid.

Outputs, for each clip, in ``--out-dir``:

- ``qual_<regime>.pdf`` and ``qual_<regime>.png`` — the figure
- ``qual_<regime>.json`` — clip PIT MAE of each method (per-frame Hungarian),
  overall and for each target regime, for the caption
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tdseries as td  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from losses.pit import align_rps_to_gt  # noqa: E402
from plots.timeframe.registry import TrackContext, get_renderer  # noqa: E402
from plots.timeframe.renderers import ROTOR_COLORS, make_log_spectrogram_series  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "dload:DREGON-LM-V4-michaels-valid-full"
SR = 16_000
N_FFT = 2048
HOP = 512
FMAX_HZ = 4000.0
REGIMES = ("zero", "low", "flight")

# regime label -> clip index. Verified on the frozen valid-full split:
#   36  251 zero / 0 low / 0 flight  (stopped rotors, low-frequency rumble)
#    8   87 zero / 59 low / 105 flight  (a stop/start transition)
#   20    0 zero / 0 low / 251 flight  (cruise)
DEFAULT_CLIPS: tuple[tuple[str, int], ...] = (
    ("zero", 36),
    ("transition", 8),
    ("cruise", 20),
)

# The regime each default clip must show, as a predicate on the frame counts.
INTENT: dict[str, Callable[[dict[str, int]], bool]] = {
    "zero": lambda c: c["zero"] > 0.8 * sum(c.values()),
    "transition": lambda c: min(c["zero"] + c["low"], c["flight"]) > 0.1 * sum(c.values()),
    "cruise": lambda c: c["flight"] > 0.8 * sum(c.values()),
}

DEFAULT_METHODS: tuple[tuple[str, str], ...] = (
    ("HB SimpleConvV2 (IF)", "zoo:hb_scv2_if"),
    ("SimpleConvV2 (real only)", "zoo:scv2_fs_v2"),
    ("NMF", "classical:nmf"),
    ("HPS", "classical:hps"),
)

STYLE = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.compression": 9,
}


# ---------------------------------------------------------------------------
# Metrics (the 10-line helpers of results/m3cur_regime_probe/regime_probe.py)
# ---------------------------------------------------------------------------


def frame_groups(t: np.ndarray) -> np.ndarray:
    """Label each frame of the ``(4, F)`` target by its regime."""
    mx, mn = t.max(0), t.mean(0)
    g = np.full(t.shape[1], "low", dtype=object)
    g[mx < 1.0] = "zero"
    g[mn >= 45.0] = "flight"
    return g


def pit_err(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-frame Hungarian match of ``(4, F)`` prediction to target -> ``|err|``."""
    n_frames = t.shape[1]
    out = np.empty((t.shape[0], n_frames))
    for i in range(n_frames):
        cost = np.abs(p[:, None, i] - t[None, :, i])
        ri, ci = linear_sum_assignment(cost)
        out[:, i] = np.abs(p[ri, i] - t[ci, i])
    return out


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

Predictor = Callable[[td.Frame], np.ndarray]


def _audio_series(frame: td.Frame) -> td.Series:
    """Return the clip waveform. Valid-full frames name it ``mixture``."""
    for name in ("mixture", "audio"):
        if name in frame:
            series = frame[name]
            if isinstance(series, td.Series):
                return series
    raise KeyError(f"No waveform entry in frame (have {list(frame.entries)})")


def _zoo_predictor(experiment: str) -> Predictor:
    import zoo

    model = zoo.load(experiment, ckpt="best", device="cpu")

    def predict(frame: td.Frame) -> np.ndarray:
        out = model(frame)
        return np.asarray(out["rps_pred"].data, dtype=np.float64)

    return predict


def _classical_predictor(name: str) -> Predictor:
    from experiments.classical_rps.predictors import CLASSICAL_TRACKERS

    if name not in CLASSICAL_TRACKERS:
        raise KeyError(f"Unknown classical tracker {name!r} (have {sorted(CLASSICAL_TRACKERS)})")
    tracker = CLASSICAL_TRACKERS[name]

    def predict(frame: td.Frame) -> np.ndarray:
        audio = np.asarray(_audio_series(frame).data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[0]
        return np.asarray(tracker(audio), dtype=np.float64)

    return predict


def _npz_predictor(spec: str) -> Predictor:
    path_str, _, key = spec.partition("#")
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path

    def predict(_frame: td.Frame) -> np.ndarray:
        obj = np.load(path)
        if isinstance(obj, np.ndarray):
            arr = obj
        else:
            names = list(obj.files)
            chosen = key or next(
                (k for k in ("rps_pred", "pred", "rps", "traj") if k in names), names[0]
            )
            arr = obj[chosen]
        return np.asarray(arr, dtype=np.float64)

    return predict


def build_predictor(source: str) -> Predictor:
    """Build the predictor named by one ``kind:argument`` source string."""
    kind, _, arg = source.partition(":")
    if kind == "zoo":
        return _zoo_predictor(arg)
    if kind == "classical":
        return _classical_predictor(arg)
    if kind == "npz":
        return _npz_predictor(arg)
    raise ValueError(f"Unknown method source {source!r} (kinds: zoo, classical, npz)")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _draw_spectrogram(ax: Any, audio: td.Series, t_start: float, t_end: float) -> None:
    """Draw the 0-4 kHz log spectrogram through the project renderer."""
    if np.ndim(audio.data) > 1:
        audio = audio.slice[str(audio.dims[0]), 0]
    track = make_log_spectrogram_series(audio, n_fft=N_FFT, hop_length=HOP, fmax=FMAX_HZ)
    ctx = TrackContext(
        ax=ax,
        name="spectrogram",
        t_start=t_start,
        t_end=t_end,
        style={"_hints": track.hints},
    )
    get_renderer("audio_spectrogram")(track.series, ctx)
    # The renderer draws raw dB; clip the color range so the combs stay visible.
    data = np.asarray(track.series.data, dtype=np.float64)
    for mesh in ax.collections:
        mesh.set_clim(float(np.percentile(data, 40.0)), float(np.percentile(data, 99.8)))
        # A vector mesh of a spectrogram is megabytes; rasterize it.
        mesh.set_rasterized(True)
    ax.set_ylabel("kHz")
    ax.set_yticks([0.0, 2000.0, 4000.0])
    ax.set_yticklabels(["0", "2", "4"])
    ax.set_xlim(t_start, t_end)


def _panel(
    ax: Any,
    label: str,
    t_gt: np.ndarray,
    gt: np.ndarray,
    t_pred: np.ndarray,
    pred: np.ndarray,
    ylim: tuple[float, float],
    t_start: float,
    t_end: float,
) -> None:
    for r in range(gt.shape[0]):
        ax.plot(t_gt, gt[r], ":", color=ROTOR_COLORS[r], lw=1.2, alpha=0.7)
    for r in range(min(pred.shape[0], len(ROTOR_COLORS))):
        ax.plot(t_pred, pred[r], "-", color=ROTOR_COLORS[r], lw=1.6, alpha=0.9)
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(*ylim)
    ax.set_ylabel("rev/s")
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.012,
        0.93,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
    )


def render_clip(
    frame: td.Frame,
    methods: list[tuple[str, Predictor]],
    *,
    width: float,
) -> tuple[Any, dict[str, Any]]:
    """Render one clip and return the figure plus its metrics record."""
    audio = _audio_series(frame)
    rps = frame["rps"]
    assert isinstance(rps, td.Series)
    gt = np.asarray(rps.data, dtype=np.float64)
    t_start = float(audio.t_start)
    t_end = t_start + float(np.asarray(audio.data).shape[-1]) / SR
    t_gt = np.linspace(t_start, t_end, gt.shape[1])
    groups = frame_groups(gt)
    counts = {g: int((groups == g).sum()) for g in REGIMES}

    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metrics: dict[str, Any] = {}
    for label, predict in methods:
        raw = predict(frame)
        if raw.ndim != 2:
            raise ValueError(f"{label}: expected a (4, T) trajectory, got {raw.shape}")
        n = raw.shape[1]
        t_pred = np.linspace(t_start, t_end, n)
        gt_on_pred = gt if n == gt.shape[1] else _resample(gt, gt.shape[1], n)
        aligned = align_rps_to_gt(raw, gt_on_pred)
        preds[label] = (t_pred, aligned)

        err = pit_err(raw, gt_on_pred)
        groups_on_pred = groups if n == gt.shape[1] else frame_groups(gt_on_pred)
        row: dict[str, Any] = {"mae": float(err.mean()), "n_frames": int(n)}
        for g in REGIMES:
            sel = groups_on_pred == g
            row[f"mae_{g}"] = float(err[:, sel].mean()) if sel.any() else None
        metrics[label] = row

    lo = min(0.0, float(gt.min()), *(float(p.min()) for _, p in preds.values()))
    hi = max(float(gt.max()), *(float(p.max()) for _, p in preds.values()))
    span = max(hi - lo, 1.0)
    ylim = (lo - 0.06 * span, hi + 0.10 * span)

    n_rows = len(methods) + 1
    height = 1.35 + 1.30 * len(methods) + 0.55
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(width, height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05] + [1.0] * len(methods)},
    )
    _draw_spectrogram(axes[0], audio, t_start, t_end)
    for ax, (label, _) in zip(axes[1:], methods, strict=True):
        t_pred, pred = preds[label]
        _panel(ax, label, t_gt, gt, t_pred, pred, ylim, t_start, t_end)
    axes[-1].set_xlabel("Time (s)")

    handles = [Line2D([0], [0], color=ROTOR_COLORS[r], lw=1.2, label=f"R{r + 1}") for r in range(4)]
    handles.append(Line2D([0], [0], color="#333333", ls=":", lw=1.2, label="truth"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.1,
        bbox_to_anchor=(0.5, -0.004),
    )
    # `hspace` goes after `tight_layout` — set through `gridspec_kw` it makes
    # matplotlib 3.10 call the figure incompatible with tight_layout.
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.subplots_adjust(hspace=0.12)

    record = {
        "clip": None,
        "recording_id": str(frame["meta"]["recording_id"]),
        "regime_counts": counts,
        "duration_s": round(t_end - t_start, 3),
        "methods": metrics,
    }
    return fig, record


def _resample(arr: np.ndarray, n_src: int, n_dst: int) -> np.ndarray:
    """Linear resample of a ``(rotor, n_src)`` track onto ``n_dst`` frames."""
    src = np.linspace(0.0, 1.0, n_src)
    dst = np.linspace(0.0, 1.0, n_dst)
    return np.stack([np.interp(dst, src, row) for row in arr])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_pair(text: str, sep: str, what: str) -> tuple[str, str]:
    label, found, value = text.partition(sep)
    if not found:
        raise argparse.ArgumentTypeError(f"{what} must be '<label>{sep}<value>', got {text!r}")
    return label.strip(), value.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        default=None,
        metavar="REGIME:INDEX",
        help="clip to render (repeatable); default: zero:36 transition:8 cruise:20",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        metavar="LABEL=SOURCE",
        help="method to draw (repeatable); source is zoo:/classical:/npz:",
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "figures"),
        help="output directory (use src/figures to write where the paper reads)",
    )
    parser.add_argument("--width", type=float, default=3.45, help="figure width in inches")
    parser.add_argument("--no-png", action="store_true", help="write the PDF only")
    args = parser.parse_args(argv)

    clips = (
        [(lbl, int(idx)) for lbl, idx in (_parse_pair(c, ":", "--clip") for c in args.clip)]
        if args.clip
        else list(DEFAULT_CLIPS)
    )
    methods_spec = (
        [_parse_pair(m, "=", "--method") for m in args.method]
        if args.method
        else list(DEFAULT_METHODS)
    )

    from data_processing.frame_datasets import DregonLMFrameDataset

    dataset = DregonLMFrameDataset(
        data_dir=args.data_dir,
        n_fft=N_FFT,
        hop_length=HOP,
        sample_rate=SR,
        channel=args.channel,
    )
    print(f"{len(dataset)} clips in {args.data_dir}")

    print("Loading methods ...")
    methods = [(label, build_predictor(source)) for label, source in methods_spec]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    for regime, index in clips:
        frame = dataset[index]
        fig, record = render_clip(frame, methods, width=args.width)
        record["clip"] = index
        record["regime"] = regime
        record["sources"] = dict(methods_spec)

        counts = record["regime_counts"]
        total = sum(counts.values())
        check = INTENT.get(regime)
        verdict = "" if check is None else ("  OK" if check(counts) else "  MISMATCH")
        print(
            f"clip {index:3d} [{regime}] {record['recording_id']}: "
            + " ".join(f"{g}={counts[g]}" for g in REGIMES)
            + f" (of {total}){verdict}"
        )
        for label, row in record["methods"].items():
            per = " ".join(
                f"{g}={row[f'mae_{g}']:.2f}" for g in REGIMES if row[f"mae_{g}"] is not None
            )
            print(f"    {label:26s} MAE {row['mae']:6.2f}   [{per}]")

        pdf = out_dir / f"qual_{regime}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        written = [pdf.name]
        if not args.no_png:
            fig.savefig(out_dir / f"qual_{regime}.png", bbox_inches="tight")
            written.append(f"qual_{regime}.png")
        plt.close(fig)
        (out_dir / f"qual_{regime}.json").write_text(json.dumps(record, indent=1) + "\n")
        written.append(f"qual_{regime}.json")
        print(f"    wrote {out_dir}/{{{', '.join(written)}}}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
