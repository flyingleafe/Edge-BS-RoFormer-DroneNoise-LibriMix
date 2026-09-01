#!/usr/bin/env python
"""Slot method against the trained regressor, on both synthetic families.

Two rows, one per synthetic family, and three columns:

  1. the clip's spectrogram, 0-1.6 kHz, so the comb the two methods read is on
     the slide next to what each made of it
  2. the GATHER + SALIENCE + SLOT decoder, untrained (`head_mode="classical"`)
  3. the best rotor-speed REGRESSOR trained on that same family

Rotor trajectories come from `comb_bench`'s DREGON-calibrated OU draw (the
default since 2026-09-01), so the dotted truth carries real telemetry's
frame-to-frame jitter rather than the two smooth sinusoids the benchmark used
before. Both rows are the SAME seed and therefore the same rotor trajectory:
`comb_bench_stochastic.stoch_comb_clip` takes its tracks from
`comb_bench.comb_clip` itself rather than re-drawing them, so the two families
share their labels exactly and the only difference between the rows is how the
lines are rendered. All four prediction panels are drawn on one shared speed
axis and carry the same dotted truth.

Configuration is the campaign's, not the defaults of the deployed real-data
decoder: `SlotCombNet(head_mode="classical", n_iter=1, k_max=32)`, which is what
`scripts/comb_slots_eval.py` runs and what the geomeans in
`docs/experiments/comb-slot-crf.md` were measured with. `k_max=200` (the
real-data setting) was measured here too and is worse on the stochastic family
(3.46 against 1.33 on this clip), so it is not what the slide should show.

    PYTHONPATH=src python writing/slides/2026-08-31_supervisor-update/make_assets_slots.py
"""

from __future__ import annotations

import itertools
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
import torch  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from data_processing.comb_bench import comb_clip  # noqa: E402
from data_processing.comb_bench_stochastic import stoch_comb_clip  # noqa: E402
from data_processing.frames import make_recording_frame  # noqa: E402
from losses.pit import align_rps_to_gt  # noqa: E402
from plots.timeframe.renderers import ROTOR_COLORS  # noqa: E402

SR = 16_000
HOP = 512
N_FFT = 4096

# The `typical` cell of `comb_bench.REGIMES`, and a seed from the frozen
# validation range the campaign scores on (`seed = 1000 + k`, k < 8).
CELL = {"centre": 75.0, "spread": 11.0, "excursion": 1.5}
SEED = 1000
F_MAX_HZ = 1600.0

# The regressors. `stoch_long_scv2` is the one the stochastic-family reference
# numbers were measured on. On the static family `m3abl_comb_scv2_s1` (the
# comb stage-1 arm) is measurably the better of the two comb-trained candidates
# — see the cell means printed by this script.
STATIC_REG = "m3abl_comb_scv2_s1"
STOCH_REG = "stoch_long_scv2"

# Cell means of the `typical` cell over the eight validation seeds, PIT-RMSE in
# rev/s. Read off the campaign's own result directories, which are the source of
# every number on the slide: results/ou_static + results/ou_static_reg (static)
# and results/ou_stoch_coh + results/ou_stoch_reg (coherent). The slot entry is
# `slots1`, which is this script's `n_iter=1` configuration.
#
# These REPLACE the pre-2026-09-01 values (static 0.037 / 2.781 / 6.146,
# coherent 1.040 / 5.527), which were measured on the two-sinusoid trajectory
# draw that `comb_bench` has since retired for being 21x too smooth.
CELL_MEANS = {
    "static": {"slots": 2.754, STATIC_REG: 3.079, "comb_floor_deep": 4.979},
    "coherent": {"slots": 8.974, STOCH_REG: 6.723, "stoch_s1id_scv2": 4.956},
}
CELL_MEANS_SOURCE = {
    "static": ["results/ou_static", "results/ou_static_reg"],
    "coherent": ["results/ou_stoch_coh", "results/ou_stoch_reg"],
}

STYLE = {
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.0,
    "legend.fontsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "pdf.compression": 9,
}


# ---------------------------------------------------------------------------
# Data and predictions
# ---------------------------------------------------------------------------


def clip(family: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(audio, rps)`` for one benchmark cell of either family."""
    if family == "static":
        audio, rps, _ = comb_clip(seed=seed, **CELL)
    else:
        audio, rps, _ = stoch_comb_clip(seed=seed, line_mode="coherent", **CELL)
    return np.asarray(audio, dtype=np.float64), np.asarray(rps, dtype=np.float64)


def bench_frame(audio: np.ndarray, rps: np.ndarray) -> td.Frame:
    """The Frame a zoo `FrameModel` consumes, exactly as `comb_slots_eval` builds it."""
    t_audio = np.arange(len(audio)) / SR
    rps_full = np.stack([np.interp(t_audio, np.arange(rps.shape[1]) * HOP / SR, r) for r in rps])
    return make_recording_frame(
        {
            "mixture": td.uniform(
                np.ascontiguousarray(audio[None].astype(np.float32)),
                SR,
                dims=("mic", "time"),
                t_start=0.0,
            ),
            "rps": td.events(
                t_audio,
                np.ascontiguousarray(rps_full.astype(np.float32)),
                dims=("rotor", "time"),
                t_start=0.0,
            ),
        },
        meta={"recording_id": "bench"},
    )


def pit_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """The benchmark's own metric: RMSE under the best rotor permutation."""
    return min(
        float(np.sqrt(((pred[list(p)] - gt) ** 2).mean()))
        for p in itertools.permutations(range(pred.shape[0]))
    )


def slot_predict(net, audio: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net.decode(torch.tensor(audio, dtype=torch.float32)[None])[0].numpy()


def zoo_predict(model, audio: np.ndarray, rps: np.ndarray) -> np.ndarray:
    return np.asarray(model(bench_frame(audio, rps))["rps_pred"].data, dtype=np.float64)


def aligned(pred: np.ndarray, rps: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``(pred_aligned, truth, pit_rmse)`` on the prediction's frame grid."""
    width = min(pred.shape[1], rps.shape[1])
    pred, truth = pred[:, :width], rps[:, :width]
    return align_rps_to_gt(pred, truth), truth, pit_rmse(pred, truth)


def spectrogram_db(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(db, freqs)`` of a 4096-point STFT, cut at `F_MAX_HZ`."""
    spec = torch.stft(
        torch.tensor(audio, dtype=torch.float32),
        n_fft=N_FFT,
        hop_length=HOP,
        window=torch.hann_window(N_FFT),
        center=True,
        return_complex=True,
    )
    power = (spec.real**2 + spec.imag**2).numpy()
    freqs = np.arange(power.shape[0]) * SR / N_FFT
    keep = freqs <= F_MAX_HZ
    db = 10.0 * np.log10(power[keep] + 1e-12)
    return db, freqs[keep]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_tracks(ax, t, pred, truth, rmse: float, title: str, ylim, show_x: bool) -> None:
    # The truth is drawn WIDE and the prediction NARROW on top of it. On the
    # static slot panel the two coincide to 0.02 rev/s, and a truth of equal
    # width would vanish under the prediction and read as "only four lines".
    for r in range(truth.shape[0]):
        ax.plot(t, truth[r], ":", color=ROTOR_COLORS[r], lw=3.0, alpha=0.55)
    for r in range(pred.shape[0]):
        ax.plot(t, pred[r], "-", color=ROTOR_COLORS[r], lw=1.4, alpha=0.98)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=6)
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_x:
        ax.set_xlabel("time (s)")
    else:
        ax.tick_params(labelbottom=False)
    ax.text(
        0.985,
        0.03,
        f"PIT-RMSE {rmse:.3f} rev/s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.95, "pad": 3.0},
    )


def draw_spectrogram(ax, db, freqs, dur_s: float, title: str, show_x: bool) -> None:
    # Per-panel contrast. The stochastic family has a loud random floor, so one
    # shared dB window renders it as flat noise; percentiles keep both legible.
    hi = float(np.percentile(db, 99.85))
    lo = float(np.percentile(db, 55.0))
    ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=(0.0, dur_s, float(freqs[0]), float(freqs[-1])),
        cmap="magma",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    ax.set_title(title, pad=6)
    ax.set_ylabel("frequency (Hz)")
    if show_x:
        ax.set_xlabel("time (s)")
    else:
        ax.tick_params(labelbottom=False)


def build_figure(rows: list[dict[str, Any]], *, width=11.0, height=6.5):
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[0.86, 1.0, 1.0],
        hspace=0.26,
        wspace=0.24,
        left=0.065,
        right=0.990,
        top=0.935,
        bottom=0.125,
    )
    lo = min(min(c["truth"].min(), c["pred"].min()) for r in rows for c in r["cols"])
    hi = max(max(c["truth"].max(), c["pred"].max()) for r in rows for c in r["cols"])
    span = max(hi - lo, 1.0)
    ylim = (lo - 0.16 * span, hi + 0.08 * span)

    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        ax = fig.add_subplot(gs[i, 0])
        draw_spectrogram(ax, row["db"], row["freqs"], row["dur_s"], row["title"], last)
        for j, col in enumerate(row["cols"]):
            axt = fig.add_subplot(gs[i, j + 1])
            t = np.linspace(0.0, row["dur_s"], col["pred"].shape[1])
            draw_tracks(axt, t, col["pred"], col["truth"], col["rmse"], col["title"], ylim, last)
            if j == 0:
                axt.set_ylabel("rotor speed (rev/s)")
            else:
                axt.tick_params(labelleft=False)

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
        bbox_to_anchor=(0.5, -0.005),
    )
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def survey(seeds: range) -> dict[str, dict[str, float]]:
    """Cell means over the validation seeds, for the caption and for CELL_MEANS."""
    import zoo
    from models.comb_slots import SlotCombNet

    net = SlotCombNet(head_mode="classical", n_iter=1, use_checkpoint=False).eval()
    regs = {
        "static": ["m3abl_comb_scv2_s1", "comb_floor_deep"],
        "coherent": [STOCH_REG],
    }
    models = {n: zoo.load(n, ckpt="best", device="cpu") for v in regs.values() for n in v}
    out: dict[str, dict[str, list[float]]] = {}
    for family in ("static", "coherent"):
        acc: dict[str, list[float]] = {"slots": []}
        for seed in seeds:
            audio, rps = clip(family, seed)
            acc["slots"].append(aligned(slot_predict(net, audio), rps)[2])
            for name in regs[family]:
                acc.setdefault(name, []).append(
                    aligned(zoo_predict(models[name], audio, rps), rps)[2]
                )
        out[family] = {k: float(np.mean(v)) for k, v in acc.items()}
        print(f"  {family}: " + "  ".join(f"{k} {v:.3f}" for k, v in out[family].items()))
    return out


def main(argv: list[str]) -> int:
    plt.rcParams.update(STYLE)
    out_dir = HERE / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "--survey" in argv:
        print("Cell means over seeds 1000-1007, `typical` cell, PIT-RMSE rev/s:")
        survey(range(1000, 1008))
        return 0

    import zoo
    from models.comb_slots import SlotCombNet

    net = SlotCombNet(head_mode="classical", n_iter=1, use_checkpoint=False).eval()
    print(f"SlotCombNet trained parameters: {sum(p.numel() for p in net.parameters())}")

    rows, record = [], {}
    for family, reg_name, label in (
        ("static", STATIC_REG, "STATIC comb"),
        ("coherent", STOCH_REG, "STOCHASTIC comb (coherent)"),
    ):
        audio, rps = clip(family, SEED)
        db, freqs = spectrogram_db(audio)
        model = zoo.load(reg_name, ckpt="best", device="cpu")
        slot_pred, slot_truth, slot_rmse = aligned(slot_predict(net, audio), rps)
        reg_pred, reg_truth, reg_rmse = aligned(zoo_predict(model, audio, rps), rps)
        print(f"  {family}: slots {slot_rmse:.3f}   {reg_name} {reg_rmse:.3f}")
        rows.append(
            {
                "title": f"{label} — 0–{F_MAX_HZ / 1000:.1f} kHz",
                "db": db,
                "freqs": freqs,
                "dur_s": len(audio) / SR,
                "cols": [
                    {
                        "title": "gather + salience + slots (0 params)",
                        "pred": slot_pred,
                        "truth": slot_truth,
                        "rmse": slot_rmse,
                    },
                    {
                        "title": f"regressor {reg_name}",
                        "pred": reg_pred,
                        "truth": reg_truth,
                        "rmse": reg_rmse,
                    },
                ],
            }
        )
        record[family] = {
            "seed": SEED,
            "cell": CELL,
            "slots_pit_rmse": slot_rmse,
            "regressor": reg_name,
            "regressor_ckpt": "best",
            "regressor_pit_rmse": reg_rmse,
            "cell_means_over_seeds_1000_1007": CELL_MEANS[family],
            "cell_means_source": CELL_MEANS_SOURCE[family],
        }

    fig = build_figure(rows)
    fig.savefig(out_dir / "slots_vs_regressor.pdf")
    fig.savefig(out_dir / "slots_vs_regressor.png")
    plt.close(fig)
    print(f"  wrote {out_dir}/slots_vs_regressor.{{pdf,png}}")

    record["config"] = {
        "slot_net": "SlotCombNet(head_mode='classical', n_iter=1, k_max=32)",
        "trained_parameters": 0,
        "data": "comb_bench.comb_clip / comb_bench_stochastic.stoch_comb_clip",
        "trajectory": "ou (DREGON-calibrated); sinusoid draw retired 2026-09-01",
    }
    (out_dir / "slots_vs_regressor.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"  wrote {out_dir}/slots_vs_regressor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
