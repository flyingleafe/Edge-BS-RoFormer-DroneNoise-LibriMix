r"""Train + validate the faithful JASA GP rotor-noise model on ``jasa-flyovers``.

Trains on ``V in {6, 8, 10} m/s`` (aft region) and validates on the held-out
``V in {7, 9}`` exactly as the paper (Sec. II C).  Writes, under ``--out``:

  - ``best.pt``            the trained :class:`~.jasa_gp.JasaGPModel`
  - ``eval_metrics.json``  per-mic held-out metrics (corr, residual, loudness dB)
  - ``eval_V{7,9}.png``    time-domain + FFT overlays (paper Figs. 5-8 style)
  - ``loudness.png``       predicted vs true loudness across the aft grid (Fig. 10)

Run locally (``python -m src.experiments.gp_rotor_noise.train_jasa_gp``) or via
omnirun on an external CPU box (the fit is tiny; no GPU needed)::

    omnirun submit --backend apocrita-cpu --gpus 0 --time 30m --yes -- \
        uv run python -m src.experiments.gp_rotor_noise.train_jasa_gp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.gp_rotor_noise import jasa_gp as J

EVAL_MICS = [(-30, 0), (-50, 0), (-70, 0), (-90, 0), (-140, 0), (-30, 20), (-30, 50), (-90, 50)]


def _loudness_db(x: np.ndarray) -> float:
    """RMS sound-pressure level [dB re 20 uPa] (a fast loudness proxy)."""
    return float(20.0 * np.log10(np.sqrt(np.mean(x**2)) / 20e-6 + 1e-12))


def _true_aligned_tonal(model: J.JasaGPModel, fl: dict, mic_xy, v: float) -> np.ndarray:
    """Ground-truth tonal at a mic, de-Dopplerized + phase-aligned like training."""
    mics = fl["mics"]
    i = int(((mics[:, 0] - mic_xy[0]) ** 2 + (mics[:, 1] - mic_xy[1]) ** 2).argmin())
    dd, _ = J.dedoppler(fl["tonal"][i], mic_xy, v)
    return J.phase_align(dd, model.target_phase, model.cfg.bpf)


def evaluate(model: J.JasaGPModel, flyovers: dict, out: Path) -> list[dict]:
    rows: list[dict] = []
    for v in J.TEST_SPEEDS:
        fl = flyovers[round(v)]
        fig, axs = plt.subplots(len(EVAL_MICS), 2, figsize=(13, 2.1 * len(EVAL_MICS)))
        for r, (x, y) in enumerate(EVAL_MICS):
            yt = _true_aligned_tonal(model, fl, (x, y), v)
            yp = model.synthesize(x, y, float(v), duration=1.0, rps=None, broadband="none")
            n = min(len(yt), len(yp))
            yt, yp = yt[:n], yp[:n]
            corr = float(np.corrcoef(yt, yp)[0, 1])
            resid = float(((yt - yp) ** 2).sum() / ((yt**2).sum() + 1e-12))
            d_loud = _loudness_db(yp) - _loudness_db(yt)
            rows.append(
                {
                    "V": v,
                    "x": x,
                    "y": y,
                    "corr": round(corr, 3),
                    "resid": round(resid, 3),
                    "dLoud_dB": round(d_loud, 2),
                }
            )
            t = np.arange(n) / J.FS
            sl = (t >= 0.15) & (t <= 0.25)
            axs[r, 0].plot(t[sl], yt[sl], "k", lw=0.8, label="true")
            axs[r, 0].plot(t[sl], yp[sl], "r", lw=0.8, label="GP")
            axs[r, 0].set_title(
                f"V={v} ({x},{y})  corr={corr:.2f}  dL={d_loud:+.1f} dB", fontsize=8
            )
            fr = np.fft.rfftfreq(n, 1 / J.FS)
            st = np.abs(np.fft.rfft(yt * np.hanning(n)))
            sp = np.abs(np.fft.rfft(yp * np.hanning(n)))
            axs[r, 1].plot(fr, st, "k", lw=0.7, label="true")
            axs[r, 1].plot(fr, sp, "r", lw=0.7, label="GP")
            axs[r, 1].set_xlim(0, 500)
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"eval_V{round(v)}.png", dpi=110)
        plt.close(fig)

    # loudness-vs-x plot (Fig. 10 style) at y=0, across the aft grid
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in J.TEST_SPEEDS:
        fl = flyovers[round(v)]
        xs = np.arange(-140, -20, 10)
        lt, lp = [], []
        for x in xs:
            yt = _true_aligned_tonal(model, fl, (x, 0), v)
            yp = model.synthesize(int(x), 0, float(v), duration=1.0, broadband="none")
            lt.append(_loudness_db(yt))
            lp.append(_loudness_db(yp[: len(yt)]))
        ax.plot(xs, lt, "s--", label=f"true V={round(v)}", alpha=0.6)
        ax.plot(xs, lp, "o-", label=f"GP V={round(v)}")
    ax.set_xlabel("x [m] (y=0)")
    ax.set_ylabel("tonal SPL [dB]")
    ax.legend(fontsize=8)
    ax.set_title("Held-out loudness: GP vs CONA ground truth")
    fig.tight_layout()
    fig.savefig(out / "loudness.png", dpi=110)
    plt.close(fig)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/jasa_gp")
    p.add_argument("--n_harm", type=int, default=24)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--no_eval", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    flyovers = J.load_flyovers()
    print(f"[train] loaded speeds {sorted(flyovers)}")
    cfg = J.JasaGPConfig(n_harm=args.n_harm, iters=args.iters, verbose=True)
    model = J.JasaGPModel(cfg)
    t0 = time.time()
    model.fit(flyovers)
    print(f"[train] fit in {time.time() - t0:.1f} s")
    model.save(out / "best.pt")
    print(f"[train] saved -> {out / 'best.pt'}")

    if not args.no_eval:
        rows = evaluate(model, flyovers, out)
        (out / "eval_metrics.json").write_text(json.dumps(rows, indent=2))
        corr = float(np.mean([r["corr"] for r in rows]))
        dloud = float(np.mean([abs(r["dLoud_dB"]) for r in rows]))
        resid = float(np.mean([r["resid"] for r in rows]))
        print(
            f"[eval] held-out mean corr={corr:.3f}  mean|dLoud|={dloud:.2f} dB  mean resid={resid:.3f}"
        )
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
