"""Two cue probes: WHICH acoustic cue does a rotor-speed predictor read?

A validation MAE says how well a model reads rotor speed, never what it reads
it FROM. These two probes deform one cue at a time and watch the prediction
move, over any list of `zoo:` checkpoints, on the CPU.

**freq** -- the frequency-scaling probe. Resampling the waveform by ``1/alpha``
multiplies every frequency by ``alpha``, so a model that reads harmonic
POSITIONS must scale its predicted speeds by exactly ``alpha`` (slope 1.00),
while a model that reads a timbre or a level fingerprint does not move
(slope 0). Reported on six DREGON cruise clips of the frozen real split as the
mean relative change against the ``alpha = 1`` prediction, with two least
squares slopes through the origin of percent shift against percent alpha: over
the whole range, and inside +-4 %. This is the operation
``data_processing.noise_augmentations.freq_scale`` applies in training, without
the label rescaling -- the point of the probe is that the label stays put.

    python scripts/rps_cue_probe.py freq --exp r4hb_scv2 hb_scv2_mag_nogate

**cutoff** -- the harmonic-cutoff probe. Low-pass the audio so that only the
harmonic orders at or below ``k_cut`` of the rotor comb survive (the cutoff is
``k_cut`` x the clip's maximum labelled rotor speed). A model that reads the
low orders a real rotor exposes keeps its score; a model that learned the high
orders only synthesis renders collapses, and typically onto the half rate.
Reported per ``k_cut`` as the PIT MAE and the fraction of rotor-frames read at
the true rate and at half the true rate.

    python scripts/rps_cue_probe.py cutoff --exp salv2_hppnet_stoch_nomix

Both probes cache one JSON per experiment under
``results/rps_probe/<probe>/<exp>.json`` and skip an experiment that already
has one, so a killed run resumes. ``--force`` recomputes. A non-default
setting gets its own suffix (``<exp>.fir-n16.json``) so two settings of one
probe never overwrite each other, and a cache whose settings do not match the
request is recomputed rather than read.

Reference numbers. The frequency curves of figure 3 of the wrap-up paper
(``writing/papers/2026-08_wrapup/figures/freq_probe_full.json``, which this
probe's JSON output matches key for key) and probe P2b of
``docs/rps-tracking-architecture-candidates.md`` (PIT MAE 2.05 / 16.2 / 39.5 /
44.8 at ``k_cut`` 80 / 40 / 20 / 10 for ``salv2_hppnet_stoch_nomix``). P2b used
a 1025-tap FIR low-pass over 16 frames; ``--lowpass fir --n-frames 16``
reproduces it, the defaults are the STFT mask over every flight frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SR, HOP, N_FFT = 16000, 512, 2048
PERMS = list(permutations(range(4)))

ALPHAS = (0.70, 0.775, 0.85, 0.925, 0.96, 0.98, 1.0, 1.02, 1.04, 1.10, 1.15, 1.22, 1.30)
K_CUTS = (80, 40, 20, 10)
CRUISE_MEAN = 45.0  # every frame's rotor mean above this -> a cruise clip
FLIGHT_MEAN = 1.0  # some rotor mean above this -> a flight frame (the cutoff probe)
LOCAL = 4.001  # the +-4 % window of the local slope
SALIENCE_GRID = (0.0, 150.0, 300)  # the salv2 / r4_l4 rate grid
N_LAYERS = 4
RATIO_FLOOR = 1.0  # rotor-frames whose truth is at or below this carry no ratio


# ─── Shared ───────────────────────────────────────────────────────────────────


def rate_grid() -> np.ndarray:
    from models.multif0.utils import linear_freq_grid

    fmin, fmax, bins = SALIENCE_GRID
    return np.asarray(linear_freq_grid(fmin, fmax, int(bins)), dtype=np.float64)


def speeds(fm: Any, frame: Any, grid: np.ndarray) -> np.ndarray:
    """``(R, T)`` rev/s from one model on one Frame, regressor or salience port.

    A regressor's ``rps_pred`` is taken as is; a salience port's layers are read
    by the peak + parabola readout its own ``rps_mae`` metric uses. This is
    ``experiments.rps_bench.Readout`` without the metric, which needs a label
    the frequency probe's deformed frames do not carry.
    """
    import torch
    import torch.nn.functional as F

    from metrics._common import get_array
    from metrics.salience_layers import peak_readout

    with torch.no_grad():
        out = fm(frame)
    if "rps_pred" in out:
        return np.asarray(get_array(out, "rps_pred"), dtype=np.float64)
    logits = torch.as_tensor(get_array(out, "salience")).unsqueeze(0)  # (1, R*G, T)
    _, fg, n_t = logits.shape
    if fg != N_LAYERS * len(grid):
        # A shared-map salience baseline (LateDeep, Basic Pitch, the published
        # HarmoF0 / HPPNet; block S levels L0-L1) emits one map on its own
        # grid: read it by its own threshold + Hungarian decode, as rps_dump does.
        key = "mixture" if "mixture" in frame else "audio"
        audio = torch.as_tensor(np.asarray(get_array(frame, key), dtype=np.float32))
        if audio.ndim == 1:
            audio = audio[None]
        with torch.no_grad():
            return np.asarray(
                fm.model.predict_rps(audio.to(fm.device))[0].detach().cpu().numpy(),
                dtype=np.float64,
            )
    layers = logits.reshape(1, N_LAYERS, len(grid), n_t).double()
    return np.asarray(peak_readout(F.logsigmoid(layers), grid)[0].numpy(), dtype=np.float64)


def pit(pred: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray]:
    """PIT MAE plus the label permuted onto ``pred``'s rotor rows."""
    from experiments.rps_bench import resample_like_metric

    g = resample_like_metric(np.asarray(gt, dtype=np.float64), pred.shape[-1])
    cost = np.abs(pred[:, None] - g[None, :]).mean(-1)
    best = min(PERMS, key=lambda p: sum(cost[k, p[k]] for k in range(4)))
    aligned = np.stack([g[best[k]] for k in range(4)])
    return float(np.abs(pred - aligned).mean()), aligned


# ─── Probe 1: frequency scaling ───────────────────────────────────────────────


def cruise_clips(frames: list, n_ch: int, want: int) -> list[int]:
    """The first ``want`` clips of the real part that are at cruise throughout.

    The test is on the mean over the four rotors of EVERY frame, so a clip that
    dips at any point is out. On ``part("real")`` it selects clips 2, 3, 4, 5,
    6, 9 -- the six DREGON cruise clips of the wrap-up paper's figure 3.
    """
    from metrics._common import get_array

    out = []
    for c in range(len(frames) // n_ch):
        gt = np.asarray(get_array(frames[c * n_ch], "rps"), dtype=np.float64)
        if (gt.mean(axis=0) >= CRUISE_MEAN).all():
            out.append(c)
    return out[:want]


def freq_probe(fm: Any, frames: list, clips: list[int], n_ch: int, alphas: list[float]) -> dict:
    """Mean predicted speed per alpha, and the response relative to alpha = 1.

    The waveform is resampled by ``1/alpha`` (``resample_poly`` up 1000, down
    ``1000 alpha``), which multiplies every frequency by ``alpha``, then
    truncated to the span the largest alpha still covers, so that every alpha is
    read over the same number of frames.
    """
    import tdseries as td
    from scipy.signal import resample_poly

    from data_processing.frames import audio_series

    grid = rate_grid()
    keep_ratio = max(alphas)
    means = []
    for a in alphas:
        vals = []
        for c in clips:
            audio = np.asarray(frames[c * n_ch]["mixture"].data, dtype=np.float64)
            if audio.ndim > 1:
                audio = audio[0]
            scaled = resample_poly(audio, 1000, int(round(a * 1000)))
            keep = int(len(audio) / keep_ratio)
            f = td.Frame({"mixture": audio_series(scaled[:keep][None].astype(np.float32), SR)})
            vals.append(float(speeds(fm, f, grid).mean()))
        means.append(float(np.mean(vals)))
        print(f"    alpha {a:5.3f}  mean speed {means[-1]:8.4f}", flush=True)
    base = means[alphas.index(1.0)]
    resp = [100.0 * (m - base) / base for m in means]
    x = 100.0 * (np.asarray(alphas) - 1.0)
    y = np.asarray(resp)
    near = np.abs(x) <= LOCAL
    return {
        "alphas": alphas,
        "clips": clips,
        "mean_speed": means,
        "response": resp,
        "slope_full": float((x * y).sum() / (x * x).sum()),
        "slope_local": float((x[near] * y[near]).sum() / (x[near] * x[near]).sum()),
    }


# ─── Probe 2: harmonic cutoff ─────────────────────────────────────────────────


def lowpass_stft(x: np.ndarray, f_cut: float) -> np.ndarray:
    """Zero every STFT bin above ``f_cut`` on the models' own 2048/512 grid."""
    import torch

    t = torch.as_tensor(x, dtype=torch.float32)
    win = torch.hann_window(N_FFT)
    spec = torch.stft(t, N_FFT, HOP, window=win, return_complex=True, center=True)
    freqs = torch.fft.rfftfreq(N_FFT, 1.0 / SR)
    spec[freqs > f_cut] = 0.0
    wave = torch.istft(spec, N_FFT, HOP, window=win, center=True, length=len(x))
    return np.asarray(wave.numpy(), dtype=np.float64)


def lowpass_fir(x: np.ndarray, f_cut: float) -> np.ndarray:
    """Zero-phase 1025-tap Hamming FIR, the low-pass of probe P2b."""
    from scipy.signal import filtfilt, firwin

    return np.asarray(filtfilt(firwin(1025, f_cut, fs=SR, window="hamming"), [1.0], x))


def flight_frames(frames: list, want: int) -> list[int]:
    """Frames whose label is not all zero; ``want`` of them, evenly spread."""
    from metrics._common import get_array

    keep = [
        i
        for i, f in enumerate(frames)
        if (np.asarray(get_array(f, "rps"), dtype=np.float64).mean(axis=1) >= FLIGHT_MEAN).any()
    ]
    if not want or want >= len(keep):
        return keep
    return sorted({keep[j] for j in np.linspace(0, len(keep) - 1, want).round().astype(int)})


def cutoff_probe(fm: Any, frames: list, sel: list[int], k_cuts: list[int], how: str) -> dict:
    """PIT MAE and the true / half rate fractions per harmonic cutoff order.

    A cutoff above Nyquist filters nothing and is the control (``k_cut`` 80 on
    an 85 rev/s rotor asks for 6.8 kHz, which is inside the band, but a slower
    clip's 80th order is not). The rate ratio is read on rotor-frames whose
    truth is above ``RATIO_FLOOR``, so a stopped rotor does not divide by zero.
    """
    import tdseries as td

    from data_processing.frames import audio_series
    from metrics._common import get_array

    grid = rate_grid()
    rows = []
    for i in sel:
        f = frames[i]
        gt = np.asarray(get_array(f, "rps"), dtype=np.float64)
        x = np.asarray(f["mixture"].data, dtype=np.float64).ravel()
        r_max = float(gt.max())
        for k in k_cuts:
            f_cut = k * r_max
            if f_cut >= 0.98 * (SR / 2):  # above Nyquist: the unfiltered control
                xf, applied = x, False
            else:
                xf = lowpass_stft(x, f_cut) if how == "stft" else lowpass_fir(x, f_cut)
                applied = True
            g = td.Frame(
                {
                    "mixture": audio_series(xf[None, :].astype(np.float32), SR),
                    "rps": f["rps"],
                    "meta": f["meta"],
                }
            )
            pred = speeds(fm, g, grid)
            mae, aligned = pit(pred, gt)
            m = aligned > RATIO_FLOOR
            r = pred[m] / aligned[m]
            rows.append(
                {
                    "frame": i,
                    "k_cut": k,
                    "filtered": applied,
                    "f_cut": f_cut if applied else None,
                    "r_max": r_max,
                    "mae": mae,
                    # the task's tolerance: within 5 % of the rate itself
                    "frac_true": float((np.abs(r - 1.0) <= 0.05).mean()),
                    "frac_half": float((np.abs(r - 0.5) <= 0.025).mean()),
                    # ... and the wider tolerance the P2b numbers were read at
                    "frac_true_p2b": float((np.abs(r - 1.0) <= 0.10).mean()),
                    "frac_half_p2b": float((np.abs(r - 0.5) <= 0.05).mean()),
                    "median_ratio": float(np.median(r)),
                }
            )
        print(
            f"    frame {i:3d}  " + "  ".join(f"k{r['k_cut']} {r['mae']:6.2f}" for r in rows[-4:]),
            flush=True,
        )
    return {
        "k_cuts": k_cuts,
        "lowpass": how,
        "frames": sel,
        "rows": rows,
        "summary": summarize(rows, k_cuts),
    }


def summarize(rows: list[dict], k_cuts: list[int]) -> dict:
    """Mean over frames of every per-frame number, one entry per cutoff order.

    ``n_filtered`` counts the frames the low-pass really touched: a cutoff
    above Nyquist is a pass-through, and on a fast rotor even ``k_cut`` 80
    lands above the band, so the control row is not the same set of frames as
    the others. Kept separate from the compute step so that the aggregate can
    be rebuilt from a cached run's rows after the summary schema changes.
    """
    keys = ("mae", "frac_true", "frac_half", "frac_true_p2b", "frac_half_p2b")
    out = {}
    for k in k_cuts:
        sub = [r for r in rows if r["k_cut"] == k]
        out[str(k)] = {
            "n_frames": len(sub),
            "n_filtered": sum(1 for r in sub if r["filtered"]),
            **{key: float(np.mean([r[key] for r in sub])) for key in keys},
            "median_ratio": float(np.median([r["median_ratio"] for r in sub])),
        }
    return out


# ─── Drivers ──────────────────────────────────────────────────────────────────


def run_freq(a: argparse.Namespace, out_dir: Path) -> None:
    import zoo
    from experiments import rps_bench as rb

    tag = variant(a)
    frames = rb.part(a.part)
    n_ch = 1 + max(int(f["meta"]["channel"]) for f in frames) if "meta" in frames[0] else 1
    clips = [int(c) for c in a.clips] if a.clips else cruise_clips(frames, n_ch, a.n_clips)
    print(f"{a.part}: {len(frames)} frames, {n_ch} mics; cruise clips {clips}", flush=True)

    res = {}
    for exp in a.exp:
        hit = load_cached(out_dir, exp, tag, a.force, {"clips": clips})
        if hit is None:
            t0 = time.time()
            print(f"  {exp}", flush=True)
            fm = zoo.load(exp, ckpt=a.ckpt, device="cpu")
            hit = freq_probe(fm, frames, clips, n_ch, list(ALPHAS))
            hit["experiment"], hit["seconds"] = exp, round(time.time() - t0, 1)
            (out_dir / f"{exp}{tag}.json").write_text(json.dumps(hit, indent=1))
            del fm
        res[exp] = hit
        print(
            f"  {exp:34s} slope full {hit['slope_full']:6.2f}  local "
            f"{hit['slope_local']:6.2f}  ({hit.get('seconds', 0)} s)",
            flush=True,
        )

    agg = {
        "alphas": list(ALPHAS),
        "clips": clips,
        "response": {e: r["response"] for e, r in res.items()},
    }
    (out_dir / f"summary{tag}.json").write_text(json.dumps(agg, indent=1))
    lines = ["experiment,alpha,mean_speed,response_pct,slope_full,slope_local"]
    for e, r in res.items():
        for al, ms, rp in zip(r["alphas"], r["mean_speed"], r["response"], strict=True):
            lines.append(f"{e},{al},{ms:.6f},{rp:.6f},{r['slope_full']:.6f},{r['slope_local']:.6f}")
    (out_dir / f"summary{tag}.csv").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir}/summary{tag}.json and summary{tag}.csv", flush=True)


def run_cutoff(a: argparse.Namespace, out_dir: Path) -> None:
    import zoo
    from experiments import rps_bench as rb

    tag = variant(a)
    frames = rb.part(a.part)
    sel = flight_frames(frames, a.n_frames)
    n_flight = len(flight_frames(frames, 0))
    print(
        f"{a.part}: {len(frames)} frames, {n_flight} of them in flight, "
        f"{len(sel)} read, low-pass {a.lowpass}",
        flush=True,
    )

    res = {}
    for exp in a.exp:
        hit = load_cached(out_dir, exp, tag, a.force, {"lowpass": a.lowpass, "frames": sel})
        if hit is None:
            t0 = time.time()
            print(f"  {exp}", flush=True)
            fm = zoo.load(exp, ckpt=a.ckpt, device="cpu")
            hit = cutoff_probe(fm, frames, sel, list(K_CUTS), a.lowpass)
            hit["experiment"], hit["seconds"] = exp, round(time.time() - t0, 1)
            (out_dir / f"{exp}{tag}.json").write_text(json.dumps(hit, indent=1))
            del fm
        # rebuilt from the rows, so a cache written by an older summary schema
        # still aggregates
        res[exp] = summarize(hit["rows"], list(K_CUTS))

    lines = [
        "experiment,k_cut,n_frames,n_filtered,mae,frac_true,frac_half,"
        "frac_true_p2b,frac_half_p2b,median_ratio"
    ]
    print(
        f"\n{'experiment':34s} {'k':>3s} {'n':>4s} {'mae':>8s} {'true':>6s} "
        f"{'half':>6s} {'true10':>7s} {'half5':>6s} {'medrat':>7s}"
    )
    for e, r in res.items():
        for k in map(str, K_CUTS):
            s = r[k]
            lines.append(
                f"{e},{k},{s['n_frames']},{s['n_filtered']},{s['mae']:.6f},{s['frac_true']:.6f},"
                f"{s['frac_half']:.6f},{s['frac_true_p2b']:.6f},{s['frac_half_p2b']:.6f},"
                f"{s['median_ratio']:.6f}"
            )
            print(
                f"{e:34s} {k:>3s} {s['n_frames']:4d} {s['mae']:8.2f} {s['frac_true']:6.2f} "
                f"{s['frac_half']:6.2f} {s['frac_true_p2b']:7.2f} {s['frac_half_p2b']:6.2f} "
                f"{s['median_ratio']:7.3f}"
            )
    (out_dir / f"summary{tag}.csv").write_text("\n".join(lines) + "\n")
    (out_dir / f"summary{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out_dir}/summary{tag}.json and summary{tag}.csv", flush=True)


def variant(a: argparse.Namespace) -> str:
    """The suffix that keeps two settings of one probe from sharing a cache file."""
    if a.probe == "freq":
        return (
            "" if a.clips is None and a.n_clips == 6 else f".c{'-'.join(map(str, a.clips or []))}"
        )
    return "" if a.lowpass == "stft" and not a.n_frames else f".{a.lowpass}-n{a.n_frames or 'all'}"


def load_cached(out_dir: Path, exp: str, tag: str, force: bool, expect: dict) -> dict | None:
    """The cached result of ``exp``, unless ``--force`` or its settings differ."""
    f = out_dir / f"{exp}{tag}.json"
    if not f.exists() or force:
        return None
    hit = json.loads(f.read_text())
    if any(hit.get(k) != v for k, v in expect.items()):
        print(f"  {exp:34s} cache is a different setting; recomputing", flush=True)
        return None
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("probe", choices=("freq", "cutoff"))
    ap.add_argument("--exp", nargs="+", required=True, help="zoo experiment names")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default="results/rps_probe", help="cache root")
    ap.add_argument("--part", default=None, help="rps_bench part (freq: real, cutoff: stoch)")
    ap.add_argument("--force", action="store_true", help="recompute cached experiments")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--n-clips", type=int, default=6, help="freq: cruise clips to read")
    ap.add_argument("--clips", nargs="*", default=None, help="freq: clip indices, overriding")
    ap.add_argument("--n-frames", type=int, default=0, help="cutoff: frames (0 = every flight)")
    ap.add_argument("--lowpass", choices=("stft", "fir"), default="stft", help="cutoff: how")
    a = ap.parse_args()

    import torch

    torch.set_num_threads(a.threads)
    a.part = a.part or ("real" if a.probe == "freq" else "stoch")
    out_dir = Path(a.out) / a.probe
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_freq if a.probe == "freq" else run_cutoff)(a, out_dir)


if __name__ == "__main__":
    main()
