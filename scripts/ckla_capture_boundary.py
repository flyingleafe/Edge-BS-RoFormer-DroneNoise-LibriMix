"""CKLA P0b — empirical capture/lock boundary over (drift-rate × SNR).

Implements the "capture behavior" diagnostic of ``docs/ckla-design.md`` §6
(campaign row P0b in ``docs/experiments/ckla.md``): on synthetic eval clips
with controlled RPS drift, classify each clip as lock/no-lock (per-frame
mean-over-rotors PIT-aligned error < 2 rev/s sustained after a 1 s warmup)
and map the lock fraction over a (drift-rate × interference-SNR) grid — an
empirical capture boundary to compare against K2's collapse point and the
EKF-PLL prediction. If a learned CKLA head shows the same boundary as K2,
the closed-loop hypothesis is refuted quantitatively.

Self-contained synthetic eval — no dataset dependency:

* Noise: the analytic static-comb synthesizer used for training
  (``data_processing.rotor_spectral_model.StaticCombNoisePool``,
  ``kind: static_comb``) with per-clip profile sampling at the same
  real-calibrated ranges as ``conf/online_mix/rps_static_comb_only.yaml``,
  so eval clips are on-distribution for the ``ckla_p0_staticcomb`` arm.
  Single-mic (models take (B, T) mono — the training stream flattens
  channels).
* Drift axis: the OU-mode synthetic RPS generator's calibrated
  ``aggressiveness`` knob (``data_processing.rps_synthesis``, threaded
  through the pool's ``synthetic_intermittent`` excitation).
* Interference: a speech-shaped proxy (numpy port of
  ``src/experiments/kalman_harmonic/phase0.py::synth_speech_proxy``) mixed
  at the target SNR (speech power relative to comb-noise power, matching
  the online-mix ``snr_db`` convention: −20 dB = quiet speech = easy), or
  ``--speech none`` for pure comb noise.
* Metric: per-clip PIT alignment via ``tasks.rps_prediction.align_rps_to_gt``
  (MSE-Hungarian, identical to the vk_valid_comparison protocol), then
  per-frame mean-over-rotors absolute error on the STFT frame grid.
* Rotation ablation (``--ablate-rotation``): for CKLA checkpoints,
  additionally evaluate a copy with the rotation path zeroed (every
  ``ComplexKLALayer``: s ← 0, ω0 ← 0, W_ω weight+bias ← 0) and report the
  per-cell delta — §6 "rotation usage" evidence. Skipped (with a note) for
  non-CKLA models.

Noise/RPS/speech are seeded per (aggressiveness, clip) and shared across the
SNR grid (only the speech scale changes) and across models, so cells differ
only along the axes under test.

Run (smoke, laptop):
  python scripts/ckla_capture_boundary.py \
    --models random:simple_conv_v2_ckla_mag random:simple_conv_v2_transformer \
    --n-clips 2 --aggressiveness 1.0 --snr-db 0 --duration 2.0 --device cpu

Run (real, cluster CPU / free GPU):
  python scripts/ckla_capture_boundary.py \
    --models ckla=r2://ml-data/artifacts/ckla_p0_staticcomb/checkpoints/best.ckpt \
             transformer=r2://ml-data/artifacts/e8_staticcomb_s1_transformer/checkpoints/best.ckpt \
    --ablate-rotation

Outputs (``--out``, default ``results/ckla_capture_boundary/``):
``results.json`` (per-cell per-model mean/median PIT-MAE + lock fraction +
per-clip detail), ``boundary_<model>.png`` lock-fraction heatmaps, and a
summary table on stdout.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
# Pin the repo's src/ ahead of site-packages (same rationale as
# scripts/rps_predictor_vk_eval.py: the editable install points at whatever
# checkout owns .venv, which on worktrees is NOT the job's checkout).
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

HOP = 512  # STFT hop of every registry RPS model (frame = HOP/sr seconds)
N_ROTORS = 4


# ── Model specs ─────────────────────────────────────────────────────────────


def parse_model_specs(items: list[str]) -> list[tuple[str, str, str | None]]:
    """``name=path`` or ``random:name`` → (label, registry_name, ckpt|None).

    Labels are the registry names, deduplicated with a numeric suffix when the
    same registry model is listed more than once.
    """
    specs: list[tuple[str, str, str | None]] = []
    seen: dict[str, int] = {}
    for item in items:
        if item.startswith("random:"):
            name, ckpt = item[len("random:") :], None
        elif "=" in item:
            name, ckpt = item.split("=", 1)
        else:
            raise SystemExit(f"--models entry {item!r}: expected name=path or random:name")
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}_{seen[name]}"
        specs.append((label, name, ckpt))
    return specs


def load_model(name: str, ckpt: str | None, device: str, seed: int):
    """Build a registry model; load a checkpoint (local or r2://) or keep the
    seeded random init (``random:``)."""
    import torch

    from models.registry import build_model

    torch.manual_seed(seed)  # reproducible random-init arms
    model = build_model(name)
    if ckpt is not None:
        path = ckpt
        if path.startswith("r2://"):
            from training.artifacts import resolve_checkpoint_uri

            path = resolve_checkpoint_uri(path, str(_ROOT / ".cache" / "r2_checkpoints"))
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def ablate_rotation(model):
    """Copy with the CKLA rotation path zeroed (design §6): for every
    ``ComplexKLALayer``, s ← 0, ω0 ← 0, W_ω weight+bias ← 0. Returns None if
    the model contains no CKLA layers (non-CKLA models: skip with a note)."""
    import torch

    from models.ckla import ComplexKLALayer

    clone = copy.deepcopy(model)
    n = 0
    with torch.no_grad():
        for mod in clone.modules():
            if isinstance(mod, ComplexKLALayer):
                mod.s.zero_()
                mod.omega0.zero_()
                mod.omega_proj.weight.zero_()
                if mod.omega_proj.bias is not None:
                    mod.omega_proj.bias.zero_()
                n += 1
    return clone if n else None


# ── Synthetic scene ─────────────────────────────────────────────────────────


def build_pool(aggressiveness: float, sr: int, duration_s: float):
    """StaticCombNoisePool configured exactly like the training policy
    ``conf/online_mix/rps_static_comb_only.yaml`` (n_harmonics 100,
    min_harm_above_floor 0.30, drone_profile_range [0,1], mic_gain_db
    [-12,0], synthetic_intermittent RPS) — except n_mics=1 (models take
    single-channel audio) and the ``aggressiveness`` under test."""
    from data_processing.rotor_spectral_model import StaticCombNoisePool

    return StaticCombNoisePool(
        sample_rate=sr,
        duration_s=duration_s,
        n_harmonics=100,
        n_mics=1,
        n_rotors=N_ROTORS,
        min_harm_above_floor=0.30,
        aggressiveness=aggressiveness,
        rps_kind="synthetic_intermittent",
        drone_profile_range=(0.0, 1.0),
        mic_gain_db=(-12.0, 0.0),
    )


def synth_speech_proxy(T: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-shaped stand-in: band-limited noise with syllabic-rate bursts.

    Numpy port of ``src/experiments/kalman_harmonic/phase0.py::
    synth_speech_proxy`` (torch → numpy rng, same spectral shape/envelope)."""
    x = rng.standard_normal(T)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(T, 1.0 / sr)
    shape = (np.maximum(f / 500.0, 0.2) ** -1.0) * (f > 100) * (f < 4000)
    x = np.fft.irfft(X * shape, n=T)
    env = (1.0 + np.sin(2.0 * np.pi * 3.0 * np.arange(T) / sr + 2.0 * np.pi * rng.uniform())) / 2.0
    return (x * env**2).astype(np.float32)


def make_clips(
    agg_idx: int, aggressiveness: float, args
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize the per-aggressiveness clip bank: (noise (N, T), rps
    (N, R, T), speech (N, T)). Seeded per (seed, agg_idx, clip) so the bank
    is identical across models and shared across the SNR grid."""
    sr, dur, n = args.sample_rate, args.duration, args.n_clips
    T = int(round(dur * sr))
    pool = build_pool(aggressiveness, sr, dur)
    noise = np.empty((n, T), dtype=np.float32)
    rps = np.empty((n, N_ROTORS, T), dtype=np.float32)
    speech = np.empty((n, T), dtype=np.float32)
    for ci in range(n):
        kids = np.random.SeedSequence([args.seed, agg_idx, ci]).spawn(2)
        audio, r, _profiles = pool.render(np.random.default_rng(kids[0]), dur)
        noise[ci] = audio[0]  # mic 0 of the single-mic render
        rps[ci] = r
        speech[ci] = synth_speech_proxy(T, sr, np.random.default_rng(kids[1]))
    return noise, rps, speech


def mix_at_snr(noise: np.ndarray, speech: np.ndarray, snr_db: float) -> np.ndarray:
    """Mixture (N, T) with speech scaled to ``snr_db`` relative to the comb
    noise per clip (online-mix convention: snr = speech power / noise power)."""
    p_n = np.mean(noise**2, axis=-1, keepdims=True)
    p_s = np.maximum(np.mean(speech**2, axis=-1, keepdims=True), 1e-12)
    scale = np.sqrt(p_n * 10.0 ** (snr_db / 10.0) / p_s)
    return (noise + scale * speech).astype(np.float32)


# ── Evaluation ──────────────────────────────────────────────────────────────


def batched_forward(model, wins: np.ndarray, device: str, batch: int) -> np.ndarray:
    """wins (N, T) float32 → (N, R, F) float32. fp32, no_grad."""
    import torch

    outs = []
    with torch.no_grad():
        for i in range(0, wins.shape[0], batch):
            t = torch.from_numpy(wins[i : i + batch]).float().to(device)
            outs.append(model(t).float().cpu().numpy())
    pred = np.concatenate(outs, axis=0)
    if pred.ndim != 3 or pred.shape[1] != N_ROTORS:
        raise RuntimeError(f"unexpected model output shape {pred.shape}")
    return pred


def eval_cell(
    model,
    mixtures: np.ndarray,
    rps: np.ndarray,
    args,
) -> dict[str, Any]:
    """One (aggressiveness, snr) cell for one model: per-clip lock/PIT-MAE."""
    from tasks.rps_prediction import align_rps_to_gt

    sr = args.sample_rate
    preds = batched_forward(model, mixtures, args.device, args.batch)
    F = preds.shape[-1]
    t_frames = np.arange(F) * HOP / sr  # torch.stft(center=True) frame centers
    t_audio = np.arange(rps.shape[-1]) / sr
    warm = int(round(args.warmup * sr / HOP))

    maes: list[float] = []
    fracs: list[float] = []
    locked: list[bool] = []
    for ci in range(preds.shape[0]):
        gtf = np.stack([np.interp(t_frames, t_audio, rps[ci, r]) for r in range(N_ROTORS)])
        aligned = align_rps_to_gt(preds[ci], gtf)
        err = np.mean(np.abs(aligned - gtf), axis=0)[warm:]  # (F - warm,)
        frac_below = float(np.mean(err < args.lock_threshold)) if err.size else 0.0
        maes.append(float(err.mean()) if err.size else float("nan"))
        fracs.append(frac_below)
        locked.append(frac_below >= args.lock_min_frac)

    return {
        "n_clips": len(maes),
        "mean_pit_mae": float(np.mean(maes)),
        "median_pit_mae": float(np.median(maes)),
        "lock_fraction": float(np.mean(locked)),
        "per_clip": {
            "pit_mae": maes,
            "frac_below_threshold": fracs,
            "locked": [bool(x) for x in locked],
        },
    }


def eval_model(model, banks, agg_grid, snr_grid, args) -> list[dict[str, Any]]:
    cells = []
    for ai, agg in enumerate(agg_grid):
        noise, rps, speech = banks[ai]
        for snr in snr_grid:
            mixtures = noise if args.speech == "none" else mix_at_snr(noise, speech, snr)
            cell = {"aggressiveness": float(agg), "snr_db": float(snr)}
            cell.update(eval_cell(model, mixtures, rps, args))
            cells.append(cell)
    return cells


# ── Outputs ─────────────────────────────────────────────────────────────────


def _grids(cells, agg_grid, snr_grid) -> tuple[np.ndarray, np.ndarray]:
    """Cells → (lock (A, S), mae (A, S)) with rows/cols in grid order."""
    lock = np.full((len(agg_grid), len(snr_grid)), np.nan)
    mae = np.full_like(lock, np.nan)
    for c in cells:
        i = agg_grid.index(c["aggressiveness"])
        j = snr_grid.index(c["snr_db"])
        lock[i, j] = c["lock_fraction"]
        mae[i, j] = c["mean_pit_mae"]
    return lock, mae


def plot_boundary(path: Path, label: str, agg_grid, snr_grid, lock, mae) -> None:
    """Lock-fraction heatmap (sequential single-hue, fixed 0..1 scale) with
    direct per-cell labels: lock fraction + mean PIT-MAE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.8 + 1.3 * len(snr_grid), 2.2 + 0.9 * len(agg_grid)))
    im = ax.imshow(lock, cmap="Blues", vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    ax.set_xticks(range(len(snr_grid)), [f"{s:+.0f}" for s in snr_grid])
    ax.set_yticks(range(len(agg_grid)), [f"{a:g}" for a in agg_grid])
    ax.set_xlabel("interference SNR (dB, speech vs comb)")
    ax.set_ylabel("RPS aggressiveness (drift rate)")
    ax.set_title(f"{label} — lock fraction (PIT err < 2 rev/s)", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for i in range(len(agg_grid)):
        for j in range(len(snr_grid)):
            if np.isnan(lock[i, j]):
                continue
            ink = "white" if lock[i, j] > 0.55 else "#1a1a1a"
            ax.text(
                j,
                i,
                f"{lock[i, j]:.2f}\nMAE {mae[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=ink,
            )
    fig.colorbar(im, ax=ax, label="lock fraction", fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_table(label: str, agg_grid, snr_grid, lock, mae) -> None:
    header = "agg\\snr".ljust(10) + "".join(f"{s:+8.0f} dB " for s in snr_grid)
    print(f"\n== {label} ==  (cell = lock fraction / mean PIT-MAE rev/s)")
    print(header)
    for i, agg in enumerate(agg_grid):
        row = f"{agg:<10g}"
        for j in range(len(snr_grid)):
            row += f"{lock[i, j]:.2f}/{mae[i, j]:6.2f} "
        print(row)


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=PATH|random:NAME",
        help="registry model + checkpoint pairs; random:NAME = random init (smoke)",
    )
    ap.add_argument(
        "--aggressiveness",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="rps_synthesis aggressiveness grid (drift-rate axis)",
    )
    ap.add_argument("--snr-db", nargs="+", type=float, default=[10.0, 0.0, -10.0, -20.0])
    ap.add_argument("--n-clips", type=int, default=16, help="clips per (aggressiveness, snr) cell")
    ap.add_argument("--duration", type=float, default=4.0, help="clip length (s)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--speech",
        choices=["proxy", "none"],
        default="proxy",
        help="interference: speech-shaped proxy or pure comb noise",
    )
    ap.add_argument(
        "--ablate-rotation",
        action="store_true",
        help="also evaluate CKLA models with the rotation path zeroed; report the delta",
    )
    ap.add_argument("--lock-threshold", type=float, default=2.0, help="lock error bar (rev/s)")
    ap.add_argument(
        "--lock-min-frac",
        type=float,
        default=0.8,
        help="min fraction of post-warmup frames below threshold to count as locked",
    )
    ap.add_argument("--warmup", type=float, default=1.0, help="warmup span excluded from lock (s)")
    ap.add_argument("--out", default="results/ckla_capture_boundary")
    args = ap.parse_args()

    agg_grid = sorted(dict.fromkeys(args.aggressiveness))
    snr_grid = sorted(dict.fromkeys(args.snr_db), reverse=True)  # hard → easy
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = parse_model_specs(args.models)

    t0 = time.time()
    print(
        f"Synthesizing {len(agg_grid)} aggressiveness banks × {args.n_clips} clips "
        f"× {args.duration:g}s (static-comb + {args.speech} interference)..."
    )
    banks = [make_clips(ai, agg, args) for ai, agg in enumerate(agg_grid)]
    print(f"  done in {time.time() - t0:.1f}s")

    results: dict[str, Any] = {
        "config": {
            **{k: v for k, v in vars(args).items() if k != "models"},
            "models": args.models,
            "aggressiveness_grid": agg_grid,
            "snr_grid": snr_grid,
        },
        "models": {},
    }

    for label, name, ckpt in specs:
        t0 = time.time()
        model = load_model(name, ckpt, args.device, args.seed)
        variants = [(label, model, False)]
        if args.ablate_rotation:
            ablated = ablate_rotation(model)
            if ablated is None:
                print(f"[{label}] --ablate-rotation: no ComplexKLALayer found, skipping ablation")
            else:
                variants.append((f"{label}__rot0", ablated, True))

        for vlabel, vmodel, is_ablated in variants:
            cells = eval_model(vmodel, banks, agg_grid, snr_grid, args)
            results["models"][vlabel] = {
                "registry_name": name,
                "checkpoint": ckpt,
                "rotation_ablated": is_ablated,
                "cells": cells,
            }
            lock, mae = _grids(cells, agg_grid, snr_grid)
            print_table(vlabel, agg_grid, snr_grid, lock, mae)
            plot_boundary(
                out / f"boundary_{_safe(vlabel)}.png", vlabel, agg_grid, snr_grid, lock, mae
            )
        if args.ablate_rotation and len(variants) == 1:
            results["models"][label]["rotation_ablation_note"] = "skipped: no ComplexKLALayer"
        if args.ablate_rotation and len(variants) == 2:
            base_lock, base_mae = _grids(results["models"][label]["cells"], agg_grid, snr_grid)
            abl_lock, abl_mae = _grids(
                results["models"][f"{label}__rot0"]["cells"], agg_grid, snr_grid
            )
            print(f"\n== {label} rotation-ablation delta ==  (rot0 − base)")
            print_table(f"{label} Δ", agg_grid, snr_grid, abl_lock - base_lock, abl_mae - base_mae)
        print(f"[{label}] evaluated in {time.time() - t0:.1f}s")

    with open(out / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out / 'results.json'} + boundary_*.png")


if __name__ == "__main__":
    main()
