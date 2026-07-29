#!/usr/bin/env python3
"""Blind-VK pseudo-label annotator for unannotated drone ego-noise datasets.

Produces per-frame ``n_rotors`` RPS pseudo-labels + confidence for every
recording of a published ``tdframe-v1`` dload dataset (first target:
``AVQ-egonoise`` — 5 mono 16 kHz pure ego-noise recordings, ~705 s), so the
recordings can later join RPS-predictor training.

Pipeline per recording (the validated geometry-free "plain" blind ladder,
``scripts/vk_blind_sweep.py``):

1. Slice into overlapping windows (default 20 s / 4 s overlap — both
   ``blind_seed`` and the VK solve degrade beyond ~25 s).
2. Per window: ``blind_seed(arms={"K","R"})`` (K = auto update-gate/bw knobs,
   R = residual re-scan — load-bearing for hidden combs) → constant init
   ``r0`` from the seed bases → CAPTURE (annealed grow schedule) → REFINE
   (fixed narrow bands, k_min=6, de-biasing), with the seed's auto knobs
   spliced in exactly as ``vk_blind_sweep.run_pipeline`` does.
3. Stitch windows: each window's rotor rows are PIT-aligned (MSE Hungarian on
   the overlap frames) to the running stitched estimate — rotor order is
   arbitrary per window — then averaged on overlaps
   (``rps_predictor_vk_eval.stitch_stack``'s convention).
4. Honest refusal per (window, rotor): rows whose mean VK confidence falls
   below ``--refuse-conf`` (``vk_spcup.REFUSE_CONF`` calibration: every
   DREGON success sits at 0.026-0.033) are NaN'd rather than emitted.

Artifacts in ``--out`` (default ``results/vk_pseudolabel/<dataset>/``):
per-recording NPZ (ft, rps with NaN where refused, stitched VK confidence,
per-window comb confidence / seed bases / walls, window bounds) and
``summary.csv``.

Run (smoke, one 20 s window):
  PYTHONPATH=src python scripts/vk_pseudolabel.py --recordings S1_seq1 --max-s 20
Full (CPU-heavy: blind_seed is ~50-100 s per 20 s window):
  omnirun submit --backend apocrita-cpu --gpus 0 --time 10h -- \
    python scripts/vk_pseudolabel.py
"""

from __future__ import annotations

import os
import sys


def _early_arg(name: str, default: str) -> str:
    """Read one ``--name value`` / ``--name=value`` arg before heavy imports."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


# Cap BLAS threads BEFORE numpy import (same rationale as vk_spcup: the VK
# solve is BLAS-bound and this box / cluster node is shared).
_OMP = _early_arg("--omp", "2")
os.environ.setdefault("OMP_NUM_THREADS", _OMP)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _OMP)
os.environ.setdefault("MKL_NUM_THREADS", _OMP)

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

# Pin the repo's src/ ahead of site-packages (same rationale as
# scripts/vk_blind_sweep.py / rps_predictor_vk_eval.py: the editable install
# points at whatever checkout owns .venv, which on omnirun worktrees is NOT
# the job's checkout). Also pin scripts/ so the CAPTURE/REFINE configs are
# imported from THIS checkout's vk_blind_annotation.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from vk_blind_annotation import CAPTURE_CFG, REFINE_CFG  # noqa: E402

from data_processing.frames import get_meta  # noqa: E402
from data_processing.streams import iter_published_frames  # noqa: E402
from data_processing.vk_blind_seeding import (  # noqa: E402
    SeedConfig,
    blind_seed,
    track_comb_confidence,
    whitened_logmag,
)
from data_processing.vk_tracking import VKResult, vk_track  # noqa: E402

SR = 16000
HOP = 512
FRAME_HOP_S = HOP / SR  # 0.032 s — the project-wide trajectory grid
# Refusal gate on mean refine-phase VK confidence, per (window, rotor).
# Calibration: vk_spcup.REFUSE_CONF (DREGON success band 0.026-0.033;
# visually-hallucinated tracks sit at <= 0.016).
REFUSE_CONF = 0.02
MAX_CHANNELS = 8  # comb evidence averages across mics; 8 is plenty


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_recordings(
    dataset: str, wanted: set[str] | None, max_s: float | None
) -> dict[str, np.ndarray]:
    """Stream ``dataset``; return ``{recording_id: (C, T) float32 @ 16 kHz}``."""
    out: dict[str, np.ndarray] = {}
    for frame in iter_published_frames(dataset):
        rid = str(get_meta(frame, "recording_id", ""))
        if wanted is not None and rid not in wanted:
            continue
        aud = frame["audio"]
        data = np.asarray(aud.data, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[0] > MAX_CHANNELS:
            data = data[:MAX_CHANNELS]
        sr = int(aud.tindex.sr)
        if max_s is not None:
            data = data[:, : int(round(max_s * sr))]
        if sr != SR:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=SR, axis=-1, res_type="soxr_hq")
        out[rid] = np.ascontiguousarray(data)
        print(f"  loaded {rid}: {data.shape} @ {SR} Hz (native {sr})", flush=True)
        if wanted is not None and set(out) >= wanted:
            break
    return out


# ── Window stitching ──────────────────────────────────────────────────────────
TAIL_MIN_S = 2.0  # tails shorter than this extend the last window instead of
# spawning a nearly-duplicate solve (worst case window-s + 2 s, still under
# the ~25 s blind_seed/VK degradation bound at the 20 s default)


def window_spans(n_frames: int, win_frames: int, hop_frames: int) -> list[tuple[int, int]]:
    """``(f0, f1)`` frame spans covering ``[0, n_frames)`` exactly."""
    if n_frames <= win_frames:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - win_frames + 1, hop_frames))
    spans = [(s, s + win_frames) for s in starts]
    tail = n_frames - spans[-1][1]
    if tail > 0:
        if tail >= int(round(TAIL_MIN_S / FRAME_HOP_S)):
            spans.append((n_frames - win_frames, n_frames))  # backed-up tail window
        else:
            spans[-1] = (spans[-1][0], n_frames)  # absorb the sliver
    return spans


def perm_align_overlap(pred: np.ndarray, ref_overlap: np.ndarray) -> np.ndarray:
    """Permutation (index array ``perm``: new row i <- pred row perm[i]) by
    MSE Hungarian between ``pred``'s overlap columns and the running stitched
    estimate (``rps_predictor_vk_eval.perm_align_overlap``'s convention)."""
    from scipy.optimize import linear_sum_assignment

    cost = np.mean((pred[:, None, :] - ref_overlap[None, :, :]) ** 2, axis=-1)
    row, col = linear_sum_assignment(cost)
    perm = np.empty(pred.shape[0], dtype=int)
    perm[col] = row
    return perm


# ── Per-recording driver ──────────────────────────────────────────────────────
def process_recording(
    rid: str,
    audio: np.ndarray,
    n_rotors: int,
    win_frames: int,
    hop_frames: int,
    refuse_conf: float,
    out_dir: Path,
) -> dict[str, Any]:
    """Windowed blind-seed + plain VK ladder + stitching; saves the NPZ."""
    n_frames = int(audio.shape[-1] / SR / FRAME_HOP_S) + 1
    ft = np.arange(n_frames) * FRAME_HOP_S
    spans = window_spans(n_frames, win_frames, hop_frames)
    seed_cfg = SeedConfig()

    # Raw accumulator (all rows — the alignment reference) and accepted-only
    # accumulator (the emitted pseudo-labels; refused rows never enter).
    acc_all = np.zeros((n_rotors, n_frames))
    cnt_all = np.zeros(n_frames)
    acc_ok = np.zeros((n_rotors, n_frames))
    cnt_ok = np.zeros((n_rotors, n_frames))

    conf_chunks: list[np.ndarray] = []
    conf_time_chunks: list[np.ndarray] = []
    seed_bases_pw = np.full((len(spans), n_rotors), np.nan)
    comb_conf_pw = np.full((len(spans), n_rotors), np.nan)
    vk_conf_pw = np.full((len(spans), n_rotors), np.nan)
    refused_pw = np.zeros((len(spans), n_rotors), dtype=bool)
    bounds = np.zeros((len(spans), 2), dtype=int)
    wall_seed = np.zeros(len(spans))
    wall_vk = np.zeros(len(spans))

    for w, (f0, f1) in enumerate(spans):
        bounds[w] = (f0, f1)
        wav = audio[:, f0 * HOP : min(audio.shape[-1], f1 * HOP)].astype(np.float64)
        ftw = np.arange(f1 - f0) * FRAME_HOP_S

        tic = time.perf_counter()
        seed = blind_seed(wav, float(SR), n_rotors, seed_cfg, arms={"K", "R"})
        wall_seed[w] = time.perf_counter() - tic

        # Geometry-free "plain" ladder, exactly vk_blind_sweep.run_pipeline:
        # auto update_gate everywhere; auto bw only for CAPTURE (its final
        # annealed band must admit k * wander) — REFINE keeps its narrow
        # de-biasing band.
        r0 = np.repeat(seed.bases[:, None], f1 - f0, axis=1)
        gate = seed.update_gate
        cap_cfg, ref_cfg = CAPTURE_CFG, REFINE_CFG
        if gate is not None:
            cap_cfg = replace(cap_cfg, update_gate=gate, bw_hz=float(seed.bw_hz or 1.5))
            ref_cfg = replace(ref_cfg, update_gate=gate)
        tic = time.perf_counter()
        cap: VKResult = vk_track(wav, r0, ftw, cap_cfg)
        ref: VKResult = vk_track(wav, cap.r_refined, ftw, ref_cfg)
        wall_vk[w] = time.perf_counter() - tic

        # PIT-align this window's rotor rows to the running stitched estimate
        # on the overlap (raw accumulator: refused rows still anchor identity).
        r_win = ref.r_refined
        conf = ref.confidence
        perm = np.arange(n_rotors)
        seen = cnt_all[f0:f1] > 0
        if seen.any():
            ref_traj = acc_all[:, f0:f1][:, seen] / cnt_all[f0:f1][seen]
            perm = perm_align_overlap(r_win[:, seen], ref_traj)
        r_win = r_win[perm]
        conf = conf[perm] if conf.size else conf
        seed_bases_pw[w] = seed.bases[perm]

        # Blind per-track comb confidence along the refined trajectories.
        white, bin_hz, st = whitened_logmag(wav.astype(np.float32), float(SR), seed_cfg)
        comb_conf_pw[w] = track_comb_confidence(white, bin_hz, st, ftw, r_win, seed_cfg)

        # Refusal per rotor row on mean VK confidence.
        conf_mean = np.mean(conf, axis=1) if conf.size else np.full(n_rotors, np.nan)
        vk_conf_pw[w] = conf_mean
        refused = ~(conf_mean >= refuse_conf)  # NaN conf -> refused
        refused_pw[w] = refused

        acc_all[:, f0:f1] += r_win
        cnt_all[f0:f1] += 1.0
        keep = ~refused
        acc_ok[keep, f0:f1] += r_win[keep]
        cnt_ok[keep, f0:f1] += 1.0

        if conf.size:
            conf_chunks.append(conf)
            conf_time_chunks.append(ref.conf_times + f0 * FRAME_HOP_S)

        print(
            f"  [{rid} win {w + 1}/{len(spans)} {ft[f0]:.1f}-{ft[f1 - 1]:.1f}s] "
            f"seed [{', '.join(f'{b:.1f}' for b in seed_bases_pw[w])}] rev/s "
            f"gate={gate if gate is None else round(gate, 1)} "
            f"bw={seed.bw_hz if seed.bw_hz is None else round(float(seed.bw_hz), 2)} | "
            f"vk_conf [{', '.join(f'{c:.3f}' for c in conf_mean)}] "
            f"comb [{', '.join(f'{c:.3f}' for c in comb_conf_pw[w])}] "
            f"refused {refused.sum()}/{n_rotors} | "
            f"seed {wall_seed[w]:.0f}s vk {wall_vk[w]:.0f}s",
            flush=True,
        )

    rps = np.full((n_rotors, n_frames), np.nan)
    have = cnt_ok > 0
    rps[have] = acc_ok[have] / cnt_ok[have]

    if conf_chunks:
        confidence = np.concatenate(conf_chunks, axis=1)
        conf_times = np.concatenate(conf_time_chunks)
        order = np.argsort(conf_times, kind="stable")
        confidence, conf_times = confidence[:, order], conf_times[order]
    else:
        confidence = np.zeros((n_rotors, 0))
        conf_times = np.zeros(0)

    comb_conf = np.nanmean(comb_conf_pw, axis=0)
    pct_refused = float(np.mean(np.isnan(rps))) * 100.0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{rid}.npz",
        ft=ft,
        rps=rps,
        confidence=confidence,
        conf_times=conf_times,
        comb_conf=comb_conf,
        comb_conf_per_window=comb_conf_pw.T,  # (R, n_win)
        vk_conf_per_window=vk_conf_pw.T,  # (R, n_win)
        refused_per_window=refused_pw.T,  # (R, n_win)
        seed_bases_per_window=seed_bases_pw,  # (n_win, R)
        window_bounds=bounds,  # (n_win, 2) frame indices [f0, f1)
        refuse_conf=refuse_conf,
        wall_seed_s=wall_seed,
        wall_vk_s=wall_vk,
    )

    return {
        "recording_id": rid,
        "duration_s": round(audio.shape[-1] / SR, 1),
        "n_windows": len(spans),
        "pct_refused": round(pct_refused, 1),
        "mean_vk_conf": round(float(np.nanmean(vk_conf_pw)), 4),
        "mean_comb_conf": round(float(np.nanmean(comb_conf_pw)), 4),
        "median_rps": ", ".join(
            "nan" if np.all(np.isnan(r)) else f"{np.nanmedian(r):.1f}" for r in rps
        ),
        "wall_seed_s": round(float(wall_seed.sum()), 1),
        "wall_vk_s": round(float(wall_vk.sum()), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--dataset", default="AVQ-egonoise", help="published tdframe-v1 dataset")
    ap.add_argument("--recordings", default=None, help="comma-separated recording_id filter")
    ap.add_argument("--n-rotors", type=int, default=4)
    ap.add_argument("--window-s", type=float, default=20.0, help="analysis window length")
    ap.add_argument("--overlap-s", type=float, default=4.0, help="window overlap")
    ap.add_argument("--max-s", type=float, default=None, help="cap seconds per recording (smoke)")
    ap.add_argument("--refuse-conf", type=float, default=REFUSE_CONF)
    ap.add_argument("--out", default=None, help="default results/vk_pseudolabel/<dataset>/")
    ap.add_argument("--omp", default="2", help="BLAS thread cap (read pre-import)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("results/vk_pseudolabel") / args.dataset
    win_frames = int(round(args.window_s / FRAME_HOP_S))
    hop_frames = int(round((args.window_s - args.overlap_s) / FRAME_HOP_S))
    if hop_frames <= 0:
        raise SystemExit("--overlap-s must be smaller than --window-s")
    wanted = (
        {r.strip() for r in args.recordings.split(",") if r.strip()} if args.recordings else None
    )

    print(f"Streaming {args.dataset} ...", flush=True)
    recs = load_recordings(args.dataset, wanted, args.max_s)
    if wanted and (missing := wanted - set(recs)):
        raise SystemExit(f"recordings not found in {args.dataset}: {sorted(missing)}")
    if not recs:
        raise SystemExit(f"no recordings matched in {args.dataset}")

    rows = []
    for rid in sorted(recs):
        print(
            f"{rid}: {recs[rid].shape[-1] / SR:.1f}s, window {args.window_s:g}s "
            f"overlap {args.overlap_s:g}s",
            flush=True,
        )
        row = process_recording(
            rid, recs[rid], args.n_rotors, win_frames, hop_frames, args.refuse_conf, out_dir
        )
        rows.append(row)
        print(
            f"[done {rid}] {row['n_windows']} windows, {row['pct_refused']}% refused, "
            f"median rps [{row['median_rps']}], seed {row['wall_seed_s']}s vk {row['wall_vk_s']}s",
            flush=True,
        )

    cols = list(rows[0].keys())
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nArtifacts written to {out_dir}/ (summary.csv, per-recording .npz)", flush=True)


if __name__ == "__main__":
    main()
