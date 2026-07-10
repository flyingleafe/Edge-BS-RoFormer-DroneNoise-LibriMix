#!/usr/bin/env python3
"""Blind (telemetry-free) RPS annotation of drone ego-noise via comb refinement.

Demonstrates that ``data_processing.rps_refinement`` — designed to *refine*
telemetry-initialised rotor-speed trajectories — can also **initialise from
nothing but the audio** and annotate unlabeled ego-noise, while its
``comb_confidence`` correctly *refuses* where no rotor comb exists.

Test bed: the published ``SPCUP19-egonoise`` dataset (10 heterogeneous student
drone rigs, IEEE SP Cup 2019 bonus task). We pick a handful of recordings
spanning different drones / channel counts, take a mid-recording segment, and
run a fully blind pipeline:

1. **Base-speed scan** — sweep a constant trajectory ``r0`` over 30..120 rev/s
   and read the summed-log-mag comb score ``score(r0)``. The top local maxima
   are candidate rotor speeds. Octave ambiguity (``score(2*r0)`` rivals
   ``score(r0)`` because its harmonics are a subset of ``r0``'s) is resolved by
   preferring the *smaller* base when the two are within a small margin.
2. **Rotor-count model selection** — for ``R in {1, 2, 4}`` initialise ``R``
   constant trajectories (one per clustered peak, else symmetric offsets around
   the base), run :func:`refine_trajectories`, and score the coherent harmonic
   least-squares residual (:func:`harmonic_lsq_residual`). Pick ``R`` at the
   residual elbow (stop when adding a rotor buys < 10 % relative improvement).
3. **Confidence + collision gating** — record per-window ``comb_confidence`` and
   flag rotor collisions (unresolvable duplicate speeds).

Artifacts land in ``results/rps_refinement/spcup/``: one ``.npz`` per recording
(scan curve, refined trajectories, confidence, per-R residuals), a
``summary.csv``, and per-recording diagnostic PNGs. CPU-only; a few minutes end
to end. Run: ``python scripts/rps_refinement_spcup.py``.
"""

from __future__ import annotations

import contextlib
import csv
import json
import time
from pathlib import Path
from typing import Any

import librosa
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

from data_processing.frames import get_meta
from data_processing.rps_refinement import (
    LogMagSpec,
    RefineConfig,
    RefinementResult,
    comb_score,
    compute_logmag,
    harmonic_lsq_residual,
    refine_trajectories,
)
from data_processing.streams import iter_published_frames

# Play nice with concurrent experiments on shared CPU boxes: torch grabbing
# every core makes *everything* slower under contention.
torch.set_num_threads(4)

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET = "SPCUP19-egonoise"
SR = 16000
SEG_S = 25.0  # segment length (s) analysed per recording
SCAN_LO, SCAN_HI, SCAN_STEP = 30.0, 120.0, 0.05  # base-speed scan grid (rev/s)
SCAN_K_MAX = 40  # harmonics summed in the scan
SCAN_CHUNK = 64  # grid points per batched comb_score call
OCTAVE_MARGIN = 0.02  # prefer smaller base when score gap below this (log10)
CLUSTER_W = 5.0  # rev/s: "multiple close maxima" window for per-peak init
COLLISION = 0.15  # rev/s: rotors closer than this anywhere -> unresolved
ELBOW = 0.10  # relative residual improvement below which we stop adding rotors
LSQ_K_MAX = 40  # harmonics in the residual least-squares fit
LOW_CONF = 0.02  # mean confidence below this -> the method "refuses"
R_GRID = (1, 2, 4)
OFFSETS: dict[int, tuple[float, ...]] = {
    1: (0.0,),
    2: (-0.5, 0.5),
    4: (-1.5, -0.5, 0.5, 1.5),
}
HARMONIC_K = (1, 2, 4, 8, 16)  # harmonics overlaid on the spectrogram
ROTOR_COLORS = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")
OUT = Path("results/rps_refinement/spcup")

# Curated selection: max rig / channel-count diversity, prefer multichannel and
# flight / ego-noise conditions (where a real comb should exist). One
# deliberately marginal case (a calibration clip) is kept so the demo can show
# the confidence gate refusing rather than only succeeding.
TARGETS: tuple[str, ...] = (
    "Diagonal_Unloading__recordings__flight__square_10m",  # DJI P4 PRO, 8ch flight
    "AGH__ego-noise__mic_array__1",  # quadrotor, 8ch ego-noise
    "KU_Leuven__SPCUP19_KU_Leuven_Team_1_recording",  # MikroKopter, 8ch
    "Idea_ssu__free_flight_1",  # DJI Phantom 4, 1ch flight
    "Maverick__5",  # YH-19HW, 3ch
    "Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission1",  # Intel Aero, 8ch static
    "AGH__calibration__1",  # calibration clip (marginal-comb control)
)


# ── Metadata + audio loading ────────────────────────────────────────────────────
def _meta_nested(frame: Any, group: str, key: str, default: Any = None) -> Any:
    """Safe read of a nested ``meta.<group>.<key>`` scalar."""
    with contextlib.suppress(KeyError, TypeError):
        return frame["meta"][group][key]
    return default


def load_segment(frame: Any) -> tuple[np.ndarray, int, float, float]:
    """Mid-recording ``(C, T)`` segment resampled to 16 kHz + native rate."""
    aud = frame["audio"]
    data = np.asarray(aud.data, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] > 8:  # cap channels: comb score averages, 8 is plenty
        data = data[:8]
    sr = int(aud.tindex.sr)
    total_s = data.shape[1] / sr
    seg = min(SEG_S, total_s)
    start = max(0.0, (total_s - seg) / 2.0)  # centred: dodges takeoff/landing
    s0, s1 = int(round(start * sr)), int(round((start + seg) * sr))
    clip = np.ascontiguousarray(data[:, s0:s1])
    if sr != SR:
        clip = librosa.resample(clip, orig_sr=sr, target_sr=SR, axis=-1, res_type="soxr_hq")
    return np.ascontiguousarray(clip.astype(np.float32)), sr, start, seg


def collect_segments() -> dict[str, dict[str, Any]]:
    """Target segments + meta, from the local cache or one dataset stream pass.

    Each entry: ``{"audio": (C, T) float32 @ 16 kHz, "team", "drone",
    "condition", "native_sr", "seg_start_s", "seg_len_s"}``. Cached as one NPZ
    per recording under ``OUT/segments/`` so re-runs skip streaming entirely.
    """
    seg_dir = OUT / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    missing = []
    for rid in TARGETS:
        path = seg_dir / f"{rid}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                out[rid] = {"audio": z["audio"], **json.loads(str(z["meta"]))}
        else:
            missing.append(rid)
    if not missing:
        print(f"All {len(out)} segments loaded from cache ({seg_dir}).", flush=True)
        return out
    wanted = set(missing)
    print(f"Streaming {DATASET} for {len(wanted)} target recordings ...", flush=True)
    for frame in iter_published_frames(DATASET):
        rid = str(get_meta(frame, "recording_id"))
        if rid not in wanted:
            continue
        audio, native_sr, seg_start, seg_len = load_segment(frame)
        meta = {
            "team": _meta_nested(frame, "system", "team"),
            "drone": _meta_nested(frame, "system", "make_model"),
            "condition": _meta_nested(frame, "operating", "condition"),
            "native_sr": native_sr,
            "seg_start_s": round(seg_start, 1),
            "seg_len_s": round(seg_len, 1),
        }
        np.savez(seg_dir / f"{rid}.npz", audio=audio, meta=json.dumps(meta))
        out[rid] = {"audio": audio, **meta}
        wanted.discard(rid)
        print(f"  fetched {rid} {audio.shape}", flush=True)
        if not wanted:
            break
    if wanted:
        print(f"WARNING: never streamed: {sorted(wanted)}", flush=True)
    return out


# ── Blind initialisation ────────────────────────────────────────────────────────
def base_speed_scan(spec: LogMagSpec, cfg: RefineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Constant-trajectory comb score over the base-speed grid (batched)."""
    grid = np.arange(SCAN_LO, SCAN_HI + SCAN_STEP / 2, SCAN_STEP)
    n = spec.n_frames
    scores = np.empty(len(grid), dtype=np.float64)
    dev = spec.logmag.device
    for i in range(0, len(grid), SCAN_CHUNK):
        chunk = grid[i : i + SCAN_CHUNK]
        r = torch.as_tensor(chunk, dtype=torch.float32, device=dev)[:, None, None].expand(-1, 1, n)
        with torch.no_grad():
            sc = comb_score(spec, r, cfg, k_max=SCAN_K_MAX)  # (G, 1)
        scores[i : i + len(chunk)] = sc[:, 0].cpu().numpy()
    return grid, scores


def detect_peaks(grid: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of the scan, sorted by score (descending)."""
    rng = float(scores.max() - scores.min())
    prom = max(0.01, 0.03 * rng)
    dist = max(1, int(round(1.5 / SCAN_STEP)))
    idx, _ = find_peaks(scores, prominence=prom, distance=dist)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(scores))])
    speeds, pk = grid[idx], scores[idx]
    order = np.argsort(pk)[::-1]
    return speeds[order], pk[order]


def choose_base(
    grid: np.ndarray, peak_speeds: np.ndarray, peak_scores: np.ndarray
) -> tuple[float, list[float], bool]:
    """Resolve the octave ambiguity: prefer the smaller base within margin."""
    primary, pscore = float(peak_speeds[0]), float(peak_scores[0])
    base, candidates, octave = primary, [primary], False
    half = primary / 2.0
    if half >= grid[0]:
        d = np.abs(peak_speeds - half)
        j = int(np.argmin(d))
        if float(d[j]) <= 1.0 and float(peak_scores[j]) >= pscore - OCTAVE_MARGIN:
            base = float(peak_speeds[j])
            candidates = [base, primary]
            octave = True
    return base, candidates, octave


def make_init(
    r_count: int, base: float, peak_speeds: np.ndarray, peak_scores: np.ndarray
) -> np.ndarray:
    """Initial constant speeds for ``r_count`` rotors: clustered peaks or offsets."""
    within = np.abs(peak_speeds - base) <= CLUSTER_W
    cl_speeds, cl_scores = peak_speeds[within], peak_scores[within]
    if r_count > 1 and len(cl_speeds) >= r_count:
        top = np.argsort(cl_scores)[::-1][:r_count]
        init = np.sort(cl_speeds[top])
    else:
        init = base + np.asarray(OFFSETS[r_count])
    return np.clip(init, SCAN_LO, SCAN_HI).astype(np.float64)


def _dedupe_rotors(r: np.ndarray) -> np.ndarray:
    """Drop rotor tracks that collapsed onto an earlier one (mean gap < COLLISION)."""
    keep = [0]
    for i in range(1, r.shape[0]):
        if all(float(np.mean(np.abs(r[i] - r[j]))) >= COLLISION for j in keep):
            keep.append(i)
    return r[keep]


def _drop_most_redundant(r: np.ndarray) -> np.ndarray:
    """Remove one rotor of the closest pair (the most redundant track)."""
    best: tuple[float, int] = (np.inf, 1)
    for i in range(r.shape[0]):
        for j in range(i + 1, r.shape[0]):
            d = float(np.mean(np.abs(r[i] - r[j])))
            if d < best[0]:
                best = (d, j)
    return np.delete(r, best[1], axis=0)


def robust_residual(
    audio: np.ndarray, r: np.ndarray, frame_times: np.ndarray, cfg: RefineConfig
) -> tuple[float, int]:
    """``harmonic_lsq_residual`` made robust to blind over-parameterisation.

    When R exceeds the true rotor count, refined tracks collapse onto (or hug)
    the same comb; the joint design matrix goes (near-)rank-deficient and the
    VP normal-equation solve can fail outright. Near-identical tracks span
    essentially the same harmonic subspace, so we drop exact duplicates first
    and then, on solver failure, iteratively remove the most redundant track
    (one of the closest pair) until the fit succeeds. The returned
    ``n_effective`` is the rotor count the fit could actually resolve.
    """
    cand = _dedupe_rotors(r)
    while True:
        try:
            out = harmonic_lsq_residual(audio, cand, frame_times, cfg, k_max=LSQ_K_MAX)
            return float(out["residual_ratio"]), int(cand.shape[0])
        except RuntimeError:
            if cand.shape[0] == 1:
                return float("nan"), 1
            cand = _drop_most_redundant(cand)


def refine_for_r(
    spec: LogMagSpec, audio: np.ndarray, cfg: RefineConfig, init: np.ndarray
) -> tuple[RefinementResult, float, int]:
    """Refine constant inits then score the coherent harmonic LSQ residual."""
    r_init = np.repeat(init[:, None], spec.n_frames, axis=1)  # (R, N)
    result = refine_trajectories(spec, r_init, cfg)
    resid, n_eff = robust_residual(audio, result.r_refined, result.frame_times, cfg)
    return result, resid, n_eff


def choose_r_by_elbow(residuals: dict[int, float]) -> int:
    """Smallest R past which adding a rotor cuts residual by < ELBOW (relative).

    A NaN residual (degenerate fit that even the robust path could not score)
    counts as "no improvement" — the elbow stops before it.
    """
    chosen = R_GRID[0]
    for prev, cur in zip(R_GRID, R_GRID[1:]):
        if not (np.isfinite(residuals[prev]) and np.isfinite(residuals[cur])):
            break
        rel = (residuals[prev] - residuals[cur]) / max(residuals[prev], 1e-12)
        if rel >= ELBOW:
            chosen = cur
        else:
            break
    return chosen


def has_collision(r_refined: np.ndarray) -> bool:
    """Any two rotor tracks within COLLISION rev/s anywhere -> unresolved."""
    r_count = r_refined.shape[0]
    if r_count < 2:
        return False
    worst = np.inf
    for i in range(r_count):
        for j in range(i + 1, r_count):
            worst = min(worst, float(np.min(np.abs(r_refined[i] - r_refined[j]))))
    return worst < COLLISION


# ── Per-recording driver ─────────────────────────────────────────────────────────
def process(rid: str, seg: dict[str, Any]) -> dict[str, Any]:
    """Full blind pipeline for one recording; returns a summary dict + saves NPZ."""
    audio = seg["audio"]
    # delta_max=2.0: the blind init sits at a scan peak, so the coarse basin is
    # narrow; iters=200: the spline correction after basin capture is small and
    # converges well before the 300-iter default (checked on KU_Leuven).
    cfg = RefineConfig(sample_rate=SR, delta_max=2.0, iters=200, device="cpu")
    spec = compute_logmag(audio, cfg)

    grid, scores = base_speed_scan(spec, cfg)
    peak_speeds, peak_scores = detect_peaks(grid, scores)
    base, candidates, octave = choose_base(grid, peak_speeds, peak_scores)

    results: dict[int, RefinementResult] = {}
    residuals: dict[int, float] = {}
    n_effective: dict[int, int] = {}
    for r_count in R_GRID:
        init = make_init(r_count, base, peak_speeds, peak_scores)
        results[r_count], residuals[r_count], n_effective[r_count] = refine_for_r(
            spec, audio, cfg, init
        )

    chosen = choose_r_by_elbow(residuals)
    result = results[chosen]
    mean_conf = float(np.mean(result.confidence))
    collision = has_collision(result.r_refined)
    base_speeds = np.median(result.r_refined, axis=1)

    np.savez(
        OUT / f"{rid}.npz",
        grid=grid,
        scores=scores,
        peak_speeds=peak_speeds,
        peak_scores=peak_scores,
        base=base,
        base_candidates=np.asarray(candidates),
        octave_ambiguity=octave,
        chosen_r=chosen,
        r_refined=result.r_refined,
        frame_times=result.frame_times,
        confidence=result.confidence,
        conf_centers=result.conf_centers,
        residual_r1=residuals[1],
        residual_r2=residuals[2],
        residual_r4=residuals[4],
        n_effective_r1=n_effective[1],
        n_effective_r2=n_effective[2],
        n_effective_r4=n_effective[4],
        base_speeds=base_speeds,
    )

    return {
        "recording_id": rid,
        "team": seg["team"],
        "drone": seg["drone"],
        "condition": seg["condition"],
        "n_channels": int(audio.shape[0]),
        "native_sr": seg["native_sr"],
        "seg_start_s": seg["seg_start_s"],
        "seg_len_s": seg["seg_len_s"],
        "base_primary": round(float(peak_speeds[0]), 2),
        "base_chosen": round(base, 2),
        "octave_ambiguity": octave,
        "chosen_r": chosen,
        "base_speeds": ", ".join(f"{v:.1f}" for v in base_speeds),
        "mean_confidence": round(mean_conf, 4),
        "residual_chosen": round(residuals[chosen], 4),
        "residual_by_r": {r: round(residuals[r], 4) for r in R_GRID},
        "n_effective": n_effective[chosen],
        "collision_unresolved": collision,
        "refused": mean_conf < LOW_CONF,
        # kept in-memory for figures (not serialised into the summary CSV):
        "_spec": spec,
        "_result": result,
        "_grid": grid,
        "_scores": scores,
        "_peaks": (peak_speeds, peak_scores),
        "_residuals": residuals,
    }


# ── Figures ──────────────────────────────────────────────────────────────────────
def make_figure(row: dict[str, Any]) -> None:
    """(a) scan curve + peaks, (b) spectrogram + refined tracks, (c) residual vs R."""
    spec: LogMagSpec = row["_spec"]
    result: RefinementResult = row["_result"]
    grid, scores = row["_grid"], row["_scores"]
    peak_speeds, peak_scores = row["_peaks"]
    residuals = row["_residuals"]
    rid = row["recording_id"]

    fig = plt.figure(figsize=(13, 8), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15), hspace=0.32, wspace=0.22)
    ax_scan = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[1, :])

    # (a) base-speed scan
    ax_scan.plot(grid, scores, color="#333333", lw=1.0)
    ax_scan.scatter(peak_speeds, peak_scores, color="#ff7f00", s=28, zorder=5, label="local maxima")
    ax_scan.axvline(row["base_chosen"], color="#e41a1c", ls="--", lw=1.3, label="chosen base")
    if row["octave_ambiguity"]:
        ax_scan.axvline(row["base_primary"], color="#377eb8", ls=":", lw=1.3, label="octave (2x)")
    ax_scan.set_xlabel("base speed r0 (rev/s)")
    ax_scan.set_ylabel("comb score (mean log-mag)")
    ax_scan.set_title("(a) blind base-speed scan")
    ax_scan.legend(fontsize=8, loc="best")
    ax_scan.grid(alpha=0.25)

    # (c) residual vs R
    r_vals = list(R_GRID)
    res_vals = [residuals[r] for r in r_vals]
    colors = ["#4daf4a" if r == row["chosen_r"] else "#bbbbbb" for r in r_vals]
    ax_res.bar([str(r) for r in r_vals], res_vals, color=colors)
    ax_res.set_xlabel("rotor count R")
    ax_res.set_ylabel("LSQ residual ratio")
    ax_res.set_title(f"(c) residual vs R (chosen R={row['chosen_r']})")
    for x, v in enumerate(res_vals):
        if np.isfinite(v):
            ax_res.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax_res.text(x, 0.02, "degenerate", ha="center", va="bottom", fontsize=8, rotation=90)
    ax_res.grid(alpha=0.25, axis="y")

    # (b) spectrogram 0-2 kHz with refined harmonic tracks
    lm = spec.logmag[0].cpu().numpy()  # channel 0
    freqs = np.arange(lm.shape[0]) * spec.bin_hz
    fmask = freqs <= 2000.0
    ax_spec.pcolormesh(
        spec.frame_times, freqs[fmask], lm[fmask], shading="auto", cmap="magma", rasterized=True
    )
    ft = result.frame_times
    halo = [patheffects.withStroke(linewidth=2.6, foreground="white", alpha=0.9)]
    for i in range(result.r_refined.shape[0]):
        color = ROTOR_COLORS[i % len(ROTOR_COLORS)]
        for j, k in enumerate(HARMONIC_K):
            track = k * result.r_refined[i]
            track = np.where(track <= 2000.0, track, np.nan)
            ax_spec.plot(
                ft,
                track,
                color=color,
                lw=1.1,
                alpha=0.95,
                path_effects=halo,
                label=f"rotor {i} (k=1,2,4,8,16)" if j == 0 else None,
            )
    ax_spec.set_xlabel("time (s)")
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_ylim(0, 2000)
    ax_spec.set_title(
        f"(b) log-mag spectrogram + refined comb  |  mean conf={row['mean_confidence']:.3f}"
        + ("  [REFUSED: low confidence]" if row["refused"] else "")
    )
    ax_spec.legend(fontsize=8, loc="upper right", framealpha=0.85)

    fig.suptitle(f"{rid}   ({row['drone']}, {row['n_channels']}ch)", fontsize=12)
    fig.savefig(OUT / f"{rid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    segments = collect_segments()

    rows: list[dict[str, Any]] = []
    for rid in TARGETS:
        if rid not in segments:
            continue
        print(f"\nProcessing {rid} ...", flush=True)
        t0 = time.time()
        row = process(rid, segments[rid])
        make_figure(row)
        rows.append(row)
        print(
            f"  [{time.time() - t0:.0f}s] base={row['base_chosen']} "
            f"(primary {row['base_primary']}, octave={row['octave_ambiguity']}) | "
            f"R={row['chosen_r']} | conf={row['mean_confidence']:.3f} | "
            f"resid={row['residual_chosen']:.3f} | "
            f"collision={row['collision_unresolved']} | refused={row['refused']}",
            flush=True,
        )

    # summary.csv
    csv_cols = [
        "recording_id",
        "team",
        "drone",
        "condition",
        "n_channels",
        "native_sr",
        "seg_start_s",
        "seg_len_s",
        "base_primary",
        "base_chosen",
        "octave_ambiguity",
        "chosen_r",
        "base_speeds",
        "mean_confidence",
        "residual_chosen",
        "n_effective",
        "collision_unresolved",
        "refused",
    ]
    with (OUT / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in csv_cols})

    # console summary table
    print("\n" + "=" * 118)
    print("SUMMARY: blind RPS annotation on SPCUP19-egonoise")
    print("=" * 118)
    hdr = (
        f"{'recording':<44}{'drone':<26}{'ch':>3}{'R':>3}{'base(rev/s)':>26}"
        f"{'conf':>8}{'resid':>8}{'flag':>10}"
    )
    print(hdr)
    print("-" * 118)
    for row in rows:
        flag = "REFUSE" if row["refused"] else ("COLLIDE" if row["collision_unresolved"] else "ok")
        print(
            f"{row['recording_id'][:43]:<44}{str(row['drone'])[:25]:<26}"
            f"{row['n_channels']:>3}{row['chosen_r']:>3}{row['base_speeds']:>26}"
            f"{row['mean_confidence']:>8.3f}{row['residual_chosen']:>8.3f}{flag:>10}"
        )
    print("=" * 118)
    print(f"Artifacts written to {OUT}/ (summary.csv, per-recording .npz + .png)")


if __name__ == "__main__":
    main()
