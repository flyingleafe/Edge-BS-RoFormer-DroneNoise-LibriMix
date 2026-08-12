#!/usr/bin/env python3
"""Blind rotor-speed annotation of drone-noise recordings that have NO telemetry.

The blind ladder (``tracking.blind_fullrange`` / ``tracking.vit2dsp``) is
calibrated against telemetry on DREGON (pooled ``err_sm`` 0.688 rev/s) and on
Michael's FLY124 (cruise 1.03 rev/s). Outside those two rigs there is no
ground truth at all, so this driver scores annotation QUALITY with LABEL-FREE
instruments only:

1. ``fvk`` — the profiled coupled-VK residual (:func:`tracking.fvk_score`) of
   the blind trajectory, and of six PERTURBED siblings at FIXED degrees of
   freedom (the harmonic cap is pinned to the blind trajectory, so every
   candidate is scored on identical cells). The siblings are the two octave
   candidates (``half`` = r/2, ``double`` = 2r) and four detunings
   (±0.3, ±1.0 rev/s). Their objectives against the blind one are the
   well-depth reading: a real minimum rises on BOTH sides, an octave failure
   shows a sibling BELOW the annotation.
2. ``ridge`` — dB of line density on the carrier over a local floor
   (:func:`tracking.score_window`, phase-6d component, the one where more is
   better), under the ``none`` control and under the two §B nulls that need no
   labels: ``offcomb`` (the half-integer carrier — THE off-comb null) and
   ``mismatch`` against a time-reversed partner (same rate marginal, destroyed
   temporal alignment). ``ridge(none) - ridge(offcomb)`` is the clearance.
3. The ladder's own guards — the blind seed's octave flag and accepted bases,
   the coarse stage's trust mode, and the per-track comb confidence.
4. Self-consistency — the within-window standard deviation of each annotated
   rotor row, which on a hover segment is a direct precision proxy.

One unit = one (recording, window, arm). Artifacts per unit: a small JSON (all
readings) plus a small NPZ (the trajectory on the 0.032 s grid). Aggregation
and the overlay PNGs are :mod:`scripts.blind_corpus_report`.

Smoke (one short window, serial, laptop):
  PYTHONPATH=src python scripts/blind_corpus.py --dataset AVQ-egonoise \
      --recordings S1_seq1 --window-s 8 --max-s 8 --jobs 1 --out results/blind_corpus/smoke

Full corpus (cluster):
  omnirun submit --backend uni-cpu --gpus 0 --cpus 8 --time 8h --yes -- \
      python scripts/blind_corpus.py --dataset AVQ-egonoise --jobs 8
"""

from __future__ import annotations

import os
import sys


def _early_arg(name: str, default: str) -> str:
    """Read one ``--name value`` / ``--name=value`` arg before heavy imports."""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


# Cap BLAS threads BEFORE numpy: the VK solve is BLAS-bound and the grid runs
# one process per core, so an unclamped pool oversubscribes the allocation.
_OMP = _early_arg("--omp", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _OMP)

import argparse  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

# Pin THIS checkout's src ahead of site-packages: the editable install points
# at whichever checkout owns .venv, which on a worktree is not this one.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

import tracking as trk  # noqa: E402
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

SR = 16000
HOP_S = 0.032  # the project-wide trajectory grid (512 / 16000)
MAX_CHANNELS = 8

#: Perturbed siblings scored at the blind trajectory's fixed degrees of freedom.
#: ``(name, multiply, add_rev_s)`` -> candidate = blind * multiply + add.
SIBLINGS: tuple[tuple[str, float, float], ...] = (
    ("half", 0.5, 0.0),
    ("double", 2.0, 0.0),
    ("detune_p03", 1.0, +0.3),
    ("detune_m03", 1.0, -0.3),
    ("detune_p10", 1.0, +1.0),
    ("detune_m10", 1.0, -1.0),
)


# ── data ──────────────────────────────────────────────────────────────────────
def prepare_windows(
    dataset: str,
    wanted: set[str] | None,
    window_s: float,
    overlap_s: float,
    max_s: float | None,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Stream ``dataset``, resample to 16 kHz, cut windows, cache them as NPZ.

    Returns the window index (one dict per window). Caching in the MAIN process
    keeps the grid workers off the network and off the decoder — a worker only
    ever memory-maps one small NPZ.
    """
    from data_processing.frames import get_meta
    from data_processing.streams import iter_published_frames

    cache_dir.mkdir(parents=True, exist_ok=True)
    index_p = cache_dir / "index.json"
    if index_p.exists():
        return json.loads(index_p.read_text())

    hop_s = window_s - overlap_s
    if hop_s <= 0:
        raise SystemExit("--overlap-s must be smaller than --window-s")

    index: list[dict[str, Any]] = []
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
        sr = int(round(float(aud.tindex.sr)))
        if max_s is not None:
            data = data[:, : int(round(max_s * sr))]
        if sr != SR:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=SR, axis=-1, res_type="soxr_hq")
        data = np.ascontiguousarray(data, dtype=np.float32)

        n = data.shape[-1]
        w = int(round(window_s * SR))
        h = int(round(hop_s * SR))
        starts = list(range(0, max(1, n - w + 1), h)) or [0]
        # A tail longer than a fifth of a window earns a backed-up window
        # rather than a short (and therefore differently-conditioned) solve.
        if n - (starts[-1] + w) > w // 5:
            starts.append(max(0, n - w))
        # A rotor track, where one exists, is carried through as a REFERENCE
        # only. The ladder never sees it; it lets a recording with telemetry
        # (DREGON's command-only room2 set) calibrate what the label-free
        # instruments read on an annotation of known accuracy.
        ref_r = ref_t = None
        if "rps" in frame:
            ent = frame["rps"]
            ref_r = np.atleast_2d(np.asarray(ent.data, dtype=np.float64))
            try:
                ref_t = np.asarray(ent.tindex.abs_stamps, dtype=np.float64)
            except AttributeError:
                ref_t = np.arange(ref_r.shape[-1]) / float(ent.tindex.sr)

        for wi, s0 in enumerate(starts):
            seg = data[:, s0 : s0 + w]
            if seg.shape[-1] < int(round(2.0 * SR)):
                continue
            uid = f"{rid}__w{wi:03d}"
            payload: dict[str, np.ndarray] = {"audio": seg}
            if ref_r is not None and ref_t is not None:
                t0, t1 = s0 / SR, (s0 + seg.shape[-1]) / SR
                sel = (ref_t >= t0) & (ref_t < t1)
                if sel.sum() > 4:
                    payload["ref_rps"] = ref_r[:, sel].astype(np.float32)
                    payload["ref_t"] = (ref_t[sel] - t0).astype(np.float32)
            with (cache_dir / f"{uid}.npz").open("wb") as fh:
                np.savez(fh, **payload)  # pyright: ignore[reportArgumentType]
            index.append(
                {
                    "uid": uid,
                    "recording_id": rid,
                    "window": wi,
                    "t0_s": round(s0 / SR, 3),
                    "dur_s": round(seg.shape[-1] / SR, 3),
                    "n_channels": int(seg.shape[0]),
                    "native_sr": sr,
                    "has_reference": "ref_rps" in payload,
                }
            )
        print(f"  {rid}: {n / SR:.1f}s, {data.shape[0]}ch -> {len(starts)} windows", flush=True)

    index_p.write_text(json.dumps(index, indent=1))
    return index


# ── the arms ──────────────────────────────────────────────────────────────────
def build_arm(arm: str, n_rotors: int):
    """Named blind recipe -> a ``Stage``. No script ever assembles a ladder."""
    if arm == "fullrange":
        return trk.blind_fullrange(n_rotors=n_rotors)
    if arm == "vit2dsp":
        return trk.vit2dsp(n_rotors=n_rotors)
    raise ValueError(f"unknown arm {arm!r}; valid: fullrange, vit2dsp")


def _stage_log(frame) -> list[dict[str, Any]]:
    return [dict(e) for e in frame["meta"]["tracking"]]


def _seed_readings(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull the guard-relevant fields out of the ladder's own stage log."""
    out: dict[str, Any] = {}
    for e in log:
        st = str(e.get("stage", ""))
        if "seed" in st:
            out["seed_bases"] = e.get("bases")
            out["seed_octave"] = e.get("octave")
            out["seed_primary"] = e.get("primary")
            out["seed_accepted"] = e.get("accepted_bases")
            out["seed_update_gate"] = e.get("update_gate")
            out["seed_bw_hz"] = e.get("bw_hz")
            out["seed_n_candidates"] = e.get("n_candidates")
        if st == "coarse_init":
            out["coarse_halved"] = e.get("halved")
            out["coarse_bases"] = e.get("bases")
            out["coarse_mode"] = e.get("coarse_mode")
        if st in ("vit2dsp", "guard"):
            for k in ("comb_conf", "conf", "reverted", "n_reverted"):
                if k in e:
                    out[f"{st}_{k}"] = e[k]
    return out


# ── the label-free instruments ────────────────────────────────────────────────
def _fvk_objectives(
    audio: np.ndarray, r: np.ndarray, ft: np.ndarray, k_max: int, alias_penalty: float
) -> dict[str, Any]:
    """F_VK of the blind trajectory and of every sibling, at FIXED cells.

    ``reference=r`` on every call pins the harmonic cap to the blind
    trajectory, so ``n_cells`` is identical across candidates and the numbers
    are comparable (the module's "fixed degrees of freedom by construction").

    Read the two octave siblings differently, because the objective is not
    symmetric in them. The harmonics of ``r/2`` are a SUPERSET of the harmonics
    of ``r`` (every line of ``r`` is an even line of ``r/2``), so a bare
    least-squares comb fit can never prefer ``r`` over ``r/2`` by much — the
    sub-harmonic is structurally favoured and its margin measures the
    objective's own bias, not the evidence. ``2r`` is a SUBSET (it drops the
    odd lines), so ``objective(2r) >> objective(r)`` IS evidence: the odd
    harmonics carry real energy and the annotation is not an octave low.
    ``alias_penalty > 0`` turns on :func:`tracking.alias_charge`, the order
    counter-term that charges a carrier for the lines it models and does not
    find — which is what breaks the ``r/2`` degeneracy.
    """
    from dataclasses import replace as _replace

    cfg = _replace(trk.FVKConfig(), sr=SR, k_max=int(k_max))
    cfg_alias = _replace(cfg, alias_penalty=float(alias_penalty))
    out: dict[str, Any] = {}
    for name, mul, add in (("blind", 1.0, 0.0), *SIBLINGS):
        cand = r * mul + add
        if np.any(cand <= 1.0):  # a non-physical carrier is not a null
            out[name] = {"objective": None, "skipped": "non_physical"}
            continue
        s = trk.fvk_score(audio, SR, cand, ft, cfg, reference=r)
        rec: dict[str, Any] = {
            "objective": s["objective"],
            "r2": s["r2"],
            "n_cells": s.get("n_cells"),
        }
        if name == "blind":
            rec["residual"] = s["residual"]
            rec["k_hi"] = s.get("k_hi")
        # The alias charge only has to separate the octave family.
        if alias_penalty > 0 and name in ("blind", "half", "double"):
            sa = trk.fvk_score(audio, SR, cand, ft, cfg_alias, reference=r)
            rec["objective_alias"] = sa["objective"]
        out[name] = rec
    return out


def _ridge_readings(audio: np.ndarray, r: np.ndarray, ft: np.ndarray) -> dict[str, Any]:
    """Ridge concentration: the blind carrier, its two §B nulls, and the octaves.

    ``ridge`` is line density ON the carrier over a LOCAL floor, so unlike the
    least-squares objective it is not fooled by a comb that models more lines
    than it finds: at ``r/2`` half the modelled teeth land on empty spectrum
    and the density falls. That makes it the independent octave discriminator
    the profiled residual cannot be — measured with the SAME pinned degrees of
    freedom (the reference is the blind trajectory in every call).
    """
    from tracking.fitness import FitnessConfig, score_window

    cfg = FitnessConfig(sr=SR)
    partner = r[:, ::-1].copy()  # same rate marginal, alignment destroyed
    out: dict[str, Any] = {}

    def _read(cand: np.ndarray, control: str, key: str) -> None:
        res = score_window(
            audio,
            ft,
            cand,
            r,  # reference is the blind trajectory: the DOF are pinned to it
            cfg=cfg,
            control=control,
            partner=partner if control == "mismatch" else None,
            n_boot=0,
        )
        if "failed" in res:
            out[key] = {"failed": res["failed"]}
            return
        sc = res["scores"].get("none") or next(iter(res["scores"].values()))
        out[key] = {
            "ridge": sc.get("ridge"),
            "n_cells_ridge": sc.get("n_cells_ridge"),
            "broadband": sc.get("broadband"),
            "phase_noise": sc.get("phase_noise"),
            "snr_median": sc.get("snr_median"),
            "admit_frac_ridge": res["cells"].get("admit_frac_ridge"),
            "line_share_ridge": res["cells"].get("line_share_ridge"),
        }

    for control in ("none", "offcomb", "mismatch"):
        _read(r, control, control)
    for key, mul in (("half", 0.5), ("double", 2.0)):
        cand = r * mul
        if np.any(cand <= 1.0):
            out[key] = {"failed": "non_physical"}
            continue
        _read(cand, "none", key)
    return out


# ── worker ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Worker:
    """One unit of the grid. A module-level callable, NOT a closure: the pool
    pickles the worker to send it to a child process, and a closure has no
    importable qualified name."""

    cache_dir: Path
    n_rotors: int
    k_max: int
    npz_dir: Path
    alias_penalty: float

    def __call__(self, unit: Unit) -> dict[str, Any]:
        cache_dir, n_rotors = self.cache_dir, self.n_rotors
        k_max, npz_dir, alias_penalty = self.k_max, self.npz_dir, self.alias_penalty
        p = unit.params
        blob = np.load(cache_dir / f"{p['uid']}.npz")
        audio = np.ascontiguousarray(blob["audio"], dtype=np.float64)

        frame = trk.tracking_frame(audio, SR, dtype=np.float64)
        arm = str(p["arm"])
        tic = time.perf_counter()
        fallback = None
        try:
            out_frame = build_arm(arm, n_rotors)(frame)
        except ValueError as exc:
            # ``coarse_init``'s takeoff bridge selects the idle rate from
            # ``c_grid <= bridge_idle_c_frac * c_hi``; on a window whose
            # detected cruise rate is low (a pre-takeoff or near-idle slice)
            # that selection is EMPTY and ``argmax`` raises. The stage that
            # exists to serve a ramp is exactly the one that fails on one, so
            # a corpus driver must not die on it: drop the coarse init and
            # run the plain ladder, recording that it did.
            if arm != "fullrange":
                raise
            fallback = f"{type(exc).__name__}: {exc}"
            out_frame = build_arm("vit2dsp", n_rotors)(frame)
        wall_ladder = time.perf_counter() - tic

        r, ft = trk.get_rps(out_frame)
        ft = np.asarray(ft, dtype=np.float64)
        ft = ft - float(ft[0]) if ft.size else ft

        row: dict[str, Any] = {
            **{k: v for k, v in p.items()},
            "wall_ladder_s": round(wall_ladder, 1),
            "arm_fallback": fallback,
            "rps_mean": [round(float(x), 3) for x in r.mean(axis=1)],
            "rps_std": [round(float(x), 4) for x in r.std(axis=1)],
            "rps_min": [round(float(x), 3) for x in r.min(axis=1)],
            "rps_max": [round(float(x), 3) for x in r.max(axis=1)],
            "spread_rev_s": round(float(r.mean(axis=1).max() - r.mean(axis=1).min()), 3),
            **_seed_readings(_stage_log(out_frame)),
        }

        tic = time.perf_counter()
        row["fvk"] = _fvk_objectives(audio, r, ft, k_max, alias_penalty)
        row["wall_fvk_s"] = round(time.perf_counter() - tic, 1)

        tic = time.perf_counter()
        row["ridge"] = _ridge_readings(audio, r, ft)
        row["wall_ridge_s"] = round(time.perf_counter() - tic, 1)

        # Derived headline numbers (the report reads these directly).
        fv = row["fvk"]
        f0 = fv["blind"]["objective"]
        row["fvk_blind"] = f0
        # PRECISION: how steep the well is at +-0.3 rev/s, as a ratio. The
        # local reading, and the one the rate estimate is judged on.
        near = [
            fv[k]["objective"]
            for k in ("detune_p03", "detune_m03")
            if fv.get(k, {}).get("objective") is not None
        ]
        row["fvk_well_p03"] = round(min(near) / f0, 3) if near and f0 else None
        # OCTAVE: 2r drops the odd lines, so a large ratio is real evidence
        # that the annotation is not an octave low. r/2 is a superset and is
        # reported raw, never as a verdict (see _fvk_objectives).
        for key in ("double", "half"):
            o = fv.get(key, {}).get("objective")
            if o is not None and f0:
                row[f"fvk_ratio_{key}"] = round(o / f0, 3)
            oa, ba = fv.get(key, {}).get("objective_alias"), fv["blind"].get("objective_alias")
            if oa is not None and ba:
                row[f"fvk_alias_ratio_{key}"] = round(oa / ba, 3)
        rd = row["ridge"]
        rn = rd.get("none", {})
        for key, tag in (
            ("offcomb", "ridge_clearance_db"),
            ("mismatch", "ridge_clearance_mismatch_db"),
            ("half", "ridge_margin_half_db"),
            ("double", "ridge_margin_double_db"),
        ):
            other = rd.get(key, {}).get("ridge")
            if rn.get("ridge") is not None and other is not None:
                row[tag] = round(rn["ridge"] - other, 3)

        # The reference, where one exists, is a CALIBRATION reading, never an
        # input: PIT-aligned error of the blind annotation against telemetry.
        if "ref_rps" in blob.files:
            from tracking.protocols import pit_align

            ref = np.asarray(blob["ref_rps"], dtype=np.float64)
            ref_t = np.asarray(blob["ref_t"], dtype=np.float64)
            ref_i = np.stack([np.interp(ft, ref_t, row) for row in ref])
            if ref_i.shape[0] == r.shape[0]:
                r_al, _perm = pit_align(r, ref_i, cost="mae")
                err = np.asarray(r_al, dtype=np.float64) - ref_i
                row["ref_mae_rev_s"] = round(float(np.mean(np.abs(err))), 4)
                row["ref_rmse_rev_s"] = round(float(np.sqrt(np.mean(err**2))), 4)
                row["ref_mean_rev_s"] = [round(float(x), 3) for x in ref_i.mean(axis=1)]

        npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_dir / f"{unit.uid}.npz", rps=r.astype(np.float32), ft=ft.astype(np.float32)
        )
        return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _f(key: str) -> list[float]:
        return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

    def _stats(key: str) -> dict[str, Any]:
        v = _f(key)
        return {
            "median": round(float(np.median(v)), 3) if v else None,
            "min": round(float(np.min(v)), 3) if v else None,
            "max": round(float(np.max(v)), 3) if v else None,
            "n": len(v),
        }

    return {
        "n_units": len(rows),
        # A window fails the octave test when the doubled carrier is NOT
        # clearly worse (the odd harmonics carry no energy) or when the
        # half carrier reads a HIGHER line density than the annotation.
        "n_octave_suspect": sum(
            1
            for r in rows
            if (r.get("fvk_ratio_double") is not None and r["fvk_ratio_double"] < 1.2)
            or (r.get("ridge_margin_half_db") is not None and r["ridge_margin_half_db"] < 0)
        ),
        "n_low_clearance": sum(
            1
            for r in rows
            if isinstance(r.get("ridge_clearance_db"), (int, float))
            and r["ridge_clearance_db"] < 1.0
        ),
        "ridge_clearance_db": _stats("ridge_clearance_db"),
        "ridge_clearance_mismatch_db": _stats("ridge_clearance_mismatch_db"),
        "ridge_margin_half_db": _stats("ridge_margin_half_db"),
        "fvk_well_p03": _stats("fvk_well_p03"),
        "fvk_ratio_double": _stats("fvk_ratio_double"),
        "fvk_ratio_half": _stats("fvk_ratio_half"),
        "wall_s_total": round(
            sum(_f("wall_ladder_s")) + sum(_f("wall_fvk_s")) + sum(_f("wall_ridge_s")), 1
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--dataset", default="AVQ-egonoise", help="published tdframe-v1 dataset")
    ap.add_argument("--recordings", default=None, help="comma-separated recording_id filter")
    ap.add_argument("--arms", default="fullrange", help="comma-separated: fullrange,vit2dsp")
    ap.add_argument("--n-rotors", type=int, default=4)
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--overlap-s", type=float, default=4.0)
    ap.add_argument("--max-s", type=float, default=None, help="cap seconds per recording (smoke)")
    ap.add_argument("--k-max", type=int, default=40, help="F_VK harmonic cap")
    ap.add_argument(
        "--alias-penalty",
        type=float,
        default=1.0,
        help="weight of the F_VK order/alias counter-term in the octave readings (0 = off)",
    )
    ap.add_argument("--windows", default=None, help="comma-separated window indices to keep")
    ap.add_argument("--out", default="results/blind_corpus/run")
    ap.add_argument("--omp", default="1", help="BLAS thread cap (read pre-import)")
    add_gridrun_args(ap, jobs=4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    cache_dir = out_dir / "windows"
    wanted = (
        {r.strip() for r in args.recordings.split(",") if r.strip()} if args.recordings else None
    )
    print(f"Preparing windows from {args.dataset} ...", flush=True)
    index = prepare_windows(
        args.dataset, wanted, args.window_s, args.overlap_s, args.max_s, cache_dir
    )
    if not index:
        raise SystemExit(f"no windows produced from {args.dataset}")
    if args.windows:
        keep = {int(w) for w in args.windows.split(",")}
        index = [e for e in index if e["window"] in keep]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    units = [
        Unit(uid=f"{e['uid']}__{arm}", params={**e, "arm": arm, "dataset": args.dataset})
        for e in index
        for arm in arms
    ]
    print(f"{len(units)} units ({len(index)} windows x {len(arms)} arms)", flush=True)

    res = gridrun_from_args(
        args,
        units,
        Worker(cache_dir, args.n_rotors, args.k_max, out_dir / "traj", args.alias_penalty),
        out_dir,
        blas_threads=int(args.omp),
        summarize=summarize,
    )
    raise SystemExit(res.exit_code)


if __name__ == "__main__":
    main()
