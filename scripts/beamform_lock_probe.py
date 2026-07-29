#!/usr/bin/env python3
"""Beamform lock probe — does coherent mic combination restore per-harmonic lock?

Closing measurement of the phase-coherence budget (vk_phase_validation ladder
follow-up). The ladder measured lock_k = |mean_t z_k| / mean_t |z_k| of the
demodulated envelope z_k = LP[x e^{-i k phi}] at telemetry-truth phase and
found free-flight audio locked at ~0.1 (k=1-2) / 0.02-0.05 (k>=5) — but those
were effectively single-channel numbers. This probe asks whether *array gain*
changes the verdict: per (window, rotor, k) it scores four channel treatments

  ch0         channel 0 only (the ladder's baseline reference),
  best_mic    max single-channel lock over the 8 mics (per rotor and k),
  das         nearfield delay-and-sum toward the rotor: per-mic advance
              tau_m = |mic_m - rotor_r| / c applied as an rfft-domain phase
              ramp, mics summed, then demodulated,
  self_steer  per-harmonic self-steered coherent sum: per-mic envelopes
              z_k^(m) combined with w_m = conj(mean_t z_k^(m)) /
              |mean_t z_k^(m)| — the weights cheat by using the data mean, so
              this is an UPPER BOUND on any coherent-combination gain
              (MVDR-lite included).

Data: every cruise window of the frozen ``beatvk-valid-raw`` protocol dataset
(3 DREGON free-flight room1 recordings + FLY124; 16 s manifest windows), audio
resampled to 16 kHz here. GT shaft phase per rotor = the RAW telemetry
linearly interpolated to audio rate, phi_r = 2 pi cumsum(r) / sr — no
smoothing, no refinement (telemetry-truth, as in the ladder's S4).

Positive control: DREGON ``motor_Motor1_70`` loaded exactly like the ladder's
S3 (mid-recording 20 s, all 8 mics), scored under two phase references:
``nominal`` (the constant setpoint — known-dead, lock ~0.01, kept as the
floor) and ``iter_warp`` (warp-refined from nominal, the ladder's best
real-motor reference, lock1 ~0.7). The control shows what each treatment does
when partial coherence IS present.

Caveat: the das treatment assumes telemetry rotor order == geometry rotor
order (DREGON motors_measured rows <-> rotorsPos rows; FLY124 rps rows <->
michaels ROTOR_ORDER) — the project-wide convention. ch0 / best_mic /
self_steer do not use geometry at all.

Output: ``results/beamform_lock_probe/lock_table.csv`` (pool, recording,
window, rotor, k, treatment, lock, ...) + ``summary.json`` + a stdout table
of mean lock per treatment x k pooled over (i) DREGON cruise windows,
(ii) FLY124 cruise windows, (iii) the motor control, and the explicit
array-gain conclusion at k <= 5.

Run::

    python scripts/beamform_lock_probe.py            # all cruise windows
    python scripts/beamform_lock_probe.py --max-windows-per-rec 2
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (the runtime-guard convention; this
# also preempts vk_phase_validation's own setdefault(2) on import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import beatvk_eval  # noqa: E402
from vk_phase_validation import (  # noqa: E402
    ENV_TRIM_S,
    EVAL_ENV_CFG,
    F_MAX,
    SR,
    build_s3_cells,
)

from data_processing.frames import resample_audio_series  # noqa: E402
from data_processing.vk_tracking import demodulate  # noqa: E402
from data_processing.warp_refinement import iter_warp_refine  # noqa: E402

KS = (1, 2, 3, 4, 5, 8, 10, 16, 20, 40)
TREATMENTS = ("ch0", "best_mic", "das", "self_steer")
C_SOUND = 343.0  # m/s
MIN_ROTOR_RPS = 5.0  # skip rotors that are effectively off within a window
CONTROL_RID = "motor_Motor1_70"
CSV_FIELDS: tuple[str, ...] = (
    "pool",
    "recording",
    "window",
    "rotor",
    "k",
    "treatment",
    "lock",
    "ref",
    "mean_rps",
)
PER_WINDOW_EST_S = 12.0  # rough per-window wall estimate for the runtime guard
BUDGET_S = 1800.0  # ~30 min local budget; beyond it windows are thinned


# ---------------------------------------------------------------------------
# Geometry


def dregon_geometry(dregon_dir: str) -> tuple[np.ndarray, np.ndarray]:
    from data_processing.dregon import get_geometry
    from data_processing.streams import resolve_source

    return get_geometry(Path(resolve_source(dregon_dir)))


def fly124_geometry() -> tuple[np.ndarray, np.ndarray]:
    from data_processing.michaels import get_geometry

    return get_geometry()


# ---------------------------------------------------------------------------
# Core measurement


def _lock(z: np.ndarray) -> np.ndarray:
    """lock = |mean_t z| / mean_t |z| along the last (time) axis."""
    return np.abs(z.mean(axis=-1)) / np.maximum(np.abs(z).mean(axis=-1), 1e-30)


def _env_trim(z: np.ndarray) -> np.ndarray:
    stride = max(1, int(round(SR / EVAL_ENV_CFG.fs_env)))
    n_trim = int(round(ENV_TRIM_S * SR / stride))
    return z[..., n_trim : z.shape[-1] - n_trim]


def _delay_and_sum(audio: np.ndarray, mic_pos: np.ndarray, rotor_xyz: np.ndarray) -> np.ndarray:
    """Nearfield delay-and-sum toward one rotor: advance mic m by tau_m, sum.

    ``x_m(t) ~ s(t - tau_m)`` with ``tau_m = |mic_m - rotor| / c``; aligning
    means evaluating ``x_m(t + tau_m)``, i.e. an rfft-domain phase ramp
    ``X_m(f) e^{+2 pi i f tau_m}`` (fractional delay; the circular wrap is
    ~0.5 ms at the edges, well inside the 1 s envelope trim). Relative delays
    only matter, so tau is re-zeroed at its minimum.
    """
    tau = np.linalg.norm(mic_pos - rotor_xyz[None, :], axis=-1) / C_SOUND
    tau -= tau.min()
    n = audio.shape[-1]
    spec = np.fft.rfft(audio, axis=-1)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec *= np.exp(2j * np.pi * freqs[None, :] * tau[:, None])
    return np.asarray(np.fft.irfft(spec.sum(axis=0), n=n))


def treatment_locks(
    audio: np.ndarray,
    r_by_rotor: dict[int, np.ndarray],
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
) -> list[dict[str, Any]]:
    """Per (rotor, k, treatment) lock rows for one window.

    ``audio``: (C, T) at SR. ``r_by_rotor``: rotor index -> (T,) audio-rate
    RPS (telemetry-truth or a recovered track). Demodulation happens once per
    (rotor, k, mic); every treatment is a cheap combination afterwards.
    """
    tracks: list[tuple[int, int]] = []
    phases: list[np.ndarray] = []
    for rotor, r_aud in sorted(r_by_rotor.items()):
        phi = 2.0 * np.pi * np.cumsum(r_aud) / SR
        mean_r = float(r_aud.mean())
        for k in KS:
            if k * mean_r <= F_MAX:
                tracks.append((rotor, k))
                phases.append(k * phi)
    if not tracks:
        return []

    z = _env_trim(demodulate(audio, np.stack(phases), EVAL_ENV_CFG))  # (C, M, T_env)
    lock_pc = _lock(z)  # (C, M) per-channel locks

    # self_steer: per-mic mean-phasor weights (data-cheating upper bound).
    zbar = z.mean(axis=-1)  # (C, M)
    w = np.conj(zbar) / np.maximum(np.abs(zbar), 1e-30)
    lock_ss = _lock((w[..., None] * z).sum(axis=0))  # (M,)

    # das: one delayed-sum virtual channel per rotor, demodulated over all
    # tracks at once; only the (rotor's channel, rotor's tracks) cells count.
    rotors = sorted(r_by_rotor)
    das = np.stack([_delay_and_sum(audio, mic_pos, rotor_pos[r]) for r in rotors])
    lock_das = _lock(_env_trim(demodulate(das, np.stack(phases), EVAL_ENV_CFG)))  # (R, M)

    rows: list[dict[str, Any]] = []
    for m, (rotor, k) in enumerate(tracks):
        locks = {
            "ch0": float(lock_pc[0, m]),
            "best_mic": float(lock_pc[:, m].max()),
            "das": float(lock_das[rotors.index(rotor), m]),
            "self_steer": float(lock_ss[m]),
        }
        rows.extend({"rotor": rotor, "k": k, "treatment": t, "lock": locks[t]} for t in TREATMENTS)
    return rows


# ---------------------------------------------------------------------------
# Free-flight windows (beatvk-valid-raw)


def _thin_windows(windows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if cap <= 0 or len(windows) <= cap:
        return windows
    idx = np.unique(np.round(np.linspace(0, len(windows) - 1, cap)).astype(int))
    return [windows[i] for i in idx]


def probe_recording(
    rec: dict[str, Any],
    geometry: tuple[np.ndarray, np.ndarray],
    max_windows: int,
) -> list[dict[str, Any]]:
    rid = rec["recording_id"]
    pool = "fly124_cruise" if rid == beatvk_eval.FLY124_REC else "dregon_cruise"
    mic_pos, rotor_pos = geometry
    audio16 = resample_audio_series(rec["audio"], SR)
    data = np.atleast_2d(np.asarray(audio16.data, dtype=np.float32))
    ts, vals = rec["ts"], rec["vals"]
    n_win = int(round(beatvk_eval.FRAME_S * 500 * SR))  # 16 s at SR

    cruise = [w for w in rec["windows"] if str(w["regime"]) == "cruise"]
    windows = _thin_windows(cruise, max_windows)
    if len(windows) < len(cruise):
        print(f"  [{rid}] thinned {len(cruise)} -> {len(windows)} cruise windows", flush=True)

    rows: list[dict[str, Any]] = []
    for w in windows:
        tic = time.perf_counter()
        i0 = int(round(float(w["start_s"]) * SR))
        seg = data[:, i0 : i0 + n_win]
        t_abs = (i0 + np.arange(seg.shape[-1])) / SR
        r_by_rotor: dict[int, np.ndarray] = {}
        for r in range(vals.shape[0]):
            r_aud = np.interp(t_abs, ts, vals[r])
            if float(r_aud.mean()) >= MIN_ROTOR_RPS:
                r_by_rotor[r] = r_aud
        for row in treatment_locks(seg, r_by_rotor, mic_pos, rotor_pos):
            rows.append(
                {
                    "pool": pool,
                    "recording": rid,
                    "window": int(w["index"]),
                    "ref": "telemetry",
                    "mean_rps": float(w["mean_rps"]),
                    **row,
                }
            )
        print(
            f"  [{rid}] window {int(w['index'])} ({float(w['mean_rps']):.1f} rev/s, "
            f"{len(r_by_rotor)} rotors): {time.perf_counter() - tic:.1f} s",
            flush=True,
        )
    return rows


# ---------------------------------------------------------------------------
# Positive control (DREGON single motor, the ladder's S3 loading)


def control_rows(dregon_dir: str, geometry: tuple[np.ndarray, np.ndarray]) -> list[dict[str, Any]]:
    mic_pos, rotor_pos = geometry
    cells = build_s3_cells(dregon_dir, quick=True)  # ladder's S3 loader (quick set)
    cell = next((c for c in cells if c.cell_id == CONTROL_RID), None)
    if cell is None:
        print(f"[control] {CONTROL_RID} not found under {dregon_dir} — control skipped")
        return []
    rotor_geom = int(cell.meta["motor"]) - 1  # MotorN -> rotorsPos row N-1
    t_aud = np.arange(cell.audio.shape[-1]) / SR

    refs: dict[str, np.ndarray] = {"nominal": cell.r_init_base}
    tic = time.perf_counter()
    r_warp, _ = iter_warp_refine(cell.audio, cell.r_init_base, cell.ft, sr=SR)
    refs["iter_warp"] = r_warp
    print(f"[control] iter_warp refine: {time.perf_counter() - tic:.1f} s", flush=True)

    rows: list[dict[str, Any]] = []
    for ref, r_ft in refs.items():
        r_aud = np.interp(t_aud, cell.ft, r_ft[0])
        for row in treatment_locks(cell.audio, {rotor_geom: r_aud}, mic_pos, rotor_pos):
            rows.append(
                {
                    "pool": "motor_control",
                    "recording": CONTROL_RID,
                    "window": -1,
                    "ref": ref,
                    "mean_rps": float(r_aud.mean()),
                    **row,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Pooling / reporting


def pooled_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[int, float]]]:
    """pool key -> treatment -> k -> mean lock (control split by phase ref)."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    keys = sorted(
        {
            (r["pool"] if r["pool"] != "motor_control" else f"motor_control[{r['ref']}]")
            for r in rows
        }
    )
    for key in keys:
        sub = [
            r
            for r in rows
            if (r["pool"] if r["pool"] != "motor_control" else f"motor_control[{r['ref']}]") == key
        ]
        table: dict[str, dict[int, float]] = {}
        for t in TREATMENTS:
            table[t] = {
                k: float(np.mean(vals))
                for k in KS
                if (vals := [r["lock"] for r in sub if r["treatment"] == t and r["k"] == k])
            }
        out[key] = table
    return out


def print_pool_tables(pooled: dict[str, dict[str, dict[int, float]]]) -> None:
    for pool, table in pooled.items():
        n_ks = sorted({k for t in TREATMENTS for k in table[t]})
        print(f"\n{pool} — mean lock_k per treatment")
        header = f"{'k':>4}" + "".join(f"{t:>12}" for t in TREATMENTS)
        print(header)
        print("-" * len(header))
        for k in n_ks:
            cells = "".join(
                f"{table[t][k]:>12.3f}" if k in table[t] else f"{'—':>12}" for t in TREATMENTS
            )
            print(f"{k:>4}{cells}")


def print_conclusion(pooled: dict[str, dict[str, dict[int, float]]]) -> None:
    print("\n=== Array-gain conclusion ===")
    free_pools = [p for p in pooled if p in ("dregon_cruise", "fly124_cruise")]
    best = 0.0
    for pool in free_pools:
        for k in (1, 2, 5):
            ch0 = pooled[pool]["ch0"].get(k, float("nan"))
            ss = pooled[pool]["self_steer"].get(k, float("nan"))
            best = max(best, ss)
            print(
                f"{pool} k={k}: ch0 {ch0:.3f} -> self-steered upper bound {ss:.3f} "
                f"(x{ss / max(ch0, 1e-9):.1f})"
            )
    verdict = (
        "coherent combination RESTORES phase evidence"
        if best >= 0.3
        else "even the data-cheating self-steered upper bound stays below ~0.3 — "
        "multichannel coherent combination does NOT restore per-harmonic phase "
        "lock on free-flight audio; the impossibility argument stands"
    )
    print(f"Max free-flight lock at k<=5 under the self-steered upper bound: {best:.3f}")
    print(f"Verdict: {verdict}")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--out", default="results/beamform_lock_probe", help="output directory")
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--dregon-dir", default="data/DREGON", help="path or dload:DREGON (geometry)")
    ap.add_argument(
        "--max-windows-per-rec",
        type=int,
        default=0,
        help="cap cruise windows per recording (0 = all, subject to the runtime guard)",
    )
    ap.add_argument("--skip-control", action="store_true", help="skip the single-motor control")
    args = ap.parse_args()

    tic = time.perf_counter()
    recs = beatvk_eval.load_recordings(args.dataset_version, None, keep_audio=True)
    print(
        f"[beamform_lock_probe] {beatvk_eval.DATASET}@{recs[0]['dataset_version'][:12]}: "
        f"{[r['recording_id'] for r in recs]}",
        flush=True,
    )

    # Runtime guard: estimate before demodulating anything; if the full cruise
    # set exceeds the ~30 min local budget, thin to the representative subset
    # (2 windows per recording) as specified.
    n_cruise = {
        r["recording_id"]: sum(1 for w in r["windows"] if str(w["regime"]) == "cruise")
        for r in recs
    }
    cap = args.max_windows_per_rec
    est = sum(n_cruise.values()) * PER_WINDOW_EST_S
    if cap <= 0 and est > BUDGET_S:
        cap = 2
        print(
            f"[guard] {sum(n_cruise.values())} cruise windows ~{est / 60:.0f} min > "
            f"{BUDGET_S / 60:.0f} min budget — thinning to 2 windows per recording",
            flush=True,
        )
    print(f"[beamform_lock_probe] cruise windows: {n_cruise}", flush=True)

    geom_dregon = dregon_geometry(args.dregon_dir)
    geom_fly124 = fly124_geometry()

    rows: list[dict[str, Any]] = []
    for rec in recs:
        geometry = geom_fly124 if rec["recording_id"] == beatvk_eval.FLY124_REC else geom_dregon
        rows.extend(probe_recording(rec, geometry, cap))
        rec["audio"] = None  # free the native-rate audio

    if not args.skip_control:
        rows.extend(control_rows(args.dregon_dir, geom_dregon))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lock_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["pool"], r["recording"], r["window"], r["rotor"])):
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    pooled = pooled_means(rows)
    print_pool_tables(pooled)
    print_conclusion(pooled)

    wall = time.perf_counter() - tic
    summary = {
        "dataset": {"name": beatvk_eval.DATASET, "version": recs[0]["dataset_version"]},
        "ks": list(KS),
        "treatments": list(TREATMENTS),
        "env_trim_s": ENV_TRIM_S,
        "control": None if args.skip_control else CONTROL_RID,
        "windows_per_rec_cap": cap,
        "n_rows": len(rows),
        "pooled_mean_lock": {
            pool: {t: {str(k): v for k, v in table[t].items()} for t in TREATMENTS}
            for pool, table in pooled.items()
        },
        "wall_s": round(wall, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ndone: {len(rows)} rows -> {out_dir}/lock_table.csv + summary.json ({wall:.0f} s)")


if __name__ == "__main__":
    main()
