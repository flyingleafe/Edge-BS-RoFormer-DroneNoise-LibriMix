#!/usr/bin/env python3
"""Frozen regression reference for the VK + pi_kalman tracking path.

The optimization campaign (issues #15/#16/#17) rewrites the hot inner loops
of ``tracking.vk_tracking`` and ``tracking.phase_increment_tracker``. Those
rewrites must not change what the pipeline computes, and the 15-window
protocol is far too slow to answer that question per commit. So this script
freezes ONE deterministic clip and dumps the intermediate arrays of ONE
flagship application, and can re-run the same computation later and diff it.

The clip: the first window of the first DREGON recording of the ``beatvk``
protocol (``tracking.protocols.BEATVK``), sliced and resampled exactly as
``scripts/beatvk_vk_arms.build_preps`` does (native -> 16 kHz soxr_hq, window
bounds from the frozen dataset manifest, 0.032 s frame grid).

The computation: the ``peeled`` arm of ``scripts/beatvk_flagship.py``, ONE
application, started from the window's RAW telemetry. Telemetry is the init
on purpose — the blind_fullrange chain that the flagship uses is expensive
and is not what this reference guards; the guarded surface is the VK envelope
solve, the least-squares peel projection, and the pi_kalman pass. Nothing is
re-implemented here: the arrays come from ``vk_envelopes`` /
``ls_project_envelopes`` / ``beatvk_flagship.run_arm``.

Run::

    python scripts/tracking_ref.py --capture results/tracking_ref
    python scripts/tracking_ref.py --compare results/tracking_ref
    python scripts/tracking_ref.py --compare results/tracking_ref --exact
    python scripts/tracking_ref.py --capture /tmp/ref --seconds 2   # smoke

Remote (the capture is minutes of CPU, so this is the normal path)::

    omnirun submit --backend apocrita-cpu --gpus 0 --time 2h --yes \\
        --env PYTHONPATH=src -- \\
        python scripts/tracking_ref.py --capture results/tracking_ref
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import — the beatvk_flagship convention, and
# part of what makes the reference reproducible across machines.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import beatvk_eval  # noqa: E402
import beatvk_flagship as flag  # noqa: E402

from tracking.protocols import (  # noqa: E402
    BEATVK,
    BEATVK_DREGON_RECS,
    WindowSpec,
    iter_windows,
    to_frame,
)

REF_NAME = "tracking_ref.npz"
#: Arrays whose exact values are the regression surface.
DUMPED = ("env_z", "env_x", "env_x_ls", "env_valid", "env_bw_track", "env_t_env", "r0", "r_next")


# ---------------------------------------------------------------------------
# the frozen clip


def load_window(
    rid: str | None, widx: int | None, *, version: str | None, seconds: float | None
) -> tuple[Any, WindowSpec, dict[str, Any]]:
    """Stream the protocol dataset and build the tracking frame of one window.

    Defaults to the FIRST window of the FIRST DREGON recording of ``beatvk``.
    Returns ``(frame, spec, provenance)``.
    """
    rid = rid or BEATVK_DREGON_RECS[0]
    recs = beatvk_eval.load_recordings(version, {rid}, keep_audio=True)
    rec = recs[0]
    specs = list(iter_windows(BEATVK, {rid: {"windows": rec["windows"]}}))
    spec = specs[0] if widx is None else next(s for s in specs if s.index == widx)

    from data_processing.frames import resample_audio_series

    # The protocol resample (native -> 16 kHz, librosa soxr_hq) then window
    # slicing by sample index — beatvk_vk_arms.build_preps, verbatim.
    audio16 = np.atleast_2d(
        np.asarray(resample_audio_series(rec["audio"], flag.SR).data, dtype=np.float32)
    )
    start, end = float(spec.start_s or 0.0), float(spec.end_s or 0.0)
    a0, a1 = int(round(start * flag.SR)), int(round(end * flag.SR))
    if seconds is not None:  # smoke runs only: a shorter clip is a DIFFERENT reference
        a1 = min(a1, a0 + int(round(seconds * flag.SR)))
    seg = audio16[:, a0:a1]
    ft = np.arange(0.0, (a1 - a0) / flag.SR - flag.FRAME_S / 2, flag.FRAME_S)
    r_meas = np.stack(
        [np.interp(ft + start, rec["ts"], rec["vals"][i]) for i in range(flag.N_ROTORS)]
    )
    frame = to_frame(
        seg,
        flag.SR,
        spec,
        rps=r_meas,
        frame_times=ft,
        rps_meas=r_meas,
        meta={"init": "raw_telemetry"},
    )
    prov = {
        "dataset": beatvk_eval.DATASET,
        "dataset_version": rec["dataset_version"],
        "recording_id": rid,
        "window_index": spec.index,
        "regime": spec.regime,
        "start_s": start,
        "end_s": end,
        "clip_seconds": (a1 - a0) / flag.SR,
        "n_channels": int(seg.shape[0]),
        "n_frames": int(len(ft)),
    }
    return frame, spec, prov


# ---------------------------------------------------------------------------
# the computation


def run_reference(frame: Any, *, peel_mode: str, channels: int) -> tuple[dict[str, Any], dict]:
    """One flagship application on ``frame``: arrays + the config that made them."""
    from tracking.stages import get_audio, get_rps
    from tracking.vk_tracking import VKConfig, ls_project_envelopes, vk_envelopes

    audio, sr = get_audio(frame)
    clip = np.asarray(audio[:channels], dtype=np.float64)
    r0, ft = get_rps(frame)

    # (a) the envelope solve the peel runs at the current track.
    cfg = VKConfig(
        fs=float(sr), bw_hz=flag.PEEL_BW_HZ, k_max=flag.PEEL_K_MAX, f_max=6000.0, n_outer=1
    )
    t_aud = np.arange(clip.shape[-1]) / sr
    r_aud = np.vstack([np.interp(t_aud, ft, r0[r]) for r in range(flag.N_ROTORS)])
    tic = time.perf_counter()
    env = vk_envelopes(clip, r_aud, cfg)
    wall_env = time.perf_counter() - tic

    # (b) the least-squares re-projection of those envelopes onto the clip.
    tic = time.perf_counter()
    env_ls, ls_diag = ls_project_envelopes(clip, env)
    wall_ls = time.perf_counter() - tic

    # (c) the whole application (peel + one pi_kalman pass), through the
    # flagship's own entry point so this can never drift from what it runs.
    tic = time.perf_counter()
    iters, app_diag = run_arm_one(clip, r0, ft, peel_mode)
    wall_app = time.perf_counter() - tic

    arrays = {
        "env_z": env.z,
        "env_x": env.x,
        "env_x_ls": env_ls.x,
        "env_valid": env.valid,
        "env_bw_track": env.bw_track,
        "env_t_env": env.t_env,
        "env_rotor": env.rotor,
        "env_k": env.k,
        "ft": ft,
        "r0": r0,
        "r_next": iters[1],
    }
    config = {
        "vk": {
            "fs": cfg.fs,
            "bw_hz": cfg.bw_hz,
            "k_min": cfg.k_min,
            "k_max": cfg.k_max,
            "f_min": cfg.f_min,
            "f_max": cfg.f_max,
            "n_outer": cfg.n_outer,
            "fs_env": cfg.fs_env,
            "lp_mode": cfg.lp_mode,
            "solver": cfg.solver,
            "couple_hz": cfg.couple_hz,
            "prune_far_pairs": cfg.prune_far_pairs,
            "p": cfg.p,
        },
        "peel": {"mode": peel_mode, "bw_hz": flag.PEEL_BW_HZ, "k_max": flag.PEEL_K_MAX},
        "pi_kalman": {
            "n_iter": flag.PI_N_ITER,
            "band_hz": flag.PI_BAND_HZ,
            "pair_mode": flag.PI_PAIR_MODE,
            "variant": "protocol",
        },
        "channels": int(clip.shape[0]),
        "arm": "peeled",
        "n_applications": 1,
        "ls_diag": ls_diag,
        "app_diag": app_diag,
        "wall_s": {
            "vk_envelopes": round(wall_env, 2),
            "ls_project": round(wall_ls, 2),
            "application": round(wall_app, 2),
        },
    }
    return arrays, config


def run_arm_one(
    clip: np.ndarray, r0: np.ndarray, ft: np.ndarray, peel_mode: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """``beatvk_flagship.run_arm``, one peeled application, no file plumbing."""
    return flag.run_arm(clip, r0, ft, "peeled", 1, "tracking_ref", peel_mode=peel_mode)


# ---------------------------------------------------------------------------
# capture / compare


def capture(out: Path, arrays: dict[str, Any], config: dict, prov: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / REF_NAME
    np.savez(
        path,
        allow_pickle=False,
        config=np.str_(json.dumps(config, sort_keys=True)),
        provenance=np.str_(json.dumps(prov, sort_keys=True)),
        **arrays,
    )
    (out / "tracking_ref.json").write_text(
        json.dumps({"provenance": prov, "config": config}, indent=2, sort_keys=True)
    )
    return path


def compare(out: Path, arrays: dict[str, Any], config: dict, exact: bool) -> int:
    """Diff fresh arrays against the stored reference. Returns an exit code."""
    path = out / REF_NAME
    if not path.is_file():
        raise SystemExit(f"no reference at {path} — run --capture first")
    failures: list[str] = []
    with np.load(path) as z:
        stored_cfg = json.loads(str(z["config"]))
        for key in _config_diff(stored_cfg, config):
            print(f"  CONFIG CHANGED  {key}")
        print(f"{'array':<16}{'shape':>18}{'max|abs|':>14}{'max|rel|':>14}   verdict")
        print("-" * 78)
        for name in DUMPED:
            if name not in z:
                failures.append(f"{name}: missing from the stored reference")
                continue
            ref = np.asarray(z[name])
            new = np.asarray(arrays[name])
            if ref.shape != new.shape:
                failures.append(f"{name}: shape {ref.shape} -> {new.shape}")
                print(f"{name:<16}{str(ref.shape):>18}{'—':>14}{'—':>14}   SHAPE CHANGED")
                continue
            d_abs, d_rel = _diffs(ref, new)
            ok = np.array_equal(ref, new) if exact else _close(ref, new)
            verdict = "identical" if np.array_equal(ref, new) else ("close" if ok else "DIFFERENT")
            print(f"{name:<16}{str(ref.shape):>18}{d_abs:>14.3e}{d_rel:>14.3e}   {verdict}")
            if not ok:
                failures.append(f"{name}: max abs {d_abs:.3e}, max rel {d_rel:.3e}")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        "\nOK — the pipeline reproduces the stored reference"
        + (" bit-for-bit" if exact else " within tolerance")
    )
    return 0


def _diffs(ref: np.ndarray, new: np.ndarray) -> tuple[float, float]:
    if ref.dtype == bool:
        n = float(np.count_nonzero(ref != new))
        return n, n / max(ref.size, 1)
    d = np.abs(np.asarray(new, dtype=np.complex128) - np.asarray(ref, dtype=np.complex128))
    scale = np.maximum(np.abs(np.asarray(ref, dtype=np.complex128)), 1e-300)
    return float(d.max()), float((d / scale).max())


def _close(ref: np.ndarray, new: np.ndarray) -> bool:
    if ref.dtype == bool:
        return bool(np.array_equal(ref, new))
    return bool(np.allclose(new, ref, rtol=1e-5, atol=1e-8))


def _config_diff(stored: dict, fresh: dict) -> list[str]:
    """Keys whose values differ, ignoring the timing/diagnostic blocks."""
    skip = {"wall_s", "ls_diag", "app_diag"}
    out: list[str] = []
    for key in sorted(set(stored) | set(fresh)):
        if key in skip:
            continue
        if stored.get(key) != fresh.get(key):
            out.append(f"{key}: {stored.get(key)!r} -> {fresh.get(key)!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", metavar="OUTDIR", help="run and store the reference")
    mode.add_argument("--compare", metavar="OUTDIR", help="re-run and diff against the reference")
    ap.add_argument("--exact", action="store_true", help="--compare: demand bit-identical arrays")
    ap.add_argument(
        "--recording", default=None, help="override the recording (default: first DREGON)"
    )
    ap.add_argument(
        "--window", type=int, default=None, help="override the window index (default: first)"
    )
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--channels", type=int, default=8, help="mic channels (<= 8)")
    ap.add_argument(
        "--peel-mode",
        default=flag.DEFAULT_PEEL_MODE,
        choices=list(flag.PEEL_MODES),
        help="peel subtraction mode (must match the stored reference)",
    )
    ap.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="truncate the clip (smoke runs only — a truncated clip is its own reference)",
    )
    opts = ap.parse_args()

    out = Path(opts.capture or opts.compare)
    tic = time.perf_counter()
    frame, spec, prov = load_window(
        opts.recording, opts.window, version=opts.dataset_version, seconds=opts.seconds
    )
    prov["channels"] = min(opts.channels, prov["n_channels"])
    prov["truncated_to_s"] = opts.seconds
    print(
        f"[tracking_ref] {prov['dataset']}@{prov['dataset_version'][:12]} "
        f"{spec.name} ({spec.regime}, {prov['clip_seconds']:.1f} s, "
        f"{prov['channels']} mics) loaded in {time.perf_counter() - tic:.0f}s",
        flush=True,
    )

    arrays, config = run_reference(frame, peel_mode=opts.peel_mode, channels=opts.channels)
    print(f"[tracking_ref] walls: {config['wall_s']}", flush=True)

    if opts.capture:
        path = capture(out, arrays, config, prov)
        print(f"[tracking_ref] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
        for name in DUMPED:
            a = np.asarray(arrays[name])
            print(f"  {name:<16}{str(a.shape):>18}  {a.dtype}")
        return
    raise SystemExit(compare(out, arrays, config, opts.exact))


if __name__ == "__main__":
    main()
