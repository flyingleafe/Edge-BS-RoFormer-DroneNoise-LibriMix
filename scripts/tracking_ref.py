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
    python scripts/tracking_ref.py --bench                          # per-stage ms
    python scripts/tracking_ref.py --bench --bench-workers 1,4 --bench-vk

``--self-check`` needs no stored reference: it runs the SAME application
twice in one process — once on cpu/exact, once on whatever
``--device`` / ``--pad-mode`` select — and diffs the two.
That is the form the GPU verification takes, because the frozen ``.npz`` is
~100 MB and does not travel to a compute node::

    python scripts/tracking_ref.py --self-check --device cuda

Remote (the capture is minutes of CPU, so this is the normal path)::

    omnirun submit --backend apocrita-cpu --gpus 0 --time 2h \\
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

from tracking.dsp import PAD_MODES, dsp_config, thread_pool  # noqa: E402
from tracking.protocols import (  # noqa: E402
    BEATVK,
    BEATVK_DREGON_RECS,
    WindowSpec,
    iter_windows,
    slice_window,
    to_frame,
)

REF_NAME = "tracking_ref.npz"
#: Arrays whose exact values are the regression surface.
DUMPED = ("env_z", "env_x", "env_x_ls", "env_valid", "env_bw_track", "env_t_env", "r0", "r_next")

#: Tolerance-mode bar per dumped array: ``("scale", f)`` means ``f`` times
#: the array's own scale (``max |ref|``), ``("abs", v)`` means ``v`` in the
#: array's own units. Boolean arrays ignore this and demand zero flips —
#: a gate decision must never move.
#:
#: ``env_x`` / ``env_x_ls`` get a looser bar than ``env_z`` on purpose, and
#: the gap is not slack: the VK normal equations at ``bw_hz = 1`` carry
#: ``rho^2 ~ 4e5``, so the assembled system's condition number is ~1e7 and
#: the solve AMPLIFIES the demodulation's complex64 rounding (~1e-7 of
#: scale, which is what ``env_z`` shows) by one to three orders, and by how
#: much depends on the clip. Measured for the scipy->torch consolidation:
#: ``env_z`` moves 1.5e-7 of scale on the full 16 s clip (1.2e-7 on the 4 s
#: smoke), ``env_x`` 7.3e-7 (3.7e-5 on the smoke), ``r_next`` 4.5e-6 rev/s,
#: and no gate flips anywhere. The tracker consumes ``r_next`` and the
#: gates, so those carry the tight bars.
TOL: dict[str, tuple[str, float]] = {
    "env_z": ("scale", 1e-5),
    "env_x": ("scale", 1e-3),
    "env_x_ls": ("scale", 1e-3),
    "env_bw_track": ("scale", 1e-9),
    "env_t_env": ("scale", 1e-9),
    "r0": ("scale", 1e-9),
    # Half a per mille of the tracker's honest 0.2 rev/s floor.
    "r_next": ("abs", 1e-4),
}


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

    # The protocol resample (native -> 16 kHz, librosa soxr_hq), then the
    # protocol's own window slicer — the same call beatvk_vk_arms.build_preps
    # makes, so this reference cannot drift from the dataset it guards.
    # (``seconds`` is for smoke runs: a shorter clip is a DIFFERENT reference.)
    audio16 = np.atleast_2d(
        np.asarray(resample_audio_series(rec["audio"], flag.SR).data, dtype=np.float32)
    )
    seg, ft, r_meas, _ = slice_window(
        audio16, flag.SR, spec, rec["ts"], rec["vals"], seconds=seconds
    )
    start, end = float(spec.start_s or 0.0), float(spec.end_s or 0.0)
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
        "clip_seconds": seg.shape[-1] / flag.SR,
        "n_channels": int(seg.shape[0]),
        "n_frames": int(len(ft)),
    }
    return frame, spec, prov


# ---------------------------------------------------------------------------
# the computation


def run_reference(frame: Any, *, peel_mode: str, channels: int) -> tuple[dict[str, Any], dict]:
    """One flagship application on ``frame``: arrays + the config that made them."""
    from tracking.top import get_audio, get_rps
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
# micro-benchmark


def _median_ms(fn: Any, iters: int) -> float:
    """Median wall time (ms) of ``fn()`` over ``iters`` calls.

    One warmup call first, unless ``iters == 1`` — the multi-second stages
    are timed once and a warmup would double the bench's wall time.
    """
    if iters > 1:
        fn()
    samples = []
    for _ in range(iters):
        tic = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - tic)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def _bench_rows(frame: Any, *, channels: int, iters: int, with_vk: bool) -> list[tuple[str, float]]:
    """Time one pass of every hot stage on ``frame`` at the CURRENT worker setting."""
    import beatvk_flagship as flag2

    from tracking.phase_increment_tracker import _demod_bank, pi_kalman_refine, zoom_lp_decimate
    from tracking.top import get_audio, get_rps
    from tracking.vk_tracking import VKConfig, ls_project_envelopes, vk_envelopes

    audio, sr = get_audio(frame)
    clip = np.asarray(audio[:channels], dtype=np.float64)
    r0, ft = get_rps(frame)
    n_t = clip.shape[-1]
    t_aud = np.arange(n_t) / sr
    y32 = clip.astype(np.float32)
    stride = max(1, int(round(sr / 62.5)))
    n_env = len(range(0, n_t, stride))
    ks = list(range(1, flag2.PEEL_K_MAX + 1))
    phi = 2.0 * np.pi * np.cumsum(np.interp(t_aud, ft, r0[0])) / sr
    band_cyc = flag2.PI_BAND_HZ / sr
    x_one = np.asarray(y32 * np.exp(-1j * phi).astype(np.complex64), dtype=np.complex64)

    rows = [
        (
            f"zoom_lp_decimate ({clip.shape[0]},T)",
            _median_ms(lambda: zoom_lp_decimate(x_one, stride, n_env, band_cyc), iters),
        ),
        (
            f"_demod_bank (K={len(ks)})",
            _median_ms(
                lambda: _demod_bank(
                    y32, phi, t_aud, ks, flag2.PI_BAND_HZ + 5.0, stride, n_env, band_cyc
                ),
                1,
            ),
        ),
    ]
    if with_vk:
        cfg = VKConfig(
            fs=float(sr), bw_hz=flag2.PEEL_BW_HZ, k_max=flag2.PEEL_K_MAX, f_max=6000.0, n_outer=1
        )
        r_aud = np.vstack([np.interp(t_aud, ft, r0[r]) for r in range(flag2.N_ROTORS)])
        rows.append(("vk_envelopes", _median_ms(lambda: vk_envelopes(clip, r_aud, cfg), 1)))
        env = vk_envelopes(clip, r_aud, cfg)
        rows.append(
            ("ls_project_envelopes", _median_ms(lambda: ls_project_envelopes(clip, env), 1))
        )
    rows.append(
        (
            "pi_kalman_refine (full)",
            _median_ms(
                lambda: pi_kalman_refine(
                    clip,
                    r0,
                    ft,
                    sr=int(sr),
                    n_iter=flag2.PI_N_ITER,
                    pair_mode=flag2.PI_PAIR_MODE,
                    band_hz=flag2.PI_BAND_HZ,
                ),
                1,
            ),
        )
    )
    return rows


def bench(
    frame: Any,
    *,
    channels: int,
    worker_counts: list[int],
    iters: int,
    with_vk: bool,
    devices: list[str],
    pad: str,
    out_json: Path | None = None,
) -> None:
    """Per-stage timings of the tracking hot path on the frozen clip.

    Stages, innermost first: one ``zoom_lp_decimate`` call (the band-select
    kernel), one ``demod_bank`` flush for rotor 0 at the full harmonic cap
    (the ``pi_kalman`` inner loop), optionally the peel's ``vk_envelopes`` +
    ``ls_project_envelopes`` (``--bench-vk``), and the whole
    ``pi_kalman_refine`` call. Repeated for every ``--bench-workers`` entry
    and every ``--bench-devices`` entry, so both the threading opt-in and the
    device choice are measured rather than assumed.

    ``out_json`` writes the whole grid as a record — the form the remote
    bench comes back in (``omnirun pull``).
    """
    print(f"[bench] channels {channels}, iters {iters}, workers {worker_counts}, {devices}")
    record: dict[str, Any] = {
        "channels": channels,
        "iters": iters,
        "pad": pad,
        "rows": [],
    }
    for dev in devices:
        for w in worker_counts:
            with dsp_config(device=dev, pad=pad), thread_pool(w):
                rows = _bench_rows(frame, channels=channels, iters=iters, with_vk=with_vk)
            print(f"  device={dev} threads={w}")
            for label, ms in rows:
                print(f"    {label:<26}{ms:12.1f} ms", flush=True)
                record["rows"].append(
                    {"device": dev, "workers": w, "stage": label, "ms": round(ms, 1)}
                )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"[bench] wrote {out_json}")


def self_check(
    frame: Any, *, peel_mode: str, channels: int, device: str, pad: str, exact: bool
) -> int:
    """Run the application twice in one process and diff CPU against ``device``.

    The reference leg is always ``cpu`` / ``exact``, so this needs no stored
    ``.npz`` and can therefore run on a compute node that has only the
    checkout and the streamed clip — which is how the GPU is verified.
    """

    tic = time.perf_counter()
    with dsp_config(device="cpu", pad="exact"):
        ref, ref_cfg = run_reference(frame, peel_mode=peel_mode, channels=channels)
    print(f"[self-check] cpu/exact leg: {ref_cfg['wall_s']} ({time.perf_counter() - tic:.0f}s)")
    tic = time.perf_counter()
    with dsp_config(device=device, pad=pad):
        new, new_cfg = run_reference(frame, peel_mode=peel_mode, channels=channels)
    print(
        f"[self-check] {device}/{pad} leg: {new_cfg['wall_s']} ({time.perf_counter() - tic:.0f}s)"
    )
    return diff_table(ref, new, exact, f"the cpu leg vs {device}/{pad}")


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


def diff_table(ref_of: dict[str, Any], new_of: dict[str, Any], exact: bool, what: str) -> int:
    """Print the per-array diff of two array dicts. Returns an exit code.

    Boolean arrays (the gate masks) report a *flip count* in the abs column
    and a flipped fraction in the rel column — a gate flip is the thing a
    device change must not cause, so it is never hidden behind a tolerance.
    """
    failures: list[str] = []
    head = f"{'array':<16}{'shape':>18}{'max|abs|':>13}{'abs/scale':>12}{'max|rel|':>12}"
    print(f"{head}   verdict")
    print("-" * 88)
    for name in DUMPED:
        if name not in ref_of:
            failures.append(f"{name}: missing from {what}")
            continue
        ref = np.asarray(ref_of[name])
        new = np.asarray(new_of[name])
        if ref.shape != new.shape:
            failures.append(f"{name}: shape {ref.shape} -> {new.shape}")
            print(f"{name:<16}{str(ref.shape):>18}{'—':>37}   SHAPE CHANGED")
            continue
        d_abs, d_rel = _diffs(ref, new)
        d_scale = d_abs / _scale(ref)
        ok = np.array_equal(ref, new) if exact else _close(name, ref, new)
        verdict = "identical" if np.array_equal(ref, new) else ("close" if ok else "DIFFERENT")
        if ref.dtype == bool:
            verdict += f" ({int(d_abs)} flips)"
        print(
            f"{name:<16}{str(ref.shape):>18}{d_abs:>13.3e}{d_scale:>12.3e}"
            f"{d_rel:>12.3e}   {verdict}"
        )
        if not ok:
            failures.append(f"{name}: max abs {d_abs:.3e}, abs/scale {d_scale:.3e}")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nOK — {what} reproduced" + (" bit-for-bit" if exact else " within tolerance"))
    return 0


def compare(out: Path, arrays: dict[str, Any], config: dict, exact: bool) -> int:
    """Diff fresh arrays against the stored reference. Returns an exit code."""
    path = out / REF_NAME
    if not path.is_file():
        raise SystemExit(f"no reference at {path} — run --capture first")
    with np.load(path) as z:
        stored_cfg = json.loads(str(z["config"]))
        for key in _config_diff(stored_cfg, config):
            print(f"  CONFIG CHANGED  {key}")
        stored = {name: np.asarray(z[name]) for name in DUMPED if name in z}
        return diff_table(stored, arrays, exact, "the stored reference")


def _diffs(ref: np.ndarray, new: np.ndarray) -> tuple[float, float]:
    if ref.dtype == bool:
        n = float(np.count_nonzero(ref != new))
        return n, n / max(ref.size, 1)
    d = np.abs(np.asarray(new, dtype=np.complex128) - np.asarray(ref, dtype=np.complex128))
    scale = np.maximum(np.abs(np.asarray(ref, dtype=np.complex128)), 1e-300)
    return float(d.max()), float((d / scale).max())


def _scale(ref: np.ndarray) -> float:
    """The array's own magnitude scale (``max |ref|``), floored away from 0."""
    if ref.dtype == bool:
        return max(float(ref.size), 1.0)
    return max(float(np.abs(np.asarray(ref, dtype=np.complex128)).max()), 1e-300)


def _close(name: str, ref: np.ndarray, new: np.ndarray) -> bool:
    """Tolerance-mode verdict for one array, per the :data:`TOL` bar."""
    if ref.dtype == bool:
        return bool(np.array_equal(ref, new))
    kind, val = TOL.get(name, ("scale", 1e-5))
    bar = val * _scale(ref) if kind == "scale" else val
    return bool(
        np.abs(np.asarray(new, np.complex128) - np.asarray(ref, np.complex128)).max() <= bar
    )


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
    mode.add_argument(
        "--bench", action="store_true", help="time the hot stages instead of capturing/diffing"
    )
    mode.add_argument(
        "--self-check",
        action="store_true",
        help="run cpu/exact and the selected device in one process and diff them",
    )
    ap.add_argument("--exact", action="store_true", help="--compare: demand bit-identical arrays")
    ap.add_argument(
        "--device", default="cpu", help="torch device of the fresh run (cpu, cuda, ...)"
    )
    ap.add_argument(
        "--pad-mode",
        default="exact",
        choices=list(PAD_MODES),
        help="transform-length padding of the fresh run (default: exact)",
    )
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
    ap.add_argument(
        "--bench-workers",
        default="1,4",
        help="--bench: comma-separated torch thread counts (0 = the whole CPU budget)",
    )
    ap.add_argument("--bench-iters", type=int, default=5, help="--bench: repeats of the FFT kernel")
    ap.add_argument(
        "--bench-vk", action="store_true", help="--bench: also time vk_envelopes + ls_project"
    )
    ap.add_argument(
        "--bench-devices",
        default=None,
        help="--bench: comma-separated torch devices (default: --device only)",
    )
    ap.add_argument("--bench-json", default=None, help="--bench: write the grid to this JSON path")
    opts = ap.parse_args()

    out = Path(opts.capture or opts.compare or ".")
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

    if opts.bench:
        bench(
            frame,
            channels=opts.channels,
            worker_counts=[int(v) for v in str(opts.bench_workers).split(",") if v.strip()],
            iters=opts.bench_iters,
            with_vk=opts.bench_vk,
            devices=[
                v.strip() for v in str(opts.bench_devices or opts.device).split(",") if v.strip()
            ],
            pad=opts.pad_mode,
            out_json=Path(opts.bench_json) if opts.bench_json else None,
        )
        return

    if opts.self_check:
        raise SystemExit(
            self_check(
                frame,
                peel_mode=opts.peel_mode,
                channels=opts.channels,
                device=opts.device,
                pad=opts.pad_mode,
                exact=opts.exact,
            )
        )

    with dsp_config(device=opts.device, pad=opts.pad_mode), thread_pool(None):
        arrays, config = run_reference(frame, peel_mode=opts.peel_mode, channels=opts.channels)
    config["dsp"] = [opts.device, opts.pad_mode]
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
