"""Profiling / regression benchmark for the coupled Vold-Kalman tracker.

Phase 1 of "VK tracker fast inference": measure where ``vk_track``
(``data_processing.vk_tracking``) spends its time on the real whole-recording
cases used by the prior evaluations, and pin down regression references so
later optimizations can prove they do not change results.

Cases — the ``wholerec_*`` trajectory files produced by the prior VK
evaluation (3 DREGON free-flight recordings + Michael's FLY124). The npz files
carry only trajectories (``ft``/``r_init``/``r_vk``/``tau``), NOT audio, so
tracked copies live in ``scripts/vk_bench_cases/`` (small, ship with the git
clone to any omnirun backend) and audio is streamed from the published
rich-frame dload datasets (``DREGON-frames`` / ``michaels-frames``,
credentials via the auto-shipped ``.env``). Each case is limited to a 20 s
in-flight window (FLY124: cruise, per-frame mean telemetry RPS > 45 — idle /
warm-up segments inflate error and are not the regime we care about).

Config families (the two real run families):

* ``refine`` — telemetry init; the validation config of
  ``results/vk_tracking/validation/summary.json``: bw 1.5, k 6-30,
  couple 20, n_outer 5, max_step 0.3, fixed schedule.
* ``blind``  — the blind-annotation config of
  ``results/vk_tracking/blind_annotation/fix_summary.json``: bw 7.0, k 6-12,
  couple 20, n_outer 8, update_gate 8.0, fixed schedule. The stored blind
  operating-point inits live in gitignored ``results/`` and are not shipped,
  so the init is telemetry + 2 rev/s (all rotors) — a blind-magnitude
  perturbation, same spirit as the capture-basin protocol.

Per (case x config): wall time, cProfile per-phase breakdown (demodulate /
envelope solve / frequency update / reconstruct, keyed to the vk_tracking
function names), peak RSS, realtime factor (audio s / wall s), and pooled MAE
vs the prior refined trajectories stored in the case npz (a consistency
number — the prior runs used full-recording chunking, so exact equality is
not expected; the *regression* reference for future optimization work is this
run's own output, saved to ``results/vk_bench/<case>_<config>.npz``).

Outputs: ``results/vk_bench/profile_report.json`` + ``profile_report.txt``
(+ the per-run reference npz files; ``omnirun pull`` brings ``results/**``
back).

Modes:
  * default            — all 4 cases x 2 configs, first 20 s.
  * ``--quick``        — single case (nosource), first 10 s (CI-style smoke).
  * ``--synthetic``    — tiny self-test (2 rotors, 5 s, 8 harmonics, mono),
                         runs anywhere in <1 min with no data access; writes
                         ``profile_report_synthetic.json``/``.txt``.

Optimization A/B knobs (fast-inference work): ``--solver banded|splu``,
``--lp-mode fft|fir|iir`` and ``--no-prune`` override the corresponding
``VKConfig`` fields on both config families; ``--out-suffix _foo`` suffixes
every output file (report + per-run npz) so a re-bench does not clobber the
recorded regression references.

Run: ``.venv/bin/python scripts/vk_bench.py [--quick | --synthetic]``
"""

from __future__ import annotations

import os

# BLAS/FFT thread budget must be set BEFORE numpy import. Unlike the
# process-parallel validation scripts, the bench runs cases sequentially, so
# it uses the whole box by default (this is the realistic single-run inference
# setting the "fast inference" push optimizes). vk_tracking's FFT stage reads
# OMP_NUM_THREADS for its worker count.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))

import argparse  # noqa: E402
import cProfile  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import pstats  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from data_processing.vk_tracking import VKConfig, vk_track  # noqa: E402

SR = 16000
FRAME_HOP_S = 0.032  # the wholerec / validation frame grid
EDGE_TRIM_S = 0.5  # metric exclusion at window edges (filter transients)
BLIND_INIT_OFFSET = 2.0  # rev/s, all rotors (see module docstring)
CASES_DIR = Path("scripts/vk_bench_cases")
OUT_DIR = Path("results/vk_bench")

# case name -> (frames dataset, sample key, cruise/in-flight RPS threshold)
CASES: dict[str, tuple[str, str, float]] = {
    "free-flight_nosource_room1": ("DREGON-frames", "free-flight_nosource_room1", 30.0),
    "free-flight_speech-low_room1": ("DREGON-frames", "free-flight_speech-low_room1", 30.0),
    "free-flight_whitenoise-low_room1": (
        "DREGON-frames",
        "free-flight_whitenoise-low_room1",
        30.0,
    ),
    "michaels_FLY124": ("michaels-frames", "FLY124", 45.0),
}
QUICK_CASE = "free-flight_nosource_room1"

# The two real run families (see module docstring for provenance).
REFINE_CFG = VKConfig(
    fs=float(SR),
    couple_hz=20.0,
    n_outer=5,
    k_min=6,
    k_max=30,
    k_schedule="fixed",
    bw_hz=1.5,
    max_step=0.3,
)
BLIND_CFG = VKConfig(
    fs=float(SR),
    couple_hz=20.0,
    n_outer=8,
    k_min=6,
    k_max=12,
    k_schedule="fixed",
    bw_hz=7.0,
    max_step=0.5,
    update_gate=8.0,
)
CONFIGS: dict[str, VKConfig] = {"refine": REFINE_CFG, "blind": BLIND_CFG}

# cProfile phase attribution: cumulative time of these vk_tracking functions.
# NB: phases nest (``vk_envelopes`` includes its internal demod; ``vk_track``
# includes everything) — the breakdown is a map of where time accumulates,
# not a disjoint partition. ``demodulate``/``_demod_tracks_fft`` are the demod
# entry points for both lp_modes; ``_demod_residual`` is the residual demod
# the gate/confidence machinery pays for on top.
PHASE_FUNCS: dict[str, tuple[str, ...]] = {
    "total_vk_track": ("vk_track",),
    "envelope_solve": ("vk_envelopes",),
    "demodulate": ("demodulate", "_demod_tracks_fft"),
    "fft_lp_decimate": ("_fft_lp_decimate",),
    "freq_update": ("_freq_update",),
    "reconstruct": ("vk_reconstruct",),
    "residual_demod": ("_demod_residual",),
    "confidence": ("_confidence",),
}


def _maxrss_mb() -> float:
    """Peak RSS of this process so far, in MB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# data access


def load_case_npz(case: str) -> dict[str, np.ndarray]:
    """Load a case's trajectory npz (tracked copy first, then results/)."""
    for root in (CASES_DIR, Path("results/vk_eval")):
        path = root / f"wholerec_{case}.npz"
        if path.exists():
            with np.load(path) as z:
                return {k: np.asarray(z[k]) for k in ("ft", "r_init", "r_vk", "tau")}
    raise FileNotFoundError(
        f"no trajectory npz for case {case!r} in {CASES_DIR}/ or results/vk_eval/"
    )


def load_case_audio(case: str, t0: float, dur_s: float) -> np.ndarray:
    """Stream one recording's audio from its published frames dataset.

    Returns ``(C, T)`` float64 at ``SR``, the ``[t0, t0 + dur_s]`` slice in
    recording-relative time (the ``ft`` clock of the case npz). Sample keys of
    the ``tdframe-v1`` datasets are bare recording ids, so the raw sample
    stream is filtered on the key *before* decoding — only the matching
    recording is decoded (shards still stream sequentially; fine on a cloud
    box, do NOT do this on the 4 GB laptop).
    """
    import dload
    import librosa

    from data_processing.streams import decode_tdframe, open_repository

    dataset, key_wanted, _ = CASES[case]
    repo = open_repository()
    ds = dload.Dataset(repo, repo.manifest(dataset, None))
    for key, fields in ds.samples():
        if key != key_wanted:
            continue
        frame = decode_tdframe((key, fields))
        series = frame["audio"]
        data = np.atleast_2d(np.asarray(series.data))
        sr_native = float(series.tindex.sr)  # type: ignore[union-attr]
        a0 = int(round(t0 * sr_native))
        a1 = min(int(round((t0 + dur_s) * sr_native)), data.shape[-1])
        seg = np.asarray(data[:, a0:a1], dtype=np.float64)
        if sr_native != SR:
            seg = librosa.resample(seg, orig_sr=sr_native, target_sr=SR, res_type="soxr_hq")
        return seg
    raise KeyError(f"recording {key_wanted!r} not found in dload dataset {dataset!r}")


def pick_window(ft: np.ndarray, r_init: np.ndarray, dur_s: float, min_rps: float) -> float:
    """Earliest window start whose frames all have mean telemetry > min_rps.

    Falls back to the window with the highest in-regime fraction when no fully
    clean window exists (short recordings / permanently noisy telemetry).
    """
    mean_r = r_init.mean(axis=0)
    ok = mean_r > min_rps
    hop = float(np.median(np.diff(ft)))
    n_win = max(1, int(round(dur_s / hop)))
    if len(ft) <= n_win:
        return float(ft[0])
    frac = np.convolve(ok.astype(np.float64), np.ones(n_win) / n_win, mode="valid")
    full = np.where(frac >= 1.0)[0]
    start = int(full[0]) if len(full) else int(np.argmax(frac))
    return float(ft[start])


# ---------------------------------------------------------------------------
# bench core


def phase_breakdown(prof: cProfile.Profile) -> tuple[dict[str, float], str]:
    """Per-phase cumulative seconds + a top-25 tottime table (text)."""
    stream = io.StringIO()
    ps = pstats.Stats(prof, stream=stream)
    raw: dict[tuple[str, int, str], tuple[int, int, float, float, Any]] = ps.stats  # type: ignore[attr-defined]
    phases: dict[str, float] = {}
    for phase, names in PHASE_FUNCS.items():
        phases[phase] = sum(entry[3] for loc, entry in raw.items() if loc[2] in names)
    ps.sort_stats("tottime").print_stats(25)
    return phases, stream.getvalue()


def run_one(
    label: str,
    audio: np.ndarray,
    frame_times: np.ndarray,
    r_init: np.ndarray,
    r_ref: np.ndarray | None,
    cfg_name: str,
    cfg: VKConfig,
    out_dir: Path,
    suffix: str = "",
) -> dict[str, Any]:
    """Profile one ``vk_track`` run and save its regression-reference npz."""
    audio = np.atleast_2d(audio)
    audio_s = audio.shape[-1] / cfg.fs
    print(
        f"[{label} / {cfg_name}] {audio.shape[0]} ch x {audio_s:.1f} s, "
        f"{r_init.shape[0]} rotors, k {cfg.k_min}-{cfg.k_max}, n_outer {cfg.n_outer} ...",
        flush=True,
    )
    prof = cProfile.Profile()
    tic = time.perf_counter()
    prof.enable()
    res = vk_track(audio, r_init, frame_times, cfg)
    prof.disable()
    wall = time.perf_counter() - tic
    phases, top_txt = phase_breakdown(prof)

    edge = (frame_times > frame_times[0] + EDGE_TRIM_S) & (
        frame_times < frame_times[-1] - EDGE_TRIM_S
    )
    mae_vs_ref = mae_init_vs_ref = None
    if r_ref is not None:
        mae_vs_ref = float(np.mean(np.abs((res.r_refined - r_ref)[:, edge])))
        mae_init_vs_ref = float(np.mean(np.abs((r_init - r_ref)[:, edge])))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{label}_{cfg_name}{suffix}.npz",
        frame_times=frame_times,
        r_init=r_init,
        r_refined=res.r_refined,
        r_env=res.r_env,
        t_env=res.t_env,
        residual_ratios=np.array(res.residual_ratios),
        max_deltas=np.array(res.max_deltas),
        edge_mask=edge,
        config_json=np.bytes_(json.dumps(asdict(cfg)).encode()),
    )

    row: dict[str, Any] = {
        "case": label,
        "config": cfg_name,
        "n_channels": int(audio.shape[0]),
        "audio_s": round(audio_s, 2),
        "wall_s": round(wall, 2),
        "realtime_factor": round(audio_s / wall, 4),
        "peak_rss_mb": round(_maxrss_mb(), 1),
        "phases_cum_s": {k: round(v, 2) for k, v in phases.items()},
        "residual_first": round(res.residual_ratios[0], 4),
        "residual_last": round(res.residual_ratios[-1], 4),
        "max_delta_last": round(res.max_deltas[-1], 4),
        "mae_vs_stored_ref": mae_vs_ref,
        "mae_init_vs_stored_ref": mae_init_vs_ref,
        "profile_top": top_txt,
    }
    print(
        f"[{label} / {cfg_name}] wall {wall:.1f} s (rtf {audio_s / wall:.3f}), "
        f"resid {res.residual_ratios[0]:.3f}->{res.residual_ratios[-1]:.3f}, "
        f"mae_vs_ref {mae_vs_ref if mae_vs_ref is None else round(mae_vs_ref, 4)}, "
        f"peak_rss {row['peak_rss_mb']:.0f} MB",
        flush=True,
    )
    return row


def run_real_case(case: str, dur_s: float, suffix: str = "") -> list[dict[str, Any]]:
    """Load one wholerec case (npz trajectories + streamed audio), run both configs."""
    z = load_case_npz(case)
    ft, r_init_full, r_vk_full = z["ft"], z["r_init"], z["r_vk"]
    min_rps = CASES[case][2]
    t0 = pick_window(ft, r_init_full, dur_s, min_rps)
    mask = (ft >= t0) & (ft < t0 + dur_s)
    ft_w = ft[mask] - t0
    audio = load_case_audio(case, t0, dur_s)
    # Trim the trajectory grid to the audio actually delivered (tail windows).
    n_keep = int(np.sum(ft_w < audio.shape[-1] / SR - FRAME_HOP_S / 2))
    ft_w = ft_w[:n_keep]
    r_init = r_init_full[:, mask][:, :n_keep]
    r_ref = r_vk_full[:, mask][:, :n_keep]
    print(f"[{case}] window {t0:.1f}-{t0 + dur_s:.1f} s, {len(ft_w)} frames", flush=True)

    rows = []
    for cfg_name, cfg in CONFIGS.items():
        init = r_init if cfg_name == "refine" else r_init + BLIND_INIT_OFFSET
        rows.append(run_one(case, audio, ft_w, init, r_ref, cfg_name, cfg, OUT_DIR, suffix))
    return rows


# ---------------------------------------------------------------------------
# synthetic self-test


def synthetic_signal(
    dur_s: float = 5.0, n_rotors: int = 2, k_max: int = 8, snr_db: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiny synthetic comb: mono audio, frame grid, true trajectories."""
    rng = np.random.default_rng(0)
    n_t = int(dur_s * SR)
    t = np.arange(n_t) / SR
    base = np.array([80.0, 65.0])[:n_rotors]
    r_true = base[:, None] + np.stack(
        [2.0 * np.sin(2 * np.pi * 0.3 * t + i) for i in range(n_rotors)]
    )
    phase = 2 * np.pi * np.cumsum(r_true, axis=-1) / SR
    audio = np.zeros(n_t)
    for i in range(n_rotors):
        for k in range(1, k_max + 1):
            audio += (1.0 / k) * np.cos(k * phase[i] + rng.uniform(0, 2 * np.pi))
    noise = rng.standard_normal(n_t)
    noise *= np.sqrt(np.sum(audio**2) / np.sum(noise**2)) * 10 ** (-snr_db / 20)
    audio += noise
    ft = np.arange(0.0, dur_s - FRAME_HOP_S / 2, FRAME_HOP_S)
    r_true_ft = np.stack([np.interp(ft, t, r_true[i]) for i in range(n_rotors)])
    return audio, ft, r_true_ft


def run_synthetic(suffix: str = "") -> list[dict[str, Any]]:
    """Self-test: same machinery, no data access, <1 min anywhere."""
    audio, ft, r_true = synthetic_signal()
    # Scaled-down cousins of the two families (only 8 harmonics exist).
    cfgs = {
        "refine": replace(CONFIGS["refine"], k_min=1, k_max=8, n_outer=3),
        "blind": replace(CONFIGS["blind"], k_min=1, k_max=8, n_outer=4),
    }
    rows = []
    for cfg_name, cfg in cfgs.items():
        init = r_true + (0.5 if cfg_name == "refine" else BLIND_INIT_OFFSET)
        rows.append(run_one("synthetic", audio, ft, init, r_true, cfg_name, cfg, OUT_DIR, suffix))
    return rows


# ---------------------------------------------------------------------------
# reporting


def write_report(rows: list[dict[str, Any]], suffix: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "edge_trim_s": EDGE_TRIM_S,
            "blind_init_offset_revs": BLIND_INIT_OFFSET,
            "configs": {name: asdict(cfg) for name, cfg in CONFIGS.items()},
            "note": (
                "mae_vs_stored_ref compares to the PRIOR full-recording chunked runs "
                "(consistency, not identity); regression reference for optimization "
                "work is this run's own <case>_<config>.npz output"
            ),
        },
        "runs": [{k: v for k, v in r.items() if k != "profile_top"} for r in rows],
    }
    json_path = OUT_DIR / f"profile_report{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "VK tracker profiling benchmark",
        f"host {platform.node()}  cpus {os.cpu_count()}  omp {os.environ.get('OMP_NUM_THREADS')}",
        "",
        f"{'case':<36}{'config':>8}{'ch':>4}{'audio_s':>9}{'wall_s':>9}{'rtf':>8}"
        f"{'rss_MB':>8}{'mae_ref':>9}",
        "-" * 91,
    ]
    for r in rows:
        mae = "-" if r["mae_vs_stored_ref"] is None else f"{r['mae_vs_stored_ref']:.4f}"
        lines.append(
            f"{r['case']:<36}{r['config']:>8}{r['n_channels']:>4}{r['audio_s']:>9.1f}"
            f"{r['wall_s']:>9.1f}{r['realtime_factor']:>8.3f}{r['peak_rss_mb']:>8.0f}{mae:>9}"
        )
    for r in rows:
        lines += [
            "",
            "=" * 91,
            f"{r['case']} / {r['config']}: phase cumulative seconds "
            "(nested — envelope_solve includes its demod; total includes all)",
        ]
        lines += [f"  {k:<18}{v:>10.2f}" for k, v in r["phases_cum_s"].items()]
        lines += ["", r["profile_top"]]
    txt_path = OUT_DIR / f"profile_report{suffix}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"\nreport: {json_path} + {txt_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])  # type: ignore[union-attr]
    ap.add_argument("--quick", action="store_true", help="single case, first 10 s")
    ap.add_argument("--synthetic", action="store_true", help="tiny local self-test (no data)")
    ap.add_argument("--duration", type=float, default=20.0, help="window length (s)")
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        default=None,
        help="restrict to these cases (default: all; useful for resubmitting leftovers)",
    )
    ap.add_argument(
        "--solver",
        choices=["banded", "splu"],
        default=None,
        help="override VKConfig.solver on both config families (A/B)",
    )
    ap.add_argument(
        "--lp-mode",
        choices=["fft", "fir", "iir"],
        default=None,
        help="override VKConfig.lp_mode on both config families (A/B)",
    )
    ap.add_argument(
        "--no-prune",
        action="store_true",
        help="disable VKConfig.prune_far_pairs on both config families (A/B)",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="suffix for every output file (report + npz) — keeps re-bench "
        "runs from clobbering the recorded regression references",
    )
    args = ap.parse_args()

    overrides: dict[str, Any] = {}
    if args.solver is not None:
        overrides["solver"] = args.solver
    if args.lp_mode is not None:
        overrides["lp_mode"] = args.lp_mode
    if args.no_prune:
        overrides["prune_far_pairs"] = False
    if overrides:
        for name in list(CONFIGS):
            CONFIGS[name] = replace(CONFIGS[name], **overrides)

    if args.synthetic:
        rows = run_synthetic(suffix=args.out_suffix)
        write_report(rows, suffix="_synthetic" + args.out_suffix)
        return

    cases = [QUICK_CASE] if args.quick else (args.cases or list(CASES))
    dur = 10.0 if args.quick else args.duration
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(run_real_case(case, dur, suffix=args.out_suffix))
        write_report(rows, suffix=args.out_suffix)  # partial report survives a timeout


if __name__ == "__main__":
    main()
