#!/usr/bin/env python3
"""Beat-VK protocol runner for the phase-increment tracker (``pi_kalman``).

Takes an existing per-recording trajectory source in the beat-VK NPZ
convention (``<recording_id>.npz`` with ``ft`` (N,) seconds from recording
start + ``rps`` (4, N) rev/s — e.g. a ``beatvk_vk_arms`` output dir such as
``results/beatvk_vk_arms/neural_traj`` or ``.../telem_init``), refines it
window-by-window with :func:`data_processing.phase_increment_tracker.
pi_kalman_refine` on the ``beatvk-valid-raw`` audio, and writes a refined
NPZ dir in the SAME convention — directly consumable by
``scripts/beatvk_eval.py --pred npz:<out>/traj``.

Protocol fit:

* Windows come from the frozen dataset manifest (never re-derived); each
  selected window is refined independently from the init trajectory
  restricted to it. Frames outside the selected windows keep the init
  values, so the output trajectory is the init with refined spans patched
  in on the 0.032 s recording grid.
* Audio is soxr-resampled to the tracker's 16 kHz (the same single-resample
  convention as ``beatvk_eval``'s ``model:`` source); the GT/telemetry side
  is untouched — scoring happens on the frozen scorer's raw-telemetry grid.
* Scoring reuses ``beatvk_eval``'s own machinery (``score_recording`` +
  ``align_rps_to_gt``), printed as a before/after per-window PIT-MAE table
  plus pooled means over the refined windows. Numbers match a subsequent
  ``beatvk_eval.py --pred npz:...`` run window-for-window.

Run (smoke: 2 steady DREGON + 2 FLY124 cruise windows from the neural
trajectories)::

    python scripts/pi_kalman_protocol.py \
        --init omnirun-outputs/bash-1e251e/results/beatvk_vk_arms/neural_traj \
        --windows free-flight_nosource_room1:1,2 FLY124:3,4 \
        --pair-mode joint --tag neural_smoke

Full pass: omit ``--windows`` (every manifest window of every recording
present in the init dir).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import beatvk_eval  # noqa: E402  (scripts/ on path)

from data_processing.phase_increment_tracker import pi_kalman_refine  # noqa: E402

SR = beatvk_eval.SR  # 16 kHz — the tracker's rate
FRAME_S = beatvk_eval.FRAME_S  # 0.032 s recording grid
N_ROTORS = beatvk_eval.N_ROTORS


def parse_window_spec(specs: list[str] | None) -> dict[str, set[int]] | None:
    """``rec:0,2`` items -> ``{rec: {0, 2}}``; None -> all windows."""
    if not specs:
        return None
    out: dict[str, set[int]] = {}
    for spec in specs:
        rid, _, idxs = spec.partition(":")
        if not idxs:
            raise SystemExit(f"bad --windows item {spec!r} (expected <recording>:<i>,<j>,...)")
        out.setdefault(rid, set()).update(int(x) for x in idxs.split(","))
    return out


def refine_recording(
    rec: dict[str, Any],
    ft_init: np.ndarray,
    rps_init: np.ndarray,
    windows: list[dict[str, Any]],
    *,
    pair_mode: str,
    n_iter: int,
    band_hz: float | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Refine the selected windows of one recording.

    Returns ``(tg, rps_out, meta)``: the 0.032 s recording grid, the init
    trajectory with refined spans patched in, and per-window run metadata.
    """
    from data_processing.frames import resample_audio_series

    audio16 = resample_audio_series(rec["audio"], SR)
    audio = np.atleast_2d(np.asarray(audio16.data, dtype=np.float64))
    n_frames = int(np.ceil(max(float(w["end_s"]) for w in rec["windows"]) / FRAME_S)) + 1
    tg = np.arange(n_frames, dtype=np.float64) * FRAME_S
    rps_out = np.vstack([np.interp(tg, ft_init, rps_init[r]) for r in range(N_ROTORS)])

    meta: list[dict[str, Any]] = []
    for w in windows:
        start, end = float(w["start_s"]), float(w["end_s"])
        a0, a1 = int(round(start * SR)), min(int(round(end * SR)), audio.shape[-1])
        if a1 - a0 < SR:
            print(f"  [skip] {rec['recording_id']} w{w['index']}: no audio in window")
            continue
        clip = audio[:, a0:a1]
        ft_w = np.arange(start, (a1 - 1) / SR - FRAME_S / 2, FRAME_S)
        r0 = np.vstack([np.interp(ft_w, ft_init, rps_init[r]) for r in range(N_ROTORS)])
        tic = time.perf_counter()
        r_hat, diag = pi_kalman_refine(
            clip, r0, ft_w - start, sr=SR, n_iter=n_iter, pair_mode=pair_mode, band_hz=band_hz
        )
        wall = time.perf_counter() - tic
        mask = (tg >= start - 1e-6) & (tg < end - 1e-6)
        for r in range(N_ROTORS):
            rps_out[r, mask] = np.interp(tg[mask], ft_w, r_hat[r])
        step = float(np.mean(np.abs(r_hat - r0)))
        meta.append(
            {
                "window": int(w["index"]),
                "regime": str(w["regime"]),
                "wall_s": round(wall, 1),
                "mean_abs_step": round(step, 4),
                "diag": diag,
            }
        )
        print(
            f"  [refined] {rec['recording_id']} w{w['index']} ({w['regime']}): "
            f"mean |step| {step:.3f} rev/s in {wall:.0f} s",
            flush=True,
        )
    return tg, rps_out, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument(
        "--init", required=True, help="init trajectory NPZ dir/file (beatvk npz: convention)"
    )
    ap.add_argument(
        "--windows",
        nargs="+",
        default=None,
        help="restrict to '<recording>:<i>,<j>' manifest windows (default: all)",
    )
    ap.add_argument("--pair-mode", default="gate", choices=("gate", "joint"))
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument(
        "--band-hz",
        default="6",
        help="demod half-band (Hz): one float or a comma list = per-iteration schedule",
    )
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--tag", default=None, help="run name (default: <init stem>_<pair-mode>)")
    ap.add_argument("--out", default="results/pi_kalman_protocol", help="output root")
    args = ap.parse_args()

    sel = parse_window_spec(args.windows)
    tag = args.tag or f"{Path(args.init).stem}_{args.pair_mode}"
    out_dir = Path(args.out) / tag
    traj_dir = out_dir / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(sel) if sel is not None else None
    recs = beatvk_eval.load_recordings(args.dataset_version, wanted, keep_audio=True)
    rec_ids = [r["recording_id"] for r in recs]
    preds = beatvk_eval.preds_from_npz(Path(args.init), rec_ids)
    print(
        f"[pi_kalman_protocol] {beatvk_eval.DATASET}@{recs[0]['dataset_version'][:12]}: "
        f"{sorted(preds)} (pair_mode={args.pair_mode})",
        flush=True,
    )

    rows_before: list[dict[str, Any]] = []
    rows_after: list[dict[str, Any]] = []
    run_meta: dict[str, Any] = {}
    for rec in recs:
        rid = rec["recording_id"]
        if rid not in preds:
            print(f"[pi_kalman_protocol] no init trajectory for {rid} — skipped", flush=True)
            continue
        windows = [
            w for w in rec["windows"] if sel is None or int(w["index"]) in sel.get(rid, set())
        ]
        if not windows:
            continue
        bands = tuple(float(b) for b in str(args.band_hz).split(","))
        band_hz = bands[0] if len(bands) == 1 else bands
        ft_init, rps_init = preds[rid]
        tg, rps_out, meta = refine_recording(
            rec,
            ft_init,
            rps_init,
            windows,
            pair_mode=args.pair_mode,
            n_iter=args.n_iter,
            band_hz=band_hz,
        )
        np.savez(traj_dir / f"{rid}.npz", ft=tg, rps=rps_out)
        run_meta[rid] = meta
        keep = {int(w["index"]) for w in windows}
        rec_nb = {**rec, "windows": windows}  # score the refined windows only
        rows_before.extend(
            r
            for r in beatvk_eval.score_recording(rec_nb, ft_init, rps_init, ["none"])
            if r["window"] in keep
        )
        rows_after.extend(
            r
            for r in beatvk_eval.score_recording(rec_nb, tg, rps_out, ["none"])
            if r["window"] in keep
        )

    if not rows_after:
        raise SystemExit("nothing refined (empty window selection?)")

    before = {(r["recording"], r["window"]): r for r in rows_before}
    print("\nBefore/after per-window PIT-MAE (rev/s), beat-VK scoring:")
    header = f"{'recording':<36}{'w':>3} {'regime':<8}{'init':>8}{'refined':>9}{'delta':>8}"
    print(header)
    print("-" * len(header))
    deltas = []
    for r in rows_after:
        b = before[(r["recording"], r["window"])]
        delta = r["mae"] - b["mae"]
        deltas.append(delta)
        print(
            f"{r['recording']:<36}{r['window']:>3} {r['regime']:<8}"
            f"{b['mae']:>8.3f}{r['mae']:>9.3f}{delta:>+8.3f}"
        )

    def pool(rows: list[dict[str, Any]], pred) -> float | None:
        v = [r["mae"] for r in rows if pred(r)]
        return float(np.mean(v)) if v else None

    print("\nPooled over refined windows:")
    for name, predfn in (
        (
            "dregon_cruise",
            lambda r: r["recording"] in beatvk_eval.DREGON_RECS and r["regime"] == "cruise",
        ),
        (
            "fly124_cruise",
            lambda r: r["recording"] == beatvk_eval.FLY124_REC and r["regime"] == "cruise",
        ),
        ("all_selected", lambda r: True),
    ):
        b, a = pool(rows_before, predfn), pool(rows_after, predfn)
        if b is not None and a is not None:
            print(f"  {name:<16} init {b:.3f} -> refined {a:.3f} ({a - b:+.3f})")

    report = {
        "init": args.init,
        "pair_mode": args.pair_mode,
        "n_iter": args.n_iter,
        "band_hz": args.band_hz,
        "dataset": {"name": beatvk_eval.DATASET, "version": recs[0]["dataset_version"]},
        "windows": args.windows,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "runs": run_meta,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"\n[pi_kalman_protocol] wrote {traj_dir}/ (+ report.json); full protocol scoring:\n"
        f"  python scripts/beatvk_eval.py --pred npz:{traj_dir} --tag {tag}",
        flush=True,
    )


if __name__ == "__main__":
    main()
