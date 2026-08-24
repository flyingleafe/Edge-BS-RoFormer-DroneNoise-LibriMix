#!/usr/bin/env python3
"""The blind tracker's row of the unified RPS leaderboard, on the V4 valid split.

``dload:DREGON-LM-V4-michaels-valid-full`` is 37 clips of 8 s cut from FOUR
parent recordings — DREGON ``free-flight_{nosource,speech-low,whitenoise-low}_room1``
and Michael's ``FLY124``. The blind ladder wants a 20 s window, so this driver
does what the corpus campaign does and annotates the PARENT recordings, then
scores the stitched full-span track on the clips' own STFT frames.

Two stages, one CLI:

1. ``annotate`` — one unit is one (recording, window). The worker is
   ``scripts/blind_corpus.Worker`` unchanged, so the ladder call, the guard
   readings and the label-free instruments are the corpus campaign's, call for
   call. Products: ``raw/<uid>.json`` (the readings), ``traj/<uid>.npz`` (the
   trajectory) and ``windows/<uid>.npz`` (the cached window audio).
2. ``score`` — stitch the windows of each recording, decode a REFUSED window to
   0 rev/s, interpolate onto every clip's 2048/512 frame grid, and pool the
   per-frame Hungarian error by regime.

**The arm.** ``vit2dsp`` is the run of record: ``tracking.vit2dsp`` is the
calibrated blind-annotation ladder that reads DREGON pooled ``err_sm`` 0.688
and FLY124 cruise 1.027 (``docs/experiments/beat-vk.md`` § Goal), and its guard
is inside ``vit2dsp_pipeline`` — the script never assembles a ladder. It is
also a CRUISE arm: it carries one rate law across the window, so the warm-up,
take-off and landing parts of this full-envelope split are its weak regime.
``--arm fullrange`` (``tracking.blind_fullrange``) is the ramp-capable
alternative and reads DREGON cruise 1.809 / FLY124 cruise 2.515 / warm-up 3.607
on the beat-VK protocol. Run both if the leaderboard has room for two rows.

**The channel convention.** One trajectory per RECORDING, from all 8
microphones: ``blind_seed`` channel-averages its whitened spectrogram and the
ladder's stage 2 is a SPATIAL joint Viterbi. The corpus campaign measured that
a mono seed collapses (DREGON PIT-MAE 1.81 -> 18.76), so the 8-channel form is
the only defensible one. The eight channels of one clip therefore share one
prediction, and the pooled MAE is identical whether a channel is counted once
or eight times.

**The zero convention** (user decision, full envelope). A window the acceptance
gates refuse, a window whose unit failed, and any span no window covers all
decode to 0 rev/s on all four rotors. Nothing is dropped from the pools.

**The gates.** ``docs/experiments/blind-corpus-annotation.md`` § "Acceptance
gates" makes the rule ARM-DEPENDENT, and this is a four-rotor arm, so the
default set is:

- ``g1`` — ``fvk_ratio_double >= 1.065``. Calibrated on DREGON room2 twice,
  17 of 17 both times.
- ``g5`` — the four annotated rates stay inside 12 rev/s of each other.

``pr`` (the per-rotor half margin, cut -1.5 dB) is available and is OFF by
default: the room1 run against MEASURED telemetry found the cut unsafe at four
rotors — it crosses on 2 of 19 windows, one of which has a PIT-MAE of 1.08
rev/s and is one of the best windows of the whole campaign. ``g4`` (continuity:
consecutive windows must agree inside their overlap to better than 1 rev/s) is
also available and OFF by default, because it refuses BOTH members of a
disagreeing pair and this split is mostly ramp. ``score`` always reports the
ungated numbers beside the gated ones, and re-scoring under a different gate
set costs seconds.

**The stitch.** Windows overlap by 4 s and the cut is at the MIDPOINT: a frame
takes the trajectory of the window whose center is nearest. No cross-fade —
two windows can name their rotors in different orders, and averaging rows
across a permutation would invent a trajectory neither window produced.

**The clip-to-parent join.** The published split's ``metadata.json`` carries
``recording_id`` and an AUDIO-RELATIVE ``start_time`` per clip, and ``score``
reads it directly — ``DregonLMFrameDataset`` looks for that file beside the
split directory, which a dload materialization does not have, so the dataset's
own ``meta`` entry is empty. ``score`` then PROVES the join rather than
trusting it: see :func:`verify_mapping`.

Smoke, measured (one window, this laptop, 4 BLAS threads)::

  python scripts/blind_valid_row.py annotate --recordings free-flight_nosource_room1 \\
      --windows 1 --jobs 1 --omp 4 --out results/blind_valid_row/smoke
  python scripts/blind_valid_row.py score --annot results/blind_valid_row/smoke

126.4 s of wall time for the 20 s window — ladder 82.6, F_VK 36.0, ridge 2.9,
per-rotor octave 4.9 — and the window reads a PIT error of 1.058 rev/s against
the parent's tachometer, which is inside the campaign's room1 cruise band of
0.97 to 1.22. The two clips it fully covers score 0.788 and 0.735 rev/s.

Full run (cluster). The four parents give 20 windows (4 + 4 + 4 DREGON, 8
FLY124), and the corpus campaign measured about 400 CPU-s per window at one
BLAS thread, so the grid is about 2.2 CPU-hours::

  omnirun submit --backend uni-cpu --gpus 0 --cpus 8 --time 2h --yes -- \\
      python scripts/blind_valid_row.py annotate --jobs 8 --omp 1 \\
          --out results/blind_valid_row/vit2dsp
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


# Cap BLAS threads BEFORE numpy — and therefore before importing blind_corpus,
# which imports numpy itself. The VK solve is BLAS-bound and the grid runs one
# process per core, so an unclamped pool oversubscribes the allocation.
_OMP = _early_arg("--omp", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _OMP)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from blind_corpus import MAX_CHANNELS, SCORE_WINDOW_S, SR, Worker  # noqa: E402

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

DATASET = "dload:DREGON-LM-V4-michaels-valid-full"
SPEC_NAME = "DREGON-LM-V4-michaels-valid-full"
N_FFT = 2048
HOP_LENGTH = 512
N_ROTORS = 4
REGIMES = ("zero", "low", "flight")

#: The four-rotor acceptance cuts (docs/experiments/blind-corpus-annotation.md).
G1_RATIO_DOUBLE_MIN = 1.065
G5_SPREAD_MAX_REV_S = 12.0
PR_HALF_MARGIN_MIN_DB = -1.5
G4_STEP_MAX_REV_S = 1.0

DEFAULT_GATES = "g1,g5"
ALL_GATES = ("g1", "g5", "pr", "g4")

#: The clip metadata names Michael's recording with a rig prefix the published
#: parent frame does not carry. One entry, and a prefix strip behind it.
CLIP_RECORDING_ALIASES = {"michaels_FLY124": "FLY124"}


def parent_recording(clip_rid: str) -> str:
    """The parent recording id a clip's ``recording_id`` refers to."""
    if clip_rid in CLIP_RECORDING_ALIASES:
        return CLIP_RECORDING_ALIASES[clip_rid]
    return clip_rid.split("_", 1)[1] if clip_rid.startswith("michaels_") else clip_rid


def parent_specs() -> list[dict[str, Any]]:
    """The PINNED parent set of the valid split, off the derivation itself.

    Reading the spec rather than a literal keeps the audio this driver
    annotates the audio the clips were cut from, at the same version.
    """
    from data_processing.derivations import SPECS

    sources = SPECS[SPEC_NAME]["gen"]["noise_sources"]
    out: list[dict[str, Any]] = []
    for src in sources:
        uri = str(src["dataset"])
        name, _, version = uri.removeprefix("dload:").partition("@")
        out.append(
            {
                "dataset": name,
                "version": version or None,
                "recording_ids": [str(r) for r in src["recording_ids"]],
            }
        )
    return out


# ── stage 1: annotate ─────────────────────────────────────────────────────────
def prepare_windows(
    specs: list[dict[str, Any]],
    cache_dir: Path,
    *,
    window_s: float,
    overlap_s: float,
    max_s: float | None = None,
    recordings: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Stream the parents, resample to 16 kHz, cut windows, cache them as NPZ.

    The window law and the cache layout are ``scripts/blind_corpus``'s; what
    differs is the source, which is a list of PINNED ``(dataset, version,
    recording_ids)`` specs because this split is cut from two published
    parents. Caching in the MAIN process keeps the grid workers off the network
    and off the decoder.

    Each window's ``t0_s`` is AUDIO-RELATIVE seconds, which is the convention
    the clip metadata's ``start_time`` also uses, so the two indices join
    directly. Each recording's telemetry is written once to ``ref/<rid>.npz``
    for the calibration reading and for the clip-to-parent verification; the
    ladder never sees it.
    """
    import librosa

    from data_processing.frames import PUBLISHED_RPS_KEYS, get_meta
    from data_processing.streams import iter_published_frames

    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = cache_dir.parent / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    index_p = cache_dir / "index.json"
    if index_p.exists():
        return json.loads(index_p.read_text())

    hop_s = window_s - overlap_s
    if hop_s <= 0:
        raise SystemExit("--overlap-s must be smaller than --window-s")

    index: list[dict[str, Any]] = []
    for spec in specs:
        wanted = {str(r) for r in spec["recording_ids"]}
        if recordings is not None:
            wanted &= recordings
        if not wanted:
            continue
        for frame in iter_published_frames(str(spec["dataset"]), spec["version"]):
            rid = str(get_meta(frame, "recording_id", ""))
            if rid not in wanted:
                continue
            aud = frame["audio"]
            data = np.asarray(aud.data, dtype=np.float32)
            if data.ndim == 1:
                data = data[None, :]
            data = data[:MAX_CHANNELS]
            sr = int(round(float(aud.tindex.sr)))
            if max_s is not None:
                data = data[:, : int(round(max_s * sr))]
            if sr != SR:
                data = librosa.resample(data, orig_sr=sr, target_sr=SR, axis=-1, res_type="soxr_hq")
            data = np.ascontiguousarray(data, dtype=np.float32)

            # EVERY rotor track the parent carries, because the published clip
            # labels do not always come from the first one: DREGON-frames now
            # ships both `motors_measured` and `motors_command`, and the
            # verification has to say which of them the clips were cut from.
            # The window's own reference — the ladder's calibration reading —
            # stays the FIRST key, which is the tachometer.
            ref_r = ref_t = None
            tracks: dict[str, np.ndarray] = {}
            for key in PUBLISHED_RPS_KEYS:
                if key not in frame:
                    continue
                ent = frame[key]
                vals = np.atleast_2d(np.asarray(ent.data, dtype=np.float64))
                try:
                    # DREGON stamps are absolute (Unix clock); every index in
                    # this driver is audio-relative seconds.
                    stamps = np.asarray(ent.tindex.abs_stamps, dtype=np.float64) - float(
                        aud.t_start
                    )
                except AttributeError:
                    stamps = np.arange(vals.shape[-1]) / float(ent.tindex.sr)
                tracks[f"r_{key}"] = vals.astype(np.float32)
                tracks[f"t_{key}"] = stamps
                if ref_r is None:
                    ref_r, ref_t = vals, stamps
            if tracks:
                tracks["keys"] = np.asarray([k[2:] for k in list(tracks) if k.startswith("r_")])
                with (ref_dir / f"{_safe(rid)}.npz").open("wb") as fh:
                    np.savez_compressed(fh, **tracks)  # pyright: ignore[reportArgumentType]

            n = data.shape[-1]
            w = int(round(window_s * SR))
            h = int(round(hop_s * SR))
            starts = list(range(0, max(1, n - w + 1), h)) or [0]
            # A tail longer than a fifth of a window earns a backed-up window
            # rather than a short (and differently conditioned) solve.
            if n - (starts[-1] + w) > w // 5:
                starts.append(max(0, n - w))
            for wi, s0 in enumerate(starts):
                seg = data[:, s0 : s0 + w]
                if seg.shape[-1] < int(round(2.0 * SR)):
                    continue
                uid = f"{_safe(rid)}__w{wi:03d}"
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
                        "t0_s": round(s0 / SR, 6),
                        "dur_s": round(seg.shape[-1] / SR, 6),
                        "n_channels": int(seg.shape[0]),
                        "native_sr": sr,
                        "has_reference": "ref_rps" in payload,
                    }
                )
            print(f"  {rid}: {n / SR:.1f}s, {data.shape[0]}ch -> {len(starts)} windows", flush=True)

    index_p.write_text(json.dumps(index, indent=1))
    return index


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def annotate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    cache_dir = out_dir / "windows"
    recordings = (
        {r.strip() for r in args.recordings.split(",") if r.strip()} if args.recordings else None
    )
    print("Preparing windows from the pinned valid-split parents ...", flush=True)
    index = prepare_windows(
        parent_specs(),
        cache_dir,
        window_s=args.window_s,
        overlap_s=args.overlap_s,
        max_s=args.max_s,
        recordings=recordings,
    )
    if not index:
        raise SystemExit("no windows produced")
    if args.windows:
        keep = {int(w) for w in args.windows.split(",")}
        index = [e for e in index if e["window"] in keep]

    units = [Unit(uid=e["uid"], params={**e, "arm": args.arm, "dataset": SPEC_NAME}) for e in index]
    print(f"{len(units)} units (arm {args.arm})", flush=True)

    res = gridrun_from_args(
        args,
        units,
        Worker(
            cache_dir,
            N_ROTORS,
            args.k_max,
            out_dir / "traj",
            args.alias_penalty,
            args.score_window_s,
            not args.no_per_rotor_octave,
        ),
        out_dir,
        blas_threads=int(args.omp),
        summarize=_annotate_summary,
    )
    return res.exit_code


def _annotate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _sum(key: str) -> float:
        return float(sum(r[key] for r in rows if isinstance(r.get(key), (int, float))))

    audio_s = float(sum(float(r.get("dur_s", 0.0)) for r in rows))
    cpu_s = (
        _sum("wall_ladder_s") + _sum("wall_fvk_s") + _sum("wall_ridge_s") + _sum("wall_octave_s")
    )
    ref = [r["ref_mae_rev_s"] for r in rows if isinstance(r.get("ref_mae_rev_s"), (int, float))]
    return {
        "n_units": len(rows),
        "audio_s": round(audio_s, 1),
        "cpu_s": round(cpu_s, 1),
        "cpu_s_per_audio_s": round(cpu_s / audio_s, 2) if audio_s else None,
        "wall_ladder_s": round(_sum("wall_ladder_s"), 1),
        "wall_fvk_s": round(_sum("wall_fvk_s"), 1),
        "wall_ridge_s": round(_sum("wall_ridge_s"), 1),
        "wall_octave_s": round(_sum("wall_octave_s"), 1),
        # A CALIBRATION reading, never an input: the per-window PIT error of
        # the blind annotation against the parent's telemetry.
        "ref_mae_rev_s": {
            "median": round(float(np.median(ref)), 3) if ref else None,
            "min": round(float(np.min(ref)), 3) if ref else None,
            "max": round(float(np.max(ref)), 3) if ref else None,
            "n": len(ref),
        },
    }


# ── stage 2: score ────────────────────────────────────────────────────────────
def _gate_window(row: dict[str, Any], gates: set[str]) -> list[str]:
    """The gate names this window FAILS (an empty list is an accepted window)."""
    failed: list[str] = []
    if "g1" in gates:
        ratio = row.get("fvk_ratio_double")
        if not isinstance(ratio, (int, float)) or ratio < G1_RATIO_DOUBLE_MIN:
            failed.append("g1")
    if "g5" in gates:
        spread = row.get("spread_rev_s")
        if not isinstance(spread, (int, float)) or spread > G5_SPREAD_MAX_REV_S:
            failed.append("g5")
    if "pr" in gates:
        margin = row.get("pr_margin_half_min_db")
        if isinstance(margin, (int, float)) and margin <= PR_HALF_MARGIN_MIN_DB:
            failed.append("pr")
    return failed


def _apply_g4(windows: list[dict[str, Any]]) -> None:
    """Continuity: mark both members of every disagreeing consecutive pair.

    The two windows of a pair are compared over their OVERLAP, on the rotor
    assignment the Hungarian match gives — a window names its rotors in its own
    order, so a raw row-by-row difference would read a permutation as a jump.
    """
    from tracking.protocols import pit_align

    for a, b in zip(windows, windows[1:], strict=False):
        lo = max(a["t0_s"], b["t0_s"])
        hi = min(a["t0_s"] + a["dur_s"], b["t0_s"] + b["dur_s"])
        if hi - lo <= 0.5:
            continue
        t = np.linspace(lo, hi, 32)
        ra = _interp_window(a, t)
        rb = _interp_window(b, t)
        rb_al, _perm = pit_align(rb, ra, cost="mae")
        step = float(np.max(np.abs(np.asarray(rb_al) - ra)))
        a["g4_step_rev_s"] = max(float(a.get("g4_step_rev_s", 0.0)), step)
        b["g4_step_rev_s"] = max(float(b.get("g4_step_rev_s", 0.0)), step)
        if step > G4_STEP_MAX_REV_S:
            for w in (a, b):
                if "g4" not in w["failed_gates"]:
                    w["failed_gates"].append("g4")


def _interp_window(win: dict[str, Any], t_abs: np.ndarray) -> np.ndarray:
    """The window's trajectory at ABSOLUTE recording times, held at the edges."""
    t_local = t_abs - win["t0_s"]
    return np.stack([np.interp(t_local, win["ft"], row) for row in win["rps"]])


def load_windows(annot_dir: Path, gates: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Every annotated window, by recording, with its trajectory and verdict."""
    raw = annot_dir / "raw"
    traj = annot_dir / "traj"
    by_rec: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(raw.glob("*.json")):
        row = json.loads(path.read_text())
        blob = np.load(traj / f"{path.stem}.npz")
        win = {
            "uid": row["uid"],
            "recording_id": str(row["recording_id"]),
            "t0_s": float(row["t0_s"]),
            "dur_s": float(row["dur_s"]),
            "rps": np.asarray(blob["rps"], dtype=np.float64),
            "ft": np.asarray(blob["ft"], dtype=np.float64),
            "failed_gates": _gate_window(row, gates),
            "fvk_ratio_double": row.get("fvk_ratio_double"),
            "spread_rev_s": row.get("spread_rev_s"),
            "pr_margin_half_min_db": row.get("pr_margin_half_min_db"),
            "ref_mae_rev_s": row.get("ref_mae_rev_s"),
            "cpu_s": round(
                sum(
                    float(row.get(k) or 0.0)
                    for k in ("wall_ladder_s", "wall_fvk_s", "wall_ridge_s", "wall_octave_s")
                ),
                1,
            ),
        }
        by_rec.setdefault(win["recording_id"], []).append(win)
    for wins in by_rec.values():
        wins.sort(key=lambda w: w["t0_s"])
        if "g4" in gates:
            _apply_g4(wins)
    # A unit that failed leaves a .err and no JSON, so its span simply has no
    # window and the zero convention covers it. Say so out loud.
    n_err = len(list(raw.glob("*.err"))) if raw.is_dir() else 0
    if n_err:
        print(f"[blind_valid_row] {n_err} failed unit(s) -> those spans decode to 0", flush=True)
    return by_rec


def predict(windows: list[dict[str, Any]], t_abs: np.ndarray, *, gated: bool) -> np.ndarray:
    """The stitched track at ABSOLUTE times: midpoint cut, refusal -> 0 rev/s.

    Each time takes the trajectory of the covering window whose center is
    nearest. A time no window covers, and a time whose chosen window is
    refused, read 0 rev/s on every rotor.
    """
    out = np.zeros((N_ROTORS, t_abs.size), dtype=np.float64)
    if not windows:
        return out
    centers = np.array([w["t0_s"] + 0.5 * w["dur_s"] for w in windows])
    starts = np.array([w["t0_s"] for w in windows])
    ends = np.array([w["t0_s"] + w["dur_s"] for w in windows])
    covers = (t_abs[None, :] >= starts[:, None]) & (t_abs[None, :] < ends[:, None])
    dist = np.where(covers, np.abs(t_abs[None, :] - centers[:, None]), np.inf)
    choice = np.argmin(dist, axis=0)
    covered = np.isfinite(dist[choice, np.arange(t_abs.size)])
    for wi, win in enumerate(windows):
        sel = covered & (choice == wi)
        if not sel.any():
            continue
        if gated and win["failed_gates"]:
            continue
        out[:, sel] = _interp_window(win, t_abs[sel])
    return out


def clip_index(dataset: str) -> tuple[Path, list[dict[str, Any]]]:
    """The materialized split directory and its per-clip metadata, in order.

    ``DregonLMFrameDataset`` looks for ``metadata.json`` beside the split
    directory, which a dload materialization does not have, so its ``meta``
    entry comes back empty. The file itself IS in the materialized tree and
    carries ``recording_id`` and ``start_time``, which is the whole join.
    """
    from data_processing.streams import resolve_source

    root = Path(resolve_source(dataset))
    payload = json.loads((root / "metadata.json").read_text())
    if len(payload) != 1:
        raise SystemExit(f"expected one split in {root}/metadata.json, found {list(payload)}")
    items = next(iter(payload.values()))
    return root, sorted(items, key=lambda it: str(it["id"]))


def verify_mapping(
    annot_dir: Path,
    root: Path,
    clips: list[dict[str, Any]],
    targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Prove that every clip is the span of the parent its metadata claims.

    Two readings, and the first one is the identity test:

    - ``raw`` — the clip's own ``rps.npy`` against the parent's telemetry
      SLICED at ``[start_time, start_time + duration)``. The derivation cuts
      exactly that slice, so a correct offset gives the same sample count and
      the same values. Nothing is interpolated and nothing is aligned, so this
      cannot be fitted. Every rotor track the parent carries is tried and the
      best one is reported: the published clips are older than the parent's
      current key set, and the reading names which track they came from.
    - ``grid`` — the parent's telemetry INTERPOLATED at the scoring times
      ``start_time + f * hop / sr`` against the clip's frame target. This is the
      time grid the score itself uses, so it is the reading that certifies the
      grid rather than the offset. Its control is the same measurement at a
      deliberately wrong offset of +4 s.
    """
    ref_dir = annot_dir / "ref"

    rows: list[dict[str, Any]] = []
    for clip in clips:
        rid = parent_recording(str(clip["recording_id"]))
        path = ref_dir / f"{_safe(rid)}.npz"
        if not path.is_file():
            continue
        blob = np.load(path)
        keys = [str(k) for k in np.asarray(blob["keys"]).tolist()]
        t0 = float(clip["start_time"])
        dur = float(clip["duration"])
        target = targets[str(clip["id"])]
        t_local = np.arange(target.shape[-1]) * (HOP_LENGTH / SR)
        raw = np.load(root / str(clip["id"]) / "rps.npy").astype(np.float64)

        best: dict[str, Any] | None = None
        for key in keys:
            ref_r = np.asarray(blob[f"r_{key}"], dtype=np.float64)
            ref_t = np.asarray(blob[f"t_{key}"], dtype=np.float64)

            def _grid_mae(offset: float, _r=ref_r, _t=ref_t, _tl=t_local, _tg=target) -> float:
                got = np.stack([np.interp(offset + _tl, _t, row) for row in _r])
                return float(np.mean(np.abs(got - _tg)))

            sel = (ref_t >= t0) & (ref_t < t0 + dur)
            cut = ref_r[:, sel]
            same_shape = bool(cut.shape == raw.shape)
            cand = {
                "id": str(clip["id"]),
                "recording_id": rid,
                "start_time": t0,
                "parent_track": key,
                "raw_same_shape": same_shape,
                "raw_max_abs_diff": (
                    round(float(np.max(np.abs(cut - raw))), 6) if same_shape else None
                ),
                "raw_frac_equal": (
                    round(float(np.mean(np.abs(cut - raw) < 1e-6)), 6) if same_shape else None
                ),
                "grid_mae_at_claimed": round(_grid_mae(t0), 4),
                "grid_mae_control_plus_4s": round(_grid_mae(t0 + 4.0), 4),
            }
            if best is None or cand["grid_mae_at_claimed"] < best["grid_mae_at_claimed"]:
                best = cand
        if best is not None:
            rows.append(best)
    diffs = [r["raw_max_abs_diff"] for r in rows if r["raw_max_abs_diff"] is not None]
    equal = [r["raw_frac_equal"] for r in rows if r["raw_frac_equal"] is not None]
    claimed = [r["grid_mae_at_claimed"] for r in rows]
    control = [r["grid_mae_control_plus_4s"] for r in rows]
    return {
        "n_clips": len(rows),
        "n_raw_shape_mismatch": sum(1 for r in rows if not r["raw_same_shape"]),
        "parent_tracks": sorted({str(r["parent_track"]) for r in rows}),
        "raw_max_abs_diff": round(max(diffs), 6) if diffs else None,
        "raw_frac_equal_min": round(min(equal), 6) if equal else None,
        "grid_mae_at_claimed_max": round(max(claimed), 4) if claimed else None,
        "grid_mae_at_claimed_median": round(float(np.median(claimed)), 4) if claimed else None,
        "grid_mae_control_median": round(float(np.median(control)), 4) if control else None,
        "n_worse_than_control": sum(1 for a, b in zip(claimed, control, strict=True) if a >= b),
        "clips": rows,
    }


def _pools(err: np.ndarray, groups: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for regime in REGIMES:
        sel = groups == regime
        if not sel.any():
            continue
        vals = err[:, sel].ravel()
        out[regime] = {
            "sum_abs": float(np.abs(vals).sum()),
            "sum_sq": float((vals**2).sum()),
            "n": int(vals.size),
        }
    return out


def _aggregate(per_clip: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, float]] = {}
    for pools in per_clip:
        for regime, stats in pools.items():
            for pool in (regime, "all"):
                cur = acc.setdefault(pool, {"sum_abs": 0.0, "sum_sq": 0.0, "n": 0.0})
                cur["sum_abs"] += stats["sum_abs"]
                cur["sum_sq"] += stats["sum_sq"]
                cur["n"] += stats["n"]
    return {
        pool: {
            "mae": cur["sum_abs"] / cur["n"] if cur["n"] else float("nan"),
            "rmse": float(np.sqrt(cur["sum_sq"] / cur["n"])) if cur["n"] else float("nan"),
            "mse": cur["sum_sq"] / cur["n"] if cur["n"] else float("nan"),
            "n": int(cur["n"]),
            "n_channel_equivalent": int(cur["n"]) * 8,
        }
        for pool, cur in sorted(acc.items())
    }


def score(args: argparse.Namespace) -> int:
    from experiments.classical_rps.valid_eval import _frame_groups, _pit_err

    annot_dir = Path(args.annot)
    gates = (
        set()
        if args.gates.strip().lower() in ("", "none")
        else {g.strip() for g in args.gates.split(",") if g.strip()}
    )
    unknown = gates - set(ALL_GATES)
    if unknown:
        raise SystemExit(f"unknown gate(s): {sorted(unknown)}; valid: {list(ALL_GATES)}")

    by_rec = load_windows(annot_dir, gates)
    root, clips = clip_index(args.dataset)
    if args.clips is not None:
        clips = clips[: int(args.clips)]

    targets: dict[str, np.ndarray] = {}
    per_clip: dict[str, list[dict[str, dict[str, float]]]] = {"gated": [], "ungated": []}
    # The campaign's bars are per RIG, so the row carries a per-recording split
    # of the gated pools beside the pooled numbers.
    per_rec: dict[str, list[dict[str, dict[str, float]]]] = {}
    clip_rows: list[dict[str, Any]] = []
    covered_s = refused_s = 0.0
    for clip in clips:
        cid = str(clip["id"])
        target = _clip_target(root / cid)
        targets[cid] = target
        rid = parent_recording(str(clip["recording_id"]))
        windows = by_rec.get(rid, [])
        t_abs = float(clip["start_time"]) + np.arange(target.shape[-1]) * (HOP_LENGTH / SR)
        groups = _frame_groups(target)
        row: dict[str, Any] = {"id": cid, "recording_id": rid, "start_time": clip["start_time"]}
        for mode in ("gated", "ungated"):
            pred = predict(windows, t_abs, gated=mode == "gated")
            err = _pit_err(pred, target)
            pools = _pools(err, groups)
            per_clip[mode].append(pools)
            row[f"mae_{mode}"] = round(float(err.mean()), 3)
            if mode == "gated":
                per_rec.setdefault(rid, []).append(pools)
                row["zero_frac"] = round(float((pred.max(0) == 0.0).mean()), 3)
        clip_rows.append(row)
        dur = target.shape[-1] * (HOP_LENGTH / SR)
        covered_s += dur
        refused_s += dur * float(row["zero_frac"])

    all_windows = [w for wins in by_rec.values() for w in wins]
    cpu_s = float(sum(w["cpu_s"] for w in all_windows))
    parent_s = float(sum(w["dur_s"] for w in all_windows))
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "annot_dir": str(annot_dir),
        "gates": sorted(gates),
        "n_clips": len(clips),
        "n_windows": len(all_windows),
        "n_windows_accepted": sum(1 for w in all_windows if not w["failed_gates"]),
        "scores": {mode: _aggregate(per_clip[mode]) for mode in ("gated", "ungated")},
        "scores_by_recording": {rid: _aggregate(p) for rid, p in sorted(per_rec.items())},
        "zero_decoded_frac": round(refused_s / covered_s, 4) if covered_s else None,
        "compute": {
            "annotation_cpu_s": round(cpu_s, 1),
            "parent_audio_annotated_s": round(parent_s, 1),
            # The tracker's own rate: CPU-seconds per second of audio it reads.
            "cpu_s_per_audio_s": round(cpu_s / parent_s, 2) if parent_s else None,
            "scored_audio_s": round(covered_s, 1),
            "cpu_s_per_scored_audio_s": round(cpu_s / covered_s, 2) if covered_s else None,
        },
        "windows": [
            {
                "uid": w["uid"],
                "recording_id": w["recording_id"],
                "t0_s": w["t0_s"],
                "failed_gates": w["failed_gates"],
                "fvk_ratio_double": w["fvk_ratio_double"],
                "spread_rev_s": w["spread_rev_s"],
                "pr_margin_half_min_db": w["pr_margin_half_min_db"],
                "g4_step_rev_s": round(float(w["g4_step_rev_s"]), 3)
                if "g4_step_rev_s" in w
                else None,
                "ref_mae_rev_s": w["ref_mae_rev_s"],
                "cpu_s": w["cpu_s"],
            }
            for w in sorted(all_windows, key=lambda x: (x["recording_id"], x["t0_s"]))
        ],
        "clips": clip_rows,
    }
    if not args.no_verify:
        summary["mapping_verification"] = verify_mapping(annot_dir, root, clips, targets)

    out_dir = Path(args.out or annot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    _print_score(summary)
    return 0


def _clip_target(clip_dir: Path) -> np.ndarray:
    """The clip's ``(4, F)`` target on the 2048/512 grid, as the dataset builds it.

    ``DregonLMFrameDataset`` reads the clip's raw telemetry and stretches it
    endpoint to endpoint onto ``T // hop + 1`` frames; the same two lines here
    keep the score on the dataset's own grid without loading its audio twice.
    """
    import soundfile as sf

    from data_processing.frame_datasets import stretch_rps_to_frames

    info = sf.info(str(clip_dir / "mixture.wav"))
    n_frames = int(info.frames) // HOP_LENGTH + 1
    raw = np.load(clip_dir / "rps.npy").astype(np.float32)
    return np.asarray(stretch_rps_to_frames(raw, n_frames), dtype=np.float64)


def _print_score(summary: dict[str, Any]) -> None:
    print(
        f"[blind_valid_row] {summary['n_windows_accepted']}/{summary['n_windows']} windows "
        f"accepted under gates {summary['gates']}; "
        f"{summary['zero_decoded_frac']} of scored time decodes to 0",
        flush=True,
    )
    for mode in ("gated", "ungated"):
        for pool, stats in summary["scores"][mode].items():
            print(
                f"  {mode:8s} {pool:7s} MAE {stats['mae']:7.2f}  "
                f"RMSE {stats['rmse']:7.2f}  n={stats['n']}"
            )
    comp = summary["compute"]
    print(
        f"  compute: {comp['annotation_cpu_s']} CPU-s over "
        f"{comp['parent_audio_annotated_s']} s of parent audio "
        f"({comp['cpu_s_per_audio_s']} CPU-s per audio-s)",
        flush=True,
    )
    ver = summary.get("mapping_verification")
    if ver:
        print(
            f"  mapping: parent track(s) {ver['parent_tracks']}, "
            f"{ver['n_raw_shape_mismatch']} shape mismatch, "
            f"{ver['raw_frac_equal_min']} of samples equal at worst, raw max abs diff "
            f"{ver['raw_max_abs_diff']} rev/s; grid MAE max "
            f"{ver['grid_mae_at_claimed_max']} against a control median of "
            f"{ver['grid_mae_control_median']}, {ver['n_worse_than_control']} clip(s) "
            f"no better than the control",
            flush=True,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    ann = sub.add_parser("annotate", help="run the blind ladder on the parent recordings")
    ann.add_argument("--arm", default="vit2dsp", choices=("vit2dsp", "fullrange", "seedvk"))
    ann.add_argument("--window-s", type=float, default=20.0)
    ann.add_argument("--overlap-s", type=float, default=4.0)
    ann.add_argument("--max-s", type=float, default=None, help="cap seconds per recording (smoke)")
    ann.add_argument("--recordings", default=None, help="comma-separated parent recording filter")
    ann.add_argument("--windows", default=None, help="comma-separated window indices to keep")
    ann.add_argument("--k-max", type=int, default=40, help="F_VK harmonic cap")
    ann.add_argument("--score-window-s", type=float, default=SCORE_WINDOW_S)
    ann.add_argument("--alias-penalty", type=float, default=1.0)
    ann.add_argument("--no-per-rotor-octave", action="store_true")
    ann.add_argument("--out", default="results/blind_valid_row/vit2dsp")
    ann.add_argument("--omp", default="1", help="BLAS thread cap (read pre-import)")
    add_gridrun_args(ann, jobs=4)
    ann.set_defaults(func=annotate)

    sc = sub.add_parser("score", help="stitch, decode refusals to zero, score the clips")
    sc.add_argument("--annot", default="results/blind_valid_row/vit2dsp")
    sc.add_argument("--dataset", default=DATASET)
    sc.add_argument("--gates", default=DEFAULT_GATES, help=f"comma list of {list(ALL_GATES)}")
    sc.add_argument("--clips", type=int, default=None, help="cap the number of clips (debug)")
    sc.add_argument("--no-verify", action="store_true", help="skip the clip-to-parent check")
    sc.add_argument("--out", default=None, help="summary.json directory (default: --annot)")
    sc.add_argument("--omp", default="1")
    sc.set_defaults(func=score)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
