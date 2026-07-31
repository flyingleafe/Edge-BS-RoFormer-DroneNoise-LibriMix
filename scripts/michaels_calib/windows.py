#!/usr/bin/env python3
"""Michael's-recording windows for the telemetry calibration (FLY124 + FLY125).

Replays the frozen beat-VK windowing protocol against the LOCAL RAW recordings,
so it works for **both** FLY124 (which is in ``beatvk-valid-raw``) and FLY125
(which is not — it is the TRAINING recording and was never published):

  ``scripts/publish_beatvk_valid.py`` — re-anchor audio to t=0, trim the leading
      and trailing maximal exact-constant telemetry runs, tile contiguous
      non-overlapping 16 s windows over the eval span, regime-tag each window
      from the 0.032 s-grid mean rps (<5 ground, <45 warmup, else cruise).
  ``scripts/beatvk_vk_arms.py:build_preps`` — soxr_hq resample of the native
      44.1 kHz 8-ch audio to 16 kHz, slice by sample index, telemetry linearly
      interpolated onto the window frame grid, ``edge`` = 0.5 s guard.

Rebuilding FLY124 this way reproduces the published window manifest and the
cached prep NPZs bit-for-bit (see :func:`selfcheck`, which is a no-op when the
frozen artefacts are not present, e.g. on a cluster worktree).

``time_offset`` / ``time_dilation`` / ``rps_offset`` / ``rps_scale`` are
parameters here, so hypotheses about the shipped alignment constants can be
built and scored **without touching** ``src/data_processing/michaels.py``.

Data root resolution (in order, first hit wins):
  1. ``$DATA_ROOT`` — if it actually contains the recordings;
  2. ``<repo>/data`` — the developer checkout;
  3. dload ``recording_with_motor_speed`` via
     :func:`data_processing.streams.ensure_local` — the cluster path (R2
     credentials ship with the job in ``.env``).
Both tree layouts are accepted: a *project data root* containing
``recording_with_motor_speed/recording_1/...`` and a *dataset root* whose
children are ``recording_1/`` and ``recording_2/`` directly (what a
dload-materialized tree looks like when the dataset was pushed from inside
``recording_with_motor_speed``).
"""

from __future__ import annotations

import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from data_processing.michaels import MICHAELS_FILES  # noqa: E402

SR = 16000
FRAME_S = 0.032
WINDOW_S = 16.0
WINDOW_FRAMES = 500
N_ROTORS = 4
DATASET = "recording_with_motor_speed"

#: recording id -> ``(wav_rel, csv_rel, time_offset, time_dilation)``, the
#: SHIPPED alignment constants (imported, never re-typed).
SHIPPED: dict[str, tuple[str, str, float, float]] = {
    "FLY124": MICHAELS_FILES[0],
    "FLY125": MICHAELS_FILES[1],
}


# ────────────────────────────────────────────────────────── data root
@functools.lru_cache(maxsize=1)
def data_root() -> tuple[Path, str]:
    """Return ``(root, how)`` — a directory under which the recordings resolve."""
    probe_rel = Path(SHIPPED["FLY124"][0])  # recording_with_motor_speed/recording_1/124.wav
    inner_rel = probe_rel.relative_to(DATASET)  # recording_1/124.wav

    def ok(root: Path) -> bool:
        return (root / probe_rel).exists() or (root / inner_rel).exists()

    env = os.environ.get("DATA_ROOT")
    if env and ok(Path(env)):
        return Path(env), f"$DATA_ROOT={env}"
    if ok(REPO / "data"):
        return REPO / "data", "<repo>/data"
    from data_processing.streams import ensure_local

    tree = ensure_local(DATASET)
    if not ok(tree):
        raise SystemExit(
            f"dload dataset {DATASET!r} materialized to {tree} but "
            f"neither {probe_rel} nor {inner_rel} is there"
        )
    return tree, f"dload:{DATASET} -> {tree}"


def michaels_paths(rid: str) -> tuple[Path, Path]:
    """``(wav, csv)`` for one recording, under whichever root resolved."""
    root, _how = data_root()
    wav_rel, csv_rel = Path(SHIPPED[rid][0]), Path(SHIPPED[rid][1])
    wav, csv = root / wav_rel, root / csv_rel
    if wav.exists() and csv.exists():
        return wav, csv
    wav = root / wav_rel.relative_to(DATASET)
    csv = root / csv_rel.relative_to(DATASET)
    if wav.exists() and csv.exists():
        return wav, csv
    raise SystemExit(f"{rid}: no {wav_rel} (nor the dataset-root layout) under {root}")


# ────────────────────────────────────────────────────────── windowing
@dataclass
class Window:
    """One 16 s window: 16 kHz audio + telemetry on the 0.032 s frame grid."""

    name: str  # "fly125_w03"
    rid: str  # "FLY125"
    widx: int
    regime: str
    start_s: float
    end_s: float
    audio: np.ndarray  # (8, T) float32 @ SR
    ft: np.ndarray  # (N,) seconds, window-relative
    r_meas: np.ndarray  # (4, N) rev/s

    @property
    def t_centre(self) -> float:
        return 0.5 * (self.start_s + self.end_s)


def window_name(rid: str, widx: int) -> str:
    return f"{rid.lower()}_w{widx:02d}"


def regime_of(mean_rps: float) -> str:
    if mean_rps < 5.0:
        return "ground"
    if mean_rps < 45.0:
        return "warmup"
    return "cruise"


def trim_constant_runs(ts: np.ndarray, vals: np.ndarray) -> tuple[float, float]:
    """Drop the leading/trailing maximal EXACT-constant telemetry runs."""
    same = np.all(vals[:, 1:] == vals[:, :-1], axis=0)
    lead = 0
    while lead < len(same) and same[lead]:
        lead += 1
    trail = 0
    while trail < len(same) and same[len(same) - 1 - trail]:
        trail += 1
    return float(ts[lead]), float(ts[len(ts) - 1 - trail])


def load_recording(
    rid: str, *, time_offset: float | None = None, time_dilation: float | None = None
) -> dict[str, Any]:
    """16 kHz audio + aligned telemetry + the 16 s window manifest.

    Defaults to the SHIPPED constants; pass either to build a hypothesis.
    """
    import librosa as lr

    from data_processing.michaels import _load_michaels_data_raw

    off = SHIPPED[rid][2] if time_offset is None else float(time_offset)
    dil = SHIPPED[rid][3] if time_dilation is None else float(time_dilation)
    wav_path, csv_path = michaels_paths(rid)
    wav, ts, ms, sr = _load_michaels_data_raw(
        wav_path, csv_path, time_offset=off, time_dilation=dil, sr=None
    )
    audio16 = np.atleast_2d(
        lr.resample(
            np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=SR, res_type="soxr_hq", axis=-1
        )
    )
    del wav
    t_end = audio16.shape[-1] / SR
    live0, live1 = trim_constant_runs(ts, ms)
    span0, span1 = max(0.0, live0), min(t_end, live1)

    windows: list[dict[str, Any]] = []
    start = span0
    while start + WINDOW_S <= span1 + 1e-9:
        grid = start + np.arange(WINDOW_FRAMES) * FRAME_S
        per_rotor = np.stack([np.interp(grid, ts, ms[r]) for r in range(N_ROTORS)])
        windows.append(
            {
                "index": len(windows),
                "name": window_name(rid, len(windows)),
                "start_s": round(start, 6),
                "end_s": round(start + WINDOW_S, 6),
                "regime": regime_of(float(per_rotor.mean())),
                "mean_rps": round(float(per_rotor.mean()), 4),
                "gt_mean": np.round(per_rotor.mean(1), 4).tolist(),
                "gt_std": np.round(per_rotor.std(1), 4).tolist(),
            }
        )
        start += WINDOW_S
    return {
        "rid": rid,
        "time_offset": off,
        "time_dilation": dil,
        "audio": audio16,
        "ts": ts,
        "vals": ms,
        "eval_span": [round(span0, 6), round(span1, 6)],
        "windows": windows,
    }


def cut_window(
    rec: dict[str, Any], widx: int, *, rps_offset: float = 0.0, rps_scale: float = 1.0
) -> Window:
    w = rec["windows"][widx]
    start, end = float(w["start_s"]), float(w["end_s"])
    a0, a1 = int(round(start * SR)), int(round(end * SR))
    seg = np.ascontiguousarray(rec["audio"][:, a0:a1])
    ft = np.arange(0.0, (a1 - a0) / SR - FRAME_S / 2, FRAME_S)
    ts, vals = rec["ts"], rec["vals"]
    r_meas = np.stack([np.interp(ft + start, ts, vals[i]) for i in range(N_ROTORS)])
    r_meas = r_meas * rps_scale + rps_offset
    return Window(
        name=str(w["name"]),
        rid=rec["rid"],
        widx=widx,
        regime=str(w["regime"]),
        start_s=start,
        end_s=end,
        audio=seg,
        ft=ft,
        r_meas=r_meas,
    )


# ─────────────────────────────────────────────────── prep cache (NPZ)
def cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.npz"


def build_cache(
    cache_dir: Path,
    rids: tuple[str, ...] = ("FLY124", "FLY125"),
    *,
    time_offsets: dict[str, float] | None = None,
    time_dilations: dict[str, float] | None = None,
    rps_offset: float = 0.0,
    rps_scale: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize every window of ``rids`` as a small NPZ; return the manifest.

    The 44.1 kHz load + resample is the memory hog (~0.6 GB for FLY125), so it
    happens ONCE, here, in the parent — workers then read ~8 MB NPZs. One
    recording is held at a time.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"data_root": data_root()[1], "recordings": {}}
    for rid in rids:
        off = (time_offsets or {}).get(rid)
        dil = (time_dilations or {}).get(rid)
        rec = load_recording(rid, time_offset=off, time_dilation=dil)
        for w in rec["windows"]:
            p = cache_path(cache_dir, str(w["name"]))
            if p.exists() and not force:
                continue
            win = cut_window(rec, int(w["index"]), rps_offset=rps_offset, rps_scale=rps_scale)
            np.savez(
                p,
                audio=win.audio.astype(np.float32),
                ft=win.ft,
                r_meas=win.r_meas,
                meta=json.dumps(
                    {
                        "name": win.name,
                        "rid": win.rid,
                        "widx": win.widx,
                        "regime": win.regime,
                        "start_s": win.start_s,
                        "end_s": win.end_s,
                    }
                ),
            )
        manifest["recordings"][rid] = {
            "time_offset": rec["time_offset"],
            "time_dilation": rec["time_dilation"],
            "eval_span": rec["eval_span"],
            "rps_offset": rps_offset,
            "rps_scale": rps_scale,
            "windows": rec["windows"],
        }
        del rec
    return manifest


def load_cached(cache_dir: Path, name: str) -> Window:
    with np.load(cache_path(cache_dir, name), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        return Window(
            name=meta["name"],
            rid=meta["rid"],
            widx=int(meta["widx"]),
            regime=meta["regime"],
            start_s=float(meta["start_s"]),
            end_s=float(meta["end_s"]),
            audio=np.asarray(z["audio"], dtype=np.float64),
            ft=np.asarray(z["ft"], dtype=np.float64),
            r_meas=np.asarray(z["r_meas"], dtype=np.float64),
        )


def selfcheck() -> bool:
    """FLY124 rebuilt here must match the frozen beat-VK prep cache.

    Returns True if the check ran and passed, False if the frozen artefacts are
    absent (cluster worktree) — never raises for a missing reference.
    """
    man_p = REPO / "results/beatvk_vk_arms/manifest.json"
    if not man_p.exists():
        print("selfcheck: no results/beatvk_vk_arms/manifest.json — skipped", flush=True)
        return False
    ref = json.loads(man_p.read_text())["recordings"]["FLY124"]["windows"]
    rec = load_recording("FLY124")
    print(f"selfcheck: eval span {rec['eval_span']}", flush=True)
    worst = 0.0
    for a, b in zip(ref, rec["windows"], strict=True):
        assert abs(a["start_s"] - b["start_s"]) < 1e-6, (a, b)
        assert a["regime"] == b["regime"], (a, b)
        p = REPO / f"results/beatvk_vk_arms/prep_cache/FLY124__w{a['index']:02d}.npz"
        if not p.exists():
            continue
        win = cut_window(rec, int(a["index"]))
        with np.load(p) as z:
            da = float(np.max(np.abs(z["audio"] - win.audio)))
            dr = float(np.max(np.abs(z["r_meas"] - win.r_meas)))
        worst = max(worst, da, dr)
        print(f"  w{a['index']:02d} max|Δaudio| {da:.3e}  max|Δr_meas| {dr:.3e}", flush=True)
    print(f"selfcheck: worst deviation {worst:.3e}", flush=True)
    return worst < 1e-6


if __name__ == "__main__":
    root, how = data_root()
    print(f"data root: {root}  ({how})")
    selfcheck()
