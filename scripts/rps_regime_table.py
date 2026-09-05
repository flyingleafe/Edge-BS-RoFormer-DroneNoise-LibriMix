"""Split a `scripts/rps_dump.py` dump by flight regime, rig and microphone group.

One mean over a validation split hides WHERE a model fails. A rotor-speed
predictor that is exact at cruise and lost on the ramp reads the same as one
that is mediocre everywhere. This CLI reads the per-frame predictions the dump
kept and reports the per-frame PIT MAE inside every cell of three axes:

* the **regime** of a frame -- ``zero-frames`` (every rotor at or below
  0.5 rev/s), ``below-30`` (some rotor between 0.5 and 30 rev/s, outside the
  slot-decoder grid), ``in-grid`` (the rest) -- crossed with the **phase** of
  its clip: ``ground`` (all rotor means below 1 rev/s), ``ramp`` (mean rotor
  range above 15 rev/s), ``cruise`` (the rest);
* the **rig** the clip came from (DREGON or FLY124), read from the source
  dataset's ``metadata.json``, because FLY124 is in no training pool and its
  numbers are cross-drone;
* the **microphone group** of the frame -- ``ch0``, ``ch1-7``, ``all`` -- from
  the ``channel`` key of the dump's ``_meta.json``. A dump frame is one
  microphone of one clip, so a model read on ``ch0`` only is read on the
  microphone every single-channel eval uses.

    python scripts/rps_regime_table.py --models r4hb_scv2 hppnet_r4_l4
    python scripts/rps_regime_table.py --dump results/rps_dump/stoch --csv s.csv

Outputs a tidy CSV (``model,rig,regime,mic_group,n_frames,pit_mae``) over every
non-empty cell, and a markdown pivot (one table per microphone group) over the
rows of ``--rows``. The ``all`` microphone group over the real split reproduces
the tables of ``docs/experiments/candidate-tests-2026-09-04.md``.

PIT is the MAE-optimal assignment over all 4! permutations, chosen once per
frame on the label resampled to the prediction's grid -- the convention of
``experiments.rps_bench.pit_mae`` and ``scripts/rps_error_profile.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from itertools import permutations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.rps_bench import resample_like_metric  # noqa: E402

PERMS = list(permutations(range(4)))

ZERO = 0.5  # a rotor at or below this is stopped, for the frame regime
GRID_LOW = 30.0  # the low edge of the slot-decoder rate grid
GROUND_MEAN = 1.0  # a clip whose every rotor mean is below this is on the ground
RAMP_RANGE = 15.0  # a clip whose mean rotor range is above this is a ramp

FRAME_REGIMES = ("zero-frames", "below-30", "in-grid")
PHASES = ("ground", "ramp", "cruise")
MIC_GROUPS = ("ch0", "ch1-7", "all")

#: default markdown rows when the rig of each clip is known (the real split)
ROWS_WITH_RIG = (
    "zero-frames",
    "below-30",
    "DREGON:ramp:in-grid",
    "FLY124:ramp:in-grid",
    "DREGON:cruise:in-grid",
    "FLY124:cruise:in-grid",
    "DREGON:cruise:all",
    "FLY124:cruise:all",
    "ramp:all",
    "ground:all",
    "all",
)
#: ... and when it is not (the synthetic parts)
ROWS_PLAIN = ("zero-frames", "below-30", "ramp:in-grid", "cruise:in-grid", "all")

RIG_DATASET = "DREGON-LM-V4-michaels-valid-full"


# ─── The split ────────────────────────────────────────────────────────────────


def pit_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """``gt`` permuted onto ``pred``'s rotor rows by the MAE-optimal assignment."""
    cost = np.abs(pred[:, None] - gt[None, :]).mean(-1)
    best = min(PERMS, key=lambda p: sum(cost[k, p[k]] for k in range(4)))
    return np.stack([gt[best[k]] for k in range(4)])


def clip_phase(gt: np.ndarray) -> str:
    """``ground`` / ``ramp`` / ``cruise`` from one clip's ``(R, T)`` label."""
    if np.all(gt.mean(axis=1) < GROUND_MEAN):
        return "ground"
    if float((gt.max(axis=1) - gt.min(axis=1)).mean()) > RAMP_RANGE:
        return "ramp"
    return "cruise"


def frame_masks(gt: np.ndarray) -> dict[str, np.ndarray]:
    """``(R, T)`` label -> one boolean ``(T,)`` mask per frame regime."""
    zero = np.asarray((gt <= ZERO).all(0))
    below = np.asarray((~zero) & np.asarray(((gt > ZERO) & (gt < GRID_LOW)).any(0)))
    return {"zero-frames": zero, "below-30": below, "in-grid": ~zero & ~below}


def clip_index(meta: list[dict]) -> tuple[list[int], int]:
    """Per-frame clip index (dataset order) and the number of clips.

    The clip id is ``recording_id`` when the dump carries one (the real split
    tags every frame with its sample directory), else ``sample_id``. Frames of
    one clip are consecutive (clip-major, microphone-minor).
    """
    order: OrderedDict[str, int] = OrderedDict()
    out = []
    for row in meta:
        key = str(row.get("recording_id", row.get("sample_id", "?")))
        out.append(order.setdefault(key, len(order)))
    return out, len(order)


def mic_group(channel: int | None) -> str:
    return "ch0" if channel == 0 else "ch1-7"


def rig_by_clip(n_clips: int, dataset: str) -> dict[int, str] | None:
    """Clip index -> rig name, from the source dataset's ``metadata.json``.

    Returns ``None`` when the dataset is unreachable or holds a different
    number of clips than the dump (a synthetic part), so the rig axis is
    simply dropped instead of being guessed.
    """
    try:
        from data_processing.streams import ensure_local

        root = ensure_local(dataset)
        rows = json.loads((root / "metadata.json").read_text())
        if isinstance(rows, dict):
            rows = next(iter(rows.values()))
    except Exception as exc:  # offline, or no such dataset
        print(f"note: no rig axis ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None
    if len(rows) != n_clips:
        print(
            f"note: no rig axis ({dataset} has {len(rows)} clips, the dump has {n_clips})",
            file=sys.stderr,
        )
        return None
    out = {}
    for i, row in enumerate(rows):
        rid = str(row.get("recording_id", ""))
        upper = rid.upper()
        out[i] = upper.split("_")[-1] if "FLY" in upper else "DREGON"
    return out


def cells(
    dump: Path, models: list[str], rigs: dict[int, str] | None
) -> dict[tuple[str, str, str, str], tuple[float, int]]:
    """``(model, rig, regime, mic_group) -> (pit_mae, n_frames)`` over every cell."""
    z = np.load(dump / "_gt.npz")
    gt_all, gt_n_t = z["rps"], z["n_t"]
    meta = json.loads((dump / "_meta.json").read_text())
    clips, n_clips = clip_index(meta)
    if rigs is not None and set(clips) - set(rigs):
        raise SystemExit("the rig map does not cover every clip of the dump")

    # the per-frame masks and the clip phase do not depend on the model
    per_frame = []
    for i in range(gt_all.shape[0]):
        g = gt_all[i, :, : int(gt_n_t[i])].astype(np.float64)
        rig = "all" if rigs is None else rigs[clips[i]]
        per_frame.append((g, clip_phase(g), rig, mic_group(meta[i].get("channel"))))

    acc: dict[tuple[str, str, str, str], list[float]] = {}
    for exp in models:
        d = np.load(dump / f"{exp}.npz")
        pred_all, pred_n_t = d["pred"], d["n_t"]
        for i, (g, phase, rig, mic) in enumerate(per_frame):
            p = np.nan_to_num(pred_all[i, :, : int(pred_n_t[i])].astype(np.float64), nan=0.0)
            if p.shape[-1] != g.shape[-1]:
                p = resample_like_metric(p, g.shape[-1])
            err = np.abs(p - pit_align(p, g)).mean(0)  # (T,) mean over rotors
            masks = frame_masks(g)
            masks["all"] = np.ones(g.shape[-1], dtype=bool)
            for regime, m in masks.items():
                n = int(m.sum())
                if not n:
                    continue
                s = float(err[m].sum())
                for r in {rig, "all"}:
                    for mg in {mic, "all"}:
                        for label in (regime, f"{phase}:{regime}"):
                            e = acc.setdefault((exp, r, label, mg), [0.0, 0])
                            e[0] += s
                            e[1] += n
    return {k: (v[0] / v[1], int(v[1])) for k, v in acc.items()}


# ─── Output ───────────────────────────────────────────────────────────────────


def split_row(spec: str, rigs: set[str]) -> tuple[str, str]:
    """``"DREGON:cruise:all"`` -> ``("DREGON", "cruise:all")``."""
    head, _, rest = spec.partition(":")
    return (head, rest) if head in rigs and rest else ("all", spec)


def markdown(
    table: dict[tuple[str, str, str, str], tuple[float, int]],
    models: list[str],
    rows: list[str],
    rigs: set[str],
) -> str:
    out = []
    for mg in MIC_GROUPS:
        present = [(split_row(r, rigs), r) for r in rows]
        present = [(k, r) for k, r in present if any((m, *k, mg) in table for m in models)]
        if not present:
            continue
        out += [f"\n{mg}:\n", "| regime (frames) | " + " | ".join(models) + " |"]
        out.append("|---|" + "---|" * len(models))
        for (rig, regime), label in present:
            n = next(
                (table[(m, rig, regime, mg)][1] for m in models if (m, rig, regime, mg) in table), 0
            )
            vals = [table.get((m, rig, regime, mg), (None, 0))[0] for m in models]
            best = min((v for v in vals if v is not None), default=None)
            cs = ["-" if v is None else (f"**{v:.2f}**" if v == best else f"{v:.2f}") for v in vals]
            out.append(f"| {label} ({n}) | " + " | ".join(cs) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dump", default="results/rps_dump/real", help="one dumped set's directory")
    ap.add_argument("--models", nargs="*", default=None, help="default: every .npz of the dump")
    ap.add_argument("--csv", default=None, help="write the tidy per-cell CSV here")
    ap.add_argument("--md", default=None, help="write the markdown pivot here (else stdout)")
    ap.add_argument("--rows", nargs="*", default=None, help="markdown rows, `[rig:]regime`")
    ap.add_argument("--rig-dataset", default=RIG_DATASET, help="source of the per-clip rig")
    ap.add_argument("--no-rigs", action="store_true", help="drop the rig axis")
    a = ap.parse_args()

    dump = Path(a.dump)
    models = a.models or sorted(f.stem for f in dump.glob("*.npz") if not f.stem.startswith("_"))
    if not models:
        raise SystemExit(f"no model .npz in {dump}")
    missing = [m for m in models if not (dump / f"{m}.npz").exists()]
    if missing:
        raise SystemExit(f"not in {dump}: {', '.join(missing)}")

    _, n_clips = clip_index(json.loads((dump / "_meta.json").read_text()))
    rigs = None if a.no_rigs else rig_by_clip(n_clips, a.rig_dataset)
    table = cells(dump, models, rigs)
    rig_names = {r for _, r, _, _ in table} - {"all"}

    lines = ["model,rig,regime,mic_group,n_frames,pit_mae"]
    for (m, r, regime, mg), (mae, n) in sorted(table.items()):
        lines.append(f"{m},{r},{regime},{mg},{n},{mae:.6f}")
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        Path(a.csv).write_text("\n".join(lines) + "\n")
        print(f"wrote {a.csv} ({len(lines) - 1} cells)", file=sys.stderr)

    rows = [str(r) for r in (a.rows or (ROWS_WITH_RIG if rigs is not None else ROWS_PLAIN))]
    md = markdown(table, models, rows, rig_names)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(md + "\n")
        print(f"wrote {a.md}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
