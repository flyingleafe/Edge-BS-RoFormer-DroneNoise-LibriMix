"""The P4 refiner probe as a library: does a refiner improve a realistic init?

Probe P4 (2026-09-03) measured the trained HG-CKLA v1 refiner
(``hb_hgckla_ref``) on the frozen real split and wrote the verdict that put
candidate C2 on the shortlist "conditional on three fixes"
(``docs/rps-tracking-architecture-candidates.md`` sections 4 and 5). The
probe was a scratch script; this module is the same five measurements, so
that a v2 run is scored against v1 by the same code and not by a rewrite.

The set is ``experiments.rps_bench.part("real")`` — 296 mono frames = 37
clips of ``DREGON-LM-V4-michaels-valid-full`` x 8 microphones, 8 s at 16 kHz,
labels on the 512-hop grid (T = 251). Scoring is ``rps_bench.pit_mae``
(MAE-optimal assignment over the 24 permutations, per frame). Clips are split
into **ground / ramp / cruise** from the labels alone, exactly as the probe
did, because the three phases answer different questions: cruise is where a
precision stage must win, ramp is where capture range decides, ground is the
zero decision.

The five measurements:

M1
    Corrupted labels, ``RPSCorruption(seed=777)`` with the frame index as the
    seed key. This reproduces the run's own validation, so it says whether
    the checkpoint and the conditioning key are the ones the run used.
M2
    Real predictions as conditioning: the dumped ``r4hb_scv2`` track (the
    best regressor) plus anything passed in ``extra_conds`` — a C1 seed, for
    example. This is the operating point that matters.
M3
    The true labels as conditioning. The refined column IS the refiner's own
    noise floor, and it is the measurement that decides whether a fixed point
    at the truth exists. v1 scores 0.405 rev/s at cruise here.
M4
    Capture range: a constant offset on all four rotors, cruise clips only.
    v1 removes a near-constant ~40 % of the offset inside +-2 rev/s.
M5
    Iteration: feed the refined track back in, ``passes`` times, for every
    conditioning of M1-M3. v1 walks out after one pass (cruise 2.09 -> 2.12
    -> 2.17 rev/s).

Usage::

    from experiments.refiner_bench import run
    run("hb_hgckla_ref_v2", out_dir=Path("results/refiner_bench/v2"))

or through the CLI ``scripts/refiner_bench.py``. ``n_frames`` truncates the
set for a smoke; it makes every number unrepresentative and is labelled as
such in the report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

import zoo
from data_processing.collate import frame_collate
from data_processing.frames import rps_series
from data_processing.rps_corruption import RPSCorruption
from experiments import rps_bench as rb

__all__ = ["PHASES", "clip_phase", "run"]

T_GT = 251  #: label frames of one 8 s clip on the 512-hop 16 kHz grid
MICS = 8  #: microphones per clip in the real part; frame i is clip i // 8
DUMP = Path("results/rps_dump/real")
DEFAULT_OFFSETS = (0.5, 1.0, 2.0, 4.0, 8.0)
PHASES = ("ground", "ramp", "cruise", "all")
HURT = 0.2  #: a frame is "hurt" when its PIT MAE grows by more than this


def clip_phase(gt: np.ndarray) -> str:
    """``(R, T)`` labels -> ``ground`` / ``ramp`` / ``cruise``.

    The probe's rule, kept verbatim: every rotor below 1 rev/s on average is
    a rotor that never spun up, and a mean per-rotor excursion above 15 rev/s
    is a spin-up or a landing. Everything else is flight.
    """
    if np.all(gt.mean(axis=1) < 1.0):
        return "ground"
    if (gt.max(axis=1) - gt.min(axis=1)).mean() > 15.0:
        return "ramp"
    return "cruise"


def _resample_to(arr: np.ndarray, n: int) -> np.ndarray:
    """``(R, T)`` -> ``(R, n)``, linear endpoint to endpoint."""
    t = arr.shape[-1]
    if t == n:
        return arr
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, row) for row in arr])


def _plain_mae(a: np.ndarray, b: np.ndarray) -> float:
    """MAE in the conditioning's own rotor order (no permutation search)."""
    return float(np.abs(a - b).mean())


def _summarize(
    inp: np.ndarray,
    ref: np.ndarray,
    gt: np.ndarray,
    phase: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    """Per-frame input/refined scores plus the per-phase aggregation."""
    n = len(inp)
    si = np.array([score(inp[i], gt[i]) for i in range(n)])
    sr = np.array([score(ref[i], gt[i]) for i in range(n)])
    rows: dict[str, dict[str, float]] = {}
    for p in PHASES:
        m = np.ones(n, bool) if p == "all" else (phase == p)
        if m.sum() == 0:
            continue
        base = float(si[m].mean())
        rows[p] = dict(
            n=int(m.sum()),
            input=base,
            refined=float(sr[m].mean()),
            rel=float((sr[m].mean() - base) / base * 100.0) if base > 0 else float("nan"),
            hurt=float(((sr[m] - si[m]) > HURT).mean()),
        )
    return rows, si, sr


def _fmt_pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:+.1f} %"


def _table(rows: dict[str, dict[str, float]], label: str = "") -> list[str]:
    """One markdown block of the input/refined/rel/hurt table."""
    out = []
    first = True
    for p in PHASES:
        if p not in rows:
            continue
        r = rows[p]
        head = (label if first else "") if label else p
        cells = [head, p] if label else [p]
        out.append(
            "| "
            + " | ".join(
                [
                    *cells,
                    str(int(r["n"])),
                    f"{r['input']:.4f}",
                    f"{r['refined']:.4f}",
                    _fmt_pct(r["rel"]),
                    f"{r['hurt'] * 100:.1f} %",
                ]
            )
            + " |"
        )
        first = False
    return out


class _Refiner:
    """The batched ``cond -> refined`` map over one fixed set of frames."""

    def __init__(self, experiment: str, frames: list, device: str, batch: int = 8) -> None:
        self.frames = frames
        self.batch = batch
        self.model = zoo.load(experiment, device=device)
        self.seconds = 0.0

    def __call__(self, conds: np.ndarray) -> np.ndarray:
        out = np.empty_like(conds)
        t0 = time.time()
        for s in range(0, len(conds), self.batch):
            e = min(s + self.batch, len(conds))
            fr = [
                self.frames[i].with_entry(
                    "rps_cond",
                    rps_series(conds[i].astype(np.float32), sample_rate=16000, hop_length=512),
                )
                for i in range(s, e)
            ]
            pred = self.model(frame_collate(fr))
            out[s:e] = np.asarray(pred["rps_pred"].data, dtype=np.float64)
        self.seconds += time.time() - t0
        return out


def _predict(experiment: str, frames: list, device: str) -> np.ndarray:
    """Regenerate a conditioning: ``zoo.load(experiment)`` over the frames.

    Same readout as ``scripts/rps_dump.py`` (a regressor's ``rps_pred`` as is,
    a salience port through the peak readout), resampled onto the 251-frame
    label grid so the refiner sees the track its dump would have held.
    """
    fm = zoo.load(experiment, device=device)
    reader = rb.Readout()
    out = np.empty((len(frames), 4, T_GT))
    for i, f in enumerate(frames):
        p, _ = reader(fm(f), f)
        a = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0)
        out[i] = _resample_to(a, T_GT)
    return out


def _load_dump(name: str, n: int) -> np.ndarray | None:
    """``results/rps_dump/real/<name>.npz`` as ``(N, 4, 251)``, or None."""
    path = DUMP / f"{name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    pred, n_t = d["pred"], d["n_t"]
    cond = np.empty((n, 4, T_GT))
    for i in range(n):
        a = np.nan_to_num(pred[i, :, : int(n_t[i])].astype(np.float64), nan=0.0)
        cond[i] = _resample_to(a, T_GT)
    return cond


def run(
    experiment: str,
    *,
    extra_conds: dict[str, np.ndarray] | None = None,
    cond_experiments: tuple[str, ...] = ("r4hb_scv2",),
    passes: int = 3,
    out_dir: Path,
    n_frames: int | None = None,
    device: str | None = None,
    batch: int = 8,
    offsets: tuple[float, ...] = DEFAULT_OFFSETS,
) -> dict[str, Any]:
    """Score one refiner checkpoint on the frozen real split (M1-M5).

    Parameters
    ----------
    experiment : str
        An experiment name ``zoo.load`` can resolve, e.g. ``hb_hgckla_ref_v2``.
    extra_conds :
        Extra conditionings for M2 and M5, ``{name: (N, 4, 251) rev/s}`` in
        the frames' own order.
    cond_experiments :
        Zoo experiments whose prediction on the frames is a conditioning
        (M2 and M5). The dump ``results/rps_dump/real/<name>.npz`` is used
        when it exists (the laptop); otherwise the prediction is regenerated
        with ``zoo.load`` (a cluster job has no dumps, only R2).
    passes : int
        Iteration depth of M5 (the refiner is applied to its own output).
    out_dir : Path
        Written: ``results.json`` (every number) and ``REPORT.md``.
    n_frames :
        Truncate the set — a SMOKE only. The frames are ordered clip by clip,
        so a truncated set holds whole clips of one or two phases and none of
        the aggregates mean anything.
    device :
        ``cuda`` when available, else ``cpu`` with 4 threads.

    Returns
    -------
    The same dict that is written to ``results.json``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = rb.part("real")
    if n_frames is not None:
        frames = frames[: int(n_frames)]
    n = len(frames)
    gt = np.stack([np.asarray(f["rps"].data, dtype=np.float64) for f in frames])
    if gt.shape[1:] != (4, T_GT):
        raise ValueError(f"expected (N, 4, {T_GT}) labels, got {gt.shape}")

    # Phase is a property of the CLIP, so it is read once per clip from the
    # first microphone and broadcast to that clip's eight frames.
    n_clips = -(-n // MICS)
    per_clip = [clip_phase(gt[c * MICS]) for c in range(n_clips)]
    phase = np.array([per_clip[i // MICS] for i in range(n)])

    refine = _Refiner(experiment, frames, device, batch=batch)
    results: dict[str, Any] = {
        "experiment": experiment,
        "device": device,
        "n_frames": n,
        "n_clips": n_clips,
        "smoke": n_frames is not None,
        "frames_per_phase": {p: int((phase == p).sum()) for p in PHASES[:-1]},
        "clips_per_phase": {p: per_clip.count(p) for p in PHASES[:-1]},
    }

    # ── M1: the run's own validation condition ────────────────────────────
    corrupt = RPSCorruption(seed=777)
    cond_m1 = np.empty_like(gt)
    gt_m1 = np.empty_like(gt)
    for i in range(n):
        c, g = corrupt(gt[i].astype(np.float32), i)
        cond_m1[i], gt_m1[i] = c, g
    ref_m1 = refine(cond_m1)
    results["M1_pit"], _, _ = _summarize(cond_m1, ref_m1, gt_m1, phase, rb.pit_mae)
    results["M1_plain"], _, _ = _summarize(cond_m1, ref_m1, gt_m1, phase, _plain_mae)

    # ── M2: realistic initializations ─────────────────────────────────────
    conds: dict[str, np.ndarray] = {}
    for name in cond_experiments:
        dumped = _load_dump(name, n)
        conds[name] = dumped if dumped is not None else _predict(name, frames, device)
    for name, arr in (extra_conds or {}).items():
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (n, 4, T_GT):
            raise ValueError(f"extra cond {name!r} must be {(n, 4, T_GT)}, got {a.shape}")
        conds[name] = a
    results["M2"] = {}
    refined_m2: dict[str, np.ndarray] = {}
    for name, cond in conds.items():
        refined_m2[name] = refine(cond)
        results["M2"][name], _, _ = _summarize(cond, refined_m2[name], gt, phase, rb.pit_mae)

    # ── M3: the oracle init, i.e. the refiner's own floor ─────────────────
    ref_m3 = refine(gt.copy())
    results["M3"], _, sr3 = _summarize(gt, ref_m3, gt, phase, rb.pit_mae)
    delta = ref_m3 - gt
    running = gt > 10.0  # a relative bias only means something on a spinning rotor
    for p in PHASES:
        m = np.ones(n, bool) if p == "all" else (phase == p)
        if m.sum() == 0 or p not in results["M3"]:
            continue
        sel = running & m[:, None, None]
        results["M3"][p].update(
            pit_median=float(np.median(sr3[m])),
            pit_p90=float(np.percentile(sr3[m], 90)),
            abs_median=float(np.median(np.abs(delta[m]))),
            abs_p90=float(np.percentile(np.abs(delta[m]), 90)),
            signed_mean=float(delta[m].mean()),
            rel_pct_running=float((delta[sel] / gt[sel]).mean() * 100.0) if sel.any() else 0.0,
        )

    # ── M4: capture range on the cruise clips ─────────────────────────────
    cruise = phase == "cruise"
    results["M4"] = {}
    if cruise.any():
        for off in offsets:
            cond = gt + off
            ref = refine(cond)
            si = np.array([rb.pit_mae(cond[i], gt[i]) for i in range(n)])
            sr = np.array([rb.pit_mae(ref[i], gt[i]) for i in range(n)])
            correction = float((ref - cond)[cruise].mean())
            results["M4"][f"{off}"] = dict(
                n=int(cruise.sum()),
                input=float(si[cruise].mean()),
                refined=float(sr[cruise].mean()),
                mean_correction=correction,
                pull=float(-correction / off),
            )

    # ── M5: iterate on the model's own output ─────────────────────────────
    results["M5"] = {}
    cases: list[tuple[str, np.ndarray, np.ndarray]] = [("corrupt777", cond_m1, gt_m1)]
    cases += [(name, cond, gt) for name, cond in conds.items()]
    cases.append(("oracle", gt.copy(), gt))
    for tag, cond, target in cases:
        series: list[np.ndarray] = [np.array([rb.pit_mae(cond[i], target[i]) for i in range(n)])]
        cur = cond
        for _ in range(passes):
            cur = refine(cur)
            series.append(np.array([rb.pit_mae(cur[i], target[i]) for i in range(n)]))
        results["M5"][tag] = {
            f"pass{p}": {
                q: float(series[p][np.ones(n, bool) if q == "all" else (phase == q)].mean())
                for q in PHASES
                if q == "all" or (phase == q).any()
            }
            for p in range(passes + 1)
        }
        if cruise.any() and passes >= 2:
            best = series[1][cruise]
            results["M5"][tag]["worse_than_pass1"] = {
                f"pass{p}": dict(
                    frac_any=float((series[p][cruise] > best).mean()),
                    frac_gt_0p2=float(((series[p][cruise] - best) > HURT).mean()),
                )
                for p in range(2, passes + 1)
            }

    results["inference_seconds"] = round(refine.seconds, 1)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "REPORT.md").write_text(_report(results))
    return results


def _report(r: dict[str, Any]) -> str:
    """The probe report's own layout, rebuilt from :func:`run`'s numbers."""
    head = "| phase | n | input | refined | rel | hurt (>0.2) |\n|---|---|---|---|---|---|"
    head2 = "| cond | phase | n | input | refined | rel | hurt |\n|---|---|---|---|---|---|---|"
    lines = [
        f"# Does `{r['experiment']}` improve a realistic RPS initialization on real audio?",
        "",
        f'Set: `rps_bench.part("real")` — {r["n_frames"]} frames = {r["n_clips"]} clips x '
        f"{MICS} mics, 8 s, 16 kHz mono, labels on the 512-hop grid (T = {T_GT}). "
        f"Device `{r['device']}`, {r['inference_seconds']} s of inference in total. "
        f"Scoring is `rps_bench.pit_mae` (MAE-optimal assignment over the 24 permutations).",
        "",
        "Phases (per clip, from the labels): "
        + ", ".join(
            f"**{p} {r['clips_per_phase'][p]} clips / {r['frames_per_phase'][p]} frames**"
            for p in PHASES[:-1]
        )
        + ".",
        "",
    ]
    if r["smoke"]:
        lines += [
            "> **SMOKE RUN.** The set was truncated to the first "
            f"{r['n_frames']} frames, which are whole clips of one or two phases only. "
            "Every number below is a wiring check, not a result.",
            "",
        ]
    lines += ["---", "", "## M1 — sanity: validation-identical conditioning", "", head]
    lines += _table(r["M1_pit"])
    lines += ["", "Plain (non-PIT, conditioning order) MAE, all frames: "]
    if "all" in r["M1_plain"]:
        lines[-1] += (
            f"{r['M1_plain']['all']['input']:.4f} -> {r['M1_plain']['all']['refined']:.4f}."
        )
    lines += ["", "---", "", "## M2 — real predictions as conditioning", "", head2]
    for name, rows in r["M2"].items():
        lines += _table(rows, label=f"`{name}`")
    lines += [
        "",
        "---",
        "",
        "## M3 — oracle init: the refiner's own noise floor",
        "",
        "Input PIT MAE is 0 by construction, so the refined column IS the floor.",
        "",
        "| phase | n | refined PIT MAE | median | p90 | median abs err | p90 abs err | signed mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in PHASES:
        if p not in r["M3"]:
            continue
        m = r["M3"][p]
        lines.append(
            f"| {p} | {int(m['n'])} | {m['refined']:.4f} | {m['pit_median']:.4f} | "
            f"{m['pit_p90']:.4f} | {m['abs_median']:.4f} | {m['abs_p90']:.4f} | "
            f"{m['signed_mean']:+.4f} |"
        )
    if "cruise" in r["M3"]:
        lines += [
            "",
            "Relative signed bias on running rotors (GT > 10 rev/s): "
            f"**{r['M3']['cruise']['rel_pct_running']:+.3f} %** at cruise.",
        ]
    lines += [
        "",
        "---",
        "",
        "## M4 — capture range (cruise clips only)",
        "",
        '"Correction" is the mean signed `refined - cond`; "pull" is that as a fraction of '
        "the offset (1.0 would be a full correction).",
        "",
    ]
    if not r["M4"]:
        lines.append("_The set holds no cruise clip, so capture range was not measured._")
    else:
        lines += [
            "| offset (rev/s) | input MAE | refined MAE | mean correction | pull |",
            "|---|---|---|---|---|",
        ]
        for off, m in r["M4"].items():
            lines.append(
                f"| +{off} | {m['input']:.4f} | {m['refined']:.4f} | "
                f"{m['mean_correction']:+.4f} | {m['pull'] * 100:.1f} % |"
            )
    lines += [
        "",
        "---",
        "",
        "## M5 — iterating the refiner on its own output",
        "",
        "| case | pass | " + " | ".join(PHASES) + " |",
        "|---|---|" + "---|" * len(PHASES),
    ]
    for tag, block in r["M5"].items():
        for key, row in block.items():
            if not key.startswith("pass"):
                continue
            cells = [f"{row[p]:.4f}" if p in row else "-" for p in PHASES]
            lines.append(f"| `{tag}` | {key[4:]} | " + " | ".join(cells) + " |")
    for tag, block in r["M5"].items():
        worse = block.get("worse_than_pass1")
        if not worse:
            continue
        lines += ["", f"`{tag}`, cruise frames, relative to the pass-1 result:", ""]
        lines += ["| | worse at all | worse by > 0.2 rev/s |", "|---|---|---|"]
        for key, row in worse.items():
            lines.append(
                f"| {key} | {row['frac_any'] * 100:.1f} % | {row['frac_gt_0p2'] * 100:.1f} % |"
            )
    lines += ["", "---", "", "Raw numbers: `results.json`.", ""]
    return "\n".join(lines)
