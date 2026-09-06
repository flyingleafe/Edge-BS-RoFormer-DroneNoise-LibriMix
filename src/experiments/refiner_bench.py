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
from collections.abc import Callable, Mapping, Sized
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, IterableDataset

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


# JHTR diagnostics deliberately leave the historical run() and its matching
# conventions untouched. This path consumes the parent's fixed Frames verbatim.
def paired_group_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired reference-minus-candidate mean; resample whole original flights.

    Microphones, crops and speech variants move together. The point estimate
    remains sample-weighted, as in the benchmark; each bootstrap draw resamples
    group sums AND counts, rather than treating unequal groups as equal samples.
    """
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    groups = np.asarray(groups, dtype=str)
    if reference.shape != candidate.shape or reference.shape != groups.shape or reference.ndim != 1:
        raise ValueError("reference, candidate and groups must be paired vectors")
    if not len(groups) or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("bootstrap requires nonempty finite paired observations")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    names, inverse = np.unique(groups, return_inverse=True)
    delta = reference - candidate
    result: dict[str, Any] = {
        "improvement": float(delta.mean()),
        "n_examples": len(groups),
        "n_groups": len(names),
        "resamples": resamples,
        "seed": seed,
    }
    if len(names) < 2:
        return {
            **result,
            "status": "unestablished",
            "reason": "fewer than two independent groups",
            "ci95": None,
        }
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for start in range(0, resamples, 256):
        indices = rng.integers(len(names), size=(min(256, resamples - start), len(names)))
        draws[start : start + len(indices)] = sums[indices].sum(1) / counts[indices].sum(1)
    return {**result, "status": "measured", "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}


def trajectory_errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """N,R,T -> per-example metrics using the existing two PIT conventions."""
    from metrics.rps import rps_mae_clip, rps_mae_frame, rps_mse

    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have identical N,R,T shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite predictions/targets must not be silently scored")
    return {
        "ordered_mae": np.abs(prediction - target).mean((1, 2)),
        "ordered_mse": ((prediction - target) ** 2).mean((1, 2)),
        "pit_mae_mae_assignment": np.array([rb.pit_mae(p, t) for p, t in zip(prediction, target)]),
        "pit_mae_mse_assignment": np.array(
            [rps_mae_frame(p, t) for p, t in zip(prediction, target)]
        ),
        "pit_mse": np.array([rps_mse(p, t) for p, t in zip(prediction, target)]),
        "pit_mae_clip": np.array([rps_mae_clip(p, t) for p, t in zip(prediction, target)]),
    }


def recovery_intervals(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    dt: float = 512 / 16000,
    tolerance: float = 1.0,
    duration: float = 0.5,
) -> dict[str, np.ndarray]:
    """All-four recovery after ONE whole-crop MSE-optimal permutation.

    A successful interval's first and last timestamps must span >= duration;
    16 samples at 32 ms span only .480 s and therefore do not qualify.
    This is an offline interval, never a causal recovery latency.
    """
    from itertools import permutations

    if prediction.shape != target.shape or prediction.ndim != 3 or dt <= 0 or duration <= 0:
        raise ValueError("expected equal N,R,T arrays and positive time scales")
    aligned = np.empty_like(prediction)
    success = np.zeros(len(prediction), bool)
    first = np.full(len(prediction), np.nan)
    longest = np.zeros(len(prediction))
    perms = list(permutations(range(prediction.shape[1])))
    for i, (pred, truth) in enumerate(zip(prediction, target)):
        perm = min(perms, key=lambda p: float(((pred[list(p)] - truth) ** 2).mean()))
        aligned[i] = pred[list(perm)]
        good = np.all(np.abs(aligned[i] - truth) <= tolerance, axis=0)
        edges = np.diff(np.r_[False, good, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        spans = (ends - starts - 1) * dt
        if len(spans):
            longest[i] = float(spans.max())
            accepted = starts[spans >= duration - 1e-12]
            if len(accepted):
                success[i], first[i] = True, accepted[0] * dt
    return {
        "success": success,
        "first_interval_start_s": first,
        "longest_interval_s": longest,
        "aligned_prediction": aligned,
    }


def precision_gates(
    oracle: np.ndarray,
    offsets: dict[str, tuple[np.ndarray, np.ndarray]],
    target: np.ndarray,
    observable: np.ndarray | None,
) -> dict[str, Any]:
    """Claim gates, never a checkpoint selector; oracle has shape N,S+1,R,T."""
    if observable is None or not np.asarray(observable, bool).any():
        return {
            "status": "unestablished",
            "reason": "no certified >=3-order, >=6dB, >=16Hz, >=0.5s observable coverage",
        }
    mask = np.asarray(observable, bool)
    if (
        mask.shape != target.shape
        or oracle.shape[0] != len(target)
        or oracle.shape[2:] != target.shape[1:]
    ):
        raise ValueError("observable must be N,R,T and oracle must be N,S+1,R,T")
    drift = np.abs(oracle - target[:, None])
    blocks = []
    for step in range(1, oracle.shape[1]):
        values = drift[:, step][mask]
        mean, p95 = float(values.mean()), float(np.percentile(values, 95))
        blocks.append(
            {"step": step, "mean": mean, "p95": p95, "pass": mean <= 0.10 and p95 <= 0.25}
        )
    corrections = {}
    for name, (initial, final) in offsets.items():
        offset = float(name.removeprefix("offset"))
        valid = mask & (target + offset >= 0) & (target + offset <= 150)
        if not valid.any():
            corrections[name] = {"pass": False, "status": "unestablished"}
            continue
        before = float(np.abs(initial - target)[valid].mean())
        after = float(np.abs(final - target)[valid].mean())
        corrections[name] = {
            "input": before,
            "final": after,
            "rotor_frames": int(valid.sum()),
            "pass": before > 0 and after <= 0.8 * before,
        }
    required = {f"offset{v:+g}" for v in (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)}
    correction_pass = required.issubset(corrections) and all(
        c["pass"] for c in corrections.values()
    )
    preservation_pass = bool(blocks) and all(b["pass"] for b in blocks)
    return {
        "status": "measured",
        "oracle": blocks,
        "signed_offsets": corrections,
        "preservation_pass": preservation_pass,
        "correction_pass": correction_pass,
        "precision_pass": preservation_pass and correction_pass,
    }


def diagnostic_guesses(target: np.ndarray) -> dict[str, tuple[np.ndarray, NDArray[np.bool_]]]:
    """Evaluation-only guesses and per-example applicability; never alter audio."""
    n, _, t = target.shape
    all_examples = np.ones(n, bool)
    cases: dict[str, tuple[np.ndarray, NDArray[np.bool_]]] = {
        "oracle": (target.copy(), all_examples)
    }
    for offset in (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0):
        raw = target + offset
        cases[f"offset{offset:+g}"] = (
            np.clip(raw, 0, 150),
            np.asarray(((raw >= 0) & (raw <= 150)).all((1, 2)), dtype=bool),
        )
    active = target > 0.5
    for name, factor in (("half", 0.5), ("double", 2.0)):
        raw = target * factor
        cases[name] = (
            np.clip(raw, 0, 150),
            np.asarray(active.any((1, 2)) & (raw <= 150).all((1, 2)), dtype=bool),
        )
    # Pick the most active row from each EXISTING example, without changing truth.
    anchor = active.sum(2).argmax(1)
    collapse = np.stack(
        [np.broadcast_to(target[i, row], target[i].shape) for i, row in enumerate(anchor)]
    )
    cases["all_collapse"] = (collapse.copy(), np.asarray(active.any((1, 2)), dtype=bool))
    duplicate, missing, wrong = target.copy(), target.copy(), target.copy()
    begin = max(0, (t - int(np.ceil(0.5 / 0.032))) // 2)
    end = min(t, begin + int(np.ceil(0.5 / 0.032)))
    for i, row in enumerate(anchor):
        duplicate[i, (row + 1) % target.shape[1]] = target[i, row]
        missing[i, row] = 0.0
        wrong[i, row, begin:end] = target[i, (row + 1) % target.shape[1], begin:end]
    cases["duplicate"] = (duplicate, np.asarray(np.any(duplicate != target, axis=(1, 2)), bool))
    cases["missing_active"] = (missing, np.asarray(active.any((1, 2)), dtype=bool))
    cases["wrong_track_0p5s"] = (wrong, np.asarray(np.any(wrong != target, axis=(1, 2)), bool))
    false_active = np.where(active, target, 60.0)
    cases["false_active"] = (false_active, np.asarray((~active).any((1, 2)), dtype=bool))
    return cases


def _json_value(value: Any) -> Any:
    """Lossless JSON representation of metadata used for equality/fingerprints."""
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if torch.is_tensor(value):
        return _json_value(value.detach().cpu().numpy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping) or (hasattr(value, "keys") and not isinstance(value, str)):
        return {str(k): _json_value(value[k]) for k in value}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"unsupported metadata type {type(value).__name__}; supply explicit serialization"
    )


def _frame_fingerprint(frame: Any) -> str:
    import hashlib

    from data_processing.frames import meta_dict

    digest = hashlib.sha256()
    digest.update(json.dumps(_json_value(meta_dict(frame)), sort_keys=True).encode())
    for key in ("mixture", "rps", "rps_cond"):
        if key not in frame:
            continue
        series = frame[key]
        array = np.ascontiguousarray(series.data)
        digest.update(key.encode())
        digest.update(str((array.dtype, array.shape, series.dims)).encode())
        digest.update(array.tobytes())
        digest.update(np.asarray(series.tindex.sample_times(), dtype=np.float64).tobytes())
    return digest.hexdigest()


def _separation_masks(target: np.ndarray) -> dict[str, np.ndarray]:
    """Natural nearest-active-neighbour separations; no generated examples."""
    distance = np.abs(target[:, :, None] - target[:, None, :])
    active = target > 0.5
    valid = active[:, :, None] & active[:, None, :]
    valid &= ~np.eye(target.shape[1], dtype=bool)[None, :, :, None]
    nearest = np.where(valid, distance, np.inf).min(2)
    bounds = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf)
    result = {"exact_coincidence": nearest == 0}
    for low, high in zip(bounds[:-1], bounds[1:]):
        result[f"separation_{low:g}_{high:g}"] = (nearest > low if low == 0 else nearest >= low) & (
            nearest < high
        )
    # A strict sign change is identifiable as a crossing; exact-coincidence
    # plateaus are reported separately rather than assigned physical identity.
    crossing = np.zeros_like(active)
    for left in range(target.shape[1]):
        for right in range(left + 1, target.shape[1]):
            delta = target[:, left] - target[:, right]
            hit = (delta[:, :-1] * delta[:, 1:] < 0) & active[:, left, 1:] & active[:, right, 1:]
            crossing[:, left, 1:] |= hit
            crossing[:, right, 1:] |= hit
    result["crossing"] = crossing
    return result


class _DiagnosticForward(Protocol):
    def __call__(
        self, audio: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]: ...


def evaluate_jhtr(
    experiment: str,
    frames: list,
    *,
    out_dir: Path,
    checkpoint: str = "best",
    device: str = "cpu",
    batch: int = 1,
    group_ids: list[str] | None = None,
    independent_sample_ids: bool = False,
    cases: tuple[str, ...] | None = None,
    observable: np.ndarray | None = None,
    observable_provenance: str | None = None,
    model: Any = None,
    smoke: bool = False,
    sample_strata: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Measure a SELECTED checkpoint, including local results/*.ckpt, on fixed data.

    ``group_ids`` must identify original recordings/flights, not microphones.
    Synthetic sample_id is usable only with certified flight_reuse=1. An
    observable mask must be independently certified from existing source
    evidence; absent that metadata the precision claim stays unestablished.
    The model argument is an already-built FrameModel, useful with saved as-run
    configs; its diagnostic call bypasses only the codec's auxiliary discard.
    """
    from data_processing.frames import meta_dict

    if not frames or batch < 1:
        raise ValueError("need nonempty fixed frames and positive batch")
    metadata = [_json_value(meta_dict(f)) for f in frames]
    if group_ids is None:
        group_ids = []
        for meta in metadata:
            key = next((k for k in ("recording_id", "flight_id") if k in meta), None)
            if key is None and independent_sample_ids and "sample_id" in meta:
                key = "sample_id"
            if key is None:
                raise ValueError(
                    "original recording/flight groups unavailable; supply group_ids, never mic indices"
                )
            group_ids.append(f"{key}:{meta[key]}")
    if len(group_ids) != len(frames):
        raise ValueError("one original group id is required per frame")
    groups = np.asarray(group_ids, dtype=str)
    gt = np.stack([np.asarray(f["rps"].data, dtype=np.float64) for f in frames])
    if gt.ndim != 3 or gt.shape[1] != 4:
        raise ValueError("fixed frames must have four equally sampled target rows")
    if observable is not None and (not observable_provenance or np.shape(observable) != gt.shape):
        raise ValueError("observable N,R,T mask requires independently recorded provenance")
    fm = model if model is not None else zoo.load(experiment, ckpt=checkpoint, device=device)
    fm.model.eval()
    diagnostic_forward = getattr(fm.model, "forward_with_diagnostics", None)
    n_blocks = getattr(fm.model, "n_blocks", None)
    if not callable(diagnostic_forward) or not isinstance(n_blocks, int) or n_blocks < 1:
        raise TypeError("model requires callable forward_with_diagnostics and positive n_blocks")
    # The frozen API supplies Tensor outputs; shape/finiteness are checked below.
    forward_diagnostics = cast(_DiagnosticForward, diagnostic_forward)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = np.asarray([_frame_fingerprint(f) for f in frames])
    for frame in frames:
        expected = np.arange(frame["mixture"].shape[-1] // 512 + 1) * 0.032
        times = np.asarray(frame["rps"].tindex.sample_times(), dtype=np.float64)
        if times.shape != expected.shape or not np.allclose(
            times - frame["mixture"].t_start, expected, atol=1e-9, rtol=0
        ):
            raise ValueError(
                "fixed labels must use physical 512/16000 timestamps; no benchmark stretching"
            )
    guesses: dict[str, tuple[np.ndarray | None, NDArray[np.bool_]]] = {
        name: value for name, value in diagnostic_guesses(gt).items()
    }
    if all("rps_cond" in f for f in frames):
        guesses["standard"] = (
            np.stack([np.asarray(f["rps_cond"].data) for f in frames]),
            np.ones(len(frames), bool),
        )
    guesses["locator"] = (None, np.ones(len(frames), bool))
    chosen = tuple(guesses) if cases is None else cases
    unknown = set(chosen) - guesses.keys()
    if unknown:
        raise ValueError(f"unknown/unavailable cases: {sorted(unknown)}")
    result: dict[str, Any] = {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "device": device,
        "smoke": smoke,
        "n_examples": len(frames),
        "n_groups": len(np.unique(groups)),
        "metadata": metadata,
        "group_ids": groups.tolist(),
        "fingerprints": fingerprints.tolist(),
        "observable": {
            "status": "certified"
            if observable is not None and observable.any()
            else "unestablished",
            "provenance": observable_provenance,
            "rotor_frames": int(np.sum(observable)) if observable is not None else 0,
            "reason": "Existing Frames do not retain per-source order powers/floors; mixture peaks cannot certify attribution.",
        },
        "selection": "parent-monitor selected checkpoint; diagnostics never select another epoch",
        "cases": {},
    }
    masks = {"all": np.ones_like(gt, bool), "stopped": gt <= 0.5, "below30": (gt > 0.5) & (gt < 30)}
    masks.update(_separation_masks(gt))
    for phase in PHASES[:-1]:
        masks[phase] = np.broadcast_to(
            np.array([clip_phase(t) == phase for t in gt])[:, None, None], gt.shape
        )
    for key in ("rig", "recording_id", "channel", "speech_present"):
        values = [str(meta.get(key, "unavailable")) for meta in metadata]
        for value in sorted(set(values) - {"unavailable"}):
            masks[f"{key}:{value}"] = np.broadcast_to(
                np.array([v == value for v in values])[:, None, None], gt.shape
            )
    for key, values in (sample_strata or {}).items():
        if np.shape(values) != (len(frames),):
            raise ValueError(f"stratum {key} needs one membership value per fixed example")
        masks[key] = np.broadcast_to(np.asarray(values, bool)[:, None, None], gt.shape)
    if observable is not None:
        masks["observable"] = np.asarray(observable, bool)
    offset_outputs = {}
    oracle_blocks = oracle_repeats = None
    for name in chosen:
        initial, applicable = guesses[name]
        chunks, repeat_chunks = [], []
        for start in range(0, len(frames), batch):
            stop = min(start + batch, len(frames))
            fr = frame_collate(frames[start:stop]).map_data(
                lambda a: torch.as_tensor(a).to(fm.device)
            )
            audio = fr["mixture"].data
            cond = (
                None
                if initial is None
                else torch.as_tensor(initial[start:stop], dtype=torch.float32, device=fm.device)
            )
            with torch.no_grad():
                pred, diagnostics = forward_diagnostics(audio, cond)
                trajectory = diagnostics["trajectories"]
                expected_shape = (stop - start, n_blocks + 1, 4, gt.shape[-1])
                if (
                    tuple(trajectory.shape) != expected_shape
                    or not torch.isfinite(trajectory).all()
                ):
                    raise ValueError(f"invalid diagnostic trajectories: {tuple(trajectory.shape)}")
                if not torch.equal(pred, trajectory[:, -1]):
                    raise ValueError("final prediction differs from final diagnostic block")
                chunks.append(trajectory.cpu().numpy())
                repeated = [trajectory[:, 0].cpu().numpy(), pred.cpu().numpy()]
                # Each public call resets hidden state, including calls 2 and 3.
                for _ in range(2):
                    pred, _ = forward_diagnostics(audio, pred)
                    if not torch.isfinite(pred).all():
                        raise ValueError("nonfinite repeated-operator output")
                    repeated.append(pred.cpu().numpy())
                repeat_chunks.append(np.stack(repeated, axis=1))
        trajectories = np.concatenate(chunks)
        repeats = np.concatenate(repeat_chunks)
        step_errors = [
            trajectory_errors(trajectories[:, s], gt) for s in range(trajectories.shape[1])
        ]
        metric_arrays = {
            key: np.stack([step[key] for step in step_errors], 1) for key in step_errors[0]
        }
        repeated_errors = np.stack(
            [trajectory_errors(repeats[:, s], gt)["ordered_mae"] for s in range(4)], 1
        )
        final, init = trajectories[:, -1], trajectories[:, 0]
        recovery = recovery_intervals(final, gt)
        initial_recovery = recovery_intervals(init, gt)
        aligned = recovery.pop("aligned_prediction")
        initial_recovery.pop("aligned_prediction")
        absolute = np.abs(final - gt)
        strata = {}
        for label, mask in masks.items():
            selected = mask & applicable[:, None, None]
            if selected.any():
                strata[label] = {
                    "rotor_frames": int(selected.sum()),
                    "ordered_mae": float(absolute[selected].mean()),
                    "mse_aligned_mae": float(np.abs(aligned - gt)[selected].mean()),
                }
            else:
                strata[label] = {"rotor_frames": 0, "status": "unestablished"}
        truth_active = gt > 0.5
        pred_active = aligned > 0.5
        activity = {}
        for label, denominator, errors in (
            ("false_active", ~truth_active, pred_active & ~truth_active),
            ("false_inactive", truth_active, ~pred_active & truth_active),
        ):
            denominator = denominator & applicable[:, None, None]
            activity[label] = {
                "count": int(denominator.sum()),
                "rate": float(errors[denominator].mean()) if denominator.any() else None,
            }
        selected = applicable
        bootstrap = (
            paired_group_bootstrap(
                metric_arrays["ordered_mae"][selected, 0],
                metric_arrays["ordered_mae"][selected, -1],
                groups[selected],
            )
            if selected.any()
            else {"status": "unestablished"}
        )
        all_four = (
            paired_group_bootstrap(
                recovery["success"][selected].astype(float),
                initial_recovery["success"][selected].astype(float),
                groups[selected],
            )
            if selected.any()
            else {"status": "unestablished"}
        )
        block_displacement = np.abs(trajectories - trajectories[:, :1])
        repeat_displacement = np.abs(repeats - repeats[:, :1])
        octave_error = (np.abs(aligned - 0.5 * gt) <= 1.0) | (np.abs(aligned - 2.0 * gt) <= 1.0)
        octave_denominator = (gt > 4.0) & applicable[:, None, None]
        result["cases"][name] = {
            "applicable_examples": int(selected.sum()),
            "strata": strata,
            "activity": activity,
            "blocks": {k: v.mean(0).tolist() for k, v in metric_arrays.items()},
            "repeated_operator_ordered_mae": repeated_errors.mean(0).tolist(),
            "full_set_block_displacement_mean": block_displacement.mean((0, 2, 3)).tolist(),
            "full_set_block_displacement_p95": np.percentile(
                block_displacement, 95, axis=(0, 2, 3)
            ).tolist(),
            "full_set_repeat_displacement_mean": repeat_displacement.mean((0, 2, 3)).tolist(),
            "full_set_repeat_displacement_p95": np.percentile(
                repeat_displacement, 95, axis=(0, 2, 3)
            ).tolist(),
            "octave_failure_fraction": float(octave_error[octave_denominator].mean())
            if octave_denominator.any()
            else None,
            "final_ordered_mae_median": float(np.median(metric_arrays["ordered_mae"][:, -1])),
            "final_ordered_mae_p90": float(np.percentile(metric_arrays["ordered_mae"][:, -1], 90)),
            "identity_improvement": bootstrap,
            "all_four_success_improvement": all_four,
            "all_four_success_rate": float(recovery["success"][selected].mean())
            if selected.any()
            else None,
        }
        np.savez_compressed(
            out_dir / f"{name}.npz",
            allow_pickle=False,
            trajectories=trajectories,
            repeats=repeats,
            target=gt,
            group_ids=groups,
            fingerprints=fingerprints,
            applicable=applicable,
            timestamps=np.stack([np.asarray(f["rps"].tindex.sample_times()) for f in frames]),
            observable=np.zeros_like(gt, bool) if observable is None else observable,
            repeated_ordered_mae=repeated_errors,
            **{f"mask_{k}": v for k, v in masks.items()},
            **metric_arrays,
            **{f"recovery_{k}": v for k, v in recovery.items()},
            **{f"initial_recovery_{k}": v for k, v in initial_recovery.items()},
        )
        if name == "oracle":
            oracle_blocks, oracle_repeats = trajectories, repeats
        if name.startswith("offset"):
            offset_outputs[name] = (init, final)
        # Incremental evidence survives a later diagnostic/device failure.
        (out_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    result["precision_gate"] = (
        precision_gates(oracle_blocks, offset_outputs, gt, observable)
        if oracle_blocks is not None
        else {"status": "unestablished", "reason": "oracle not evaluated"}
    )
    result["repeated_preservation_gate"] = (
        precision_gates(oracle_repeats, {}, gt, observable)
        if oracle_repeats is not None
        else {"status": "unestablished"}
    )
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def compare_jhtr(reference: Path, candidate: Path, *, out_path: Path) -> dict[str, Any]:
    """Compare fixed selected runs with paired 10k flight/recording bootstraps."""
    reference, candidate = Path(reference), Path(candidate)
    left = json.loads((reference / "results.json").read_text())
    right = json.loads((candidate / "results.json").read_text())
    if left["fingerprints"] != right["fingerprints"] or left["group_ids"] != right["group_ids"]:
        raise ValueError("comparison requires identical ordered Frames and original groups")
    results = {
        "reference": str(reference),
        "candidate": str(candidate),
        "seed_variability": "not measured by a single-seed trajectory bootstrap",
        "cases": {},
    }
    for name in sorted(left["cases"].keys() & right["cases"].keys()):
        with np.load(reference / f"{name}.npz") as a, np.load(candidate / f"{name}.npz") as b:
            if not np.array_equal(a["target"], b["target"]) or not np.array_equal(
                a["applicable"], b["applicable"]
            ):
                raise ValueError(f"{name}: targets or applicability differ")
            if name != "locator" and not np.array_equal(
                a["trajectories"][:, 0], b["trajectories"][:, 0]
            ):
                raise ValueError(f"{name}: conditional controls did not receive identical guesses")
            mask = a["applicable"]
            if not mask.any():
                results["cases"][name] = {"status": "unestablished"}
                continue
            row = {}
            for metric in ("ordered_mae", "pit_mae_mae_assignment", "pit_mae_mse_assignment"):
                before, after = a[metric][mask, -1], b[metric][mask, -1]
                stats = paired_group_bootstrap(before, after, a["group_ids"][mask])
                relative = float(1 - after.mean() / before.mean()) if before.mean() > 0 else None
                row[metric] = {
                    **stats,
                    "relative_reduction": relative,
                    "full_set_mechanism_metric_gate": relative is not None
                    and relative >= 0.10
                    and stats["ci95"] is not None
                    and stats["ci95"][0] > 0,
                }
            row["all_four_success"] = paired_group_bootstrap(
                b["recovery_success"][mask].astype(float),
                a["recovery_success"][mask].astype(float),
                a["group_ids"][mask],
            )
            row["strata"] = {}
            for key in sorted(k for k in a.files if k.startswith("mask_")):
                if key not in b or not np.array_equal(a[key], b[key]):
                    raise ValueError(f"{name}: {key} differs")
                membership = a[key] & mask[:, None, None]
                counts = membership.sum((1, 2))
                covered = counts > 0
                if not covered.any():
                    row["strata"][key.removeprefix("mask_")] = {"status": "unestablished"}
                    continue
                before = (np.abs(a["trajectories"][:, -1] - a["target"]) * membership).sum((1, 2))[
                    covered
                ] / counts[covered]
                after = (np.abs(b["trajectories"][:, -1] - b["target"]) * membership).sum((1, 2))[
                    covered
                ] / counts[covered]
                stats = paired_group_bootstrap(before, after, a["group_ids"][covered])
                row["strata"][key.removeprefix("mask_")] = {
                    **stats,
                    "metric": "ordered MAE, each covered example weighted equally",
                    "reference": float(before.mean()),
                    "candidate": float(after.mean()),
                    "degradation_limit": max(0.1, 0.05 * float(before.mean())),
                    "degradation_guard_pass": bool(
                        after.mean() - before.mean() <= max(0.1, 0.05 * before.mean())
                    ),
                }
            results["cases"][name] = row
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def campaign_config(experiment: str, *, config_path: Path | None = None) -> Any:
    """Load an as-run resolved config, or compose a new experiment recipe."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from training.config import register_configs

    if config_path is not None:
        return OmegaConf.load(config_path)
    register_configs()
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
        return compose(config_name="config", overrides=[f"experiment={experiment}"])


def load_campaign_model(cfg: Any, checkpoint: str, *, device: str) -> Any:
    """Use the trainer's model/codec and checkpoint resolver, no zoo publication."""
    from training.config import build_task_and_codec, instantiate_model
    from utils.checkpoints import resolve_checkpoint_uri
    from zoo.frame_model import FrameModel

    task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model)
    state = torch.load(resolve_checkpoint_uri(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return FrameModel(
        model.to(device).eval(), codec, task, experiment=str(cfg.experiment_name), device=device
    )


def _config_mapping(value: object, path: str) -> dict[str, object]:
    """Narrow resolved config nodes without accepting sequences/scalars as recipes."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be a string-keyed configuration mapping")
    # OmegaConf's container union does not encode its key type after this check.
    return cast(dict[str, object], value)


def fixed_dataset_size(dataset: Dataset) -> int:
    """Reject streams and unsized datasets before finite indexed benchmark reads."""
    if isinstance(dataset, IterableDataset) or not isinstance(dataset, Sized):
        raise TypeError("campaign validation requires a sized, map-style fixed dataset")
    return len(dataset)


def check_recipe_parity(
    parent: Any, child: Any, *, conditional_bridge: bool = False
) -> dict[str, Any]:
    """Strict resolved-tree comparison with narrowly enumerated task allowances."""
    from omegaconf import OmegaConf

    before = _config_mapping(
        OmegaConf.to_container(parent, resolve=True, throw_on_missing=True), "parent"
    )
    after = _config_mapping(
        OmegaConf.to_container(child, resolve=True, throw_on_missing=True), "child"
    )
    allowed = []
    for tree in (before, after):
        tree.pop("experiment_name", None)
        # Architecture can change; task, rates, rotor count and conditioning cannot.
        model = _config_mapping(tree["model"], "model")
        task_model = {k: model[k] for k in ("task", "task_params") if k in model}
        task_params = _config_mapping(task_model.setdefault("task_params", {}), "task_params")
        task_params.setdefault("use_cond", False)
        tree["model"] = task_model
        params = _config_mapping(model.get("params", {}), "model.params")
        task_model["io"] = {
            k: params.get(k, default)
            for k, default in (("num_rotors", 4), ("hop_length", 512), ("sample_rate", 16000))
        }
    if conditional_bridge:
        child_model = _config_mapping(after["model"], "child.model")
        child_task = _config_mapping(child_model["task_params"], "child.model.task_params")
        if child_task.get("use_cond") is not True:
            raise ValueError("conditional bridge must enable use_cond")
        expected_loss = OmegaConf.to_container(
            OmegaConf.load(Path(__file__).resolve().parents[2] / "conf/loss/mse_cond.yaml"),
            resolve=True,
        )
        if after["loss"] != expected_loss:
            raise ValueError("conditional bridge must use the unchanged mse_cond loss")
        for split, seed in (("train", 20260729), ("valid", 777)):
            data = _config_mapping(after["data"], "child.data")
            dataset = _config_mapping(data[split], f"child.data.{split}")
            params = _config_mapping(dataset["params"], f"child.data.{split}.params")
            if params.get("rps_corruption") != {"seed": seed}:
                raise ValueError(f"{split} corruption must have only the established seed {seed}")
            params.pop("rps_corruption")
        parent_model = _config_mapping(before["model"], "parent.model")
        parent_task = _config_mapping(parent_model["task_params"], "parent.model.task_params")
        child_task["use_cond"] = parent_task.get("use_cond", False)
        after["loss"] = before["loss"]
        allowed = [
            "use_cond=true",
            "loss=mse_cond",
            "train.rps_corruption.seed=20260729",
            "valid.rps_corruption.seed=777",
        ]

    def differences(a: Any, b: Any, prefix: str = "") -> list[dict[str, Any]]:
        if isinstance(a, dict) and isinstance(b, dict):
            result = []
            for key in sorted(a.keys() | b.keys()):
                path = f"{prefix}.{key}" if prefix else key
                if key not in a or key not in b:
                    result.append({"path": path, "parent": a.get(key), "child": b.get(key)})
                else:
                    result.extend(differences(a[key], b[key], path))
            return result
        return [] if a == b else [{"path": prefix, "parent": a, "child": b}]

    diff = differences(before, after)
    return {
        "pass": not diff,
        "allowed_task_bridge": allowed,
        "differences": diff,
        "note": "Composed current recipes; historical comparisons require their saved as-run configs.",
    }


def check_fixed_frame_parity(cfg: Any) -> dict[str, Any]:
    """Exercise all 256/paired512 samples: byte/time/meta/id and target-row parity.

    The baseline and conditional data are independently instantiated. A second
    conditional instantiation checks reproducibility beyond cached __getitem__.
    No waveform RNG, stream, corruption or label logic is reimplemented here.
    """
    from itertools import permutations

    from omegaconf import OmegaConf

    from data_processing.frames import meta_dict
    from training.config import build_dataset

    spec = _config_mapping(OmegaConf.to_container(cfg.data.valid, resolve=True), "data.valid")
    params = _config_mapping(spec["params"], "data.valid.params")
    expected_n = params["n"]
    target = spec["_target_"]
    if not isinstance(expected_n, int) or isinstance(expected_n, bool):
        raise TypeError("fixed validation n must be an integer")
    if not isinstance(target, str):
        raise TypeError("fixed validation _target_ must be a class path")
    paired = target.endswith("SpeechPairedSynthValidDataset")
    if expected_n != (512 if paired else 256):
        raise ValueError("full fixed validation requires 256 samples, or 512 speech-paired samples")
    base_spec = OmegaConf.create(spec)
    base_spec.params.rps_corruption = None
    cond_spec = OmegaConf.create(spec)
    cond_spec.params.rps_corruption = {"seed": 777}
    baseline = build_dataset(base_spec)
    conditional = build_dataset(cond_spec)
    repeated = build_dataset(OmegaConf.create(OmegaConf.to_container(cond_spec, resolve=True)))
    sizes = [fixed_dataset_size(ds) for ds in (baseline, conditional, repeated)]
    if not all(size == expected_n for size in sizes):
        raise AssertionError("validation adapter changed the sample count")
    fingerprints = []
    for i in range(expected_n):
        left, right, again = baseline[i], conditional[i], repeated[i]
        if "rps_cond" in left or "rps_cond" not in right:
            raise AssertionError(f"sample {i}: incorrect conditioning presence")
        if set(right) != set(left) | {"rps_cond"}:
            raise AssertionError(f"sample {i}: adapter changed frame entries")
        if _json_value(meta_dict(left)) != _json_value(meta_dict(right)):
            raise AssertionError(f"sample {i}: metadata/id/order changed")
        for key in left:
            if key == "meta":
                continue
            a, b = left[key], right[key]
            if a.dims != b.dims or not a.tindex.equal(b.tindex):
                raise AssertionError(f"sample {i}/{key}: dimensions/timestamps changed")
            aa, bb = np.asarray(a.data), np.asarray(b.data)
            if aa.dtype != bb.dtype or aa.shape != bb.shape:
                raise AssertionError(f"sample {i}/{key}: dtype/shape changed")
            if key == "rps":
                equal = any(np.array_equal(aa[list(p)], bb) for p in permutations(range(4)))
            else:
                equal = aa.tobytes() == bb.tobytes()
            if not equal:
                raise AssertionError(
                    f"sample {i}/{key}: bytes changed beyond an allowed whole-row permutation"
                )
        if _frame_fingerprint(right) != _frame_fingerprint(again):
            raise AssertionError(f"sample {i}: conditional regeneration is not deterministic")
        if _frame_fingerprint(right) != _frame_fingerprint(conditional[i]):
            raise AssertionError(f"sample {i}: repeated access is not deterministic")
        fingerprints.append(_frame_fingerprint(right))
    if paired:
        for i in range(expected_n // 2):
            a, b = conditional[i], conditional[i + expected_n // 2]
            for key in ("rps", "rps_cond"):
                if not np.array_equal(a[key].data, b[key].data) or not a[key].tindex.equal(
                    b[key].tindex
                ):
                    raise AssertionError(f"speech pair {i}: different {key} alignment/guesses")
            for key in ("sample_id", "channel"):
                if meta_dict(a).get(key) != meta_dict(b).get(key):
                    raise AssertionError(f"speech pair {i}: different {key}")
    return {
        "pass": True,
        "n": expected_n,
        "speech_paired": paired,
        "fingerprints": fingerprints,
        "proof": "all non-target series bytes, all timestamps/dims/meta/ids, row-only targets, deterministic guesses",
    }


def profile_jhtr(cfg: Any, *, device: str = "cpu", smoke: bool = False) -> dict[str, Any]:
    """Run the actual model forward/backward; no optimizer/schedule/data changes.

    Full profiling uses the inherited batch at 1/4 s training and 8 s validation,
    as the trainer does. CPU smoke uses one 1 s example and is not a matched profile.
    """
    from training.config import instantiate_model

    torch.manual_seed(int(cfg.seed))
    model = instantiate_model(cfg.model).to(device)
    cuda = torch.device(device).type == "cuda"
    if not smoke and not cuda:
        raise ValueError("matched full profile requires a GPU; use --smoke for bounded CPU proof")
    rows = []
    for seconds in (1,) if smoke else (1, 4, 8):
        training = seconds != 8
        batch = 1 if smoke else int(cfg.data.batch_size or cfg.batch_size)
        samples = seconds * 16000
        # Known finite nonzero input, not a new dataset or training regimen.
        audio = torch.randn(batch, samples, device=device) * 0.1
        cond = torch.full((batch, 4, samples // 512 + 1), 80.0, device=device)
        model.train(training)
        model.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=torch.device(device).type,
                enabled=cuda and bool(cfg.amp),
                dtype=torch.bfloat16
                if getattr(cfg, "amp_dtype", "float16") == "bfloat16"
                else torch.float16,
            ),
        ):
            prediction = model(
                audio, cond if bool(cfg.model.task_params.get("use_cond", False)) else None
            )
            objective = prediction.square().mean()
        if cuda:
            torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - start
        if prediction.shape != cond.shape or not torch.isfinite(prediction).all():
            raise AssertionError("model smoke/profile produced invalid output")
        start = time.perf_counter()
        if training:
            objective.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise AssertionError("model smoke/profile produced absent or nonfinite gradients")
        if cuda:
            torch.cuda.synchronize()
        rows.append(
            {
                "seconds": seconds,
                "batch": batch,
                "training": training,
                "frames": prediction.shape[-1],
                "forward_seconds": forward_seconds,
                "backward_seconds": time.perf_counter() - start if training else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
            }
        )
        del audio, cond, prediction, objective
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise AssertionError("nonfinite model parameters")
    return {
        "device": device,
        "hardware": torch.cuda.get_device_name() if cuda else "CPU",
        "parameters": sum(p.numel() for p in model.parameters()),
        "smoke": smoke,
        "amp": cuda and bool(cfg.amp),
        "timing": "single cold forward/backward; not steady-state latency",
        "profiles": rows,
    }
