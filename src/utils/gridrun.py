"""Restartable parallel unit-JSON harness (the pattern ~13 deleted scripts copied).

One *unit* of work = one JSON file under ``<out_dir>/raw/<uid>.json``. The
harness runs the units on a process pool, skips the units whose JSON already
exists (resume), turns a unit exception into a ``<uid>.err`` traceback file
without killing the pool, then aggregates all unit JSONs into
``<out_dir>/summary.json`` and prints it. The live exemplars of the pattern
are ``scripts/sr_dp_probe.py`` and ``scripts/jb_probe.py``.

Usage::

    from utils.gridrun import Unit, run_grid

    def worker(unit: Unit) -> dict:
        ...  # heavy imports INSIDE the worker (numpy/torch)
        return {"window": unit.params["window"], "mae": ...}

    units = [Unit(uid=f"cost__{w}", params={"window": w}) for w in WINDOWS]
    result = run_grid(units, worker, "results/my_probe", jobs=8)
    raise SystemExit(result.exit_code)

Layering: ``utils`` is the bottom layer — this module is stdlib-only at
module level (no numpy/torch). Workers do their own heavy imports.

Pickling note: ``worker`` must be a module-level function (the pool pickles
it by reference). The default start method is the platform default (fork on
Linux); pass ``mp_context="spawn"`` for torch/CUDA workers.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "GridResult",
    "Unit",
    "add_gridrun_args",
    "gridrun_from_args",
    "run_grid",
    "unit_path",
]

#: The BLAS pools capped in the parent BEFORE workers start: process-level
#: parallelism instead of thread-level (the convention every hand-copied
#: harness used). ``setdefault`` — an explicit user environment wins.
_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class Unit:
    """One restartable unit: a filesystem-safe id + a JSON-serializable payload."""

    uid: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridResult:
    """Outcome of one :func:`run_grid` call."""

    out_dir: Path
    n_units: int
    n_ok: int
    n_skipped: int
    n_failed: int
    summary: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return 1 if self.n_failed else 0


def _safe_uid(uid: str) -> str:
    """Filesystem-safe unit id (the exemplars' ``':' -> '_'`` convention)."""
    return uid.replace(":", "_").replace("/", "_").replace(" ", "_")


def unit_path(out_dir: str | Path, uid: str) -> Path:
    """The JSON path of one unit: ``<out_dir>/raw/<uid>.json``."""
    return Path(out_dir) / "raw" / f"{_safe_uid(uid)}.json"


def _write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write (tmp + ``os.replace``), parent dirs created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _run_unit(worker: Callable[[Unit], dict], out_dir: str, unit: Unit) -> tuple[str, str]:
    """Run one unit; a unit exception becomes a ``.err`` file, never a pool kill."""
    out = unit_path(out_dir, unit.uid)
    err = out.with_suffix(".err")
    try:
        payload = worker(unit)
        _write_json(out, payload)
        if err.exists():
            err.unlink()
        return unit.uid, "ok"
    except Exception:  # noqa: BLE001 — one bad unit must not kill the grid
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        return unit.uid, "ERROR"


def run_grid(
    units: Sequence[Unit],
    worker: Callable[[Unit], dict],
    out_dir: str | Path,
    *,
    jobs: int = 4,
    resume: bool = True,
    blas_threads: int | None = 1,
    summarize: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    summary_name: str = "summary.json",
    mp_context: str | None = None,
    verbose: bool = True,
) -> GridResult:
    """Run ``worker`` over ``units``, one JSON per unit, restartable.

    - ``resume=True`` skips every unit whose JSON already exists (a stale
      ``.err`` without a JSON is retried); ``resume=False`` re-runs all.
    - ``jobs <= 1`` runs serially in-process (debuggable stack traces).
    - ``blas_threads`` caps the BLAS thread pools through the environment
      before workers start (``None`` leaves the environment alone).
    - ``summarize(rows)`` maps ALL unit payloads (sorted by filename, so
      completed units from earlier runs are included) to the summary dict
      written to ``<out_dir>/<summary_name>`` and printed.
    """
    out_dir = Path(out_dir)
    if blas_threads is not None:
        for var in _BLAS_VARS:
            os.environ.setdefault(var, str(blas_threads))

    seen: set[str] = set()
    for u in units:
        if _safe_uid(u.uid) in seen:
            raise ValueError(f"duplicate unit uid {u.uid!r}")
        seen.add(_safe_uid(u.uid))

    n_skipped = 0
    todo: list[Unit] = []
    for u in units:
        if resume and unit_path(out_dir, u.uid).exists():
            n_skipped += 1
            if verbose:
                print((u.uid, "skip"), flush=True)
        else:
            todo.append(u)

    statuses: dict[str, str] = {}
    if jobs <= 1:
        for u in todo:
            uid, status = _run_unit(worker, str(out_dir), u)
            statuses[uid] = status
            if verbose:
                print((uid, status), flush=True)
    elif todo:
        ctx = multiprocessing.get_context(mp_context) if mp_context else None
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            futs = [pool.submit(_run_unit, worker, str(out_dir), u) for u in todo]
            for f in as_completed(futs):
                uid, status = f.result()
                statuses[uid] = status
                if verbose:
                    print((uid, status), flush=True)

    n_ok = sum(1 for s in statuses.values() if s == "ok")
    n_failed = sum(1 for s in statuses.values() if s == "ERROR")

    raw = out_dir / "raw"
    rows = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))] if raw.is_dir() else []
    summary = summarize(rows) if summarize is not None else {"n_units": len(rows)}
    _write_json(out_dir / summary_name, summary)
    if verbose:
        print(json.dumps(summary, indent=1), flush=True)
        if n_failed:
            print(f"!! {n_failed} unit(s) failed — see {out_dir}/raw/*.err", flush=True)
    return GridResult(
        out_dir=out_dir,
        n_units=len(units),
        n_ok=n_ok,
        n_skipped=n_skipped,
        n_failed=n_failed,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# argparse mixin


def add_gridrun_args(parser: Any, *, jobs: int = 4) -> None:
    """Add the standard harness flags (``--jobs``, ``--no-resume``) to a parser."""
    parser.add_argument("--jobs", type=int, default=jobs, help="parallel worker processes")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="re-run every unit (default: skip units whose JSON already exists)",
    )


def gridrun_from_args(
    args: Any,
    units: Sequence[Unit],
    worker: Callable[[Unit], dict],
    out_dir: str | Path,
    **kwargs: Any,
) -> GridResult:
    """:func:`run_grid` driven by :func:`add_gridrun_args` flags."""
    return run_grid(
        units,
        worker,
        out_dir,
        jobs=int(args.jobs),
        resume=not getattr(args, "no_resume", False),
        **kwargs,
    )
