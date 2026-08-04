# src/utils/ — `utils` package

This directory holds the `utils` Python package. It is editable-installed via
`pyproject.toml` (`[tool.hatch.build.targets.wheel] packages = [..., "src/utils"]`).

`__init__.py` contains the legacy "ZFTurbo"-style helpers (config loader,
audio I/O, `demix`/TTA inference, LoRA and weight-loading utilities) that used
to live in `utils.py` at the project root. The model factory
(`get_model_from_config`/`build_model_from_config`) moved to
`src/models/registry.py::LEGACY_MODEL_BUILDERS`, so `utils` is now a **leaf
package** — it imports no other project package.

## Why a package, not a module

This used to be a sub-package (`utils.data`, the in-repo timeseries algebra)
before it was replaced by the PyPI `tdseries` package and deleted (see
docs/refactor-unified-framework.md). Converting `utils.py` → `utils/__init__.py`
preserved every legacy import while opening the namespace for future sub-packages.

## Files

| Path | Purpose | Details |
|------|---------|---------|
| `dataloader_benchmark.py` | Reusable `benchmark_dataloader(...)` helper for finite and infinite PyTorch loaders; reports batch/example/effective-audio-clip throughput. | Use before optimizing online-mixing dataloaders. |
| `paths.py` | The **data** root, which is not the code root: `get_data_root()` / `get_data_path("DREGON")` / `get_datasets_path` / `get_results_path`. `$DATA_ROOT` → the main checkout (first line of `git worktree list`) → the git toplevel. | `data/`, `datasets/`, `results/` and `omnirun-outputs/` are gitignored, so a worktree does not have them — code must come from `Path(__file__)` (this checkout), data from here. Getting this backwards means a worktree silently runs the other branch's code, or finds no data at all. |
| `gridrun.py` | The restartable parallel unit-JSON harness (`Unit`, `run_grid`, `add_gridrun_args`/`gridrun_from_args`): one JSON per unit under `<out>/raw/`, skip-if-exists resume, per-unit `.err` on failure, `summary.json`. Stdlib-only (utils is the bottom layer). | Replaces the ProcessPoolExecutor boilerplate ~13 scripts hand-copied; live exemplars of the pattern: `scripts/sr_dp_probe.py`, `scripts/jb_probe.py`. Used by `scripts/rps_eval.py`. |

## Adding new sub-packages

Create a new subdirectory with its own `__init__.py` and `AGENTS.md`. Do not
add unrelated helpers to `__init__.py` — they belong in a sub-module so the
top-level namespace stays comprehensible.
