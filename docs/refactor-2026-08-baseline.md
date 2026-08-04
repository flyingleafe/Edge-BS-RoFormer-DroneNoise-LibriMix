# Refactor 2026-08 — Phase 0 baseline

Date: 2026-08-04. Recorded after the Phase 0 guardrails landed
(`docs/refactor-2026-08-plan.md` § 4, Phase 0). Compare against these numbers
after each subsequent phase.

## Pytest

Default run (`uv run pytest -q`, addopts `-m "not slow and not network"`):

| Metric | Count |
|---|---|
| Collected (full, `-m ""`) | 746 |
| Passed | 726 |
| Skipped | 3 |
| Deselected by markers | 17 |
| Failed | 0 |

Deselected breakdown (module-level marks):

- `network` (1): `tests/training/test_artifacts_r2_integration.py` (live R2;
  previously self-enabled whenever `.env` held `R2_ACCOUNT_ID` — now opt-in
  via `pytest -m network`)
- `slow` (16): `tests/training/test_loop.py` (7),
  `tests/data_processing/test_generated_noise.py` (6),
  `tests/tasks/test_rps_regression.py` (3 — also keeps its golden-artifact
  skipif)

## import-linter

`uv run lint-imports` — 10 root packages (`utils`, `data_processing`,
`models`, `tasks`, `losses`, `metrics`, `plots`, `training`, `experiments`,
`fwh_rotor_sim`), 169 files, 317 dependencies analyzed. Contracts, all KEPT:

1. `nothing imports experiments` — fixed by relocating the egonoise-GP
   inference core to `data_processing/egonoise_gp.py` (was 2 edges from
   `data_processing.gp_noise`).
2. `models must not import training` — fixed by relocating
   `resolve_checkpoint_uri`/`load_r2_env` to `src/utils/checkpoints.py`
   (was 1 edge from `models.htdemucs_ft`; `training.artifacts` re-exports).

Deliberately NOT added (fails today): "utils is a leaf" —
`utils.get_model_from_config` lazily imports `models.*` (the legacy ZFTurbo
chain). Revisit in phase 3 when the model registry is dict-ified.

Known cross-package edges that remain (phase 2 scope, no contract yet):
`data_processing -> training` (2), `data_processing -> tasks` (2),
`tasks <-> losses`, `data_processing <-> training` cycles — see plan § 3.5.

Note: import-linter caches its graph (grimp cache). If a contract result
looks stale after a file move, re-run with `lint-imports --no-cache`.

## Pre-commit

`lint-imports` wired as a flake-generated pre-commit hook (`flake.nix`,
whole-graph, fires on any `*.py` change) next to the existing
`validate-experiment-docs` hook (already wired before Phase 0).
