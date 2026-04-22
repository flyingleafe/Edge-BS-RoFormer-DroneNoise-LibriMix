# tests/ — Postdoc System Tests

Pytest test suite for the postdoc experiment orchestration system. Does not cover model code — focuses on CLI, config, job management, and local backend.

## Why this directory exists

Postdoc is infrastructure that must work correctly before any experiment can run. These tests prevent regressions in job submission, scheduling, tracking, and config resolution.

## Test Files

| File | Tests |
|------|-------|
| `conftest.py` | Shared pytest fixtures |
| `test_cli.py` | CLI commands (submit, list, status, etc.) |
| `test_config.py` | PostdocConfig loading from `postdoc.yaml` |
| `test_context.py` | Backend wiring (scheduler, storage, tracker) |
| `test_experiment.py` | Experiment YAML loading and config resolution |
| `test_run_job.py` | Job runner (train → eval subprocess orchestration) |
| `test_local_scheduler.py` | LocalScheduler GPU allocation and job lifecycle |
| `test_local_storage.py` | LocalStorage artifact persistence |
| `test_tracker.py` | JobTracker (SQLite-backed state and metrics) |
| `test_integration.py` | End-to-end integration tests |

## Running

```bash
pytest tests/
```

## Gotchas

- Tests require a `postdoc.yaml` in the project root
- Integration tests may need GPU access for full coverage
- Tests do NOT cover model training — that's validated via manual runs and W&B