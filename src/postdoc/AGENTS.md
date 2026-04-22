# src/postdoc/ — Experiment Orchestration

CLI tool for submitting, scheduling, monitoring, and analyzing ML experiments. Installed via `pyproject.toml` entry point (`postdoc = "postdoc.cli:app"`).

## Why this directory exists

Manages the full experiment lifecycle: from YAML definition through GPU-scheduled training and evaluation to result tracking. Replaces ad-hoc training commands with reproducible, tracked experiments.

## Architecture

```
postdoc CLI (Typer) → cli.py
    ├─ job submit   → JobTracker (SQLite) → Scheduler (LocalScheduler)
    │                                        │ GPU alloc + subprocess
    │                                   run_job.py
    │                                  ┌─────┴─────┐
    │                               train.py   final_valid.py
    ├─ job list/status/logs/resume/cancel
    └─ results show/compare
```

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Typer CLI commands |
| `config.py` | `PostdocConfig` dataclass from `postdoc.yaml` |
| `experiment.py` | Experiment YAML loading and config resolution |
| `run_job.py` | Job runner: train → eval subprocess orchestration |
| `context.py` | Wires storage/scheduler/tracker from backend config |
| `interfaces/` | ABCs: `Scheduler`, `StorageBackend`, `JobTracker` |
| `backends/local/` | Local GPU scheduler, disk storage |
| `backends/cloud/` | Stub — `NotImplementedError` |

## Job Lifecycle

```
DEFINED → SUBMITTED → TRAINING → EVAL → DONE
                ↓          ↓
              QUEUED    FAILED
```

- Jobs queue on `NoCapacityError`; `drain_queue()` picks up when GPUs free
- Failed jobs: `postdoc job resume <id>` resumes from best checkpoint
- Error classification: OOM, NaN, DataLoading, CUDA, Unknown

## CLI Commands

```bash
postdoc job submit <experiment.yaml>   # Submit experiment(s)
postdoc job list [--state <state>]     # List jobs
postdoc job status <job_id>            # Status + metrics
postdoc job logs <job_id> [--tail]     # View logs
postdoc job resume <job_id>            # Resume from best checkpoint
postdoc job cancel <job_id>            # Cancel running job
postdoc results show <job_id>          # Show results
postdoc results compare <job_id...>    # Compare multiple jobs
```

## Gotchas

- Cloud backend is not implemented — local only
- Job state is persisted in SQLite (`postdoc_jobs.db`)
- `postdoc.yaml` must exist in project root with `backend: local` and GPU count