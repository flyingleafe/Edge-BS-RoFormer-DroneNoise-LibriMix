# Job Layer Design — `postdoc` Experiment Platform

**Date:** 2026-03-21
**Status:** Reviewed (2 iterations) — awaiting user approval
**Scope:** Job layer — experiment definition, submission, lifecycle management, results storage, failure handling, and guardrails.

## Context

This project is the foundation of a PhD research platform for low-SNR harmonic noise suppression in speech. The researcher currently runs experiments manually via wrapper scripts on a rented vast.ai box (~$150/month constant). The job layer is the first sub-project of a larger platform (Data, Job, Literature, Reporting, Ideas layers) because compute cost and experiment scheduling are the top pain points.

### Research Background

The core research idea is self-supervised learning for speech enhancement in the presence of harmonic noise from rotating machinery (drones, engines, propellers). Models predict both denoised speech and motor speeds (RPM). A small labeled subset (DREGON) provides ground-truth RPM for supervised training; a much larger unlabeled pool uses predicted RPM with regularization losses. This approach aims to outperform prior work by combining explicit noise modeling (via RPM) with scale (via abundant unlabeled data).

### Goals

- **Eliminate idle compute spend** — GPUs are provisioned only while jobs run, then torn down.
- **Elastic scaling** — run on the cheapest available cloud (GCP initially, vast.ai via direct API as second iteration) transparently.
- **Reproducibility** — every job is pinned to a git commit; results are traceable.
- **Automated failure handling** — triage, local repro, debug, one retry; escalate on second failure.
- **Agent-friendly** — CLI interface usable by both humans and Claude Code agents orchestrated via zeroclaw.

### Non-Goals (for this sub-project)

- Dataset management (Data layer — next sub-project)
- Literature search and RAG (Literature layer)
- Automated reporting and paper drafting (Reporting layer)
- Idea management and experiment suggestion (Ideas layer)
- Cost estimator agent (will be elaborated separately; for now, cost estimation is basic)

---

## 1. Experiment Definition

Each experiment is a single YAML file in `experiments/`. It composes three concerns: ML configuration, infrastructure requirements, and run lifecycle.

```yaml
# experiments/ssl-dcunet-rpm-v1.yaml

# --- ML config (what to train) ---
model:
  base_config: configs/5_Baseline_dcunet.yaml  # reuses existing model configs
  overrides:                                     # per-experiment parameter tweaks
    training.lr: 3.0e-4
    training.batch_size: 4

dataset:
  name: dregon-librimix-v2        # references a DVC-tracked dataset
  split_strategy: rpm_aware       # [PLANNED] ensures RPM-labeled data appears in every batch

loss:                              # [PLANNED] — requires train.py modifications
  primary: si_sdr
  auxiliary:
    - type: rpm_prediction
      weight: 0.1
      supervised_only: false      # uses regularization loss when no ground truth
    - type: rpm_smoothness
      weight: 0.05

# --- Infra config (where/how to run) ---
resources:
  gpu: A10G
  gpu_count: 1
  disk_size: 100
  cloud: gcp                      # GCP via SkyPilot initially; vast.ai added later via direct API

# --- Run config (lifecycle) ---
run:
  max_duration: 6h
  checkpoint_interval: 30m
  estimated_cost: null            # filled by cost estimator before submission

wandb:
  project: harmonic-noise-suppression
  tags: [ssl, rpm, dcunet]
```

### Design Decisions

- **`base_config` reuses existing configs.** The 21 YAML configs already in `configs/` are not rewritten. Experiment YAMLs reference them and apply overrides. This avoids config duplication and keeps existing training scripts compatible.
- **`overrides` uses dot-notation paths** into the base config, allowing surgical per-experiment tweaks without forking config files.
- **`resources.cloud`** specifies the target cloud. Initially GCP only (via SkyPilot). vast.ai will be added as a second iteration via a direct API wrapper (see Section 4).
- **Experiment YAMLs are version-controlled.** They live in `experiments/` and are diffable, reviewable, and searchable.
- **The experiment design agent generates and modifies these files.** The human or agent workflow is: create/edit YAML, then submit.
- **`dataset` and `loss` sections marked [PLANNED]** are aspirational schema — they document the target format but require `train.py` modifications to be functional. Initial implementation maps only to existing `train.py` CLI arguments (see Section 4, Config Translation).

---

## 2. Job Lifecycle

Every job has two execution stages: **TRAINING** and **EVAL**. Each can succeed or fail independently.

### State Machine

```
DEFINED → SUBMITTED → PROVISIONING → TRAINING → EVAL → COMPLETING → DONE
                                                     |          |
                                                   FAILED    FAILED
                                                     |          |
                                                LOCAL_REPRO  LOCAL_REPRO
                                                     |          |
                                                  FIXING     FIXING
                                                     |          |
                                                RESUBMITTED  RESUBMITTED
                                                     |          |
                                                  TRAINING     EVAL
                                                     |          |
                                                (2nd fail)  (2nd fail)
                                                     |          |
                                                 ESCALATED  ESCALATED
```

### State Descriptions

| State | Description |
|-------|-------------|
| **DEFINED** | Experiment YAML exists in `experiments/`. Not yet submitted. |
| **SUBMITTED** | `postdoc job submit` has estimated cost (warning if over soft ceiling), translated the YAML into a SkyPilot task, and launched it. Code auto-committed to an experiment branch. Cost estimation is inline during submission, not a separate state. |
| **PROVISIONING** | SkyPilot is finding and setting up the instance: clone repo at pinned commit, install deps, pull dataset from R2. |
| **TRAINING** | `train.py` is running. Checkpoints sync to R2 at `checkpoint_interval`. Metrics stream to W&B. |
| **EVAL** | Training completed. `final_valid.py` runs on the trained model. |
| **COMPLETING** | Eval finished. Agent extracts key metrics into structured results store on R2. Logs archived. Instance torn down. If metric extraction fails, job still transitions to DONE but with `meta.json` flagged `metrics_incomplete: true`. |
| **DONE** | Terminal success state. All artifacts on R2, metrics in local DB. |
| **FAILED** | Training or eval crashed. Logs saved to R2. Instance torn down. Enters failure handling flow. |
| **LOCAL_REPRO** | Agent checks out the experiment branch and attempts to reproduce the failure locally (CPU, tiny data slice, few steps). For eval failures, runs eval on a single sample. |
| **FIXING** | Agent has reproduced the failure. Uses systematic-debugging skill to diagnose and fix. Commits fix to the experiment branch. |
| **RESUBMITTED** | Fix committed. Job resubmitted from last safe checkpoint (training) or from the trained model (eval-only rerun). One attempt only. |
| **ESCALATED** | Second failure on the same issue. Human is notified with: error category, logs, what the agent tried, and the agent's diagnosis. |

### Instance Teardown

Automatic after COMPLETING, FAILED (once logs are saved), or ESCALATED. No idle instances ever.

---

## 3. Code Pinning & Experiment Branches

Every submitted job is pinned to an exact git commit for reproducibility.

### Submission Flow

1. `postdoc job submit experiments/foo.yaml` is invoked.
2. If there are uncommitted changes, they are **auto-committed to an experiment branch** named `exp/<experiment-name>` (e.g., `exp/ssl-dcunet-rpm-v1`). Main is never touched.
3. The commit hash is recorded in the job metadata.
4. The remote job clones the repo at that exact commit.

### Branch Conflict Prevention

If an experiment branch already exists and has a running job, `postdoc job submit` refuses to submit and asks for a different experiment name. This prevents overwriting code that a running job depends on.

### Implications

- **Reproducibility** — you always know what code produced which results.
- **Safe local iteration** — you can keep working locally while jobs run on pinned code.
- **Agent fixes land on experiment branches** — failure handling commits fixes to the same experiment branch, never main. Merging successful experiment branches back to main is a deliberate human action.

---

## 4. SkyPilot Integration

### Cloud Strategy

**Phase 1 (initial):** GCP only via SkyPilot. SkyPilot has first-class GCP support with reliable provisioning, auto-teardown, and spot instance recovery.

**Phase 2 (future iteration):** vast.ai via a direct API wrapper in `src/postdoc/vast.py`. SkyPilot's vast.ai integration is community-contributed and not production-ready — it lacks reliable auto-teardown, spot recovery, and lifecycle management. Rather than building on an unstable foundation, vast.ai will be integrated directly using its REST API with the same `postdoc` interface. The `resources.cloud` field in experiment YAMLs will then accept `[gcp, vast]` as a preference list.

### Task Compilation

`postdoc job submit` compiles the experiment YAML into a SkyPilot task definition:

- `resources` section maps directly to SkyPilot resource spec.
- A **standard bootstrap script** is generated with fail-fast semantics:

```bash
#!/bin/bash
set -euo pipefail

# 1. Clone repo at pinned commit
git clone <repo-url> --branch <exp-branch> --single-branch
cd harmonic-noise-suppression
git checkout <pinned-commit>

# 2. Install dependencies
uv sync

# 3. Pull dataset from R2
# (DVC pull or direct R2 download — depends on data layer implementation)
postdoc data pull <dataset-name>

# 4. Start periodic checkpoint sync (background)
postdoc checkpoint sync-daemon <job-id> --interval <checkpoint_interval> &
SYNC_PID=$!
trap "kill $SYNC_PID 2>/dev/null" EXIT

# 5. Run training
python train.py \
  --model_type <model_type> \
  --config_path <merged-config-path> \
  --data_path <data-path> \
  --results_path <results-path> \
  --valid_path <valid-path> \
  --device_ids 0 \
  ${START_CHECKPOINT:+--start_check_point "$START_CHECKPOINT"}

# 6. Run evaluation
python final_valid.py \
  --model_type <model_type> \
  --config_path <merged-config-path> \
  --data_path <data-path> \
  --results_path <results-path> \
  --device_ids 0

# 7. Final sync of results to R2
postdoc results sync <job-id>

# 8. Instance self-terminates via SkyPilot
```

Each step fails fast due to `set -euo pipefail`. If dataset pull fails, training never starts. If training fails, eval never runs.

### Config Translation

The experiment YAML must be translated to the actual `train.py` / `final_valid.py` CLI arguments. `postdoc` handles this mapping:

| Experiment YAML field | Maps to |
|----------------------|---------|
| `model.base_config` | `--config_path` (after merging overrides) |
| `model.base_config` → inferred from config | `--model_type` |
| `dataset.name` | `--data_path` (resolved to local path after pull) + `--dataset_type` |
| `resources.gpu_count` | `--device_ids` (e.g., `0 1` for 2 GPUs) |
| Resubmission checkpoint | `--start_check_point` |
| `dataset` split paths | `--valid_path` |
| *(system-generated)* | `--results_path` (set by `postdoc` runtime to local job output dir) |

Fields marked [PLANNED] in the experiment YAML (`loss`, `split_strategy`) are not translated until `train.py` is modified to support them.

### Config Merging

The `base_config` and `overrides` from the experiment YAML are merged into a single resolved config file at submission time. This resolved config is committed alongside the experiment YAML so the job has a self-contained, unambiguous configuration.

### Checkpoint Sync During Training

A background `postdoc checkpoint sync-daemon` process runs alongside training. At each `checkpoint_interval`, it uploads new checkpoints to R2 at `r2://postdoc/jobs/<job-id>/training/checkpoints/`. This ensures that if the instance is preempted or crashes mid-training, checkpoints are not lost with the instance. The daemon also uploads a `checkpoint_manifest.json` listing all synced checkpoints with their training step and timestamp.

---

## 5. Results Store

### R2 Layout

```
r2://postdoc/
  jobs/
    <job-id>/
      config.yaml          # frozen experiment config (resolved, with overrides applied)
      commit.txt           # git commit hash + experiment branch name
      training/
        checkpoints/       # periodic + final model checkpoints
        logs/              # stdout/stderr from training
        wandb_run_id.txt   # W&B run ID for linking
      eval/
        metrics.json       # { si_sdr: ..., pesq: ..., stoi: ..., sdr: ... }
        samples/           # enhanced audio examples
      meta.json            # job state, timestamps, cost, failure info if any
```

### Local Results DB

A SQLite database at `~/.local/share/postdoc/postdoc.db` (or `$XDG_DATA_HOME/postdoc/postdoc.db`) mirrors `meta.json` and `metrics.json` from all jobs. It syncs from R2 when queried via `postdoc results list/show/compare`. Stored outside the repo to avoid accidental deletion on re-clone and to prevent corruption from concurrent agent sessions.

Purpose: fast local queries across all experiments without R2 round-trips. This powers `postdoc results compare` and will later feed the reporting layer.

### W&B Integration

W&B remains the live metrics dashboard during training. Every job logs to the configured W&B project with tags from the experiment YAML. The structured results on R2 are the **source of truth** for final evaluation metrics; W&B is for live monitoring and training curves.

**W&B config precedence:** The `wandb` section in experiment YAMLs overrides the global `postdoc.yaml` defaults. The global config provides defaults (project, entity); experiments can override `tags` and optionally `project` if needed.

---

## 6. CLI Interface

The `postdoc` CLI is the single interface for both humans and agents.

### Commands

```bash
# Job management
postdoc job submit <experiment.yaml>          # estimate cost → submit to SkyPilot
postdoc job submit experiments/*.yaml         # batch submit multiple experiments
postdoc job status <job-id>                   # current state + live metrics summary
postdoc job list [--state running|failed|done] # list jobs with filters
postdoc job logs <job-id> [--tail]            # fetch logs from R2
postdoc job cancel <job-id>                   # teardown instance immediately
postdoc job retry <job-id>                    # resubmit from last safe checkpoint

# Cost
postdoc cost estimate <experiment.yaml>       # estimate cost without submitting
postdoc cost report [--month 2026-03]         # monthly spend summary

# Results
postdoc results show <job-id>                 # key metrics table for one job
postdoc results compare <job-id> [<job-id>...]  # side-by-side metric comparison
postdoc results export [--format csv|json]    # dump structured results
postdoc results sync                          # pull latest results from R2 into local DB
```

### Package Structure

Uses the same abstract interface architecture defined in `2026-03-23-job-layer-v0.1-local.md`. The cloud backend implements `StorageBackend`, `Scheduler`, and uses the shared `JobTracker`.

```
src/postdoc/
  __init__.py
  cli.py                    # Typer CLI entry point
  config.py                 # Load and validate postdoc.yaml
  context.py                # PostdocContext, create_context() factory
  experiment.py             # Parse experiment YAMLs, merge configs
  run_job.py                # Shared job execution logic (backend-agnostic)
  cost.py                   # Cost estimation (basic for now)
  interfaces/
    __init__.py
    storage.py              # StorageBackend ABC
    scheduler.py            # Scheduler ABC
    tracker.py              # JobTracker (SQLite), JobState, JobRecord
  backends/
    local/
      __init__.py
      storage.py            # LocalStorage(StorageBackend)
      scheduler.py          # LocalScheduler(Scheduler)
    cloud/
      __init__.py
      storage.py            # R2Storage(StorageBackend)
      scheduler.py          # SkyPilotScheduler(Scheduler)
```

Installed as `postdoc` entry point via `pyproject.toml`. Backend is selected via `postdoc.yaml`, `POSTDOC_BACKEND` env var, or `--backend` CLI flag.

### Batch Submit Behavior

`postdoc job submit experiments/*.yaml` submits all matched experiments. Jobs are submitted sequentially (one SkyPilot launch at a time) but run concurrently once provisioned. Before submitting the batch, the aggregate estimated cost is shown. If the aggregate exceeds the soft monthly ceiling, a single warning is shown with a prompt to continue or abort. `--force` skips the prompt.

---

## 7. Guardrails & Safety

### Compute Guardrails

| Guardrail | Behavior |
|-----------|----------|
| **Soft monthly ceiling** | Configurable in `postdoc.yaml`. When cumulative spend approaches the limit, `postdoc job submit` warns with current spend + job estimate. Override with `--force`. |
| **Per-job max duration** | From `run.max_duration` in experiment YAML. SkyPilot auto-terminates if exceeded. |
| **Cost estimation** | Basic for now (GPU hourly rate x estimated duration). A dedicated cost estimator agent will improve this later using historical job data. |

### Code & Data Safety

| Rule | Rationale |
|------|-----------|
| Experiment branches auto-created; main never touched by agents | Keeps main clean; experiment code is isolated |
| Jobs run from pinned commits | Local iteration can't corrupt running experiments |
| R2 job directories are append-only from the job's perspective | Jobs write their own directory, never delete others |
| Checkpoints on R2 are source of truth | Local storage is a cache |

### Agent Guardrails

| Agents CAN | Agents CANNOT |
|------------|---------------|
| Generate experiment YAMLs | Force-push any branch |
| Submit jobs via CLI | Delete R2 data |
| Read logs and results | Modify main branch directly |
| Make code fixes on experiment branches | Resubmit more than once per failure |
| Commit to experiment branches | Override budget ceiling without human approval |

### Action Logging

All agent actions (submissions, fixes, retries, escalations) are logged in `postdoc.db` with timestamps, agent session IDs, and context. This provides an audit trail.

---

## 8. Global Configuration

Single `postdoc.yaml` at repo root:

```yaml
storage:
  r2_bucket: postdoc
  r2_endpoint: <cloudflare-account-endpoint>

compute:
  clouds: [gcp]              # Phase 1: GCP only. vast.ai added in Phase 2 via direct API.
  monthly_budget_soft: 100  # USD, soft ceiling
  default_gpu: A10G

wandb:
  project: harmonic-noise-suppression
  entity: <wandb-username>

notifications:
  method: stdout  # start simple; can add slack/telegram later
```

---

## 9. Failure Handling Detail

### Failure Categories

| Category | Example | Local Repro Strategy | Typical Fix |
|----------|---------|---------------------|-------------|
| **OOM** | CUDA out of memory | Skip local repro (GPU-specific, not reproducible on CPU) | Reduce batch size, enable gradient checkpointing, reduce model size |
| **NaN/Divergence** | Loss becomes NaN | Run few steps with same config on CPU | Lower LR, add gradient clipping, check data |
| **Data Loading** | Missing files, format mismatch | Attempt to load dataset locally | Fix data paths, re-pull dataset |
| **CUDA/Driver** | CUDA version mismatch | Cannot repro locally — classified as infra | Retry on different instance |
| **Infra/Timeout** | Instance preempted, network timeout | Cannot repro locally — classified as infra | Auto-retry once |
| **Eval-specific** | Metric computation fails, shape mismatch | Run eval on single sample locally | Fix eval script |

### Flow

1. Job fails. Logs and last checkpoint saved to R2 (via sync daemon, so recent checkpoints are already there). Instance torn down.
2. Agent reads logs, classifies failure into one of the categories above.
3. **Infra failures** (CUDA/driver, preemption, timeout): auto-retry once on a different instance. Skip local repro — nothing to fix in code.
4. **OOM failures**: skip local repro (not reproducible on CPU). Agent applies mechanical fix (reduce batch size, enable gradient checkpointing) based on log analysis. Commits fix to experiment branch, resubmits.
5. **Code/ML failures** (NaN, data loading, eval-specific): agent checks out experiment branch, runs minimal local repro (CPU, tiny data, few steps). For eval failures, runs eval on a single sample.
6. If reproduced: agent uses systematic-debugging skill. Commits fix to experiment branch.
7. Resubmits from last safe checkpoint (training failure) or from trained model (eval failure). **One resubmission allowed.**
8. If second failure on same issue: state becomes ESCALATED. Human notified with structured report.

### Safe Checkpoint Determination

Before resubmitting from a checkpoint, the agent verifies the checkpoint is safe:
- File is not corrupted (loadable)
- Training metrics at that checkpoint were not already anomalous (no NaN, loss was decreasing)
- If the failure was NaN/divergence, the agent rolls back to an earlier checkpoint where metrics were healthy

---

## 10. Future Integration Points

These are not in scope for this sub-project but the design accommodates them:

- **Data layer** — `postdoc data pull` is a placeholder in the bootstrap script. Will be implemented when the data layer is built. For now, datasets are pulled via existing mechanisms (manual sync or DVC).
- **Cost estimator agent** — `postdoc cost estimate` currently does basic arithmetic (GPU rate x duration). Will be replaced with a smarter agent that uses historical job data to predict duration and cost for new experiments.
- **Reporting layer** — `postdoc results export` and the local SQLite DB provide the structured data that the reporting agent will consume.
- **Experiment design agent** — generates and modifies experiment YAMLs. The YAML format is designed to be agent-friendly: composable, diffable, and self-documenting.
- **Notifications** — currently `stdout`. Adding Slack/Telegram webhook is a config change, not an architecture change.
