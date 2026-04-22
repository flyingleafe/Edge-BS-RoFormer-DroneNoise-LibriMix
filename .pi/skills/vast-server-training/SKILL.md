---
name: vast-server-training
description: Run training on the vast-server GPU host via `postdoc submit` (SkyPilot managed jobs). Use when launching, monitoring, or cancelling training jobs on the remote GPU host.
---

# Vast-Server Training

All training runs on vast-server. `postdoc submit <shell-command>` is the universal interface — it wraps `sky jobs launch` on the `ssh/vast-server` node pool.

**Every submit is git-native**: the local working tree must be clean, the HEAD is pushed to `origin`, and the server runs `git reset --hard <SHA>` + `uv sync` before executing the command. Uncommitted changes do NOT make it to the server. See `src/postdoc/AGENTS.md` for details.

## One-time setup

If `postdoc check` fails or shows the pool unregistered:

```bash
mkdir -p ~/.sky
cp docs/skypilot/ssh_node_pools.yaml.example ~/.sky/ssh_node_pools.yaml
postdoc pool-up     # wraps `sky ssh up` — installs k3s + sky runtime on vast-server
postdoc check
```

Also verify the server can pull the repo:

```bash
ssh vast-server "git ls-remote $(git remote get-url origin) HEAD"
```

If that fails, configure git auth on the server (SSH key with repo access, or deploy key) before submitting.

See `docs/skypilot/README.md` for the fuller walkthrough.

## Daily workflow

```bash
# Make sure your changes are committed first.
git add -A && git commit -m "what you're about to run"

# Submit — preflight pushes HEAD, server checks out that SHA and uv syncs.
postdoc submit python train.py --model_type dccrn --config configs/dccrn_dregon.yaml

# Name it, and request 2 GPUs:
postdoc submit -n dccrn-dregon --gpus 2 python train.py ...

# Iterate fast without committing (NOT RECOMMENDED for real runs — the server
# will run HEAD, NOT your uncommitted edits):
postdoc submit --dirty python quick_test.py

# Already pushed by hand / CI? Skip the push:
postdoc submit --skip-push python train.py ...

# Peek at the generated task YAML without launching (still runs preflight+push):
postdoc submit --dry-run python train.py ...

# Pass env vars:
postdoc submit -e WANDB_MODE=online python train.py ...

# For complex specs (file_mounts, multi-step setup), write a task YAML and
# bypass the git wrapper:
postdoc submit -f my_task.sky.yaml

# List jobs (queued + running); --all adds finished:
postdoc list
postdoc list --all

# Stream logs for a job:
postdoc logs <job-id>

# Controller/scheduler logs (why wasn't my job picked up?):
postdoc logs <job-id> --controller

# Cancel:
postdoc cancel <job-id>          # or: postdoc cancel --all -y

# Drop into a shell on the host (not a job — just ssh):
postdoc ssh

# Web UI (jobs + infra + GPU utilisation):
postdoc dashboard
```

## What `postdoc submit` actually does

1. **Local preflight**: verify clean tree, `git push origin HEAD:refs/heads/<branch>` (or `refs/postdoc/<sha>` if detached). Fail loudly on dirty tree or non-ff push.
2. Generates a SkyPilot task YAML with `infra: ssh/vast-server`, an `envs:` block with `POSTDOC_GIT_SHA` / `POSTDOC_GIT_URL` / `POSTDOC_REPO_DIR`, a setup step that clones (or reuses) the repo + `git reset --hard $SHA` + `uv sync` + `dvc pull`, and your command wrapped by `cd $REPO_DIR && source .venv/bin/activate`.
3. `sky jobs launch -y --detach-run <task.yaml>` — queues as a managed job.

SkyPilot handles queueing, log capture, auto-recovery on controller restart.

## Working with experiment configs

The experiment YAMLs in `experiments/` are **inputs to the training script's `--config` flag**, not to `postdoc`. Pattern:

```bash
postdoc submit python train.py \
    --model_type dccrn \
    --config configs/dccrn_dregon.yaml \
    --results_dir results/dccrn_dregon
```

If a group of runs shares a setup, write a one-line shell script in the repo (`./scripts/run_dccrn.sh` etc.) and submit it:

```bash
postdoc submit ./scripts/run_dccrn.sh
```

Do **not** reintroduce an experiment-YAML mode inside `postdoc`. Structure is the script's concern.

## Raw-ssh fallback (rare)

Only for ad-hoc debugging that doesn't need queueing or logging:

```bash
postdoc ssh
# then inside:
nvidia-smi
tail -f results/.../train.log
```

Anything that should be reproducible or survive disconnection → use `postdoc submit`.

## Syncing results back

Managed-job logs live on the vast-server controller and are retrievable with `postdoc logs`. Checkpoints flow through wandb artifacts (see `docs/data-and-artifacts.md`). Datasets via DVC. The legacy `./sync_results.sh` is still available if you need a full rsync.

## Gotchas

- **Dirty tree blocks submit.** This is the feature, not a bug. If you're iterating, `git commit` between tries (cheap commits are fine; use `--amend` to squash later). `--dirty` is an emergency escape that does NOT ship uncommitted code.
- **Non-fast-forward push blocks submit.** `git fetch && git rebase origin/<branch>` and retry.
- First-ever `postdoc submit` on a fresh vast-server is slow — spins up the jobs controller + clones the repo + `uv sync` from cold. After that, fast.
- `--detach` is the default. Re-attach to logs anytime with `postdoc logs <id>`.
- Two GPUs means two concurrent jobs max. Further submits queue automatically.
- `~/harmonic-noise-suppression` on the server **is** the postdoc checkout. Every submit runs `git reset --hard <SHA>` there. **Do not leave uncommitted edits in that directory on vast-server** — they'll be wiped by the next submit. For manual hacking, clone somewhere else (e.g. `~/scratch/hns-debug`).
