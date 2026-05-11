---
name: vast-server-training
description: Run training on the vast-server GPU host via `postdoc submit` (direct SSH, simplified backend). Use when launching, monitoring, or cancelling training jobs on the remote GPU host.
---

# Vast-Server Training

All training runs on the GPU host pointed at by `$POSTDOC_HOST` (default
`vast-server-2`; set in the project `.envrc`). `postdoc submit <shell-command>`
is the universal interface — it wraps a plain `ssh` + `nohup` launcher.
SkyPilot is opt-in only via `--cloud`.

**Every submit is git-native**: the local working tree must be clean, HEAD is
pushed to `origin`, and the server runs `git fetch && git reset --hard <SHA> &&
uv sync && dvc pull` inside the shared checkout before executing the command.
Uncommitted changes do NOT make it to the server. See
`src/postdoc/AGENTS.md` for the exact flow.

## Architecture (one paragraph)

`postdoc submit` SSHes to `$POSTDOC_HOST`, probes free GPUs with `nvidia-smi`,
and if any are free `nohup bash run.sh`s the job out of
`/root/.postdoc/jobs/<name>__<id>/` against the shared checkout at
`/root/harmonic-noise-suppression`. If no GPU is free, the job descriptor is
appended to `/root/.postdoc/queue.fifo` and the **`postdoc-queue` tmux
daemon** (running `postdoc-runner` from the project's venv) picks it up when a
GPU frees. State lives entirely on the server's filesystem; no controller, no
Ray, no k3s.

## One-time host setup (already done for `vast-server-2`)

Replicate only if commissioning a fresh GPU box. Server-side prerequisites:
`git`, `uv`, `python3`, `tmux`, `nvidia-smi`.

```bash
# On the server, as root:
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519     # for github auth
ssh-keyscan -t ed25519,rsa,ecdsa github.com >> ~/.ssh/known_hosts
# → add the printed pubkey as a write-enabled Deploy Key on the repo.
git config --global user.name  "<you> (vast-server-2)"
git config --global user.email "<you>@users.noreply.github.com"
git clone git@github.com:flyingleafe/harmonic-noise-suppression.git \
    /root/harmonic-noise-suppression
cd /root/harmonic-noise-suppression
export PATH="/root/.local/bin:$PATH"
uv sync --no-dev
# Then put .env (WANDB_API_KEY + AWS_* + R2_ACCOUNT_ID) and
# `dvc remote modify --local r2 endpointurl https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`
# in place — see docs/data-and-artifacts.md.
```

Locally: set `POSTDOC_HOST` (or pass `--host`). The project `.envrc` already
exports the default.

```bash
postdoc check        # probes nvidia-smi over SSH; lists free GPUs
postdoc queue-start  # tmux session 'postdoc-queue' running postdoc-runner
postdoc queue-status
```

## Daily workflow

```bash
# Make sure your changes are committed first.
git add -A && git commit -m "what you're about to run"

# Submit — preflight pushes HEAD, server checks out that SHA in the shared repo.
postdoc submit python train.py --model_type dccrn --config configs/dccrn_dregon.yaml

# Name it, and request 2 GPUs (if the host has them):
postdoc submit -n dccrn-dregon --gpus 2 python train.py ...

# Iterate fast without committing (NOT RECOMMENDED for real runs — the server
# will run HEAD, NOT your uncommitted edits):
postdoc submit --dirty python quick_test.py

# Already pushed by hand / CI? Skip the push:
postdoc submit --skip-push python train.py ...

# Peek at the resolved plan without launching (still runs preflight+push):
postdoc submit --dry-run python train.py ...

# Force the SkyPilot cloud backend (fresh container per job) instead of direct:
postdoc submit --cloud python train.py ...

# Pass env vars (only honoured by the cloud backend; the direct backend
# inherits the server's environment + .env):
postdoc submit -e WANDB_MODE=online --cloud python train.py ...

# List jobs (running + queued); --all adds finished:
postdoc list
postdoc list --all

# One-line job summary:
postdoc status <name>__<id>

# Tail logs:
postdoc logs <name>__<id>           # last 50 lines
postdoc logs -f <name>__<id>        # follow

# Cancel:
postdoc cancel <name>__<id>

# Drop into a shell on the host (not a job — just ssh):
postdoc ssh

# Point at a different host ad hoc:
postdoc check --host vast-server
postdoc submit --host other-box python train.py ...
```

## What `postdoc submit` actually does

1. **Local preflight** (`git_state.snapshot`): verify clean tree,
   `git push origin HEAD:refs/heads/<branch>` (or `refs/postdoc/<sha>` if
   detached). Fail loudly on dirty tree or non-ff push.
2. SSH-probe GPUs via `nvidia-smi --query-gpu=...`.
3. **Free GPUs ≥ requested** → write `/root/.postdoc/jobs/<name>__<id>/{run.sh,job.json,log.txt}`, then `nohup bash run.sh` it. `run.sh` does `cd $REPO_DIR && git fetch && git reset --hard $SHA && uv sync && dvc pull && <cmd>`.
4. **Otherwise** → append the job JSON to `/root/.postdoc/queue.fifo`. The
   `postdoc-queue` tmux daemon picks it up when GPUs free.

## Working with experiment configs

The experiment YAMLs in `experiments/` are **inputs to the training script's
`--config` flag**, not to `postdoc`. Pattern:

```bash
postdoc submit python train.py \
    --model_type dccrn \
    --config configs/dccrn_dregon.yaml \
    --results_dir results/dccrn_dregon
```

If a group of runs shares a setup, write a one-line shell script in the repo
(`./scripts/run_dccrn.sh` etc.) and submit it:

```bash
postdoc submit ./scripts/run_dccrn.sh
```

Do **not** reintroduce an experiment-YAML mode inside `postdoc`. Structure is
the script's concern.

## Syncing results back

- **Datasets**: `dvc pull` (locally or in-job via the per-job script).
- **Checkpoints + metrics**: `wandb` artifacts.
- **Raw logs**: stream with `postdoc logs <name>__<id>`; the file lives at
  `/root/.postdoc/jobs/<name>__<id>/log.txt`. The legacy `./sync_results.sh`
  is still available for a full rsync of the `results/` dir.

See `docs/data-and-artifacts.md` for the full DVC + wandb story.

## Gotchas

- **Dirty tree blocks submit.** Feature, not bug. Commit between tries (cheap
  commits are fine; squash with `--amend` / `git rebase -i` later). `--dirty`
  is an emergency escape that does NOT ship uncommitted code.
- **Non-fast-forward push blocks submit.** `git fetch && git rebase
  origin/<branch>` and retry, or `--skip-push` if you've already pushed.
- **`/root/harmonic-noise-suppression` on the server is the shared checkout.**
  Every submit runs `git reset --hard <SHA>` there. **Do not leave uncommitted
  edits in that directory** — they'll be wiped by the next submit. For manual
  hacking, clone somewhere else (e.g. `~/scratch/hns-debug`).
- **One GPU on `vast-server-2`.** A second concurrent submit will queue. Watch
  with `postdoc list`. (Old `vast-server` had two.)
- **Queue daemon needs to be running for queued jobs to launch.** Check with
  `postdoc queue-status`; (re)start with `postdoc queue-start`.
- **First submit after server reboot is slowest.** `uv sync` reuses caches but
  re-resolves the lockfile; subsequent jobs are fast.
- **Server-side secrets are not auto-synced.** `.env` (WANDB + AWS/R2) and
  `.dvc/config.local` live on the server and are gitignored. If `dvc pull`
  fails in jobs, that's where to look.
