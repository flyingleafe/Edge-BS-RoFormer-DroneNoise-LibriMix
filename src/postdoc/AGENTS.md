# src/postdoc/ — Job-Runner CLI

Thin Typer wrapper around two backends:

- **direct (default)** — plain SSH to `$POSTDOC_HOST` (default `vast-server-2`).
  Jobs are launched with `nohup bash run.sh` against a shared checkout at
  `/root/harmonic-noise-suppression`. State lives in `/root/.postdoc/` on the
  host. A `tmux` daemon (`postdoc-queue`) drains the queue FIFO when GPUs are
  busy.
- **cloud (opt-in via `--cloud`)** — `sky jobs launch` (SkyPilot managed jobs).
  Fresh container per job. Kept for the rare case where the direct host is
  full / offline; not used day-to-day.

Routing in `cmd_submit`: free-GPU probe → direct if enough free, else cloud
(unless forced with `--direct` / `--cloud`).

## Philosophy

Jobs are shell commands. Everything structured (experiment YAMLs, hyperparam
sweeps, configs) is the *training script's* concern via its own flags. The
runner SSHes, captures logs to a file, queues when GPUs are busy. ~450 LOC
total; no k3s, no Ray, no controller pod.

**Transport is git.** Every submit pushes the local HEAD to `origin` and the
*job* does `git fetch && git reset --hard <SHA> && uv sync && dvc pull` on the
shared checkout before running. Bit-for-bit reproducibility; no silent drift.

**Why direct, not SkyPilot managed jobs (the earlier design).** Managed jobs
spin a fresh pod per job, forcing fresh `git clone` + `uv sync` + `dvc pull`
every time — duplicating ~31 GB per concurrent job on a single on-prem host.
We tried it, paid the startup cost on every submit, and rolled back. The
`cloud` backend stays as an emergency burst path; it is **not** the default.

## Files

| File | Purpose | LOC |
|---|---|---|
| `cli.py` | Typer commands. Top-level dispatch + queue-daemon control + thin wrappers around `direct` / `cloud`. | ~380 |
| `direct.py` | Plain-SSH backend: GPU probe, job-dir layout under `/root/.postdoc/jobs/`, `nohup` launcher, cancel. | ~280 |
| `cloud.py` | SkyPilot managed-jobs backend (opt-in). | ~80 |
| `queue.py` | `postdoc-runner` entry point — the FIFO-draining loop that runs inside the `postdoc-queue` tmux session on the host. | ~200 |
| `task.py` | SkyPilot task-dict builder (cloud backend only). | ~160 |
| `git_state.py` | Preflight: clean-tree check, push HEAD, capture SHA + URL. | ~95 |
| `infer.py` | `postdoc infer` — local tool that runs a trained model on audio files. | ~200 |
| `__init__.py` | Empty. | 1 |

## Commands

```
# Daily:
postdoc submit <cmd...>    [--name -n] [--gpus -g N]
                           [--remote origin] [--dirty] [--skip-push]
                           [--env/-e K=V] [--dry-run]
                           [--direct|--cloud]
                           [--host vast-server-2] [--user root]
postdoc list               [--all]
postdoc status <name>__<id>
postdoc logs <name>__<id>  [-f] [-n LINES]
postdoc cancel <name>__<id>
postdoc check              # nvidia-smi over SSH, lists free GPUs
postdoc ssh                # interactive shell on $POSTDOC_HOST

# Queue daemon (runs in a tmux session on the host):
postdoc queue-start
postdoc queue-stop
postdoc queue-status

# Local-only:
postdoc infer <run-dir> --input ... --output ...
```

## Submit flow (direct backend)

1. `cli.cmd_submit` collects positional args as the command.
2. **Git preflight** (`git_state.snapshot`):
   - Verify clean tree (fail unless `--dirty`).
   - Read `HEAD` SHA + `origin` URL.
   - `git push origin HEAD:refs/heads/<branch>` (or `refs/postdoc/<sha>` if
     detached). Fail on non-ff with a hint.
3. **Backend choice**: probe free GPUs via `nvidia-smi` over SSH. If
   `free ≥ --gpus`, direct; else cloud. `--direct` / `--cloud` force.
4. **Direct path** (`direct.submit_direct`):
   - Ensure `/root/.postdoc/{jobs,queue.fifo}` exist on host.
   - Pick next job id by scanning `jobs/`.
   - Write `jobs/<name>__<id>/{run.sh,job.json,log.txt}`.
   - `nohup bash run.sh >> log.txt 2>&1 &`; record PID into `job.json`.
   - `run.sh` does `cd $REPO_DIR && git fetch && git reset --hard $SHA &&
     uv sync && (uv run dvc pull || true) && <cmd>`.
5. **Queued path** (no free GPUs): append the job JSON to
   `/root/.postdoc/queue.fifo`. The `postdoc-runner` daemon in the
   `postdoc-queue` tmux session reads the FIFO, polls running jobs every
   10 s, and launches queued jobs when their GPU requirement fits.

## One-time host setup

Server-side prerequisites: `git`, `uv`, `python3`, `tmux`, `nvidia-smi`.

1. Generate an SSH keypair on the host and add the pubkey as a **write-enabled
   Deploy Key** on the GitHub repo (`Allow write access`). The queue daemon
   runs detached from any agent — the server needs its own credential.
2. `ssh-keyscan github.com >> ~/.ssh/known_hosts` on the host.
3. `git config --global user.{name,email} ...` so server-side commits are
   attributed.
4. Clone the repo at `/root/harmonic-noise-suppression`. Postdoc assumes this
   exact path (`DEFAULT_REPO_DIR` in `direct.py`).
5. `cd /root/harmonic-noise-suppression && uv sync --no-dev` — populates the
   shared `.venv` and installs `postdoc-runner`.
6. Write server-side `.env` (`WANDB_API_KEY`, `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID`) and
   `dvc remote modify --local r2 endpointurl
   https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`.
7. Verify locally with `postdoc check` (GPU probe) and `postdoc submit
   --dry-run ...` (git preflight).
8. `postdoc queue-start` to spin up the tmux daemon.

Locally: set `POSTDOC_HOST` in `.envrc` (or pass `--host`). Project default is
`vast-server-2`.

## Env overrides

| Var | Default | Effect |
|---|---|---|
| `POSTDOC_HOST` | `vast-server` (constant); overridden to `vast-server-2` in project `.envrc` | Default host for SSH |
| `POSTDOC_USER` | `root` | Default user on the host |
| `POSTDOC_DEFAULT_GPUS` | `1` | Default GPUs per job |
| `POSTDOC_CLUSTER` | `postdoc` | Cluster name (cloud backend only) |

## Gotchas

- **Dirty tree → submit fails.** By design. Commit first. `--dirty` is the
  escape hatch; uncommitted changes are *not* shipped.
- **Non-fast-forward push → submit fails.** Rebase onto `origin/<branch>`
  first. `--skip-push` bypasses the push entirely.
- **Server needs git read+write access to origin.** Deploy key with write
  enabled; the queue runner runs `git fetch` from inside each job script.
  Verify: `ssh $POSTDOC_HOST 'cd /root/harmonic-noise-suppression && git fetch'`.
- **`git reset --hard` clobbers the shared checkout.** Every submit does this.
  **Do not leave uncommitted edits** in `/root/harmonic-noise-suppression` on
  the host. For manual hacking use a second clone (e.g. `~/scratch/hns/`).
- **`.venv` is shared.** Diverging `pyproject.toml` across concurrent jobs
  could cause one job's `uv sync` to mutate the venv another is using.
  Negligible when the lockfile is stable.
- **No auto-recovery.** If the host reboots mid-run, jobs are lost. The cloud
  backend (`--cloud`) gets you managed-job recovery at the cost of per-job
  setup time.
- **`infer` is a local tool.** It takes a *path to a run directory* (with
  `config.yaml` + `training/*.ckpt`). Pull artefacts via wandb or rsync first.

## Not covered here

- Dataset/checkpoint sync — see `docs/data-and-artifacts.md`.
- Experiment-specific recipes — see `.pi/skills/run-experiment/SKILL.md`.
- The hidden `pool-up`/`cluster-up`/`dashboard` stubs in `cli.py` are
  back-compat error messages from the old SkyPilot era; safe to ignore.
