# SkyPilot setup

`postdoc` submits jobs via SkyPilot managed jobs onto an SSH node pool
(vast-server, currently). This doc gets you there in ~5 minutes.

## Prerequisites

* `ssh vast-server` works (key-based, no password prompt). The alias lives in
  `~/.ssh/config` on your laptop.
* `uv sync` (or `pip install -e .[dev]`) — this pulls in `skypilot>=0.11`.
* **The server can clone your git repo.** `postdoc submit` uses git as the
  transport, not rsync. Verify:
  ```bash
  ssh vast-server "git ls-remote $(git remote get-url origin) HEAD"
  ```
  If that fails, fix auth (SSH key / deploy key / credential helper on the
  server) before continuing.

## One-time bootstrap

1. Register the node pool:

   ```bash
   mkdir -p ~/.sky
   cp docs/skypilot/ssh_node_pools.yaml.example ~/.sky/ssh_node_pools.yaml
   # edit if your host alias isn't `vast-server`
   ```

2. Install SkyPilot agents on the host. This pushes a small k3s runtime plus
   the SkyPilot runtime onto the machine. Idempotent; safe to re-run.

   ```bash
   postdoc pool-up        # wraps `sky ssh up`
   ```

3. Verify:

   ```bash
   postdoc check          # wraps `sky check ssh` + `sky status`
   sky gpus list --infra ssh/vast-server
   ```

   You should see the pool listed as enabled and the GPUs visible.

## Daily use

Every `postdoc submit` does:

1. Verify your working tree is clean (fail otherwise).
2. `git push` HEAD to `origin`.
3. Generate a SkyPilot task that `git reset --hard`s to that SHA on the server,
   runs `uv sync`, and then your command.

```bash
# Normal submit:
postdoc submit python train.py --model_type dccrn --config configs/dccrn.yaml

# Name it:
postdoc submit -n dccrn-dregon python train.py ...

# Multiple GPUs:
postdoc submit --gpus 2 python train.py ...

# Dirty tree (uncommitted changes are NOT shipped — only HEAD is):
postdoc submit --dirty python quick_test.py

# Already pushed manually? Skip the push step:
postdoc submit --skip-push python train.py ...

# Non-default git remote:
postdoc submit --remote upstream python train.py ...

# Watch queue:
postdoc list

# Stream logs:
postdoc logs <job-id>

# Cancel:
postdoc cancel <job-id>

# Dry-run (still runs preflight + push, then prints the task YAML):
postdoc submit --dry-run python train.py ...

# Bypass the git wrapper entirely — use a hand-written SkyPilot task YAML:
postdoc submit -f my_task.sky.yaml
```

## How it works under the hood

Each `postdoc submit` runs a **local git preflight**, then builds a SkyPilot
task dict (`src/postdoc/task.py`). The generated task looks like:

```yaml
resources:
  infra: ssh/vast-server
  accelerators: "*:1"
envs:
  POSTDOC_GIT_SHA:  <your HEAD sha>
  POSTDOC_GIT_URL:  git@github.com:you/repo.git
  POSTDOC_REPO_DIR: ~/harmonic-noise-suppression   # reuses the existing clone on vast-server
setup: |
  # install uv if missing
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  # clone if missing; fetch + hard-reset to the SHA
  [ -d $POSTDOC_REPO_DIR/.git ] || git clone $POSTDOC_GIT_URL $POSTDOC_REPO_DIR
  cd $POSTDOC_REPO_DIR
  git fetch --all --prune --tags
  git reset --hard $POSTDOC_GIT_SHA
  uv sync
  ls *.dvc datasets/*.dvc >/dev/null 2>&1 && uv run dvc pull
run: |
  cd $POSTDOC_REPO_DIR
  source .venv/bin/activate
  <your shell command>
```

That YAML is passed to `sky jobs launch -y`. SkyPilot handles:

* Queueing (by default 1 job per GPU).
* Log capture (`sky jobs logs <id>`).
* Dashboard (`sky dashboard`).
* Auto-recovery on controller restart.

Our CLI is ~400 LOC; everything else is SkyPilot's + git.

## Common gotchas

* **Dirty tree → submit fails.** Commit first. `--dirty` forces submit but does
  **not** ship uncommitted changes — only HEAD is on the remote.
* **Non-fast-forward push → submit fails.** Rebase onto `origin/<branch>` and
  retry. `--skip-push` bypasses the push step.
* **Server needs git read access.** Test with
  `ssh vast-server "git ls-remote $(git remote get-url origin) HEAD"`.
* **`setup` block runs as a fresh shell** — any env vars / activations must be
  re-applied. If your training needs `.env`, pass it via `postdoc submit -e KEY=VAL`.
* **`uv sync` runs every submit.** Fast no-op when `uv.lock` is unchanged.
* **First run is slow** — `sky ssh up` installs k3s, and the initial `sky jobs
  launch` spins up the controller. Subsequent submissions are fast.
* **Running jobs survive CLI exit.** `postdoc submit` defaults to `--detach`.
  Reattach logs anytime with `postdoc logs <id>`.

## When not to use postdoc

* Ad-hoc debugging: `ssh vast-server`, run things directly.
* Quick one-off scripts that don't need queueing or logging.

For those, `postdoc ssh` drops you into a shell on the host.
