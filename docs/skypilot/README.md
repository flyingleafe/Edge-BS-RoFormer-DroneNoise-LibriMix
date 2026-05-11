# SkyPilot setup (cloud backend, opt-in only)

> **Read this only if you need the `--cloud` backend.** The default
> `postdoc submit` flow is plain SSH to `$POSTDOC_HOST` (`vast-server-2`).
> See `src/postdoc/AGENTS.md` and `.pi/skills/vast-server-training/SKILL.md`
> for the daily workflow.

`postdoc submit --cloud` shells out to `sky jobs launch`, which schedules a
managed job onto an SSH node pool. We keep this around as a burst path when
the direct host is full or down; it is **not** the default.

## Prerequisites

* `ssh <pool-host>` works key-based.
* `uv sync` (or `pip install -e .[dev]`) — pulls in `skypilot>=0.11`.
* The server can clone the repo:
  ```bash
  ssh <pool-host> "git ls-remote $(git remote get-url origin) HEAD"
  ```

## One-time bootstrap

1. Register the node pool:

   ```bash
   mkdir -p ~/.sky
   cp docs/skypilot/ssh_node_pools.yaml.example ~/.sky/ssh_node_pools.yaml
   # edit the alias to match your ~/.ssh/config host
   ```

2. Install the SkyPilot agents on the host (k3s runtime + SkyPilot runtime;
   idempotent):

   ```bash
   sky ssh up
   ```

3. Verify:

   ```bash
   sky check ssh
   sky status
   sky gpus list --infra ssh/<pool-host>
   ```

## Daily use

```bash
# Force the cloud backend (otherwise direct is preferred):
postdoc submit --cloud python train.py --model_type dccrn --config configs/dccrn.yaml

# Inspect generated task YAML without launching:
postdoc submit --cloud --dry-run python train.py ...

# Watch the SkyPilot queue / dashboard directly:
sky jobs queue
sky dashboard
```

`postdoc list` / `postdoc logs` / `postdoc cancel` only operate on the **direct
backend's** job state (`/root/.postdoc/jobs/`). For cloud jobs use the `sky`
CLI: `sky jobs queue`, `sky jobs logs <id>`, `sky jobs cancel <id>`.

## What the cloud task looks like

`src/postdoc/task.py` builds a task dict roughly:

```yaml
resources:
  infra: ssh/<pool-host>
  accelerators: "*:1"
envs:
  POSTDOC_GIT_SHA:  <your HEAD sha>
  POSTDOC_GIT_URL:  git@github.com:you/repo.git
  POSTDOC_REPO_DIR: ~/harmonic-noise-suppression
setup: |
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
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

That YAML is passed to `sky jobs launch -y`. SkyPilot handles queueing, log
capture (`sky jobs logs`), the dashboard, and auto-recovery on controller
restart — the perks we forfeit by using the direct backend.

## Common gotchas

* **Dirty tree → submit fails.** Same as direct; the git preflight runs first
  regardless of backend.
* **`sky ssh up` is slow on first run** (installs k3s, GPU operator, etc.).
* **Repo clone inside a fresh pod is slow** every job — this is exactly why
  the direct backend is the default.
* **Server needs git read access.** Test with the `git ls-remote` check above.

## When not to use postdoc

Ad-hoc debugging that doesn't need queueing or logging: `postdoc ssh`, then
run things directly.
