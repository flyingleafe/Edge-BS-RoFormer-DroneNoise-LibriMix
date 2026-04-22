# src/postdoc/ — Job-Runner CLI

Thin Typer wrapper around SkyPilot **cluster jobs** (`sky launch -c postdoc` once + `sky exec` per submit) running on an **SSH node pool** (currently: `vast-server`).

## Philosophy

Jobs are shell commands. Everything structured (experiment YAMLs, hyperparam sweeps, configs) is the *training script's* concern via its own flags. The runner runs the command, captures logs, queues when GPUs are busy — all delegated to SkyPilot. We own ~450 LOC; SkyPilot handles the rest.

**One persistent cluster; one shared host directory; many jobs.** The cluster pod mounts `/root/harmonic-noise-suppression` from the vast-server host via a k8s `hostPath` volume. All jobs on the cluster see the same `.venv/`, `datasets/`, `results/`, `wandb/`. No per-job downloads. No duplication. Two concurrent jobs fit in the host's disk budget because they share it.

**Transport is git, not rsync.** Every submit pushes the local HEAD to `origin` and the *job* (not the cluster) does `git reset --hard <SHA>` + `uv sync` on the mounted repo before running. Bit-for-bit reproducibility; no silent drift.

**Not managed jobs.** We use `sky exec`, not `sky jobs launch`. Managed jobs create a fresh pod per job — which would force fresh `git clone` + `uv sync` + `dvc pull` every time, duplicating ~31 GB of data per concurrent job. Wrong primitive for a single on-prem host. See the top-level `AGENTS.md` `Philosophy` for the reasoning.

## Files

| File | Purpose | LOC |
|---|---|---|
| `cli.py` | Typer commands; all but `infer` just shell out to `sky`. | ~280 |
| `task.py` | Build SkyPilot task dicts: `build_bootstrap_task` (cluster-up, hostPath mount), `build_exec_task` (per-submit, git reset + uv sync + user cmd). | ~160 |
| `git_state.py` | Preflight helpers: clean-tree check, push HEAD, capture SHA+URL. | ~95 |
| `infer.py` | `postdoc infer` — local tool, runs a trained model on audio files. Takes a path to a run directory. | ~200 |
| `__init__.py` | Empty. | 1 |

## Commands

```
# One-time / rare:
postdoc pool-up                   sky ssh up           (k3s on vast-server)
postdoc pool-down                 sky ssh down
postdoc cluster-up [--pool --gpus --repo-dir --dry-run]
                                  sky launch -c postdoc <bootstrap.yaml>
postdoc cluster-down [-y]         sky down postdoc
postdoc cluster-status            sky status postdoc

# Daily:
postdoc submit <cmd...>    [--name -n] [--gpus -g N]
                           [--remote origin] [--dirty] [--skip-push]
                           [--env/-e K=V] [--dry-run]
                           [--auto-up/--no-auto-up]
postdoc list               [--all] [--no-refresh]
postdoc status <job-id>
postdoc logs <job-id>      [--controller] [--no-follow]
postdoc cancel <ids...>|--all [-y]
postdoc ssh                [--pool NAME]
postdoc dashboard
postdoc check                    # `sky check ssh` + `sky status`
postdoc pool-up                  # `sky ssh up` — bootstrap node pool
postdoc infer <run-dir> --input ... --output ...
```

## Submit flow

1. `cli.cmd_submit` collects extra positional args as the command.
2. **Git preflight** (`git_state.snapshot`):
   - Verify clean tree (fail unless `--dirty`).
   - Read `HEAD` SHA + `origin` URL.
   - `git push origin HEAD:refs/heads/<branch>` (or `refs/postdoc/<sha>` if detached). Fail on non-ff with a hint.
3. If `--auto-up` (default) and cluster is not UP, `sky launch -c postdoc <bootstrap.yaml>` first.
4. `task.build_exec_task` produces a task dict (no `infra`, no `setup` — `sky exec` ignores those) with:
   - `resources.accelerators = ':N'` (any GPU, N count)
   - `envs: {POSTDOC_GIT_SHA, POSTDOC_GIT_URL, POSTDOC_REPO_DIR}`
   - `run` — `cd $REPO_DIR && git fetch && git reset --hard $SHA && uv sync && dvc pull && source .venv/bin/activate && <command>`
5. Dump to a temp YAML; `sky exec postdoc -d -n <name> <yaml>`.

The cluster is long-lived; all of this runs inside the same pod across submissions.

## Cluster (bootstrap) flow

`build_bootstrap_task` produces a task dict with:
   - `resources.infra = ssh/<pool>`, `resources.accelerators = ':2'` (reserve both GPUs for the cluster)
   - `envs: {POSTDOC_REPO_DIR}`
   - `setup` — install uv, pre-sync the venv
   - `run` — `echo ready; sleep` (keeps the pod alive for exec's)
   - `config.ssh.pod_config.spec` — injects a raw k8s `hostPath` volume so `/root/harmonic-noise-suppression` on the **host** is mounted at the same path inside the **pod**. `securityContext.runAsUser: 0` matches the host's root ownership of that directory.

## One-time host setup

See `docs/skypilot/README.md`. Short version:

1. Ensure `ssh vast-server` works with key auth.
2. Copy `docs/skypilot/ssh_node_pools.yaml.example` → `~/.sky/ssh_node_pools.yaml`.
3. `postdoc pool-up` (wraps `sky ssh up` — installs k3s + GPU operator on the host; slow first time).
4. `postdoc check` to verify the pool is enabled.
5. **Repo exists on host** at `/root/harmonic-noise-suppression` — postdoc mounts this path into the pod; it must pre-exist on vast-server. First-time setup: `ssh vast-server 'git clone <url> ~/harmonic-noise-suppression'`.
6. `postdoc cluster-up` to launch the persistent pod (can also happen implicitly on first `postdoc submit`).

## Env overrides

| Var | Default | Effect |
|---|---|---|
| `POSTDOC_SSH_POOL` | `vast-server` | Default pool name |
| `POSTDOC_DEFAULT_GPUS` | `1` | Default GPUs per job |
| `POSTDOC_CLUSTER_GPUS` | `2` | GPUs the cluster pod reserves (= max concurrent jobs) |
| `POSTDOC_REPO_DIR` | `/root/harmonic-noise-suppression` | Host path mounted into the pod (same path inside). Must exist on vast-server. |
| `POSTDOC_CLUSTER` | `postdoc` | Cluster name used for `sky launch -c` / `sky exec` / `sky queue` |

## Gotchas

- **Dirty tree → submit fails.** By design. Commit first. `--dirty` is the escape hatch; uncommitted changes are *not* shipped (git is the transport).
- **Non-fast-forward push → submit fails.** Rebase onto `origin/<branch>` first. `--skip-push` skips the push entirely.
- **Server needs git read access to origin.** The per-job script runs `git fetch` from inside the pod using the mounted `.git/`'s configured credentials. If it fails, `ssh vast-server 'cd ~/harmonic-noise-suppression && git fetch'` must work first.
- **`git reset --hard` clobbers the shared checkout.** Every submit does this. **Do not leave uncommitted edits** in `/root/harmonic-noise-suppression` on vast-server. For manual hacking, use a second clone (`~/scratch/hns/`).
- **Concurrency race window.** Two near-simultaneous submits of different SHAs will run `git reset --hard` against the same working tree within seconds of each other. Safe in practice because Python caches imports at startup. If bitten, the follow-up is per-job `git worktree add .postdoc/runs/<id>/` — not yet implemented.
- **`.venv` is shared.** A diverging `pyproject.toml` across concurrent jobs could cause one job's `uv sync` to mutate the venv another is using. Negligible when the lockfile is stable; avoid submitting two jobs at radically different dependency-graphs simultaneously.
- **First submit is slow.** `sky ssh up` installs k3s + GPU operator; `cluster-up` installs uv + syncs the venv. After that, submits are fast.
- **No managed-jobs auto-recovery.** We traded it for shared-host-volume semantics. If vast-server reboots mid-run, jobs are lost. Follow-up: watcher-agent that detects failure, classifies, and re-submits.
- **`infer` is a local tool.** It takes a *path to a run directory* (containing `config.yaml` + `training/*.ckpt`). Runs don't live locally by default under this flow — pull via wandb or rsync first.

## Not covered here

- SkyPilot itself — see `docs/skypilot/README.md` and https://docs.skypilot.co.
- Dataset/checkpoint sync — see `docs/data-and-artifacts.md`.
- Experiment-specific recipes — see `.pi/skills/run-experiment/SKILL.md`.
