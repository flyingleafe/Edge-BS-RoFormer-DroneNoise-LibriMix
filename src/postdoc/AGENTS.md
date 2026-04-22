# src/postdoc/ — Job-Runner CLI

Thin Typer wrapper around SkyPilot **managed jobs** (`sky jobs launch|queue|logs|cancel`) running on an **SSH node pool** (currently: `vast-server`).

## Philosophy

Jobs are shell commands. Everything structured (experiment YAMLs, hyperparam sweeps, configs) is the *training script's* concern via its own flags. The runner runs the command, captures logs, retries on failure, queues when GPUs are busy — all delegated to SkyPilot. We own ~400 LOC; SkyPilot handles the rest.

**Transport is git, not rsync.** Every submit pushes the local HEAD to `origin` and the remote does `git reset --hard <SHA>` before running. This guarantees bit-for-bit reproducibility and kills silent drift from uncommitted changes.

See the top-level `AGENTS.md` `Philosophy` section for the reasoning.

## Files

| File | Purpose | LOC |
|---|---|---|
| `cli.py` | Typer commands; all but `infer` just shell out to `sky`. | ~270 |
| `task.py` | Build a SkyPilot task dict with git-checkout + `uv sync` setup. | ~100 |
| `git_state.py` | Preflight helpers: clean-tree check, push HEAD, capture SHA+URL. | ~95 |
| `infer.py` | `postdoc infer` — local tool, runs a trained model on audio files. Takes a path to a run directory. | ~200 |
| `__init__.py` | Empty. | 1 |

## Commands

```
postdoc submit <cmd...>    [--name -n] [--gpus -g N] [--pool NAME]
                           [--remote origin] [--repo-dir PATH]
                           [--dirty] [--skip-push]
                           [--file/-f task.yaml] [--env/-e K=V]
                           [--dry-run] [--detach|--attach]
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
3. `task.build_task` produces a task dict with:
   - `resources.infra = ssh/<pool>`, `resources.accelerators = '*:N'`
   - `envs: {POSTDOC_GIT_SHA, POSTDOC_GIT_URL, POSTDOC_REPO_DIR}`
   - `setup` — install `uv`, clone if missing, `git fetch --all`, `git reset --hard $SHA`, `uv sync`, `dvc pull`
   - `run` — `cd $REPO_DIR && source .venv/bin/activate && <command>`
   - **no `workdir:`** (git is the transport)
4. Dump to a temp YAML; `sky jobs launch -y -n <name> --detach-run <yaml>`.

Managed jobs survive controller restarts and auto-recover on transient infra failures.

## One-time host setup

See `docs/skypilot/README.md`. Short version:

1. Ensure `ssh vast-server` works with key auth.
2. Copy `docs/skypilot/ssh_node_pools.yaml.example` → `~/.sky/ssh_node_pools.yaml`.
3. `postdoc pool-up` (wraps `sky ssh up` — installs SkyPilot runtime on the host).
4. `postdoc check` to verify.
5. **Git auth**: the server must be able to `git clone $origin_url`. Either the same SSH key already on the host has read access to the repo, or configure a deploy key / credential helper. Test with `ssh vast-server 'git ls-remote <origin_url>'`.

## Env overrides

| Var | Default | Effect |
|---|---|---|
| `POSTDOC_SSH_POOL` | `vast-server` | Default pool name for new submits |
| `POSTDOC_DEFAULT_GPUS` | `1` | Default `--gpus` |
| `POSTDOC_REPO_DIR` | `~/harmonic-noise-suppression` | Path on the remote where the repo is checked out (reuses the pre-existing clone on vast-server) |

## Gotchas

- **Dirty tree → submit fails.** By design. Commit first. `--dirty` is the escape hatch; uncommitted changes are *not* shipped (git is the transport).
- **Non-fast-forward push → submit fails.** Rebase onto `origin/<branch>` first. `--skip-push` skips the push entirely (useful when CI or a teammate already pushed the commit).
- **Server needs git read access to origin.** The `setup:` step runs `git clone` / `git fetch` using the server's own credentials. Test with `ssh <host> 'git ls-remote <origin_url>'`.
- **`uv sync` runs every job.** Fast no-op when the lockfile is unchanged. First launch ever on a fresh host is slow (installs uv + resolves deps).
- **`setup` is a fresh shell.** No `.bashrc`/`.env` auto-loads. Pass env vars via `postdoc submit -e KEY=VAL`.
- **First submit is slow.** `sky ssh up` installs k3s; the first managed-job launch spins up the jobs controller. Subsequent submissions are fast.
- **`infer` is a local tool.** It takes a *path to a run directory* (containing `config.yaml` + `training/*.ckpt`). Runs don't live locally by default under SkyPilot — pull from wandb or rsync first.
- **`git reset --hard` clobbers the server checkout.** `POSTDOC_REPO_DIR` defaults to `~/harmonic-noise-suppression` on vast-server — the same directory you'd `cd` into for manual work. **Do not leave uncommitted edits there.** Every submit wipes them (by design: the remote reflects the pushed SHA exactly).
- **No watcher-agent yet.** The "companion agent that edits code on failure" idea lives as a TODO. SkyPilot's managed-job auto-recovery handles the transient-failure baseline meanwhile.

## Not covered here

- SkyPilot itself — see `docs/skypilot/README.md` and https://docs.skypilot.co.
- Dataset/checkpoint sync — see `docs/data-and-artifacts.md`.
- Experiment-specific recipes — see `.pi/skills/run-experiment/SKILL.md`.
