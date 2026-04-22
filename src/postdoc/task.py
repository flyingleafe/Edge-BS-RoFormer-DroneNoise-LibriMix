"""SkyPilot task-YAML generator (git-native mode).

Every postdoc task:
  1. Clones (or updates) the project repo on the remote to a pinned SHA that
     was just pushed from the local machine. No rsync; git is the transport.
  2. Runs `uv sync` in the repo root so the environment matches the lockfile.
  3. Pulls DVC-tracked datasets.
  4. Runs the user's shell command inside the uv-managed venv.

Schema reference: https://docs.skypilot.co/en/latest/reference/yaml-spec.html
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_POOL = os.environ.get("POSTDOC_SSH_POOL", "vast-server")
DEFAULT_GPUS = int(os.environ.get("POSTDOC_DEFAULT_GPUS", "1"))
DEFAULT_REPO_DIR = os.environ.get("POSTDOC_REPO_DIR", "~/.postdoc/repo")


# Setup: install uv, clone/fetch repo, hard-reset to the submitted SHA,
# uv sync, dvc pull. Idempotent; safe to re-run across jobs.
SETUP_TEMPLATE = """\
set -eo pipefail
echo "[postdoc setup] SHA=$POSTDOC_GIT_SHA  URL=$POSTDOC_GIT_URL  DIR=$POSTDOC_REPO_DIR"

# 1. Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  echo "[postdoc setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone repo if missing.
mkdir -p "$(dirname "$POSTDOC_REPO_DIR")"
if [ ! -d "$POSTDOC_REPO_DIR/.git" ]; then
  echo "[postdoc setup] cloning $POSTDOC_GIT_URL"
  git clone "$POSTDOC_GIT_URL" "$POSTDOC_REPO_DIR"
fi

# 3. Fetch + hard-reset to the pinned SHA. Fetches all refs so that
#    refs/postdoc/<sha> (used for detached HEAD submits) resolves.
cd "$POSTDOC_REPO_DIR"
git fetch --all --prune --tags
git fetch origin "+refs/postdoc/*:refs/postdoc/*" 2>/dev/null || true
git reset --hard "$POSTDOC_GIT_SHA"
git submodule update --init --recursive 2>/dev/null || true
echo "[postdoc setup] repo at $(git rev-parse HEAD)"

# 4. Install project env (fast no-op when lockfile unchanged).
uv sync

# 5. Pull DVC-tracked datasets if any are referenced.
if ls *.dvc datasets/*.dvc >/dev/null 2>&1; then
  uv run dvc pull 2>&1 | tail -20 || echo "WARNING: dvc pull failed; job may lack data"
fi
echo "[postdoc setup] done"
"""

# Run: re-export PATH, cd into repo, activate uv venv, exec user command.
RUN_TEMPLATE = """\
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$POSTDOC_REPO_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[postdoc run] $(git rev-parse --short HEAD) :: {command_label}"
{command}
"""


def build_task(
    command: str,
    *,
    git_sha: str,
    git_url: str,
    name: str | None = None,
    gpus: int = DEFAULT_GPUS,
    pool: str = DEFAULT_POOL,
    repo_dir: str = DEFAULT_REPO_DIR,
    envs: dict[str, str] | None = None,
    extra_resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a SkyPilot task dict that checks out ``git_sha`` on the remote."""
    base_envs = {
        "POSTDOC_GIT_SHA": git_sha,
        "POSTDOC_GIT_URL": git_url,
        "POSTDOC_REPO_DIR": repo_dir,
    }
    if envs:
        base_envs.update(envs)

    # Keep the run script tidy: label is a short echo, command runs verbatim.
    label = command.replace("\n", " ")[:120]
    run_script = RUN_TEMPLATE.format(command=command, command_label=label)

    task: dict[str, Any] = {}
    if name:
        task["name"] = name
    task["resources"] = {"infra": f"ssh/{pool}"}
    if gpus:
        task["resources"]["accelerators"] = f"*:{gpus}"
    if extra_resources:
        task["resources"].update(extra_resources)
    task["envs"] = base_envs
    task["setup"] = SETUP_TEMPLATE
    task["run"] = run_script
    return task


def dump_task_yaml(task: dict[str, Any], path: Path) -> Path:
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return path
