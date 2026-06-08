"""
Centralised path resolution for data and artifacts.

In a worktree, ``data/``, ``datasets/``, and ``results/`` reside in the
*main* repository checkout — not in the worktree.  Set ``DATA_ROOT`` in
``.env`` to the main repo root so that all code (scripts, notebooks, tests)
resolves those directories correctly regardless of where it runs.

Usage::

    from utils.paths import get_data_path, get_datasets_path, get_results_path

    dregon_dir = get_data_path("DREGON")
    dregon_lm  = get_datasets_path("DREGON-LM")
    eval_dir   = get_results_path("evaluation")

If ``DATA_ROOT`` is not set, falls back to the Git repository root (the
default works for a regular single-checkout workflow).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load once at import time — idempotent, so repeated calls by other modules are harmless.
load_dotenv()

_DATA_ROOT: Path | None = None


def _resolve_data_root() -> Path:
    """Resolve DATA_ROOT from environment or fall back to git root."""
    raw = os.environ.get("DATA_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()

    # Fallback 1: try the main worktree (git worktree list).
    # In a linked worktree, 'git rev-parse --show-toplevel' returns the
    # worktree itself — not the main checkout.  The main checkout is the
    # first line of 'git worktree list'.
    try:
        result = subprocess.run(
            ["git", "worktree", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        first_line = result.stdout.strip().split("\n")[0]
        # Format: "<path> <hash> [<branch>]"
        main_path = first_line.split()[0]
        candidate = Path(main_path).resolve()
        # Only accept if it looks plausible (has a .git dir/file)
        if (candidate / ".git").exists() or (candidate / "data").exists():
            return candidate
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass

    # Fallback 2: git rev-parse --show-toplevel (works for single checkout)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Last resort: current working directory
        return Path.cwd().resolve()


def get_data_root() -> Path:
    """Return the root directory containing ``data/``, ``datasets/``, ``results/``."""
    global _DATA_ROOT
    if _DATA_ROOT is None:
        _DATA_ROOT = _resolve_data_root()
    return _DATA_ROOT


def get_data_path(subpath: str = "") -> Path:
    """Resolve ``data/<subpath>`` (e.g. ``"DREGON"`` → ``<DATA_ROOT>/data/DREGON``)."""
    p = get_data_root() / "data"
    if subpath:
        p = p / subpath
    return p


def get_datasets_path(subpath: str = "") -> Path:
    """Resolve ``datasets/<subpath>`` (e.g. ``"DREGON-LM"`` → ``<DATA_ROOT>/datasets/DREGON-LM``)."""
    p = get_data_root() / "datasets"
    if subpath:
        p = p / subpath
    return p


def get_results_path(subpath: str = "") -> Path:
    """Resolve ``results/<subpath>`` (e.g. ``"evaluation"`` → ``<DATA_ROOT>/results/evaluation``)."""
    p = get_data_root() / "results"
    if subpath:
        p = p / subpath
    return p
