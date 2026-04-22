"""Git preflight for `postdoc submit`.

Every submitted job runs at an explicit commit SHA that has been pushed to the
origin remote. The remote host then `git fetch` + `git reset --hard <SHA>` to
reproduce the tree bit-for-bit. No rsync, no "works on my laptop".
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _run(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            list(args), text=True, stderr=subprocess.PIPE, cwd=cwd,
        ).strip()
    except subprocess.CalledProcessError as e:
        raise GitError(
            f"`{' '.join(args)}` failed (rc={e.returncode}):\n{e.stderr.strip()}"
        ) from e


def in_git_repo(cwd: Path | None = None) -> bool:
    try:
        _run("git", "rev-parse", "--git-dir", cwd=cwd)
        return True
    except GitError:
        return False


def head_sha(cwd: Path | None = None) -> str:
    return _run("git", "rev-parse", "HEAD", cwd=cwd)


def is_dirty(cwd: Path | None = None) -> bool:
    return bool(_run("git", "status", "--porcelain", cwd=cwd))


def remote_url(remote: str = "origin", cwd: Path | None = None) -> str:
    return _run("git", "remote", "get-url", remote, cwd=cwd)


def current_branch(cwd: Path | None = None) -> str | None:
    """Current branch name, or None if detached HEAD."""
    try:
        return _run("git", "symbolic-ref", "--short", "HEAD", cwd=cwd)
    except GitError:
        return None


def push_head(remote: str = "origin", cwd: Path | None = None) -> str:
    """Push current HEAD to the remote. Return the refspec pushed.

    - On a branch: push to ``refs/heads/<branch>``. Fails on non-ff.
    - Detached HEAD: push to ``refs/postdoc/<sha>`` so the commit is reachable
      on origin without touching any branch.
    """
    branch = current_branch(cwd=cwd)
    sha = head_sha(cwd=cwd)
    if branch is None:
        refspec = f"HEAD:refs/postdoc/{sha}"
    else:
        refspec = f"HEAD:refs/heads/{branch}"
    try:
        subprocess.run(
            ["git", "push", remote, refspec],
            check=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise GitError(
            f"git push {remote} {refspec} failed:\n"
            f"{e.stderr.strip()}\n\n"
            f"If the branch diverged, rebase onto {remote}/{branch} and retry.\n"
            f"If there's no upstream, run `git push -u {remote} {branch}` once."
        ) from e
    return refspec


def snapshot(cwd: Path | None = None, *, allow_dirty: bool, skip_push: bool,
             remote: str = "origin") -> dict[str, str]:
    """Preflight bundle used by `postdoc submit`.

    Returns ``{sha, url, branch_or_detached, refspec}``. Raises on dirty tree
    (unless allow_dirty) or on failed push (unless skip_push).
    """
    if not in_git_repo(cwd=cwd):
        raise GitError(f"not a git repository: {cwd or '.'}")
    dirty = is_dirty(cwd=cwd)
    if dirty and not allow_dirty:
        raise GitError(
            "working tree is dirty — commit or stash first, or pass --dirty "
            "(uncommitted changes will NOT be shipped; the remote reflects HEAD only)."
        )
    sha = head_sha(cwd=cwd)
    url = remote_url(remote, cwd=cwd)
    branch = current_branch(cwd=cwd) or f"(detached:{sha[:8]})"
    if skip_push:
        refspec = "(push skipped)"
    else:
        refspec = push_head(remote, cwd=cwd)
    return {"sha": sha, "url": url, "branch": branch, "refspec": refspec, "dirty": str(dirty)}
