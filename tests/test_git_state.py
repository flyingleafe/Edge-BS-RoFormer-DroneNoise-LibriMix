"""Tests for `postdoc.git_state`.

Each test creates a throwaway git repo in tmp_path and exercises the helpers
against it. We use a local file:// "remote" so push is real but contained.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from postdoc import git_state


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return a working repo (``work/``) with an ``origin`` pointing at a bare."""
    bare = tmp_path / "bare.git"
    subprocess.check_call(["git", "init", "--bare", "-q", str(bare)])
    work = tmp_path / "work"
    work.mkdir()
    _git("init", "-q", "-b", "main", cwd=work)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    _git("remote", "add", "origin", str(bare), cwd=work)
    (work / "README").write_text("hi\n")
    _git("add", ".", cwd=work)
    _git("commit", "-q", "-m", "init", cwd=work)
    return work


def test_in_git_repo(tmp_path, repo):
    assert git_state.in_git_repo(cwd=repo)
    assert not git_state.in_git_repo(cwd=tmp_path.parent)


def test_head_sha(repo):
    sha = git_state.head_sha(cwd=repo)
    assert len(sha) == 40
    assert sha == _git("rev-parse", "HEAD", cwd=repo)


def test_is_dirty(repo):
    assert not git_state.is_dirty(cwd=repo)
    (repo / "new.txt").write_text("x")
    assert git_state.is_dirty(cwd=repo)


def test_remote_url(repo):
    url = git_state.remote_url(cwd=repo)
    assert url.endswith("bare.git")


def test_current_branch(repo):
    assert git_state.current_branch(cwd=repo) == "main"


def test_current_branch_detached(repo):
    sha = git_state.head_sha(cwd=repo)
    _git("checkout", "-q", "--detach", sha, cwd=repo)
    assert git_state.current_branch(cwd=repo) is None


def test_push_head_branch(repo):
    refspec = git_state.push_head(cwd=repo)
    assert refspec == "HEAD:refs/heads/main"
    # Ref should now exist on the bare remote.
    remote = git_state.remote_url(cwd=repo)
    out = subprocess.check_output(
        ["git", "ls-remote", remote, "refs/heads/main"], text=True,
    ).strip()
    assert out  # non-empty → ref exists


def test_push_head_detached_uses_postdoc_ref(repo):
    sha = git_state.head_sha(cwd=repo)
    _git("checkout", "-q", "--detach", sha, cwd=repo)
    refspec = git_state.push_head(cwd=repo)
    assert refspec == f"HEAD:refs/postdoc/{sha}"
    remote = git_state.remote_url(cwd=repo)
    out = subprocess.check_output(
        ["git", "ls-remote", remote, f"refs/postdoc/{sha}"], text=True,
    ).strip()
    assert out


def test_snapshot_happy_path(repo):
    snap = git_state.snapshot(cwd=repo, allow_dirty=False, skip_push=False)
    assert snap["branch"] == "main"
    assert snap["sha"] == git_state.head_sha(cwd=repo)
    assert snap["refspec"] == "HEAD:refs/heads/main"
    assert snap["dirty"] == "False"


def test_snapshot_dirty_fails_without_flag(repo):
    (repo / "new.txt").write_text("x")
    with pytest.raises(git_state.GitError, match="dirty"):
        git_state.snapshot(cwd=repo, allow_dirty=False, skip_push=False)


def test_snapshot_dirty_passes_with_flag(repo):
    (repo / "new.txt").write_text("x")
    snap = git_state.snapshot(cwd=repo, allow_dirty=True, skip_push=False)
    assert snap["dirty"] == "True"


def test_snapshot_skip_push(repo):
    snap = git_state.snapshot(cwd=repo, allow_dirty=False, skip_push=True)
    assert snap["refspec"] == "(push skipped)"
    # Nothing pushed to origin.
    remote = git_state.remote_url(cwd=repo)
    out = subprocess.check_output(
        ["git", "ls-remote", remote], text=True,
    ).strip()
    assert "refs/heads/main" not in out


def test_snapshot_not_a_repo(tmp_path):
    with pytest.raises(git_state.GitError, match="not a git repository"):
        git_state.snapshot(cwd=tmp_path, allow_dirty=False, skip_push=True)


def test_push_head_non_ff_fails(repo, tmp_path):
    """If the remote has advanced past our HEAD, push fails with a helpful error."""
    # 1. Seed the bare with main (so later clones aren't empty).
    git_state.push_head(cwd=repo)

    # 2. Second clone → advance origin/main past `repo`.
    other = tmp_path / "other"
    subprocess.check_call(
        ["git", "clone", "-q", "-b", "main",
         git_state.remote_url(cwd=repo), str(other)]
    )
    _git("config", "user.email", "t@t", cwd=other)
    _git("config", "user.name", "t", cwd=other)
    (other / "a.txt").write_text("a")
    _git("add", ".", cwd=other)
    _git("commit", "-q", "-m", "a", cwd=other)
    _git("push", "-q", "origin", "main", cwd=other)

    # 3. `repo` makes a commit on a diverged history → push must fail.
    (repo / "b.txt").write_text("b")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "b", cwd=repo)
    with pytest.raises(git_state.GitError, match="push"):
        git_state.push_head(cwd=repo)
