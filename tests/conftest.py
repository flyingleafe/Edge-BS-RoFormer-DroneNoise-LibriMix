import pytest


FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
FAKE_URL = "git@github.com:user/repo.git"


@pytest.fixture
def fake_git(monkeypatch):
    """Monkeypatch `postdoc.git_state` so CLI submit tests don't shell out to git."""
    from postdoc import git_state
    calls: list[tuple[str, dict]] = []

    def _snapshot(*, cwd=None, allow_dirty=False, skip_push=False, remote="origin"):
        calls.append(("snapshot", {
            "cwd": cwd, "allow_dirty": allow_dirty,
            "skip_push": skip_push, "remote": remote,
        }))
        return {
            "sha": FAKE_SHA, "url": FAKE_URL,
            "branch": "main",
            "refspec": "HEAD:refs/heads/main",
            "dirty": "True" if allow_dirty else "False",
        }

    monkeypatch.setattr(git_state, "snapshot", _snapshot)
    return type("FakeGit", (), {"calls": calls})


@pytest.fixture
def fake_sky(monkeypatch):
    """Replace subprocess.run so no real `sky` calls happen.

    Records every call as `fake_sky.calls` (list of argv lists).
    """
    calls: list[list[str]] = []

    class _Result:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(argv, *args, **kwargs):
        calls.append(list(argv))
        return _Result()

    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)

    # Pretend `sky` is on PATH.
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    return type("FakeSky", (), {"calls": calls})
