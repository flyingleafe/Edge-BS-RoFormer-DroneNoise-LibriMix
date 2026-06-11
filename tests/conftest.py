import pytest

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
FAKE_URL = "git@github.com:user/repo.git"


@pytest.fixture
def fake_git(monkeypatch):
    """Monkeypatch `postdoc.git_state.snapshot` so CLI tests don't shell out to git."""
    from postdoc import git_state

    calls: list[tuple[str, dict]] = []

    def _snapshot(*, cwd=None, allow_dirty=False, skip_push=False, remote="origin"):
        calls.append(
            (
                "snapshot",
                {
                    "cwd": cwd,
                    "allow_dirty": allow_dirty,
                    "skip_push": skip_push,
                    "remote": remote,
                },
            )
        )
        return {
            "sha": FAKE_SHA,
            "url": FAKE_URL,
            "branch": "main",
            "refspec": "HEAD:refs/heads/main",
            "dirty": "True" if allow_dirty else "False",
        }

    monkeypatch.setattr(git_state, "snapshot", _snapshot)
    return type("FakeGit", (), {"calls": calls})


@pytest.fixture
def fake_sky(monkeypatch):
    """Replace subprocess.run so no real `sky`/`ssh`/`git` calls happen.

    Records every call as ``.calls``. Tests can customize responses with:
        fake_sky.stdout_for = ("sky status postdoc", "some stdout")
        fake_sky.rc_for     = ("sky status postdoc", 1)
    The prefix is matched against the space-joined argv.
    """
    calls: list[list[str]] = []
    state = {"stdout_for": None, "rc_for": None}

    class _Result:
        def __init__(self, rc=0, stdout="", stderr=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(argv, *args, **kwargs):
        calls.append(list(argv))
        cmd_str = " ".join(argv)
        rc = 0
        stdout = ""
        if state["stdout_for"] and cmd_str.startswith(state["stdout_for"][0]):
            stdout = state["stdout_for"][1]
        if state["rc_for"] and cmd_str.startswith(state["rc_for"][0]):
            rc = state["rc_for"][1]
        return _Result(rc=rc, stdout=stdout)

    import subprocess

    monkeypatch.setattr(subprocess, "run", _fake_run)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    class FakeSky:
        def __init__(self):
            self.calls = calls
            self._state = state

        @property
        def stdout_for(self):
            return self._state["stdout_for"]

        @stdout_for.setter
        def stdout_for(self, v):
            self._state["stdout_for"] = v

        @property
        def rc_for(self):
            return self._state["rc_for"]

        @rc_for.setter
        def rc_for(self, v):
            self._state["rc_for"] = v

    return FakeSky()
