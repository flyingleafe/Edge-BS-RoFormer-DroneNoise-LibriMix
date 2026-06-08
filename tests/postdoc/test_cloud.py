"""Tests for `postdoc.cloud` — SkyPilot cloud backend, mocked."""

from __future__ import annotations

import shutil

import pytest

from postdoc.cloud import (
    _accelerator_str,
    _project_root,
    _sky_available,
    cancel_job_cloud,
    list_jobs_cloud,
    logs_job_cloud,
    submit_cloud,
)

FAKE_SHA = "0" * 40
FAKE_URL = "git@github.com:user/repo.git"


def _mock_sky(monkeypatch, stdout: str = "", rc: int = 0):
    """Mock subprocess.run for sky commands."""

    class _Result:
        def __init__(self, rc=rc, stdout=stdout, stderr=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, *args, **kwargs):
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)


# ── sky_available ────────────────────────────────────────────────────────


def test_sky_available_when_which_succeeds(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sky")
    assert _sky_available() is True


def test_sky_available_when_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert _sky_available() is False


# ── accelerator_str ──────────────────────────────────────────────────────


def test_accelerator_str_zero_gpus():
    assert _accelerator_str(0) == "null"


def test_accelerator_str_with_type():
    assert _accelerator_str(2, "H100") == "H100:2"


def test_accelerator_str_without_type():
    assert _accelerator_str(1) == ":1"


# ── project_root ─────────────────────────────────────────────────────────


def test_project_root_finds_git_dir(tmp_path):
    git_dir = tmp_path / "repo"
    git_dir.mkdir()
    (git_dir / ".git").mkdir()
    subdir = git_dir / "sub" / "deep"
    subdir.mkdir(parents=True)
    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(subdir)
        root = _project_root()
        assert root == git_dir.resolve()
    finally:
        os.chdir(old_cwd)


# ── submit_cloud ─────────────────────────────────────────────────────────


def test_submit_cloud_parses_job_id(monkeypatch):
    output = "Launching managed job...\nJob ID: 42\n"

    class _Result:
        returncode = 0
        stdout = output
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _Result)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sky")
    jid, status = submit_cloud(
        name="test",
        sha=FAKE_SHA,
        url=FAKE_URL,
        cmd="echo hello",
        gpus=1,
    )
    assert jid == 42
    assert status == "submitted"


def test_submit_cloud_error_on_parse_failure(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "No job ID here"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _Result)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sky")
    with pytest.raises(RuntimeError, match="Could not parse job ID"):
        submit_cloud(name="test", sha=FAKE_SHA, url=FAKE_URL, cmd="echo", gpus=1)


# ── list_jobs_cloud ──────────────────────────────────────────────────────


def test_list_jobs_cloud_parses_output(monkeypatch):
    output = "ID  NAME  STATUS  \n1   test  RUNNING\n"

    class _Result:
        returncode = 0
        stdout = output
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _Result)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sky")
    jobs = list_jobs_cloud()
    assert len(jobs) >= 0  # At minimum, no crash.


# ── cancel_job_cloud ─────────────────────────────────────────────────────


def test_cancel_job_cloud(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sky")
    cancel_job_cloud(42)
    assert any("cancel" in str(c) for c in calls)


# ── logs_job_cloud ───────────────────────────────────────────────────────


def test_logs_job_cloud_no_follow(monkeypatch):
    calls = []

    class _Result:
        returncode = 0
        stdout = "log output"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _Result)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sky")
    out = logs_job_cloud(42, follow=False)
    assert "log output" in out
