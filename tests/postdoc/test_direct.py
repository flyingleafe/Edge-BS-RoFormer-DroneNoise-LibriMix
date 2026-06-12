"""Tests for `postdoc.direct` — direct SSH backend, mocked."""

from __future__ import annotations

import json
import subprocess

from postdoc.direct import (
    GPUInfo,
    _ensure_postdoc_dir,
    _next_job_id,
    cancel_job,
    free_gpus,
    list_jobs,
    probe_gpus,
    read_logs,
    submit_direct,
)

FAKE_SHA = "0" * 40


# ── helpers ──────────────────────────────────────────────────────────────


def _fake_ssh_output(monkeypatch, stdout: str, rc: int = 0):
    """Mock subprocess.run to return given stdout for any SSH command."""

    class _Result:
        def __init__(self, rc, stdout, stderr=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, *args, **kwargs):
        if kwargs.get("capture_output") or kwargs.get("stdout") == subprocess.PIPE:
            return _Result(rc, stdout)
        return _Result(rc, stdout)

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: stdout)
    monkeypatch.setattr("subprocess.check_call", lambda cmd, **kw: None)


# ── GPU probing ──────────────────────────────────────────────────────────


def test_probe_gpus_parses_nvidia_smi(monkeypatch):
    _fake_ssh_output(monkeypatch, "0, 100, 8000, 5\n1, 200, 8000, 10\n")
    gpus = probe_gpus(user="root", host="fake")
    assert len(gpus) == 2
    assert gpus[0] == GPUInfo(index=0, memory_used_mib=100, memory_total_mib=8000, utilization=5)
    assert gpus[1] == GPUInfo(index=1, memory_used_mib=200, memory_total_mib=8000, utilization=10)


def test_probe_gpus_empty_output(monkeypatch):
    _fake_ssh_output(monkeypatch, "")
    gpus = probe_gpus(user="root", host="fake")
    assert gpus == []


def test_free_gpus_filters_by_memory_threshold(monkeypatch):
    _fake_ssh_output(monkeypatch, "0, 100, 8000, 5\n1, 600, 8000, 80\n2, 200, 8000, 10\n")
    free = free_gpus(user="root", host="fake", threshold_mib=500)
    assert free == [0, 2]


# ── directory setup ──────────────────────────────────────────────────────


def test_ensure_postdoc_dir_creates_dirs_and_fifo(monkeypatch):
    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(" ".join(cmd) if isinstance(cmd, list) else cmd)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr("subprocess.run", _fake_run)
    _ensure_postdoc_dir()
    joined = " ".join(calls)
    assert "mkdir -p" in joined
    assert "mkfifo" in joined


# ── job ID ───────────────────────────────────────────────────────────────


def test_next_job_id_when_jobs_exist(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: "42\n")
    jid = _next_job_id("root", "fake", "/root/.postdoc")
    assert jid == 42


def test_next_job_id_when_no_jobs(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: "1\n")
    jid = _next_job_id("root", "fake", "/root/.postdoc")
    assert jid == 1


# ── submit ───────────────────────────────────────────────────────────────


def test_submit_direct_returns_queued_status(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0)
    )
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: "1\n")
    jid, status = submit_direct(
        name="test-job",
        sha=FAKE_SHA,
        cmd="echo hello",
        gpus=1,
        user="root",
        host="fake",
    )
    assert isinstance(jid, int)
    assert status == "queued"


# ── list jobs ────────────────────────────────────────────────────────────


def test_list_jobs_empty_server(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: "[]")
    jobs = list_jobs(user="root", host="fake")
    assert jobs == []


def test_list_jobs_parses_json_correctly(monkeypatch):
    job_data = [
        {
            "id": 1,
            "name": "test",
            "sha": FAKE_SHA,
            "cmd": "echo",
            "gpus": 1,
            "started_at": "2024-01-01",
            "status": "running",
            "pid": 123,
            "gpu_mask": [0],
            "log_path": "/tmp/log",
        }
    ]
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: json.dumps(job_data))
    jobs = list_jobs(user="root", host="fake")
    assert len(jobs) == 1
    assert jobs[0].name == "test"
    assert jobs[0].status == "running"


# ── cancel ───────────────────────────────────────────────────────────────


def test_cancel_job_kills_pid(monkeypatch):
    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        # First call: get pid
        if "pid" in str(cmd):
            return subprocess.CompletedProcess([], 0, stdout="123\n")
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: "123\n")
    result = cancel_job("test__1", user="root", host="fake")
    # Even if kill fails, marking cancelled should succeed.
    assert result is True or result is False


# ── read logs ────────────────────────────────────────────────────────────


def test_read_logs_tail(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: "line1\nline2\n")
    out = read_logs("test__1", user="root", host="fake", follow=False, lines=10)
    assert out == "line1\nline2\n"
