"""CLI tests — verify correct SSH/sky calls for the new direct-SSH backend."""
from __future__ import annotations

from typer.testing import CliRunner

from postdoc.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# queue daemon
# ---------------------------------------------------------------------------

def test_queue_start_starts_tmux(fake_sky):
    r = runner.invoke(app, ["queue-start"])
    assert r.exit_code == 0, r.output
    tmux_calls = [c for c in fake_sky.calls
                  if c and c[0] == "ssh"]
    assert any("tmux new-session" in " ".join(c) for c in tmux_calls)


def test_queue_stop_kills_tmux(fake_sky):
    r = runner.invoke(app, ["queue-stop"])
    assert r.exit_code == 0
    ssh_calls = [c for c in fake_sky.calls if c and c[0] == "ssh"]
    assert any("tmux kill-session" in " ".join(c) for c in ssh_calls)


def test_queue_status_checks_tmux(fake_sky):
    r = runner.invoke(app, ["queue-status"])
    assert r.exit_code == 0
    ssh_calls = [c for c in fake_sky.calls if c and c[0] == "ssh"]
    assert any("tmux has-session" in " ".join(c) for c in ssh_calls)


# ---------------------------------------------------------------------------
# submit — preflight only (--dry-run avoids backend calls)
# ---------------------------------------------------------------------------

def test_submit_without_command_errors(fake_sky, fake_git):
    r = runner.invoke(app, ["submit"])
    assert r.exit_code != 0
    assert "no command" in r.output.lower()


def test_submit_dirty_flag_propagates_to_preflight(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--dirty", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0, r.output
    _, kwargs = fake_git.calls[0]
    assert kwargs["allow_dirty"] is True
    assert "WARNING" in r.output or "dirty" in r.output.lower()


def test_submit_skip_push_propagates(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--skip-push", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0, r.output
    _, kwargs = fake_git.calls[0]
    assert kwargs["skip_push"] is True


def test_submit_env_vars_passed(fake_sky, fake_git):
    r = runner.invoke(app, [
        "submit", "--dry-run",
        "-e", "WANDB_MODE=online", "-e", "FOO=bar", "echo", "hi",
    ])
    assert r.exit_code == 0, r.output
    # dry-run output includes backend + command
    assert "backend=cloud" in r.output
    assert "echo hi" in r.output


# ---------------------------------------------------------------------------
# git error exits non-zero
# ---------------------------------------------------------------------------

def test_git_error_exits_nonzero(fake_sky, monkeypatch):
    from postdoc import git_state

    def _boom(**_kw):
        raise git_state.GitError("tree is dirty")

    monkeypatch.setattr(git_state, "snapshot", _boom)
    r = runner.invoke(app, ["submit", "python", "train.py"])
    assert r.exit_code == 3
    assert "tree is dirty" in r.output


# ---------------------------------------------------------------------------
# ssh utility
# ---------------------------------------------------------------------------

def test_ssh_uses_plain_ssh(fake_sky):
    r = runner.invoke(app, ["ssh"])
    assert r.exit_code == 0
    ssh_calls = [c for c in fake_sky.calls if c and c[0] == "ssh"]
    assert any("vast-server" in " ".join(c) for c in ssh_calls)


# ---------------------------------------------------------------------------
# backwards-compat stubs emit helpful errors
# ---------------------------------------------------------------------------

def test_cluster_up_emits_migration_error(fake_sky):
    r = runner.invoke(app, ["cluster-up"])
    assert r.exit_code == 1
    assert "queue-start" in r.output


def test_pool_up_emits_migration_error(fake_sky):
    r = runner.invoke(app, ["pool-up"])
    assert r.exit_code == 1
    assert "pool-up" in r.output.lower()


def test_dashboard_emits_migration_error(fake_sky):
    r = runner.invoke(app, ["dashboard"])
    assert r.exit_code == 1
    assert "Ray dashboard" in r.output or "ssh" in r.output.lower()
