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
    tmux_calls = [c for c in fake_sky.calls if c and c[0] == "ssh"]
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
    r = runner.invoke(
        app,
        [
            "submit",
            "--dry-run",
            "-e",
            "WANDB_MODE=online",
            "-e",
            "FOO=bar",
            "echo",
            "hi",
        ],
    )
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


# ---------------------------------------------------------------------------
# submit routing + error paths
# ---------------------------------------------------------------------------


def test_submit_direct_backend_forced(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--direct", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0, r.output
    assert "backend=direct" in r.output


def test_submit_cloud_backend_forced(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--cloud", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0, r.output
    assert "backend=cloud" in r.output


def test_submit_direct_and_cloud_mutually_exclusive(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--direct", "--cloud", "--dry-run", "echo", "hi"])
    assert r.exit_code == 2


def test_submit_no_sync_flag(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--no-sync", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0, r.output
    # --no-sync is passed, dry-run shows backend info
    assert "backend=" in r.output


def test_submit_env_malformed(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--dry-run", "-e", "FOO", "echo", "hi"])
    assert r.exit_code == 2


# ---------------------------------------------------------------------------
# list / status / logs / cancel
# ---------------------------------------------------------------------------


def test_list_no_jobs(fake_sky, monkeypatch):
    from postdoc import direct as direct_mod

    monkeypatch.setattr(direct_mod, "list_jobs", lambda **kw: [])
    r = runner.invoke(app, ["list"])
    assert r.exit_code == 0
    assert "No jobs" in r.output


def test_status_not_found(fake_sky):
    r = runner.invoke(app, ["status", "nonexistent__999"])
    assert r.exit_code == 1


def test_logs_prints_tail(fake_sky):
    r = runner.invoke(app, ["logs", "job__1", "--no-follow", "--lines", "10"])
    assert r.exit_code == 0


def test_cancel_calls_direct_backend(fake_sky, monkeypatch):
    from postdoc import direct as direct_mod

    monkeypatch.setattr(direct_mod, "cancel_job", lambda name_and_id, **kw: True)
    r = runner.invoke(app, ["cancel", "job__1"])
    assert r.exit_code == 0
    assert "cancelled" in r.output.lower()


# ---------------------------------------------------------------------------
# check / probe commands
# ---------------------------------------------------------------------------


def test_check_probes_gpus(fake_sky, monkeypatch):
    from postdoc import direct as direct_mod
    from postdoc.direct import GPUInfo

    monkeypatch.setattr(
        direct_mod,
        "probe_gpus",
        lambda **kw: [GPUInfo(index=0, memory_used_mib=100, memory_total_mib=8000, utilization=5)],
    )
    r = runner.invoke(app, ["check"])
    assert r.exit_code == 0


def test_probe_is_alias_for_check(fake_sky, monkeypatch):
    from postdoc import direct as direct_mod

    monkeypatch.setattr(direct_mod, "probe_gpus", lambda **kw: [])
    r = runner.invoke(app, ["probe"])
    assert r.exit_code == 0


# ---------------------------------------------------------------------------
# backward-compat stubs
# ---------------------------------------------------------------------------


def test_pool_down_emits_noop(fake_sky):
    r = runner.invoke(app, ["pool-down"])
    assert r.exit_code == 0


def test_cluster_down_emits_noop(fake_sky):
    r = runner.invoke(app, ["cluster-down"])
    assert r.exit_code == 0


def test_queue_stub_prints_help(fake_sky):
    r = runner.invoke(app, ["queue"])
    assert r.exit_code == 1
