"""CLI tests — verify we shell out to the right `sky` invocations.

Mocks `subprocess.run` via the `fake_sky` fixture and `git_state.snapshot`
via `fake_git`.
"""
from __future__ import annotations

from typer.testing import CliRunner

from postdoc.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# cluster lifecycle
# ---------------------------------------------------------------------------

def test_cluster_up_calls_sky_launch(fake_sky):
    r = runner.invoke(app, ["cluster-up"])
    assert r.exit_code == 0, r.output
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky[:4] == ["sky", "launch", "-y", "-c"]
    assert "postdoc" in sky
    assert sky[-1].endswith(".sky.yaml")


def test_cluster_up_dry_run_emits_hostpath(fake_sky):
    r = runner.invoke(app, ["cluster-up", "--dry-run"])
    assert r.exit_code == 0
    assert "hostPath" in r.output
    assert "harmonic-noise-suppression" in r.output
    # No sky invocation at all.
    assert not [c for c in fake_sky.calls if c and c[0] == "sky"]


def test_cluster_down(fake_sky):
    runner.invoke(app, ["cluster-down", "-y"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky == ["sky", "down", "postdoc", "-y"]


def test_cluster_status(fake_sky):
    runner.invoke(app, ["cluster-status"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky == ["sky", "status", "postdoc"]


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def _cluster_running(fake_sky):
    """Make `sky status postdoc` report the cluster as present."""
    fake_sky.stdout_for = ("sky status postdoc", "postdoc  ssh/vast-server  UP")
    return fake_sky


def test_submit_calls_sky_exec(fake_sky, fake_git):
    _cluster_running(fake_sky)
    r = runner.invoke(app, ["submit", "python", "train.py"])
    assert r.exit_code == 0, r.output
    # Preflight ran.
    assert fake_git.calls and fake_git.calls[0][0] == "snapshot"
    # Last sky call is `sky exec`.
    sky_calls = [c for c in fake_sky.calls if c and c[0] == "sky"]
    exec_call = [c for c in sky_calls if len(c) >= 3 and c[1] == "exec"][-1]
    assert exec_call[:4] == ["sky", "exec", "postdoc", "-d"]
    assert exec_call[-1].endswith(".sky.yaml")


def test_submit_dry_run_prints_yaml_no_sky_exec(fake_sky, fake_git):
    r = runner.invoke(app, ["submit", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0
    # Preflight still ran.
    assert len(fake_git.calls) == 1
    # No `sky exec` should have been called.
    sky_calls = [c for c in fake_sky.calls if c and c[0] == "sky"]
    assert not [c for c in sky_calls if len(c) >= 2 and c[1] == "exec"]
    # YAML has no infra (sky exec ignores it) but does have accelerators + git envs.
    assert "accelerators:" in r.output
    assert ":1" in r.output
    assert "POSTDOC_GIT_SHA" in r.output
    assert "git reset --hard" in r.output
    assert "infra:" not in r.output


def test_submit_without_command_errors(fake_sky, fake_git):
    r = runner.invoke(app, ["submit"])
    assert r.exit_code != 0
    assert "no command" in r.output.lower()


def test_submit_dirty_flag_propagates_to_preflight(fake_sky, fake_git):
    _cluster_running(fake_sky)
    r = runner.invoke(app, ["submit", "--dirty", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0
    _, kwargs = fake_git.calls[0]
    assert kwargs["allow_dirty"] is True
    assert "WARNING" in r.output or "dirty" in r.output.lower()


def test_submit_skip_push_propagates(fake_sky, fake_git):
    _cluster_running(fake_sky)
    r = runner.invoke(app, ["submit", "--skip-push", "--dry-run", "echo", "hi"])
    assert r.exit_code == 0
    _, kwargs = fake_git.calls[0]
    assert kwargs["skip_push"] is True


def test_submit_auto_ups_cluster_when_missing(fake_sky, fake_git):
    """When the cluster isn't present, submit should trigger `sky launch` first."""
    # No cluster: `sky status postdoc` returns non-zero.
    fake_sky.rc_for = ("sky status postdoc", 1)
    r = runner.invoke(app, ["submit", "python", "train.py"])
    assert r.exit_code == 0, r.output
    sky_calls = [c for c in fake_sky.calls if c and c[0] == "sky"]
    kinds = [c[1] for c in sky_calls if len(c) >= 2]
    # Must launch before exec.
    assert "launch" in kinds and "exec" in kinds
    assert kinds.index("launch") < kinds.index("exec")


def test_submit_no_auto_up_skips_launch(fake_sky, fake_git):
    fake_sky.rc_for = ("sky status postdoc", 1)
    r = runner.invoke(app, ["submit", "--no-auto-up", "python", "train.py"])
    assert r.exit_code == 0
    sky_calls = [c for c in fake_sky.calls if c and c[0] == "sky"]
    kinds = [c[1] for c in sky_calls if len(c) >= 2]
    assert "launch" not in kinds


def test_submit_env_vars_serialize(fake_sky, fake_git):
    r = runner.invoke(app, [
        "submit", "--dry-run",
        "-e", "WANDB_MODE=online", "-e", "FOO=bar", "echo", "hi",
    ])
    assert r.exit_code == 0
    assert "WANDB_MODE: online" in r.output
    assert "FOO: bar" in r.output


# ---------------------------------------------------------------------------
# queue / logs / cancel
# ---------------------------------------------------------------------------

def test_list(fake_sky):
    runner.invoke(app, ["list"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky[:3] == ["sky", "queue", "postdoc"]
    assert "--all" not in sky


def test_list_all(fake_sky):
    runner.invoke(app, ["list", "--all"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert "--all" in sky


def test_logs_follows_by_default(fake_sky):
    runner.invoke(app, ["logs", "42"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky[:3] == ["sky", "logs", "postdoc"]
    assert "42" in sky
    assert "--no-follow" not in sky


def test_logs_no_follow(fake_sky):
    runner.invoke(app, ["logs", "42", "--no-follow"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert "--no-follow" in sky


def test_cancel_ids(fake_sky):
    runner.invoke(app, ["cancel", "1", "2", "3", "-y"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky[:3] == ["sky", "cancel", "postdoc"]
    assert "-y" in sky and "1" in sky and "2" in sky and "3" in sky


def test_cancel_all(fake_sky):
    runner.invoke(app, ["cancel", "--all", "-y"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert "--all" in sky


def test_cancel_without_ids_or_all_errors(fake_sky):
    r = runner.invoke(app, ["cancel"])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# ssh / dashboard / check / pool-up
# ---------------------------------------------------------------------------

def test_ssh_uses_plain_ssh(fake_sky):
    runner.invoke(app, ["ssh"])
    ssh = [c for c in fake_sky.calls if c and c[0] == "ssh"][-1]
    assert "vast-server" in ssh


def test_pool_up(fake_sky):
    runner.invoke(app, ["pool-up"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky == ["sky", "ssh", "up"]


def test_pool_down(fake_sky):
    runner.invoke(app, ["pool-down"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky == ["sky", "ssh", "down"]


def test_git_error_exits_nonzero(fake_sky, monkeypatch):
    from postdoc import git_state

    def _boom(**_kw):
        raise git_state.GitError("tree is dirty")

    monkeypatch.setattr(git_state, "snapshot", _boom)
    r = runner.invoke(app, ["submit", "python", "train.py"])
    assert r.exit_code == 3
    assert "tree is dirty" in r.output
