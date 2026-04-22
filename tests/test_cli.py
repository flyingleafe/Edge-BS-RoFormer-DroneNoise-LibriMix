"""CLI tests — verify we shell out to the right `sky` invocations.

These tests do *not* call SkyPilot. They mock subprocess and assert the argv
we construct. That's the whole surface of our CLI on top of SkyPilot.
"""
from __future__ import annotations

from typer.testing import CliRunner

from postdoc.cli import app


runner = CliRunner()


def test_submit_generates_task_and_calls_sky_jobs_launch(fake_sky, fake_git):
    result = runner.invoke(app, ["submit", "python", "train.py", "--x", "1"])
    assert result.exit_code == 0, result.output
    # Git preflight ran once.
    assert len(fake_git.calls) == 1
    assert fake_git.calls[0][0] == "snapshot"
    # Only one sky call expected.
    sky_calls = [c for c in fake_sky.calls if c and c[0] == "sky"]
    assert len(sky_calls) == 1
    argv = sky_calls[0]
    assert argv[:3] == ["sky", "jobs", "launch"]
    assert "-y" in argv
    assert "--detach-run" in argv
    # Last arg is the generated task-yaml path.
    assert argv[-1].endswith(".sky.yaml")


def test_submit_with_file_passes_through(fake_sky, fake_git, tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("name: x\nrun: echo hi\n")
    result = runner.invoke(app, ["submit", "-f", str(task)])
    assert result.exit_code == 0, result.output
    # --file bypasses the git wrapper entirely.
    assert fake_git.calls == []
    sky_calls = [c for c in fake_sky.calls if c and c[0] == "sky"]
    assert sky_calls[0] == ["sky", "jobs", "launch", "-y", "--detach-run", str(task)]


def test_submit_dry_run_prints_yaml_no_sky(fake_sky, fake_git):
    result = runner.invoke(app, ["submit", "--dry-run", "echo", "hi"])
    assert result.exit_code == 0
    # Git preflight still runs (push) before the dry-run returns.
    assert len(fake_git.calls) == 1
    assert "echo hi" in result.output
    assert "POSTDOC_GIT_SHA" in result.output
    # No sky calls at all.
    assert not [c for c in fake_sky.calls if c and c[0] == "sky"]


def test_submit_without_command_errors(fake_sky, fake_git):
    result = runner.invoke(app, ["submit"])
    assert result.exit_code != 0
    assert "no command" in result.output.lower()


def test_submit_with_both_file_and_command_errors(fake_sky, fake_git, tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("run: echo hi\n")
    result = runner.invoke(app, ["submit", "-f", str(task), "echo", "hi"])
    assert result.exit_code != 0


def test_list_calls_sky_jobs_queue(fake_sky):
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky[:3] == ["sky", "jobs", "queue"]
    assert "--refresh" in sky
    assert "--all" not in sky


def test_list_all_adds_flag(fake_sky):
    runner.invoke(app, ["list", "--all"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert "--all" in sky


def test_logs_follows_by_default(fake_sky):
    runner.invoke(app, ["logs", "42"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky[:3] == ["sky", "jobs", "logs"]
    assert "42" in sky
    assert "--no-follow" not in sky


def test_logs_controller_flag(fake_sky):
    runner.invoke(app, ["logs", "42", "--controller", "--no-follow"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert "--controller" in sky
    assert "--no-follow" in sky


def test_cancel_ids(fake_sky):
    runner.invoke(app, ["cancel", "1", "2", "3", "-y"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky[:3] == ["sky", "jobs", "cancel"]
    assert "-y" in sky
    assert "1" in sky and "2" in sky and "3" in sky


def test_cancel_all(fake_sky):
    runner.invoke(app, ["cancel", "--all", "-y"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert "--all" in sky


def test_cancel_without_ids_or_all_errors(fake_sky):
    result = runner.invoke(app, ["cancel"])
    assert result.exit_code != 0


def test_ssh_calls_plain_ssh(fake_sky):
    runner.invoke(app, ["ssh"])
    # No sky call
    assert not any(c for c in fake_sky.calls if c and c[0] == "sky")
    ssh = [c for c in fake_sky.calls if c and c[0] == "ssh"][-1]
    assert ssh[0] == "ssh"
    # Pool name default == host alias
    assert "vast-server" in ssh


def test_pool_up(fake_sky):
    runner.invoke(app, ["pool-up"])
    sky = [c for c in fake_sky.calls if c and c[0] == "sky"][-1]
    assert sky == ["sky", "ssh", "up"]


def test_submit_env_vars_serialize_into_task(fake_sky, fake_git):
    # --dry-run prints the YAML; use that to inspect envs.
    result = runner.invoke(app, [
        "submit", "--dry-run",
        "-e", "WANDB_MODE=online", "-e", "FOO=bar",
        "echo", "hi",
    ])
    assert result.exit_code == 0
    assert "WANDB_MODE: online" in result.output
    assert "FOO: bar" in result.output
    # Git envs are injected automatically.
    assert "POSTDOC_GIT_SHA" in result.output
    assert "POSTDOC_GIT_URL" in result.output


def test_submit_dirty_flag_propagates_to_preflight(fake_sky, fake_git):
    result = runner.invoke(app, ["submit", "--dirty", "--dry-run", "echo", "hi"])
    assert result.exit_code == 0
    _, kwargs = fake_git.calls[0]
    assert kwargs["allow_dirty"] is True
    assert "WARNING" in result.output or "dirty" in result.output.lower()


def test_submit_skip_push_flag_propagates(fake_sky, fake_git):
    result = runner.invoke(app, ["submit", "--skip-push", "--dry-run", "echo", "hi"])
    assert result.exit_code == 0
    _, kwargs = fake_git.calls[0]
    assert kwargs["skip_push"] is True


def test_submit_remote_flag_propagates(fake_sky, fake_git):
    result = runner.invoke(app, ["submit", "--remote", "upstream", "--dry-run", "echo", "hi"])
    assert result.exit_code == 0
    _, kwargs = fake_git.calls[0]
    assert kwargs["remote"] == "upstream"


def test_submit_git_error_exits_nonzero(fake_sky, monkeypatch):
    from postdoc import git_state

    def _boom(**_kw):
        raise git_state.GitError("tree is dirty")

    monkeypatch.setattr(git_state, "snapshot", _boom)
    result = runner.invoke(app, ["submit", "python", "train.py"])
    assert result.exit_code == 3
    assert "tree is dirty" in result.output
