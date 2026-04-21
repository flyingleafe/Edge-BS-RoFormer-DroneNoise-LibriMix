import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from postdoc.cli import app

runner = CliRunner()


@pytest.fixture
def cli_env(sample_postdoc_yaml, sample_experiment_yaml, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # sample_postdoc_yaml is already inside tmp_path; ensure it's named postdoc.yaml
    target = tmp_path / "postdoc.yaml"
    if not target.exists():
        import shutil
        shutil.copy(sample_postdoc_yaml, target)
    return tmp_path, sample_experiment_yaml


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_job_list_empty(cli_env):
    result = runner.invoke(app, ["job", "list"])
    assert result.exit_code == 0
    assert "No jobs" in result.stdout


@patch("postdoc.cli.subprocess.run")
def test_job_submit_creates_job(mock_git, cli_env):
    tmp_path, exp_yaml = cli_env
    mock_git.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
    result = runner.invoke(app, ["job", "submit", str(exp_yaml)])
    assert result.exit_code == 0
    assert "submitted" in result.stdout.lower() or "queued" in result.stdout.lower()


def test_job_status_nonexistent(cli_env):
    result = runner.invoke(app, ["job", "status", "nonexistent"])
    assert result.exit_code != 0 or "not found" in result.stdout.lower()


def test_results_show_nonexistent(cli_env):
    result = runner.invoke(app, ["results", "show", "nonexistent"])
    assert result.exit_code != 0 or "not found" in result.stdout.lower()
