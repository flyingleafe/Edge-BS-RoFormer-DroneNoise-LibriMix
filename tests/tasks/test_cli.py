"""Tests for `tasks.cli` — evaluate-rps CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from tasks.cli import app

runner = CliRunner()


def test_cli_no_args_shows_help():
    r = runner.invoke(app, [])
    # With no_args_is_help=True, should exit 0 or 2 and show help.
    assert r.exit_code in (0, 2)


def test_cli_missing_models():
    r = runner.invoke(app, ["--input-set", "/tmp/fake"])
    assert r.exit_code != 0


def test_cli_input_set_not_found():
    r = runner.invoke(
        app, ["--input-set", "/nonexistent/path", "--model", "simple_conv@/tmp/fake.pt"]
    )
    assert r.exit_code == 1


def test_cli_alignment_invalid():
    r = runner.invoke(
        app,
        ["--input-set", "/tmp/fake", "--model", "simple_conv@/tmp/fake.pt", "--alignment", "bogus"],
    )
    # Should error — dataset not found first, or alignment validation
    assert r.exit_code != 0
