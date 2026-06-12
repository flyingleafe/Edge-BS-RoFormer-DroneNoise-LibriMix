"""Tests for `utils.paths` — central path resolution."""

from __future__ import annotations

from pathlib import Path


def test_get_data_path_with_subpath(monkeypatch):
    from utils.paths import get_data_path

    monkeypatch.setattr("utils.paths._DATA_ROOT", Path("/fake/root"))
    assert get_data_path("DREGON") == Path("/fake/root/data/DREGON")
    assert get_data_path() == Path("/fake/root/data")
    # Reset cache to avoid polluting other tests.
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)


def test_get_datasets_path_no_subpath(monkeypatch):
    from utils.paths import get_datasets_path

    monkeypatch.setattr("utils.paths._DATA_ROOT", Path("/fake/root"))
    assert get_datasets_path() == Path("/fake/root/datasets")
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)


def test_get_results_path_with_subpath(monkeypatch):
    from utils.paths import get_results_path

    monkeypatch.setattr("utils.paths._DATA_ROOT", Path("/fake/root"))
    assert get_results_path("eval") == Path("/fake/root/results/eval")
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)


def test_data_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    # Force re-resolution.
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)
    from utils.paths import get_data_root

    assert get_data_root() == tmp_path.resolve()


def test_data_root_fallback_to_git_worktree(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)

    main_path = tmp_path / "main"
    main_path.mkdir()
    (main_path / ".git").mkdir()  # Simulate a git repo.

    def fake_run(args, **kwargs):
        if args[0] == "git" and args[1] == "worktree":
            # Return mock worktree list output.
            result = subprocess.CompletedProcess(
                args, 0, stdout=f"{main_path} abcdef [main]\n", stderr=""
            )
            return result
        raise AssertionError(f"Unexpected subprocess call: {args}")

    monkeypatch.setattr("subprocess.run", fake_run)
    from utils.paths import get_data_root

    monkeypatch.setattr("utils.paths._DATA_ROOT", None)
    assert get_data_root() == main_path.resolve()
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)


def test_data_root_fallback_to_git_rev_parse(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)

    def fake_run(args, **kwargs):
        if args[0] == "git":
            if args[1] == "worktree":
                # Simulate worktree failure.
                raise subprocess.CalledProcessError(1, args)
            elif args[1] == "rev-parse":
                return subprocess.CompletedProcess(args, 0, stdout=f"{tmp_path}\n", stderr="")
        raise AssertionError(f"Unexpected subprocess call: {args}")

    monkeypatch.setattr("subprocess.run", fake_run)
    from utils.paths import get_data_root

    monkeypatch.setattr("utils.paths._DATA_ROOT", None)
    assert get_data_root() == tmp_path.resolve()
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)


def test_data_root_fallback_to_cwd(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)

    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("utils.paths.Path.cwd", lambda: tmp_path.resolve())
    from utils.paths import get_data_root

    monkeypatch.setattr("utils.paths._DATA_ROOT", None)
    assert get_data_root() == tmp_path.resolve()
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)


def test_data_root_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)
    from utils.paths import get_data_root

    r1 = get_data_root()
    # Clear env but cache should persist.
    monkeypatch.delenv("DATA_ROOT", raising=False)
    r2 = get_data_root()
    assert r1 == r2 == tmp_path.resolve()
    monkeypatch.setattr("utils.paths._DATA_ROOT", None)
