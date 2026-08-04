"""Tests for `tasks.cli` — the live ``evaluate-rps`` console script.

``evaluate-rps`` is declared in ``pyproject.toml`` ``[project.scripts]`` and is
the documented replacement for the deleted per-question eval scripts
(``docs/migration.md``), so this file asserts real behaviour: exact exit codes,
the error text each failure mode prints, and one end-to-end run over a
two-sample synthetic input set that checks the JSON and LaTeX artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torchaudio
from typer.testing import CliRunner

from tasks.cli import app

runner = CliRunner()

SR = 16000


def _make_input_set(root, n_samples=2, duration_s=1.0):
    """Write a minimal DREGON-LM-style split: ``sample_XXXXX/{mixture.wav,rps.npy}``."""
    root.mkdir(parents=True, exist_ok=True)
    n = int(SR * duration_s)
    rng = np.random.default_rng(0)
    meta = {}
    t = np.arange(n) / SR
    for i in range(n_samples):
        sid = f"sample_{i:05d}"
        d = root / sid
        d.mkdir()
        rps = 60.0 + 5.0 * np.arange(4)[:, None] + np.zeros((4, 128))
        # A two-harmonic comb per rotor, so the audio is nominally on-task.
        wav = np.zeros(n)
        for r in rps[:, 0]:
            for k in (1, 2):
                wav += 0.05 * np.sin(2 * np.pi * k * r * t)
        torchaudio.save(
            str(d / "mixture.wav"),
            torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
            SR,
        )
        np.save(d / "rps.npy", rps + rng.normal(0, 0.1, rps.shape))
        meta[sid] = {"id": sid, "input_snr": -10.0 + i}
    (root / "metadata.json").write_text(json.dumps(meta))
    return root


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A randomly-initialised ``simple_conv`` state_dict on disk."""
    from models.registry import build_model

    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = build_model("simple_conv", n_fft=2048, hop_length=512, num_rotors=4)
    torch.save(model.state_dict(), path)
    return path


# ── argument handling ────────────────────────────────────────────────────


def test_cli_no_args_shows_usage():
    r = runner.invoke(app, [])
    # no_args_is_help=True: click's "no arguments" exit code plus the usage text.
    assert r.exit_code == 2
    assert "Usage: evaluate-rps" in r.output
    assert "--input-set" in r.output


def test_cli_missing_models_names_the_option():
    r = runner.invoke(app, ["--input-set", "/tmp/fake"])
    assert r.exit_code == 2
    assert "Missing option" in r.output
    assert "--model" in r.output


def test_cli_input_set_not_found():
    r = runner.invoke(
        app, ["--input-set", "/nonexistent/path", "--model", "simple_conv@/tmp/fake.pt"]
    )
    assert r.exit_code == 1
    assert "input set not found: /nonexistent/path" in r.output


def test_cli_checkpoint_not_found(tmp_path):
    """The input set is validated first, then the model spec — a real
    directory with a bogus checkpoint must fail in ``load_predictor``."""
    _make_input_set(tmp_path / "valid", n_samples=1)
    r = runner.invoke(
        app,
        ["--input-set", str(tmp_path / "valid"), "--model", f"simple_conv@{tmp_path}/nope.pt"],
    )
    assert r.exit_code != 0
    assert isinstance(r.exception, FileNotFoundError)


def test_cli_alignment_invalid(tmp_path, ckpt):
    _make_input_set(tmp_path / "valid", n_samples=1)
    r = runner.invoke(
        app,
        [
            "--input-set",
            str(tmp_path / "valid"),
            "--model",
            f"simple_conv@{ckpt}",
            "--alignment",
            "bogus",
            "--quiet",
        ],
    )
    assert r.exit_code != 0
    assert isinstance(r.exception, ValueError)
    assert "bogus" in str(r.exception)


# ── end to end ───────────────────────────────────────────────────────────


def test_cli_end_to_end_writes_json_and_tex(tmp_path, ckpt):
    out = tmp_path / "out" / "results.json"
    tex = tmp_path / "out" / "table.tex"
    _make_input_set(tmp_path / "valid", n_samples=2)
    r = runner.invoke(
        app,
        [
            "--input-set",
            str(tmp_path / "valid"),
            "--model",
            f"simple_conv@{ckpt}",
            "--output",
            str(out),
            "--tex",
            str(tex),
            "--quiet",
        ],
    )
    assert r.exit_code == 0, r.output + str(r.exception)

    data = json.loads(out.read_text())
    assert data["models"] == [f"simple_conv@{ckpt}"]
    assert data["input_set"] == str(tmp_path / "valid")
    assert data["alignment"] == "stft_timestamps"
    (agg,) = data["results"]
    for key in ("mse", "rmse", "mae_clip", "r2_mean"):
        assert np.isfinite(agg[key]), key
    (per_sample,) = data["per_sample"]
    assert len(per_sample) == 2

    body = tex.read_text()
    assert body.startswith(r"\begin{tabular}")
    assert r"\end{tabular}" in body


def test_cli_verbose_prints_a_summary_line(tmp_path, ckpt):
    _make_input_set(tmp_path / "valid", n_samples=1)
    r = runner.invoke(
        app, ["--input-set", str(tmp_path / "valid"), "--model", f"simple_conv@{ckpt}"]
    )
    assert r.exit_code == 0, r.output + str(r.exception)
    assert "Loading predictor: simple_conv@" in r.output
    assert "MSE=" in r.output and "RMSE=" in r.output
