"""Unit tests for the harness cores of ``scripts/bench.py`` and
``scripts/probe_ckpt.py`` (no GPU, no network, no model builds)."""

from __future__ import annotations

from pathlib import Path

import bench
import probe_ckpt
import pytest
import torch


# ── bench harness ───────────────────────────────────────────────────────────
def test_timeit_counts_calls_and_returns_ms() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1

    ms = bench.timeit(fn, torch.device("cpu"), iters=5, warmup=2)
    assert calls["n"] == 7
    assert isinstance(ms, float)
    assert ms >= 0.0


def test_shape_parsing_default_and_override() -> None:
    assert bench._shape(None, (32, 250)) == (32, 250)
    assert bench._shape("4,8", (32, 250)) == (4, 8)
    with pytest.raises(SystemExit):
        bench._shape("4,8,16", (32, 250))


def test_target_registry_is_complete() -> None:
    assert set(bench.TARGETS) == {"ckla_scan", "cqt", "grouped_branches", "noise_gen"}


# ── probe_ckpt ──────────────────────────────────────────────────────────────
def test_parse_ref_zoo_forms() -> None:
    exp, uri = probe_ckpt.parse_ref("zoo:gen_w4_lik_wind_mm")
    assert exp == "gen_w4_lik_wind_mm"
    assert uri == "r2://ml-data/artifacts/gen_w4_lik_wind_mm/checkpoints/best.ckpt"
    exp, uri = probe_ckpt.parse_ref("zoo:exp1/last.ckpt")
    assert (exp, uri) == ("exp1", "r2://ml-data/artifacts/exp1/checkpoints/last.ckpt")


def test_parse_ref_passthrough_and_errors() -> None:
    assert probe_ckpt.parse_ref("results/x/best.ckpt") == (None, "results/x/best.ckpt")
    assert probe_ckpt.parse_ref("r2://b/k") == (None, "r2://b/k")
    with pytest.raises(ValueError):
        probe_ckpt.parse_ref("zoo:")


def test_load_state_unwraps_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    state = {
        "gen.wind.raw_gate": torch.tensor([-2.0]),
        "gen.core.weight": torch.randn(4, 6),
    }
    path = tmp_path / "ck.ckpt"
    torch.save({"state_dict": state}, path)

    loaded = probe_ckpt.load_state(str(path))
    assert set(loaded) == set(state)

    probe_ckpt.report_params(loaded, ".wind.")
    out = capsys.readouterr().out
    assert "raw_gate" in out
    assert "softplus" in out
    assert "core.weight" not in out  # filtered out

    probe_ckpt.report_spectra(loaded, "core")
    out = capsys.readouterr().out
    assert "gen.core.weight" in out
