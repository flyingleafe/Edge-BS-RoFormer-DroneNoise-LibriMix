"""Loop smoke test: tiny synthetic map-style dataset + 2-layer model on the
rps_prediction task, 2 epochs CPU, wandb mocked out entirely.

``wandb.init`` reassigns ``wandb.log``/``wandb.run`` internally when a real
run starts, which silently defeats a plain ``monkeypatch.setattr(wandb,
"log", ...)`` done *before* ``run_training`` calls ``wandb.init`` — so this
mocks the ``wandb`` name binding inside ``training.loop`` itself, replacing
it with a tiny recording stub for the duration of each test.
"""

from __future__ import annotations

import math

import training.loop as loop_module
from tests.training.conftest import make_tiny_config
from training.loop import run_training


class _FakeRun:
    id = "fake-run-id"


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def init(self, *args, **kwargs):
        return _FakeRun()

    def log(self, data, *args, **kwargs):
        self.logged.append(dict(data))

    def finish(self, *args, **kwargs):
        pass


def test_run_training_writes_checkpoints_and_produces_finite_loss(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name="tiny_loop",
        epochs=2,
        n_train=6,
        n_valid=4,
        batch_size=2,
    )
    result = run_training(cfg)

    run_dir = tmp_path / "tiny_loop"
    assert (run_dir / "best.ckpt").is_file()

    train_losses = [row["train/loss"] for row in fake_wandb.logged if "train/loss" in row]
    assert len(train_losses) == 2  # one per epoch
    assert all(math.isfinite(v) for v in train_losses)

    assert "best_mse" in result
    assert math.isfinite(result["best_mse"])
    assert math.isfinite(result["final_epoch"])


def test_run_training_refuses_to_overwrite_a_nonempty_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())

    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_loop2", epochs=1)
    run_training(cfg)

    try:
        run_training(cfg)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError on a second run without resume=true")
