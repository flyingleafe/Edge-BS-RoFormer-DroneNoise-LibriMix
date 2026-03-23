import pytest
from pathlib import Path
from postdoc.experiment import load_experiment, resolve_config, build_train_args, build_eval_args


def test_load_experiment(sample_experiment_yaml):
    exp = load_experiment(sample_experiment_yaml)
    assert exp["model"]["type"] == "dcunet"
    assert exp["dataset"]["name"] == "test-dataset"


def test_resolve_config_applies_overrides(sample_experiment_yaml, sample_base_config, tmp_path):
    exp = load_experiment(sample_experiment_yaml)
    resolved_path = resolve_config(exp, tmp_path / "resolved.yaml")
    import yaml
    with open(resolved_path) as f:
        resolved = yaml.safe_load(f)
    assert resolved["training"]["lr"] == 3.0e-4
    assert resolved["training"]["batch_size"] == 4
    assert resolved["audio"]["sample_rate"] == 16000


def test_resolve_config_preserves_base(sample_experiment_yaml, sample_base_config, tmp_path):
    import yaml
    with open(sample_base_config) as f:
        before = yaml.safe_load(f)
    exp = load_experiment(sample_experiment_yaml)
    resolve_config(exp, tmp_path / "resolved.yaml")
    with open(sample_base_config) as f:
        after = yaml.safe_load(f)
    assert before == after


def test_build_train_args(sample_experiment_yaml, tmp_path):
    exp = load_experiment(sample_experiment_yaml)
    resolved = tmp_path / "resolved.yaml"
    resolve_config(exp, resolved)
    args = build_train_args(
        exp, resolved,
        results_path=tmp_path / "results",
        data_path=[Path("/data/test")],
        valid_path=[Path("/data/valid")],
        device_ids=[0],
    )
    assert args[0:2] == ["--model_type", "dcunet"]
    assert "--config_path" in args
    assert "--results_path" in args
    assert "--device_ids" in args


def test_build_eval_args(sample_experiment_yaml, tmp_path):
    exp = load_experiment(sample_experiment_yaml)
    resolved = tmp_path / "resolved.yaml"
    resolve_config(exp, resolved)
    args = build_eval_args(
        exp, resolved,
        checkpoint_path=tmp_path / "model.ckpt",
        valid_path=[Path("/data/valid")],
        store_dir=tmp_path / "eval_output",
        device_ids=[0],
    )
    assert "--start_check_point" in args
    assert "--store_dir" in args
    assert "--model_type" in args


def test_build_train_args_multiple_data_paths(sample_experiment_yaml, tmp_path):
    exp = load_experiment(sample_experiment_yaml)
    resolved = tmp_path / "resolved.yaml"
    resolve_config(exp, resolved)
    args = build_train_args(
        exp, resolved,
        results_path=tmp_path / "results",
        data_path=[Path("/data/a"), Path("/data/b")],
        valid_path=[Path("/data/valid")],
        device_ids=[0],
    )
    assert str(Path("/data/a")) in args
    assert str(Path("/data/b")) in args
