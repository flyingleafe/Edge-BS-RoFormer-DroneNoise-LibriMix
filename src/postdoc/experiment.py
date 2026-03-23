from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_experiment(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config(experiment: dict, output_path: Path) -> Path:
    base_config_path = Path(experiment["model"]["base_config"])
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    overrides = experiment.get("model", {}).get("overrides", {})
    for dotted_key, value in overrides.items():
        _set_nested(config, dotted_key.split("."), value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def build_train_args(
    experiment: dict,
    resolved_config: Path,
    results_path: Path,
    data_path: list[Path],
    valid_path: list[Path],
    device_ids: list[int],
    start_checkpoint: Path | None = None,
) -> list[str]:
    args = [
        "--model_type", experiment["model"]["type"],
        "--config_path", str(resolved_config),
        "--results_path", str(results_path),
        "--data_path", *[str(p) for p in data_path],
        "--device_ids", *[str(d) for d in device_ids],
    ]
    if valid_path:
        args.extend(["--valid_path", *[str(p) for p in valid_path]])
    if start_checkpoint:
        args.extend(["--start_check_point", str(start_checkpoint)])
    return args


def build_eval_args(
    experiment: dict,
    resolved_config: Path,
    checkpoint_path: Path,
    valid_path: list[Path],
    store_dir: Path,
    device_ids: list[int],
) -> list[str]:
    return [
        "--model_type", experiment["model"]["type"],
        "--config_path", str(resolved_config),
        "--start_check_point", str(checkpoint_path),
        "--valid_path", *[str(p) for p in valid_path],
        "--store_dir", str(store_dir),
        "--device_ids", *[str(d) for d in device_ids],
    ]


def _set_nested(d: dict, keys: list[str], value) -> None:
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
