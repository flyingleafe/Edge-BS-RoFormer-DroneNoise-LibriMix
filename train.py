"""Unified training entry point (docs/refactor-unified-framework.md § "train.py
/ eval.py"). Composition + dispatch only — logic lives in ``src/training/``.

Examples::

    python train.py experiment=rps_simple_conv_v2_v4
    python train.py experiment=rps_simple_conv_v2_v4 validate_only=true
    python train.py data=dregon_lm_v4 model=simple_conv_v2 loss=pit_mse \\
        metrics=rps experiment_name=my_run epochs=50
"""

from __future__ import annotations

import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from training.config import register_configs
from training.loop import run_training
from training.validate import validate_config

register_configs()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    _seed_everything(cfg.seed)

    problems = validate_config(cfg)
    if problems:
        print(f"Config validation found {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        if cfg.validate_only:
            sys.exit(1)
        raise SystemExit("Refusing to train: pipeline failed validation (see problems above).")

    print(f"Config valid for experiment {cfg.experiment_name!r}.")
    if cfg.validate_only:
        return

    print(OmegaConf.to_yaml(cfg))
    result = run_training(cfg)
    print(f"Training finished: {result}")


if __name__ == "__main__":
    main()
