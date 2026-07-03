# experiments/ — Historical Experiment Records (superseded by conf/experiment/)

This directory contains YAML files that were historically the input to `postdoc submit`, back when `train.py` was config-file-driven via `ml_collections`/`OmegaConf`. **They are no longer consumed by anything** — not the runner, not `train.py`. New experiments are defined in `conf/experiment/<name>.yaml` (one git-committed Hydra file per experiment, composing `data`/`model`/`loss`/`metrics` groups) and run with `python train.py experiment=<name>`; see `docs/refactor-unified-framework.md` § "Hydra config architecture" and § "Execution waves" (wave 4: `conf/experiment/*.yaml` replicating this historical catalogue).

## Current role

These YAMLs remain useful only as:

1. **Reproducibility records** of past experiments — the model type, config, overrides, and tags that defined a given run. Cross-reference `REPLICATION.md` for the catalogue of historical experiments and their `conf/experiment/` replication status.
2. **Reference** when writing the equivalent `conf/experiment/<name>.yaml` — the model type/base config/overrides here map to `model:`/`data:`/`loss:` group overrides there (legacy per-model YAMLs are still consumed via `legacy_config_path` on `conf/model` entries that route through the old `utils.get_model_from_config` dispatch — see `configs/AGENTS.md`).

## Format (historical reference — do not act on this directly)

### Training experiment

```yaml
model:
  type: dcunet                              # Matched utils.py:get_model_from_config()
  base_config: configs/5_Baseline_dcunet.yaml
  overrides:                                # Dotted-key overrides — were merged into base config
    training.lr: 3.0e-4
dataset:
  name: DN-LM
eval:
  metrics: [estoi, si_sdr, pesq]
wandb:
  tags: [dcunet, baseline]
```

The current equivalent is a `conf/experiment/<name>.yaml`:

```yaml
# @package _global_
defaults:
  - override /data: dregon_lm_v4
  - override /model: dcunet
  - override /loss: pit_mse
  - override /metrics: rps
experiment_name: dcunet_baseline
```

run with:

```bash
postdoc submit python train.py experiment=dcunet_baseline
# eval:
postdoc submit python eval.py experiment=dcunet_baseline checkpoint=results/dcunet_baseline/best.ckpt
```

(Or one job that does both, via `bash -c '... && ...'`.)

## Do not re-introduce YAML submission

Past attempts to make `postdoc` parse these YAMLs introduced structural coupling between the runner and the model/training machinery. See root `AGENTS.md` "Philosophy": jobs are shell commands; structure is the training script's (now `train.py`'s Hydra config tree) concern.

## Gotchas

- `model.type` here refers to the legacy `utils.py:get_model_from_config()` dispatch — still reachable today via `legacy_config_path` on a `conf/model` entry, but new work should use a native `conf/model/*.yaml` where possible.
- `checkpoint` paths in eval-only YAMLs are historical — re-check existence before using; current checkpoints resolve via `results/<experiment_name>/best.ckpt` or the R2 artifact store (`src/training/artifacts.py`).
- If you find yourself typing the same long command repeatedly, write a `conf/experiment/*.yaml` instead of a shell script — that is the durable, git-committed unit of reproducibility now.
