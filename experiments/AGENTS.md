# experiments/ — Experiment Definitions

YAML files that define complete experiment specifications for the `postdoc` CLI. Each file specifies a model, its config, optional overrides, dataset, eval metrics, and W&B tags.

## Why this directory exists

Decouples experiment specification from code. Enables version-controlled, reproducible experiment runs via `postdoc job submit`.

## Format

### Training experiment

```yaml
model:
  type: dcunet                              # Must match utils.py:get_model_from_config()
  base_config: configs/5_Baseline_dcunet.yaml
  overrides:                                # Dotted-key overrides merged into base config
    training.lr: 3.0e-4
    training.batch_size: 4
dataset:
  name: DN-LM                               # or DREGON-LM
eval:
  metrics: [estoi, si_sdr, pesq]
wandb:
  tags: [dcunet, experiment]
```

### Eval-only experiment

```yaml
eval_only: true
checkpoint: results/dcunet_rps/best_model.ckpt
model:
  type: dcunet
  base_config: configs/7a_DCUNet_RPS_DREGON.yaml
dataset:
  name: DREGON-LM
eval:
  metrics: [estoi, si_sdr, pesq]
wandb:
  tags: [eval, dcunet, rps, dregon]
```

## Creating a New Experiment

1. Copy `_example.yaml` as a template
2. Set `model.type` to a valid key from `utils.py:get_model_from_config()`
3. Set `model.base_config` to the corresponding config YAML path
4. Add any `model.overrides` for hyperparameter changes
5. Set `dataset.name` and `eval.metrics`
6. Add descriptive W&B tags
7. Submit: `postdoc job submit experiments/<name>.yaml`

## Existing Experiments

| Pattern | Description |
|---------|-------------|
| `train_dcunet_*` | DCUNet training variants |
| `train_dccrn_*` | DCCRN training variants |
| `train_rps_predictor_*` | Standalone RPS predictor training |
| `eval_*` | Eval-only experiments |

## Gotchas

- `model.type` must exactly match a key in `utils.py:get_model_from_config()` — typos cause config resolution failures
- `model.overrides` uses dotted keys (e.g., `training.lr`, not nested YAML)
- `checkpoint` in eval-only experiments must point to an existing `.ckpt` file