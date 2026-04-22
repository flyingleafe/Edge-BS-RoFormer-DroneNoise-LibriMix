# experiments/ — Experiment Definitions (legacy / documentation)

This directory contains YAML files that were historically the input to `postdoc submit`. **They are no longer consumed directly by the runner.** Under the SkyPilot-based `postdoc`, jobs are plain shell commands — see `src/postdoc/AGENTS.md` and `.pi/skills/run-experiment/SKILL.md`.

## Current role

These YAMLs remain useful as:

1. **Reproducibility records** of past experiments — the model type, config, overrides, and tags that defined a given run.
2. **Templates** you copy-paste into a `train.py` invocation. Pattern:
   ```bash
   postdoc submit python train.py \
       --model_type dccrn \
       --config configs/5_Baseline_dccrn.yaml \
       --wandb_tags dccrn,baseline,dregon
   ```
3. **Starter for a shell script** if a specific experiment gets run repeatedly. Put the script under `scripts/` and `postdoc submit scripts/foo.sh`.

## Format (historical reference)

### Training experiment

```yaml
model:
  type: dcunet                              # Matches utils.py:get_model_from_config()
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

To translate the above into a command:

```bash
postdoc submit python train.py \
    --model_type dcunet \
    --config configs/5_Baseline_dcunet.yaml \
    --training.lr 3.0e-4 \
    --wandb_tags dcunet,baseline
# then chain eval:
postdoc submit python final_valid.py \
    --model_type dcunet \
    --start_check_point <path> \
    --metrics estoi si_sdr pesq
```

(Or one job that does both, via `bash -c '... && ...'`.)

## Do not re-introduce YAML submission

Past attempts to make `postdoc` parse these YAMLs introduced structural coupling between the runner and the model/training machinery. See root `AGENTS.md` "Philosophy": jobs are shell commands; structure is the training script's concern via its own flags.

## Gotchas

- `model.type` must exactly match a key in `utils.py:get_model_from_config()`.
- `checkpoint` paths in eval-only YAMLs are historical — re-check existence before using.
- If you find yourself typing the same long command repeatedly, write a small shell script and check it in.
