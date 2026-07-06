# conf/ — Hydra config tree (the experiment system)

One experiment = one git-committed YAML in `conf/experiment/` composing the
component groups. Run: `python train.py experiment=<name>`; evaluate:
`python eval.py experiment=<name> [checkpoint=...]`. Contract:
`docs/refactor-unified-framework.md` § "Hydra config architecture".
Historical experiments are catalogued in `REPLICATION.md` (repo root).

## Groups

| Group | Meaning |
|---|---|
| `data/` | Dataset source (folder datasets via `frame_datasets`, online-mix wrappers referencing `conf/online_mix/*.yaml` policies) |
| `model/` | Task name + task params + model instantiation via `_target_` into `models.registry` — either `build_model` (native RPS registry, flat `params`), or `build_legacy_inline` (`params.model_type` + the ZFTurbo config tree inlined under `params.config`, routed through `utils.build_model_from_config`; this replaced the former `legacy_config_path`→`configs/*.yaml` indirection) |
| `loss/` | Loss composition (entries instantiate `src/losses` Frame adapters; multiple terms via `losses.composite`) |
| `metrics/` | MetricSuite membership (must include the monitor metric) |
| `optim/` | Optimizer + scheduler + monitor |
| `logging/` | wandb (entity/project; run name = `experiment_name`) |
| `artifacts/`, `lora/` | R2 artifact uploads; LoRA seam |

## Conventions (enforced by train.py)

- `experiment_name` names everything: wandb run, `results/<name>/`, R2
  prefix `artifacts/<name>/`. Results dir collision → error unless
  `resume=true`.
- Dirty git tree → hard error unless `allow_dirty=true`; commit hash is
  logged to wandb.
- `validate_only=true` runs pre-run spec validation (+ one-batch CPU smoke
  test) and exits — run this before submitting a GPU job.
- Rates in configs are exact rationals: write `[16000, 512]`-style pairs,
  never `31.25`.

## Adding an experiment

1. Pick/create component configs in the groups.
2. Add `conf/experiment/<name>.yaml` (`# @package _global_`, defaults
   overrides, `experiment_name: <name>`).
3. `python train.py experiment=<name> validate_only=true` on a machine
   with the dataset; commit the YAML before the real run.
