---
name: run-experiment
description: End-to-end ML experiment workflow: configuration, training, evaluation, and result analysis. Use when the user wants to train a model, run evaluation, or orchestrate an experiment.
---

# Run an Experiment

Complete workflow for running ML experiments in this repository.

**First: apply the Bootstrap** (see root AGENTS.md). Reflect on what the experiment is really testing, whether the setup is optimal, and what could go wrong before executing.

## Steps

1. **Define the experiment.** What model, dataset, conditioning, and hypothesis?

2. **Check prerequisites.**
   - Dataset exists? → `data_processing/AGENTS.md`
   - Config YAML exists? → `configs/AGENTS.md`
   - Model type registered? → Check `models/AGENTS.md` for valid keys
   - RPS needed? → Ensure `load_rps: true` in config

3. **Create experiment definition.**
   - Postdoc YAML in `experiments/` → see `experiments/AGENTS.md` for format
   - Or use direct `train.py` command → see `models/AGENTS.md` for model-specific commands

4. **Execute.**
   - **Postdoc** (recommended): `postdoc job submit experiments/<name>.yaml`
   - **Direct**: `python train.py --model_type <key> --config_path <path> ...`
   - **Remote GPU**: See `vast-server-training` skill

5. **Monitor.** `postdoc job status <id>` or `postdoc job logs <id> --tail`

6. **Evaluate.** `python final_valid.py ...` or postdoc eval experiment.

7. **Analyze results.**
   - `./sync_results.sh` first (mandatory)
   - Then use `generate-model-comparisons` skill

8. **Finish with `record-and-remember`.** Record setup, results, conclusions.

## Common Pitfalls

- **Create datasets first** — training fails silently with empty data dirs
- **Match model_type key exactly** to `utils.py:get_model_from_config()` entries
- **RPS experiments require `load_rps: true`** in config and `rps.npy` in dataset
- **Always sync results before analysis** — `./sync_results.sh`