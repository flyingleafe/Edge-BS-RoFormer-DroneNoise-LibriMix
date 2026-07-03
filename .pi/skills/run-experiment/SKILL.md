---
name: run-experiment
description: End-to-end ML experiment workflow - configuration, training, evaluation, and result analysis. Use when the user wants to train a model, run evaluation, or orchestrate an experiment.
---

# Run an Experiment

Complete workflow for running ML experiments in this repository.

**First: apply the Bootstrap** (see root AGENTS.md). Reflect on what the experiment is really testing, whether the setup is optimal, and what could go wrong before executing.

## Steps

1. **Define the experiment.** What model, dataset, conditioning, and hypothesis?

2. **Check prerequisites.**
   - Dataset exists? → `data_processing/AGENTS.md`
   - Config YAML exists? → `configs/AGENTS.md`
   - Model type registered? → Check `src/models/AGENTS.md` for valid keys
   - RPS needed? → Ensure `load_rps: true` in config

3. **Compose the training command.**
   - Pick the right `--model_type` — see `src/models/AGENTS.md`.
   - Pick the right `--config` — see `configs/AGENTS.md`.
   - The entire command is what you submit; there is **no `postdoc`-specific YAML format**. If you have a reusable group of flags, write a shell script under `scripts/` and submit that.

4. **Submit.**
   ```bash
   postdoc submit python train.py --model_type <key> --config configs/<name>.yaml [--results_dir ...] [other flags...]
   ```
   Optional: `-n <name>` (job name), `--gpus N`, `--dry-run` (inspect task YAML), `-e KEY=VAL` (env vars), `-f <task.sky.yaml>` (full task spec).

   See `vast-server-training` skill for the full CLI surface and `docs/skypilot/README.md` for setup.

   **Slurm timeout fallback pattern:** when asked to try `gpushort` first and escalate only if the short job hits walltime, submit the short job, then a long `sae` job with `--dependency=afternotok:<short_job_id>`. In the dependent job, inspect `sacct` for the short job's top-level state and run training only if it starts with `TIMEOUT`; exit without training on ordinary failures. This avoids masking data/code errors as a long rerun.

5. **Monitor.** `postdoc list`, `postdoc logs <job-id> -f`, `postdoc status <job-id>`, `postdoc dashboard`.

6. **Evaluate.** Submit a second job for eval, or chain train + eval in one shell script and submit that script.
   ```bash
   postdoc submit bash -c 'python train.py ... && python final_valid.py ...'
   ```

7. **Analyze results.**
   - `./scripts/sync_results.sh` first (mandatory)
   - Then use `generate-model-comparisons` skill

8. **Finish with `record-and-remember`.** Record setup, results, conclusions.

## Common Pitfalls

- **Create datasets first** — training fails silently with empty data dirs
- **Match model_type key exactly** to `utils.py:get_model_from_config()` entries
- **RPS experiments require `load_rps: true`** in config and `rps.npy` in dataset
- **Always sync results before analysis** — `./scripts/sync_results.sh`
