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
   - Experiment config exists? → `conf/AGENTS.md` (`conf/experiment/<name>.yaml` + its sibling `.md` doc)
   - Model registered? → Check `src/models/AGENTS.md` for valid `_target_`/keys
   - RPS needed? → the experiment's `conf/data` entry sets `load_rps`

3. **Pick / create the experiment config.**
   - Experiments live in `conf/experiment/<name>.yaml` (Hydra), selected with `experiment=<name>`. See `conf/AGENTS.md`.
   - New experiment? add `conf/experiment/<name>.yaml` **and** its sibling `conf/experiment/<name>.md` doc — the pre-commit hook (`scripts/validate_experiment_docs.py`) enforces this.
   - Override anything on the CLI, e.g. `optim.max_epochs=50 data=<other>`.

4. **Run.** There is no bespoke job runner — a job is just the training command.
   ```bash
   python train.py experiment=<name>                          # on a GPU box, run it directly
   ./scripts/sbatch.sh -- python train.py experiment=<name>   # on a Slurm cluster
   ```
   `./scripts/sbatch.sh` defaults to `gpushort` (≤1 h) and takes `--partition=sae` for longer jobs; or drive it from the `autoresearch` extension (`/autoresearch`, `slurm_submit_short`/`slurm_submit_long`).

   **Slurm timeout fallback pattern:** when asked to try `gpushort` first and escalate only if the short job hits walltime, submit the short job, then a long `sae` job with `--dependency=afternotok:<short_job_id>`. In the dependent job, inspect `sacct` for the short job's top-level state and run training only if it starts with `TIMEOUT`; exit without training on ordinary failures. This avoids masking data/code errors as a long rerun.

5. **Monitor.** Locally: watch stdout / the wandb run. On Slurm: `squeue -u $USER`, `sacct -j <id>`, and tail the log under `/gpfs/scratch/acw592/logs/` (or `slurm_logs` from the autoresearch extension).

6. **Evaluate.** Run the single eval entry point (it absorbed the old `valid`/`final_valid`/`eval_cross` scripts):
   ```bash
   python eval.py experiment=<name>
   ```
   Or chain: `python train.py experiment=<name> && python eval.py experiment=<name>`.

7. **Analyze results.**
   - `./scripts/sync_results.sh` first (mandatory)
   - Then use `generate-model-comparisons` skill

8. **Finish with `record-and-remember`.** Record setup, results, conclusions.

## Common Pitfalls

- **Create datasets first** — training fails silently with empty data dirs
- **Match model_type key exactly** to `utils.py:get_model_from_config()` entries
- **RPS experiments require `load_rps: true`** in config and `rps.npy` in dataset
- **Always sync results before analysis** — `./scripts/sync_results.sh`
