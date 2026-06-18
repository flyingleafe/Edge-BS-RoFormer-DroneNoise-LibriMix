import type { AutoresearchSession } from "./index.js";

export const SCRATCH = "/gpfs/scratch/acw592";
export const DATASETS_ROOT = `${SCRATCH}/datasets`;
export const RESULTS_ROOT = `${SCRATCH}/results/autoresearch`;

export function kickoffPrompt(session: AutoresearchSession): string {
  const metrics = session.metrics || "user-specified validation metrics";
  const trainingArgs = session.trainingArgs.trim() || "<no extra training args supplied>";
  const ideas = session.initialIdeas.trim() || "No initial ideas were supplied; propose a small diverse first batch yourself.";

  return `[AUTORESEARCH WORKFLOW: RPS architecture search]

Session: ${session.id}
Artifacts directory: ${session.artifactDir}
Dataset: ${session.dataset}
Data root: ${DATASETS_ROOT}/${session.dataset}
Results root: ${RESULTS_ROOT}/${session.id}
Target validation metrics: ${metrics}
Baseline model: ${session.baseline}
Training script: python train_rps_predictor.py
Training budget for every baseline/candidate job: --epochs 50 --patience 10
Extra training args fixed for all comparable runs:
${trainingArgs}

Initial idea seed from user:
${ideas}

You are now entering the autoresearch loop. Follow the loop instructions in the system prompt exactly. Start by inspecting the scaffolded artifact files, then train the baseline (${session.baseline}) with the exact same training parameters candidates will use. Do not propose or submit candidate jobs until the baseline job has been submitted.
`;
}

export function loopInstructions(session: AutoresearchSession): string {
  return `
## Active Workflow: Architectural Autoresearch Loop

You are running a project-specific autoresearch loop inspired by Karpathy's autoresearch framework. The goal is to improve RPS-prediction model architecture with evidence-driven Slurm experiments. Use gpushort for short <=1h jobs and sae for longer training jobs.

### Fixed session context

- Session id: ${session.id}
- Dataset: ${session.dataset}
- Dataset path: ${DATASETS_ROOT}/${session.dataset}
- Results path prefix: ${RESULTS_ROOT}/${session.id}
- Baseline model: ${session.baseline}
- Target validation metrics: ${session.metrics}
- Artifact directory committed to git: ${session.artifactDir}
- Ideas log: ${session.ideasPath}
- Experiment log: ${session.experimentsPath}
- Leaderboard: ${session.leaderboardPath}

### High-level loop

Each cycle starts from a clean git commit and an up-to-date leaderboard. Maintain four durable artifacts in git:

1. ideas.md — hypotheses, architectural ideas, expected mechanism, and why each might beat the baseline.
2. experiments.md — submitted jobs, exact commands, job ids, failures, fixes, and findings.
3. leaderboard.md — compact score table with baseline and candidates.
4. session.json — fixed session parameters and paths.

Check and update these files throughout the loop. Do not store checkpoints in git; checkpoints and run outputs live under ${RESULTS_ROOT}/${session.id}/...

### Required order

1. Inspect relevant code/docs before editing:
   - src/tasks/rps-prediction/AGENTS.md
   - src/models/AGENTS.md
   - train_rps_predictor.py
   - src/models/rps_predictor.py
2. Submit the baseline (${session.baseline}) first with the same parameters as candidates.
3. Only after baseline submission, propose a small hypothesis-first batch of candidate architecture versions.
4. For each candidate:
   - write a hypothesis in ideas.md before implementing;
   - implement the model and register it in train_rps_predictor.py::MODEL_REGISTRY;
   - smoke-test with one forward pass locally (no long login-node training);
   - submit a Slurm job with slurm_submit_short (<=1h gpushort) or slurm_submit_long (>1h/better sae GPUs).
5. Submit jobs until the first job is queued/pending and not running immediately. When slurm_status or a submit tool shows PENDING/queued, stop submitting new jobs and monitor existing jobs instead.
6. If a job fails due to a code bug, inspect slurm_logs, fix the bug, run the one-forward smoke test again, and restart the failed job. Record the failure and fix in experiments.md.
7. When jobs finish, parse best validation metrics from logs/results, update leaderboard.md, summarize findings in experiments.md, and commit code + research artifacts to git.
8. Start the next cycle only from the fresh commit and fresh leaderboard.

### Training command pattern

Use Slurm submit tools rather than bash for training. Use slurm_submit_short for jobs expected to finish within 1 hour on gpushort; use slurm_submit_long for longer jobs on sae. The command passed to the submit tool should follow this shape:

python train_rps_predictor.py \\
  --model <model_key> \\
  --device cuda:0 \\
  --data_root ${DATASETS_ROOT}/${session.dataset} \\
  --save_path ${RESULTS_ROOT}/${session.id}/<model_key> \\
  --epochs 50 \\
  --patience 10 \\
  ${session.trainingArgs.trim() || "<extra args if any>"}

Use job names like ar_${session.safeId}_<model_key>. Keep job names short enough for Slurm/log readability.

### Smoke test requirement

Before submitting any new model version, run a one-forward-pass smoke test similar to:

python - <<'PY'
import torch
from train_rps_predictor import get_model
model_name = "<model_key>"
model = get_model(model_name, n_fft=2048, hop_length=512, num_rotors=4).eval()
audio = torch.randn(2, 48000)
with torch.no_grad():
    out = model(audio)
print(model_name, tuple(out.shape))
assert out.ndim == 3, out.shape
assert out.shape[0] == 2 and out.shape[1] == 4, out.shape
PY

This is the only required smoke test. Do not run long training directly on the login node.

### Metric and leaderboard rules

- The baseline must be trained with identical dataset/training args and appears in the leaderboard.
- Record every submitted candidate, including failed ones.
- For target metrics, respect direction: lower is better for MSE/RMSE/MAE; higher is better for R² unless the user specified otherwise.
- Prefer best validation/final evaluation metrics printed by train_rps_predictor.py:
  - Per-frame: PIT MSE=..., RMSE=..., MAE=..., R²=...
  - Per-clip: MAE=...
- Include commit hash, job id, log path, save path, status, and notes in leaderboard.md.

### Git rules

- Before each new cycle, require a clean git status except for deliberate autoresearch/code changes in the current cycle.
- Commit experiment descriptions, ideas, and leaderboard with the code changes that produced them.
- Do not commit /gpfs outputs, checkpoints, logs, caches, or large generated files.
- Commit only coherent states: implemented + smoke-tested candidates and updated artifacts. Failed attempts can be recorded in experiments.md; commit code for failed attempts only if it is intentionally kept for comparison.

### Slurm tools

Dedicated tools are available:
- slurm_submit_short — submit <=1h gpushort jobs through ./sbatch.sh.
- slurm_submit_long — submit longer sae jobs through ./sbatch.sh.
- slurm_submit — compatibility/generic submit tool; prefer the explicit short/long tools.
- slurm_status — check squeue/sacct.
- slurm_logs — tail logs under ${SCRATCH}/logs.

Use these instead of raw sbatch/squeue/log-tail bash unless debugging the tools themselves.
`;
}

export function ideasTemplate(session: AutoresearchSession): string {
  return `# Autoresearch Ideas — ${session.id}

Dataset: \`${session.dataset}\`  
Baseline: \`${session.baseline}\`  
Target metrics: ${session.metrics}

## Initial user idea seed

${session.initialIdeas.trim() || "_(none supplied yet)_"}

## Hypothesis log

Add entries before implementation.

| ID | Status | Model key | Hypothesis | Expected mechanism | Risk |
|----|--------|-----------|------------|--------------------|------|
| H0 | planned | ${session.baseline} | Baseline reference trained with identical parameters. | Establish comparable score floor. | Short runs may undertrain; long sae jobs may queue longer. |
`;
}

export function experimentsTemplate(session: AutoresearchSession): string {
  return `# Autoresearch Experiments — ${session.id}

## Fixed context

- Dataset: \`${DATASETS_ROOT}/${session.dataset}\`
- Results root: \`${RESULTS_ROOT}/${session.id}\`
- Baseline: \`${session.baseline}\`
- Target metrics: ${session.metrics}
- Training budget: 50 epochs, patience 10; use gpushort <= 1:00:00 for short trials or sae for longer runs
- Extra training args: \`${session.trainingArgs.trim() || ""}\`

## Experiment log

Record exact commands, job IDs, log paths, failures, fixes, and conclusions.

### E0 — Baseline ${session.baseline}

Status: planned

Command shape:

\`\`\`bash
python train_rps_predictor.py --model ${session.baseline} --device cuda:0 --data_root ${DATASETS_ROOT}/${session.dataset} --save_path ${RESULTS_ROOT}/${session.id}/${session.baseline} --epochs 50 --patience 10 ${session.trainingArgs.trim()}
\`\`\`
`;
}

export function leaderboardTemplate(session: AutoresearchSession): string {
  return `# Autoresearch Leaderboard — ${session.id}

Target metrics: ${session.metrics}

| Rank | Model | Commit | Status | Job ID | PIT MSE | RMSE | MAE frame | MAE clip | R² | Save path | Log | Notes |
|------|-------|--------|--------|--------|---------|------|-----------|----------|----|-----------|-----|-------|
| — | ${session.baseline} | TBD | planned | TBD | TBD | TBD | TBD | TBD | TBD | \`${RESULTS_ROOT}/${session.id}/${session.baseline}\` | TBD | Baseline must be trained first. |
`;
}
