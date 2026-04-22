# AGENTS.md — Harmonic Noise Suppression

## Purpose

Suppress harmonic noise from rotating equipment (drones, motors) in speech recordings at ultra-low SNR (0 to −30 dB). Achieve SOTA results. Publish papers.

**Current focus**: Paper 2 — RPS-conditioned speech enhancement on DREGON-LM dataset.

## The Bootstrap

Every non-trivial task passes through this gate — the user's query **and every non-trivial subtask it decomposes into**. This is the one recursive invariant of the whole framework; skills never restate it.

### Triviality gate (base case)

A task is **trivial** iff it is a single tool call with an obvious result, and nothing non-obvious can go wrong (e.g. `ls`, reading a known file, looking up a key from a table already on screen).

- **Trivial** → act directly. No reflection.
- **Non-trivial** → reflect (below), then act.

### Reflect (3 questions)

Narrower subtasks get shorter answers. The top-level query deserves a full pass; a subtask may need only a sentence each.

1. **Intent.** What outcome is really wanted, beyond the literal ask?
2. **Decompose.** What sub-problems does this split into? Is there a better decomposition? *Each non-trivial sub-problem is itself a task — re-enter this gate for it.*
3. **Approach & risk.** Best route per part. Failure modes, implicit assumptions, missing information (can you find it, or must you ask?).

### Then act

Route to the right skill (table below), execute, and close with `record-and-remember` **iff** anything non-trivial happened.

## Skills

### Meta-skills (routing / lifecycle — apply to every non-trivial task)

| Skill | When to use |
|-------|-------------|
| `solve-problem` | Task requires fixing, adding, improving, or building something |
| `answer-question` | Task is information retrieval ("how does X work?", "where is Y?") |
| `record-and-remember` | Closing step of any interaction where something non-trivial happened |

### Project skills (concrete workflows)

| Skill | When to use |
|-------|-------------|
| `run-experiment` | Task requires training a model, running evaluation, orchestrating an experiment |
| `generate-model-comparisons` | Publication-ready plots/tables from eval results |
| `generate-slidev-presentation` | Academic presentations with mermaid + result figures |
| `examine-presentation-slides` | Start Slidev and visually inspect slides |
| `vast-server-training` | Run training on remote GPU via tmux |

## Directory Map

Every non-gitignored directory has an `AGENTS.md` describing what it contains and why. Read it before working in that directory.

| Directory | Purpose | Key details |
|-----------|---------|-------------|
| `models/` | Model implementations | Model type keys, RPS conditioning, adding new models |
| `configs/` | YAML config files for model variants | Naming conventions, config structure |
| `src/postdoc/` | Experiment orchestration CLI | Job submission, GPU scheduling, lifecycle |
| `experiments/` | Experiment YAML definitions | Format, creating new experiments |
| `data_processing/` | Dataset creation and RPS processing | DN-LM, DREGON-LM creation scripts |
| `notebooks/` | Jupyter notebooks for analysis | Result analysis, data exploration |
| `docs/` | Design docs and debugging guides | Postdoc specs, training loop docs, R2 sync (`docs/data-and-artifacts.md`) |
| `tests/` | Postdoc system tests | Test structure, running tests |

Root-level scripts: `train.py`, `valid.py`, `final_valid.py`, `dataset.py`, `metrics.py`, `utils.py`, etc. See code comments for details.

## Key Facts

- **Model types**: Registered in `utils.py:get_model_from_config()` — see `models/AGENTS.md` for full table
- **Datasets**: DN-LM (Paper 1), DREGON-LM (Paper 2) — see `data_processing/AGENTS.md`
- **RPS conditioning**: Rotor speed → RotorEncoder → fusion strategy — see `models/AGENTS.md`
- **Experiment running**: `postdoc job submit` or direct `train.py` — see `run-experiment` skill
- **Results**: Always `./sync_results.sh` before analysis

## Rules

1. **The Bootstrap applies recursively.** Every non-trivial task and subtask passes through it. Triviality is the only stopper.
2. **Route before acting** (for non-trivial tasks). Skills exist; use them.
3. **Subdirectory AGENTS.md is truth.** Read the relevant one before working in any directory.
4. **Code comments = concrete; AGENTS.md = conceptual/structural.**
5. **Sync results before analysis.** Preferred: `dvc pull` (datasets) + `wandb.use_artifact(...)` (checkpoints) — see `docs/data-and-artifacts.md`. Legacy rsync fallback: `./sync_results.sh`.
6. **Record iff non-trivial.** If the task taught, solved, or revealed a gap — run `record-and-remember`. If nothing was learned, skip.

## References

- Paper 1: "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement" (Liu et al., Drones 2025)
- Paper 2: RPS-conditioned speech enhancement (inspired by Gulli et al., EURASIP 2025)