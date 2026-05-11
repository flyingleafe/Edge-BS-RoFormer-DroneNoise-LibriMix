# AGENTS.md — Harmonic Noise Suppression

## Purpose

Speech enhancement under harmonic noise from rotating sources, at ultra-low SNR (0 to −30 dB). Drones are an instrumented case study; the method targets rotating-source noise generally (C1).

**See [`GOALS.md`](./GOALS.md)** at project root for the durable goal statement, constraints (C1–C8), deadline structure, and the live portfolio of experimental bets. Read it at the start of any non-trivial research push.

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
| `vast-server-training` | Run training on remote GPU (via `postdoc submit` / SkyPilot) |

## Directory Map

Every non-gitignored directory has an `AGENTS.md` describing what it contains and why. Read it before working in that directory.

| Directory | Purpose | Key details |
|-----------|---------|-------------|
| `models/` | Model implementations | Model type keys, RPS conditioning, adding new models |
| `configs/` | YAML config files for model variants | Naming conventions, config structure |
| `src/postdoc/` | Job-runner CLI — thin wrapper over SkyPilot managed jobs on an SSH node pool | `postdoc submit <shell-command>`; see `src/postdoc/AGENTS.md` and `docs/skypilot/` |
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
- **Experiment running**: `postdoc submit <shell-command>` — thin SkyPilot wrapper. Jobs are plain shell commands; configs are the training script's concern. See `run-experiment` skill, `src/postdoc/AGENTS.md`, `docs/skypilot/README.md`
- **Playwright (browser automation)**: Installed via `python312Packages.playwright` + `playwright-driver.browsers` in `flake.nix`. The nixpkgs package shadows any `uv`-installed version via `PYTHONPATH` ordering. Uses NixOS-patched Chromium/headless_shell — no `playwright install` needed. Env vars `PLAYWRIGHT_BROWSERS_PATH` and `PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS` are set in the shell hook.
- **Results**: Always `./sync_results.sh` before analysis

## Philosophy (earned, do not relitigate)

1. **Off-the-shelf first.** Before building infra (schedulers, runners, trackers, queues, sync tools), name ≥2 existing tools and state why each is rejected. The default failure mode here is producing code when a mature tool exists — invoke `research-before-build` at the *start* of any design task, not only before the closing artifact.

2. **Jobs are shell commands.** A runner runs a command on a GPU, captures logs, restarts on failure. Structured inputs (experiment YAMLs, hyperparam sweeps, configs) are the *training script's* concern via its own flags. We do not invent DSLs for what shell already does.

3. **Bespoke code goes where it's novel.** For this project the novel part is *an agent that watches training, fixes bugs, and resumes automatically*. Queueing, GPU allocation, env setup, retries, multi-cloud orchestration — delegate to mature tools. LOC we own is a liability, not an achievement.

4. **Rewrites interrogate interfaces first, implementations second.** On any rewrite, the old input/output shape is the first thing to question. Preserving it "because it was there" is the default LLM mistake. Ask what the interface *should* be in a world without legacy.

5. **Propose alternatives at the top decision level.** When sketching architecture, list choices at the highest fork (tool A vs tool B vs bespoke) *before* optimizing within any one of them. Debate is only useful at the level where the decision actually lives.

6. **Decompose before prescribing.** When a goal or stakeholder opinion arrives, resist the urge to immediately produce an implementation plan. Ask "what would it mean to achieve this?" (verify) and "what would it take?" (decompose) repeatedly until you have a specification. Only then plan. Builder bias is real — especially in LLMs.

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