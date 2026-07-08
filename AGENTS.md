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
| `create-typst-report` | Create a new Typst report in `writing/reports/` |
| `create-typst-slides` | Create a new Typst slide deck in `writing/slides/` |
| `reimplement-model` | Port a paper model into the project framework |
| `create-dregon-dataset` | (Re)create any DREGON-LM dataset variant |
| `improve-plot-visibility` | Inspect and improve generated plots |
| `load-real-propeller-geometry` | Load real propeller chord/twist into the FWH simulator |

## Task Routing — where to look first

The Skills table routes by *action*; the Directory Map routes by *location*. This table routes by *intent* — given what you're trying to do, which skill to invoke and which `AGENTS.md` / files to read **before** touching code. Always read the linked doc first; subdirectory `AGENTS.md` is truth (Rule 3).

| If the task is… | Skill | Read first |
|-----------------|-------|------------|
| **Manipulating audio / telemetry / any aligned signal** (loading, slicing, concat, shifting RPS/IMU/VAD alongside audio) | — | The in-repo `src/utils/data` timeseries algebra was replaced by the PyPI `tdseries` package (`import tdseries as td`) and deleted — see `docs/refactor-unified-framework.md` § "tdseries migration guide" for the old→new API table and gotchas. Full `tdseries` API/design: `https://github.com/flyingleafe/tflib` (source repo for the `tdseries` PyPI package; not checked out on this machine). |
| **Creating or processing a dataset** (DREGON-LM, DN-LM, mixing, RPS extraction) | `create-dregon-dataset` | `src/data_processing/AGENTS.md` (recording inventory, variants, canonical command, gotchas); loaders `src/data_processing/dregon.py`, `michaels.py`; scripts `scripts/create_dregon_librimix.py`, `scripts/create_dataset.py`. |
| **Loading data into a training loop** (Dataset/wiring, multichannel flattening) | — | `src/data_processing/AGENTS.md` § "Multichannel Training & Evaluation Wiring" (`DregonLMFrameDataset`, `NoiseRPSDataset`) + `docs/refactor-unified-framework.md` for the `tdseries`/Frame data model. Streaming from R2 (no `datasets/` checkout): `src/data_processing/streams.py` (`DloadFrameDataset`, `dload:` URIs) + `conf/data/dregon_lm_v4_stream.yaml` — see `docs/data-and-artifacts.md`. |
| **Implementing / reimplementing a model** (need examples + the interface contract) | `reimplement-model` | `src/tasks/AGENTS.md` **first** (the contract the model must satisfy) → `src/models/AGENTS.md` (registry, RPS support, "Adding a front-end"). Example impls to mirror: `src/models/dcunet_refactored.py`, `src/models/multif0/`, `src/models/rps_predictor.py`. |
| **Adding a spectral front-end** | — | `src/models/AGENTS.md` § "Spectral front-ends" / "Adding a new front-end". |
| **RPS conditioning** (RotorEncoder, fusion, predictor interface) | — | `src/models/AGENTS.md` + `src/tasks/rps-prediction/AGENTS.md`. |
| **Running an experiment** (train / eval / orchestrate) | `run-experiment` | `conf/AGENTS.md` (the Hydra experiment tree — `conf/experiment/`). Run directly with `python train.py experiment=<name>` on a local GPU; for remote GPU (Slurm/Colab/Kaggle) submit the same command via `omnirun` (Key Facts below). Legacy: `./scripts/sbatch.sh`, the `autoresearch` extension. |
| **Running online-mixed RPS-predictor training** (random mixtures each epoch, fixed validation, curriculum/augment stages) | `run-experiment` | `src/data_processing/AGENTS.md` § "Online Mixing for RPS Prediction" **first**, then `conf/AGENTS.md`, then `python train.py experiment=<name>` with a `conf/data` entry that wraps `OnlineMixIterableDataset` (policy YAMLs live at `conf/online_mix/*.yaml`; `samples_per_validation` set in `conf/config.yaml`/the experiment file). |
| **Producing reports / comparison plots / tables** | `generate-model-comparisons` (+ `improve-plot-visibility`) | Sync results first (Rule 5). Then `notebooks/AGENTS.md`; generator `eval.py` + `src/plots` comparison plots. |
| **Producing a presentation / slides** | `create-typst-slides` | `writing/AGENTS.md`; results figures via `eval.py` + `src/plots`. |
| **Producing a report** | `create-typst-report` | `writing/AGENTS.md`; results figures via `eval.py` + `src/plots`. |
| **FWH rotor / acoustic simulation, propeller geometry** | `load-real-propeller-geometry` | `src/fwh_rotor_sim/AGENTS.md`. |
| **Syncing datasets / checkpoints across machines** | — | `docs/data-and-artifacts.md` (dload + W&B artifacts + `omnirun pull`; legacy rsync fallback). |

## Directory Map

Every non-gitignored directory has an `AGENTS.md` describing what it contains and why. Read it before working in that directory.

| Directory | Purpose | Key details |
|-----------|---------|-------------|
| `src/models/` | Model implementations + pluggable spectral front-ends | Model type keys, RPS conditioning, adding new models; front-end system; see `src/models/AGENTS.md` |
| `src/tasks/` | Task interface definitions (what a model must implement for each ML task) | RPS prediction, speech enhancement; see `src/tasks/AGENTS.md` |
| `conf/` | Hydra config tree — `experiment/`, `model/` (native `_target_`; legacy models inlined via `build_legacy_inline`), `data/`, `loss/`, `metrics/`, `optim/`, plus online-mix policies in `conf/online_mix/` | See `conf/AGENTS.md` |
| `src/utils/` | The `utils` package — legacy ZFTurbo helpers in `__init__.py` | See `src/utils/AGENTS.md` for layout |
| `src/fwh_rotor_sim/` | FWH rotor acoustic simulator (BEMT + Farassat 1A), differentiable | See `src/fwh_rotor_sim/AGENTS.md` |
| `src/data_processing/` | Dataset creation and RPS processing | DN-LM, DREGON-LM loaders; offline creation scripts live in `scripts/` |
| `scripts/` | Standalone scripts not on the train/eval path | Offline dataset creation (`create_dataset.py`, `create_dregon_librimix.py`), dataset publishing (`publish_frame_datasets.py`), legacy Slurm/sync helpers (`sbatch.sh`, `sync_results.sh` — superseded by omnirun), benchmarks, config checks |
| `writing/` | Reports, slides, and papers (Typst + LaTeX) | See `writing/AGENTS.md` for templates, build chain, and visual-check workflow. |
| `notebooks/` | Jupyter notebooks for analysis | Result analysis, data exploration |
| `docs/` | Design docs and debugging guides | Training loop docs, experiment log (`docs/experiments/`), R2 sync (`docs/data-and-artifacts.md`) |
| `tests/` | Postdoc system tests | Test structure, running tests |

Root-level scripts: **only** `train.py` (the only training entry point, Hydra-driven — `python train.py experiment=<name>`) and `eval.py` (the only evaluation entry point; absorbs the former `valid.py`/`final_valid.py`/`eval_cross.py`/`eval_narrow_sr.py`/`generate_comparison.py`/`plot_per_snr.py`). Everything else that is not a `src/` package lives under `scripts/`: offline dataset-creation (`scripts/dataset.py`, `scripts/create_dataset.py`, `scripts/create_dregon_librimix.py`), dataset publishing (`scripts/publish_frame_datasets.py`), legacy Slurm/sync helpers (`scripts/sbatch.sh`, `scripts/sync_results.sh` — superseded by omnirun), benchmarks and config checks. See code comments for details. (The former root-level `utils.py` is now `src/utils/__init__.py` — `from utils import ...` keeps working; the former root `metrics.py` is now the `src/metrics` package.)

## Key Facts

- **Model types**: Registered in `utils.get_model_from_config()` (now at `src/utils/__init__.py`) — see `src/models/AGENTS.md` for full table
- **Datasets**: DN-LM (Paper 1), DREGON-LM (Paper 2) — see `data_processing/AGENTS.md`
- **RPS conditioning**: Rotor speed → RotorEncoder → fusion strategy — see `src/models/AGENTS.md`
- **Online-mixed RPS training**: loader logic is `src/data_processing/online_mixing.py`; durable YAML policies live at `conf/online_mix/*.yaml`; training uses `python train.py experiment=<name>` with a `conf/data` entry that wraps `OnlineMixIterableDataset` (e.g. `conf/data/online_mix_v4_michaels.yaml`) and a fixed, non-mixed validation split. `samples_per_validation` is a top-level Hydra field (`conf/config.yaml`). Cache location is `ONLINE_MIX_SOURCE_CACHE_DIR` in `.env` (default `.cache/online_mix_sources`).
- **Experiment running**: `python train.py experiment=<name>` is the only training entry point — on a local GPU box just run it there. For remote GPUs submit the same command via omnirun (below). Jobs are plain shell commands — there is no bespoke job-runner. See the `run-experiment` skill.
- **Remote GPU jobs (omnirun)**: `omnirun submit --backend <backend> --gpus 1 --time 30m --yes -- python train.py experiment=<name>`. Backends (user-global `~/.config/omnirun/config.toml`): `apocrita-short` (Slurm gpushort, <=1h), `apocrita-long` (Slurm sae GPU, long jobs, account-gated — GPU work only), `apocrita-cpu` (Slurm `compute`, **CPU-only**, open-access, <=10d — dataset gen/preprocessing, submit `--gpus 0`), `colab` (T4 — needs the local keep-alive daemon; allocation is a lottery), `kaggle` (P100 — ~1MB kernel source cap, needs the slim-snapshot clone recipe). Requires a clean pushed HEAD; `.env` ships per-job automatically (R2+WANDB creds → dload streaming works on any backend); cluster jobs run in shared worktrees `$PROJECT_ROOT/.trees/<sha12>` reusing the checkout's `.venv` (`uv sync --frozen`); outputs (`results/**`) come back via `omnirun pull <job>`; `omnirun backends check` revives the SSH ControlMaster after expiry. Repo-level `omnirun.toml` = job defaults only. Details: `docs/data-and-artifacts.md` § "Job running (omnirun)". Legacy fallback on Apocrita login nodes: `./scripts/sbatch.sh [slurm_params] -- <command>` (gpushort default, `--partition=sae` for long).
- **Data (dload)**: datasets live on the R2 bucket `ml-data-new` — remote in `dload.toml`, version pins in `dload.lock` (24 datasets: raw sources, per-split DREGON-LM-*, rich `tdframe-v1` frame datasets). Training streams them via `src/data_processing/streams.py` (`DloadFrameDataset`, `dload:NAME[@VER][/subpath]` URIs, `frames:NAME` specs); `dload pull <name>` prefetches. Cache knobs `DLOAD_CACHE_DIR`/`DLOAD_CACHE_BUDGET` in `.env`. See `docs/data-and-artifacts.md` + `src/data_processing/AGENTS.md` § publishing.
- **Pi autoresearch**: Project-local extension `.pi/extensions/autoresearch/` provides `/autoresearch`, `/autoresearch-resume`, plus `slurm_submit_short` (gpushort <=1h), `slurm_submit_long` (sae longer jobs), compatibility `slurm_submit`, `slurm_status`, and `slurm_logs`. It scaffolds git-tracked research artifacts under `autoresearch/<session>/`; checkpoints/results remain under `/gpfs/scratch/acw592/results/autoresearch/`. Use `/autoresearch-resume` to reattach to an existing `autoresearch/<session>/session.json` after fixing or reloading the extension.
- **Playwright (browser automation)**: Installed via `python312Packages.playwright` + `playwright-driver.browsers` in `flake.nix`. The nixpkgs package shadows any `uv`-installed version via `PYTHONPATH` ordering. Uses NixOS-patched Chromium/headless_shell — no `playwright install` needed. Env vars `PLAYWRIGHT_BROWSERS_PATH` and `PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS` are set in the shell hook.
- **Results**: sync before analysis — `omnirun pull <job>` for omnirun jobs, W&B artifacts for checkpoints (Rule 5); legacy rsync fallback `./scripts/sync_results.sh`

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
5. **Sync results before analysis.** Preferred: `dload pull <name>` (datasets) + `omnirun pull <job>` (job outputs, `results/**`) + `wandb.use_artifact(...)` (checkpoints) — see `docs/data-and-artifacts.md`. Legacy rsync fallback: `./scripts/sync_results.sh`.
6. **Record iff non-trivial.** If the task taught, solved, or revealed a gap — run `record-and-remember`. If nothing was learned, skip.

## References

- Paper 1: "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement" (Liu et al., Drones 2025)
- Paper 2: RPS-conditioned speech enhancement (inspired by Gulli et al., EURASIP 2025)