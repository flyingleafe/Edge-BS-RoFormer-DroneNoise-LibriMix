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
| `writeup` | Produce a complete slide deck or report end-to-end: work inventory since the last artifact → narrative (user checkpoint) → restricted creator agent + adversarial critic loop (dynamic workflow `writeup-build`). Use this — not the scaffolding skills below — whenever a whole deck/report is wanted. |
| `run-experiment` | Task requires training a model, running evaluation, orchestrating an experiment |
| `generate-model-comparisons` | Publication-ready plots/tables from eval results |
| `create-typst-report` | Create a new Typst report in `writing/reports/` |
| `create-typst-slides` | Create a new Typst slide deck in `writing/slides/` |
| `reimplement-model` | Port a paper model into the project framework |
| `create-dregon-dataset` | (Re)create any DREGON-LM dataset variant |
| `improve-plot-visibility` | Inspect and improve generated plots |

## Task Routing — where to look first

The Skills table routes by *action*; the Directory Map routes by *location*. This table routes by *intent* — given what you're trying to do, which skill to invoke and which `AGENTS.md` / files to read **before** touching code. Always read the linked doc first; subdirectory `AGENTS.md` is truth (Rule 3).

| If the task is… | Skill | Read first |
|-----------------|-------|------------|
| **Manipulating audio / telemetry / any aligned signal** (loading, slicing, concat, shifting RPS/IMU/VAD alongside audio) | — | The in-repo `src/utils/data` timeseries algebra was replaced by the PyPI `tdseries` package (`import tdseries as td`) and deleted — see `docs/refactor-unified-framework.md` § "tdseries migration guide" for the old→new API table and gotchas. Full `tdseries` API/design: `https://github.com/flyingleafe/tflib` (source repo for the `tdseries` PyPI package; not checked out on this machine). |
| **Creating or processing a dataset** (DREGON-LM, DN-LM, SE-valid, mixing, RPS extraction) | `create-dregon-dataset` | `docs/refactor-data-pipelines.md` (the data-layer architecture) **first**, then `src/data_processing/AGENTS.md` (recording inventory, variants, gotchas). Every raw source is a `src/data_processing/sources/` registry entry; every derived dataset is a frozen pipeline spec in `src/data_processing/derivations.py`, materialized by `python scripts/derive.py derive <NAME>`; the per-sample mixing cores are `src/data_processing/mixing.py`. |
| **Loading data into a training loop** (Dataset/wiring, multichannel flattening) | — | `src/data_processing/AGENTS.md` § "Multichannel Training & Evaluation Wiring" (`DregonLMFrameDataset`, `NoiseRPSDataset`) + `docs/refactor-unified-framework.md` for the `tdseries`/Frame data model. Streaming from R2 (no `datasets/` checkout): `src/data_processing/streams.py` (`DloadFrameDataset`, `dload:` URIs) + `conf/data/dregon_lm_v4_stream.yaml` — see `docs/data-and-artifacts.md`. |
| **Implementing / reimplementing a model** (need examples + the interface contract) | `reimplement-model` | `src/tasks/AGENTS.md` **first** (the contract the model must satisfy) → `src/models/AGENTS.md` (registry, RPS support, "Adding a front-end"). Example impls to mirror: `src/models/dcunet_refactored.py`, `src/models/multif0/`, `src/models/rps_predictor.py`. |
| **Adding a spectral front-end** | — | `src/models/AGENTS.md` § "Spectral front-ends" / "Adding a new front-end". |
| **RPS conditioning** (RotorEncoder, fusion, predictor interface) | — | `src/models/AGENTS.md` + `src/tasks/rps-prediction/AGENTS.md`. |
| **Running an experiment** (train / eval / orchestrate) | `run-experiment` | `conf/AGENTS.md` (the Hydra experiment tree — `conf/experiment/`). Run directly with `python train.py experiment=<name>` on a local GPU; for remote GPU (Slurm/Colab/Kaggle) submit the same command via `omnirun` (Key Facts below). |
| **Running online-mixed RPS-predictor training** (random mixtures each epoch, fixed validation, curriculum/augment stages) | `run-experiment` | `src/data_processing/AGENTS.md` § "Online Mixing for RPS Prediction" **first**, then `conf/AGENTS.md`, then `python train.py experiment=<name>` with a `conf/data` entry that wraps `OnlineMixFrameDataset` (policy YAMLs live at `conf/online_mix/*.yaml`; `samples_per_validation` set in `conf/config.yaml`/the experiment file). A policy compiles to one infinite `dload.Pipeline` — `data_processing.online_mixing.build_online_mix_pipeline`. |
| **Producing reports / comparison plots / tables** | `generate-model-comparisons` (+ `improve-plot-visibility`) | Sync results first (Rule 5). Then `src/plots/AGENTS.md` — `plots.dwym(frame)` is the front door for a quick figure, `eval.py` + the `src/plots` comparison plots for publication figures, `scripts/table.py` for a pivot table over tidy CSV/JSON. |
| **Producing a presentation / slides** | `writeup` | `writing/AGENTS.md`; results figures via `eval.py` + `src/plots`; scaffolding recipe used by the creator agent is `create-typst-slides`. |
| **Producing a report** | `writeup` | `writing/AGENTS.md`; results figures via `eval.py` + `src/plots`; scaffolding recipe used by the creator agent is `create-typst-report`. |
| **Rotor-speed tracking / VK order tracking / trajectory refinement** (blind annotation, refinement stages, protocol evaluation) | `solve-problem` | `src/tracking/AGENTS.md` (module map, the `Stage: Frame -> Frame` contract, the **purity rule**) → `docs/vk-order-tracking-design.md`. Evaluate with `python scripts/rps_eval.py --protocol <beatvk\|vk37> --pred ... [--refine ...]`; campaign log `docs/experiments/rps-refine-precision.md`. |
| **Loading a trained model / finding a checkpoint** | — | `src/zoo/AGENTS.md`: `zoo.checkpoints()` lists what is on R2, `zoo.load(experiment)` returns a `FrameModel` (`td.Frame` in → `td.Frame` out). Inspect one without a dataset via `python scripts/probe_ckpt.py --ckpt zoo:<experiment>`. |
| **Exploring data or results in a notebook** | — | `notebooks/AGENTS.md` (which notebook exists and what it drives) + `docs/notebook-primitives-tutorial.md`. The primitives are `plots.explore` (`datasets`/`meta_table`/`grid`/`pick`) and `plots.dwym`; notebooks stay thin — put logic in `src/`. |
| **Syncing datasets / checkpoints across machines** | — | `docs/data-and-artifacts.md` (dload + W&B artifacts + `omnirun pull`). |

## Directory Map

Every non-gitignored directory has an `AGENTS.md` describing what it contains and why. Read it before working in that directory.

| Directory | Purpose | Key details |
|-----------|---------|-------------|
| `src/models/` | Model implementations + pluggable spectral front-ends | Model type keys, RPS conditioning, adding new models; front-end system; see `src/models/AGENTS.md` |
| `src/tasks/` | Task definitions + codecs (the Frame ↔ tensor seam) | The four tasks — `speech_enhancement`, `rps_prediction`, `salience_rps`, `noise_generation` — with one subdirectory each; see `src/tasks/AGENTS.md` |
| `src/tracking/` | The rotor-speed tracking stack (Vold–Kalman, refinement, blind seeding, beam/DP search) | Pure array code; `Stage: td.Frame -> td.Frame` composition, frozen protocol window specs. **Purity rule**: imports only numpy/scipy/torch/tdseries/utils — never `data_processing`, `models`, `training`. See `src/tracking/AGENTS.md` |
| `src/zoo/` | Model-type + checkpoint registry, and `FrameModel` | `zoo.checkpoints()` / `zoo.load(experiment)`; R2 is the source of truth, the index is the gitignored `.checkpoints-cache.json`. See `src/zoo/AGENTS.md` |
| `src/framespec/` | The frame-shape vocabulary (`FrameSpec`, `SeriesSpec`, `TimeKind`) | Leaf package with no internal imports, so `losses`/`metrics` can sit below `tasks`. Import `framespec` directly — the `tasks.spec` shim is gone |
| `src/plots/` | All plotting | `plots.dwym(frame)` is the front door (frame-level dispatch + coercion); `plots.explore` holds the notebook primitives; renderers and comparison plots underneath. See `src/plots/AGENTS.md` |
| `src/losses/`, `src/metrics/` | Frame-level losses and metrics | One implementation each; depend on `framespec`, not on `tasks`. See their `AGENTS.md` |
| `src/training/` | Training loop, config building, checkpointing, R2 artifact upload | `build_task_and_codec` / `instantiate_model` are the config seam. See `src/training/AGENTS.md` |
| `src/experiments/` | Contract-fenced research sandbox | May import anything; **nothing imports it** (import-linter contract) |
| `conf/` | Hydra config tree — `experiment/`, `model/` (native `_target_`; legacy models inlined via `build_legacy_inline`), `data/`, `loss/`, `metrics/`, `optim/`, plus online-mix policies in `conf/online_mix/` | See `conf/AGENTS.md` |
| `src/utils/` | The `utils` package — legacy ZFTurbo helpers in `__init__.py` | See `src/utils/AGENTS.md` for layout |
| `src/data_processing/` | Dataset creation, streaming and RPS processing | `sources/` registry (raw datasets) → `derivations.py` (derived-dataset pipeline specs) → `streams.py`/`online_mixing.py` (consumption); driver `scripts/derive.py` |
| `scripts/` | Thin, parameterized CLIs — never imported by `src/` | Six generic tools carry the load: `se_eval.py` (SE scoring), `rps_eval.py` (protocol × prediction × refinement), `table.py` (pivot tables over tidy CSV/JSON), `bench.py` (kernel micro-benchmarks), `probe_ckpt.py` (checkpoint inspection) and the `utils.gridrun` harness they share; plus `derive.py` (the single dataset driver) and the config/stream checks (`check_stream.py`, `check_experiment_configs.py`). Campaign-specific scripts are held only until their campaign closes. Remote jobs and result sync go through omnirun. |
| `writing/` | Reports, slides, and papers (Typst + LaTeX) | See `writing/AGENTS.md` for templates, build chain, and visual-check workflow. |
| `notebooks/` | Jupyter notebooks for analysis | Result analysis, data exploration |
| `docs/` | Design docs and debugging guides | Training loop docs, experiment log (`docs/experiments/`), R2 sync (`docs/data-and-artifacts.md`) |
| `tests/` | Pytest suite, one directory per `src/` package | Layout table + which markers are deselected by default; see `tests/AGENTS.md` |

Root-level scripts: **only** `train.py` (the only training entry point, Hydra-driven — `python train.py experiment=<name>`) and `eval.py` (the only evaluation entry point; absorbs the former `valid.py`/`final_valid.py`/`eval_cross.py`/`eval_narrow_sr.py`/`generate_comparison.py`/`plot_per_snr.py`). Everything else that is not a `src/` package lives under `scripts/`: dataset materialization/publishing (`scripts/derive.py` — the single driver; the per-dataset creation/publisher CLIs are deleted, their recipes now live as specs in `src/data_processing/derivations.py`), protocol evaluators, benchmarks and config checks. The legacy Slurm/sync helpers (`sbatch.sh`, `sync_results.sh`) are deleted — omnirun replaces them. See code comments for details. (The former root-level `utils.py` is now `src/utils/__init__.py` — `from utils import ...` keeps working; the former root `metrics.py` is now the `src/metrics` package.)

## Key Facts

- **Model types**: Unified listing `models.registry.model_types()` — merges the RPS, legacy (`LEGACY_MODEL_BUILDERS`), and noise-gen registries plus the direct `conf/model` `_target_` factories, all in `src/models/registry.py` — see `src/models/AGENTS.md` for full table
- **Datasets**: DN-LM (Paper 1), DREGON-LM (Paper 2) — see `data_processing/AGENTS.md`. Every dataset is declared exactly once: raw sources in `src/data_processing/sources/`, derived datasets as `derivations.SPECS`; `python scripts/derive.py list|derive|adopt` is the only driver.
- **RPS conditioning**: Rotor speed → RotorEncoder → fusion strategy — see `src/models/AGENTS.md`
- **Checkpoints (zoo)**: `zoo.checkpoints()` lists every experiment's checkpoints, config presence and eval metrics; `zoo.load("<experiment>", ckpt="best")` returns a `FrameModel` — Hydra-composed model + its codec, `td.Frame` in → `td.Frame` out. R2 (the `training.artifacts.ArtifactStore` bucket/prefix) is the source of truth; the index is the **gitignored** `.checkpoints-cache.json` at repo root, refreshed incrementally. `tasks.checkpoints.load_model("Type@ckpt.pt")` remains the bare-`state_dict` path for RPS models. See `src/zoo/AGENTS.md`.
- **Tracking (VK / refinement)**: the whole rotor-speed tracking stack is `src/tracking` — pure array code with a `Stage: td.Frame -> td.Frame` contract (`tracking.pipeline(...)` composes stages, each appending diagnostics to `meta["tracking"]`), plus the frozen evaluation protocols as data (`tracking.protocols`). It must not import `data_processing`/`models`/`training`; an import-linter contract enforces this. See `src/tracking/AGENTS.md`.
- **Plotting**: `from plots import dwym; dwym(frame)` is the one front door — it coerces entry names, dispatches on frame shape (se / salience / noise_gen / rps / audio / timeframe), and takes a `{label: Frame}` dict for aligned comparisons. Publication figures still go through `eval.py` + the `src/plots` comparison plots. See `src/plots/AGENTS.md`.
- **Notebook primitives**: `plots.explore` — `datasets()` (the dload catalog), `meta_table()`, `grid()` (spectrogram thumbnails), `pick()` (one coerced sample, ready for `dwym` or a `FrameModel`). Notebooks are thin drivers; if a cell grows past ~5 lines the logic belongs in `src/`. Tutorial: `docs/notebook-primitives-tutorial.md`.
- **Parallel probes (gridrun)**: `utils.gridrun` is THE restartable parallel unit-JSON harness — one unit = one `<out>/raw/<uid>.json`, existing units are skipped on resume, a unit exception becomes a `.err` file instead of killing the pool, and everything aggregates into `summary.json`. Use `add_gridrun_args`/`gridrun_from_args` in any new sweep CLI; do not hand-copy a process pool again.
- **Online-mixed RPS training**: loader logic is `src/data_processing/online_mixing.py`; durable YAML policies live at `conf/online_mix/*.yaml`; training uses `python train.py experiment=<name>` with a `conf/data` entry that wraps `OnlineMixIterableDataset` (e.g. `conf/data/online_mix_v4_michaels.yaml`) and a fixed, non-mixed validation split. `samples_per_validation` is a top-level Hydra field (`conf/config.yaml`). Cache location is `ONLINE_MIX_SOURCE_CACHE_DIR` in `.env` (default `.cache/online_mix_sources`).
- **Experiment running**: `python train.py experiment=<name>` is the only training entry point — on a local GPU box just run it there. For remote GPUs submit the same command via omnirun (below). Jobs are plain shell commands — there is no bespoke job-runner. See the `run-experiment` skill.
- **Remote GPU jobs (omnirun)**: `omnirun submit --backend <backend> --gpus 1 --time 30m --yes -- python train.py experiment=<name>`. Backends (user-global `~/.config/omnirun/config.toml`): `apocrita-short` (Slurm gpushort, <=1h), `apocrita-long` (Slurm sae GPU, long jobs, account-gated — GPU work only), `apocrita-cpu` (Slurm `compute`, **CPU-only**, open-access, <=10d — dataset gen/preprocessing, submit `--gpus 0`), `colab` (T4 — needs the local keep-alive daemon; allocation is a lottery), `kaggle` (P100 — ~1MB kernel source cap, needs the slim-snapshot clone recipe). Requires a clean pushed HEAD; `.env` ships per-job automatically (R2+WANDB creds → dload streaming works on any backend); cluster jobs run in shared worktrees `$PROJECT_ROOT/.trees/<sha12>` reusing the checkout's `.venv` (`uv sync --frozen`); outputs (`results/**`) come back via `omnirun pull <job>`; `omnirun backends check` revives the SSH ControlMaster after expiry. Repo-level `omnirun.toml` = job defaults only. Details: `docs/data-and-artifacts.md` § "Job running (omnirun)".
- **Data (dload)**: datasets live on the R2 bucket `ml-data-new` — remote in `dload.toml`, version pins in `dload.lock` (26 datasets: raw sources, per-split DREGON-LM-*, DN-LM-{train,valid} derived datasets, rich `tdframe-v1` frame datasets). Training streams them via `src/data_processing/streams.py` (`DloadFrameDataset`, `dload:NAME[@VER][/subpath]` URIs, `frames:NAME` specs); `dload pull <name>` prefetches. Cache knobs `DLOAD_CACHE_DIR`/`DLOAD_CACHE_BUDGET` in `.env`. See `docs/data-and-artifacts.md` + `src/data_processing/AGENTS.md` § publishing.
- **Pi autoresearch**: Project-local extension `.pi/extensions/autoresearch/` provides `/autoresearch`, `/autoresearch-resume`, plus `slurm_submit_short` (gpushort <=1h), `slurm_submit_long` (sae longer jobs), compatibility `slurm_submit`, `slurm_status`, and `slurm_logs`. It scaffolds git-tracked research artifacts under `autoresearch/<session>/`; checkpoints/results remain under `/gpfs/scratch/acw592/results/autoresearch/`. Use `/autoresearch-resume` to reattach to an existing `autoresearch/<session>/session.json` after fixing or reloading the extension.
- **Playwright (browser automation)**: Installed via `python312Packages.playwright` + `playwright-driver.browsers` in `flake.nix`. The nixpkgs package shadows any `uv`-installed version via `PYTHONPATH` ordering. Uses NixOS-patched Chromium/headless_shell — no `playwright install` needed. Env vars `PLAYWRIGHT_BROWSERS_PATH` and `PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS` are set in the shell hook.
- **Results**: sync before analysis — `omnirun pull <job>` for omnirun jobs, W&B artifacts for checkpoints (Rule 5)

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
5. **Sync results before analysis.** `dload pull <name>` (datasets) + `omnirun pull <job>` (job outputs, `results/**`) + `wandb.use_artifact(...)` (checkpoints) — see `docs/data-and-artifacts.md`.
6. **Record iff non-trivial.** If the task taught, solved, or revealed a gap — run `record-and-remember`. If nothing was learned, skip.
7. **Worktrees live in `.worktrees/`** — every agent, one directory. Create one with `git worktree add .worktrees/<name> -b <branch>`, and enter it with Claude Code's `EnterWorktree` **`path`** argument. Do NOT call `EnterWorktree` with `name`: that argument hardcodes `.claude/worktrees/`, which this project does not use. The directory is not a setting in Claude Code 2.1, and a symlink at `.claude/worktrees` is refused, so this rule is the only enforcement. A worktree that holds its own `.venv` is not relocatable — after `git worktree move`, rewrite the old path inside `.venv/bin/*` and the editable-install `.pth` (or run `uv sync`).

## References

- Paper 1: "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement" (Liu et al., Drones 2025)
- Paper 2: RPS-conditioned speech enhancement (inspired by Gulli et al., EURASIP 2025)