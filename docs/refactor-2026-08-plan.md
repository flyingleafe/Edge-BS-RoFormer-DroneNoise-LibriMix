# Repository Refactor Plan — 2026-08-04

Scope: the nine points from the 2026-08-04 session, plus the test audit.
Evidence: five parallel code surveys (VK tracking, scripts, plots+notebooks,
model registry + fwh_rotor_sim, tests) and an AST-level import-graph scan
(240 modules, 480 internal edges).

## Status (2026-08-04)

**Executed.** Phases 0–6 are merged on `worktree-refactoring`; Phase 7 (this
closeout) is the last one. The import graph is now 179 files / 365 internal
dependencies, and all three import-linter contracts are KEPT.

| Phase | Commit(s) | What landed |
|---|---|---|
| 0 — safety net | `323c757` | `slow`/`network` markers + the default `-m` filter, first import contracts, cycle-edge fixes |
| 1 — deletions | `d578cb3` | `fwh_rotor_sim` (superseded by the external `auraflow` repo), 26 dead scripts, 18 stale notebooks |
| 2 — tracking | `f1a7350` (2a), `22d41bb` (2c), `96dc11e` (complete) | `src/tracking` + `src/framespec` extracted, the layers contract, the `Stage: Frame -> Frame` API, the vit2dsp ladder promoted to `tracking/pipelines.py` |
| 3 — registries | `62081bb` (3a), `f2beefb` (3b) | One `models.registry.model_types()`, `utils` is a leaf, `src/zoo` over the R2 artifact store |
| 4 — plots | `8d2ad92` | `plots.dwym` front door + `plots.coerce` + renderer consolidation |
| 5 — scripts | `ae6979e` | Six generic tools (`se_eval`, `table`, `bench`, `probe_ckpt`, `rps_eval`, `utils.gridrun`) replace the scripts zoo; `sbatch.sh`/`sync_results.sh` retired |
| 6 — notebooks | `699797b` | `plots.explore` primitives, focused `rps_tracking`/`noise_generation` notebooks, the SE tutorial |
| 7 — tests + docs | *this patch* | `tracking/phase_noise.py`, the test-audit remnants, every AGENTS.md refreshed, this status section |

### Known remaining debts

1. **`scripts/rps_eval.py --pred model:<key>` still routes through
   `rps_predictor_vk_eval.MODELS`** (and `beatvk_eval.preds_from_model` with
   it). Migrating to `zoo.load` is not a drop-in: `MODELS` is a curated
   *key → (experiment, ckpt URI, epoch)* table whose keys name published
   result rows, and the stitched-chmean inference calls the bare `nn.Module`
   through `vkev.predict_windows`/`stitch_stack`, not a `FrameModel`. Do it
   together with a `zoo`-side named-checkpoint alias, once the beat-VK
   campaign closes and its result keys stop being load-bearing.
2. **`notebooks/speech_enhancement.ipynb` is unwritten by design** (§6 decision
   5). It is the Phase 6 ergonomics probe: the user builds it by hand from
   `docs/notebook-primitives-tutorial.md` and the friction list drives the
   next ergonomics fixes. Do not write it for them.
3. **`zoo.checkpoints()` eval-metric columns are empty** until `eval.py` runs
   again per experiment (it uploads `eval/metrics.json` next to the
   checkpoints) and a `zoo.refresh()` picks the new objects up.
4. **Campaign scripts held in `scripts/`** until their campaign closes, then
   delete or fold into a generic tool: `sr_dp_probe.py`, `jb_probe.py`
   (the two live `utils.gridrun` exemplars), `beatvk_rescore.py`,
   `eval_noise_gen_variants.py`, `cd_iter_sweep.sh`, plus the beat-VK core
   (`beatvk_eval.py`, `beatvk_vk_arms.py`, `rps_predictor_vk_eval.py`,
   `vk_blind_annotation.py`, `rps_refine_lab.py`, `vk_*.py`).
5. **`tasks/spec.py` is still a back-compat shim** over `framespec`; remove it
   once no config or checkpoint references the old path.
6. **`salience_rps` has no task subdirectory** — it is documented inside
   `src/tasks/rps-prediction/AGENTS.md` as a readout variant. Split it out if
   it grows its own training/eval conventions.

## 1. Goals

1. A person can find each algorithm in one obvious package.
2. Stages and models compose as TimeFrame → TimeFrame callables.
3. One plotting entry point (`plot_dwym`) covers 90% of figures.
4. A dload-style registry lists all model types and checkpoints.
5. `scripts/` holds only thin, parameterized CLIs.
6. Notebooks are short because the library does the work.
7. import-linter contracts prevent structural drift.
8. Significant net LOC reduction, no loss of essential function.

## 2. Current state — survey summary

| Area | Finding |
|---|---|
| Import graph | 4 package cycles: `data_processing⇄training`, `data_processing⇄tasks`, `data_processing⇄experiments`, `tasks⇄losses`. `scripts` makes 139 imports into `data_processing`. |
| VK tracking | 7 modules, ~5,700 LOC inside `data_processing`. Pure array code, zero coupling to frames/dload. All pipeline composition lives in `scripts/` (`vk_blind_annotation.py` 3,175 LOC). Ad-hoc `Prepared` dataclass instead of TimeFrame. |
| scripts/ | 77 files, 29,513 LOC. 43% is library code imported by other scripts and tests (via `sys.path` hacks). 22% is 15 copies of the same parallel-harness boilerplate. 26 files are closed one-offs. |
| plots | Track-level dispatch exists (`resolve_renderer_key`) and is good. No frame-level dispatch, no multi-frame input, no environment detection, no audio widget in the library. Spectrogram drawing exists in ≥8 independent implementations. `ROTOR_COLORS` defined 3×. |
| Model registry | 4 parallel registries (23-type if/elif chain, 56-key dict, 2-key dict, direct factories). No checkpoint index: 62 `r2://` URIs hardcoded across ≥8 scripts + 2 notebooks. No metrics index. Checkpoints are bare `state_dict`s with no config snapshot. |
| Codec seam | `Codec.to_inputs / call_model / to_frame` already is the Frame→Frame wrapper. Only 3 call sites use it. `FrameModel(model, codec)` is ~10 lines. |
| fwh_rotor_sim | 1,283 LOC. Zero imports from `src`, `tests`, `conf`, `scripts`. Superseded by the external `auraflow` repo (not referenced anywhere in-repo — record the pointer at deletion). |
| notebooks | 26 notebooks; ~17 stale, superseded, or import-broken. Helper libs are numpy/pandas, not TimeFrame-aware. |
| tests | 746 tests collect cleanly, zero broken imports. Real issues: a live-R2 network test self-enabled by `.env`, an author-only golden test, 2 tests importing from `scripts/`/`notebooks/` via `sys.path`, dead `slow` marker, a fake test file in `fwh_rotor_sim`. Zero coverage: `src/experiments/`, most baseline models, `training/lora.py`, `plots/cli.py`. |

## 3. Target architecture

### 3.1 Packages (end state)

```
utils              bottom layer; no internal imports
framespec          NEW (extraction of tasks/spec.py): FrameSpec / SeriesSpec /
                   TimeKind — the frame-shape vocabulary everything shares
data_processing    sources, frames, streams, mixing, online_mixing,
                   derivations, synthesis (name kept — no rename)
tracking           NEW. VK / refinement / beam-search stack. Depends only on
                   utils + framespec + tdseries. No data or model imports.
models             torch modules + frontends. One MODEL_TYPES dict.
losses, metrics    Frame-level functions. Depend on framespec, NOT on tasks.
tasks              task definitions + codecs (the Frame<->tensor seam);
                   imports and bundles losses/metrics defaults
zoo                NEW. Model-type + checkpoint + metrics registry;
                   FrameModel wrappers. Depends on models/tasks/training.artifacts.
plots              rendering; depends on tasks + tdseries only
training           loop, artifacts, validate; top of src
experiments        sandbox; may import anything; nothing imports it
scripts/           thin CLIs only; import src, never the reverse
notebooks/         import src only
```

Notes:
- `fwh_rotor_sim` is deleted (§5, phase 1).
- `localization` (if revived) follows the `tracking` pattern.

Inversion of the losses/metrics ↔ tasks relationship (decided 2026-08-04):
losses/metrics currently import `tasks.spec` (15 edges) plus
`tasks.rps_prediction.align_rps_to_gt` (1 edge). The dependency is on the
spec *vocabulary*, not on task definitions. Fix:
1. Extract `tasks/spec.py` → `src/framespec/` (leaf package, no internal deps).
2. Move `align_rps_to_gt` → `losses/pit.py` (PIT machinery already lives
   there; `tasks/rps_prediction.py:607` already lazy-imports `losses.pit`).
3. Move `losses._common.get_tensor` (used by `tasks/codecs.py`) wherever it
   deduplicates best — legal either way once tasks sits above losses.
After this, tasks imports losses/metrics freely and bundles per-task defaults.

### 3.2 The Stage contract (point 1)

Every tracking stage is a callable `Stage: td.Frame -> td.Frame`:

- Input frame entries: `audio` (mic, time)@sr, `rps` (rotor, time)@frame_rate
  — the current candidate trajectories — plus optional `rps_meas`,
  `valid_span` entries.
- Output: the same frame with `rps` replaced, and stage diagnostics appended
  under `meta.tracking[<stage_name>]` (scores, guard verdicts, config).
- The array cores (`vk_track`, `blind_seed`, `pi_kalman_refine`,
  `joint_beam_track`, `refine_coherent`, `iter_warp_refine`, …) stay as
  they are; stages are thin adapters in `tracking/stages.py`.
- Composition: plain function composition, or
  `tracking.pipeline(stage_a, stage_b, ...)` returning a Stage. The canonical
  ladders (`vit2dsp`, blind seeding arms, cd_iter) become named, importable
  pipeline definitions in `tracking/pipelines.py` — promoted out of
  `scripts/vk_blind_annotation.py` and `scripts/rps_refine_lab.py`.
- `Prepared` (scripts/vk_validation.py) is replaced by a td.Frame with the
  entries above; one adapter builds it from a dload recording.

### 3.3 The zoo (points 5 + 6)

R2 is the source of truth; the local index is a **gitignored cache**
(`.checkpoints-cache.json` at repo root), populated by listing the bucket
(decided 2026-08-04 — no git-committed registry file).

Cache mechanics:
- `zoo.refresh()` lists `s3://ml-data/artifacts/*/checkpoints/*` (the repo
  currently has zero `list_objects` calls — this adds the one place that
  does it) and merges per-experiment metadata: matching
  `conf/experiment/<name>.yaml` (or a flag when none exists), object
  mtimes/etags, and eval metrics (below).
- Incremental refresh: the cache stores its population timestamp and the
  set of known experiment prefixes. A refresh first diffs cheap signals —
  new `conf/experiment/*.yaml` names and new/changed top-level prefixes —
  and re-lists only what changed. `zoo.checkpoints()` auto-refreshes when
  the cache is stale (age threshold) or missing; `refresh(full=True)`
  forces a complete re-list.
- Metrics: `eval.py` uploads its `eval/metrics.json` (+ `per_snr.csv`)
  next to the checkpoints via the existing `ArtifactStore`
  (`artifacts/<exp>/eval/`). The cache refresh harvests these, so metrics
  travel with the artifact store and appear on any machine after a refresh
  — no local state, no merge noise.

API (`src/zoo/`):
- `zoo.model_types() -> dict[str, TypeInfo]` — union of the (dict-ified)
  legacy chain + RPS registry + noise-gen registry + direct factories.
- `zoo.checkpoints(task=None) -> table` — names, metadata, metrics
  (from the cache; auto-refresh as above).
- `zoo.load(name, device=...) -> FrameModel` — Hydra-compose by experiment
  name, `resolve_checkpoint_uri`, `load_state_dict`, wrap in codec.
- `FrameModel.__call__(frame: td.Frame) -> td.Frame` — the existing codec
  triple packaged as one callable; batch-of-one convenience for notebooks.
- Validation seed: the 62 hardcoded `r2://` URIs, the `MODELS` dict in
  `scripts/rps_predictor_vk_eval.py`, and `generator_lab.VARIANTS` become
  the acceptance checklist — every one must be discoverable via
  `zoo.checkpoints()` after the first full refresh.
- `eval.py` gains `r2://` checkpoint resolution (one line) + the metrics
  upload above.

### 3.4 plot_dwym (point 4)

`plots.dwym(obj, **hints)` where `obj` is a Frame, a list of Frames, or a
dict `{label: Frame}`:

1. Frame-level dispatch (new, `plots/dwym.py`): inspect entry names + specs.
   `mixture/target/enhanced` → SE triple; `audio + rps[_pred]` → RPS overlay;
   `salience` → salience heatmap; generated-vs-real pair → noise-gen grid;
   plain audio → spectrogram + waveform. Falls through to the existing
   track-level `resolve_renderer_key` for unknown entries.
2. Multi-frame input renders aligned comparison rows (model A vs model B,
   real vs generated).
3. Environment awareness: `get_ipython()` detection. In a notebook, return a
   rich display object (figure + `IPython.display.Audio` players per audio
   entry). Otherwise return the bare Figure (and `make-plot` saves files).
4. Coercion of raw frames into the common form (`plots/coerce.py`): raw
   DREGON / Michael's frames carry many timeseries and no `rps` entry per
   se. Before dispatch, `dwym` normalizes: entry-name synonyms map to the
   canonical vocabulary (e.g. `motor_rps`/`motor_speed`/telemetry rotor
   channels → `rps`, per-source alias tables seeded from what the
   notebooks already do by hand); unit/rate sanity checks; extra
   timeseries kept as additional tracks rather than dropped. Every
   coercion prints a one-line warning naming the mapping it applied, so a
   wrong guess is visible and overridable via hints
   (e.g. `dwym(frame, rps="motor_speed_filtered")`).
5. Escape hatches: `hints` flow to renderers (existing `PlotTrack.hints`
   channel); every underlying figure function stays public.
6. Consolidation in the same pass: one spectrogram implementation
   (`make_spectrogram_series`), one `ROTOR_COLORS`, one RPS-overlay renderer;
   port the SE and noise-gen figures out of `val_logging`/notebook libs into
   `plots/`; `val_logging` becomes a thin wandb adapter over the same
   renderers. Delete `slide_comparison.py` (dead).

### 3.5 import-linter contracts (initial set)

```toml
[tool.importlinter]
root_packages = ["utils", "framespec", "data_processing", "tracking",
                 "models", "tasks", "losses", "metrics", "zoo", "plots",
                 "training", "experiments"]

# 1. Layers (highest first). Independent siblings share a layer.
[[tool.importlinter.contracts]]
name = "core layers"
type = "layers"
layers = [
  "training",
  "zoo | plots",
  "tasks",
  "metrics",            # metrics may import losses (1 existing edge)
  "losses",
  "models | tracking | data_processing",
  "framespec",
  "utils",
]

# 2. tracking is a leaf library
[[tool.importlinter.contracts]]
name = "tracking stays pure"
type = "forbidden"
source_modules = ["tracking"]
forbidden_modules = ["data_processing", "models", "training", "plots"]

# 3. experiments is a sandbox: nothing imports it
[[tool.importlinter.contracts]]
name = "nothing imports experiments"
type = "forbidden"
source_modules = ["utils", "framespec", "data_processing", "tracking",
                  "models", "tasks", "losses", "metrics", "zoo", "plots",
                  "training"]
forbidden_modules = ["experiments"]
```

Cycle breaks / inversions required before the layers contract passes:
- **losses/metrics → tasks inversion** (§3.1): extract `tasks/spec.py` →
  `framespec`; move `align_rps_to_gt` → `losses/pit.py`. After this, the
  `tasks → losses` edges become legal (tasks bundles losses/metrics).
- `data_processing → training` (2 edges), `→ tasks` (2), `→ experiments` (2):
  move or invert the few offending imports (identify exactly during phase 2).
- `models → training` (1 edge): relocate the symbol.
Wire `lint-imports` into the pre-commit hook and CI. Adopt contracts
incrementally: start with `forbidden` contracts that already pass, add the
layers contract at the end of phase 2.

## 4. Phases

Order: safety first, then deletions, then extractions, then the new
surfaces, then notebooks. Each phase ends with: full pytest run,
`lint-imports`, and a short commit series (no mega-commit).

### Phase 0 — Guardrails (small, immediate)

1. Test safety: put `tests/training/test_artifacts_r2_integration.py` behind
   `@pytest.mark.network` with `-m "not network"` in `addopts`; apply the
   `slow` marker to `test_rps_regression`, `test_generated_noise`,
   `test_loop`, `test_ckla` heavy parts, or delete the marker.
2. Wire `scripts/validate_experiment_docs.py` into a real pre-commit config,
   or delete the false claim from its docstring and docs.
3. Add the first import-linter contracts that pass today (forbidden:
   `src → scripts`, `src → notebooks`, "nothing imports experiments" after
   fixing the 2 `data_processing → experiments` edges).
4. Baseline: record current pytest counts and eval numbers to compare after
   each phase.

### Phase 1 — Deletions (~9,000 LOC + 17 notebooks)

1. `src/fwh_rotor_sim/` (1,283 LOC) + `notebooks/fwh_rotor_audio_generator.ipynb`
   + the `load-real-propeller-geometry` skill + `pyproject` wheel entry +
   doc rows. Leave one paragraph in `docs/` pointing to `auraflow`.
2. Dead scripts, per the survey verdict table (26 files, ~6,400 LOC):
   the `/gpfs`-pinned diagnostics, closed research lines (WP16/WP18/WP20
   drivers, michaels_calib runners, beamform/refine-gate probes,
   rps_refinement gen-1 trio, vk_spcup, vk_pseudolabel, repair_librispeech,
   calibrate_rps_jitter, chain_train.sh, cd_iter_sweep.sh, …).
   HOLD: `sr_dp_probe.py` until WP21 closes; `eval_noise_gen_variants.py`
   while the generator campaign is live.
3. Stale notebooks (17): analyze_results, explore_data, inspect_dregon_librimix,
   inference_comparison, dregon_hcqt_viz, rps_measured_vs_command,
   dregon_rps_minimal, classical_baselines_comparison, rps_experiment_results,
   rps_cv_analysis, rps_bad_samples_analysis, rps_evaluation_interactive,
   EVAL_BY_HAND, noise_gen_real_vs_generated, drone_embedding_explorer,
   jasa_gp_interactive, noise_four_way_comparison (+ four_way_lib.py).
4. Doc debt from the same pass: 11 deleted-script names still referenced in
   `docs/experiments/`; the dataset-name drift notes.

### Phase 2 — `src/tracking/` extraction (point 1)

1. Move the 7-module cluster (5,658 LOC): `vk_tracking`, `rps_refinement`,
   `vk_blind_seeding`, `phase_increment_tracker`, `joint_beam_tracker`,
   `rotor_dp`, `warp_refinement`. `rps_corruption` stays in data (it is a
   training-data seam). Resolve `MIXER`: move the control-mode mixer math to
   `tracking/rotors.py`; `rps_synthesis` imports it from there.
2. Promote the leaked privates that cross the new boundary
   (`_fft_workers`, `_second_diff`, `_zoom_lp_decimate`, …) to public names.
3. Add `tracking/stages.py` (Frame adapters, §3.2) and
   `tracking/pipelines.py` (vit2dsp ladder, blind-seed arms, cd_iter chain —
   promoted from `vk_blind_annotation.py` / `rps_refine_lab.py`).
4. Move the script-resident libraries into src per the survey map:
   protocol windows + `build_preps`/`fullrange_init` + `beatvk_eval.score_recording`
   → `tracking/protocols.py` + `tracking/scoring.py` (or `data_processing/`
   for the window providers — decide by import direction);
   `phase_noise_cov/estimate.py` → `tracking/phase_noise.py`.
5. Update ~40 script import sites, 9 test files, the `vk_blind_sweep`
   module-identity guard, AGENTS.md rows. Remove the 3 `sys.path` hacks in
   tests.
6. The losses/metrics ↔ tasks inversion (§3.1): extract `framespec`, move
   `align_rps_to_gt`, update the 16 import sites.
7. Fix the remaining package cycles (§3.5) and land the layers contract.

### Phase 3 — zoo (points 5 + 6)

1. Dict-ify `utils.build_model_from_config`'s if/elif chain; delete the
   ZFTurbo types with no `conf/model` reference (survey says most).
2. Build `src/zoo/` (§3.3): R2 lister, gitignored `.checkpoints-cache.json`,
   incremental refresh. Acceptance: all 62 previously-hardcoded URIs are
   discoverable through `zoo.checkpoints()`.
3. `FrameModel` wrapper; `eval.py` r2 support + metrics upload to
   `artifacts/<exp>/eval/` via `ArtifactStore`.
4. Delete the half-dead `tasks/checkpoints.py::load_model` suppression
   branch; point everything at `zoo.load`.
5. Write the missing `src/tasks/speech-enhancement/AGENTS.md` + update the
   tasks table (4 tasks, not 2).

### Phase 4 — plots (point 4)

Per §3.4. Acceptance: one notebook cell `plots.dwym(frame)` renders the
right figure for an SE pair, an RPS frame, a salience frame, a noise-gen
pair, and a `{label: frame}` dict — with audio players in the notebook and
bare figures outside it.

### Phase 5 — scripts consolidation (point 2)

Build the six generic tools from the survey, then delete what they replace:
- `scripts/se_eval.py` (replaces eval_se_perclip/models/anchors,
  diagnose_dcunet_loss, the 2 shell loops)
- `scripts/table.py` (replaces f1_tables, f2_ladder_table,
  se_valid_composition, the report/fit table scripts)
- `scripts/gridrun.py` + `src/utils/gridrun.py` (the restartable parallel
  unit-JSON harness, replaces 15 hand copies; port `sr_dp_probe` first)
- `scripts/bench.py` (replaces the 4 bench_* + vk_bench_opt_job.sh)
- `scripts/rps_eval.py` (replaces rps_refiner_eval, neural_seeded_vk,
  pi_kalman_protocol, neural_reanchor scoring halves; wraps
  `tracking.protocols` + `zoo`)
- `scripts/probe_ckpt.py` (replaces the ckpt-probe one-offs)
End state: `scripts/` ≈ 2,800–3,100 LOC (from 29,513; ≈90% reduction).
Decide on retiring `sbatch.sh`/`sync_results.sh` (326+93 LOC, doc-referenced).

### Phase 6 — notebook primitives + notebooks (points 7, 8, 9)

1. `src/utils/explore.py` (or `data_processing/explore.py`): metadata → 
   rich table rendering, grid views over dataset spans, sample pickers over
   dload/frames datasets. Built on `plots.dwym` for per-sample display.
2. New focused notebooks, each a thin driver over the primitives:
   - `notebooks/rps_tracking.ipynb` — data + tracking stages + zoo predictors
   - `notebooks/noise_generation.ipynb` — successor of generator_lab
   - `notebooks/speech_enhancement.ipynb` — **left UNWRITTEN** (see §6)
   - keep: `generalization_explorer`, `se_baselines_explorer` (until the SE
     notebook replaces them), `geom_calibration`, `stage0_rotor_rtf`,
     `michael_data_analysis`, `visualize_models` (audit imports)
3. The tutorial document for the unwritten notebook: how to list datasets,
   pull a span, list checkpoints, instantiate a FrameModel, run it, call
   `plots.dwym`, and compose tracking stages. You build the notebook by hand
   and report every friction point; we fix ergonomics from that list.

### Phase 7 — tests + docs closeout

1. Apply the test-audit verdicts: move `tests/utils/plots/` → `tests/plots/`;
   fix stale docstrings; rewrite/delete `test_cli.py`; add a happy-path
   checkpoint-load test (now against `zoo.load`); promote or delete the
   `notebooks/geom_calibration.py` test target.
2. New tests: tracking stages (Frame contract round-trip), zoo listing +
   load, `plots.dwym` dispatch table.
3. Update all AGENTS.md files (root, data_processing, models, tasks, tests,
   notebooks) + CLAUDE.md routing table to the new structure.
4. Final import-linter tightening; record the friction-report fixes.

## 5. LOC accounting (estimate)

| Change | LOC |
|---|---|
| Delete fwh_rotor_sim | −1,283 |
| Delete dead scripts | −6,400 |
| Scripts → generic tools | −7,200 |
| Scripts libraries → src (with dedup) | −8,300 (4,500 relocated) |
| Plots consolidation | −800 |
| Legacy model types + dead loaders | −500 |
| New code (tracking adapters, zoo, dwym, explore, tools) | +3,500 |
| **Net (src+scripts, excl. notebooks)** | **≈ −21,000** |

Plus ~17 notebooks deleted and their MB-scale outputs.

## 6. Decisions (resolved in review, 2026-08-04)

1. **No `data_processing` → `data` rename.** The name stays.
2. **losses/metrics sit BELOW tasks.** Tasks import and bundle them; the
   spec vocabulary moves to `framespec` (§3.1).
3. **Zoo index is a gitignored cache populated from R2 listing** with
   incremental refresh — not a git-committed file (§3.3).
4. **`plots.dwym` coerces raw frames** into the common form with printed
   warnings (§3.4).
5. **Unwritten tutorial notebook: `speech_enhancement.ipynb`** (per plan
   recommendation, accepted).
6. **Retire `sbatch.sh` + `sync_results.sh`** in phase 5; update
   CLAUDE.md/skills to omnirun-only.
7. **Protocol windows split**: window specs in `tracking/protocols.py`,
   recording loading injected — keeps the tracking-purity contract.
8. **`experiments/` stays a contract-fenced sandbox**; delete
   `kalman_harmonic` (research line already killed).

## 7. Execution notes

- Everything happens on this worktree branch (`worktree-refactoring`);
  merge to main per phase, not at the end.
- Phases 2–5 each get a dedicated verification step: pytest, lint-imports,
  one end-to-end smoke (`train.py` dry-compose over all experiments via
  `check_experiment_configs.py`, one `eval.py` run on a small valid set).
- Memory/docs updates ride with each phase (Rule 6), not at the end.
