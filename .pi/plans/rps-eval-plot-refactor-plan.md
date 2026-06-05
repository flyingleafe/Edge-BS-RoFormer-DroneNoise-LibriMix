# Plan: Task-Separated Evaluation & Plotting — RPS Prediction first

## Why

Evaluation and plotting code for the RPS-prediction task is scattered across
**~15 root scripts** and **≥6 notebooks**, each re-implementing the same three
primitives — *load model → run inference → (compute metrics | draw figure)* —
with subtle, undocumented divergences (e.g. GT↔pred frame alignment by
`min-length crop` vs `F.interpolate`; R² macro-per-sample vs global; three
different `compute_metrics`). The result: duplicated bugs, irreproducible
figures, and no single command to compare a new checkpoint against baselines.

This is not a "rewrite the scripts" task. It is an **interface** task
(AGENTS.md philosophy #4): the right top-level abstraction is the **ML task**.

## The organizing principle: *task = a type of function the model approximates*

| Task | Signature | Variants |
|------|-----------|----------|
| Noise suppression | `f: Audio → Audio` | RPS-assisted: `f: (Audio, RPS) → Audio` |
| **RPS prediction** (this plan) | `f: Audio → RPS` | classical (non-learned) predictors share the same signature |

For each task there is **one** standardized way to *evaluate* (a fixed metric
set + per-sample computation + aggregation) and **a small fixed menu** of ways
to *visualize* (inspect inputs/outputs side-by-side; present/compare aggregate
metrics). Everything else is a thin caller.

Both surfaces (eval + plot) must work identically as a **Python API** (for
notebooks/scripts) and a **CLI** (for one-shot reproducible artifacts).

## Research-before-build gate (done)

- **CLI framework** → `typer` (already a project dependency; `postdoc` uses
  it). Do **not** hand-roll argparse. ✔
- **Aligned audio↔RPS algebra** → `src/utils/data` (`UniformSeries`,
  `TimeFrame`) already exists and is the canonical home for "audio + co-recorded
  telemetry". RPS-on-frame-grid and GT-resampling belong there, not re-derived
  per script. ✔
- **Metrics** → no off-the-shelf RPS-tracking metric library exists; a single
  ~50-LOC canonical module is justified (replaces 3+ copies). ✔
- **Plot dispatch** → a plain `dict[str, callable]` registry + typer; no plugin
  framework. ✔

---

## Target architecture

### 0. Task-agnostic foundations (shared by *all* tasks)

These are not RPS-specific and must not live inside the RPS module.

**`Model@<ckpt>` loader — `src/tasks/checkpoints.py` (or `src/utils/checkpoints.py`).**
Loading a model is the same operation for every task: *instantiate a class,
`load_state_dict`*. The loader is therefore **task-agnostic**:
```python
load_model("DCCRN@results/.../best.pt")  # -> nn.Module, ready for any task
```
- **Today** (bare `state_dict` checkpoints): spec is `Type@ckpt`; the registry
  resolves `Type` → class + default constructor args. `Type` is required
  because current `best.pt` files carry no class info.
- **wandb-artifact checkpoints already carry `model_type` in artifact
  `metadata`** (see `train.py:_log_best_checkpoint_artifact`). So a spec like
  `wandb://<run>/model:best` can resolve the type *without* `@Type`.
- **Future (backward-compatible): "extended checkpoints"** — a checkpoint that
  embeds `{class_path, constructor_kwargs, state_dict}`. Then `load_model(ckpt)`
  needs no `Type@`. The loader must accept all three forms; older bare files
  keep working unchanged. *This is a separate, later increment — not blocking.*
- The two model registries to unify under one resolver: suppression/encoder
  models in `utils.get_model_from_config`, and RPS models in
  `train_rps_predictor.MODEL_REGISTRY`. (Classical RPS predictors are *not*
  `nn.Module` + state_dict; they register as `@`-less names — see §A.)

### A. Evaluation — `src/tasks/rps_prediction.py`

The RPS-prediction **task module**: a single file exposing a Python API **and**
feeding the shared typer CLIs. (A thin shim `src/evaluate_rps_predictor.py` may
re-export for discoverability, but the substance lives here.) Responsibilities,
each previously duplicated:

1. **`RPSPredictor` is a `typing.Protocol`, not a wrapper class** — structural
   interface `predict(audio, sr) -> (n_rotors, n_frames)`. A learned `nn.Module`
   whose `forward` does this, a classical function, or any future object all
   satisfy it *without inheritance or hand-wrapping*. The single user-facing
   factory is **`load_predictor(spec)`**, which owns learned-vs-classical
   dispatch and is **idempotent**:
   - already a predictor → returned as-is (no double-wrap);
   - `"Type@ckpt"` / path → calls the §0 task-agnostic `load_model` (returns a
     bare `nn.Module`) and wraps it in a tiny adapter handling
     eval/no-grad/batching/`(R,F)`-numpy-out (task knowledge that does **not**
     belong in `load_model`);
   - `"cepstral" | "hps" | "pyin" | "matched_filter" | "nmf"` → classical fn from
     `classical_rps_predictors.py`.
   Learned and classical are thus interchangeable inputs to eval and plots, and
   the caller never writes `RPSPredictor(load_model(...))`.
2. **Input set = `Iterable[TimeFrame]`** (no bespoke sample dataclass, no id
   tuple). The `TimeFrame` (from `utils.data`) carries:
   - `tracks={"audio": UniformSeries@sr, "rps": EventSeries}` — RPS is stored as
     an **`EventSeries`** (timestamped motor telemetry at its native rate), *not*
     a pre-resampled array. This is the honest representation and is precisely
     what the timestamp canon consumes; a pre-stretched array would lose the
     timestamps and re-introduce the shape-stretch trap.
   - `tags={"id": "sample_00299", "input_snr": -12.3, "recording_id": ...}` —
     time-invariant sample metadata in the new `TimeFrame.tags` field (§E). The
     **`id`** keys per-sample `EvalResult` rows and figure captions; storing
     **`input_snr`** here means per-SNR stratification reads `frame.tags` and
     needs **no external `metadata.json` join** (unlike `compute_rps_per_snr.py`
     today). Loaders populate tags once, from the dataset's `metadata.json`.
   Structural mismatches (missing `audio`/`rps`, absent `id`) surface as
   immediate runtime errors. Loaders that produce this iterable:
   - `load_input_set(path)` — standard on-disk format (`sample_*/{mixture.wav,
     rps.npy}` + `metadata.json`); covers DREGON-LM train/valid and any future
     dataset in that layout. **A path is the default addressing — no registry/DSL.**
   - `load_dregon_freeflight()`, `load_dregon_highsnr(...)`,
     `load_external(...)` — plain named functions for the non-directory sets
     (continuous take, real-recording chunk extraction, DJI logs). Same return
     type; not a registry.
3. **Inference + alignment** — one function: run `predictor.predict(audio, sr)`
   → predicted `UniformSeries` at frame rate `sr/hop`; align GT by the
   **timestamp canon** in one line — `frame["rps"].interpolate_uniform(sr=sr/hop)`
   (the new `EventSeries` method, §E). Step-2 (±1-frame) reconciliation = crop
   (non-issue; see Alignment). No `np.interp` stopgap — the `utils.data`
   extension (§E) lands first as a prerequisite.
4. **Metrics — one canonical implementation**: PIT-aware MSE/RMSE, MAE
   (frame & clip), macro-per-sample R², per-rotor. Returns per-sample records.
5. **Aggregation**: overall, per-rotor, **per-SNR stratified** (subsumes
   `compute_rps_per_snr.py`; reads `frame.tags["input_snr"]`, no join), with
   mean/median/std + LaTeX/CSV/JSON emit.

**Python API sketch**
```python
from tasks.rps_prediction import load_predictor, load_input_set, evaluate
preds   = load_predictor("SimpleConv@results/rps_exp/best.pt")  # or "cepstral"
samples = load_input_set("datasets/DREGON-LM/valid")  # Iterable[TimeFrame] (tags carry id, snr)
result  = evaluate(preds, samples)          # EvalResult: per-sample + aggregate
result.per_snr()                            # stratified table (reads frame.tags["input_snr"])
result.to_json("metrics.json")
result.to_wandb(run)                        # tables + summary + artifact (see §D)
```
`evaluate` accepts **only** `Iterable[TimeFrame]` (each frame self-describing via
`tracks` + `tags`); `load_input_set` is a convenience that turns a dataset path
into that iterable. Pass any other iterable (exotic loaders, notebook-built
frames) directly.

**CLI sketch** (`--input-set` takes a **path** for the standard format, or a
bare name resolved to one of the exotic loader functions — a thin CLI-only
convenience, not a Python-API registry)
```
evaluate-rps --input-set=datasets/DREGON-LM/valid \
  --model=SimpleConv@.../best.pt --model=cepstral \
  --metrics=rmse,mae,r2 --stratify=snr \
  -o results/.../metrics.json --tex papers/.../table.tex
```

### B. Plotting — `src/utils/plots/rps_prediction/`

```
src/utils/plots/
  __init__.py            # shared render registry + `make_plot` CLI entry
  rps_prediction/
    __init__.py          # PLOT_TYPES registry (name -> fn) for this task
    sample_comparison.py # spectrogram + GT + per-model prediction rows (+ GT overlay)
    summary_metrics.py   # bar charts (RMSE/MAE/R²) across models, with error bars
    per_snr.py           # metric vs SNR-bin lines/bars across models
    training_curves.py   # train/val MSE + R² from training_log.csv (+ naive baseline)
    full_sequence.py     # 3-panel: spectrogram / pred-vs-GT timeline / per-frame MSE
```

Each plot function: `fn(*, samples|result, models, ax|fig=None, **style) -> Figure`
so it is directly callable in notebooks, and registered by a dotted name
(`rps_prediction.sample_comparison`) for the CLI.

**CLI sketch** (exactly the user's target ergonomics)
```
make-plot --type=rps_prediction.sample_comparison \
  --sample=datasets/DREGON-LM/valid/sample_00299 \
  --model=SimpleConv@.../best.pt --model=DCCRN@.../best.pt \
  -o plot_comparison.pdf
```
Plot types that consume **aggregate** results (`summary_metrics`, `per_snr`)
take `--results=metrics.json` (produced by §A) instead of `--sample`, so
eval and plot compose cleanly.

### C. Wiring
- `[project.scripts]` in `pyproject.toml`: add `make-plot` and `evaluate-rps`.
- Keep `load_model` (§0) + `RPSPredictor` (§A) as the single seam shared by
  eval and plotting (a plot that runs a model calls the same loader; it does
  **not** re-load checkpoints itself).
- Shared plumbing (model-spec parse, input-set loader, inference) lives in §0/§A;
  plotting imports it. No inference logic duplicated in `plots/`.

### D. wandb as the eval-results store (research findings)

Current wandb usage: scalar `wandb.log` during training; best checkpoint as a
`wandb.Artifact(type="model")` with `metadata={model_type, best_metric}`
(`train.py`). No Tables/Images logged yet. `docs/data-and-artifacts.md` already
declares “Metrics/logs → wandb; eval audio → local”. Evaluation results should
slot into that contract via four wandb primitives:

| EvalResult part | wandb primitive | Why |
|------|------|------|
| headline aggregate (RMSE/MAE/R²) | `run.summary[...]` | shows in the runs table; sortable/filterable across models |
| per-sample + per-rotor + per-SNR rows | `wandb.Table` | queryable, sortable, **comparable across runs** in the wandb UI; powers built-in `wandb.plot` bars/lines without us shipping matplotlib |
| standard figures (§B) | `wandb.Image(fig)` | inline visual diff between checkpoints |
| full `metrics.json` + `per_sample.json` | `wandb.Artifact(type="eval")` | versioned, content-addressed, pullable by notebooks via `run.use_artifact` — mirrors the model-artifact pattern |

**Run association — resolved: dedicated eval run.** Each `evaluate-rps`
invocation creates its **own** run that `use_artifact`s the model checkpoints it
evaluates (lineage edges eval-run → model-A, model-B) and logs one comparison
Table keyed by model spec + summary + figures. Rejected the alternative (log
into each model's *training* run): it mutates finished runs, scatters a
cross-model comparison across pages, and has no home for classical baselines.
**Decider:** we routinely **re-evaluate trained models on new data** — that must
not keep appending to / corrupting the original training run; a fresh eval run
per evaluation is the clean unit (one run = one comparison).

`EvalResult.to_wandb(run)` encapsulates this so the local-JSON path
(`results/<exp>/eval/<input-set>/`) and the wandb path stay one code path with
different sinks. Local layout (decision §3) remains the source of truth; wandb
is an additional, optional sink — not a replacement.

### E. `utils.data` extension (prerequisite increment — lands before §A)

The task layer leans on two `utils.data` additions. They are general, belong in
the library, and ship **first**, each with Hypothesis property tests matching
the module's exactness ethos (see `src/utils/data/AGENTS.md`).

**E1 — `TimeFrame.tags`.** New field `tags: Mapping[str, Hashable] = {}` for
*time-invariant* sample metadata (`id`, `input_snr`, `recording_id`). Algebra:
- `slice` / `shift` / `select` / `with_track` → **preserved** (metadata does not
  move with time);
- `concat` → **equality required on shared keys** (raise
  `IncompatibleSeriesError` on conflict), union otherwise — consistent with the
  library's no-silent-merge rule.

**E2 — resampling / interpolation**, built on one shared evaluator:
```python
# base.TimeSeries
def interpolate(self, times, *, kind="linear", fill="clamp") -> np.ndarray
    # value(s) at absolute query times (float s OR int ticks); -> (len(times), *value_shape)
    # UniformSeries: interp between samples;  EventSeries: interp between event values
# UniformSeries
def resample(self, new_sr, *, kind="linear") -> UniformSeries   # uniform->uniform, same domain
# EventSeries
def interpolate_uniform(self, sr, *, t_start=None, t_end=None, kind="linear") -> UniformSeries
```
Resolved semantics:
- **D1** — **no `EventSeries.resample`** (events→events is ill-defined); the
  named op is `interpolate_uniform`.
- **D2** — extrapolation default **`fill="clamp"`** (= `np.interp` endpoint
  hold), with `"nan"`/`"error"` opt-in. Clamp reproduces legacy behavior.
- **D3** — **`kind="linear"`** only for now (matches the legacy `np.interp`
  canon); `"previous"`/step deferred until measured (RPS command is piecewise).
- **D4** — output grid **`phase=0`**, sample `k` at `t_start + k/sr` — reproduces
  legacy `stft_times = arange(F)*hop/sr`, keeping the canon bit-comparable to
  golden numbers. Multi-channel `(M, R)` interpolates per-channel along axis 0.

With E2, the RPS alignment canon is exactly `frame["rps"].interpolate_uniform(
  sr=audio_sr/hop)` — no task-layer `np.interp` stopgap.

---

## Script inventory → replacement map

### Plotting scripts (→ `src/utils/plots/rps_prediction/` + `make-plot`)

| File | What it does | Replaced by |
|------|--------------|-------------|
| `plot_rps_samples.py` | spectrogram + target + predicted RPS per sample (.npz) | `rps_prediction.sample_comparison` |
| `plot_rps_comparison_long.py` | GT on top + 3 model rows, GT dotted overlay | `rps_prediction.sample_comparison` (multi-model) |
| `plot_rps_comparison_with_spectrogram.py` | spectrogram + GT + 3 model rows | `rps_prediction.sample_comparison` (`--spectrogram`) |
| `generate_rps_comparison_plots.py` | per-sample compare fig + summary bar + timeseries | `sample_comparison` + `summary_metrics` |
| `generate_rps_comparison_table.py` | RMSE/MAE/R² table | `evaluate-rps --tex/--csv` (§A) |
| `generate_rps_slides.py` | summary bar charts from summary json | `rps_prediction.summary_metrics` |
| `plot_rps_training.py` | train/val MSE + R² curves | `rps_prediction.training_curves` |
| `plot_per_snr.py` (RPS part) | per-SNR metric lines | `rps_prediction.per_snr` (suppression part stays for that task) |
| `analyze_rps_full_sequence.py` (plot half) | 3-panel full-take figure | `rps_prediction.full_sequence` |

### Evaluation / inference scripts (→ `src/tasks/rps_prediction.py` + `evaluate-rps`)

| File | What it does | Replaced by |
|------|--------------|-------------|
| `eval_rps_val.py` | inference over full DREGON-LM val + metrics.json | `evaluate(... input-set=dregon-lm:valid)` |
| `evaluate_rps_predictor.py` (root) | eval best model on 5 val + external | input-sets `dregon-lm` + `external` |
| `evaluate_rps_predictor_samples.py` | 3 models on random val samples + plots | eval + `sample_comparison` |
| `evaluate_rps_long_samples.py` | eval on ~8 s long samples | input-set variant / `--duration` |
| `compute_rps_per_snr.py` | per-SNR stratified metrics + tex | `result.per_snr()` / `--stratify=snr` |
| `analyze_rps_high_snr.py` (eval half) | high-SNR chunk extraction + eval | input-set `dregon-highsnr` |
| `analyze_rps_full_sequence.py` (eval half) | inference on full take + per-frame MSE | input-set `dregon-freeflight` |
| `extract_long_rps_samples.py` / `extract_specific_rps_samples.py` | pre-extract sample subsets | folded into input-set loaders (selection args) |
| `generate_rps_samples.py` | save .npz inference triples | replaced by in-memory `EvalResult` (no on-disk .npz hop) |
| `classical_rps_predictors.py` | classical baselines | **kept**, wrapped by `RPSPredictor` protocol; registered as `@`-less specs |

### Notebooks (de-duplicate by importing the new API)

`rps_evaluation_interactive.ipynb`, `rps_experiment_results.ipynb`,
`rps_bad_samples_analysis.ipynb`, `rps_cv_analysis.ipynb` currently re-define
`compare_all_models`, `compute_metrics_interpolated`, bar charts inline. After
the refactor they `from evaluate_rps_predictor import ...` /
`from utils.plots.rps_prediction import ...`. No notebook keeps its own
inference/metric/plot code.

---

## Decisions

**Resolved (user):**
- **`Model@ckpt` loader is task-agnostic** — lives in `src/tasks/checkpoints.py`,
  shared by every task; extended (self-describing) checkpoints come later,
  backward-compatibly (§0).
- **Task module location** — `src/tasks/rps_prediction.py` (not a root
  `evaluate_*` script).
- **Local results layout** — `results/<exp>/eval/<input-set>/` is good for now;
  wandb is an additional sink (§D).
- **Predictor API** — `RPSPredictor` is a `typing.Protocol`; one idempotent
  factory `load_predictor(spec)` (no hand-wrapping `RPSPredictor(load_model(...))`).
- **Input set** — `evaluate` accepts **only** `Iterable[TimeFrame]`; no bespoke
  dataclass, no id tuple. Each frame is self-describing: `tracks={audio:
  UniformSeries, rps: EventSeries}` + `tags={id, input_snr, ...}`. Datasets are
  **path-addressed** (`load_input_set(path)`); exotic sets are plain named
  functions — no registry/DSL.
- **`utils.data` extension (§E, lands first)** — add `TimeFrame.tags`
  (time-invariant; concat requires equality on shared keys) and resampling:
  `TimeSeries.interpolate`, `UniformSeries.resample`,
  `EventSeries.interpolate_uniform`. **D1** no `EventSeries.resample`; **D2**
  `fill="clamp"` default; **D3** `kind="linear"` only; **D4** output `phase=0`.
  RPS canon = `frame["rps"].interpolate_uniform(sr=audio_sr/hop)`.

**Resolved (investigation — see “Alignment” section below):**
- **Frame-alignment canon = timestamp-based interpolation** (`np.interp` on
  real motor-time vs STFT-frame-center grids), implemented via `utils.data`
  exact resampling. This replaces the legacy **shape-stretch**
  (`F.interpolate(size=n_frames)`) used in training + golden eval, which is only
  *accidentally* correct on DREGON-LM (equal spans) and silently misaligns when
  motor/audio spans differ (e.g. the 47 s-motor / 53 s-audio free-flight take).
  The ±1-frame *pred-vs-target reconciliation* (crop vs re-interp) is declared a
  **non-issue** (sub-permille) — pick crop for simplicity.

- **`Type@ckpt` vocabulary** — `Type` names are exactly the `MODEL_REGISTRY`
  keys (e.g. `simple_conv`, `dccrn_enc_rps`). Bare `best.pt` carry no class
  info, so the CLI passes `Type` until extended checkpoints land.

- **wandb run association** (§D) — **dedicated eval run** per `evaluate-rps`
  invocation (not logging into the model's training run); decided by the
  re-evaluation-on-new-data workflow, which must not mutate training runs.

**All design decisions resolved — ready to implement per Sequencing.**

## Alignment: two independent steps (investigated — the canon rests on this)

The legacy code conflates *two* unrelated operations under “interpolate vs
crop”. They must be treated separately.

**Step 1 — GT RPS (native motor rate ~929 Hz) → STFT frame grid.** Raw `rps.npy`
is stored at motor rate, co-extensive with the audio chunk
(`create_dregon_librimix.py` slices both to the same span); every consumer
resamples it. Two methods coexist:
- **(A) timestamp-based** — `np.interp(stft_times, motor_times, rps)` on the
  *real* grids `motor_times=arange(N)/motor_sr`, `stft_times=arange(F)*hop/sr`.
  Used by `analyze_rps_full_sequence.py`, `analyze_rps_high_snr.py`.
- **(B) shape-stretch** — `F.interpolate(rps, size=n_frames)`, which ties array
  *endpoints* and ignores timestamps. Used by **`train_rps_predictor.py`
  (training targets)**, `dataset.py`, `train.py`, `valid.py`,
  `generate_rps_samples.py`, `generate_rps_comparison_plots.py`, `eval_cross.py`.

(A) and (B) agree only when motor and audio spans match. On DREGON-LM they do
(→ sub-frame difference, from `align_corners=False` + N-1-vs-N spacing), so the
shape-stretch is *accidentally* fine — and training + the golden
`eval_rps_val.py` are mutually consistent (both (B)). On the free-flight take
(motor ≈47 s, audio ≈53 s) (B) misaligns by seconds; that is *why* the analysis
scripts use (A). **Canon = (A)**, implemented via `utils.data` exact tick
resampling (its raison d'être), which subsumes both `np.interp` and the
shape-stretch.

**Step 2 — model output `T_pred` vs GT-on-grid `T_target`, off by ±1 frame**
(head pooling). Reconciled by *crop to min* (`eval_rps_val.py`, classical, all
`plot_rps_comparison_*`) or *re-interp GT to `T_pred`*
(`generate_rps_comparison_plots.py`). ±1 of ~256 frames → **non-issue**;
standardize on crop.

## Beyond this plan: deduplicating train + eval loops (user intent, noted)

The end goal is broader than eval/plot. Training and evaluation loops are
*structurally identical across tasks* — dataloading, the main loop,
backprop/optimizer step (train-only), and wandb logging are all the same,
differing only by parametrization (model, dataset, loss, metric set). The
`tasks/` package is the seam for that future unification:
- a `Task` captures `{model adapter, dataset/input-set loader, loss, metric
  set, logging}`;
- one shared `train_loop` / `eval_loop` consumes a `Task`, replacing the
  per-task `train.py` / `train_rps_predictor.py` / `eval_*` divergence.

This refactor (RPS eval + plot) is deliberately the **first slice** of that
larger move: it establishes the task-agnostic loader (§0), the per-task
`RPSPredictor` adapter, the canonical metric set, and the input-set loader —
exactly the components a unified loop will later consume. We are *not* building
the unified loop now (no premature abstraction), but every interface here is
chosen so it drops into one without rework.

## Verification — golden-artifact regression (the acceptance gate)

The refactor is **correct iff it reproduces the existing published artifacts
from the same checkpoints**. The repo already contains the golden outputs to
regress against — no new ground truth needed. Verification is a *script*
(`tests/tasks/test_rps_regression.py` + a `make verify-rps` target), run before
any legacy file is deleted.

### Golden artifacts inventory (what we diff against)

| Golden artifact | Path | What it pins |
|------|------|------|
| Per-sample DREGON-LM valid metrics | `results/rps_predictor_comparison/val_inference/per_sample_metrics.json` (600 rows: `mse,mae_frame,mae_clip,ss_tot,r2`) | the **strongest** test: exact inference + metric math, per sample |
| Per-SNR stratified table | `results/rps_predictor_comparison/per_snr_metrics.{json,csv}` | aggregation + SNR bucketing |
| Paper per-SNR LaTeX table | `papers/rps-from-drone-sound/figures/rps_per_snr_table.tex` (Overall: MSE 5.15, RMSE 2.27, MAE/frame 1.36, MAE/clip 0.56, R² 0.8353) | the **published numbers** — must reproduce digit-for-digit |
| Autoresearch-attempt aggregate | `results/rps_extra_evals_aggregate.json` + `results/rps_extra_evals_report.md` | 10 SimpleConv variants × {full_sequence, single_rotor, high_snr}; e.g. `simple_conv` full-seq MSE 104.14 / R² 0.7724, in-flight MSE 24.45 |
| Autoresearch checkpoints | `results/rps_exp_*/best_*.pt` | the inputs that produce the above |
| Paper figures | `papers/.../figures/{fig_full_sequence,fig_highsnr_outlier,fig_qualitative_combined,fig_training_curves,rps_per_snr}.{pdf,png}` | plotting (§B) outputs |

### Method (numeric — the load-bearing half)

1. **Pin determinism.** All regression runs on `device=cpu`, fixed seeds,
   single-threaded; no cudnn nondeterminism. RPS inference is feed-forward so
   CPU is bit-stable across runs.
2. **Per-sample first, aggregate second.** Re-run `evaluate(SimpleConv@<the
   checkpoint behind the golden file>, dregon-lm:valid)` and assert the new
   per-sample table matches `per_sample_metrics.json` within `rtol=1e-4,
   atol=1e-4`. Per-sample equality is far stricter than aggregate equality
   (600 independent assertions vs 1) and localizes any divergence to a sample.
3. **Reproduce the paper table** from those per-sample rows through
   `result.per_snr()` and assert each cell equals `rps_per_snr_table.tex`
   (rounded to the printed precision — these are the numbers in the paper).
4. **Autoresearch sweep.** For each `results/rps_exp_*/best_*.pt`, re-run the
   matching input-set (`dregon-freeflight`, single-rotor, `dregon-highsnr`) and
   assert against `rps_extra_evals_aggregate.json` (`mse,mae,r2,*_inflight`).
   This is the “re-evaluate autoresearch experiments on old DREGON-LM and see
   results match” check the user asked for, across all 10 variants.
5. **Report regeneration.** Re-emit `rps_extra_evals_report.md` and
   `rps_per_snr_table.tex` and `diff` against the committed files (numeric
   cells only; ignore whitespace/date lines).

### Method (figures — the visual half)

Pixel-diffing matplotlib PDFs across versions is brittle. Two-layer check:
- **Data layer (asserted):** every plot function returns/accepts the arrays it
  draws; the regression test compares *those arrays* to the legacy script’s
  arrays (exact). A figure that draws the same data as before is correct
  regardless of cosmetic drift.
- **Visual layer (spot-check):** regenerate the 5 paper figures via `make-plot`
  and eyeball them against the committed `.png`s using the
  `examine-presentation-slides` / `improve-plot-visibility` skills. Not
  automated; a human/agent sign-off, recorded in the closing notes.

### The alignment-canon caveat (must be handled explicitly)

The canon switches Step-1 alignment from legacy **shape-stretch (B)** to
**timestamp-interp (A)** (see “Alignment” section). On DREGON-LM these differ
only **sub-frame**, so the golden numbers should reproduce to tight tolerance —
but not necessarily bit-identical. Protocol:
1. First reproduce with **legacy shape-stretch (B)** + crop, exactly as each
   golden file was generated → expect **bit-identical** (`rtol=1e-4`). This
   proves the refactor is a faithful move, not a behavior change.
2. *Then* switch to canon **(A)**, re-run, and **record the delta** (expected
   sub-frame, i.e. tiny on DREGON-LM; potentially large only on span-mismatched
   recordings, where (A) is the *correct* answer and (B) was wrong). Treat as an
   intentional, documented change; re-baseline golden files in a separate,
   labeled commit after the user approves the delta.
3. Step-2 reconciliation stays *crop* throughout (non-issue), so it never
   contributes to either comparison.

This separation (faithful-move proof → intentional-improvement delta) prevents a
real bug from hiding inside “we changed alignment anyway”.

## Sequencing

1. All design decisions resolved (see Decisions). Proceed.
2. **Build the regression harness first** (against current scripts/golden
   files), so later steps have a red/green signal from the outset.
3. **Land the `utils.data` extension (`§E`)**: `TimeFrame.tags` + `interpolate` /
   `resample` / `interpolate_uniform`, each with Hypothesis property tests.
   Prerequisite for the canon.
4. Land the eval seam (`§A` + `§0`): loader + protocol + input-set loaders +
   inference + metrics + aggregation, thin CLI. Port `eval_rps_val.py` first;
   green on per-sample DREGON-LM regression (legacy alignment) before moving on.
5. Run the autoresearch sweep regression → green across 10 variants. Then flip
   alignment canon, record deltas, re-baseline if approved.
6. Land plotting (`§B`) one type at a time; each green on its data-layer
   regression + visual spot-check. Order: `sample_comparison`,
   `summary_metrics`, `per_snr`, `training_curves`, `full_sequence`.
7. Regenerate paper table + report + figures; diff vs committed.
8. Rewire notebooks to import; delete legacy scripts in one commit per group
   (only after their regression is green).
9. Update root `AGENTS.md` skills table + add a `docs/` note; record gotchas.

## Success criterion

1. **Reproduction:** `make verify-rps` is green — per-sample DREGON-LM metrics,
   the paper per-SNR table, and all 10 autoresearch-variant aggregates match
   their golden files (within tolerance, modulo the documented alignment
   delta); the 5 paper figures regenerate and pass visual spot-check.
2. **Reduction:** ≥90% of RPS plotting/eval LOC in root scripts + notebooks
   deleted, replaced by imports of `tasks.rps_prediction` /
   `utils.plots.rps_prediction` + two CLI entry points; every legacy figure
   reproducible by one `make-plot` / `evaluate-rps` invocation.
