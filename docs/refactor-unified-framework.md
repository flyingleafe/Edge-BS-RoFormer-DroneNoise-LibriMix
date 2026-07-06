# Unified Framework Refactor — Binding Design (2026-07-02)

This document is the **contract** for the repo-wide refactor executed on branch
`refactor/tdseries-framework`. Every implementation agent reads this before
touching code. Deviations require an explicit note in the final report.

## Goals

1. Replace the in-repo `src/utils/data` timeseries library with the PyPI
   package **`tdseries`** (>=0.1.0), installed like any other dependency.
2. Dataset adapters producing `tdseries.Frame` iterables live in
   `src/data_processing/` (a `src/` package like any other).
3. All losses/metrics in `src/losses/` + `src/metrics/`; all plotting in
   `src/plots/`. DRY — one canonical implementation of everything.
4. Exactly one `train.py` and one `eval.py` at repo root, driven by
   **Hydra** composable configs. A pre-run **spec validation** checks that
   data ⊇ model input, and model output (∪ data) ⊇ every loss/metric.
5. Every experiment = one git-committed YAML in `conf/experiment/`, one
   wandb run named after it, one `results/<experiment>/` subfolder.

Non-goals (v1): rewriting model internals (they stay tensor-based
`nn.Module`s); GPU re-runs of historical experiments; migrating
`fwh_rotor_sim` (it never used `utils.data`).

## Final layout

```
conf/                      # Hydra config tree (NEW)
  config.yaml              # root defaults + run conventions
  data/                    # dataset source configs
  data/online_mix/         # online-mix policy YAMLs (moved from configs/online_mix_*.yaml)
  model/                   # model configs (task + architecture + _target_)
  loss/                    # loss configs
  metrics/                 # metric-suite configs
  optim/                   # optimizer + scheduler configs
  logging/                 # wandb settings
  experiment/              # ONE FILE PER EXPERIMENT (composition via defaults)
src/
  data_processing/         # moved from ./data_processing — Frame-producing adapters + torch Datasets
  losses/                  # NEW consolidated losses
  metrics/                 # NEW consolidated metrics
  plots/                   # moved from src/utils/plots — all plotting
  tasks/                   # task specs (FrameSpec) + task registry + codecs
  training/                # NEW: loop, checkpointing, wandb, optim factory
  models/                  # unchanged (tensor-based nn.Modules + registry)
  utils/                   # legacy ZFTurbo helpers only; src/utils/data DELETED
train.py                   # the only training entry point
eval.py                    # the only evaluation entry point
results/<experiment>/      # one subfolder per experiment
```

Deleted at the end: `src/utils/data/`, `tests/utils/data/`, `valid.py`,
`final_valid.py`, `train_rps_predictor.py`, `train_noise_gen.py`,
`train_noise_generation.py`, `eval_cross.py`, `eval_narrow_sr.py`,
`plot_per_snr.py`, `generate_comparison.py` (absorbed into `eval.py` +
`src/plots`), legacy batch scripts (`rps_experiment.sh` etc.).
Dataset *creation* scripts (`create_dataset.py`, `create_dregon_librimix*.py`)
stay at root — they are offline one-shot tools, not train/eval.

## tdseries migration guide (utils.data → tdseries)

`import tdseries as td`. Full API: `~/Research/PhD/projects/tflib/DESIGN.md`
(package installed from PyPI; same code).

| Old (`utils.data`) | New (`tdseries`) |
|---|---|
| `UniformSeries.from_samples(x, sr, t_start=t)` | `td.uniform(x, sr, dims=("channel","time"), t_start=t)` — name real dims! audio `("mic","time")` or `("channel","time")`, mono `("time",)` |
| `EventSeries.from_events(ts, vals, t_start=a, t_end=b)` | `td.events(ts, vals, dims=("rotor","time"), t_start=a, t_end=b)` |
| `SegmentSeries(...)` (VAD/labels) | `td.spans(starts, ends, values, ids=...)` |
| `TimeFrame.from_tracks(tracks, tags=T, global_data=G)` | `td.Frame({**tracks, "mic_pos": td.wrap(G["mic_positions"], dims=("mic", None)), "rotor_pos": td.wrap(G["rotor_positions"], dims=("rotor", None)), "meta": td.Frame(T)})` |
| `ts.samples` | `s.data` |
| `ts.sr` | `s.tindex.sr` (float) / `s.tindex.rate` (exact `Fraction`) |
| `ts.t_start`, `.duration`, `.t_end` | same names (float secs); `_ticks` variants exist |
| `ts.slice(a, b)` (secs) | `s.time[a:b]` |
| `ts.slice_ticks(a, b)` | `s.ticks[a:b]` or `s.slice_ticks(a, b)` |
| `ts.shift(dt)` | same (float secs or int ticks) |
| `ts.concat(other)` | same. **`+` is NOT concat** — do not use `+`. |
| `es.abs_timestamps` | `s.tindex.abs_stamps` (StampIndex) |
| `es.interpolate(times)` | `s.interpolate(times)` → np.ndarray (float64), time axis kept in place |
| `us.channel_shape` | `s.shape` minus the time axis: `tuple(n for d, n in zip(s.dims, s.shape) if d != "time")` |
| `tf[key]` | `frame[key]` — same semantics (absolutized copy, frame unchanged) |
| `tf.with_track(k, v)` | `frame.with_entry(k, v)` |
| `tf.select/drop/merge` | same names |
| `tf.tags["x"]` | `frame["meta"]["x"]` (nested invariant Frame survives time ops) |
| `tf.global_data["mic_positions"]` | `frame["mic_pos"].data` (a `td.wrap`-ed Series sharing the "mic" dim) |
| `tf.slice(a,b)` / `.duration` | `frame.time[a:b]` / `frame.duration` |
| `DomainError` | `td.DomainError` |
| `IncompatibleSeriesError` | `td.IncompatibleError` |
| `TICKS_PER_SECOND`, secs↔ticks | `td.TICKS_PER_SECOND`, `td.secs_to_ticks`, `td.ticks_to_secs` |

### Gotchas (load-bearing)

- **Rational sample rates are strict.** `td.uniform(x, sr)` rejects
  non-integral float sr. STFT frame rates must be exact fractions:
  `Fraction(16000, 512)` or the tuple `(16000, 512)` — never `31.25`.
  Never compute `sr = n / duration` as a float.
- **Time axis is named, not positional.** Old code assumed time == last
  axis. Keep time last in `dims`, but always pass explicit `dims` so shared
  dims (`"mic"`, `"rotor"`) line up across entries; that's what makes
  `frame.slice["mic", 0]` slice audio *and* mic_pos together.
- **Metadata**: scalar tags go into a nested `td.Frame({...})` entry named
  `"meta"` (invariant → survives time slicing). Geometry arrays go in as
  `td.wrap` entries with a shared dim name. Convention fixed here:
  entry names `"meta"`, `"mic_pos"`, `"rotor_pos"`.
- Dims named the same must either carry indexes into a shared domain or
  have equal sizes (RangeIndex default) — Frame construction validates
  tree-wide. If sizes differ legitimately, use a different dim name.
- `interpolate` returns a plain ndarray, not a Series. To get a Series on
  a new grid use `s.resample(new_sr)`.
- No `# type: ignore`, no `# noqa` — restructure until ruff+pyright pass
  (a PostToolUse hook enforces this on every write).
- **Frame tags** (`tf.tags["recording_id"]`, `input_snr`, …) become the
  nested invariant `"meta"` Frame. Shared helpers live in
  `src/data_processing/frames.py` (NEW):
  `get_meta(frame, key, default=None)`, `with_meta(frame, **tags)`
  (returns a new Frame with an updated `"meta"` entry), and
  `meta_dict(frame) -> dict`. All ported code goes through these helpers —
  no ad-hoc `frame["meta"]` chains at call sites.
- **Series-level plot tags** (`plot.renderer`, `plot.freqs`,
  `plot.rps_pred` in `src/plots/timeframe/`) do NOT move into the data
  model. The plots package defines its own `PlotTrack` dataclass
  (`series: td.Series`, `renderer: str | None`, `hints: dict`);
  `make_spectrogram_series` / `make_salience_series` return `PlotTrack`s,
  and `plot_timeframe` accepts a mix of raw Series/Frame entries and
  `PlotTrack`s. Renderer dispatch: explicit `renderer` first, else by index
  type (GridIndex 1-D→waveform, 2-D→spectrogram/heatmap; StampIndex→rps;
  SpanIndex→spans).

## Task typing: FrameSpec

New module `src/tasks/spec.py` (pure, numpy-free, no torch import):

```python
@dataclass(frozen=True)
class SeriesSpec:
    dims: tuple[str, ...]                 # ordered dim names, e.g. ("batch","mic","time")
    time: Literal["grid", "stamps", "spans"] | None = "grid"
    rate: tuple[int, int] | None = None   # exact rate constraint, None = any
    dtype: str | None = None              # e.g. "float32"; None = any floating

@dataclass(frozen=True)
class FrameSpec:
    entries: Mapping[str, SeriesSpec | FrameSpec]
    optional: frozenset[str] = frozenset()

def spec_of(frame: td.Frame) -> FrameSpec: ...          # infer from a live Frame
def check_subsumes(provided: FrameSpec, required: FrameSpec) -> list[str]
    # [] == ok; else human-readable mismatch messages
```

`src/tasks/task.py`:

```python
@dataclass(frozen=True)
class Task:
    name: str
    input_spec: FrameSpec    # batched ("batch" leading dim)
    output_spec: FrameSpec

TASKS: dict[str, Task]  # speech_enhancement, rps_prediction, salience_rps, noise_generation
```

Canonical task signatures (entry names are the contract):

- `speech_enhancement`: in `{mixture: (batch,[mic,]time)@grid, rps?: (batch,rotor,time)@grid}`
  → out `{enhanced: (batch,[mic,]time)@grid, rps_pred?: (batch,rotor,time)@grid}`
- `rps_prediction`: in `{mixture: (batch,time)@grid}` → out
  `{rps_pred: (batch,rotor,time)@grid}` (frame rate = sr/hop as exact fraction)
- `salience_rps`: same input → `{salience: (batch,freq,time)@grid}`
- `noise_generation`: in `{rps: (batch,rotor,time)@grid, mic_pos, rotor_pos, drone_id}`
  → out `{audio: (batch,mic,time)@grid}`

Models stay tensor `nn.Module`s. Each task ships a **codec** in
`src/tasks/<task>.py`: `to_inputs(batch: Frame) -> dict[str, Tensor]` and
`to_frame(outputs, batch: Frame) -> Frame`. Batching: `collate_frames`
in `src/data_processing/collate.py` stacks equal-shape Frames along a new
leading `"batch"` dim (RangeIndex).

Losses/metrics are classes declaring what they consume:

```python
class Loss(Protocol):
    requires_pred: FrameSpec     # keys from model output
    requires_target: FrameSpec   # keys from dataset batch
    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor: ...
```

**Pre-run validation** (`src/training/validate.py`, run at the start of both
train.py and eval.py, also standalone via `python train.py ... validate_only=true`):

1. dataset.spec ⊇ model input_spec (ignoring "batch")
2. model output_spec ∪ dataset.spec ⊇ each loss/metric's requirements
3. scheduler monitor metric exists in the metric suite
4. one-batch CPU smoke test: draw/synthesize a batch, run forward, check
   `spec_of(output) ⊇ output_spec`.

## Hydra config architecture

Dependency: `hydra-core>=1.3`. Root `conf/config.yaml`:

```yaml
defaults:
  - data: ???
  - model: ???
  - loss: ???
  - metrics: ???
  - optim: adamw_plateau
  - logging: wandb
  - _self_
experiment_name: ???        # set by the experiment file
seed: 0
validate_only: false
allow_dirty: false          # refuse to train on a dirty git tree unless true
results_root: results
```

An experiment file `conf/experiment/<name>.yaml` (package `_global_`):

```yaml
# @package _global_
defaults:
  - override /data: dregon_lm_v4_michaels_online
  - override /model: simple_conv_v2_uni_gru128
  - override /loss: pit_mse
  - override /metrics: rps
experiment_name: c10_uni_gru128_online
model:
  grad_clip: null
```

Conventions enforced by `train.py`:

- run dir = `results/${experiment_name}/` — refuses to overwrite unless
  `resume=true`;
- wandb: entity `flyingleafe`, project `harmonic-noise-suppression`,
  run name = `experiment_name`, tags = [task, data name]; git commit hash
  logged; dirty tree → hard error unless `allow_dirty=true`;
- config snapshot saved to the run dir and to wandb.

Component configs use `_target_` + structured-config dataclasses registered
in a ConfigStore (`src/training/config.py`) so Hydra type-checks at compose
time; `hydra.utils.instantiate` builds datasets/models/losses/optimizers.
Model configs declare `task: <task name>`.

## Losses (`src/losses/`)

- `spectral.py`: `MultiScaleSTFT` (moved from `src/models/generative/losses.py`),
  auraloss wrappers (`MultiResolutionSTFTLoss` config'd).
- `pit.py`: `pairwise_mse`, `pit_mse_loss` (moved from `train_rps_predictor.py`),
  `segmented_pit_mse` (from `src/models/multif0/utils.py`).
- `masked.py`: `masked_loss` (quantile hard mining, from `train.py`).
- `regularizers.py`: `smoothness_penalty`, `second_difference` — the ONE
  implementation; the inline copy in `train_rps_predictor.py` dies.
- `salience.py`: BCE-on-salience with pos_weight handling.
- `composite.py`: weighted sum-of-losses combinator (replaces `choice_loss`).
Every loss is a small class with `requires_pred`/`requires_target` specs.

## Metrics (`src/metrics/`)

- `separation.py`: `sdr`, `si_sdr`, `l1_freq`, `neg_log_wmse`, `aura_stft`,
  `aura_mrstft`, `bleedless`, `fullness` (from root `metrics.py`), `pesq`,
  `stoi`, `estoi` (from `final_valid.py`).
- `rps.py`: PIT-aware MSE/RMSE/MAE-frame/MAE-clip/R² — the ONE
  implementation (kills the inline copies in `train_rps_predictor.py` and
  `src/tasks/rps_prediction.py`); PIT alignment reuses `align_rps_to_gt`.
- `perf.py`: RTF, FLOPs, peak GPU mem (from `final_valid.py`).
- `suite.py`: `MetricSuite` — named collection evaluated per sample,
  aggregated mean/median, grouped by metadata key (e.g. `input_snr`).
Root `metrics.py` is deleted after the move.

## Plots (`src/plots/`)

Move `src/utils/plots/*` → `src/plots/*` (imports `utils.plots.X` →
`plots.X`; `make-plot` script entry updated). Then DRY pass:

- `constants.py`: `ROTOR_COLORS`, model display-name/color registry (kills
  per-report copies).
- `renderers.py` stays the engine; add canonical `render_rps_overlay`;
  all spectrograms go through `make_spectrogram_series` (kill manual STFTs
  in `sample_comparison._plot_spectrogram`, paper scripts).
- `comparison.py`: unified leaderboard bars / per-SNR line+errorbar
  (absorbs `plot_leaderboard`, `plot_per_rotor_mae`, `plot_per_snr`,
  root `plot_per_snr.py::plot_comparison`, `generate_comparison.py` plots).
- Report/slide `prepare*.py` scripts get re-pointed at the library.

## train.py / eval.py

`train.py` (hydra main): compose → validate specs → build (data, model,
losses, metrics, optim, sched) → generic loop in `src/training/loop.py`
(AMP, grad accum, grad clip, early stop, ReduceLROnPlateau or others,
checkpoint best+periodic, wandb logging, optional EMA) → supports both
map-style and iterable (online-mix, `samples_per_validation`) datasets.

`eval.py` (hydra main): same composition; loads checkpoint
(`checkpoint=...` or `results/<name>/best.ckpt`); runs MetricSuite over the
eval dataset → `results/<name>/eval/{metrics.json, per_sample.csv,
per_snr.csv}` (+ optional stems, plots via src/plots). Absorbs valid.py,
final_valid.py (PESQ/STOI/RTF/FLOPs are just metrics in the suite),
eval_cross.py (cross-eval = experiment config with a different data
group), eval_narrow_sr.py, generate_comparison.py.

## Future expansions (design headroom — do NOT implement yet)

Decided 2026-07-03. These shape interfaces; no code for them exists yet:

1. **Arbitrary under-annotated audio datasets as Frame sources.** A dataset
   adapter's only obligation is "emit `td.Frame`s + declare a `FrameSpec`".
   Nothing may assume the presence of `rps`/`target` beyond what a spec
   declares — losses/metrics already declare what they need, and the
   validator already treats absent-but-optional entries correctly. Data
   configs must stay *composable to a list of named sources* (each with its
   own spec and sampling weight) without changing the trainer.
2. **Self-supervised objectives on under-annotated data, trained alongside
   supervised losses.** The loss list must tolerate per-source
   applicability: a loss whose `requires_target` is not satisfied by a
   given source's spec is *skipped for batches from that source*, not an
   error, once multi-source training lands. Keep loss composition
   declarative (CompositeLoss) so adding an SSL term = adding a conf/loss
   entry.
3. **Joint / adversarial training of the RPS predictor and the noise
   generator on unannotated noise audio.** The training loop's
   model/optimizer construction must stay behind a narrow seam (build →
   step → validate) so a future "training scheme" abstraction (N models,
   N optimizers, alternating steps) can replace the single-model scheme
   without rewriting checkpointing/logging/artifacts.

Also decided: LoRA fine-tuning keeps a config seam (`lora.*`) in the
unified trainer; multi-GPU validation is dropped; checkpoints AND selected
validation samples are uploaded as artifacts to Cloudflare R2 (bucket
`ml-data`, creds via `.env`: `R2_ACCOUNT_ID` + AWS keys, s3fs client).

## Execution waves

0. Branch + this doc + mechanical moves (`git mv data_processing
   src/data_processing`, `git mv src/utils/plots src/plots`), pyproject
   updates (deps + packages + script entries), repo-wide import fix.
1. tdseries port: agents port `src/data_processing`, `src/plots`,
   `src/tasks`, root scripts, writing/ scripts, notebooks, tests. Then
   delete `src/utils/data` + `tests/utils/data`; full pytest gate.
2. `src/losses` + `src/metrics` extraction (old entry points keep working
   by importing from the new modules); plots DRY pass; tests.
3. FrameSpec/task layer, `src/training`, `conf/` tree, new `train.py` /
   `eval.py`; delete old entry points; smoke tests (tiny synthetic data,
   1 epoch CPU) for each task family.
4. `conf/experiment/*.yaml` replicating the historical catalogue
   (docs/experiment-history — see recon), each passing
   `validate_only=true`; `REPLICATION.md` with the drop-list discussion.
5. Docs: update CLAUDE.md, all AGENTS.md; final full pytest + ruff/pyright.

Gate for every wave: `uv run pytest` green, ruff+pyright clean on touched
files, one commit per wave.
