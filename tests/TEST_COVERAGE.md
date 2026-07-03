# Test Coverage — All core modules

> Last updated: 2026-06-08  
> FWH simulator excluded per request.  
> Scope: `src/` + `train_rps_predictor.py` + `data_processing/` + `models/rps_predictor.py`.
> Excluded: standalone report/slide scripts (under `writing/`) and one-off scripts under `scripts/` (benchmarks, config checks). Several former root scripts (`generate_comparison.py`, `plot_per_snr.py`, `eval_cross.py`) were absorbed into `eval.py`; `replicate_paper.py` was removed.
> Excluded: dataset creation scripts (`scripts/create_dataset.py`, `scripts/create_dregon_librimix.py`, `scripts/create_dregon_librimix_v3.py`).
> Excluded: legacy `src/utils/__init__.py` (ZFTurbo), `src/postdoc/infer.py`, `src/postdoc/queue.py` — deferred per decision.

---

## 1. Summary

| Package / module | Stmts | Miss | Cover | Test file(s) |
|------------------|-------|------|-------|-------------|
| `src/utils/data/__init__.py` | 7 | 0 | **100%** | — (re-exports only) |
| `src/utils/data/_ticks.py` | 16 | 1 | **94%** | (exercised by data tests) |
| `src/utils/data/base.py` | 28 | 7 | **75%** | (exercised by data tests) |
| `src/utils/data/event.py` | 197 | 50 | **75%** | `tests/utils/data/test_event.py` |
| `src/utils/data/frame.py` | 205 | 42 | **80%** | `tests/utils/data/test_frame.py` |
| `src/utils/data/segment.py` | 185 | 44 | **76%** | `tests/utils/data/test_segment.py` |
| `src/utils/data/uniform.py` | 223 | 56 | **75%** | `tests/utils/data/test_uniform.py` |
| `src/utils/__init__.py` | 322 | 294 | **9%** | — (legacy ZFTurbo — **EXCLUDED**) |
| `src/utils/paths.py` | 44 | 18 | **59%** | — |
| `src/postdoc/__init__.py` | 1 | 0 | **100%** | — |
| `src/postdoc/git_state.py` | 51 | 0 | **100%** | `tests/test_git_state.py` |
| `src/postdoc/task.py` | 52 | 6 | **88%** | `tests/test_task.py` |
| `src/postdoc/cli.py` | 213 | 100 | **53%** | `tests/test_cli.py` |
| `src/postdoc/cloud.py` | 82 | 51 | **38%** | — |
| `src/postdoc/direct.py` | 103 | 55 | **47%** | — |
| `src/postdoc/queue.py` | 367 | 367 | **0%** | — (**EXCLUDED**) |
| `src/postdoc/infer.py` | 133 | 124 | **7%** | — (**EXCLUDED**) |
| `src/tasks/__init__.py` | 0 | 0 | **100%** | — |
| `src/tasks/checkpoints.py` | 45 | 17 | **62%** | `tests/tasks/test_rps_regression.py` (partial) |
| `src/tasks/rps_prediction.py` | 274 | 156 | **43%** | `tests/tasks/test_rps_regression.py` |
| `src/tasks/cli.py` | ~86 | ~86 | **0%** | — (not imported during tests) |
| `src/utils/plots/__init__.py` | ~22 | ~22 | **0%** | — (not imported during tests) |
| `src/utils/plots/cli.py` | ~78 | ~78 | **0%** | — (not imported during tests) |
| `src/utils/plots/rps_prediction/*.py` | ~200 | ~200 | **0%** | — (not imported during tests) |
| `train_rps_predictor.py` | 339 | 314 | **7%** | `tests/tasks/test_rps_regression.py` (partial, via import) |
| `data_processing/__init__.py` | 2 | 2 | **0%** | — |
| `data_processing/dregon.py` | 270 | 270 | **0%** | — |
| `data_processing/external_recordings.py` | 95 | 95 | **0%** | — |
| `data_processing/michaels.py` | 130 | 130 | **0%** | — |
| `data_processing/noise_rps_dataset.py` | 160 | 160 | **0%** | — |
| `models/rps_predictor.py` | 361 | 286 | **21%** | (exercised via `train_rps_predictor` import) |
| **Total** | **~2,600** | **~1,180** | **~55%** | |

---

## 2. What is already tested (property/invariant documentation)

### 2.1 `src/utils/data/` — Fixed-point time-series algebra

The most thoroughly tested subsystem. Tests use **Hypothesis** property-based testing with
int64 tick-preserving strategies (`tests/utils/data/strategies.py`).

#### Core invariant: slice-concat identity

```
∀ series, ∀ a ≤ b ≤ c ∈ domain:
    series.slice(a, b).concat(series.slice(b, c)) == series.slice(a, c)
```

This is the **single algebraic invariant** that guarantees correctness of all time operations.
Every container type (`UniformSeries`, `EventSeries`, `SegmentSeries`, `TimeFrame`) tests it
with random cut points drawn by Hypothesis.

#### `test_uniform.py` — UniformSeries

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_slice_identity` | `us.slice(t_start, t_end) == us` | No-op boundary |
| `test_slice_concat_is_no_op` | sub-sample cut rejoin is exact | `slice(a,b) + slice(b,c) == slice(a,c)` |
| `test_many_slices_concat_no_op` | up to 6 random cuts rejoin to original | N-ary decomposition |
| `test_slice_reports_exact_domain` | `sliced.t_start_ticks == a`, `sliced.t_end_ticks == b` | Domain fidelity |
| `test_exact_boundary_cut_no_overlap` | cutting at sample boundary gives perfect split | Integer `sr` exactness |
| `test_sub_sample_cut_overlaps_one_sample` | sample cell spanning the cut appears in both halves; dedup on concat | Sub-sample cell algebra |
| `test_slice_outside_domain_raises` | `DomainError` for out-of-bounds cuts | Guard invariant |
| `test_concat_rejects_rate_mismatch` | `IncompatibleSeriesError` for `sr` mismatch | Type safety |
| `test_shift_preserves_samples` | `shift()` only changes anchor, not sample data | O(1) semantics |
| `test_shift_roundtrip` | `us.shift(d).shift(-d) == us` | Inverse property |
| `test_concat_across_gap` | concat with time gap auto-aligns | Gap algebra |
| `test_multichannel_slice_concat` | 3-part cut of `(4, 100)` multichannel rejoins identity | Channel-agnostic |
| `test_interpolate_at_sample_points_1d` | interpolation at sample grid returns exact samples | No drift |
| `test_interpolate_midpoint` | midpoint between two samples is linear average | Linear interpolation fidelity |
| `test_interpolate_multichannel` | per-channel interpolation for `(2, 3)` signals | Multi-channel correctness |
| `test_interpolate_clamp_extrap` | clamped extrapolation returns nearest endpoint | Extrapolation policy |
| `test_interpolate_nan_extrap` | `fill="nan"` gives NaN outside domain | Extrapolation policy |
| `test_resample_same_rate_approx_identity` | resampling to same rate preserves signal | Resampling fidelity |
| `test_resample_half_rate` | `resample(rate/2)` halves sample count | Downsampling |
| `test_resample_phase_zero` | resampled grid starts at `t_start` | Grid alignment |

**Not yet tested:** `_cut_to_indices` non-integer `sr` fallback path; `time_to_index`;
`sample_times_ticks`; `timestamps`/`timestamp_ticks` properties; `_floats.py` helpers (legacy, unused).

#### `test_event.py` — EventSeries

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_slice_identity` | `es.slice(t_start, t_end) == es` | No-op boundary |
| `test_slice_concat_no_op` | 3-cut rejoin matches single cut of whole | Slice-concat algebra |
| `test_many_cuts_rejoin` | up to 6 random cuts rejoin to identity | N-ary decomposition |
| `test_events_at_cut_go_right` | event exactly at `t_cut` goes to right half (half-open) | Half-open invariant |
| `test_getitem_returns_float_seconds` | `__getitem__` returns `(float_sec, value)` | Public API contract |
| `test_empty_series` | zero-event series slice returns zero events with correct domain | Degenerate case |
| `test_slice_outside_domain_raises` | `DomainError` for out-of-bounds | Guard invariant |
| `test_shift_changes_t_start` | shift only changes anchor, relative timestamps unchanged | O(1) semantics |
| `test_shift_roundtrip` | `es.shift(d).shift(-d) == es` | Inverse property |
| `test_concat_across_gap` | auto-shift across time gap | Gap algebra |
| `test_concat_rejects_value_shape_mismatch` | `(2,1)` vs `(3,1)` rejected as incompatible | Type safety — time-last axis |
| `test_slice_multidim_values_is_time_last` | `(4, M)` RPS values slice correctly (last axis) | Axis convention invariant |
| `test_interpolate_no_values_raises` | `interpolate` on series without values raises | Guard invariant |
| `test_interpolate_at_event_times_1d` | interpolation at event times returns exact values | No drift |
| `test_interpolate_midpoint` | midpoint is linear interpolation | Linear fidelity |
| `test_interpolate_multichannel` | per-channel interpolation | Multi-channel |
| `test_interpolate_clamp_extrap` | clamped extrapolation | Extrapolation policy |
| `test_interpolate_uniform_basic` | conversion to `UniformSeries` preserves values | Type conversion |
| `test_interpolate_uniform_phase_zero` | output grid phase is 0 (aligned to `t_start`) | Grid convention |
| `test_interpolate_uniform_custom_domain` | custom `(t_start, t_end)` sub-domain works | Domain overrides |

**Not yet tested:** `from_events` with float timestamps (the `np.issubdtype(np.floating)` branch);
`from_ticks` with explicit `dur:`; `_from_relative` direct path; `values` as `None` in concat/equal;
interpolation `fill="nan"` and `fill="error"` paths; `interpolate` on empty event series with
non-default fill; `kind != "linear"` error.

#### `test_segment.py` — SegmentSeries

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_slice_identity` | `ss.slice(t_start, t_end) == ss` | No-op boundary |
| `test_slice_concat_no_op` | 3-cut rejoin | Slice-concat algebra |
| `test_many_cuts_rejoin` | up to 6 random cuts | N-ary decomposition |
| `test_splitting_a_straddling_segment` | segment `[0.2, 0.8)` split at 0.5s → rejoin restores original via `ids` | Identity-tag algebra |
| `test_unrelated_segments_meeting_at_seam_are_not_merged` | distinct segments at seam remain separate | No false merges |
| `test_shift` | shift moves absolute coordinates, preserves relative storage | O(1) semantics |
| `test_shift_roundtrip` | inverse property | Algebraic idempotence |
| `test_concat_across_gap` | auto-shift across time gap | Gap algebra |

**Not yet tested:** validation errors in `__post_init__` (unsorted segments, end ≤ start,
domain-bound violations); `from_segments` with float inputs; `from_ticks` explicit `dur:` argument;
segments with values; `interpolate` (always raises `TypeError` — by design);
`equal` for non-`SegmentSeries`; `equal` with different ids/values;
`concat` with type mismatch error path; `concat` with `values=None` on one side;
`__getitem__` with values.

#### `test_frame.py` — TimeFrame

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_slice_identity` | `tf.slice(t_start, t_end) == tf` | No-op for frame |
| `test_slice_concat_no_op` | 3-cut rejoin across all tracks | Frame-level slice-concat |
| `test_slice_concat_across_full_domain` | explicit boundary invariant | Half-split rejoin |
| `test_many_cuts_rejoin` | up to 5 random cuts across mixed tracks | N-ary frame decomposition |
| `test_select_and_drop` | column-wise projection | Column algebra |
| `test_select_missing_raises` | `KeyError` for unknown keys | Guard invariant |
| `test_merge_two_frames` | column union of disjoint tracks | Column algebra |
| `test_merge_key_collision_raises` | `ValueError` on collision | Guard invariant |
| `test_merge_different_domains` | merge with `overwrite=True` across different domains | Domain reconciliation |
| `test_with_track_expands_domain` | adding a late track extends hull | Domain expansion |
| `test_slice_outside_domain_raises` | `DomainError` for out-of-bounds | Guard invariant |
| `test_track_domain_need_not_match_frame` | track domain can differ from frame hull | Heterogeneous domains |
| `test_shift` | shift moves all tracks | O(1) frame shift |
| `test_shift_roundtrip` | inverse property | Idempotence |
| `test_concat_allows_different_keys` | frames with different track sets concat fine | Sparse track algebra |
| `test_concat_with_gap` | time-gap concat works for frame | Gap algebra |
| `test_slice_heterogeneous_domains` | tracks with different domains slice correctly | Heterogeneous domain slicing |
| `test_tags_default` | empty tags dict default | Default invariant |
| `test_tags_preserved_on_slice` | tags survive `slice` | Metadata persistence |
| `test_tags_preserved_on_shift` | tags survive `shift` | Metadata persistence |
| `test_tags_preserved_on_select` | tags survive `select` | Metadata persistence |
| `test_tags_preserved_on_with_track` | tags survive `with_track` | Metadata persistence |
| `test_tags_concat_disjoint_union` | disjoint tag keys merge as union | Tag merge semantics |
| `test_tags_concat_equal_shared_keys` | equal values on shared keys pass | Tag merge semantics |
| `test_tags_concat_conflict_raises` | conflicting tag values raise on concat | Tag merge guard |
| `test_tags_merge_preserves` | tags survive `merge` (union) | Tag merge semantics |
| `test_tags_merge_conflict_raises` | conflicting tag values raise on merge | Tag merge guard |

**Not yet tested:** `global_data` merge/conflict semantics; `__post_init__` validation
(hull bounds, negative dur); `from_tracks` with float args; `_from_local` direct path;
`__contains__`, `__iter__`, `__len__`, `keys`, `values`, `items`; `__add__` dunder;
`__eq__` with non-TimeFrame; `__hash__`; `equal` with different tags/global_data;
`_global_leaf_equal` and `_global_equal` helpers; `concat` with `other` having tracks
the self doesn't have (left-missing case).

### 2.2 `src/postdoc/` — Job-runner CLI

#### `test_git_state.py` — Git snapshot helpers (100% coverage)

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_in_git_repo` | `in_git_repo()` truth in/out of repo | Detection |
| `test_head_sha` | `head_sha()` returns 40-char hex | Format invariant |
| `test_is_dirty` | clean vs dirty detection | Git status mapping |
| `test_remote_url` | resolves origin URL | URL resolution |
| `test_current_branch` | returns branch name | Branch detection |
| `test_current_branch_detached` | returns `None` in detached HEAD | Edge case |
| `test_push_head_branch` | push creates ref on remote | Side-effect |
| `test_push_head_detached_uses_postdoc_ref` | detached HEAD uses `refs/postdoc/<sha>` | Branchless push convention |
| `test_snapshot_happy_path` | clean push → snapshot dict has all keys | Happy path |
| `test_snapshot_dirty_fails_without_flag` | `GitError` on dirty without `allow_dirty` | Cleanliness guard |
| `test_snapshot_dirty_passes_with_flag` | `allow_dirty=True` marks `dirty="True"` | Override |
| `test_snapshot_skip_push` | `skip_push=True` → `refspec="(push skipped)"` | Skip semantics |
| `test_snapshot_not_a_repo` | raises `GitError` | Error path |
| `test_push_head_non_ff_fails` | non-fast-forward push raises | Divergence guard |

#### `test_task.py` — SkyPilot task YAML generators (88% coverage)

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_bootstrap_has_hostpath_mount` | pod spec has correct hostPath volumes and mounts | YAML structure |
| `test_bootstrap_runs_as_root` | `securityContext.runAsUser == 0` | Security |
| `test_bootstrap_requests_cluster_gpus` | GPU count/type appear in accelerators | Resource spec |
| `test_bootstrap_passes_pool` | pool name appears in infra field | Pool routing |
| `test_bootstrap_setup_installs_uv_and_syncs` | bootstrap script includes `uv install` and `uv sync` | Setup commands |
| `test_exec_envs_contain_git_info` | exec task has `POSTDOC_GIT_SHA`, `POSTDOC_GIT_URL`, `POSTDOC_REPO_DIR` | Environment vars |
| `test_exec_resources_only_accelerators_no_infra` | exec task omits `infra` (cluster-owned) | Resource isolation |
| `test_exec_no_setup_or_workdir` | exec task has no `setup`, `workdir`, or `config` | Shape constraint |
| `test_exec_run_pins_sha_and_syncs` | run script does `git reset --hard`, `uv sync`, `dvc pull`, activate, and the command | Run script content |
| `test_exec_gpus_zero_drops_accelerators` | `gpus=0` omits accelerators entirely | Zero-GPU |
| `test_exec_env_overrides_merge` | user env vars merge with base, don't clobber git envs | Env merge |

#### `test_cli.py` — CLI command verification (53% coverage)

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_queue_start_starts_tmux` | `queue-start` initiates `tmux new-session` over SSH | Queue lifecycle |
| `test_queue_stop_kills_tmux` | `queue-stop` sends `tmux kill-session` | Queue lifecycle |
| `test_queue_status_checks_tmux` | `queue-status` checks `tmux has-session` | Queue introspection |
| `test_submit_without_command_errors` | empty args → exit ≠ 0 | Usage guard |
| `test_submit_dirty_flag_propagates_to_preflight` | `--dirty` → `allow_dirty=True` in snapshot | Flag plumbing |
| `test_submit_skip_push_propagates` | `--skip-push` → `skip_push=True` | Flag plumbing |
| `test_submit_env_vars_passed` | `--env KEY=VALUE` appears in dry-run output | Env forwarding |
| `test_git_error_exits_nonzero` | `GitError` → exit code 3, message in stderr | Error routing |
| `test_ssh_uses_plain_ssh` | `postdoc ssh` invokes `ssh vast-server` | Utility command |
| `test_cluster_up_emits_migration_error` | `cluster-up` → explains `queue-start` | Backward compat |
| `test_pool_up_emits_migration_error` | `pool-up` → helpful error | Backward compat |
| `test_dashboard_emits_migration_error` | `dashboard` → explains SSH alternative | Backward compat |

**Not yet tested:** `submit` with `--cloud` / `--direct` routing; `list`, `status`, `logs`, `cancel`
commands; `queue-start` detection of already-running session; `queue-start` verification failure;
`check`/`probe` commands; `infer` command; `--no-sync` flag; `--name`/`-n` override; `_auto_name`
formatting; `_format_jobs` table; error on `--direct` and `--cloud` together;
backward-compat stubs: `pool-down`, `cluster-down`, `cluster-status`, `queue`.

### 2.3 `src/tasks/` — Task-separated evaluation & training

#### `test_rps_regression.py` — RPS prediction regression (covers `rps_prediction.py` 43%, `checkpoints.py` 62%)

| Test | What it verifies | Property |
|------|-----------------|----------|
| `test_artifacts_exist` | checkpoint and dataset are on disk | Data integrity |
| `test_eval_runs_and_produces_well_formed_output` | 10-sample eval returns all expected metric keys with sane ranges | Output shape contract |
| `test_aggregate_structure` | all aggregate keys present, RMSE = sqrt(MSE), timing positive | Aggregate invariants |

**Numeric invariants validated:**
- Per-sample keys: `sample`, `mse`, `mae_frame`, `mae_clip`, `ss_tot`, `r2`, `input_snr`
- Sanity bounds: `mse ≥ 0`, `ss_tot > 0`, `r2 ≤ 1.0`
- Golden consistency: all 10 samples match committed `per_sample_metrics_10.json` to `rtol=1e-4, atol=1e-4`
- Aggregate consistency: mean metrics match golden to `rtol=1e-4`

**Not yet tested (in `rps_prediction.py`):**
- `load_predictor` with classical tracker names (pyin, cepstral, hps, nmf, matched_filter)
- `load_predictor` with `nn.Module` or already-loaded `RPSPredictor`
- `load_input_set` with missing metadata.json; with nested `{"train": [...], "valid": [...]}` metadata
- `_align_shape_stretch` alignment strategy
- `alignment` error path (invalid strategy)
- `evaluate` with string spec (lazy load) vs object predictor
- `EvalResult.per_snr()`, `.to_json()`, `.to_wandb()`
- `_stratify_per_snr` / `_bin_stats` directly
- CLI shim (`src/tasks/cli.py`): argument parsing, output formatting, LaTeX table generation
- `_ClassicalPredictor` / `_ModelPredictor` wrappers individually
- Per-channel SNR in per-sample rows

**Not yet tested (in `checkpoints.py`):**
- `load_model` with suppression model types
- `load_model` error paths: no `@` in spec, empty type, empty path, missing file, unknown type
- `_make_suppression_registry` function

---

## 3. Modules with minimal or no coverage

### 3.1 `src/utils/__init__.py` — 9% (legacy ZFTurbo helpers) — **EXCLUDED**

**What it is:** A large (322-line) util module from the ZFTurbo codebase containing:
- `load_config()` — YAML/OmegaConf config loading
- `get_model_from_config()` — model factory with ~20 model types
- `read_audio_transposed()`, `normalize_audio()`, `denormalize_audio()`
- `apply_tta()` — test-time augmentation for source separation
- `demix()` — the main inference loop with chunked processing, TQDM, RPS conditioning, mixed precision
- `load_not_compatible_weights()`, `load_lora_weights()`, `bind_lora_to_model()` — weight loading
- `draw_spectrogram()` — visualization

**Why low coverage:** This is legacy code that predates the structured `src/` layout.
It imports from `models.*` and has 20 model-type branches in `get_model_from_config()`.
Testing it properly requires:
- Mocking model imports (or loading all model dependencies)
- Testing each `demix` mode branch (demucs vs generic)
- Testing the chunked processing loop with batch accumulation
- Testing RPS conditioning path within `demix`
- Testing TTA logic

**Current status:** Exercises the `load_config` and `get_model_from_config` paths indirectly
through the regression test (which uses `tasks.checkpoints.load_model` → calls `get_model_from_config`).

### 3.2 `src/utils/paths.py` — 59%

**What it is:** Central path resolution for `data/`, `datasets/`, `results/` directories.
Resolves `DATA_ROOT` from `.env`, falls back to git worktree list or `git rev-parse --show-toplevel`.

**Missing coverage:** Fallback paths — `DATA_ROOT` not set path (worktree detection, git toplevel,
cwd fallback). Only the happy path through `.env` is exercised in tests.

**Tests needed:**
- `get_data_path()`, `get_datasets_path()`, `get_results_path()` with subpath
- `_resolve_data_root()` fallback to `git worktree list`
- `_resolve_data_root()` fallback to `git rev-parse --show-toplevel`
- `_resolve_data_root()` fallback to `cwd`
- `get_data_root()` caching behavior

### 3.3 `src/postdoc/cloud.py` — 38%

**What it is:** SkyPilot managed-jobs backend. Generates task YAML, calls `sky jobs launch`,
parses job ID from output. Also provides `list_jobs_cloud`, `cancel_job_cloud`, `logs_job_cloud`.

**Missing coverage:** All functions (`submit_cloud`, `list_jobs_cloud`, `cancel_job_cloud`,
`logs_job_cloud`) and helpers (`_sky_available`, `_accelerator_str`, `_project_root`).
Testing requires mocking `subprocess.run` for `sky` commands.

### 3.4 `src/postdoc/direct.py` — 47%

**What it is:** Direct SSH backend for the job runner. GPU probing via `nvidia-smi`,
job state management via server filesystem (`/root/.postdoc/jobs/`, `queue.fifo`),
submission, listing, cancellation, log reading.

**Missing coverage:** `probe_gpus`, `free_gpus`, `submit_direct`, `list_jobs`, `cancel_job`,
`read_logs`, `_next_job_id`, `_ensure_postdoc_dir`. Testing requires mocking SSH calls.

### 3.5 `src/postdoc/queue.py` — 0% — **EXCLUDED**

**What it is:** The queue runner daemon (367 lines). Runs in a tmux session on
vast-server. Maintains a GPU allocation table, reads job descriptors from a
named FIFO (`/root/.postdoc/queue.fifo`), launches jobs when GPUs free up,
polls running jobs, and marks them done/failed.  Also handles re-queuing
when insufficient GPUs.

**Tests needed:** The core allocator logic (GPU allocation table, job lifecycle
state machine) can be tested in isolation.  The SSH/FIFO I/O would need mocking.
Key testable units:
- `GPUAllocator`: allocate/free GPUs, find contiguous free blocks
- `JobState` transitions: queued → launching → running → done/failed/cancelled
- `_parse_nvidia_smi` output parsing
- `run_one()` job execution script generation
- FIFO read/write round-trip with mock file descriptor

### 3.6 `src/postdoc/infer.py` — 7% — **EXCLUDED**

**What it is:** Local inference runner (`postdoc infer`). Resolves run directories, finds
checkpoints, loads models, processes audio files through `demix`.

**Missing coverage:** Almost all functions (`_resolve_run`, `_resolve_checkpoint`, `infer_cmd`).
Testing requires mocking torch, soundfile, model loading, and the `demix` pipeline.

### 3.7 `src/utils/data/_floats.py` — 0% (no coverage data)

**What it is:** Legacy float-comparison helpers (`tclose`, `t_atol_at`, `grid_atol`).
Marked as **unused** in `AGENTS.md` — the library now uses exact int comparisons.
Should be removed or marked as deprecated, not tested.

### 3.8 `src/tasks/cli.py` — 0% (no coverage data)

**What it is:** Typer CLI for `evaluate-rps` — argument parsing, output formatting,
LaTeX table generation.  All substance lives in `tasks.rps_prediction`.

**Tests needed:** Use `CliRunner` (like `test_cli.py`) to verify:
- Required `--input-set` and at least one `--model`
- `--output` JSON path, `--tex` LaTeX path
- Multi-model comparison output
- `--alignment` validation (reject unknown values)
- Error paths: missing input dir, no models

### 3.9 `src/utils/plots/` — 0% (no coverage data)

**What it is:** Shared plot infrastructure — a registry of plot functions
and a CLI entry point.  Sub-package `rps_prediction/` provides:
- `full_sequence.py` — full-sequence RPS prediction plots
- `per_snr.py` — per-SNR stratification plots
- `sample_comparison.py` — audio sample vs prediction comparisons
- `summary_metrics.py` — aggregate metric bar charts
- `training_curves.py` — training loss/learning rate curves

These are shared **infrastructure** (used by `generate_comparison.py` and
similar report scripts), not standalone report scripts themselves.  They
require coverage as library code under `src/`.

**Tests needed:**
- `src/utils/plots/__init__.py`: `register()`, `get_plot_fn()`, `list_plot_types()`,
  duplicate registration error, unknown name error
- `src/utils/plots/cli.py`: CLI argument parsing (Typer `CliRunner`)
- Each `rps_prediction/*.py`: smoke test with synthetic data —
  verify figure is produced, axes are labeled, no exceptions

### 3.10 `train_rps_predictor.py` — 7%

**What it is:** Standalone training script for RPS prediction models (797 lines).  Contains:

- **`DREGONRPSDataset`** — PyTorch `Dataset` that loads `mixture.wav` + `rps.npy` from DREGON-LM
  and resamples RPS to STFT frames via shape-stretch (endpoint-to-endpoint interpolation).
  Handles both mono `(T,)` and multichannel `(C, T)` audio.
- **`RPSPredictionHead`** — FPN-style auxiliary head that predicts `(4, T_stft)` from
  encoder feature pyramids.  Used by `DCUNetEncRPS` and `DCCRNEncRPS`.
- **`DCUNetEncRPS`** — DCUNet complex conv encoder with `RPSPredictionHead` for standalone
  RPS prediction.  Faithful copy of the encoder from `models/dcunet.py`.
- **`DCCRNEncRPS`** — DCCRN complex conv encoder with `RPSPredictionHead`.  Supports
  `lite` variant with fewer layers/channels.
- **`MODEL_REGISTRY`** — Maps 13 model names to constructors (SimpleConv variants, DCUNet,
  DCCRN, DCCRN-Lite).
- **`get_model()`** — Model factory from registry.
- **`pairwise_mse()` / `pit_mse_loss()`** — Permutation-invariant MSE loss over 4! = 24 rotor
  permutations.  Uses pre-computed `_ROTOR_PERMS` tensor.
- **`_flatten_channels()`** — Flattens `(B, C, T)` multichannel batches to `(B*C, T)`
  for models that expect mono input.
- **`wandb_init()`** — WandB logging initialisation.
- **`evaluate()`** — Validation loop returning PIT MSE, standard MSE, MAE per-frame,
  MAE per-clip, per-sample R² (macro-averaged).
- **`train_model()`** — Full training loop with AdamW, ReduceLROnPlateau, AMP, gradient
  clipping, early stopping, naive baseline, temporal smoothness regularisation.
- **`main()`** — Argument parsing.  Supports single-model or `--train_all` modes.

**Why 7% coverage:** The 25 covered statements come from imports triggered by
`tasks.checkpoints.load_model` (which imports `MODEL_REGISTRY` and `get_model` from
this module).  None of the training/inference logic is covered.

**Test strategy:**

*Unit-testable (no GPU needed):*
- `DREGONRPSDataset.__len__` / `__getitem__` with mock file system
- `stft_time_frames()` arithmetic
- `get_model()` with known/unknown model names
- `pairwise_mse()` shape and value correctness on synthetic tensors
- `pit_mse_loss()`: identity (loss=0 when pred==target), invariance under permutation,
  monotonicity (loss increases with perturbation)
- `_flatten_channels()` mono vs multichannel transformation
- `evaluate()` on synthetic data with a mock model (identity or constant)

*Integration-testable (requires GPU or small model on CPU):*
- `DCUNetEncRPS` forward pass shape check
- `DCCRNEncRPS` forward pass shape check (lite and full)
- `RPSPredictionHead` shape check
- `train_model()` smoke test with 2 epochs, 2 batches → verify loss decreases

### 3.11 `models/rps_predictor.py` — 21%

**What it is:** Model definitions for SimpleConv and its variants (361 lines).
Imported by `train_rps_predictor.py`'s `MODEL_REGISTRY`.  21% coverage comes
from model class definitions being imported during test runs — none of the
`forward()` methods are exercised in tests.

**Models defined:**
- `SimpleConv` — lightweight CNN on log-magnitude spectrograms
- `SimpleConvV2`, `SimpleConvWide` — architectural variants
- `SimpleConvTCN` — temporal conv net variant
- `SimpleConvMultiScale` — multi-scale spectrogram processing
- `SimpleConvBiGRU`, `SimpleConvBiGRUV2` — GRU-augmented variants
- `SimpleConvMagPhaseBiGRU` — magnitude + phase spectrogram input
- `SimpleConvAttnPool` — attention-pooled variant
- `SimpleConvSENext` — squeeze-and-excitation variant

**Test strategy:** Each model's `forward()` → check output shape `(B, 4, T_stft)`
given synthetic `(B, T_audio)` input.  Can be done on CPU with `torch.no_grad()`.
Test property: output shape invariant under different audio lengths.

### 3.12 `data_processing/` — 0% (all modules)

Dataset loading infrastructure — converts raw recordings into `TimeFrame`-based
representations.  Entirely uncovered.

#### 3.12.1 `data_processing/dregon.py` (270 stmts)

**What it is:** DREGON dataset loader with TimeFrame-native interface.

- **`discover_recordings()`** — walks a local DREGON mirror, returns recording metadata.
- **`download_dregon_dataset()`** — downloads ZIPs from dregon.inria.fr, unpacks them.
- **`_parse_mic_positions_txt()`** — parses mic geometry from `micPos.txt`.
- **`_load_rotor_positions()`** — loads rotor geometry from `.mat` files.
- **`get_geometry()`** — returns `(mic_positions, rotor_positions)` for a recording.
- **`clean_command_spikes()`** — median-filter cleaning of motor command telemetry spikes.
- **`load_timeframe()`** — loads one DREGON recording as a `TimeFrame` with tracks:
  `"audio"` (UniformSeries), `"motors_measured"` / `"motors_command"` (EventSeries),
  `"imu_accel"` / `"imu_gyro"` (EventSeries), `"source_position"` (EventSeries).
- **`load_dregon_timeframes()`** — loads all recordings for a split type
  (`in_flight_source`, `in_flight_noise`, etc.).

**Test strategy:**
- `_parse_mic_positions_txt()` with synthetic `.txt` content
- `clean_command_spikes()`: known spike patterns → cleaned output
- `_unpack_zip()` / `_download_file()` with mocked `urllib`
- `load_timeframe()` with a minimal recording (2-channel WAV + CSV motor log)
- `discover_recordings()` with tmp_path fixture

#### 3.12.2 `data_processing/michaels.py` (130 stmts)

**What it is:** Michael's dataset loader.  Handles DJI flight-controller CSV logs
with per-motor rotation speeds, aligned to WAV recordings via empirical time offsets.

- **`MotorData`** — frozen dataclass with `timestamps`, `measured` (4-rotor RPS),
  `command` (commanded RPS).  Methods: `slice_by_time()`.
- **`MichaelsRecord`** — duck-type compatible with `DREGONRecord`.  Fields:
  `audio`, `audio_timestamps`, `motors` (MotorData), `sample_rate`, `recording_id`.
  Method: `slice_by_time(start_sec, end_sec)` returns a new record.
- **`MichaelsDataset`** — file registry with 4 recordings (offsets −0.94, −0.40,
  −20.63, −26.27 s), the last two behind existence checks.
- **`load()`** / **`load_michaels_timeframes()`** — load individual or all recordings
  as `TimeFrame`.

**Test strategy:**
- `MotorData.slice_by_time()` correctness
- `MichaelsRecord.slice_by_time()` returns correct sub-chunk of audio + motors
- `MichaelsDataset` file registry structure
- `load()` with a recording fixture (mock CSV + WAV)
- Test that `slice_by_time` preserves audio/RPS alignment within the empirical offsets

#### 3.12.3 `data_processing/external_recordings.py` (95 stmts)

**What it is:** Loads external drone noise recordings (DJI flight-log CSVs +
multi-channel WAV) as `TimeFrame` objects.

- **`_find_col_indices()`** — column index lookup in CSV header.
- **`load_external_recording()`** — loads WAV (via soundfile) and CSV (via csv module)
  into a `TimeFrame` with `"audio"` (UniformSeries), `"motors_measured"` (EventSeries),
  `"imu_accel"` / `"imu_gyro"` (EventSeries).
- **`load_recordings_batch()`** — batch loader over a directory tree.

**Test strategy:**
- `_find_col_indices()` with known header → returns correct indices
- `load_external_recording()` with synthetic CSV + WAV → correct tracks
- Motor speed columns parsed and converted from RPM to RPS
- Time alignment between audio and motor samples

#### 3.12.4 `data_processing/noise_rps_dataset.py` (160 stmts)

**What it is:** Combined chunkable noise+RPS dataset for training generative
noise models.  Pulls from DREGON (`in_flight_noise` split) and Michael's set.

- **`upsample_rps_to_audio_rate()`** — linearly interpolates motor speeds onto audio
  sample grid.  Deduplicates motor timestamps (DREGON has some duplicates that
  cause NaN in `interp1d`).
- **`ChunkableNoiseRPSDataset`** — PyTorch `Dataset` returning `(rps_audio_rate, audio_target)`.
  Supports chunking with overlap, filtering by recording source, and metadata collection.
- **`NoiseChunk`** named tuple — `(rps_audio_rate: Tensor, audio_target: Tensor, metadata: dict)`.

**Test strategy:**
- `upsample_rps_to_audio_rate()`: linear interpolation correctness, dedup behavior
- `ChunkableNoiseRPSDataset` with mock `TimeFrame` data:
  - Chunk count and shapes match expected values
  - Overlap handling
  - Source filtering (DREGON-only, Michaels-only)
- `NoiseChunk` structure

**What it is:** Shared plot infrastructure — a registry of plot functions
and a CLI entry point.  Sub-package `rps_prediction/` provides:
- `full_sequence.py` — full-sequence RPS prediction plots
- `per_snr.py` — per-SNR stratification plots
- `sample_comparison.py` — audio sample vs prediction comparisons
- `summary_metrics.py` — aggregate metric bar charts
- `training_curves.py` — training loss/learning rate curves

These are shared **infrastructure** (used by `generate_comparison.py` and
similar report scripts), not standalone report scripts themselves.  They
require coverage as library code under `src/`.

**Tests needed:**
- `src/utils/plots/__init__.py`: `register()`, `get_plot_fn()`, `list_plot_types()`,
  duplicate registration error, unknown name error
- `src/utils/plots/cli.py`: CLI argument parsing (Typer `CliRunner`)
- Each `rps_prediction/*.py`: smoke test with synthetic data —
  verify figure is produced, axes are labeled, no exceptions

---

## 4. Test infrastructure

### 4.1 `tests/conftest.py` — Shared fixtures

| Fixture | Purpose |
|---------|---------|
| `fake_git` | Monkeypatches `postdoc.git_state.snapshot` to return fake data without shelling out |
| `fake_sky` | Monkeypatches `subprocess.run` and `shutil.which` to intercept all CLI commands; records every argv |

### 4.2 `tests/utils/data/strategies.py` — Hypothesis strategies

| Strategy | Generates |
|----------|-----------|
| `time_anchors` | Random int64 tick origins (small ~0–1e13 and Unix-scale ~1.6–1.8e18) |
| `sample_rates` | `{10, 100, 1000, 8000, 16000, 44100}` — plausible audio/telemetry rates |
| `uniform_series()` | Valid `UniformSeries` with random samples, sr, and tick anchor |
| `cut_points()` | Float cut points inside a uniform series domain |
| `cut_points_ticks()` | Distinct int64 cut points inside `[t_start, t_end)` |
| `event_series()` | Valid `EventSeries` with up to 32 events |
| `segment_series()` | Valid `SegmentSeries` with up to 8 segments |

All strategies build series via `from_ticks`/`from_events` with int arguments to guarantee
exactness — no float quantisation in Hypothesis-generated data.

---

## 5. What "100% coverage" requires (gap analysis)

### Priority 1: `src/utils/data/` — already well-tested

| Module | Current | What's needed to reach 100% |
|--------|---------|---------------------------|
| `_ticks.py` | 94% (1 line missed) | Test `_c_to_ticks` with edge case input (line 58, likely `None` guard) |
| `base.py` | 75% (7 lines) | Test `__eq__` with non-`TimeSeries`, `__add__` dunder, `__hash__`, `equal` abstract to raise; the `duration` property with float arithmetic |
| `event.py` | 75% (50 lines) | Validation paths in `__post_init__` (negative dur, events outside domain, unsorted); `from_events` with float timestamps; `interpolate` `fill="nan"` / `fill="error"`; `kind != "linear"` error; empty series interpolate |
| `frame.py` | 80% (42 lines) | `global_data` merge/conflict; `__post_init__` hull validation; `from_tracks` float args; dunder methods (`__iter__`, `__len__`, etc.); `_global_equal` helpers; `concat` with missing tracks |
| `segment.py` | 76% (44 lines) | `__post_init__` validation (unsorted, end≤start, domain bounds); `from_segments` float input path; concat type mismatch; `equal` edge cases; `__getitem__` with values |
| `uniform.py` | 75% (56 lines) | Non-integer `sr` `_cut_to_indices` path; `time_to_index`; `sample_times_ticks`; `timestamps`/`timestamp_ticks`; `values`/`channel_shape`; `__getitem__`; `concat` incompatible grid offset; `interpolate` empty series, `fill="error"`, `kind != "linear"` |
| `_floats.py` | 0% (unused) | **Remove** — legacy, unused per `AGENTS.md` |

### Priority 2: `src/postdoc/` — CLI & backend

| Module | Current | What's needed |
|--------|---------|--------------|
| `task.py` | 88% (6 lines) | Test `dump_task_yaml` and `task_to_yaml` with multiline strings; `_LiteralDumper` representative |
| `cli.py` | 53% (100 lines) | Test `submit` routing paths (--direct, --cloud); list, status, logs, cancel commands; check/probe commands; infer command; error paths (--direct + --cloud simultaneously); --no-sync; backward-compat stubs |
| `direct.py` | 47% (55 lines) | Mock SSH to test `probe_gpus` (nvidia-smi parsing), `free_gpus`, `submit_direct` (queued path), `list_jobs`, `cancel_job`, `read_logs`; `_next_job_id`; `_ensure_postdoc_dir` |
| `cloud.py` | 38% (51 lines) | Mock `sky` CLI to test `submit_cloud`, `list_jobs_cloud`, `cancel_job_cloud`, `logs_job_cloud`; `_sky_available`; `_accelerator_str`; `_project_root` |
| `infer.py` | 7% (124 lines) | Most complex — requires mocking model loading, torch, soundfile. Consider smoke tests with a tiny model. At minimum: `_resolve_run` (exists/not-dir/no-config), `_resolve_checkpoint` (best/latest/epoch/substring) |
| `queue.py` | 0% (367 lines) | Core allocator + job lifecycle state machine testable in isolation (no SSH). GPUAllocator allocate/free/contiguous; JobState transitions; FIFO parse/write; `_parse_nvidia_smi` |

### Priority 3: `src/tasks/` — Task modules

| Module | Current | What's needed |
|--------|---------|--------------|
| `checkpoints.py` | 62% (17 lines) | Error paths for `load_model` (no `@`, empty type, empty path, missing file, unknown type); suppression model registry |
| `rps_prediction.py` | 43% (156 lines) | Classical predictor loading; `_ClassicalPredictor` wrapper; `_ModelPredictor` wrapper; `_align_shape_stretch`; `load_input_set` metadata variants; `evaluate` with string spec and with shape_stretch alignment; `EvalResult.per_snr()`, `.to_json()`, `.to_wandb()`; `_stratify_per_snr` helper |
| `cli.py` | 0% (not in report) | Typer CLI test: argument parsing, JSON/LaTeX output paths; error on missing --model, missing --input-set |

### Priority 4: `src/utils/plots/` and `src/tasks/cli.py`

| Module | Current | What's needed |
|--------|---------|--------------|
| `src/utils/plots/__init__.py` | 0% (~22 lines) | Test `register()`, `get_plot_fn()` (happy + unknown), `list_plot_types()`, duplicate registration error |
| `src/utils/plots/cli.py` | 0% (~78 lines) | Typer CLI test: argument parsing for each plot type |
| `src/utils/plots/rps_prediction/*` | 0% (~200 lines) | Smoke tests with synthetic data: each plot function produces a `Figure` with expected structure |
| `src/tasks/cli.py` | 0% (~86 lines) | `CliRunner` test: required args, output paths, multi-model, alignment validation, error paths |

### Priority 5: `src/utils/__init__.py` (**EXCLUDED**) and `src/utils/paths.py`

| Module | Current | What's needed |
|--------|---------|--------------|
| `__init__.py` | 9% (294 lines) | **Excluded** — legacy ZFTurbo code, deferred for later rewrite/retirement. |
| `paths.py` | 59% (18 lines) | Test `_resolve_data_root` fallback paths (worktree detection, git toplevel, cwd); cache behavior |

### Priority 6: `train_rps_predictor.py` — training pipeline

| Component | Lines | What's needed |
|-----------|-------|--------------|
| `DREGONRPSDataset` | ~30 | `__len__`, `__getitem__` with mock filesystem (mono + multichannel); STFT frame count correctness |
| `stft_time_frames()` | 3 | Arithmetic correctness for known lengths |
| `get_model()` | 6 | Happy path (all 13 names), unknown-name error |
| `pairwise_mse()` | 5 | Shape `(B,4,4)` on synthetic tensors; value correctness |
| `pit_mse_loss()` | 15 | Identity (loss=0 when pred==target), permutation invariance, monotonicity |
| `_flatten_channels()` | 8 | Mono no-op, multichannel reshape, channel count return |
| `evaluate()` | 30 | Output keys, PIT vs std MSE on synthetic mock model, per-sample R² range |
| `wandb_init()` | 15 | Disabled mode when no key; run-id file written |
| `train_model()` | 100 | Smoke test (2 epochs, 2 batches, CPU): loss decreases, best checkpoint saved |
| `main()` | 60 | Argparse: `--model`, `--train_all`, `--pit_loss`/`--no_pit_loss` |

### Priority 7: `models/rps_predictor.py` — model forward passes

| What's needed |
|--------------|
| Each model variant: `forward()` returns `(B, 4, T_stft)` given `(B, T_audio)` input |
| SimpleConv: verify log-mag spectrogram extraction + Conv1d pipeline |
| Shape invariance: different audio lengths produce correct `T_stft` |
| `torch.no_grad()` CPU inference only — no GPU needed |

### Priority 8: `data_processing/` — dataset loaders

| Module | What's needed |
|--------|--------------|
| `dregon.py` | `_parse_mic_positions_txt()` with synthetic .txt; `clean_command_spikes()` on known spike patterns; `load_timeframe()` with a minimal recording fixture; `discover_recordings()` in tmp_path |
| `michaels.py` | `MotorData.slice_by_time()` correctness; `MichaelsRecord.slice_by_time()` produces correct sub-chunk; `MichaelsDataset` file registry structure |
| `external_recordings.py` | `_find_col_indices()` with known header; `load_external_recording()` with synthetic CSV + WAV → correct tracks; RPM→RPS conversion |
| `noise_rps_dataset.py` | `upsample_rps_to_audio_rate()` linear interpolation + dedup; `ChunkableNoiseRPSDataset` chunk count/shapes, overlap, source filtering |

**Shared pattern for all data_processing tests:** Use `tmp_path` fixtures with minimal synthetic WAV files (e.g. 1 second of zeros at 44100 Hz via `soundfile`/`scipy`) and synthetic CSV/mat/npy metadata.  No real DREGON downloads needed.

---

## 6. Concrete test plan for >90% coverage

Test names use the convention `test_<situation>_<expected_behavior>`.
Tests target **currently uncovered** code paths only (covered paths are already safe).
Each test includes a brief property/invariant statement.

### 8.1 `src/utils/data/_ticks.py` (94% → 100%)

| Test | Property |
|------|----------|
| `test_c_to_ticks_from_float` | `_c_to_ticks(float)` quantises via `round(f * TPS)` |
| `test_c_to_ticks_from_int_is_id` | `_c_to_ticks(int)` returns the int unchanged (line 58) |

### 8.2 `src/utils/data/base.py` (75% → 100%)

| Test | Property |
|------|----------|
| `test_add_is_concat` | `ts + other == ts.concat(other)` for all concrete types |
| `test_eq_not_time_series_returns_notimplemented` | `ts == 42` returns `NotImplemented` |
| `test_eq_different_type_returns_false` | `UniformSeries(…) == EventSeries(…)” is `False` |
| `test_eq_delegates_to_equal` | `(a == b) == a.equal(b)` for same-type series |
| `test_hash_is_id` | `hash(ts) == id(ts)` — identity-based (frozen dataclass override) |

### 8.3 `src/utils/data/event.py` (75% → 95%)

#### Validation paths (16 lines)

| Test | Property |
|------|----------|
| `test_init_rejects_2d_timestamps` | `ValueError` for `ndim != 1` timestamps |
| `test_init_rejects_value_shape_mismatch_last_axis` | `values.shape[-1] != len(ts)` → `ValueError` |
| `test_init_rejects_negative_dur_ticks` | `dur_ticks < 0` → `ValueError` |
| `test_init_rejects_negative_relative_timestamp` | `ts[0] < 0` → `ValueError` |
| `test_init_rejects_event_after_domain_end` | `ts[-1] >= dur_ticks` → `ValueError` |
| `test_init_rejects_unsorted_timestamps` | non-monotonic timestamps → `ValueError` |

#### Constructor branches (8 lines)

| Test | Property |
|------|----------|
| `test_from_events_with_float_timestamps` | `from_events(…, t_start=0.5, t_end=1.5)` with float seconds → correct int64 domain |
| `test_from_events_empty_infers_zero_domain` | zero events, no t_start/t_end → `dur_ticks=0`, `t_start_ticks=0` |
| `test_from_ticks_with_explicit_dur` | `from_ticks(ts, vals, t_start=100, dur=1000)` sets domain correctly |

#### Interpolation edge paths (22 lines)

| Test | Property |
|------|----------|
| `test_interpolate_with_int_query_times` | query as int64 ticks → converted to float seconds internally |
| `test_interpolate_empty_series_fill_error` | `fill="error"` on empty series → `DomainError` |
| `test_interpolate_empty_series_fill_nan` | `fill="nan"` on empty series → all-NaN output |
| `test_interpolate_unsupported_kind` | `kind="cubic"` → `ValueError` |
| `test_interpolate_fill_nan_extrapolation` | query outside event span with `fill="nan"` → NaN at those positions |
| `test_interpolate_fill_error_extrapolation` | query outside event span with `fill="error"` → `DomainError` |
| `test_interpolate_fill_error_within_span` | `fill="error"` but all queries inside span → no error |
| `test_interpolate_unsupported_fill_value` | `fill="extrap"` → `ValueError` |

#### Concat / equal error paths (10 lines)

| Test | Property |
|------|----------|
| `test_concat_rejects_non_event_series` | `es.concat(UniformSeries(…))` → `IncompatibleSeriesError` |
| `test_concat_rejects_one_has_values_one_does_not` | `es(vals=None).concat(es(vals=…))` → `IncompatibleSeriesError` |
| `test_equal_non_event_series_is_false` | `es.equal(UniformSeries(…))` → `False` |
| `test_equal_different_t_start_is_false` | same data, different `t_start_ticks` → `False` |
| `test_equal_one_none_values_one_not` | `es(vals=None).equal(es(vals=…))` → `False` |
| `test_equal_different_dur_is_false` | same data, different `dur_ticks` → `False` |

#### `interpolate_uniform` guards (3 lines)

| Test | Property |
|------|----------|
| `test_interpolate_uniform_rejects_nonpositive_sr` | `sr ≤ 0` → `ValueError` |
| `test_interpolate_uniform_rejects_zero_dur` | `t_end == t_start` → `ValueError` |

### 8.4 `src/utils/data/frame.py` (80% → 95%)

#### `__post_init__` validation + normalisation (11 lines)

| Test | Property |
|------|----------|
| `test_init_rejects_negative_dur_ticks` | `dur_ticks < 0` → `ValueError` |
| `test_init_rejects_non_timeseries_track` | track not a `TimeSeries` → `TypeError` |
| `test_init_rejects_t_start_after_hull_start` | `t_start_ticks > min(track starts)` → `ValueError` |
| `test_init_rejects_t_end_before_hull_end` | `t_end_ticks < max(track ends)` → `ValueError` |
| `test_init_normalises_none_tags_to_empty_dict` | `tags=None` → `tags == {}` |
| `test_init_normalises_none_global_data_to_empty_dict` | `global_data=None` → `global_data == {}` |

#### `from_tracks` error path (1 line)

| Test | Property |
|------|----------|
| `test_from_tracks_empty_without_domain_raises` | empty tracks dict, no `t_start`/`t_end` → `ValueError` |

#### Dict-like protocol (8 lines)

| Test | Property |
|------|----------|
| `test_len_is_track_count` | `len(tf)` equals number of tracks |
| `test_iter_yields_keys` | `list(tf)` equals `list(tf.keys())` |
| `test_values_returns_abs_tracks` | `tf.values()` tracks have absolute timestamps |
| `test_items_returns_abs_tracks` | `tf.items()` yields `(name, abs_track)` pairs |
| `test_contains_key` | `"audio" in tf` is `True`; `"missing" in tf` is `False` |

#### Domain properties (4 lines)

| Test | Property |
|------|----------|
| `test_t_start_is_float_seconds` | `tf.t_start == ticks_to_secs(tf.t_start_ticks)` |
| `test_t_end_is_float_seconds` | `tf.t_end == ticks_to_secs(tf.t_end_ticks)` |
| `test_duration_is_float_seconds` | `tf.duration == ticks_to_secs(tf.dur_ticks)` |

#### `global_data` semantics (8 lines)

| Test | Property |
|------|----------|
| `test_merge_global_data_union` | disjoint keys → union in result |
| `test_merge_global_data_equal_shared_keys` | equal numpy arrays on shared keys → pass |
| `test_merge_global_data_conflict_raises` | different arrays on shared key → `IncompatibleSeriesError` |
| `test_concat_global_data_union` | disjoint keys → union |
| `test_concat_global_data_conflict_raises` | conflicting shared key → `IncompatibleSeriesError` |

#### `concat` missing-track path (3 lines)

| Test | Property |
|------|----------|
| `test_concat_other_has_track_self_does_not` | track only in `other` → present in result at correct offset |

#### `shift` zero-delta fast path (1 line)

| Test | Property |
|------|----------|
| `test_shift_zero_is_identity` | `tf.shift(0) is tf` (same object, not just equal) |

#### Dunders + `equal` (10 lines)

| Test | Property |
|------|----------|
| `test_add_is_concat` | `tf1 + tf2 == tf1.concat(tf2)` |
| `test_eq_non_timeframe_returns_notimplemented` | `tf == 42` → `NotImplemented` |
| `test_eq_delegates_to_equal` | `(tf1 == tf2) == tf1.equal(tf2)` for TimeFrame |
| `test_hash_is_id` | `hash(tf) == id(tf)` |
| `test_equal_different_t_start` | same tracks, different `t_start` → `False` |
| `test_equal_different_track_set` | different keys → `False` |
| `test_equal_different_tags` | different tag values → `False` |
| `test_equal_different_global_data` | different global_data → `False` |
| `test_global_leaf_equal_numpy_arrays` | `np.array([1,2])` vs `np.array([1,2])` → `True` |
| `test_global_leaf_equal_different_shapes` | shape mismatch → `False` |

### 8.5 `src/utils/data/segment.py` (76% → 95%)

#### Validation (16 lines)

| Test | Property |
|------|----------|
| `test_init_rejects_2d_starts` | 2-D starts → `ValueError` |
| `test_init_rejects_end_not_greater_than_start` | `ends[i] <= starts[i]` → `ValueError` |
| `test_init_sorts_unsorted_segments` | unsorted starts → auto-sorted, values/ids reordered |
| `test_init_rejects_wrong_shape_ids` | `ids.shape != (M,)` → `ValueError` |
| `test_init_rejects_wrong_values_last_axis` | `values.shape[-1] != M` → `ValueError` |
| `test_init_rejects_negative_dur_ticks` | `dur_ticks < 0` → `ValueError` |
| `test_init_rejects_start_before_t_start` | `starts[0] < 0` (relative) → `ValueError` |
| `test_init_rejects_end_after_t_end` | `ends max > dur_ticks` → `ValueError` |

#### `from_segments` float-input path (6 lines)

| Test | Property |
|------|----------|
| `test_from_segments_with_float_starts_ends` | float seconds → correct int64 conversion |
| `test_from_segments_with_float_t_start_t_end` | float `t_start`/`t_end` → correct domain |
| `test_from_segments_empty_infers_zero_domain` | zero segments → `t_start=0`, `dur=0` |

#### Concat / equal error paths (8 lines)

| Test | Property |
|------|----------|
| `test_concat_rejects_non_segment_series` | `ss.concat(UniformSeries(…))` → `IncompatibleSeriesError` |
| `test_concat_rejects_one_has_values_one_does_not` | mismatch → `IncompatibleSeriesError` |
| `test_equal_non_segment_series` | `ss.equal(UniformSeries(…))` → `False` |
| `test_equal_different_ids` | different ids → `False` |
| `test_equal_different_values` | different values → `False` |
| `test_equal_one_has_values_one_does_not` | mismatch → `False` |

#### `__getitem__` with values (2 lines)

| Test | Property |
|------|----------|
| `test_getitem_with_values_returns_4_tuple` | `ss[i] → (start_s, end_s, values[…, i], id)` |

#### Properties (7 lines)

| Test | Property |
|------|----------|
| `test_starts_returns_float_seconds` | `.starts` returns float seconds |
| `test_ends_returns_float_seconds` | `.ends` returns float seconds |
| `test_abs_starts_is_float_seconds` | `.abs_starts` is float, not ticks |
| `test_abs_ends_is_float_seconds` | `.abs_ends` is float, not ticks |

### 8.6 `src/utils/data/uniform.py` (75% → 95%)

#### Validation (6 lines)

| Test | Property |
|------|----------|
| `test_init_rejects_0d_samples` | scalar sample → `ValueError` |
| `test_init_rejects_nonpositive_sr` | `sr <= 0` → `ValueError` |
| `test_init_rejects_phase_out_of_range` | `phase = 1.5` → `ValueError` |

#### Non-integer sr fallback in `_cut_to_indices` (7 lines)

| Test | Property |
|------|----------|
| `test_cut_to_indices_non_integer_sr` | `sr=44.1` (float): cut at known offsets → correct sample count |
| `test_cut_to_indices_non_integer_sr_edge_clamp` | indices clamped to `[0, n_samples]` |

#### Phase clamping in slice (3 lines)

| Test | Property |
|------|----------|
| `test_slice_phase_clamped_to_minus_one` | sub-sample cut that would produce `phase > 0` → wraps |

#### Tool methods (18 lines)

| Test | Property |
|------|----------|
| `test_timestamps_returns_float_seconds` | `.timestamps` are float seconds relative to `t_first_edge` |
| `test_timestamp_ticks_returns_int64` | `.timestamp_ticks` are int64 relative to `t_start_ticks` |
| `test_channel_shape_for_mono` | mono → `channel_shape == ()` |
| `test_channel_shape_for_stereo` | stereo → `channel_shape == (2,)` |
| `test_values_is_samples` | `.values is .samples` |
| `test_getitem_returns_along_last_axis` | `us[i] == us.samples[..., i]` |
| `test_sample_times_ticks_returns_int64` | `.sample_times_ticks()` returns int64 |
| `test_time_to_index_integer_sr` | `time_to_index(t)` returns correct cell for integer sr |
| `test_time_to_index_non_integer_sr` | `time_to_index(t)` for `sr=44.1` |

#### `concat` incompatible offset paths (5 lines)

| Test | Property |
|------|----------|
| `test_concat_rejects_incompatible_grid_float_offset` | phase offset not near-integer → `IncompatibleSeriesError` |
| `test_concat_rejects_incompatible_grid_offset_value` | integer offset not `n_self` or `n_self-1` → `IncompatibleSeriesError` |

#### Interpolation error paths (14 lines)

| Test | Property |
|------|----------|
| `test_interpolate_with_int_query_times` | int64 query → converted to float internally |
| `test_interpolate_empty_series_fill_error` | `fill="error"` on empty → `DomainError` |
| `test_interpolate_empty_series_fill_nan` | `fill="nan"` on empty → NaN output with correct shape |
| `test_interpolate_empty_series_zero_samples` | empty samples but non-empty query → zero-filled output |
| `test_interpolate_unsupported_kind` | `kind="cubic"` → `ValueError` |
| `test_interpolate_fill_nan_extrapolation_multichannel` | multichannel NaN fill → NaN in correct positions |
| `test_interpolate_fill_error_extrapolation` | `fill="error"` with out-of-span query → `DomainError` |
| `test_interpolate_fill_error_within_span` | `fill="error"` but all queries inside span → no error |
| `test_interpolate_unsupported_fill` | `fill="extrap"` → `ValueError` |
| `test_resample_rejects_nonpositive_sr` | `resample(0)` → `ValueError` |

### 8.7 `src/utils/paths.py` (59% → 95%)

| Test | Property |
|------|----------|
| `test_data_root_from_env` | `DATA_ROOT` set in env → `get_data_root()` returns that path |
| `test_data_root_from_git_worktree_list` | mock `git worktree list` output → parses first line |
| `test_data_root_from_git_rev_parse` | mock `git rev-parse --show-toplevel` → returns that path |
| `test_data_root_fallback_to_cwd` | all git commands fail → returns `Path.cwd()` |
| `test_data_root_cached` | second call to `get_data_root()` returns same object (cached) |
| `test_get_data_path_with_subpath` | `get_data_path("DREGON")` → `<root>/data/DREGON` |
| `test_get_datasets_path_no_subpath` | `get_datasets_path()` → `<root>/datasets` |
| `test_get_results_path_with_subpath` | `get_results_path("eval")` → `<root>/results/eval` |
| `test_path_resolution_ignores_trailing_slash` | subpath with trailing slash handled correctly |

### 8.8 `src/postdoc/task.py` (88% → 100%)

| Test | Property |
|------|----------|
| `test_exec_with_name_includes_name_field` | `name="my-job"` → `task["name"] == "my-job"` |
| `test_dump_task_yaml_writes_multiline_as_block_scalar` | multi-line setup script → `\|` in YAML output |
| `test_dump_task_yaml_returns_path` | returns the `Path` passed in |
| `test_task_to_yaml_returns_string` | `task_to_yaml(task)` returns `str` with block scalars |

### 8.9 `src/postdoc/cli.py` (53% → 90%)

All use `fake_sky` + `fake_git` from conftest + `CliRunner`.

#### Submit routing + error paths

| Test | Property |
|------|----------|
| `test_submit_direct_backend_forced` | `--direct` forces backend=direct in dry-run |
| `test_submit_cloud_backend_forced` | `--cloud` forces backend=cloud in dry-run |
| `test_submit_direct_and_cloud_mutually_exclusive` | `--direct --cloud` → exit code 2, error message |
| `test_submit_no_sync_flag_disables_sync` | `--no-sync` implies `--skip-push` |
| `test_submit_auto_name_from_command` | no `--name` → auto-generated name contains command stem |
| `test_submit_env_malformed` | `--env FOO` (no `=`) → exit code 2 |

#### List / status / logs / cancel

| Test | Property |
|------|----------|
| `test_list_no_jobs` | `list_jobs` returns empty → "No jobs found" |
| `test_list_active_only_by_default` | finished jobs filtered out unless `--all` |
| `test_status_not_found` | non-existent `name__id` → exit 1, "Job not found" |
| `test_logs_prints_tail` | `logs` command calls `tail -50` on server |
| `test_cancel_calls_direct_backend` | `cancel` calls `direct_mod.cancel_job` |

#### Check / probe commands

| Test | Property |
|------|----------|
| `test_check_probes_gpus` | `check` calls `direct_mod.probe_gpus` |
| `test_probe_is_alias_for_check` | `probe` delegates to `cmd_check` |

#### Backward-compat stubs

| Test | Property |
|------|----------|
| `test_pool_down_emits_noop` | `pool-down` prints "no longer needed", exit 0 |
| `test_cluster_down_emits_noop` | `cluster-down` prints teardown message |
| `test_cluster_status_delegates_to_check` | `cluster-status` calls `cmd_check` |
| `test_queue_stub_prints_help` | `queue` prints usage hint, exit 1 |

### 8.10 `src/postdoc/direct.py` (47% → 90%)

Mock `subprocess.run` to intercept SSH calls.

| Test | Property |
|------|----------|
| `test_probe_gpus_parses_nvidia_smi` | synthetic nvidia-smi output → correct `GPUInfo` list |
| `test_probe_gpus_empty_output` | no GPUs → empty list |
| `test_free_gpus_filters_by_memory_threshold` | GPUs with <500 MiB used → in free list |
| `test_ensure_postdoc_dir_creates_dirs_and_fifo` | verifies `mkdir -p` + `mkfifo` are called |
| `test_next_job_id_when_jobs_exist` | mock glob returns dirs → max id + 1 |
| `test_next_job_id_when_no_jobs` | no dirs → returns 1 |
| `test_submit_direct_creates_job_dir_and_writes_fifo` | job desc JSON written to `queue.fifo` as base64 |
| `test_submit_direct_returns_queued_status` | return value `(job_id, "queued")` |
| `test_list_jobs_parses_json_correctly` | mock SSH output with JSON → list of `JobInfo` |
| `test_list_jobs_empty_server` | empty job dirs → empty list |
| `test_cancel_job_kills_pid` | valid pid → `kill <pid>` called, status set to "cancelled" |
| `test_cancel_job_no_pid` | no pid → status still set to "cancelled" |
| `test_read_logs_tail` | `read_logs(…, lines=50)` → `tail -50` |
| `test_read_logs_follow` | `read_logs(…, follow=True)` → `tail -F` |

### 8.11 `src/postdoc/cloud.py` (38% → 90%)

Mock `subprocess.run` for `sky` CLI.

| Test | Property |
|------|----------|
| `test_sky_available_when_which_succeeds` | `shutil.which("sky")` returns path → `True` |
| `test_sky_available_when_missing` | `shutil.which("sky")` returns `None` → `False` |
| `test_accelerator_str_zero_gpus` | `_accelerator_str(0)` → `"null"` |
| `test_accelerator_str_with_type` | `_accelerator_str(2, "H100")` → `"H100:2"` |
| `test_accelerator_str_without_type` | `_accelerator_str(1)` → `":1"` |
| `test_project_root_finds_git_dir` | walking up from subdir finds `.git` parent |
| `test_submit_cloud_generates_task_yaml_and_launches` | mock `sky jobs launch` → task YAML written, returns `(job_id, "submitted")` |
| `test_submit_cloud_parses_job_id` | parse `Job ID: 42` from sky output |
| `test_submit_cloud_error_on_parse_failure` | sky output without job ID → `RuntimeError` |
| `test_list_jobs_cloud_parses_output` | mock `sky jobs queue` output → list of `CloudJobInfo` |
| `test_cancel_job_cloud` | mock `sky jobs cancel 42 -y` called |
| `test_logs_job_cloud_no_follow` | mock `sky jobs logs 42 --no-follow` |

### 8.12 `src/tasks/checkpoints.py` (62% → 95%)

| Test | Property |
|------|----------|
| `test_load_model_no_at_symbol` | `spec="no_at_symbol"` → `ValueError` |
| `test_load_model_empty_type` | `spec="@/path/to/ckpt.pt"` → `ValueError` |
| `test_load_model_empty_path` | `spec="simple_conv@"` → `ValueError` |
| `test_load_model_missing_checkpoint` | `spec="simple_conv@/nonexistent.pt"` → `FileNotFoundError` |
| `test_load_model_unknown_type` | `spec="unknown_type@/path/to/ckpt.pt"` → `ValueError` |
| `test_load_model_with_state_dict_wrapper` | checkpoint has `{"state_dict": …}` → unwraps correctly |

### 8.13 `src/tasks/rps_prediction.py` (43% → 90%)

#### `load_predictor` with classical trackers (targets lines 53-87)

| Test | Property |
|------|----------|
| `test_load_predictor_returns_existing_predictor_as_is` | pass `RPSPredictor` → same object returned |
| `test_load_predictor_returns_nn_module_wrapped` | pass `nn.Module` → wrapped in `_ModelPredictor` |
| `test_load_predictor_classical_cepstral` | `"cepstral"` → `_ClassicalPredictor` calling `cepstral_tracker` |
| `test_load_predictor_classical_hps` | `"hps"` → `_ClassicalPredictor` calling `hps_tracker` |
| `test_load_predictor_classical_pyin` | `"pyin"` → `_ClassicalPredictor` calling `pyin_single_f0` |
| `test_load_predictor_classical_matched_filter` | `"matched_filter"` → `_ClassicalPredictor` |
| `test_load_predictor_classical_nmf` | `"nmf"` → `_ClassicalPredictor` |
| `test_load_predictor_unknown_string` | `"unknown_spec"` → `ValueError` |
| `test_load_predictor_rejects_non_string_non_predictor` | pass `42` → `TypeError` |

#### `_ModelPredictor` and `_ClassicalPredictor` wrappers

| Test | Property |
|------|----------|
| `test_model_predictor_mono_audio` | `(T,)` input → `(1, T)` → model → `(R, F)` |
| `test_model_predictor_multichannel_audio` | `(C, T)` input → model treats as batch → `(C, R, F)` |
| `test_classical_predictor_delegates_to_function` | `.predict(audio, sr)` → calls underlying fn |

#### `load_input_set` variants (targets lines 183-250)

| Test | Property |
|------|----------|
| `test_load_input_set_missing_directory` | nonexistent path → `FileNotFoundError` |
| `test_load_input_set_nested_metadata` | metadata with `{"valid": [{…}]}` → tags populated |
| `test_load_input_set_flat_metadata` | metadata as list → tags populated |
| `test_load_input_set_no_metadata` | no `metadata.json` → tags have only `"id"` |
| `test_load_input_set_multichannel_audio` | multichannel wav → audio kept as `(C, T)` |
| `test_load_input_set_wrong_sample_rate` | audio not 16 kHz → `ValueError` |

#### `_align_shape_stretch`

| Test | Property |
|------|----------|
| `test_align_shape_stretch_produces_correct_shape` | `EventSeries` with `(4, M)` values → output `(4, n_frames)` |
| `test_align_shape_stretch_no_values_raises` | EventSeries without values → `ValueError` |

#### `evaluate` error paths

| Test | Property |
|------|----------|
| `test_evaluate_with_string_spec` | `evaluate("simple_conv@/path", samples)` → lazy-loads predictor |
| `test_evaluate_with_shape_stretch_alignment` | `alignment="shape_stretch"` → uses `_align_shape_stretch` |
| `test_evaluate_with_unknown_alignment` | `alignment="bogus"` → `ValueError` |
| `test_evaluate_missing_audio_track` | TimeFrame without `"audio"` → `KeyError` |
| `test_evaluate_missing_rps_track` | TimeFrame without `"rps"` → `KeyError` |
| `test_evaluate_audio_not_uniform_series` | audio track is EventSeries → `TypeError` |
| `test_evaluate_rps_not_event_series` | rps track is UniformSeries → `TypeError` |
| `test_evaluate_per_channel_snr_tag` | `input_snr_per_channel` in tags → per-channel SNR in output rows |

#### `EvalResult` methods

| Test | Property |
|------|----------|
| `test_eval_result_per_snr_stratifies` | per-SNR rows bucketed into `[-30,-25)`, …, `[-5,0)` plus Overall |
| `test_eval_result_to_json_writes_file` | `result.to_json(path)` → valid JSON with all keys |
| `test_eval_result_to_wandb_without_wandb_installed` | `wandb` not installed → prints warning, no crash |

### 8.14 `src/tasks/cli.py` (0% → 90%)

Use `CliRunner`:

| Test | Property |
|------|----------|
| `test_cli_no_args_errors` | no arguments → help text or error |
| `test_cli_missing_input_set` | `--model simple_conv@/path` only → error |
| `test_cli_missing_models` | `--input-set /path` only → error |
| `test_cli_input_set_not_found` | `--input-set /nonexistent` → exit 1, error message |
| `test_cli_output_json_path` | `--output /tmp/out.json` → writes JSON |
| `test_cli_tex_output` | `--tex /tmp/table.tex` → writes LaTeX |
| `test_cli_multi_model_output` | two `--model` specs → both evaluated, summary table printed |
| `test_cli_alignment_invalid` | `--alignment bogus` → `ValueError` from evaluate |
| `test_cli_quiet_mode` | `--quiet` → no stdout from evaluate |

### 8.15 `src/utils/plots/` (0% → 90%)

#### `src/utils/plots/__init__.py`

| Test | Property |
|------|----------|
| `test_register_adds_to_registry` | `register("test.plot", fn)` → `get_plot_fn("test.plot") is fn` |
| `test_register_duplicate_raises` | same name twice → `ValueError` |
| `test_get_plot_fn_unknown_raises` | unknown name → `ValueError` with known names listed |
| `test_list_plot_types_returns_sorted` | `list_plot_types()` returns sorted names |

#### `src/utils/plots/rps_prediction/*.py`

Each plot function receives synthetic `EvalResult` data and returns a
`matplotlib.figure.Figure`.  Smoke tests verify:

| Test | Property |
|------|----------|
| `test_full_sequence_plot_returns_figure` | `full_sequence(…)` returns `Figure` with labeled axes |
| `test_per_snr_plot_returns_figure` | `per_snr(…)` returns `Figure` |
| `test_sample_comparison_plot_returns_figure` | `sample_comparison(…)` returns `Figure` |
| `test_summary_metrics_plot_returns_figure` | `summary_metrics(…)` returns `Figure` |
| `test_training_curves_plot_returns_figure` | `training_curves(…)` returns `Figure` |

### 8.16 `train_rps_predictor.py` (7% → 90%)

Tests marked **(P)** are property-based.  All others are unit tests with synthetic data.

#### Pure functions (no GPU, no I/O)

| Test | Property |
|------|----------|
| `test_stft_time_frames_typical` | `audio_len=16000, hop=512` → 32 frames |
| `test_stft_time_frames_edge` | `audio_len=0` → 1 frame (center padding) |
| `test_get_model_known_name` | `get_model("simple_conv")` returns `SimpleConv` instance |
| `test_get_model_unknown_name_raises` | `get_model("bogus")` → `ValueError` |
| `test_get_model_all_registry_keys` | all 13 keys produce callable model instances |
| `test_pairwise_mse_shape` **(P)** | `(B,4,T)` inputs → `(B,4,4)` output |
| `test_pairwise_mse_identity` **(P)** | `pairwise_mse(x, x)` → zero diagonal |
| `test_pairwise_mse_permutation_invariance` **(P)** | adding noise + permuting rotors → correct matching pattern |
| `test_pit_mse_loss_identity` **(P)** | `pit_mse_loss(x, x)` ≈ 0 |
| `test_pit_mse_loss_permutation_invariant` **(P)** | `pit_mse_loss(x, x[perm])` ≈ 0 for any perm |
| `test_pit_mse_loss_increases_with_noise` **(P)** | adding noise → loss increases monotonically |
| `test_flatten_channels_mono_noop` | `(B,T)` input → same output, C=1 |
| `test_flatten_channels_multichannel` | `(B,3,T)` input → `(B*3,T)` output, rps broadcast |

#### Dataset (mock filesystem)

| Test | Property |
|------|----------|
| `test_dataset_len_matches_sample_count` | `len(ds)` equals number of `sample_*` dirs |
| `test_dataset_getitem_returns_audio_rps_tuple` | `__getitem__` → `(Tensor(T,), Tensor(4, F))` |
| `test_dataset_getitem_multichannel_audio` | stereo audio → `(Tensor(2,T), Tensor(4,F))` |
| `test_dataset_resamples_rps_to_stft_frames` | RPS shape matches STFT frame count |

#### Models (CPU, no grad, 1-second audio)

| Test | Property |
|------|----------|
| `test_simple_conv_forward_shape` | `(B=2, T=16000)` → `(2, 4, ~31)` |
| `test_dcunet_enc_rps_forward_shape` | same input → `(2, 4, T_stft)` |
| `test_dccrn_enc_rps_forward_shape` | same input → correct output shape |
| `test_dccrn_lite_rps_forward_shape` | lite variant → correct output shape |
| `test_model_output_invariant_under_audio_length` **(P)** | doubling audio length doubles `T_stft` |

#### `evaluate()` with mock model

| Test | Property |
|------|----------|
| `test_evaluate_with_identity_model` | mock model returns GT → PIT MSE ≈ 0, R² ≈ 1 |
| `test_evaluate_output_keys_present` | dict has `mse`, `std_mse`, `mae_frame`, `mae_clip`, `r2`, `r2_median` |
| `test_evaluate_mae_per_rotor_length` | `mae_per_rotor` list has 4 elements |
| `test_evaluate_multichannel_iterates_channels` | stereo input → twice as many R² values |

#### `wandb_init()`

| Test | Property |
|------|----------|
| `test_wandb_init_no_key_disables` | no WANDB_API_KEY → `wandb.init(mode="disabled")` |
| `test_wandb_init_writes_run_id` | with key → `wandb_run_id.txt` written to save_path |

### 8.17 `models/rps_predictor.py` (21% → 90%)

For each model variant, verify `forward()` shape contract: `(B, T) → (B, 4, F)`.
All tests on CPU with `torch.no_grad()`.

| Test | Property |
|------|----------|
| `test_simple_conv_v2_forward` | output shape `(B, 4, T_stft)` |
| `test_simple_conv_wide_forward` | same invariant |
| `test_simple_conv_tcn_forward` | same invariant |
| `test_simple_conv_multiscale_forward` | same invariant |
| `test_simple_conv_bigru_forward` | same invariant |
| `test_simple_conv_bigru_v2_forward` | same invariant |
| `test_simple_conv_magphase_bigru_forward` | same invariant |
| `test_simple_conv_attn_pool_forward` | same invariant |
| `test_simple_conv_se_next_forward` | same invariant |
| `test_all_models_shape_invariant` **(P)** | for each model, `forward(x).shape[-1]` depends only on `x.shape[-1]`, not batch size |

### 8.18 `data_processing/dregon.py` (0% → 90%)

All tests use `tmp_path` fixtures with minimal synthetic files.

#### Pure helpers

| Test | Property |
|------|----------|
| `test_parse_mic_positions_txt_valid` | micPos matrix → `(8, 3)` array |
| `test_parse_mic_positions_txt_no_matrix_raises` | garbage content → `ValueError` |
| `test_clean_command_spikes_removes_spikes` | known spike pattern → median-filtered |
| `test_clean_command_spikes_noop_on_clean` | clean signal → unchanged |

#### I/O with tmp_path fixtures

| Test | Property |
|------|----------|
| `test_download_file_skips_existing` | dest exists → no download, returns path |
| `test_unpack_zip_actual_zip` | real .zip with `.wav` inside → extracted |
| `test_unpack_zip_raw_wav_misnamed_as_zip` | .wav with .zip extension → copied directly |
| `test_discover_recordings_finds_dirs` | tmp_path with recording dirs → correct metadata |
| `test_load_rotor_positions_from_mat` | synthetic .mat → `(4, 3)` array |
| `test_get_geometry_returns_correct_shapes` | mic `(8, 3)`, rotors `(4, 3)` |

#### `load_timeframe()` (integration-style with synthetic files)

| Test | Property |
|------|----------|
| `test_load_timeframe_returns_timeframe` | returns `TimeFrame` with expected tracks |
| `test_load_timeframe_tracks_have_correct_types` | `"audio"` is `UniformSeries`, `"motors_measured"` is `EventSeries` |
| `test_load_timeframe_tags_include_split` | tags contain `split`, `flight_type`, `recording_id` |
| `test_load_timeframe_global_data_has_geometry` | `global_data` has `mic_positions`, `rotor_positions` |
| `test_load_timeframe_all_tracks_aligned` | slicing `[t_a, t_b)` cuts all tracks simultaneously |
| `test_load_dregon_timeframes_returns_list` | `load_dregon_timeframes(split)` → list of TimeFrames |

---

## 7. Summary: expected coverage gains

| Module | Current | Target | New tests |
|--------|---------|--------|-----------|
| `_ticks.py` | 94% | 100% | 2 |
| `base.py` | 75% | 100% | 5 |
| `event.py` | 75% | 95% | 24 |
| `frame.py` | 80% | 95% | 30 |
| `segment.py` | 76% | 95% | 20 |
| `uniform.py` | 75% | 95% | 24 |
| `paths.py` | 59% | 95% | 9 |
| `task.py` | 88% | 100% | 4 |
| `cli.py` (postdoc) | 53% | 90% | 16 |
| `direct.py` | 47% | 90% | 14 |
| `cloud.py` | 38% | 90% | 12 |
| `checkpoints.py` | 62% | 95% | 6 |
| `rps_prediction.py` | 43% | 90% | 27 |
| `tasks/cli.py` | 0% | 90% | 9 |
| `plots/__init__.py` | 0% | 90% | 4 |
| `plots/rps_prediction/*` | 0% | 90% | 5 |
| `train_rps_predictor.py` | 7% | 90% | 28 |
| `models/rps_predictor.py` | 21% | 90% | 11 |
| `data_processing/dregon.py` | 0% | 90% | 16 |
| **Total** | **~55%** | **~92%** | **~266** |

**Excluded modules** (not targeting >90%):
- `src/utils/__init__.py` — legacy ZFTurbo, 322 lines
- `src/postdoc/infer.py` — requires full model-loading chain, 124 lines
- `src/postdoc/queue.py` — complex daemon, 367 lines

**Also excluded:** `src/utils/plots/cli.py` (simple argument routing, can be added
alongside `tasks/cli.py` with the same `CliRunner` pattern when needed).

---

## 8. Test organisation (proposed)

```
tests/
├── conftest.py                        # fake_git, fake_sky fixtures
├── test_cli.py                        # postdoc CLI (12→28 tests)
├── test_git_state.py                  # git_state module (14 tests)
├── test_task.py                       # SkyPilot task YAML gen (12→16 tests)
├── tasks/
│   ├── test_rps_regression.py         # RPS eval smoke + golden (3 tests)
│   ├── test_checkpoints.py            # [NEW] checkpoints.load_model (6 tests)
│   ├── test_rps_prediction.py         # [NEW] load_predictor, align, evaluate (27 tests)
│   └── test_cli.py                    # [NEW] evaluate-rps CLI (9 tests)
├── postdoc/
│   ├── test_direct.py                 # [NEW] direct SSH backend (14 tests)
│   └── test_cloud.py                  # [NEW] cloud backend (12 tests)
├── utils/
│   ├── test_paths.py                  # [NEW] path resolution (9 tests)
│   ├── data/
│   │   ├── strategies.py              # Hypothesis strategies
│   │   ├── test_ticks.py              # [NEW] _c_to_ticks + base dunders (7 tests)
│   │   ├── test_uniform.py            # UniformSeries (17→41 tests)
│   │   ├── test_event.py              # EventSeries (17→41 tests)
│   │   ├── test_segment.py            # SegmentSeries (7→27 tests)
│   │   └── test_frame.py              # TimeFrame (17→47 tests)
│   └── plots/
│       ├── test_registry.py           # [NEW] register/get/list (4 tests)
│       └── test_rps_plots.py          # [NEW] smoke tests per plot fn (5 tests)
├── train/
│   └── test_rps_predictor.py          # [NEW] pure fns, model shapes, dataset (28 tests)
└── data_processing/
    └── test_dregon.py                 # [NEW] dregon loader (16 tests)
```

---

## 8. Running coverage

```bash
# Run all tests with coverage (all modules)
python -m pytest tests/ --cov=src --cov=train_rps_predictor --cov=data_processing \
    --cov=models/rps_predictor --cov-report=term-missing

# Run only the fast unit tests (no regression tests requiring data)
python -m pytest tests/ --cov=src --cov=train_rps_predictor --cov=data_processing \
    --cov-report=term-missing -k "not test_rps_regression"

# Run with Hypothesis example database disabled (fresh random draws each time)
python -m pytest tests/utils/data/ --cov=src/utils/data --hypothesis-show-statistics

# Generate HTML report
python -m pytest tests/ --cov=src --cov=train_rps_predictor --cov=data_processing \
    --cov-report=html
```
