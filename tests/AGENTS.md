# tests/ — Framework Test Suite

Pytest suite for the whole library: data pipelines, tracking, losses, metrics,
models, tasks, plots, the training loop, the zoo, the thin `scripts/` CLIs, and
utilities. One directory per `src/` package — the layout mirrors `src/`.

## Layout

Directories mirror `src/` package by package.

| Path | Covers |
|---|---|
| `data_processing/` | Raw-source registry + DREGON/Michael's loaders, `derivations` specs and fingerprints, the dload↔tdseries `streams` bridge, Frame datasets, the online-mix compiler (offline `test_online_mixing.py`, published-frames `test_online_mixing_frames.py`, SE/`audio_pool` `test_se_online_mixing.py`), the generated-noise spawn producer, static-comb model, RPS synthesis, RPS corruption, time-warp, harmonicity |
| `tracking/` | The VK/refinement stack, one file per module: `test_vk_tracking.py`, `test_vk_kscaled.py` (k-scaled bandwidth + WP18 weight), `test_vk_blind_seeding.py`, `test_rps_refinement.py`, `test_phase_increment_tracker.py`, `test_phase_noise.py` (WP18 rank-one covariance estimator), `test_warp_refinement.py`, `test_joint_beam_tracker.py`, `test_rotor_dp.py`, `test_stages.py` (the Frame stage contract), `test_pipelines.py` (frozen vit2dsp ladder + config registry), `test_protocols.py` (window specs as data) |
| `losses/` | spectral, SI-SDR, masked, PIT, regularizers, salience, spectral/spatial likelihood, composite |
| `metrics/` | separation, RPS (PIT-aware), salience, perf + `MetricSuite` |
| `models/` | RPS predictors, CKLA (+ conditional refiner), front-end arms (G2 IF/HCQT, G4 comb, G8 pyramid), salience-RPS, generative noise gen (equivalence, RPS jitter, spectral stats), the SE baselines' conf-yaml build path (MP-SENet, SGMSE, TF-GridNet, Edge-BS-RoFormer), `test_registry_listing.py` (registry ↔ `conf/model/*.yaml` cross-check) |
| `framespec/` | `FrameSpec` structural typing (`test_spec.py`) |
| `tasks/` | `tasks.checkpoints` `Type@ckpt` loading, the live `evaluate-rps` CLI, RPS prediction + its golden-artifact regression, noise-generation codec |
| `plots/` | plot registry, RPS plots, `plots.dwym` dispatch + `data_processing.canonical` coercion, `plots.explore` notebook primitives |
| `training/` | training loop, collate, `validate_config`, val-logging, R2 `ArtifactStore` (fake client; the one real-R2 test is `network`-marked). `conftest.py` + `_fixtures.py` live here |
| `zoo/` | `zoo.cache` refresh/listing against a fake S3 client, `zoo.FrameModel` round-trip. No network |
| `scripts/` | The importable cores of the generic tools — `se_eval.py`, `table.py`, `bench.py`/`probe_ckpt.py`. `conftest.py` puts `scripts/` on `sys.path` |
| `utils/` | `utils.paths` resolution, `utils.gridrun` (resume, error isolation, aggregation) |

Root-level modules (no natural package home):

| File | Covers |
|---|---|
| `test_basic_pitch.py` | PyTorch Basic Pitch port vs the frozen `basic_pitch_ref.npz` fixture |
| `test_fkla_model.py` | Vendored flat-KLA op/layer + the FKLA RPS model against a NumPy fp64 port |
| `test_wind_wake_gen.py` | Wind-wake noise channel physics |
| `test_noise_augmentations.py` | G6 strong noise-chunk augmentations |
| `test_plot_timeframe.py` | Generic `plot_timeframe` machinery |
| `test_dataloader_benchmark.py` | Dataloader throughput smoke |
| `test_geom_calibration.py` | The **frozen** `notebooks/geom_calibration.py` study — the one deliberate `notebooks/` import; see the file's header for why it is not moved into `src` |

## Running

**Never run an unbounded `pytest`** — some tests build large tensors and a wide
`-k` selection has OOM-frozen small machines. The default `addopts` filter in
`pyproject.toml` (`-m "not slow and not network"`) now deselects the heavy and
the live-R2 tests, so a whole-suite run is survivable, but a bounded
subdirectory or single file is still the right default:

```bash
pytest tests/losses -q
pytest tests/data_processing/test_generated_noise.py -q
```

Opt back in explicitly when you mean it: `-m slow`, `-m network` (the latter
needs real R2 credentials in `.env`).

Run inside the nix devshell (`nix develop`) so torch and the editable install
resolve.

## If adding a feature

Add its test under the matching subdirectory (mirror `src/`'s layout). New
model → `tests/models/`; new loss → `tests/losses/`; new tracking stage →
`tests/tracking/`; new task behaviour → `tests/tasks/`. Keep fixtures local to
the subdirectory's `conftest.py` where possible.

Tests import `src` packages by name (`from tracking import ...`) — never
through a `sys.path` hack into `scripts/` or `notebooks/`. The two exceptions
are deliberate and documented in place: `tests/scripts/conftest.py` (the thin
CLIs are not importable packages) and `tests/test_geom_calibration.py`.
