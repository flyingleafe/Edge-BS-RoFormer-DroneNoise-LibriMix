> **Architecture:** `docs/refactor-data-pipelines.md` is the design doc for the
> data layer described here — read it first for the *why*. Every dataset is
> declared exactly once: raw sources in `sources/`, derived datasets as frozen
> pipeline specs in `derivations.py`, materialized only through
> `scripts/derive.py`.

# data_processing/ — Dataset Creation and RPS Processing

Contains scripts and modules for creating and processing training datasets.

## Why this directory exists

Dataset creation and preprocessing is a separate concern from model training.
Raw datasets need to be downloaded, processed, and mixed into training-ready
formats before any experiment can run.

## Files

The data layer has three layers, each declared exactly once — see
`docs/refactor-data-pipelines.md` for the architecture.

### Layer 1 — raw sources (`sources/`)

| File | Purpose |
|------|---------|
| `sources/__init__.py` | **The** external-dataset registry (`REGISTRY`): one `SourceDataset` per raw dataset — a pinned `DownloadSpec` / custom `fetcher` / `raw_dataset` dload pin, plus a `builder(raw_dir) -> Iterator[(key, td.Frame)]` producing rich `tdframe-v1` recording Frames. DREGON gets no preferential treatment: it is one entry like MIMII or AVQ. Helpers: `get`/`list_names`/`raw_root`/`iter_frames`/`iter_recording_frames`/`geometry`/`dataset_meta`. Torch-free. |
| `sources/_common.py` | Shared builder helpers: `meta_frame`/`audio_frame`/`clean_dict`/`mic_ring`/`safe_key`/`iter_audio_files`/`read_audio_file`, plus the `LAYOUT = "tdframe-v1"` marker. |
| `sources/dregon.py` | DREGON builder + raw-tree loaders (`load_timeframe`, `load_dregon_timeframes`, `discover_recordings`, `get_geometry`, `clean_command_spikes`, `download_dregon`). |
| `sources/michaels.py` | Michael's DJI Matrice 100 rig: `MICHAELS_FILES` alignment constants (paths **relative to the `recording_with_motor_speed` tree**; use `resolve_raw_root()` to resolve any root form), `load_raw_aligned`, `build`/`build_frame`, `load_michaels_timeframes`, `get_geometry`. |
| `sources/{aerosonicdb,avq,droneaudio,hornbase,hustmotor,kaist,mimii,spcup19}.py` | One builder module per external harmonic-noise dataset (provenance + `build`). |
| `downloaders.py` | Reproducible fetch helpers (thin wrappers, no bespoke sync): `zenodo_fetch`/`http_fetch`/`mendeley_fetch` (`requests`), `hf_fetch` (`huggingface_hub`), `gdrive_fetch` (`gdown`), `extract_zip`. Idempotent; heavy imports deferred. |

### Layer 2 — derived datasets (`derivations.py`, `mixing.py`)

| File | Purpose |
|------|---------|
| `derivations.py` | **Every** derived dataset as a frozen dload pipeline spec (`SPECS`) + its module-level generator: `generate_dregon_lm_split` / `generate_dn_lm_split` (LibriMix-style `sample-dir-v1` mixes), `generate_source_frames` (any sources builder as a derivation), `generate_frame_subset`, `generate_raw_subset`, `generate_pcm16_mono`, `generate_beatvk_valid`, `generate_se_valid`. Also `PARENTS` (pinned parent URIs, drift-guarded by tests), `SE_CATEGORY_NOISE`/`SE_HELDOUT_SPEAKERS`, `HISTORICAL_PINS`, and `build_pipeline`/`dataset_meta`/`fingerprint`. Torch-free. Driver: `scripts/derive.py` (`list`/`derive`/`adopt`). |
| `mixing.py` | The **single copy** of the per-sample mixing math (torch-free, disk-free): `load_audio`/`adjust_length`/`normalize_audio`, `scale_noise_to_snr`/`mix_audio`/`mix_at_snr`/`calculate_snr` (offline LibriMix convention — speech is the reference), `scale_source_to_snr`/`mix_at_source_to_noise_snr`/`is_silent` (online/streaming convention — noise is the reference), `resolve_motor_tracks`/`find_inflight_window`/`extract_multichannel_noise_chunk`, `load_noise_source_frames`, `render_multichannel_sample`, `iter_real_valid_clips`, `mix_dn_lm`. |
| `frames.py` | Shared `td.Frame` conventions: `audio_series`/`rps_series`/`resample_audio_series`, `get_meta`/`meta_dict`/`with_meta`, `adapt_recording_frame` (rich published recording → the minimal audio+rps noise Frame), `make_recording_frame`. |

### Layer 3 — consumption (streams, online mixing, torch Datasets)

| File | Purpose |
|------|---------|
| `streams.py` | dload ↔ tdseries bridge: `DloadFrameDataset`, the generic `tdframe-v1` Frame codec (`frame_to_sample`/`sample_to_frame`), pipeline combinators (`to_frames`/`frame_windows`/`mix_frames`/`resample_frames`), `ensure_local`/`resolve_source` (`dload:` URIs), `iter_published_frames`, `open_repository`/`local_repository`. |
| `online_mixing.py` | The online-mix **compiler**: a policy YAML → one infinite `dload.Pipeline` of per-sample `td.Frame`s (`build_online_mix_pipeline`), built from `build_noise_stream` (real `kind: frames` records, synthetic engines, `kind: audio_pool`) and `build_speech_stream` (`include`/`exclude` speaker filters). Curriculum staging + per-sample-id augmentation RNG (`make_rng`) are unchanged. |
| `frame_datasets.py` | torch `Dataset` adapters: `DregonLMFrameDataset`, `DNLMFrameDataset`, `SEValidFrameDataset`, `OnlineMixFrameDataset` (thin wrapper over the compiled pipeline; `flatten_channels` = a `flat_map` stage, `rps_corruption` = a `map` stage), `NoiseGenFrameDataset`. |
| `noise_rps_dataset.py` | `NoiseRPSDataset` — combined chunkable dataset over DREGON `in_flight_noise` + Michael's (accepts `frames:NAME[@VER]` specs, `dload:` URIs, or local paths). |
| `generated_noise.py` | `GeneratedNoisePool` — a trained `PositionalHarmonicNoiseGen` as a noise **source** (`kind: generated`). One background **spawn** producer process renders chunks into a shared-memory ring buffer; fork DataLoader workers read finished chunks (lock-free seqlock). See § "Generated noise source". |
| `gp_noise.py` | `GPRotorNoisePool` — a trained per-drone egonoise GP as a noise **source** (`kind: gp`, G3). See § "GP rotor-noise source". |
| `rotor_spectral_model.py` | `StaticCombNoisePool` — analytic static-comb noise source (`kind: static_comb`). |
| `rps_corruption.py` | Synthetic corruption of clean RPS tracks for the **conditional refiner**. Seeded via `make_rng(seed, sample_id)`; wired through `frame_datasets.{DregonLMFrameDataset,OnlineMixFrameDataset}(rps_corruption=...)`, which then emit an extra `rps_cond` entry. |
| `noise_augmentations.py` | Strong **noise-chunk** augmentation family (`policy.noise_augmentations`, G6). See § "Strong noise augmentations". |
| `time_warp.py` | `policy.noise_time_warp` — angular resampling of the noise+RPS pair. |
| `harmonicity.py` | `measure_harmonicity(audio, sr)` → `Harmonicity`. Torch-free; the **analysis-stage** measure of "how harmonic" a noise source is (not baked into publish). |
| `rps_synthesis.py`, `rps_refinement.py`, `vk_tracking.py`, `vk_blind_seeding.py`, `phase_increment_tracker.py`, `warp_refinement.py`, `collate.py` | RPS synthesis / refinement / Vold–Kalman tracking / batching (unchanged by the data refactor). |


## DREGON Recording Inventory

All recordings are **8-channel, 44.1 kHz**; motor rate ≈ 929 Hz.

| Split | Recording ID | Duration | `motors_measured` | `motors_command` | Notes |
|-------|-------------|----------|-------------------|------------------|-------|
| `in_flight_noise` | `free-flight_nosource_room1` | 71 s | ✓ | ✓ | pure drone noise |
| `in_flight_noise` | `free-flight_nosource_room2` | 82 s | — | ✓ | pure drone noise |
| `in_flight_noise` | `hovering_nosource_room2` | 45 s | — | ✓ | hovering |
| `in_flight_noise` | `updown_nosource_room2` | 54 s | — | ✓ | up-down manoeuvre |
| `in_flight_noise` | `rectangle_nosource_room2` | 44 s | — | ✓ | rectangle path |
| `in_flight_noise` | `spinning_nosource_room2` | 41 s | — | ✓ | spinning |
| `in_flight_source` | `free-flight_speech-low_room1` | 63 s | ✓ | ✓ | drone + co-recorded speech (low level) |
| `in_flight_source` | `free-flight_speech-high_room1` | 53 s | ✓ | ✓ | drone + co-recorded speech (high level) |
| `in_flight_source` | `free-flight_whitenoise-low_room1` | 65 s | ✓ | ✓ | drone + co-recorded whitenoise (low) |
| `in_flight_source` | `free-flight_whitenoise-high_room1` | 61 s | ✓ | ✓ | drone + co-recorded whitenoise (high) |
| `noise_free` | `silent-flight_whitenoise-low_room1` | 78 s | — | — | no drone motors |
| `clean_source` | `clean_speech_*`, `clean_whitenoise_*`, `clean_chirps_*` | 1–10 s | — | — | isolated sources, no drone |
| `motor` | `motor_Motor{1-4}_{50,60,70,80,90}`, `motor_allMotors_70` | 25–45 s | — | — | individual/combined motor runs |

### Telemetry structure: `motors_command` vs `motors_measured`

Both tracks are EventSeries with values shape `(4, M)` (time-last, 4 rotors).

- **`motors_command`**: commanded RPS from the flight controller.  Present in
  ALL recordings with motor data.  Suffers from two logging artefacts:
  - *Leading freeze*: the first N samples are stuck at a constant high value
    before the real command sequence begins.  `clean_command_spikes` zeros this.
  - *Trailing freeze*: the last 45–1577 samples are also stuck at a constant
    value — the logger stopped updating before the motors spun down.  Landing
    is **NOT** visible in command values.
- **`motors_measured`**: actual measured rotor speeds.  Present only in
  `free-flight_*_room1` recordings (5 total).  Shows the real spindown during
  landing (drops to ~55–78 RPS in the last few seconds before the trailing
  freeze).  **Use this for in-flight window detection when available.**

### In-flight window (takeoff / landing trim)

All recordings begin with ~5–13 s of pre-takeoff/ramp-up:

| Recording | trim_start | trim_end | inflight | detect key |
|-----------|-----------|---------|---------|-----------|
| free-flight_nosource_room1 | 9.4 s | 0.2 s | 59.8 s | measured |
| free-flight_nosource_room2 | 8.1 s | 0.0 s | 70.7 s | command |
| hovering_nosource_room2 | 7.2 s | 0.0 s | 32.8 s | command |
| updown_nosource_room2 | 5.7 s | 0.0 s | 41.6 s | command |
| rectangle_nosource_room2 | 7.6 s | 0.0 s | 33.4 s | command |
| spinning_nosource_room2 | 8.7 s | 0.0 s | 28.0 s | command |
| free-flight_speech-low_room1 | 9.8 s | 0.4 s | 50.7 s | measured |
| free-flight_speech-high_room1 | 8.6 s | 0.4 s | 42.9 s | measured |
| free-flight_whitenoise-low_room1 | 12.2 s | 0.4 s | 49.9 s | measured |
| free-flight_whitenoise-high_room1 | 9.8 s | 0.4 s | 47.5 s | measured |

(Computed with `min_motor_rps: 30.0`, the recommended default.)

`mixing.find_inflight_window(tf, motor_key, min_motor_rps, clean=…)` implements
this; `mixing.resolve_motor_tracks` picks the key — `motors_measured` for
detection when present (it shows the real spindown), falling back to
`motors_command`. The saved `rps.npy` always comes from `motors_command`
(cleaner signal). Published `*-frames` datasets expose a single, already-clean
`rps` entry, so detection there runs on the command track.

---

## Dataset Variants

Every variant below is a **frozen spec** in `derivations.SPECS`; there are no
creation CLIs. `python scripts/derive.py list -v` prints them with their notes
and fingerprints; `derive <NAME>` materializes one, `adopt <NAME> --commit`
points a derivation ref at an existing historical pin.

### DN-LM (DroneNoise-LibriMix) — Paper 1

Specs `DN-LM-train` / `DN-LM-valid` (generator `dn_lm`, derivable).

- Sources: LibriSpeech `train-clean-100` + `drone_audio/Binary_Drone_Audio/yes_drone`
  (the label-1 recordings; the raw tree's `*/unknown/` mixes ESC-50/WN/silence
  negatives, which the paper's DN-LM excluded)
- 1 s samples, 16 kHz mono, SNR −30…0 dB; inverse-distance attenuation
- Split: 6480 train / 720 valid

### DREGON-LM V4 — current (multichannel)

Specs `DREGON-LM-V4-{train,valid}`, `DREGON-LM-V4-michaels-{train,valid}`,
`DREGON-LM-V4-michaels-valid-full` (generator `dregon_lm`, **adopt-only** —
the published bytes are the historical uploads; the generators reproduce the
recipe but the mixing RNG is not byte-stable across machines).

- `mode: "synthesized"` (train): noise chunks from the published frames
  datasets + LibriSpeech, mixed per channel. Per sample: `mixture.wav (T, 8)`,
  `vocals.wav`, `noise.wav`, `rps.npy (4, M)`, `meta.json`.
- `mode: "real_valid"` (valid): raw 8-channel clips from `in_flight_source`
  recordings, **no mixing**. `mixture.wav` = the raw recording, `rps.npy` =
  telemetry; there is no `vocals`/`noise` (no clean reference exists).
  `params.max_non_overlapping: true` emits every non-overlapping clip of every
  in-flight window instead of `num_samples` random draws.
- Shared V4 knobs (`_V4_PARAMS`): 1 s train / 8 s valid, 16 kHz,
  SNR U[−30, 0], `speech_per_channel: independent`,
  `source_white_noise_prob: 0.3`, `min_motor_rps: 30.0`.

### Superseded variants

The V1/V2/V3/test recipes and the one-off `rps_*` probe sets are consumed as
plain pinned uploads and are **not** re-derived — see
`derivations.HISTORICAL_PINS` (and git history for the deleted CLIs).

## Composing a noise pool from published frames

`mixing.load_noise_source_frames(specs, sample_rate=...)` builds a
`list[td.Frame]` noise pool from declarative specs — the **same schema** the
online-mix `kind: frames` source uses, so a derivation and a training policy
select noise identically:

| Key | Meaning |
|-----|---------|
| `dataset` | published frames dataset name, optionally `NAME@version` (required) |
| `splits` / `split` | filter on frame `meta.split` (e.g. `in_flight_noise`) |
| `recording_ids` | keep only these recording ids |
| `exclude_recording_ids` | drop these recording ids |
| `take` | cap the number of recordings |

```python
[
  {"dataset": "DREGON-frames", "splits": ["in_flight_noise"],
   "exclude_recording_ids": ["free-flight_nosource_room1"]},
  {"dataset": "michaels-frames", "recording_ids": ["FLY125"]},
]
```

Fixes are baked in at **derivation** time — DREGON's `motors_command` is
already `clean_command_spikes`-cleaned, michaels' `rps` already aligned — so
`adapt_recording_frame` renames the rotor track to the generic `rps` entry and
`mixing.resolve_motor_tracks` re-applies nothing. Michael's telemetry logs at
~29 Hz vs DREGON's ~929 Hz, so `rps.npy` is shorter; handled downstream via
the per-sample `motor_sample_rate`.


## Online Mixing for RPS Prediction

`data_processing/online_mixing.py` contains the config-in/stream-out online mixer:
`OnlineMixIterableDataset.from_config(cfg)` returns an infinite `IterableDataset`
that uses `TimeFrame` internally for aligned noise+RPS slicing and yields
model-ready `(audio, rps_target)` tensors. Unaligned speech/source audio is plain
NumPy/memmap/tensor data, not a new project container.

Fresh-session map for online RPS training:
1. Read this section, then `configs/AGENTS.md` § "Online-mixing configs".
2. Loader implementation: `src/data_processing/online_mixing.py`.
3. Durable policies: `conf/online_mix/online_mix_*.yaml`.
4. Training integration: `python train.py experiment=<name>` where the experiment
   overrides `data:` to a `conf/data/*.yaml` entry wrapping `OnlineMixIterableDataset`
   / `OnlineMixFrameDataset.from_yaml` over the policy YAML (see
   `conf/data/online_mix_v4_michaels.yaml`), plus `samples_per_validation=<N>`
   (top-level Hydra field, `conf/config.yaml`). The stream is infinite;
   `samples_per_validation` defines the arbitrary validation cadence/"epoch" size.
5. Validation remains fixed via the `valid:` entry of the `conf/data/*.yaml` config
   (a plain `DregonLMFrameDataset` over `<dataset>/valid`); do **not** use an
   advancing online validation stream for early stopping.

Source/cache interface rules:
- Public interface is config-in/stream-out. Do not add source-cache prep scripts
  or cache-specific CLI flags; cache/memmap optimizations belong behind
  `AudioFileSourcePool.from_config(...)`.
- `cache.mode: packed_int16` creates/reuses a packed PCM16 source cache. Its
  location is `sources.speech[].cache.dir`, normally
  `${oc.env:ONLINE_MIX_SOURCE_CACHE_DIR,.cache/online_mix_sources}`. Put
  `ONLINE_MIX_SOURCE_CACHE_DIR=/large/partition/online_mix_sources` in `.env` on
  machines where repo-local `.cache/` is the wrong partition.

### Generated noise source (`kind: generated`)

A trained noise generator can be listed as a `sources.noise` entry exactly like a
real recording — the payoff being *unlimited* rotating-noise variety with an
*exact* RPS label. Implementation: `data_processing/generated_noise.py`
(`GeneratedNoisePool`), wired into `build_noise_pool(...)` (the dispatcher
`OnlineMixIterableDataset.from_config` now uses). Example config:
`conf/online_mix/online_mix_generated_augment_example.yaml`.

Why the process/buffer design (option C): the mixer runs in **forked** DataLoader
workers, and CUDA cannot init in a forked child. So one **spawn** producer owns
the single generation CUDA context and renders batches into a **shared-memory ring
buffer** (`torch` shared tensors); the fork workers only read finished chunks.
Reads are lock-free via a per-slot **seqlock** (`version` odd=writing; reader
retries if odd or changed across its copy) — no mutex across the spawn/fork split.
Generation rate is **decoupled** from consumption: workers sample-with-replacement
from filled slots, so a slow GPU just means more chunk reuse (fine for an
augmentation source). The producer is started once in the main process (never in a
worker); `close()`/`atexit` tears it down.

Config fields (defaults in parens): `checkpoint` (bundle path, required); `drone`
(codebook key + geometry source, michaels/dregon); `n_harmonics` (**must match the
checkpoint**); `device` (cuda:0 — the one extra context); `gen_batch` (32);
`random_phase` (true — per-chunk harmonic phases for extra texture, model stays in
eval); `refresh` (true; **false** = fill the buffer once for a reproducible fixed
bank); `rps.kind` (`synthetic_intermittent` only) + `rps.aggressiveness` (1.0);
`buffer.slots` (512 ≈ 384 MB) + `buffer.warmup` (16); `weight` (mix weight, 1.0
per source item — a bare `[dregon, michaels, generated]` list is duration-weighted
within reals, then real-pool vs generated at these pool-level weights).
Determinism caveat: a live (`refresh: true`) stream is **not** seed-reproducible
(buffer contents depend on timing) — keep validation on real/fixed sources, or use
`refresh: false`.

**Vicinal `interp` mode** (E7 — `conf/online_mix/rps_generated_only_interp.yaml`):
instead of one fixed `drone`, an `interp:` sub-block makes each producer batch
sample a *novel* drone along the DREGON↔Michael's embedding segment, so the
consumer (e.g. an RPS predictor) sees a continuum of timbres/geometries. Per
batch: draw `α ~ U(alpha.low, alpha.high)`; `z = (1−α)·z0 + α·z1 +
N(0, embedding_noise·‖z1−z0‖)`; `rotor_interp` linearly interpolates rotor
positions at α; `jitter_sigma: interp` blends the learned per-drone OU σ at α
(or a float, or `off`) and is forced ON at eval; `mic_sampling` picks a rig
(`rigs`, `prob`) **independently** of α and jitters each mic by
`N(0, jitter_std m)` — the vicinity of the real arrays. `endpoints` are codebook
names. α also feeds the `rps_synthesis` `drone_profile` blend. Requires a
**flat conditioned checkpoint** (`_CodebookConditionedNoiseGen` state_dict with
`codebook.codes.*` + optional `log_jitter_sigma.*`; the modern `training.loop`
format) — `_load_generator` rebuilds the exact composite via
`models.registry.build_noise_gen_model` (spectral-norm / per-drone-σ aware),
so the reduced `save_bundle` (no σ) is single-drone only. `checkpoint` accepts
an `r2://` URI (auto-downloaded via `training.artifacts.resolve_checkpoint_uri`);
set `dregon_dir: dload:DREGON` on cloud so the producer can load DREGON geometry.

Benchmark notes:
- Noise-gen inference (`PositionalHarmonicNoiseGen`, 236k params, mostly FFTs) is
  ~128 ms per 1 s 8-mic chunk on CPU (batched); GPU is far faster, which is why the
  producer renders on the GPU in `gen_batch` batches.
- Early local smoke using generated `DREGON-LM-V4-michaels/train/**/vocals.wav`
  is obsolete; online training should use original LibriSpeech files.
- Correct V4-Michaels setup (`data/librispeech/LibriSpeech/train-clean-100`,
  DREGON train noise excluding `free-flight_nosource_room1`, plus Michael's
  `FLY125` only), `batch_size=16`, `num_workers=4`,
  `speech_per_channel=independent`: direct FLAC decode was about `2.9 batch/s`;
  internal `cache.mode: packed_int16` creates/reuses
  `${ONLINE_MIX_SOURCE_CACHE_DIR:-.cache/online_mix_sources}/*` and reaches about
  `13.5 batch/s` / `1728 audio-clip/s` on cache reuse. Set
  `ONLINE_MIX_SOURCE_CACHE_DIR` in `.env` to place this cache on another partition.
- Fixed precomputed loader is about `21 batch/s` / `5394 audio-clip/s`.
Optimize only behind the same public API.

### GP rotor-noise source (`kind: gp`) — G3

`data_processing/gp_noise.py` (`GPRotorNoisePool`): the per-drone **egonoise
GP** checkpoints (`train_egonoise_gp.py`; `r2://ml-data/artifacts/gp_egonoise/
{dregon,matrice100}/best.pt`) as an online-mix noise source with exact
synthetic RPS labels. Architecture mirrors the static comb, not the neural
producer: the GP posterior is batch-queried **once at pool init** on a dense
rps grid at the rig mic positions and reduced to a `(G, M, 2H+1)` coefficient
table (~1.6 MB, picklable; the gpytorch model is dropped), so per-chunk
synthesis is pure numpy in the fork workers (~200 ms per 1 s 8-mic chunk):
rps-interpolate coefficients at the chunk-mean rps, FM-synthesize the comb at
`render_fs` 24 kHz (anti-aliased), add the checkpoint's σ_b(rps) colored
broadband, decimate to 16 kHz, normalize global RMS. Key config: `checkpoint`
(r2:// ok), `drone` (dregon/michaels — geometry + `rps_synthesis` profile),
`mic_mode: shell` (default — rig mics projected radially onto the GP's
training shell; native rig positions are ~3 lengthscales out-of-support and
mean-revert), `rotor_mode: per_rotor` (default — Σ_r S(mic, rps_r)/R, exact
non-degenerate per-rotor labels; `mean` = four_way_lib convention, degenerate
labels), `broadband`, `rps.kind: synthetic_intermittent` **only** (GP support
is rps 40–85; no full-flight excitation). Policy example:
`conf/online_mix/g3_gp_aug_dload.yaml`; batch doc
`docs/experiments/g3-gp-curriculum.md`.

### Strong noise augmentations (`policy.noise_augmentations`) — G6

`data_processing/noise_augmentations.py`: six strong transforms of the
**noise chunk** (audio + RPS pair, applied before speech mixing — unlike
`policy.augmentations`, which is post-mix on the mixture and provably weak
for RPS prediction: polarity is an exact no-op for mag/IF front-ends, gain a
log-offset). Same `probability` + `choices` fire/choice schema; on a hit the
chunk Frame is rebuilt in `time_warp.apply_time_warp`'s output convention
(audio exactly `target_len` + a clean uniform 100 Hz `rps` track), so
downstream mixing/target interpolation are untouched. The six: `freq_scale`
(resample by α∈U(0.75,1.3) at natural scaled length — the sourcing pipeline
oversamples the noise window by α_high so the downstream target_len crop
never pads; labels ×α on the scaled time base — the one that manufactures
new (audio, RPS) pairs), `spectral_recolor` (smooth random EQ ±8 dB, 10 log-spaced anchors,
per channel), `random_reverb` (deterministic 200-RIR synthetic bank: RT60
U(0.1,0.8) s, DRR U(3,15) dB, exp-decay colored tails; RMS renormalized),
`tooth_dropout` (zero ±2 STFT bins around k·rps_r(t), 1–4 random teeth k≤25 —
label-aware), `spec_mask` (SpecAugment bands/time masks), `floor_inject`
(1/f^tilt floor at U(−20,0) dB rel RMS). STFT ops run on the model's own
2048/512 grid. Keep the 50k unaugmented warmup stage — G5 measured it as
load-bearing. Policy: `conf/online_mix/g6_strongaug_dload.yaml`; batch doc
`docs/experiments/g1-vk-parity.md` § "Phase G6"; tests
`tests/test_noise_augmentations.py`.

### Speech-enhancement target mode (`task: speech_enhancement`) — F1 baselines

The online mixer doubles as a **speech-enhancement** training stream. Set
`task: speech_enhancement` at the top of the policy YAML and
`OnlineMixIterableDataset` yields `(mixture, clean_speech)` instead of
`(audio, rps_target)`: the clean target is the gain-scaled speech exactly as
mixed (SNR of the returned pair == the drawn SNR), post-mix augmentation
(`random_gain`/`random_polarity`) is applied *identically* to mixture and
target, and RPS interpolation is skipped (so telemetry-free noise sources
work). The stream is **mono** — a random mic channel is picked from
multichannel noise (DREGON/Michael's 8-ch). `OnlineMixFrameDataset` packs each
pair into a `{mixture, target, meta}` Frame (the DN-LM layout the SE task /
`losses.MaskedLoss` consume). Speech always drawn (a clean reference is
required). See `conf/online_mix/se_{drone_only,all_harmonic}.yaml`,
`docs/experiments/f1-se-blind-baselines.md`.

### Telemetry-free audio noise pool (`kind: audio_pool`)

A dload-backed audio dataset exposed as a noise pool **without** any rotor
telemetry (`DloadAudioPool`): random recording (shard weighted by sample
count), random channel, resample to 16 kHz, loop/pad to the chunk. Streams
lazily at *shard* granularity via dload's `PackReader` (never materializes the
whole dataset — works on 258 GiB MIMII / 88 GiB DroneAudioSet where
`TimeFrameNoisePool` would OOM). Handles both `tdframe-v1` (audio under the
`audio` entry) and raw-audio datasets; skips non-audio samples (e.g.
`new-drone-noises` csv flight logs); zip-blob datasets (`zenodo_drone_noises`)
unsupported. Usable only for `speech_enhancement` (no rotor track).

```yaml
sources:
  noise:
    - kind: audio_pool
      dataset: MIMII                 # any dload dataset name
      channel: random               # or an int
      holdout: {split: train, valid_shards: 2}   # leak-free train/valid split
      include_keys: [S1_seq1]       # optional: restrict to named recordings
      exclude_keys: []              # optional: drop named recordings
      weight: 1.0                   # per-source weight in the MixedNoisePool
```

`include_keys` / `exclude_keys` restrict the pool to specific recordings — a
sample key is kept when it *equals* or *contains* a listed entry (exact key or
substring), `exclude_keys` applied last. Keys are shard-local (the manifest has
no key list), so filtering stays shard-lazy: a shard is dropped (draw weight
zeroed) the first time it is opened and found to hold no match, and the draw
retried.

**Cost, and when to publish a subset instead.** Because the manifest has no key
list, *every* shard must be opened (i.e. downloaded) at least once before the
pool knows it holds nothing wanted. Filtering a many-shard dataset down to a few
recordings therefore pays the dataset's full download for a tiny slice. The F2
replication originally selected the 5 AVQ *ego-noise* sequences this way
(`dataset: AVQ`, `channel: 0`, `include_keys: [S1_seq1, …]`) and paid ~4 GiB
across 11 shards for ~12 minutes of audio — and hit a second problem: AVQ's
352 MB shard is a 42-part multipart upload whose ETag s3transfer rejects on some
boto3 builds (Kaggle), aborting the download of a perfectly intact object. Both
costs vanish if the wanted audio is its own dataset, so it now reads the
purpose-built **`AVQ-egonoise`** (see Catalog) with no `include_keys`/`channel`
at all. Rule of thumb: use `include_keys` for ad-hoc/exploratory restriction;
publish a derived subset (the `frame_subset` generator in `derivations.py` is the template)
once a fixed small selection becomes an experiment's durable noise pool.

`holdout` reserves the last `valid_shards` whole shards (= whole recording
groups) as the *valid* partition and the rest as *train* (single-shard
datasets fall back to a per-shard sample-index split at `fraction`, default
0.1). `derivations.generate_se_valid` uses `split: valid` with the same
`valid_shards` to build the fixed SE valid sets, kept complementary to the
training pools' `split: train`. `AudioFileSourcePool` also gained an
`exclude:` list (path-substring drop) — used to hold LibriSpeech speakers out
of training speech. The map-style `SEValidFrameDataset` streams a published SE
valid set (`SE-valid-drone` / `SE-valid-harmonic` / `SE-valid-avq-survey`) as
`{mixture, target, meta}` frames for `eval.py`; `local_root=<dir>` instead reads
an **unpublished** set from a local dload repository
(`streams.local_repository`, written via `streams.local_repository`). The
builder's rate / SNR grid / duration are per-target-set presets
(`DATASET_PRESETS`, CLI-overridable), so 8 kHz replications reuse it unchanged.

### Published rich-frame noise source (`kind: frames`)

The fixed rich-frame datasets published by the `source_frames` derivation
(`DREGON-frames`, `michaels-frames`; dload `tdframe-v1` layout, decoded by
`data_processing.streams`) can feed the noise pool directly. **Fixes are baked
in at publish time** — DREGON `motors_command` is already
`clean_command_spikes`-cleaned, michaels `rps` is already aligned — so the
loader re-applies nothing: it renames the rotor track to the generic `rps`
entry (the no-cleaning path of `_resolve_motor_tracks`), keeps only
`audio` + `rps` + `meta` per recording (IMU/GPS/raw telemetry dropped, one
frame decoded at a time), and soxr-resamples audio to the pool `sample_rate`.

```yaml
sources:
  noise:
    - kind: frames
      dataset: DREGON-frames        # dload dataset name (tdframe-v1)
      # version: <manifest hash>    # optional; default = dload.lock pin / latest
      splits: [in_flight_noise]     # optional filter on frame meta.split
      exclude_recording_ids: [free-flight_nosource_room1]
      min_motor_rps: 30.0
    - kind: frames
      dataset: michaels-frames
      recording_ids: [FLY125]       # bare published ids (not michaels_FLY125)
```

Also accepts `split` (singular), `recording_ids`, and `take` (cap the number
of recordings). A real source (`dregon`/`michaels`/`frames`) that sets an
**explicit `weight`** becomes its own sub-pool in a `MixedNoisePool` at that
pool-level weight instead of joining the duration-weighted real merge (whose
weight stays 1.0 per unweighted item) — the knob that lets a long auxiliary
corpus enter at a modest share, e.g. `AVQ-egonoise-vkrps` (~617 s) at
`weight: 0.5` vs the merged DREGON+michaels 2.0 in
`conf/online_mix/beatvk_avq_dload.yaml` (= 20% of noise chunks; being mono,
~3% of training frames under `flatten_channels`). Unweighted-only source
lists behave exactly as before. Nuance vs `kind: dregon`: after adaptation there is no
separate `motors_measured` detect track, so the in-flight window is detected
on the cleaned `motors_command` — the command's trailing logging freeze is not
trimmed (same behaviour as the command-only room2 recordings). Likewise,
`noise_rps_dataset.build_noise_rps_datasets` accepts
`dregon_dir="frames:DREGON-frames[@VERSION]"` /
`michaels_dir="frames:michaels-frames"` in place of local folders.

## Stream sanity-checking (`scripts/check_stream.py`)

**Rule: run this against any new or edited online-mix policy BEFORE submitting
training jobs, and re-run it whenever the data path is refactored** (loader,
wrapper, flattening, policy resolution). It exists because of the CKLA staging
bug (docs/experiments/ckla.md § "THE STAGING BUG"): `flatten_channels=True`
turns each generated chunk into C=8 training frames, so `until:` stage
boundaries (in chunk units) sat at effective epoch ~80 instead of 10 and the
staged augmentations silently never fired for ~3 weeks of experiments —
config inspection and the provenance print both looked correct; only the
stream itself showed the truth.

```bash
python scripts/check_stream.py --policy conf/online_mix/<name>.yaml \
    --flatten --samples-per-epoch 5000 --epochs 0 5 10 12 [--probes 48]
python scripts/check_stream.py --experiment <name>   # checks exactly what
    # `train.py experiment=<name>` trains on (policy path, flatten_channels,
    # samples_per_validation pulled from the composed Hydra config)
```

It measures, on the actually-generated stream: (1) the chunk→frame expansion
ratio C; (2) each stage boundary in chunks / frames / effective epochs (WARN
beyond `--warn-boundary-epoch`); (3) **empirical** per-key fire rates of
`augmentations` / `noise_augmentations` / `noise_time_warp` at the probed
epochs vs the configured probabilities (exact binomial test), plus the
label-diff rate (freq_scale must change labels on every nonzero-RPS fire —
all-zero-RPS full-flight chunks are exempt, `0*alpha == 0`; post-mix augs
must never touch labels);
(4) per-id determinism. Nonzero exit on any FAIL — usable as a submission
gate. Fire detection compares each sample against a control stream whose aug
block has `probability: 1e-9` (never removes the key — a present-but-missed
block still consumes the fire-decision RNG draw, so stripping it would shift
the whole downstream RNG stream and fake a ~1.0 rate).

## Multichannel Training & Evaluation Wiring

### Training (unified `train.py`, via `data_processing.frame_datasets.DregonLMFrameDataset`)

The legacy `train_rps_predictor.py` (and its `DREGONRPSDataset` /
`_flatten_channels`) has been deleted — see docs/refactor-unified-framework.md.
`DregonLMFrameDataset.__getitem__` (this package's `frame_datasets.py`) returns
a `td.Frame` per sample instead of a raw `(audio, rps)` tensor pair:
- Mono files (`C=1`): `frame["mixture"]` is `(time,)`
- Multichannel files (`C>1`): `frame["mixture"]` is `(mic, time)`

`frame["rps"]` is `(rotor, time)` on the STFT frame grid in both cases.
`_flatten_channels`'s channel-as-extra-batch-item trick (broadcasting a
`(B, 4, F)` RPS target across `(B, C, T)` audio) is now reproduced at the
*data* level (C9, REPLICATION.md § C9): `channel=<int>` on
`DregonLMFrameDataset` still selects one mic deterministically (a genuinely
mono `(time,)` Frame per sample); `flatten_channels=True` instead expands
each multichannel sample into `n_channels` separate mono-view Frames (one
per mic, `len(dataset) = n_samples * n_channels`), each broadcasting the
recording's single RPS target and tagging `meta.channel` with which mic it
came from — `conf/data/dregon_lm_v4_8ch_flat.yaml`.

`SimpleConv` (and all `rps_predictor` model variants) accept any leading batch
shape before the time dimension — `torch.stft` treats everything before `T` as
batch.  DCUNet/DCCRN are **not** covered by channel flattening; they handle
1-channel mono only.

### Evaluation (`src/tasks/rps_prediction.py`)

`load_input_set(path)` yields `TimeFrame` objects where:
- Mono audio: `audio.samples` shape `(T,)`
- Multichannel audio: `audio.samples` shape `(C, T)`

`_ModelPredictor.predict(audio)`:
- `(T,)` → unsqueeze → model → `(R, F)` (mono, backward-compat)
- `(C, T)` → model directly → `(C, R, F)` (channel treated as batch)

`evaluate(predictor, samples)` expands mono pred to `(1, R, F)` and loops over
channels, emitting one row per `(sample, channel)` pair in `EvalResult.per_sample`:

```
{"sample": "sample_00000", "channel": 0, "mse": …, "mae_frame": …, "r2": …, "input_snr": …}
```

Aggregate has both `n_samples` (distinct samples) and `n_rows` (= n_samples × C).

---

## RPS Processing

`sources/dregon.py` provides:
- `load_timeframe(sample)` — load one raw recording as a `td.Frame`
- `load_dregon_timeframes(data_dir, splits=…)` — all recordings in `splits`
  (`data_dir` may be the `DREGON` tree or its parent)
- `clean_command_spikes(command)` — `(4, M)` in/out; zeros the leading freeze,
  median-filters along time. Time is the **last** axis. Already applied in the
  published `DREGON-frames`, so never re-apply it to those.

`mixing.py` provides the pool-side helpers:
- `resolve_motor_tracks(tf)` → `(detect_key, rps_key, needs_cleaning)`
- `find_inflight_window(tf, motor_key, min_motor_rps, clean=…)` → `(t0, t1)`
- `extract_multichannel_noise_chunk` / `extract_non_overlapping_multichannel_chunks`

---


## Publishing datasets to dload

Datasets are managed by `dload` (PyPI `dload-ml`); the R2 remote (bucket
`ml-data-new`) lives in `dload.toml`, version pins in `dload.lock` (repo
root).

**Raw trees** are committed once with the CLI — `dload commit NAME --from
data/NAME` (sample key = file relpath minus extension, field = the extension;
caveat: the CLI does **not** skip hidden files — `drone_audio` needed a custom
walker) — and then referenced by a `sources.REGISTRY` entry's `raw_dataset`.

**Everything else is a derivation.** `python scripts/derive.py derive <NAME>`
runs the spec's pipeline once and commits it as a memoized, content-addressed
version, then pins it. Three manifest layouts exist, and consumers
(`streams.ensure_local` / `DloadFrameDataset` decode dispatch) distinguish them
via `meta["layout"]`:

| Layout | Shape | Produced by |
|--------|-------|-------------|
| `sample-dir-v1` | one `sample_NNNNN/` dir per sample + a `_meta` bookkeeping sample carrying the split `metadata.json`; `meta["fields"]` maps field → filename | `dregon_lm`, `dn_lm` generators |
| `tdframe-v1` | one rich `td.Frame` per recording/clip via the generic Frame codec (`streams.frame_to_sample`) | `source_frames`, `frame_subset`, `beatvk_valid`, `se_valid` generators |
| `raw-files` / `pcm16-mono-v1` | byte-exact file passthrough / mono int16 PCM | `raw_subset`, `pcm16_mono` generators |

After any commit: `dload pin NAME && git add dload.lock`, then commit+push.

**`adopt_only` specs** are historical uploads whose bytes predate the spec.
Re-deriving would push a near-duplicate copy (the mixing RNG is not byte-stable
across machines), so `derive.py adopt <NAME>` instead points the derivation ref
at the existing `dload.lock` pin (dry-run by default; `--commit` to write).


### Catalog (pinned in `dload.lock` — 43 datasets)

- **Raw sources** (7, CLI convention, from `data/`): `DREGON`, `librispeech`,
  `drone_audio`, `music`, `new-drone-noises`, `recording_with_motor_speed`,
  `zenodo_drone_noises`.
- **Derived DREGON-LM** (16, sample-dir convention, per split):
  `DREGON-LM-{train,valid}`, `DREGON-LM-V2-{train,valid}`,
  `DREGON-LM-V3-{train,valid}`, `DREGON-LM-V4-{train,valid}`,
  `DREGON-LM-V4-michaels-{train,valid}`, `DREGON-LM-V4-michaels-valid-full`,
  `DREGON-LM-test-{train,valid}`,
  `DREGON-LM-rps_{eval_long,eval_specific,train_specific}_samples`.
- **DN-LM** (2, sample-dir; dload *derived datasets* — `derivations.py`):
  `DN-LM-{train,valid}` (6480/720, drone-only noise; no `rps` field).
- **Rich frames** (2, `tdframe-v1`): `DREGON-frames`, `michaels-frames`.
- **External harmonic-noise datasets** (10, `tdframe-v1`; registry
  `sources/`, driver `scripts/derive.py`, see
  `docs/external-datasets-plan.md` + [[external-harmonic-datasets]] memory):
  `MIMII` (54057; industrial fan/pump/slider/valve, 8-ch 16 kHz 10 s, 3 SNR
  tiers), `MIMII-DG` (17999; fan/gearbox/bearing/slider/valve mono, domain-shift
  sections), `drone-detection-samples` (180320; mono 16 kHz binary
  drone/no-drone), `DroneAudioSet` (2313; 2 quads × 2 throttles × 3 rooms, 8-ch,
  drone-only/source-only/mixed subsets), `AeroSonicDB` (1895; aircraft flyover +
  rich aircraft/engine/prop meta), `SPCUP19-egonoise` (278; 10 heterogeneous
  drone-team ego-noise rigs, 1–16 ch, mic geometry in meta where exposed),
  `HornBase` (1080; horn/not-horn — tonal, not rotating-source), `HUSTmotor`
  (24; 6 health states × 4 speeds, acoustic + X/Y/Z vibration),
  `KAIST-rotating-acoustic` (5; sound-pressure at 3010 RPM), `AVQ` (12;
  audio-visual quadrotor — onboard 8-ch array, 44.1 kHz, rotor ego-noise + a
  moving speech source; labeled seqs carry `angle_vad` DOA/VAD + `mic_pos`;
  builder `build_avq`, http+extract). Every recording Frame carries
  `system`/`observation`/`operating`/`label` meta (make/model, how observed —
  onboard vs flyover — SNR, condition). Harmonicity measured separately
  (`harmonicity.py`, analysis stage).
- **Byte-exact raw companions** (1, raw-files convention): `AVQ-raw` (26; the
  AVQ videos + `cameraParams.mat` + `.docx` docs + raw mic_pos/angle_vad/
  av_calibration mats — everything except the per-channel `MONO-*.wav`, which is
  the audio in `AVQ`). Generator: `derivations` `raw_subset`.
- **Purpose-built derived subsets** (2, `tdframe-v1`): `AVQ-egonoise` (5; the
  pure rotor ego-noise sequences of `AVQ` — `S1_seq1`/`S1_seq2`/`S1_seq3`/
  `S2_seq1`/`S2_seq2`, the only AVQ recordings *without* an `angle_vad` entry
  and therefore without the speech source — **channel 0 only, 16 kHz mono**,
  705 s in one 43 MiB shard, AVQ's per-recording `meta` + provenance kept).
  Generator: `derivations` `frame_subset`. It exists so the F2 pools
  (`conf/online_mix/se_avq_survey.yaml`, `se_survey_alldrone.yaml`,
  `se_survey_allharmonic.yaml`, `derivations.SE_CATEGORY_NOISE`'s `avq_ego`
  category) can be a plain `audio_pool` with **no `include_keys` and no
  `channel`**: manifests carry no key list, so key filtering over `AVQ` had to
  download all 11 shards (~4 GiB) to find those 5 recordings, and one AVQ shard
  is a 352 MB **multipart** object that s3transfer's ETag validation rejects on
  some boto3 builds (Kaggle: `S3DownloadFailedError ... did not match expected
  ETag`) — a false alarm (the object's sha256 matches its content address), but
  a fatal one on those backends. `AVQ`/`AVQ-raw` are unchanged.
  `AVQ-egonoise-vkrps` (7; `AVQ-egonoise` joined with the blind-VK RPS
  **pseudo-labels** of `scripts/vk_pseudolabel.py` @ fa5053fc — one Frame per
  contiguous accepted segment, recordings split at NaN/refused spans (a frame
  is accepted iff all 4 rotor labels are finite), segments >= 10 s kept:
  ~617 s total, one 38 MiB shard. Each Frame: mono 16 kHz `audio` + `rps`
  `(rotor, time)` StampIndex Series on the 0.032 s grid (michaels-frames
  events convention, no-cleaning `_resolve_motor_tracks` path) + provenance
  meta (annotator commit, `refuse_conf`, per-segment mean VK confidence).
  Spec `AVQ-egonoise-vkrps` (adopt-only — a GPU annotator sits in the loop); consumed as `kind: frames` in
  `conf/online_mix/beatvk_avq_dload.yaml` (beat-VK R2 arm). Labels are
  cruise-only (66–117 rev/s), not telemetry — treat as pseudo-ground-truth.
- **Fixed SE validation sets** (3, `tdframe-v1`, `{mixture,target,meta}` per
  clip; builder `derivations.generate_se_valid`): `SE-valid-drone`,
  `SE-valid-harmonic` (F1), `SE-valid-avq-survey` (F2, 250 clips).

Consumption paths: `DloadFrameDataset` / `dload:NAME[@VER][/subpath]` URIs /
`frames:NAME` specs — see `streams.py`'s module docstring and
`docs/data-and-artifacts.md` (end-to-end flow, cache env vars, measured
streaming numbers).

---

## Gotchas

- **Datasets are gitignored** — create locally, `dload pull <name>`, or stream/reference via `data_processing.streams` (`conf/data/*_stream.yaml`, `dload:` URIs) before training.
- **`michaels_dir` in `conf/data/noise_rps_dregon_michaels*.yaml` is stale**: those configs set `michaels_dir: data/new-drone-noises`, but `load_michaels_timeframes` hardcodes `recording_with_motor_speed/`-relative paths — the value is effectively ignored; don't copy it into new configs (behavior intentionally left unchanged, flagged here).
- **`new-drone-noises` coverage**: 103 of its 108 recordings have **no alignment constants** — only the aligned ones (FLY124/FLY125 via `MICHAELS_FILES`) are in `michaels-frames`; the rest exist raw-only in the `new-drone-noises` dload dataset.
- **`motors_command` trailing freeze**: the last 45–1577 samples are identical
  (logger stopped).  `_find_inflight_window` strips this when using `motors_measured`;
  when only command is available, the end trim is effectively 0 s.  **Never use
  raw command tail samples as ground-truth RPS.**
- **Motor/IMU/source telemetry is time-last `(…, M)`**, carried as a `tdseries`
  `StampIndex`-backed Series (values `.data`, timestamps `.tindex.abs_stamps`) —
  see `docs/refactor-unified-framework.md` § "tdseries migration guide" (the old
  `EventSeries`/`src/utils/data` API was deleted). `load_timeframe` stores
  motor/imu/source telemetry as `(4|3, M)`.  Older code used `(M, 4)` + `.T`;
  if you see `(M, 0)` after slicing or a `.T` on motor values, it predates the
  convention fix (June 2026).
- **`load_timeframe(target_sr=…)` resampling**: `librosa.resample` must receive
  the `(n_ch, N)` array with `axis=-1`.  The wrong axis (resampling the 8-element
  channel dimension) loops over ~3M tiny signals and hangs for minutes.  Always
  use `res_type="soxr_hq"` (`resampy`/kaiser not installed).
- **`--min_motor_rps`** defaults to 0 (backward-compat).  Always pass `30.0` for
  new datasets to exclude takeoff and any visible landing.
- **`--real_valid` valid set has no `vocals.wav`** — only `mixture.wav` and
  `rps.npy`.  Do not run `eval.py`'s speech-enhancement metrics on it; it is
  for RPS prediction evaluation only.
- **Valid clip count ceiling**: with `--real_valid --valid_duration 8.0`, only
  ~15 non-overlapping clips exist across the 2 default recordings.  Larger
  `--num_valid` will overlap; this is fine for RPS eval but be aware.
- **Online-mix leakage/source split**: for V4-Michaels online RPS training, use
  original LibriSpeech under `data/librispeech/LibriSpeech/train-clean-100`, not
  generated `datasets/.../train/**/vocals.wav`; exclude DREGON
  `free-flight_nosource_room1`; use Michael's `FLY125` for training and reserve
  `FLY124` for validation.
- **`source_white_noise_prob` vs `white_noise_prob`**: the former replaces speech
  with white noise as the *target source*; the latter adds WN *on top of* speech.
  For diversity training use `--source_white_noise_prob 0.2–0.4`.
