# Plan: Online-Mixing Dataloader for RPS Prediction

Status: reviewed design draft, ready for discussion before implementation.

## Task understanding

Build a fast online-mixing training stream for RPS prediction. Instead of
reading precomputed `mixture.wav + rps.npy` samples, the dataloader should draw
random aligned rotor-noise/RPS segments, draw random speech or other source
segments, mix them on the fly, and return the same training-loop interface as
`DREGONRPSDataset`: `(audio, rps_target)` where `audio` is `(C, T)` or `(T,)`
and `rps_target` is `(4, T_stft)`.

A major design requirement is future **curriculum/scheduling** of mixture
parameters: source lists, source weights, speech/no-speech probability, SNR
range/distribution, and augmentation probabilities should be functions of
training progress. This affects the core dataset design: sample generation must
be driven by a deterministic global sample id, not by “whatever RNG state this
worker currently has”.

---

## Why

The current `DREGONRPSDataset` reads precomputed 1-second mixtures from disk
(`mixture.wav` + `rps.npy` per sample). This means:

1. **Fixed dataset** — the model sees the same SNR distribution, same slice
   locations, and same speech utterances every epoch.
2. **No cheap curriculum** — changing SNR/source/augmentation distributions
   requires regenerating the dataset or adding ad-hoc logic outside the data
   pipeline.
3. **Storage bloat** — multichannel precomputed variants duplicate the same
   noise/speech material many times.
4. **Slow iteration** — re-creating a variant with different SNR range,
   duration, or source mix requires re-running `create_dregon_librimix.py`.

Goal: an **online-mixing dataloader** that streams random mixtures at or above
current training throughput (≥11 batch/s for `simple_conv_v2_gru96` on H100),
while making curricula/schedules first-class.

---

## Hard requirements / acceptance criteria

### Throughput

| Metric | Target |
|--------|--------|
| Steady-state batches/s | **≥11** with current RPS predictor workload |
| Batch shape | current default: batch 32, 8ch flattened downstream → 256 clips/batch |
| Clip format | 1 s @ 16 kHz, float32 by the time it reaches the model |
| Workers | must work with 4 workers; may tune to 8 |
| GPU starvation | no sustained dataloader wait in profiler |

Acceptance: use `utils.dataloader_benchmark.benchmark_dataloader(...)` to
iterate at least 1000 batches (or a long enough timed window) through the
DataLoader with the same `batch_size`, `num_workers`, `pin_memory`, and
`prefetch_factor` as training, and separately run one short H100 training job to
confirm no throughput regression.

### Functional

1. **Noise segment** — sample a random aligned segment from configured rotating
   noise sources (initially DREGON `in_flight_noise`; then Michael's; later any
   `TimeFrame` with `audio` + RPS-like track).
2. **RPS target** — interpolate telemetry onto the model's STFT frame times,
   returning `(4, T_stft)` in Hz.
3. **Optional source segment** — sample speech/white-noise/other source audio;
   allow `source_prob=0` for noise-only phases.
4. **SNR mixing** — if a source is present, mix at a scheduled/configurable SNR
   distribution, default uniform `[-30, 0)` dB.
5. **Multichannel** — preserve full noise channel set. Speech/source can be
   shared across channels or independently sampled per channel.
6. **Augmentations** — configurable and schedulable pre-source, pre-noise, and
   post-mixture transforms.
7. **Compatibility** — training loop should only need dataset construction
   changes; `_flatten_channels` remains valid.
8. **Determinism** — a fixed seed and fixed global sample ids produce the same
   samples independent of DataLoader worker count.
9. **Validation stability** — validation should remain fixed/reproducible; do
   not use an advancing random online validation stream for early stopping.

---

## Non-goals for first implementation

- Do not build a general experiment/config framework. Add a minimal YAML loader
  for a persisted dataset/mixer definition, and keep existing argparse training
  flow intact.
- Do not replace `create_dregon_librimix.py`; online mixing is for training
  diversity, while precomputed/real-valid datasets remain useful for fixed
  evaluation.
- Do not implement heavy augmentations before proving the basic mixer is not a
  bottleneck.
- Do not solve optimizer-step-exact scheduling in Phase 1. Sample-count-exact
  scheduling is sufficient and avoids shared-state races with DataLoader
  prefetching.

---

## Public interface rule: config in, stream out

The user-facing interface must be dead simple:

```python
cfg = OmegaConf.load("path/to/online_mix.yaml")
train_stream = OnlineMixIterableDataset.from_config(cfg)
```

or equivalently one training flag pointing at that config. There should be no
separate scripts or required manual cache-preparation steps for speech/source
pools. If a faster internal cache is configured or preferred and it does not
exist, the dataloader/pool creates it during initialization and prints a clear
status message. The most naïve implementation (read individual files every
sample) and the optimized implementation (packed memmap/cache) must have the
same public constructor and YAML schema.

---

## Data/config representation rule: reuse existing abstractions

Do not invent new data containers for sampled noise or speech segments.

- **Complex aligned data:** use existing `TimeFrame`s and `TimeSeries` tracks.
  Rotating noise with audio + RPS telemetry is a `TimeFrame`; slicing a segment
  is `tf.slice(...)`.
- **Simple unaligned audio:** use plain NumPy arrays, memmaps, or tensors at the
  function boundary. Packed speech with fixed sample rate/SNR metadata does not
  need a segment class.
- **Metadata:** prefer `TimeFrame.tags` / `global_data` for time-invariant
  aligned-recording metadata, or simple side tables for pool indexes.
- **Configs:** persist YAML and load with OmegaConf. Prefer `DictConfig` nodes,
  `OmegaConf.to_container(...)`, and small pure helper functions over custom
  config dataclasses.

---

## Key design decision: infinite `IterableDataset` with deterministic global sample ids

The online mixer should be an `IterableDataset`, not a map-style `Dataset`.
The conceptual sample space is infinite; an "epoch" is just a training-loop
cadence: consume an arbitrary number of samples, then run fixed validation.

`global_sample_id` is still the anchor for reproducibility and scheduling, but
it is generated by the stream rather than passed in as `__getitem__(idx)`.

Recommended design:

- Implement `OnlineMixIterableDataset(torch.utils.data.IterableDataset)`.
- Do not implement `__len__` or `__getitem__` for the online training stream.
- The training loop owns `samples_per_validation` (the arbitrary number of
  online samples to consume before running validation) and stops each training
  interval after that many samples or equivalent batches.
- Use `shuffle=False`; online randomness already provides data diversity.
- Each yielded item carries or is generated from an absolute `global_sample_id`.
- Derive a per-sample RNG by hashing `(base_seed, global_sample_id)`.
- Do not use persistent mutable `self.rng` for stochastic choices inside sample
  generation.

This makes schedules exact:

```python
policy = schedule(global_sample_id)
rng = make_rng(base_seed, global_sample_id)
sample = generate_sample(policy, rng)
```

With multiple workers, shard the global-id stream inside `__iter__` using
`torch.utils.data.get_worker_info()`. A simple deterministic scheme is for
worker `w` out of `W` workers to yield ids:

```python
global_sample_id = start_sample_id + w + k * W
```

Worker id is allowed to choose which ids a worker owns, but all random choices
for a sample must still be derived only from `(base_seed, global_sample_id)`.
Worker-local state may be used for lazy file handles/memmaps, but not for the
random decisions that define a sample.

Changing `num_workers` may change which worker produces a given id, but should
not change the sample associated with that id. If exact prefix order across
worker counts matters, add an ordered merge or generate finite id windows in the
main process; Phase 1 only requires deterministic samples per id.

For distributed training later, shard the global id stream by rank as well, but
keep `global_sample_id` itself absolute so schedules remain comparable across
runs.

---

## Architecture

```
DataLoader workers
  ├── NoisePool / NoiseCache      existing TimeFrame objects for aligned noise+RPS
  ├── SourcePool(s)               plain arrays/memmaps or existing audio loaders
  ├── schedule function           global_sample_id -> resolved OmegaConf node
  ├── AugmentationPipeline        schedulable transforms
  └── OnlineMixIterableDataset.__iter__()
          1. generate/shard an infinite stream of global_sample_id values
          2. evaluate schedule
          3. create deterministic per-sample RNG
          4. sample noise segment
          5. maybe sample source segment(s)
          6. apply pre-mix augmentations
          7. SNR mix or noise-only passthrough
          8. apply post-mix augmentations
          9. interpolate RPS to STFT frame grid
         10. yield `(mixture, rps_target)`
```

### Component 1: NoisePool / NoiseCache

Use the project `TimeFrame` abstraction for aligned audio + telemetry. Do **not**
invent new record/segment container datatypes for noise. A loaded recording is a
`TimeFrame` with named tracks such as `audio` and an RPS-like track; a sampled
noise segment is just `recording_tf.slice(t0, t1)`.

Initial source support:

- DREGON `in_flight_noise`, trimmed with `min_motor_rps=30.0`.
- Michael's recordings via the existing loader once integrated.
- Source specs should mirror `create_dregon_librimix.py` conventions where
  possible: `dregon-split:in_flight_noise`, `dregon-id:<id>`, `michaels:all`,
  etc.

Important details:

- Use `resolve_motor_tracks`-style logic from the existing dataset pipeline:
  prefer measured RPS where available for validity/windowing, but use cleaned
  command or generic `rps` track when that is the available aligned target.
- Preserve the invariant that time is the last axis: audio `(C, T)`, RPS
  `(4, M)`.
- Select recordings with weights proportional to available valid duration,
  unless overridden by a schedule.
- Cache noise audio/RPS in memory where practical. DREGON at 16 kHz float32 is
  small; even a few hundred seconds of 8ch audio is well under 1 GB.
- With Linux `fork`, a read-only cache built in the main process is shared by
  workers through copy-on-write. Still support lazy worker loading as a fallback
  for platforms that use `spawn`.

Pseudo-interface:

```python
class NoisePool:
    # Internally stores existing TimeFrame objects plus lightweight metadata
    # such as source group and valid sampling windows. Metadata can live in
    # TimeFrame.tags/global_data or side tables; do not introduce a new segment
    # container class.
    recordings: list[TimeFrame]

    def sample_timeframe(self, cfg, rng, duration_s: float) -> TimeFrame:
        """Return a sliced TimeFrame containing at least audio + RPS tracks."""
        ...
```

### Component 2: SourcePool(s)

`SourcePool` provides random source audio. Do **not** introduce a speech-segment
container. For simple unaligned audio, return a plain NumPy array plus any
primitive metadata needed for debugging/logging. If a future source has aligned
side information, represent it as a `TimeFrame` instead of creating a new custom
container.

Phase 1 can support LibriSpeech speech only; the schedule interface should not
assume speech is the only source.

Storage implementation is private to `SourcePool` and must not change the
public dataloader interface.

- The config names source roots/groups, not prebuilt cache artifacts.
- On construction, `SourcePool` may check whether an internal packed cache/index
  exists. If not, it should print a clear message such as
  `Creating source cache for librispeech-train-clean-100 ...` and build it.
- A naïve implementation that reads individual files every time and an advanced
  implementation backed by int16 memmaps must have the same constructor/config
  interface.
- Store packed PCM as **int16** unless there is a strong reason to store float32.
  Train-clean-100 is ~100 h:
  - float32 mono 16 kHz: ~21.5 GiB
  - int16 mono 16 kHz: ~10.7 GiB
- Convert selected slices to float32 in the worker.

If an internal packed cache is used, its index should use **sample offsets**, not
ambiguous byte offsets:

```python
# private implementation detail; shape (N, 4):
# [sample_offset, num_samples, speaker_id, source_group_id]
index: np.ndarray[np.int64]
```

The initial version may use directory-tree `torchaudio.load` with a small
per-worker LRU cache, then later switch to an internal memmap cache without any
change to training code or YAML schema.

### Component 3: Configs and schedules

Persist the dataset/mixer definition as YAML and load it with OmegaConf. Prefer
`DictConfig`/`ListConfig` nodes and small pure functions over custom config
dataclasses. A schedule maps progress to a resolved OmegaConf node (or a plain
`dict` produced by `OmegaConf.to_container`) containing scalar probabilities,
source weights, distribution specs, and augmentation parameters.

Do not introduce a custom policy datatype unless implementation experience
shows OmegaConf nodes are genuinely painful. The Phase-1 constant schedule can
be as simple as:

```python
cfg = OmegaConf.load(path)
policy_cfg = cfg.policy  # resolved constant DictConfig
```

Schedule units:

- **Default: `sample`** — exact and worker-independent.
- Optional later: `validation_interval`/`epoch` labels — syntactic sugar for
  sample ranges, not evidence of a finite dataset.
- Later: `optimizer_step` — only if truly needed; DataLoader prefetching makes
  step-exact worker-side schedules awkward and less reproducible.

Schedule support:

- **Phase 1 implements constant policies only**, loaded as OmegaConf YAML so the
  dataset definition is persisted in the repo.
- The schedule API should leave room for later piecewise/linear policies, but
  non-constant schedules are explicitly deferred.
- Future primitives: piecewise constant values, linear interpolation for numeric
  ranges/probabilities, and source-list/weight interpolation where new sources
  ramp from weight 0 to a target weight over a sample interval.

Example matching the requested curriculum:

```yaml
unit: sample
samples_per_validation: 9000
base_seed: 1234

stages:
  - until: 10000
    source_prob: 0.0              # noise only; SNR ignored
    noise_sources:
      dregon-split:in_flight_noise: 1.0
    augmentations: {}

  - until: 30000
    source_prob: {linear: [0.0, 1.0]}
    snr_db:
      uniform:
        low: -30
        high: {linear: [-25, 0]}  # expands [-30, -25) -> [-30, 0)
    noise_sources:
      dregon-split:in_flight_noise: 1.0
    speech_sources:
      librispeech-train-clean-100: 1.0
    augmentations: {}

  - until: null
    source_prob: 1.0
    snr_db: {uniform: {low: -30, high: 0}}
    noise_sources:
      dregon-split:in_flight_noise: 0.8
      michaels:all: {linear_from_previous: [0.0, 0.2, 20000]}
    speech_sources:
      librispeech-train-clean-100: 0.8
      librispeech-train-other-100: {linear_from_previous: [0.0, 0.2, 20000]}
    augmentations:
      random_gain: 1.0
      random_eq: {linear: [0.0, 0.5], over: 20000}
      channel_drop: {linear: [0.0, 0.2], over: 20000}
```

The exact YAML grammar can be simplified during implementation. In Phase 1,
`online_mix.yaml` may contain only constants; the important contract is that
`schedule(global_sample_id)` returns a fully resolved OmegaConf node or plain
mapping and can later grow non-constant stages without changing the dataset
interface.

Source roots and optional cache policy live in the same config. Cache paths are
implementation details, not separate CLI inputs:

```yaml
sources:
  noise:
    - name: dregon-in-flight
      kind: dregon
      root: data/DREGON
      split: in_flight_noise
      min_motor_rps: 30.0
  speech:
    - name: librispeech-train-clean-100
      kind: librispeech
      root: data/LibriSpeech/train-clean-100
      cache:
        mode: auto        # auto | file_lru | packed_int16 | none
        dir: .cache/online_mix_sources
```

### Component 4: AugmentationPipeline

Stages operate on `(C, T)` tensors/arrays and receive both RNG and the resolved
OmegaConf/plain mapping policy.

Initial low-cost transforms:

| Stage | Scope | Notes |
|-------|-------|-------|
| `RandomGain` | source/noise/mixture | scalar or per-channel |
| `RandomEQ` | source/noise | cheap biquad; add only after benchmark |
| `FractionalDelay` / phase jitter | noise | useful for channel robustness |
| `ChannelDrop` | noise/mixture | preserve shape, zero selected channels |
| `RandomPolarity` | source/mixture | cheap |
| `SoftClip` | mixture | optional |

Keep all Phase-1 augmentations CPU time-domain. The model owns STFT/front-end
processing, so SpecAugment-like operations belong in model/training code, not
this dataset.

### Component 5: OnlineMixIterableDataset

Pseudo-interface:

```python
class OnlineMixIterableDataset(IterableDataset):
    def __init__(
        self,
        noise_pool: NoisePool,
        source_pools: dict[str, SourcePool],
        schedule,  # callable: global_sample_id -> resolved OmegaConf/plain mapping
        duration: float = 1.0,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        start_sample_id: int = 0,
    ): ...

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        k = 0
        while True:
            global_sample_id = self.start_sample_id + worker_id + k * num_workers
            k += 1
            yield self.generate_sample(global_sample_id)

    def generate_sample(self, global_sample_id: int):
        policy = self.schedule(global_sample_id)
        rng = make_rng(self.schedule.base_seed, global_sample_id)

        noise_tf = self.noise_pool.sample_timeframe(policy, rng, self.duration)
        noise_audio = extract_audio_array(noise_tf["audio"])       # np.ndarray (C, T)
        rps_track = noise_tf["rps"]                                # existing TimeSeries

        if rng.random() < policy.source_prob:
            source = sample_source_array(policy, rng, duration=self.duration, channels=noise_audio.shape[0])
            source = apply_augment("source", source, policy, rng)
            noise_audio = apply_augment("noise", noise_audio, policy, rng)
            mixture = snr_mix(source, noise_audio, sample_snr(policy.snr_db, rng), policy)
        else:
            mixture = apply_augment("noise", noise_audio, policy, rng)

        mixture = apply_augment("mixture", mixture, policy, rng)
        rps_target = interpolate_rps_to_stft_grid(rps_track, ...)
        return torch.from_numpy(mixture), torch.from_numpy(rps_target)
```

RPS interpolation should be timestamp-based for online data, not just
shape-stretching raw RPS to `n_frames`. Shape-stretch is only safe for the old
precomputed DREGON-LM chunks where audio and `rps.npy` are coextensive by
construction.

---

## Mixing details

### SNR convention

For RPS prediction, the target is RPS, not clean speech. SNR controls how much
speech/source masks the rotating noise. Define:

```text
SNR dB = 10 log10(source_power / noise_power)
```

So `-30 dB` means the source is 30 dB below the noise. To mix:

```python
scale = sqrt(noise_power * 10 ** (snr_db / 10) / (source_power + eps))
mixture = noise + scale * source
```

This is the opposite scaling from speech-enhancement code that treats speech as
the target and scales noise to reach a desired SNR. The implementation should
name variables explicitly (`source_power`, `noise_power`) and include a unit
test for this convention.

### Per-channel policy

Defaults:

- `speech_per_channel="independent"` to match current multichannel synthetic
  training diversity.
- `snr_per_channel=False` initially, because per-channel SNR changes mixture
  statistics substantially. Add `snr_per_channel=True` as a config option.

### Clipping / normalization

After mixing, either:

- leave float32 amplitudes unconstrained if current model training already
  expects arbitrary normalized float waveforms, or
- apply a conservative peak normalization only when required.

Do not silently normalize every sample to unit peak without recording this in
metadata/config; it changes the SNR/gain curriculum.

---

## Throughput expectations and caveats

### Data sizes

| Data | Approx size |
|------|-------------|
| DREGON in-flight noise, 16 kHz, float32, 8ch, ~266 s | ~136 MB |
| Additional 8ch noise, 16 kHz, float32, ~200 s | ~102 MB |
| LibriSpeech train-clean-100 packed int16 mono | ~10.7 GiB |
| LibriSpeech train-clean-100 packed float32 mono | ~21.5 GiB |

The original estimate of ~5.4 GB for float32 train-clean-100 is too low; that
is closer to compressed FLAC scale, not raw float32 PCM.

### I/O

Worst common case: 32 samples/batch × 8 independent source channels × 11
batches/s = 2816 one-second source reads/s. At int16, each read is 32 KB, so
sequential-equivalent throughput is ~90 MB/s. The issue is not bandwidth but
random small-read behavior on GPFS/network storage. If profiling shows page
fault or I/O jitter:

1. configure the source cache directory to use local NVMe/scratch;
2. increase DataLoader prefetch;
3. add a per-worker segment cache or shuffle buffer behind the same interface;
4. fall back to shared-source-per-channel for early experiments.

Do not assume mmap is always free on network filesystems; benchmark it.

### CPU

Basic slicing + RMS scaling is cheap. EQ/filter augmentations can dominate if
implemented naively. Therefore Phase 1 should include only gain/SNR mixing and
maybe channel-drop; add EQ after benchmarking.

### Prefetching

Start with:

```python
DataLoader(
    train_stream,
    batch_size=32,
    shuffle=False,          # IterableDataset; global sample id controls randomness
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
```

For an infinite `IterableDataset`, prefer keeping the iterator alive across
validation intervals:

```python
train_iter = iter(train_loader)
for epoch_idx in range(num_validation_intervals):
    for _ in range(batches_per_validation):
        batch = next(train_iter)
        train_step(batch)
    validate_on_fixed_set()
```

`persistent_workers=True` is usually appropriate here because workers are not
restarted at arbitrary "epoch" boundaries. If exact restart/resume from a sample
id is needed, store `next_global_sample_id` in the checkpoint and reconstruct the
stream with `start_sample_id=next_global_sample_id`.

---

## Integration with existing training

Minimal new flags:

```text
--online_mix
--mix_config path/to/online_mix.yaml
--samples_per_validation 9000   # arbitrary cadence before fixed validation
--data_root datasets/DREGON-LM-RealValid   # still used for fixed valid set
```

Do not expose separate `--prepare_*`, `--speech_pool_mmap`, or
`--speech_pool_index` flags. Cache paths and cache strategy are internal details
under the config's source/cache section; training code should simply instantiate
the dataloader from config and iterate it.

When `--online_mix` is false, keep the existing `DREGONRPSDataset` path.

When `--online_mix` is true:

- Training data is an infinite `OnlineMixIterableDataset` stream.
- A training "epoch" means: consume `samples_per_validation` online samples
  (or `ceil(samples_per_validation / batch_size)` batches), then run validation.
- Validation dataset should remain fixed, preferably current real-valid or
  precomputed valid data, to keep early stopping comparable.
- `shuffle=False` for the online training loader.
- The naïve baseline computation should not consume an advancing random
  training stream if it needs to be comparable. Either compute it from a fixed
  probe sample set or from the noise pool's RPS distribution.

Salience-model path:

- Datasets should expose the same `(audio, rps_target)` format for all RPS
  predictor families.
- Salience-model training now derives BCE salience targets on the fly in the
  training loop from `batch[1]` (STFT-frame RPS). This avoids any dataset-specific
  third item and works for both precomputed and online-mixed data.
- Eval is unchanged: salience models run `predict_rps(audio)` and compare to the
  same RPS target as direct-regression models.

---

## Source cache lifecycle

There should be no separate user-visible preprocessing command for speech/source
pools. The public workflow is:

```python
cfg = OmegaConf.load("path/to/online_mix.yaml")
train_stream = OnlineMixIterableDataset.from_config(cfg)
```

Under the hood, `SourcePool.from_config(cfg.sources.speech)` can choose one of
several implementations:

1. naive file-backed reads from the configured source root;
2. file-backed reads plus a per-worker LRU cache;
3. packed PCM16 memmap plus private index.

All three must expose the same `SourcePool.sample_array(...)` behavior and use
the same YAML schema. If the selected implementation wants a packed cache and it
is missing, it creates it lazily during initialization:

```text
Creating source cache for librispeech-train-clean-100 at ...
```

Cache creation algorithm, if needed:

1. Find all `.flac`/audio files under the configured source root.
2. Load/resample to 16 kHz mono.
3. Scale/clamp to int16 PCM unless explicitly configured otherwise.
4. Append to a private flat binary/cache file.
5. Record private index entries such as
   `[sample_offset, num_samples, speaker_id, source_group_id]`.
6. Filter utterances shorter than requested duration or mark them as requiring
   wrap/pad. Prefer filtering for 1 s chunks.

For multi-machine training, each node needs access to the source root and/or the
cache directory. Prefer local scratch cache paths if shared filesystem random
access is slow, but keep that as config and internal cache policy—not a separate
preparation step.

---

## Implementation roadmap

### Phase 0 — benchmark + existing-code alignment (0.5 day)

1. Confirm source-loading helpers to reuse from `create_dregon_librimix.py` and
   `data_processing.michaels`.
2. Write a tiny benchmark for random source-pool reads from the intended
   filesystem.
3. Use PCM16/int16 for packed speech/source audio unless a benchmark shows a
   measurable degradation. Convert to float32 after slicing in workers.

### Phase 1 — minimal online mixer (1-2 days)

1. Implement `NoisePool` for DREGON `in_flight_noise` using `TimeFrame` and
   `min_motor_rps=30.0` trimming.
2. Implement `SourcePool.from_config(...)` with the same interface for naïve
   file-backed reads and internal cache-backed reads. If a configured cache is
   missing, create it automatically during initialization.
3. Implement `OnlineMixIterableDataset` with deterministic
   `global_sample_id -> rng` and a constant resolved OmegaConf/plain-mapping
   policy loaded from YAML. Do not implement non-constant schedules in Phase 1.
4. Implement source/no-source, SNR mixing, timestamp-based RPS interpolation.
5. Wire `--online_mix` into `train_rps_predictor.py` so the training loop
   consumes an arbitrary `samples_per_validation` from the infinite stream, then
   runs fixed validation.
6. Verify output shapes and train for a short run.

### Phase 2 — source-cache throughput hardening (1 day)

1. Add/optimize the internal packed int16 cache implementation behind
   `SourcePool.from_config(...)`; do not add a separate preparation script.
2. Run DataLoader-only and training-throughput benchmarks.
3. If needed, tune `num_workers`, `prefetch_factor`, local scratch cache paths,
   or per-worker segment cache.

### Phase 3 — non-constant scheduling (future, 1-2 days)

1. Extend the schedule function beyond Phase-1 constant YAML to piecewise
   constants and linear interpolation.
2. Add source-list/weight scheduling.
3. Add tests that specific sample ids resolve to expected policies.
4. Add a smoke curriculum: noise-only → narrow SNR → full SNR.

### Phase 4 — augmentations (1-2 days)

1. Add low-cost scheduled augmentations: gain, channel drop, polarity.
2. Benchmark.
3. Add EQ/fractional delay only if throughput remains safe.

### Phase 5 — broader source support (optional)

1. Add Michael's source specs.
2. Add non-speech source pools such as DREGON clean white noise/chirps.
3. Add industrial rotating-source datasets when available.

---

## Resolved design choices

1. **Dataset/mixer definition:** persisted as YAML now.
2. **Packed source audio dtype:** PCM16/int16 by default; convert to float32 in
   workers before mixing.
3. **Validation:** keep fixed real-valid/precomputed validation for stable early
   stopping and model comparison.
4. **Phase 1 scheduling:** implement only constant YAML policies, but design the
   API so non-constant schedules can be added later. The schedule is evaluated
   against `global_sample_id` from an infinite `IterableDataset` stream, not a
   finite map-style dataset index.
5. **Salience models:** no dataset special-case. Training computes salience BCE
   targets on the fly from the common `(audio, rps_target)` batch; eval unchanged.
