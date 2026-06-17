# Plan: Online-Mixing Dataloader for RPS Prediction

## Why

The current `DREGONRPSDataset` reads precomputed 1-second mixtures from disk
(`mixture.wav` + `rps.npy` per sample). This means:

1. **Fixed dataset** — the 9000 training clips are frozen; the model sees the
   same SNR distribution, same slicing locations, same speech utterances every
   epoch.
2. **No augmentations** — no random gain, EQ, or phase perturbations that could
   improve generalisation.
3. **Storage bloat** — 9000 samples × 8ch × 1s × 16kHz × 4 bytes ≈ 4.6 GB on
   disk for a single dataset variant.
4. **No rapid iteration** — re-creating a variant with different SNR range or
   duration requires re-running `create_dregon_librimix.py` (~10 min).

Goal: an **online-mixing dataloader** that reads source recordings (drone noise
with RPS telemetry + LibriSpeech speech), mixes them on the fly with random
parameters, and streams batches to the GPU at ≥11 batch/s (matching the current
`simple_conv_v2_gru96` training throughput on H100).

---

## Requirements

### Throughput (hard constraints from observed training)

| Metric | Value | Source |
|--------|-------|--------|
| Batches/s (steady-state) | **≥11** | training_logs.txt, epoch 2+ avg |
| Effective clips/s per batch | **256** (32 × 8ch, flattened) | `_flatten_channels` in train loop |
| Per-batch data to GPU | **16.4 MB** (256 × 64 KB clips) | 1s @ 16 kHz × float32 |
| Target dataset `__len__` | **≥9000** (matches current epoch size) | training_logs.txt |
| Workers | 4 (keep, can bump to 8) | current `train_rps_predictor.py` |

### Functional requirements

1. **Random noise segment** — slice from any DREGON in_flight_noise or Michael's
   recording at a random position, including its RPS telemetry.
2. **Random speech segment** — slice from any LibriSpeech utterance.
3. **SNR mixing** — mix speech + noise at random SNR per sample (and optionally
   per channel), range configurable (default −30…0 dB).
4. **RPS target** — output RPS upsampled to the STFT frame grid (same
   `F.interpolate` as today), ready for PIT-MSE loss.
5. **Augmentation pipeline** — configurable sequence of random transforms
   applied before mixing and/or to the mixture.
6. **Multichannel support** — 8-channel noise; speech can be shared across
   channels or independent per channel.
7. **Interface compatible** with the existing training loop — `DREGONRPSDataset`
   currently yields `(audio, rps)` where `audio.shape = (C, T)` or `(T,)` and
   `rps.shape = (4, n_frames)`.
8. **Deterministic mode** — fixed seed per worker for reproducible runs.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      DataLoader (4-8 workers)                     │
│                                                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │  Worker 0  │  │  Worker 1  │  │    ...     │  │  Worker N  │  │
│  │            │  │            │  │            │  │            │  │
│  │ NoiseCache │  │ NoiseCache │  │  NoiseCache│  │ NoiseCache │  │
│  │ (shared)   │  │ (shared)   │  │  (shared)  │  │ (shared)   │  │
│  │            │  │            │  │            │  │            │  │
│  │ SpeechPool │  │ SpeechPool │  │  SpeechPool│  │ SpeechPool │  │
│  │ (mmap)     │  │ (mmap)     │  │  (mmap)    │  │ (mmap)     │  │
│  │            │  │            │  │            │  │            │  │
│  │ Mixer +    │  │ Mixer +    │  │  Mixer +   │  │ Mixer +    │  │
│  │ Augmenter  │  │ Augmenter  │  │  Augmenter │  │ Augmenter  │  │
│  └────┬───────┘  └────┬───────┘  └────┬───────┘  └────┬───────┘  │
│       │               │               │               │           │
│       └───────┬───────┴───────┬───────┴───────┬───────┘           │
│               │         prefetch queue        │                   │
│               │       (collated batches)      │                   │
├───────────────┴───────────────────────────────────┬───────────────┤
│               Main Process                        │               │
│         `_flatten_channels` → model               │               │
│         RPS targets → PIT-MSE loss                │               │
└───────────────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

### Component 1: NoiseCache

Loads all drone-noise recordings into RAM at init, provides random slice access.

**Storage (at 16 kHz, float32):**

| Source | Total duration | Channels | Size |
|--------|---------------|----------|------|
| DREGON in_flight_noise (6 recordings) | ~266 s in-flight | 8 | ~136 MB |
| Michael's recordings (est.) | ~200 s | 1-2 | ~13 MB |
| **Total** | | | **< 150 MB** |

Fits easily in RAM. With `fork` multiprocessing, workers share pages via COW.

```python
class NoiseCache:
    """In-memory cache of all noise recordings with RPS telemetry."""

    recordings: list[NoiseRecording]  # each has audio (C, T) + rps_raw (4, M) + timestamps

    def sample_segment(
        self, duration: float, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Returns (audio [C, T], rps_stft [4, F], origin_id).

        Picks a random recording weighted by duration, slices at random offset.
        """
        ...
```

**Key details:**
- Recordings are resampled to 16 kHz once at cache construction (via
  `dregon.load_timeframe(target_sr=16000)` which uses `soxr_hq`).
- RPS telemetry stored as raw (Hz, at motor timestamps); the STFT-grid
  interpolation happens inside `__getitem__` (same `F.interpolate` as now).
- Weights for recording selection are proportional to in-flight duration
  (excluding takeoff/landing via `min_motor_rps=30`).
- Michael's recordings (mono/stereo) are broadcast to 8 channels to match
  DREGON's format, or kept at native channel count (handled by
  `_flatten_channels` downstream).

### Component 2: SpeechPool

Provides random-access to 1-second speech segments from LibriSpeech.

**Challenge:** LibriSpeech train-clean-100 is ~5.4 GB at 16 kHz — too large to
cache fully in RAM. Need a structure that supports O(1) random seek + read.

**Approach: Pre-concatenated memory-mapped array**

Pre-processing step (one-time, ~5 minutes):

```bash
# Concatenate all utterances into one flat file + build offset index
python scripts/prepare_speech_pool.py \
  --source data/LibriSpeech/train-clean-100 \
  --output data/speech_pool/speech.bin \
  --index  data/speech_pool/index.npy \
  --sr 16000
```

The index is a 2D array of shape `(N_utterances, 3)` = `[file_offset, num_samples, speaker_id]`.
At runtime, the SpeechPool memory-maps `speech.bin`:

```python
class SpeechPool:
    """Memory-mapped random access to LibriSpeech utterances."""

    data: np.memmap  # (total_samples,) float32
    index: np.ndarray  # (N, 3) — [offset, length, speaker_id]

    def sample_segment(
        self, duration: float, rng: np.random.Generator
    ) -> torch.Tensor:
        """Returns random 1-second segment from a random utterance.
        If the utterance is shorter than `duration`, wrap or pad.
        """
        ...
```

**Fallback (no pre-processing):** If the speech pool hasn't been built, fall
back to `torchaudio.load` from the LibriSpeech directory tree, with a small
in-memory LRU cache (capacity ~500 utterances ≈ 80 MB). This is slower but
zero-setup — fine for initial experimentation.

**Alternative considered — utterance-level shuffle buffer:** A large ring buffer
(e.g. 8192 segments) that background threads continuously refill by loading
random utterances and slicing them. Rejected because the memory-mapped array is
simpler, faster (no background threads needed), and provides true random access.
The shuffle buffer adds complexity with no throughput benefit over mmap.

### Component 3: Augmentation Pipeline

A configurable sequence of random transforms applied to speech, noise, or the
mixture. Implemented as composable `nn.Module`-like stages (or plain functions)
that operate on `(C, T)` tensors + metadata.

```
Speech ──► PreSpeechAug ──┐
                           ├──► SNR Mix ──► PostMixAug ──► (audio, rps)
Noise  ──► PreNoiseAug  ──┘       │
                                   └──► RPS Resample ──► rps_target
```

**Proposed augmentations (initial set):**

| Stage | Scope | Description | Cost |
|-------|-------|-------------|------|
| `RandomGain` | speech, noise | Scale by random factor [0.8, 1.2] | ~0 |
| `RandomEQ` | speech, noise | 2nd-order shelving/bell filter, random freq/gain | Low (biquad) |
| `RandomPhaseJitter` | noise | Sub-sample delay via fractional shift | Low (resample offset) |
| `ChannelDrop` | noise | Randomly zero out 1-2 channels | ~0 |
| `SNRMix` | mixture | Apply per-channel SNR from configurable range | ~0 |
| `RandomClip` | mixture | Soft clip at random threshold | ~0 |

**Design principle:** Augmentations are performed **in the time domain** on the
CPU, before the batch is pinned and transferred. The model computes its own STFT,
so no spec-level augmentations here. If SpecAugment is desired, it goes in the
model's forward pass.

Each augmentation is a callable with signature:
```python
def __call__(self, audio: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """audio: (C, T) or (T,). Returns same shape."""
```

The pipeline is a `nn.Sequential`-style chain. Augmentations are only applied
during training (skipped at eval via a flag).

### Component 4: OnlineMixDataset

Wraps NoiseCache + SpeechPool + Augmentation pipeline into a `torch.utils.data.Dataset`.

```python
class OnlineMixDataset(Dataset):
    """On-the-fly speech+noise mixer for RPS prediction."""

    def __init__(
        self,
        noise_cache: NoiseCache,
        speech_pool: SpeechPool,
        duration: float = 1.0,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        snr_range: tuple[float, float] = (-30, 0),
        speech_per_channel: Literal["shared", "independent"] = "independent",
        augment: AugmentationPipeline | None = None,
    ):
        ...
        self.rngs = np.random.default_rng(seed)    # per-worker seeded

    def __len__(self):
        return 9000  # virtual epoch size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self.rngs[idx % self._n_workers]     # worker-local RNG

        # 1. Sample noise segment
        noise_seg, rps_raw, _ = self.noise_cache.sample_segment(self.duration, rng)

        # 2. Sample speech segment(s)
        if self.speech_per_channel == "shared":
            speech_seg = self.speech_pool.sample_segment(self.duration, rng)
            speech_seg = speech_seg.unsqueeze(0).expand(noise_seg.shape[0], -1)
        else:
            n_ch = noise_seg.shape[0]
            segments = [self.speech_pool.sample_segment(self.duration, rng)
                        for _ in range(n_ch)]
            speech_seg = torch.stack(segments)  # (C, T)

        # 3. Pre-mix augmentations
        speech_seg = self.augment.apply("speech", speech_seg, rng)
        noise_seg  = self.augment.apply("noise", noise_seg, rng)

        # 4. SNR mixing
        snr_db = rng.uniform(*self.snr_range)  # or per-channel
        mixture = self._snr_mix(speech_seg, noise_seg, snr_db)

        # 5. Post-mix augmentations
        mixture = self.augment.apply("mixture", mixture, rng)

        # 6. RPS target (resample to STFT grid)
        n_frames = mixture.shape[-1] // self.hop_length + 1
        rps_target = F.interpolate(
            rps_raw.unsqueeze(0), size=n_frames, mode="linear", align_corners=False
        ).squeeze(0)

        return mixture, rps_target
```

#### Worker setup + RNG isolation

Each DataLoader worker gets its own `np.random.Generator` seeded from a master
seed + worker_id. The `__getitem__` uses `rng` for all stochastic choices
(recording selection, slice offset, SNR draw, augmentation params) so the
pipeline is reproducible when the worker seed is fixed.

The NoiseCache is inherited via `fork` (read-only, shared COW pages). The
SpeechPool memory-map is inherited per-worker (Linux `mmap` semantics — the
mapping is shared across fork).

---

## Throughput Analysis

### Cost breakdown per batch (32 samples, 8ch, 1s, 16 kHz)

| Operation | Time | Notes |
|-----------|------|-------|
| Noise random slice (cached) | **~0** | Already in RAM, just array slicing + indexing |
| Speech random slice (mmap) | **~5 µs** | np.memmap → slice → torch tensor |
| Speech per-channel (independent) | **~160 µs** | 8× the above, vectorised |
| Pre-noise aug (gain) | **~1 µs** | scalar multiply |
| Pre-speech aug (gain + optional EQ) | **~10 µs** | biquad filter if enabled |
| SNR mix (per channel) | **~10 µs** | `rms` → `scale = sqrt(speech_power / noise_power / 10^(snr/10))` |
| Post-mix aug (optional clip) | **~1 µs** | clamp |
| RPS resample (interpolate) | **~10 µs** | `F.interpolate` 4×33 → 4×32 |
| Collate + pin_memory | **~100 µs** | DataLoader collate + CUDA pin |
| **Total CPU work per sample** | **~200 µs** | worst case (all augs enabled) |

Per batch (32 samples): **~6.4 ms CPU work** × 4 workers in parallel → each
worker does ~1.6 ms of CPU work per batch while the GPU processes for 91 ms.

### Bottleneck check

| Resource | Required | Available | Headroom |
|----------|----------|-----------|----------|
| CPU (mixing per batch) | 1.6 ms/worker | 91 ms GPU time | **56×** |
| Memory (noise cache) | <150 MB | >500 GB (H100 node) | **trivial** |
| Disk I/O (speech mmap) | ~280 KB/batch | >2 GB/s (NVMe/GPFS) | **7000×** |
| Pin-memory transfer | 16.4 MB/batch | ~32 GB/s (PCIe Gen5) | **180×** |

**Conclusion:** The online mixer will NOT be the bottleneck. The CPU work per
batch (~6.4 ms total, parallelised over 4 workers) is negligible compared to the
GPU's 91 ms per batch. The GPU remains the bottleneck at 11 batch/s.

**Limiting case — independent speech per channel with full augmentations:**
At 352 speech segments/s (32 × 11 × 8), each requiring a 64 KB memmap read,
total I/O is 22 MB/s — far below any modern storage system's capability.

### Prefetching strategy

The existing `DataLoader(num_workers=4, pin_memory=True, prefetch_factor=2)`
(prefetch_factor=2 means 2 batches per worker = 8 batches in the queue) gives
~730 ms of buffered work. This is sufficient to absorb any transient I/O jitter.

If profiling shows CPU-side latency spikes (e.g. from the biquad EQ), increase
to `num_workers=8` or increase `prefetch_factor`.

---

## Integration with Existing Training Loop

### Minimal changes to `train_rps_predictor.py`

Replace:

```python
train_ds = DREGONRPSDataset(
    os.path.join(args.data_root, "train"), n_fft, hop, salience_fn=salience_fn
)
```

With:

```python
noise_cache = NoiseCache.build(
    dregon_dir=os.path.join(args.data_root, ".."),
    michaels_dir="data/new-drone-noises",
    sample_rate=16000,
    min_motor_rps=30.0,
)
speech_pool = SpeechPool(
    mmap_path="data/speech_pool/speech.bin",
    index_path="data/speech_pool/index.npy",
    librispeech_dir="data/LibriSpeech/train-clean-100",  # fallback
)
train_ds = OnlineMixDataset(
    noise_cache=noise_cache,
    speech_pool=speech_pool,
    duration=1.0,
    sample_rate=16000,
    n_fft=n_fft,
    hop_length=hop_length,
    snr_range=(args.snr_min, args.snr_max),
    speech_per_channel="independent",
    augment=get_train_augmentations(),
)
```

The rest of the training loop (optimizer, scaler, evaluation) stays identical.
The `_flatten_channels` call works unchanged because the output format
`(C, T)`, `(4, F)` is preserved.

### Config knobs (CLI flags to add)

```
--online_mix              (flag, replaces --data_root)
--snr_min -30
--snr_max 0
--duration 1.0
--speech_per_channel {shared,independent}
--speech_pool_mmap data/speech_pool/speech.bin
--noise_sources "dregon-split:in_flight_noise,michaels:all"
--augmentations "gain,eq,clip"
```

When `--online_mix` is not set, the existing `DREGONRPSDataset` path is used
(backward compatible).

---

## Pre-processing: speech_pool

### `scripts/prepare_speech_pool.py`

**Input:** `data/LibriSpeech/train-clean-100/` directory tree.

**Output:**
- `data/speech_pool/speech.bin` — flat float32 array of concatenated utterances,
  all resampled to 16 kHz.
- `data/speech_pool/index.npy` — `(N_utterances, 3)` int64 array:
  `[byte_offset, num_samples, speaker_id]`.

**Algorithm:**
1. `find` all `.flac` files under `train-clean-100`
2. For each file: `torchaudio.load` → resample to 16 kHz → convert to mono
   → append to `speech.bin` → record `(byte_offset, num_samples, speaker_id)`
3. Write `index.npy` as `uint64` array.

**Estimated output size:** ~5.4 GB (train-clean-100 at 16 kHz mono float32).
This is large but fine for a single file — GPFS handles it. For environments
with limited disk, keep the file on a scratch volume.

**Caveat:** The mmap file won't work across machines without copying the `.bin`
file. For multi-host training (distributed), each node needs its own copy or
shared filesystem access (both satisfied by GPFS on the HPC cluster).

---

## Implementation Roadmap

### Phase 1 — Minimal viable (1-2 days)

1. `NoiseCache` loading DREGON recordings via `dregon.load_timeframe(target_sr=16000)`
   - Skip Michael's for now (add in Phase 2)
   - Store audio as `(C, T)` float32 tensors + raw RPS + motor timestamps
2. `SpeechPool` with the fallback path (directory-tree + LRU cache, no mmap)
3. `OnlineMixDataset` with basic SNR mixing only (no augmentations)
4. Wire into training loop via `--online_mix` flag
5. Verify throughput matches 11 batch/s on H100

### Phase 2 — Performance (1 day)

1. `scripts/prepare_speech_pool.py` — build the memory-mapped speech pool
2. Switch `SpeechPool` default to mmap, keep directory fallback
3. Add `num_workers=8` tuning if needed
4. Profile and verify zero GPU starvation

### Phase 3 — Augmentations (2-3 days, optional)

1. Implement augmentation stages (gain, EQ, phase jitter, channel drop, clip)
2. Add `--augmentations` flag
3. Ablation study: which augmentations improve RPS prediction?

### Phase 4 — Evaluation compatibility (1 day, optional)

1. Support `--real_valid` mode (no mixing for valid set — use raw
   `in_flight_source` recordings as today)
2. Ensure eval path determinism (fixed seed, no augmentations)

---

## Open Questions

1. **Michael's recording channel count** — Michael's recordings are mono.
   Should we broadcast to 8ch in NoiseCache, or keep native channel count
   and let `_flatten_channels` handle variable C? Answer: keep native
   channel count; the variable-C path is simpler and `_flatten_channels`
   already handles it.

2. **SNR per channel vs per sample** — Independent SNR per channel means
   8 different SNR draws per sample. This multiplies the diversity but
   changes the statistics significantly. Keep as configurable default
   ("per-sample" for now, "per-channel" as an option).

3. **RPS target when using Michael's recordings** — Michael's motor telemetry
   is ~29 Hz vs DREGON's ~929 Hz. The `F.interpolate` resampling from raw
   RPS→STFT grid already handles this, so no special treatment needed. But
   the lower-rate telemetry may alias fast RPS changes — worth monitoring.

4. **Speech duration shorter than target** — Some LibriSpeech utterances are
   <1 second. Options: (a) skip them, (b) zero-pad, (c) loop-wrap. The
   simplest is (a): filter utterances shorter than `duration` in the
   SpeechPool index. This removes <1% of train-clean-100.

5. **Determinism with online mixing** — Since every epoch draws different
   random slices, the valid set metrics will fluctuate. For reliable
   early-stopping comparisons, either (a) fix the valid set as precomputed
   clips (current approach), or (b) use a fixed seed for the valid-set
   DataLoader that yields the same sequence every epoch. Option (b) is
   preferred for consistency: seed the valid RNG with a constant and don't
   advance it between epochs.

6. **Warmup cost** — The first epoch will pay the Linux page-cache warmup as
   the speech pool mmap is faulted in. This is a one-time ~5s penalty per
   training run (amortised over 50 epochs).
