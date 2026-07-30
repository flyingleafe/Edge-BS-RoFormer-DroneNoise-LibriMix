# Refactor: one dload-pipeline data layer

Branch `refactor/data-pipelines`. Status: in progress.

## Problem

The data layer grew three parallel generations of machinery that duplicate each
other:

1. **Bespoke per-dataset loaders** — `dregon.py` (656 LOC) and `michaels.py`
   (232 LOC) get privileged loader modules with their own downloaders, geometry
   getters, and telemetry fixups, while the 10 other external datasets share the
   uniform `external_datasets.py` registry (download spec + builder →
   `tdframe-v1`). `external_recordings.py` (223 LOC) is dead code.
2. **Bespoke publisher scripts** — `publish_frame_datasets.py` (DREGON/michaels),
   `publish_external_datasets.py` (10 externals), `publish_avq_{raw,egonoise,
   vkrps}.py`, `publish_beatvk_valid.py`, `build_se_valid.py` — six drivers all
   doing "iterate samples → `repo.commit` with a layout meta".
3. **Bespoke mixed-dataset CLIs** — `create_dregon_librimix.py` (1777 LOC, 30
   flags), `create_dregon_librimix_v3.py`, `create_dataset.py`, `dataset.py`
   (ZFTurbo legacy) — duplicating the same audio helpers (`load_audio`,
   `adjust_length`, `calculate_snr`, `mix_at_snr`) in three files, while
   `derivations.py` already re-declares the same recipes as dload pipelines and
   imports the CLI cores through a `sys.path` shim.

Consumption is already uniform (`streams.DloadFrameDataset`, `dload:` URIs,
`frames:` specs, online-mix `kind: frames`); creation is not.

## Target architecture

Every dataset is defined **once**, in exactly one of two registries, and
materialized **exclusively** through dload:

```
external origin (zenodo / hf / mendeley / gdrive / http / local-only)
   │  sources registry entry: fetch spec + builder(raw_dir) → td.Frame stream
   ▼
raw dload dataset            (only for project-local raws: DREGON,
                              new-drone-noises, librispeech, …)
   │  derivations SPECS: generate_source_frames / generate_*_lm_split /
   │  generate_frame_subset / generate_se_valid / … (dload pipelines,
   ▼  frozen JSON specs, recipe_version, parent pins in the fingerprint)
derived dload datasets       (*-frames, DREGON-LM-*, DN-LM-*, SE-valid-*,
                              AVQ-egonoise*, …)
   │  streams.py (DloadFrameDataset / dload: URIs / frames: specs)
   ▼
training & eval
```

- `src/data_processing/sources/` — **the** external-dataset registry. One
  module per dataset (DREGON and michaels included — no preferential
  treatment), each exporting a uniform `SourceDataset` entry: `DownloadSpec`
  (or `None` for local-only raws already committed to dload) + `builder`.
- `src/data_processing/derivations.py` — **every** derived dataset as a frozen
  pipeline spec. Historical pins are adopted in place; fresh work derives.
- `src/data_processing/mixing.py` — the pure per-sample mixing cores (hoisted
  out of the deleted CLIs; torch-free), shared by the derivation generators.
- `scripts/derive.py` — the single driver: `list` / `derive` / `adopt` /
  `publish-source`.

## Deleted

`scripts/{create_dregon_librimix,create_dregon_librimix_v3,create_dataset,
dataset,build_se_valid,publish_frame_datasets,publish_external_datasets,
publish_avq_raw,publish_avq_egonoise,publish_avq_vkrps,publish_beatvk_valid}.py`;
`src/data_processing/{dregon,michaels,external_datasets,
external_recordings}.py`; all `.ipynb_checkpoints/`.

Dataset *recreation* after this refactor: add/edit a spec, then
`python scripts/derive.py derive <NAME>`. Historical recipes remain in git
history and in the adopted specs' `note` fields.

## Consumer migrations

- online-mix noise sources `kind: dregon` / `kind: michaels` (local-dir
  loaders) are removed; every real-recording pool is `kind: frames` over a
  published frames dataset (fixes are baked in at derivation time).
  `conf/online_mix/*.yaml` migrated accordingly.
- `noise_rps_dataset`, `frame_datasets`, `generated_noise`, GP experiments,
  localization: geometry/loaders come from the sources registry or published
  frames — never from bespoke loader modules.

## Online mixing: pipelines, not pool classes

`online_mixing.py` re-implemented sampling imperatively (TimeFrameNoisePool,
MixedNoisePool, AudioFileSourcePool, DloadAudioPool — weighted record choice,
shard LRU, packed caches, redraw loops). All of that is dload combinator
composition:

- duration-weighted record choice → `dload.choice(per-record window streams,
  weights=durations, seed=...)`;
- random window cut → `random_stream(seed).map(cut_window)` (one uniform per
  chunk);
- speech files → `ds.samples().shuffle(buf, seed).repeat().map(decode+cut)`
  with the cut drawn from `dload.seeded(key, "window")` (the canonical
  per-key randomness idiom); C independent channels = `.window(C)` + stack;
- the bespoke `packed_int16` speech cache → a **derived dataset**
  (`librispeech-pcm16`, `Repository.derive` — memoized preprocessing is what
  derive IS); decode dispatch (flac vs pcm16) by manifest layout, one code path;
- `audio_pool` holdouts → shard-subset manifests (`dload.Dataset(repo,
  manifest_subset)`), key filters → `.filter(...)`, non-audio redraw →
  `.filter(...)`;
- augmentations firing → per-sample-id RNG (`make_rng`) kept exactly (the
  check_stream control-stream methodology depends on draw-count stability),
  driven by an explicit id stream (`from_iterable(count, shard=True)` —
  IterableSourceNode striping reproduces the old `worker_id + k*num_workers`
  global-id assignment exactly);
- generated/GP/static-comb noise → `random_stream(seed).map(synthesize)`;
  the neural generator's CUDA producer/ring-buffer is the one genuinely
  stateful resource and stays behind one factory function.

The train stream is one infinite `Pipeline` consumed through
`dload.torch.as_iterable_dataset`; `OnlineMixFrameDataset.from_yaml` keeps its
Hydra-facing signature (configs unchanged in shape) but is a thin wrapper over
`build_online_mix_pipeline(cfg)`. `flatten_channels` becomes a `flat_map`,
`rps_corruption` a `map` reading the sample id from the item (no lockstep
id recomputation).

## Verification gates

- `pytest tests/data_processing -q` green (baseline: 169 passed).
- `python scripts/check_experiment_configs.py` — all experiments compose
  (baseline: 150 OK / 0 FAIL).
- `python scripts/check_stream.py --experiment <one online-mix experiment>`
  unchanged stream behaviour for a migrated policy.
- Line-count delta measured per area (reported in the PR).
