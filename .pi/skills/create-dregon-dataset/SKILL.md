---
name: create-dregon-dataset
description: Create any variant of the DREGON-LM dataset (mono, multichannel synthesised, or multichannel with real valid set), or any other derived dataset. Use when the user asks to (re)create a training/validation dataset from DREGON recordings and LibriSpeech.
---

# Create a derived dataset (DREGON-LM and friends)

> Bootstrap applies. Read `docs/refactor-data-pipelines.md` (the data-layer
> architecture) and `src/data_processing/AGENTS.md` (recording inventory,
> telemetry gotchas) before acting.

## The one model

There are no per-dataset creation CLIs any more. Every dataset is declared
**exactly once**:

- **raw sources** — one entry per dataset in `src/data_processing/sources/`
  (a pinned `DownloadSpec`/fetcher + a `builder(raw_dir) -> (key, td.Frame)`);
- **derived datasets** — one frozen spec in
  `src/data_processing/derivations.SPECS` (a `gen` sub-spec that is the
  derivation *fingerprint*, plus registry metadata), materialized by a
  module-level generator function;
- **per-sample mixing math** — `src/data_processing/mixing.py`, shared by the
  derivation generators and the online-mix stream.

`scripts/derive.py` is the single driver:

```bash
python scripts/derive.py list --check-remote -v   # specs, fingerprints, published refs
python scripts/derive.py derive <NAME>            # materialize + commit + pin
python scripts/derive.py adopt  <NAME> --commit   # point the derivation ref at the existing pin
```

## Prerequisites

Raw parents are dload-managed and pinned in `dload.lock`; the generators
resolve them through their pinned URIs, so a manual pull is only a prefetch:

```bash
dload pull DREGON               # ~4 GB — DREGON recordings + telemetry
dload pull librispeech          # LibriSpeech train-clean-100
```

## Creating a new variant

1. Add a `SPECS` entry. Reuse an existing `gen` block as the template — e.g.
   `DREGON-LM-V4-michaels-train` for a synthesized multichannel split, or
   `DREGON-LM-V4-michaels-valid` for a real-valid (raw clip) split. Compose the
   noise pool from **published frames datasets** (`noise_sources`, the same
   spec schema the online-mix `kind: frames` source uses), never `data/` paths.
2. `python scripts/derive.py list` — confirm the new fingerprint.
3. `python scripts/derive.py derive <NAME>` — materializes, commits, pins.
4. `git add dload.lock && git commit && git push`.

**Bump `recipe_version` on ANY behavioural change to a generator**, otherwise
the unchanged fingerprint keeps serving the stale snapshot.

## Key recipe decisions (the V4 params)

| Decision | Why |
|----------|-----|
| `min_motor_rps: 30.0` | Excludes pre-takeoff (command freeze artefact) and landing. Detection uses the frames dataset's cleaned rotor track. |
| `source_white_noise_prob: 0.3` | 30 % of train samples use white noise as the target source instead of speech — improves generalisation to non-speech sources. |
| `mode: real_valid` | Valid set = raw `in_flight_source` clips, no synthetic mixing. Drone + co-recorded source *is* the mixture; no clean reference exists. Good for RPS evaluation on real recordings. |
| `sample_duration: 8.0` (valid) | Longer clips capture more RPS variation than 1 s train clips. |
| 8-channel output | Each sample is `(T, 8)` WAV. At train time the channel axis is flattened into the batch (`flatten_channels=True`); at eval time per-channel metrics are logged separately. |

## The SE validation sets

`generate_se_valid` builds them from the *same* stream builders the training
policies compile to, with the complementary filters: valid-side noise holdouts
(`SE_CATEGORY_NOISE`) and `include`-ing exactly `SE_HELDOUT_SPEAKERS`. Silent
draws are rejected upstream — a silent noise draw zeroes both the mixture and
the clean target through the source-to-noise scaling.
`iter_se_valid_category` is public so unpublished probe categories (e.g.
`drone_seen`) can be materialized on demand by
`notebooks/generalization_lib.py`.

## Training on the dataset

```bash
python train.py experiment=<name>   # the only training entry point
```

with a `conf/data/*.yaml` entry pointing at the pinned dataset (see
`conf/data/dregon_lm_v4_michaels.yaml`).

## Evaluation on the dataset

```python
from models.rps_predictor import SimpleConv
from tasks.rps_prediction import _ModelPredictor, load_input_set, evaluate

model = SimpleConv(); model.load_state_dict(torch.load("results/.../best.ckpt"))
predictor = _ModelPredictor(model, "cuda")

result = evaluate(predictor, load_input_set("datasets/DREGON-LM-V4-michaels/valid"))
print(result.aggregate)          # {mse, rmse, mae_frame, mae_clip, r2_mean, n_samples, n_rows}
result.to_json("results/eval.json")
```

Per-sample rows have a `channel` column (0–7 multichannel, 0 mono);
`n_rows = n_samples × n_channels`.

## Critical gotchas

- **Fingerprints are the identity.** Editing a generator without bumping
  `recipe_version` silently serves the stale memoized snapshot.
- **`adopt_only` specs** are historical uploads: re-deriving would push a
  near-duplicate copy (the mixing RNG is not byte-stable across machines). Use
  `adopt`, not `derive`, for those.
- **`motors_command` trailing freeze**: the last 45–1577 raw samples of command
  are identical (logger stopped before landing). The published frames datasets
  have `clean_command_spikes` baked in; **never take raw tail samples as
  ground-truth RPS.**
- **`real_valid` clips have no `vocals`**: only `mixture` + `rps`. Do not run
  speech-enhancement eval on them.
- **`source_white_noise_prob` ≠ `white_noise_prob`**: the former replaces speech
  with WN as the target source; the latter adds WN on top of speech.
- **Telemetry is time-last `(4, M)`** — a `(M, 4)` shape or a `.T` on motor
  values predates the June 2026 convention fix.
- **librosa resample axis**: audio resampling must use `axis=-1` on the
  `(n_ch, N)` array with `res_type="soxr_hq"` (`frames.resample_audio_series`).
  The wrong axis hangs for minutes.
