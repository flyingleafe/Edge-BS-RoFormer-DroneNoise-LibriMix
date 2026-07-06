---
name: create-dregon-dataset
description: Create any variant of the DREGON-LM dataset (mono, multichannel synthesised, or multichannel with real valid set). Use when the user asks to (re)create a training/validation dataset from DREGON recordings and LibriSpeech.
---

# Create DREGON Dataset

> Bootstrap applies. Read `data_processing/AGENTS.md` before acting — it has
> the full recording inventory, telemetry gotchas, and the flag reference table.

## Prerequisites

Both raw sources must be locally present (DVC-tracked under `data/`):

```bash
dvc pull data/DREGON.dvc        # ~4 GB — DREGON recordings + telemetry
dvc pull data/librispeech.dvc   # LibriSpeech train-clean-100
```

If DVC pull fails, DREGON can be auto-downloaded from `dregon.inria.fr` by the
script itself (pass `download=True` to `load_dregon_timeframes` or let the script
handle it).

## Recommended command (multichannel + real valid)

```bash
python scripts/create_dregon_librimix.py \
  --multichannel --real_valid \
  --output_dir datasets/DREGON-LM-RealValid \
  --num_train 6000   --duration 1.0 \
  --num_valid 30     --valid_duration 8.0 \
  --min_motor_rps 30.0 \
  --source_white_noise_prob 0.3 \
  --speech_per_channel independent \
  --snr_min -30 --snr_max 0
```

**`--min_motor_rps 30.0` is mandatory for in-flight-only samples.** Without it
the first ~8–13 s of every recording (pre-takeoff + ramp-up) will be sampled.
See `data_processing/AGENTS.md` § "In-flight window" for per-recording trims.

## Key design decisions

| Decision | Why |
|----------|-----|
| `--min_motor_rps 30.0` | Excludes pre-takeoff (command freeze artefact) and landing (visible in `motors_measured` for room1 recordings). Detection uses `motors_measured` when available, falls back to `motors_command`. |
| `--source_white_noise_prob 0.3` | 30 % of train samples use white noise as the target source instead of speech — improves generalisation to non-speech sources without additional data. |
| `--real_valid` | Valid set = raw `in_flight_source` clips, no synthetic mixing. Drone + co-recorded source is the mixture; no clean reference exists. Good for RPS evaluation on real recordings. |
| `--valid_duration 8.0` | Longer clips capture more RPS variation than 1 s train clips. ~15 non-overlapping 8 s clips available across the 2 default valid recordings. |
| 8-channel output | Each sample is `(T, 8)` WAV. At train time, channel axis is flattened into batch (`_flatten_channels` in `train_rps_predictor.py`). At eval time, per-channel metrics are logged separately. |

## Mono (legacy / Paper 2 baseline)

```bash
python scripts/create_dregon_librimix.py \
  --output_dir datasets/DREGON-LM \
  --num_train 6000 --num_valid 600 \
  --duration 3.0 \
  --min_motor_rps 30.0 \
  --motor_combo_fraction 0.2
```

## Publish after creation

```bash
dvc add datasets/DREGON-LM-RealValid && dvc push
git add datasets/DREGON-LM-RealValid.dvc datasets/.gitignore
git commit -m "dataset: DREGON-LM-RealValid" && git push
```

## Training on the dataset

After dataset creation, train the RPS predictor:

```bash
python train_rps_predictor.py \
  --model simple_conv_bigru_v2 \
  --data_root datasets/DREGON-LM-RealValid \
  --batch_size 32 --epochs 200 \
  --pit_loss --smoothness_weight 0.01
```

The dataloader handles both mono `(T,)` and multichannel `(C, T)` audio
transparently via `_flatten_channels`.

## Evaluation on the dataset

```python
from models.rps_predictor import SimpleConv
from src.tasks.rps_prediction import _ModelPredictor, load_input_set, evaluate

model = SimpleConv(); model.load_state_dict(torch.load("results/.../best_simple_conv.pt"))
predictor = _ModelPredictor(model, "cuda")

result = evaluate(predictor, load_input_set("datasets/DREGON-LM-RealValid/valid"))
print(result.aggregate)          # {mse, rmse, mae_frame, mae_clip, r2_mean, n_samples, n_rows}
print(result.per_sample[:3])     # [{sample, channel, mse, mae_frame, r2, input_snr}, ...]
result.to_json("results/eval.json")
```

Per-sample rows have `channel` column (0–7 for multichannel, 0 for mono) and
optionally `input_snr_channel` (per-channel SNR from multichannel train metadata).
`n_rows = n_samples × n_channels`.

## Critical gotchas

- **`motors_command` trailing freeze**: the last 45–1577 raw samples of command
  are identical (logger stopped before landing). `_find_inflight_window` strips
  this correctly. **Never take raw tail samples as ground-truth RPS.**
- **Valid set has no `vocals.wav`**: `--real_valid` clips are raw recordings.
  Only `mixture.wav` + `rps.npy`. Do not run speech-enhancement eval on them.
- **`--source_white_noise_prob` ≠ `--white_noise_prob`**: former replaces speech
  with WN; latter adds WN on top of speech. Use `source_white_noise_prob` for
  source diversity, not `white_noise_prob`.
- **EventSeries shape is time-last `(4, M)`** — see `src/utils/data/AGENTS.md`.
  If you see motor shapes like `(M, 4)` or a `.T` on `.values`, it predates
  the June 2026 convention fix.
- **librosa resample axis**: `load_timeframe(target_sr=…)` must resample with
  `axis=-1` on the `(n_ch, N)` array. Wrong axis hangs for minutes. Already
  fixed in `dregon.py`; mention if you see a hang during loading.
