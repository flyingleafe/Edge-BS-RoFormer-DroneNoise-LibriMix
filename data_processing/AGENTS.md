# data_processing/ — Dataset Creation and RPS Processing

Contains scripts and modules for creating and processing training datasets.

## Why this directory exists

Dataset creation and preprocessing is a separate concern from model training.
Raw datasets need to be downloaded, processed, and mixed into training-ready
formats before any experiment can run.

## Files

| File | Purpose |
|------|---------|
| `dregon.py` | DREGON dataset loading — TimeFrame-native. `load_timeframe(sample)` returns a `TimeFrame` with tracks `"audio"` (UniformSeries), `"motors_measured"` / `"motors_command"` (EventSeries), etc. Tags hold scalar metadata; `global_data` holds mic/rotor positions. |
| `michaels.py` | Michael's drone-noise dataset (DJI WAVs + flight-controller CSVs in `data/new-drone-noises/`). Uses its own local `MotorData` dataclass. |
| `noise_rps_dataset.py` | `NoiseRPSDataset` — combined chunkable dataset over DREGON `in_flight_noise` + Michael's. |
| `external_recordings.py` | Loads external DJI recordings as `TimeFrame`. |
| `__init__.py` | Package init |

---

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

(Computed with `--min_motor_rps 30.0`, which is the recommended default.)

The helper `_find_inflight_window(tf, motor_key, min_motor_rps)` in
`create_dregon_librimix.py` implements this.  It prefers `motors_measured`
for detection (real spindown) and falls back to `motors_command`.  The saved
`rps.npy` always comes from `motors_command` (cleaner signal).

---

## Dataset Variants

### DN-LM (DroneNoise-LibriMix) — Paper 1

Script: `create_dataset.py` (root level).

- Sources: LibriSpeech `train-clean-100` + DroneAudioDataset
- 1 s samples, 16 kHz mono, SNR −30…0 dB
- Split: 6480 train / 720 valid

### DREGON-LM (mono) — Paper 2 baseline

Script: `create_dregon_librimix.py` (root level), no flags.

- Sources: DREGON `in_flight_noise` (train) + LibriSpeech; `in_flight_source`
  recordings (valid) also mixed with LibriSpeech
- Default: 3 s samples, 16 kHz mono
- Per sample: `mixture.wav` `(1, T)`, `vocals.wav`, `noise.wav`, `rps.npy (4, M)`
- Motor-combo synthetic samples: `--motor_combo_fraction 0.2` (20 % of train)

### DREGON-LM (multichannel) — current

Script: `create_dregon_librimix.py --multichannel`

- Sources: same as mono but all 8 mic channels kept together
- Per sample: `mixture.wav (T, 8)`, `vocals.wav (T, 8)`, `noise.wav (T, 8)`,
  `rps.npy (4, M)` — rps is shared across channels (same recording = same motors)
- Speech/SNR independent per channel by default (`--speech_per_channel independent`)
- Channel axis = minibatch at train time; eval reports per-channel metrics

### DREGON-LM-RealValid — recommended for RPS evaluation

`--multichannel --real_valid` together:

- **Train**: synthesised multichannel mixtures from `in_flight_noise` + LibriSpeech
- **Valid**: raw 8-channel clips from `in_flight_source` recordings, **no mixing**
  - `mixture.wav (T, 8)` = raw recording (drone + co-recorded source)
  - `rps.npy (4, M)` = motor telemetry
  - No `vocals.wav` / `noise.wav` (no clean reference exists)
  - `source_type` in metadata: `"speech"` or `"whitenoise"`
  - ~15 non-overlapping 8-second clips available (2 recordings × ~7–8 clips each)

---

## Canonical Dataset Creation Command

```bash
python create_dregon_librimix.py \
  --multichannel --real_valid \
  --output_dir datasets/DREGON-LM-RealValid \
  --num_train 6000   --duration 1.0 \
  --num_valid 30     --valid_duration 8.0 \
  --min_motor_rps 30.0 \
  --source_white_noise_prob 0.3 \
  --speech_per_channel independent \
  --snr_min -30 --snr_max 0
```

Key flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--multichannel` | off | Produce full 8-channel samples |
| `--real_valid` | off | Valid = raw in_flight_source clips (no mixing) |
| `--min_motor_rps` | 0.0 | **Set to 30.0** to exclude takeoff/landing from all splits |
| `--duration` | 3.0 | Train sample length (s); 1.0 recommended |
| `--valid_duration` | 8.0 | Valid clip length (s) for `--real_valid` |
| `--source_white_noise_prob` | 0.0 | Fraction of train samples using WN *instead of* speech |
| `--white_noise_prob` | 0.0 | Adds WN *on top of* speech (legacy, usually 0) |
| `--speech_per_channel` | independent | `independent` = different utterance+SNR per channel |
| `--valid_recording_ids` | speech-low,whitenoise-low | Comma-separated IDs for `--real_valid` |
| `--motor_combo_fraction` | 0.2 | Synthetic motor combos (mono pipeline only) |

---

## Multichannel Training & Evaluation Wiring

### Training (`train_rps_predictor.py`)

`DREGONRPSDataset.__getitem__` returns:
- Mono files (`C=1`): `audio (T,)`, `rps (4, F)`
- Multichannel files (`C>1`): `audio (C, T)`, `rps (4, F)`

The training loop calls `_flatten_channels(audio, rps)`:
- `(B, T)` → no-op, C=1
- `(B, C, T)` → `(B*C, T)` audio; `(B, 4, F)` → broadcast+reshape → `(B*C, 4, F)` rps

`SimpleConv` (and all `rps_predictor` model variants) accept any leading batch
shape before the time dimension — `torch.stft` treats everything before `T` as
batch.  DCUNet/DCCRN are **not** covered by this flattening; they handle 1-channel
mono only.

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

`dregon.py` provides:
- `load_timeframe(sample)` — load a recording as `TimeFrame`
- `clean_command_spikes(command)` — `(4, M)` in/out; zeros leading freeze,
  applies median filter along time axis.  Time is the **last** axis.
- `load_dregon_timeframes(data_dir, splits=…)` — load all recordings in splits
- `_find_inflight_window(tf, motor_key, min_motor_rps)` — in `create_dregon_librimix.py`

---

## Publishing a processed dataset

```bash
dvc add datasets/DREGON-LM-RealValid && dvc push
git add datasets/DREGON-LM-RealValid.dvc datasets/.gitignore
git commit -m "dataset: DREGON-LM-RealValid" && git push
```

See `docs/data-and-artifacts.md` for the end-to-end CPU → GPU → laptop flow.

---

## Gotchas

- **Datasets are gitignored** — create locally or `dvc pull` before training.
- **`motors_command` trailing freeze**: the last 45–1577 samples are identical
  (logger stopped).  `_find_inflight_window` strips this when using `motors_measured`;
  when only command is available, the end trim is effectively 0 s.  **Never use
  raw command tail samples as ground-truth RPS.**
- **EventSeries values are time-last `(…, M)`** — see `src/utils/data/AGENTS.md`.
  `load_timeframe` stores motor/imu/source telemetry as `(4|3, M)`.  Older code
  used `(M, 4)` + `.T`; if you see `(M, 0)` after slicing or a `.T` on motor
  values, it predates the convention fix (June 2026).
- **`load_timeframe(target_sr=…)` resampling**: `librosa.resample` must receive
  the `(n_ch, N)` array with `axis=-1`.  The wrong axis (resampling the 8-element
  channel dimension) loops over ~3M tiny signals and hangs for minutes.  Always
  use `res_type="soxr_hq"` (`resampy`/kaiser not installed).
- **`--min_motor_rps`** defaults to 0 (backward-compat).  Always pass `30.0` for
  new datasets to exclude takeoff and any visible landing.
- **`--real_valid` valid set has no `vocals.wav`** — only `mixture.wav` and
  `rps.npy`.  Do not run `final_valid.py` speech-enhancement eval on it; it is
  for RPS prediction evaluation only.
- **Valid clip count ceiling**: with `--real_valid --valid_duration 8.0`, only
  ~15 non-overlapping clips exist across the 2 default recordings.  Larger
  `--num_valid` will overlap; this is fine for RPS eval but be aware.
- **`source_white_noise_prob` vs `white_noise_prob`**: the former replaces speech
  with white noise as the *target source*; the latter adds WN *on top of* speech.
  For diversity training use `--source_white_noise_prob 0.2–0.4`.
