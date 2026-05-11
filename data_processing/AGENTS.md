# data_processing/ — Dataset Creation and RPS Processing

Contains scripts and modules for creating and processing training datasets. Currently handles the DREGON dataset with its motor telemetry (RPS) data.

## Why this directory exists

Dataset creation and preprocessing is a separate concern from model training. Raw datasets need to be downloaded, processed, and mixed into training-ready formats before any experiment can run.

## Files

| File | Purpose |
|------|---------|
| `dregon.py` | DREGON dataset loading, RPS processing, and DREGON-LM creation utilities |
| `michaels.py` | Michael's drone-noise dataset (DJI WAVs + flight-controller CSVs in `data/new-drone-noises/`). `MichaelsRecord` is duck-type compatible with `DREGONRecord` (`audio`, `audio_timestamps`, `motors`, `slice_by_time`). Handles per-file CSV-vs-audio time-offset alignment. |
| `noise_rps_dataset.py` | `NoiseRPSDataset` — combined chunkable dataset over DREGON `in_flight_noise` + Michael's. Each item: RPS upsampled to audio rate + matched recorded noise. Use `build_noise_rps_datasets(...)` to get train/val with held-out per-recording time tails. |
| `__init__.py` | Package init |

## Datasets

### Michael's drone noise + RPS

Files in `data/new-drone-noises/`:
- `103_2.wav` + `FLY103.csv` (offset −0.94 s, valid window 12–100 s)
- `108_2.wav` + `FLY108.csv` (offset −0.40 s, valid window 9–88 s)

The CSV columns `Motor:Speed:{RFront,LFront,LBack,RBack}` log motor RPM at
~30 Hz; the loader divides by 60 to match DREGON's RPS units (Hz). Audio is
resampled to the requested `sample_rate` (default 16 kHz). Empirical
time-offsets between the WAV and CSV timelines come from the legacy
`drone_audition` repo and are stored in `michaels.MICHAELS_FILES`.

### DN-LM (DroneNoise-LibriMix) — Paper 1

Created by `create_dataset.py` (root level, not this directory):
- Sources: LibriSpeech `train-clean-100` + DroneAudioDataset
- Duration: 2h total, 1s samples at 16 kHz mono
- SNR range: −30 dB to 0 dB
- Split: 6480 train / 720 valid

### DREGON-LM — Paper 2

Created by `create_dregon_librimix.py` (root level) using functions from `dregon.py`:
- Sources: DREGON dataset (real UAV flight recordings + motor telemetry) + LibriSpeech
- Key feature: `rps.npy` per sample — 4 rotors at ~929 Hz
- Sample duration: 8.224s (131584 samples at 16 kHz)
- Downloads DREGON from HuggingFace via `datasets` library

## RPS Processing

`dregon.py` provides utilities for:
- Loading raw DREGON motor speed data
- Resampling RPS from motor sampling rate (~929 Hz) to STFT frame rate
- Creating per-sample `rps.npy` files with shape `(4, n_motor_samples)`

## Publishing a processed dataset

After creation, publish via DVC so other machines (`postdoc job submit` preflight)
can auto-pull:

```bash
dvc add datasets/DREGON-LM && dvc push
git add datasets/DREGON-LM.dvc datasets/.gitignore && git commit -m "dataset: DREGON-LM v1" && git push
```

See `docs/data-and-artifacts.md` for the end-to-end CPU → GPU → laptop flow.

## Gotchas

- Datasets are gitignored — they must be created locally OR pulled via `dvc pull` before training
- RPS data shape is `(4, n_motor_samples)` — `RotorEncoder` in `models/dcunet.py` resamples this to STFT frames
- `create_dregon_librimix.py` auto-downloads DREGON from HuggingFace