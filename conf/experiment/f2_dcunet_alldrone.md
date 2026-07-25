---
experiment: f2_dcunet_alldrone
training_config: conf/experiment/f2_dcunet_alldrone.yaml
batch: docs/experiments/f2-survey-replication.md
---

# `f2_dcunet_alldrone`

## Motivation

Step 2 of the F2 ladder. Step 1 (`f2_dcunet_avq_survey`) reproduces the
Mukhutdinov et al. 2023 (IEEE Access) DCUNet protocol exactly, training on a
single drone's ego-noise. This arm changes **one thing and one thing only**: the
training noise pool is widened from the 5 AVQ ego-noise sequences to *all* drone
noise sources in the data lake (the F1 Pass A pool — DREGON + Michael's FLY125
real frames + drone_audio + DroneAudioSet + SPCUP19-egonoise + new-drone-noises)
with AVQ retained, so the pool is a strict superset of step 1's.

The question is whether the survey-protocol DCUNet result survives — or improves
with — drone-noise diversity, and hence how much of the F1/F2 gap is explained
by noise-pool breadth rather than by architecture or pipeline.

Full batch context: [F2 — 2023 IEEE Access survey replication](../../docs/experiments/f2-survey-replication.md).

## Setup

Identical to `f2_dcunet_avq_survey` in every respect except `override /data`:
8 kHz, 3.0 s (24000-sample) crops, speech = LibriSpeech train-clean-100 with the
F1 held-out ~10 % of speakers reserved, train SNR ~ U(−25, −5) dB mixed on the
fly, DCUNet-10 (5 encoder + 5 decoder complex convs), STFT n_fft 512 / hop 128 /
dim_f 256, SI-SDR loss only, Adam lr 1e-3, LR-on-plateau patience 5 (×0.1),
early stop patience 10, batch 32, `samples_per_validation: 41580`.

**Validation is deliberately unchanged**: the same fixed `SE-valid-avq-survey`
set (250 clips at SNR {−25,−20,−15,−10,−5} dB, 8 kHz) as steps 1 and 3, so the
monitor, the LR-plateau trigger and early stopping all see exactly the same
distribution in every arm and the step-to-step deltas are directly comparable.

Hydra wiring — data `f2_alldrone` (train stream
`conf/online_mix/se_survey_alldrone.yaml`, valid `SE-valid-avq-survey`) · model
`f2_dcunet_survey` · loss `si_sdr_8k` · metrics `separation_basic_8k`. Train
with `python train.py experiment=f2_dcunet_alldrone` and evaluate with
`python eval.py experiment=f2_dcunet_alldrone metrics=separation_full`.

## Conclusion

Reported comparatively in the batch write-up — see
[F2 — 2023 IEEE Access survey replication](../../docs/experiments/f2-survey-replication.md).
