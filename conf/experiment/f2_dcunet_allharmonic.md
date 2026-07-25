---
experiment: f2_dcunet_allharmonic
training_config: conf/experiment/f2_dcunet_allharmonic.yaml
batch: docs/experiments/f2-survey-replication.md
---

# `f2_dcunet_allharmonic`

## Motivation

Step 3 of the F2 ladder, and the widest arm. Steps 1 and 2 train on one drone's
ego-noise and on all drone noise respectively; this arm widens the training
noise pool once more, to the full **category-uniform harmonic** pool of F1
Pass B (drone + MIMII + MIMII-DG + AeroSonicDB + HUSTmotor +
KAIST-rotating-acoustic + HornBase), with AVQ kept inside the drone category so
the pool remains a strict superset of steps 1 and 2.

Together the three arms isolate how the survey-protocol DCUNet number moves as
the training noise goes from one drone → all drones → all harmonic noise, which
is the F1 training distribution. If the score degrades monotonically, the F1
DCUNet weakness is a noise-diversity/capacity effect rather than a pipeline
defect.

Full batch context: [F2 — 2023 IEEE Access survey replication](../../docs/experiments/f2-survey-replication.md).

## Setup

Identical to `f2_dcunet_avq_survey` in every respect except `override /data`:
8 kHz, 3.0 s (24000-sample) crops, speech = LibriSpeech train-clean-100 with the
F1 held-out ~10 % of speakers reserved, train SNR ~ U(−25, −5) dB mixed on the
fly, DCUNet-10 (5 encoder + 5 decoder complex convs), STFT n_fft 512 / hop 128 /
dim_f 256, SI-SDR loss only, Adam lr 1e-3, LR-on-plateau patience 5 (×0.1),
early stop patience 10, batch 32, `samples_per_validation: 41580`.

**Validation is deliberately unchanged**: the same fixed `SE-valid-avq-survey`
set (250 clips at SNR {−25,−20,−15,−10,−5} dB, 8 kHz) as steps 1 and 2, so the
monitor, the LR-plateau trigger and early stopping all see exactly the same
distribution in every arm and the step-to-step deltas are directly comparable.
Note this makes step 3 an out-of-distribution-ish test by construction: most of
its training noise is not drone ego-noise, but it is still scored on AVQ.

Hydra wiring — data `f2_allharmonic` (train stream
`conf/online_mix/se_survey_allharmonic.yaml`, valid `SE-valid-avq-survey`) ·
model `f2_dcunet_survey` · loss `si_sdr_8k` · metrics `separation_basic_8k`.
Train with `python train.py experiment=f2_dcunet_allharmonic` and evaluate with
`python eval.py experiment=f2_dcunet_allharmonic metrics=separation_full`.

## Conclusion

Reported comparatively in the batch write-up — see
[F2 — 2023 IEEE Access survey replication](../../docs/experiments/f2-survey-replication.md).
