---
experiment: f2_dcunet_avq_survey
training_config: conf/experiment/f2_dcunet_avq_survey.yaml
batch: docs/experiments/f2-survey-replication.md
---

# `f2_dcunet_avq_survey`

## Motivation

Faithful replication of the DCUNet arm of Mukhutdinov et al. 2023 (IEEE Access,
"Deep Learning Models for Single-Channel Speech Enhancement on Drones"), the
survey whose 12-model benchmark DCUNet won — run here to check whether we
reproduce its reported DCUNet result before drawing any further conclusion from
F1, where DCUNet came out weakest of five architectures.

Replication target (paper, at −15 dB input SNR): SI-SDR **+3.7 dB**, eSTOI
**0.4**, PESQ **1.9** — measured there on the AS drone; AVQ ego-noise should be
no harder. Those numbers were measured at 8 kHz on TIMIT while we run at 16 kHz
on LibriSpeech, so treat them as a **reference point, not an exact bar** (see
the batch doc's § Deliberate deviations).

Full batch context: [F2 — 2023 IEEE Access survey replication](../../docs/experiments/f2-survey-replication.md).

## Setup

Protocol: 16 kHz (project-native — the paper's 8 kHz is deliberately not
replicated), 3.0 s (48000-sample) crops, noise = the 5 AVQ ego-noise
sequences (first channel only, same recordings for train and valid), speech =
LibriSpeech train-clean-100 with the F1 held-out ~10 % of speakers reserved for
the valid set, train SNR ~ U(−25, −5) dB mixed on the fly, fixed valid at SNR
{−25,−20,−15,−10,−5} dB. DCUNet-10 (5 encoder + 5 decoder complex convs), STFT
64 ms / 16 ms (the paper's resolution, in ms) → n_fft 1024, hop 256, dim_f 512,
dim_t 188. SI-SDR loss only.
Adam, lr 1e-3, LR-on-plateau patience 5 (×0.1), early stop patience 10,
batch 32.

Hydra wiring — data `f2_avq_survey` (online-mix SE stream + fixed local
`SE-valid-avq-survey`) · model `f2_dcunet_survey` · loss `si_sdr` · metrics
`separation_basic` (the standard 16 kHz groups). Build the
valid set once with
`python data_processing.derivations (se_valid generator) --dataset avq --local-repo datasets/se-valid-local`,
then train with `python train.py experiment=f2_dcunet_avq_survey` and evaluate
with `python eval.py experiment=f2_dcunet_avq_survey metrics=separation_full`.

## Conclusion

Reported comparatively in the batch write-up — see
[F2 — 2023 IEEE Access survey replication](../../docs/experiments/f2-survey-replication.md).
