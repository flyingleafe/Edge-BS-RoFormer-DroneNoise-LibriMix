---
experiment: f1_htdemucs_b
training_config: conf/experiment/f1_htdemucs_b.yaml
batch: docs/experiments/f1-se-blind-baselines.md
---

# `f1_htdemucs_b`

## Motivation

HTDemucs v4 (hybrid time+spectrogram transformer, ~42M) fine-tuned from the
official Meta music-separation checkpoint, Pass B (all harmonic noises,
category-uniform) of the F1 blind speech-enhancement baseline program — the
diversity arm for the pretrained model.

Full batch context: [F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).
All pretrained-adaptation design decisions (checkpoint staging, resampling,
mono->stereo, 4->2 head remap, fine-tune regime) are documented in
[`f1_htdemucs_a.md`](./f1_htdemucs_a.md) and apply unchanged.

## Setup

Hydra wiring — data `se_baselines_b` (category-uniform all-harmonic online-mix
SE stream + fixed SE-valid-drone) · model `f1_htdemucs` · loss `si_sdr_mrstft`
· metrics `separation_basic` (eval with `metrics=separation_full` for
PESQ/eSTOI). Train with `python train.py experiment=f1_htdemucs_b`, evaluate
with `python eval.py experiment=f1_htdemucs_b metrics=separation_full`.

## Conclusion

Reported comparatively in the batch write-up — see
[F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).
