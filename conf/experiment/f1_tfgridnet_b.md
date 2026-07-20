---
experiment: f1_tfgridnet_b
training_config: conf/experiment/f1_tfgridnet_b.yaml
batch: docs/experiments/f1-se-blind-baselines.md
---

# `f1_tfgridnet_b`

## Motivation

TF-GridNet (dense full+sub-band dual-path, mid-size ~8.4M), Pass B (all harmonic noises (category-uniform)) of the F1 blind speech-enhancement baseline
program — the diversity arm. One rung of the 5-arch x 2-pass floor comparison.

Full batch context: [F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).

## Setup

Hydra wiring — data `se_baselines_b` (online-mix SE stream + fixed SE-valid-drone) ·
model `f1_tfgridnet` · loss `masked_mse` · metrics `separation_basic`
(eval with `metrics=separation_full` for PESQ/eSTOI). Train with
`python train.py experiment=f1_tfgridnet_b`, evaluate with
`python eval.py experiment=f1_tfgridnet_b metrics=separation_full`.

## Conclusion

Reported comparatively in the batch write-up — see
[F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).
