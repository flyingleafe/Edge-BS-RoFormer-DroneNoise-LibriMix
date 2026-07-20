---
experiment: f1_tfgridnet_a
training_config: conf/experiment/f1_tfgridnet_a.yaml
batch: docs/experiments/f1-se-blind-baselines.md
---

# `f1_tfgridnet_a`

## Motivation

TF-GridNet (dense full+sub-band dual-path, mid-size ~8.4M), Pass A (drone noises only) of the F1 blind speech-enhancement baseline
program — the drone-focused floor. One rung of the 5-arch x 2-pass floor comparison.

Full batch context: [F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).

## Setup

Hydra wiring — data `se_baselines_a` (online-mix SE stream + fixed SE-valid-drone) ·
model `f1_tfgridnet` · loss `masked_mse` · metrics `separation_basic`
(eval with `metrics=separation_full` for PESQ/eSTOI). Train with
`python train.py experiment=f1_tfgridnet_a`, evaluate with
`python eval.py experiment=f1_tfgridnet_a metrics=separation_full`.

## Conclusion

Reported comparatively in the batch write-up — see
[F1 — SE blind baselines](../../docs/experiments/f1-se-blind-baselines.md).
