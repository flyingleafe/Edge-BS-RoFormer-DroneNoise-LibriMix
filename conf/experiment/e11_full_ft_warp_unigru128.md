---
experiment: e11_full_ft_warp_unigru128
training_config: conf/experiment/e11_full_ft_warp_unigru128.yaml
batch: docs/experiments/e10-full-flight.md
---

# `e11_full_ft_warp_unigru128`

## Motivation

Stage 2 of the E11 full-flight sim->real curriculum for `simple_conv_v2_uni_gru128`: fine-tune on
real data with time-warp, warm-started from the full-flight-pretrained
`e11_full_aug_unigru128`. Uses the SAME real+timewarp recipe as the
`e11_real_warp_unigru128` baseline, so the only difference is the sim pretraining
— isolating its effect. Validated on the FULL-envelope real split.

## Setup

Data `e11_real_warp` (real online mix + time-warp + aug), model `simple_conv_v2_uni_gru128`,
warm-start `checkpoint: r2://ml-data/artifacts/e11_full_aug_unigru128/checkpoints/best.ckpt`,
patience 20, batch 16. Valid `dload:DREGON-LM-V4-michaels-valid-full`.

Train: `python train.py experiment=e11_full_ft_warp_unigru128`.

## Conclusion

Ran 2026-07-12. Sim curriculum: full-envelope PIT-MSE 179.6 vs baseline 253.0 vs E12 190.0. See [e10-full-flight.md](../../docs/experiments/e10-full-flight.md).

*(Backfilled 2026-08-20.)*
