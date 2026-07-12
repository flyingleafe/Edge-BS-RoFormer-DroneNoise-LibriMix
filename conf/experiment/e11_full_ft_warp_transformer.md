---
experiment: e11_full_ft_warp_transformer
training_config: conf/experiment/e11_full_ft_warp_transformer.yaml
batch: docs/experiments/e10-full-flight.md
---

# `e11_full_ft_warp_transformer`

## Motivation

Stage 2 of the E11 full-flight sim->real curriculum for `simple_conv_v2_transformer`: fine-tune on
real data with time-warp, warm-started from the full-flight-pretrained
`e11_full_aug_transformer`. Uses the SAME real+timewarp recipe as the
`e11_real_warp_transformer` baseline, so the only difference is the sim pretraining
— isolating its effect. Validated on the FULL-envelope real split.

## Setup

Data `e11_real_warp` (real online mix + time-warp + aug), model `simple_conv_v2_transformer`,
warm-start `checkpoint: r2://ml-data/artifacts/e11_full_aug_transformer/checkpoints/best.ckpt`,
patience 20, batch 16. Valid `dload:DREGON-LM-V4-michaels-valid-full`.

Train: `python train.py experiment=e11_full_ft_warp_transformer`.

## Conclusion

_Pending run._
