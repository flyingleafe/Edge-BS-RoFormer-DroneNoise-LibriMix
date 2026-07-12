---
experiment: e11_full_aug_scv2
training_config: conf/experiment/e11_full_aug_scv2.yaml
batch: docs/experiments/e10-full-flight.md
---

# `e11_full_aug_scv2`

## Motivation

E11 fixes the two E10 failures for `simple_conv_v2`: (1) the generator now goes to
**true silence at zero RPS** (emitter smoothstep silence-fade, `e11_noisegen_silence`)
instead of a DC pedestal; (2) **augmentation is restored** — it is a domain-gap
*reducer*, and removing it in E10 let the model overfit synthetic cruise texture
and under-predict real cruise (~51 vs 80). Full-flight envelope + full-envelope
valid retained.

## Setup

Data `e11_full_aug` (50% neural gen full_flight from `e11_noisegen_silence` +
50% static-comb full_flight + LibriSpeech + augmentation), model `simple_conv_v2`, loss
`pit_mse`, metrics `rps`. epochs 200, patience 20, batch 16,
`samples_per_validation=5000`. Needs `e11_noisegen_silence` best.ckpt in R2.
Neural-gen source needs a GPU producer. Kaggle/P100: override train policy to
`e11_full_aug_p100.yaml`. Valid `dload:DREGON-LM-V4-michaels-valid-full`.

Train: `python train.py experiment=e11_full_aug_scv2`.

## Conclusion

_Pending run._
