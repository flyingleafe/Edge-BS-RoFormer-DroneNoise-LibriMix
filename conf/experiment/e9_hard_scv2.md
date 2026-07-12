---
experiment: e9_hard_scv2
training_config: conf/experiment/e9_hard_scv2.yaml
batch: docs/experiments/e9-hard-combined.md
---

# `e9_hard_scv2`

## Motivation

Sibling of [`e9_hard_transformer`](./e9_hard_transformer.md) for the plain
`simple_conv_v2` arch (real-trained baseline PIT MSE 9.71). Extends the E9
"hard" combined generated-noise recipe to all three archs so the sim→real
transfer is measured per-architecture on the **clean** free-flight-only
validation split (`DREGON-LM-V4-michaels-valid` rebuilt with `min_motor_rps=50`,
dropping the FLY124 ground warm-up that had inflated the old val metric).
Stage 1 of the sim→real curriculum; stage 2 = `e9_hard_scv2_ft_real`.

## Setup

Data `rps_hard_combined` (50% neural generator + 50% static-comb + LibriSpeech +
50% augmentation, fixed clean real valid), model `simple_conv_v2`, loss
`pit_mse`, metrics `rps`. Patience 20, batch 16, `samples_per_validation=5000`.
The neural-generator source needs a GPU producer (`device: cuda:0`). Cloud:
override train policy to `rps_hard_combined_dload.yaml`, valid to
`dload:DREGON-LM-V4-michaels-valid`. See the
[E9 batch doc](../../docs/experiments/e9-hard-combined.md).

Train: `python train.py experiment=e9_hard_scv2`.

## Conclusion

_Pending run._
