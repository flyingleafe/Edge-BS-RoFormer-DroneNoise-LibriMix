---
experiment: e10_full_transformer
training_config: conf/experiment/e10_full_transformer.yaml
batch: docs/experiments/e10-full-flight.md
---

# `e10_full_transformer`

## Motivation

Train `simple_conv_v2_transformer` on the FULL-FLIGHT combined synthetic noise task with NO
augmentation, validated on the FULL-envelope real split. The full-flight RPS
trajectories (rps_synthesis.generate_full_flight) + zero->silence amplitude make
the synthetic data span warm-up/takeoff/landing/ground, so the predictor learns
the regimes real recordings contain instead of only cruise. Deliberately simple
(no augmentation, no hard shortcuts): the earlier E7-E9 "overfits" were domain
mismatch, addressed here by reducing the mismatch, not by making the task harder.

## Setup

Data `e10_full_flight` (50% neural gen full_flight from
`e10_noisegen_fullrange` + 50% static-comb full_flight + LibriSpeech, no aug),
model `simple_conv_v2_transformer`, loss `pit_mse`, metrics `rps`. epochs 200, patience 20,
batch 16, `samples_per_validation=5000`. Needs the retrained generator
(`e10_noisegen_fullrange` best.ckpt) in R2. Neural-gen source needs a GPU
producer. Kaggle/P100: override train policy to `e10_full_flight_p100.yaml`.
Valid streams `dload:DREGON-LM-V4-michaels-valid-full`.

Train: `python train.py experiment=e10_full_transformer`.

## Conclusion

_Pending run._
