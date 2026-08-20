---
experiment: e11_real_warp_scv2
training_config: conf/experiment/e11_real_warp_scv2.yaml
batch: docs/experiments/e10-full-flight.md
---

# `e11_real_warp_scv2`

## Motivation

Real-data-only baseline for `simple_conv_v2` with time-warp augmentation — the
comparison point for the E11 sim->real full-flight curriculum. Trained only on
real DREGON+FLY125 online mixtures (E5 time-warp recipe), no synthetic noise, no
warm-start, validated on the FULL-envelope real split so it is directly
comparable to the sim-pretrained + real-finetuned models.

## Setup

Data `e11_real_warp` (real online mix + time-warp + augmentation), model
`simple_conv_v2`, loss `pit_mse`, metrics `rps`. epochs 200, patience 20, batch 16.
Valid `dload:DREGON-LM-V4-michaels-valid-full`.

Train: `python train.py experiment=e11_real_warp_scv2`.

## Conclusion

Ran 2026-07-12. Real-only cruise-trained baseline: full-envelope PIT-MSE 183.4. See [e10-full-flight.md](../../docs/experiments/e10-full-flight.md).

*(Backfilled 2026-08-20.)*
