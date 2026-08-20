---
experiment: e12_real_fullflight_scv2
training_config: conf/experiment/e12_real_fullflight_scv2.yaml
batch: docs/experiments/e10-full-flight.md
---

# `e12_real_fullflight_scv2`

## Motivation

Diagnostic ceiling for the sim->real full-flight work. The E11 sim-pretrained +
real-finetuned models scored only ~200+ aggregate PIT-MSE on the full-envelope
real validation set, raising the worry that models simply collapse toward the
mean and cannot predict RPS across regimes at all. This run removes synthetic
noise from the question: train `simple_conv_v2` on the REAL full flight by
keeping the whole powered envelope of the DREGON noise source
(`min_motor_rps: 0` instead of 30), so training sees the real warm-up /
take-off ramp windows, not just cruise. If a model trained directly on real
full-flight data predicts the low-RPS regimes well, the limitation is sim->real
transfer; if it also collapses, the limitation is the data/task itself.

## Setup

Data `e12_real_fullflight` (real DREGON in-flight w/ `min_motor_rps: 0` +
FLY125, online mix + E5 time-warp + augmentation), model `simple_conv_v2`,
loss `pit_mse`, metrics `rps`. epochs 200, patience 20, batch 16. Valid
`dload:DREGON-LM-V4-michaels-valid-full` (FLY124 — no leakage).

Train: `python train.py experiment=e12_real_fullflight_scv2`.

## Conclusion

Ran 2026-07-12. Full-envelope PIT-MSE 145.1 vs curriculum 132.5 vs baseline 183.4. See [e10-full-flight.md](../../docs/experiments/e10-full-flight.md).

*(Backfilled 2026-08-20.)*
