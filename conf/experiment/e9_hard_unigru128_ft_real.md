---
experiment: e9_hard_unigru128_ft_real
training_config: conf/experiment/e9_hard_unigru128_ft_real.yaml
batch: docs/experiments/e9-hard-combined.md
---

# `e9_hard_unigru128_ft_real`

## Motivation

Stage 2 of the E9 sim->real curriculum for `simple_conv_v2_uni_gru128`: fine-tune on **real** data
(DREGON in_flight_noise + Michael's FLY125 online mix, no augmentation),
warm-started from the stage-1 `e9_hard_unigru128` `best.ckpt` (converged on the
hard combined generated-noise task). Tests whether a brief real fine-tune closes
the residual sim->real gap left after training on generated noise. Validated on
the **clean** free-flight-only `DREGON-LM-V4-michaels-valid` (min_motor_rps=50).

## Setup

Data `e9_real_ft` (real online mix, no aug, michaels `min_motor_rps:50` so
FLY125 ground warm-up is excluded), model `simple_conv_v2_uni_gru128`, loss `pit_mse`, metrics
`rps`. Warm-start `checkpoint: r2://ml-data/artifacts/e9_hard_unigru128/checkpoints/best.ckpt`
(loaded `strict=False` before training). Renewed patience 20, batch 16,
`samples_per_validation=5000`. Cloud: override train policy to
`e9_real_ft_dload.yaml`, valid to `dload:DREGON-LM-V4-michaels-valid`.

Train: `python train.py experiment=e9_hard_unigru128_ft_real` (after stage 1
uploads best.ckpt).

## Conclusion

Ran 2026-07-12. Best clean-valid PIT-MSE **11.1** (R² 0.74) — the best E9 number. See the conclusion of [e9-hard-combined.md](../../docs/experiments/e9-hard-combined.md).

*(Backfilled 2026-08-20.)*
