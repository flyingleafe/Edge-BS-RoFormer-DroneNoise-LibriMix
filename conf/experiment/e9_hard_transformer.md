---
experiment: e9_hard_transformer
training_config: conf/experiment/e9_hard_transformer.yaml
batch: docs/experiments/e9-hard-combined.md
---

# `e9_hard_transformer`

## Motivation

E7 (neural-generated) and E8 (static-comb) each failed to transfer to real RPS
prediction on their own (real val PIT MSE ~189–225, R² < 0). E8's static comb
helped the transformer modestly, and the RPS distribution roughly matches real,
so the gap is not amplitude-shortcut-only nor RPS-range. E9 makes the task
**harder** so the predictor cannot exploit either source's idiosyncrasy: the
noise pool is **50% neural generator + 50% static-comb**, with LibriSpeech
mixtures and **50% mixture-level augmentation from the start**, trained with
**patience 20**. The converged `last.ckpt` (not the early val-best `best.ckpt`)
is the artifact for the sim→real failure-mode diagnostic.

## Setup

Data `rps_hard_combined` (50/50 generated+static_comb noise, fixed real valid),
model `simple_conv_v2_transformer`, loss `pit_mse`, metrics `rps`. Patience 20,
batch 16, `samples_per_validation=5000`. The neural-generator source needs a GPU
producer (`device: cuda:0`). Cloud: override train policy to
`rps_hard_combined_dload.yaml`, valid to `dload:DREGON-LM-V4-michaels-valid`.
See the [E9 batch doc](../../docs/experiments/e9-hard-combined.md).

Train: `python train.py experiment=e9_hard_transformer`.

## Conclusion

Two runs. 2026-07-11 on the **contaminated** valid: best PIT-MSE 176.5 (R² −7.9). 2026-07-12 rerun on the clean valid (`min_motor_rps=50`): best PIT-MSE **19.2** (R² 0.59) — sim transfer is real. See the conclusion of [e9-hard-combined.md](../../docs/experiments/e9-hard-combined.md).

*(Backfilled 2026-08-20.)*
