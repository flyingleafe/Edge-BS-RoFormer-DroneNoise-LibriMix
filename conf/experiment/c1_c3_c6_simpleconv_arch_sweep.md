---
experiment: c1_c3_c6_simpleconv_arch_sweep
training_config: conf/experiment/c1_c3_c6_simpleconv_arch_sweep.yaml
batch: docs/experiments/simpleconv-rps-architecture-search.md
---

# `c1_c3_c6_simpleconv_arch_sweep`

## Motivation

Searches SimpleConv encoder / temporal-head architectures for direct RPS prediction from drone audio, offline and under online mixing.

Full batch context: [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).

## Setup

C1/C3/C6 — first RPS predictor training + the 10-variant SimpleConv arch-zoo sweep, all on the ORIGINAL DREGON-LM (v1) dataset. These three labels refer to the SAME historical sweep event (see REPLICATION.md § C1/ C3/C6): `simple_conv` (this file's default `model`) is both the literal "first" RPS predictor trained (C1) and the baseline of the 10-variant comparison (C3/C6). Dataset later flagged "trivially easy" / dataset-flawed once DREGON-LM-V2 existed for comparison (1 s-equivalent structure, train/ valid overlap risk) — kept for historical value, not recommended for new work. To reproduce a different sweep variant, override `model`: python train.py experiment=c1_c3_c6_simpleconv_arch_sweep model=simple_conv_bigru Full variant list (registry keys, all have a conf/model/<key>.yaml): simple_conv (this file), simple_conv_bigru, simple_conv_bigru_v2, simple_conv_v2, simple_conv_tcn, simple_conv_magphase_bigru, simple_conv_attn_pool, simple_conv_wide, simple_conv_multiscale, simple_conv_se_next. Historical command: train_rps_predictor.py --model <key> --epochs 200 --patience 15 --batch_size 16 --lr 0.001 --weight_decay 0.0001 --grad_clip 5.0.

Hydra wiring — data `dregon_lm_v1` · model `simple_conv` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c1_c3_c6_simpleconv_arch_sweep`,
evaluate with `python eval.py experiment=c1_c3_c6_simpleconv_arch_sweep`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).
