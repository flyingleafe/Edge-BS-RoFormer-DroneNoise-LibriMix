---
experiment: c10_arch_sweep_online
training_config: conf/experiment/c10_arch_sweep_online.yaml
batch: docs/experiments/simpleconv-rps-architecture-search.md
---

# `c10_arch_sweep_online`

## Motivation

Searches SimpleConv encoder / temporal-head architectures for direct RPS prediction from drone audio, offline and under online mixing.

Full batch context: [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).

## Setup

C10 — 26-variant SimpleConv architecture sweep, ONLINE-MIX arm (autoresearch session 20260617-012233, 2026-06-18 rerun). Same 26 model keys as conf/experiment/c10_arch_sweep_offline.yaml — override `model` per variant. The dedicated winner (simple_conv_v2_uni_gru128, PIT MSE 7.33, best overall) has its own named file: conf/experiment/c10_uni_gru128_online.yaml (mirrors the worked example in docs/refactor-unified-framework.md § "Hydra config architecture"). Clipped-GRU follow-up (uni_gru* variants only, grad_clip=0.5): python train.py experiment=c10_arch_sweep_online model=simple_conv_v2_uni_gru grad_clip=0.5 Historical command: train_rps_predictor.py --model <key> --data_root datasets/DREGON-LM-V4-michaels --online_mix --mix_config conf/online_mix/online_mix_v4_michaels_train_no_room1_gpfs.yaml --samples_per_validation 5000 --epochs 200 --patience 50 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse. Results on HPC scratch (not synced) — see REPLICATION.md § C10.

Hydra wiring — data `online_mix_v4_michaels` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c10_arch_sweep_online`,
evaluate with `python eval.py experiment=c10_arch_sweep_online`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).
