---
experiment: c10_uni_gru128_online
training_config: conf/experiment/c10_uni_gru128_online.yaml
batch: docs/experiments/simpleconv-rps-architecture-search.md
---

# `c10_uni_gru128_online`

## Motivation

Searches SimpleConv encoder / temporal-head architectures for direct RPS prediction from drone audio, offline and under online mixing.

Full batch context: [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).

## Setup

C10 winner — simple_conv_v2_uni_gru128, online-mix arm (PIT MSE 7.33, R² 0.82; best result of the whole 26-variant × 2-regime sweep). Notably the causal uni-GRU family "fails offline" (NaNs/collapses in c10_arch_sweep_offline) but becomes the best online-mix variant — see REPLICATION.md § C10. This file mirrors the worked example in docs/refactor-unified-framework.md § "Hydra config architecture" (that doc's illustrative `data: dregon_lm_v4_michaels_online` is this repo's actual `online_mix_v4_michaels` group). Results on HPC scratch (not synced) — see REPLICATION.md § C10.

Hydra wiring — data `online_mix_v4_michaels` · model `simple_conv_v2_uni_gru128` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c10_uni_gru128_online`,
evaluate with `python eval.py experiment=c10_uni_gru128_online`.

## Conclusion

The online-mixing arm winner: PIT MSE **7.33**, R^2 0.82.
