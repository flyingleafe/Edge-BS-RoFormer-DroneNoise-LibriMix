---
experiment: rps_simple_conv_v2_v4
training_config: conf/experiment/rps_simple_conv_v2_v4.yaml
batch: docs/experiments/simpleconv-rps-architecture-search.md
---

# `rps_simple_conv_v2_v4`

## Motivation

Searches SimpleConv encoder / temporal-head architectures for direct RPS prediction from drone audio, offline and under online mixing.

Full batch context: [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).

## Setup

SimpleConvV2 on DREGON-LM-V4 — the reference online-mixing-free RPS baseline (mirrors the historical `train_rps_predictor.py --model simple_conv_v2 --data_root datasets/DREGON-LM-V4` run).

Hydra wiring — data `dregon_lm_v4` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=rps_simple_conv_v2_v4`,
evaluate with `python eval.py experiment=rps_simple_conv_v2_v4`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [SimpleConv RPS architecture search](../../docs/experiments/simpleconv-rps-architecture-search.md).
