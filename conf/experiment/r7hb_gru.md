---
experiment: r7hb_gru
training_config: conf/experiment/r7hb_gru.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `r7hb_gru`

## Motivation

Regime cell R7 — the mixed one-stage pool with the stochastic family as the
synthetic half — on the causal unidirectional-GRU head (`simple_conv_v2_uni_gru128`). This file is
`r7hb_scv2` with the model swapped and nothing else changed.

The published R7 row measured only `simple_conv_v2`. The regime matrix reads
every regime on the same three trunks, so the two remaining trunks are filled
in here. The pool stays `conf/online_mix/hb_stochmixed_dload.yaml` — real
45.5%, the honest silence arm 9.1%, the stochastic family 22.7% and the
analytic static comb 22.7% — with one stage and no warm start.

Train: `python train.py experiment=r7hb_gru`.

## Conclusion

Pending.
