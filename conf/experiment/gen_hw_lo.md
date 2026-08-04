---
experiment: gen_hw_lo
training_config: conf/experiment/gen_hw_lo.yaml
batch: docs/experiments/wind-channel-likelihood.md
---

# `gen_hw_lo`

## Motivation

Hybrid objective at spatial weight 0.005, WAKE-GATED wind. At 0.05 the wind channel was
degenerate — 98% of predicted variance on DREGON — so that comparison tested
nothing. Check `scripts/probe_ckpt.py --report variance_share` before reading the scores.

Full batch context: [Wind channel by likelihood](../../docs/experiments/wind-channel-likelihood.md).

## Conclusion

Reported in the batch write-up.
