---
experiment: gen_h1_hybrid_wind
training_config: conf/experiment/gen_h1_hybrid_wind.yaml
batch: docs/experiments/wind-channel-likelihood.md
---

# `gen_h1_hybrid_wind`

## Motivation

Hybrid objective (marginal Rice + cross-microphone coherence), multi-observer,
WAKE-GATED wind. Keeps the coherent fit that the pure spatial objective destroyed while
still giving the wind channel a gradient it can earn.

Full batch context: [Wind channel by likelihood](../../docs/experiments/wind-channel-likelihood.md).

## Conclusion

Reported in the batch write-up.
