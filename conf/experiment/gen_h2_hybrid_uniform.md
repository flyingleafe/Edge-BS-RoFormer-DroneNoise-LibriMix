---
experiment: gen_h2_hybrid_uniform
training_config: conf/experiment/gen_h2_hybrid_uniform.yaml
batch: docs/experiments/wind-channel-likelihood.md
---

# `gen_h2_hybrid_uniform`

## Motivation

Hybrid objective (marginal Rice + cross-microphone coherence), multi-observer,
UNIFORM-exposure control (gate removed, same capacity). Keeps the coherent fit that the pure spatial objective destroyed while
still giving the wind channel a gradient it can earn.

Full batch context: [Wind channel by likelihood](../../docs/experiments/wind-channel-likelihood.md).

## Conclusion

Reported in the batch write-up.
