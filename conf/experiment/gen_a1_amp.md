---
experiment: gen_a1_amp
training_config: conf/experiment/gen_a1_amp.yaml
batch: docs/experiments/amplitude-target-training.md
---

# `gen_a1_amp`

## Motivation

Amplitude-target arm 1: per-drone codebook, DREGON + Michael's, 8 mics. The A/B partner of `gen_m1_refined` — same data and architecture, but the objective fits the Vold-Kalman amplitude envelopes (`decomp-frames-v1`) instead of a multi-scale STFT distance on rendered audio.

## Conclusion

Ran (v1 decompositions; ep10 comb readout in the batch doc). The v1 targets carry the flat-bandwidth underestimate, so the v2 arms supersede this run. See the batch doc for numbers.
