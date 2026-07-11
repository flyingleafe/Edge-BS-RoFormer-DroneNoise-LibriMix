# E9 — "Hard" Combined Generated-Noise Task

**Status:** built, running — **Date:** 2026-07-12

## Motivation

Neither generated-noise family transfers to real RPS prediction on its own:

- **E7** (neural generator, vicinal interp): real val PIT MSE ~222–225, R² ≈ −10
  across all three archs — the predictor reverse-engineers the generator's
  amplitude dynamics (a shortcut absent in real data).
- **E8** (analytic static-comb, amplitudes RPS-independent): helped the
  transformer (225→189, R² −10.1→−7.3) but not the smaller heads, and all still
  fail (~10–12 rev/s off vs the real baseline's ~2.7). The synthetic RPS
  distribution roughly matches real (mean ~79), so it is not an RPS-range gap.

So removing the amplitude shortcut is necessary-but-insufficient. E9 makes the
training task **harder** so no single-source shortcut survives:

- **50% neural generator + 50% static-comb** noise (MixedNoisePool, weights
  1:1) — the model must read RPS from both the neural gen's rich, variable
  amplitudes *and* the static comb's amplitude-free combs, so neither source's
  idiosyncratic amplitude→RPS mapping is a reliable shortcut; the common,
  transferable cue is the comb *frequency*.
- LibriSpeech mixtures at −30..0 dB.
- **50% mixture-level augmentation from the start** (`random_gain`,
  `random_polarity`, `channel_drop`) — further discourages amplitude/level
  memorisation.
- **Patience 20** (vs E7/E8's 5/8) — train longer.

Only the transformer is run first (the arch E8 helped). The converged
`last.ckpt` (added to `training.loop` this batch — `best.ckpt` is the early
val-best, wrong for this) is the artifact for the sim→real failure-mode
diagnostic (pred vs GT RPS on a real clip).

## Setup

`e9_hard_transformer` — data `rps_hard_combined`, model
`simple_conv_v2_transformer`, `pit_mse`, `rps`, patience 20,
`samples_per_validation=5000`. Neural-gen source runs a GPU producer.

```bash
omnirun submit --backend colab --gpu-type L4 --gpus 1 --time 4h --yes -- \
  python train.py experiment=e9_hard_transformer \
    data.train.params.path=conf/online_mix/rps_hard_combined_dload.yaml \
    "data.valid.params.data_dir='dload:DREGON-LM-V4-michaels-valid'"
```

## Conclusion

_Pending run._
