---
experiment: m3mixv2_transformer
training_config: conf/experiment/m3mixv2_transformer.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3mixv2_transformer`

## Motivation

Re-runs one-stage mixed training for the IF-front-end transformer with a validation set that matches
the training mixture, because the original verdict measured the stopping rule
rather than the method.

`m3abl_mixed_transformer` scored 103.8 / 6.37 at 25 epochs and stopped there. The same
synthetic family validated on a set containing its own synthetic sources
(`stoch_long_scv2`, MixedRealSynthValidDataset) trained for 228 epochs and
converged. Nine times the training length, from the validation set alone. The
mixed arms were then compared against curriculum arms that had more epochs AND
a full pre-training stage, so "mixed training is worse" was never a controlled
comparison.

Full batch context: [Generator refined labels](../../docs/experiments/generator-refined-labels.md).

## Setup

Hydra wiring — data `m3mix_matchedval` · model `simple_conv_v2_transformer_if` · loss `pit_mse` ·
metrics `rps`. Train with `python train.py experiment=m3mixv2_transformer`.

The training stream is `conf/online_mix/m3abl_mixed_dload.yaml` unchanged: real
50 % (DREGON `in_flight_noise` minus room1, plus FLY125), neural generator 25 %,
static comb 25 %. Validation is `MixtureMatchedValidDataset` at the policy's own
proportions — 296 real (the whole frozen split, never subsampled) + 148
generated + 148 comb = 592 frames.

Monitor is `mae_frame`, not `mse`: val/mse carries 16-55 % relative jitter
between adjacent epochs on these runs against 10-39 % for val/mae_frame.
`patience` is 30 and `epochs` 300, so the run is allowed to converge.

## Conclusion

Pending — the cell is running.
