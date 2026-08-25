---
experiment: stoch_s1q_colab
training_config: conf/experiment/stoch_s1q_colab.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1q_colab`

## Motivation

Arm P's noise pool on the causal unidirectional-GRU trunk instead of the
bidirectional one. Everything about the data is identical; the trunk is the
only difference.

The trunk has prior evidence behind it, which is more than the last several
noise-model levers had. Across the old comb-only family the causal trunk was
the better synthetic-only transferrer on every axis that matters here: 183.7
against 204.0 on the aggregate, and — the cell this campaign is furthest from —
a stopped-rotor mean absolute error of 4.73 against the convolutional trunk's
5.64, the best of any synthetic-only model measured. Its ramp cell was also the
better of the two, 24.24 against 26.32.

That pattern is worth taking seriously for a transfer problem. A bidirectional
head sees a whole window before answering and can settle on a window-level
average; a causal one answers from what has arrived, which is a harder task on
the training family and a more local one on a recording it has never heard.

Data `stoch_s1p` (arm G's pool plus the recording floor — the one change arm L
proved), model `simple_conv_v2_uni_gru128`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1q_gru`.

## Conclusion

PENDING — the run has not finished.


A copy of `stoch_s1q_gru` run on the Colab backend, identical except for the
run name, submitted because the cluster GPU partition was saturated. It exists
as its own config so the checkpoint can be scored per regime by name.
