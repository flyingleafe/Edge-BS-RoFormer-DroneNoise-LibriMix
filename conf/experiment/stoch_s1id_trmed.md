---
experiment: stoch_s1id_trmed
training_config: conf/experiment/stoch_s1id_trmed.yaml
batch: docs/experiments/stochastic-transfer.md
---

# stoch_s1id_trmed — a 4 x 128 transformer head (2.21M params) on arm ID's stochastic stream

## Motivation

Arm ID's stream and `simple_conv_v2`'s encoder, with the BiGRU temporal head
replaced by a 4 x 128 transformer head (2.21M params).

## Why the arm exists

Transformers have failed on every synthetic stage 1 this project has run, but
always on a NARROW noise family: `m3abl_comb_transformer_s1` reached 1802.0
validation PIT-MSE on the analytic static comb and `m3cur_transformer_s1`
reached 316.9 on generator + comb, against `simple_conv_v2`'s 204.0 and 325.5
on the same two streams. The recorded reading was that a comb-only stage 1
kills the transformer and helps the conv trunk.

The stochastic family is far wider than either, so the failure may have been
the data and not the architecture. These arms test that, and they bracket
capacity so a result can be attributed:

| arm | temporal head | params |
|---|---|---|
| `stoch_s1id_tr` | 2 x 64, 4 heads (stock) | 1.48M |
| `stoch_s1id_trmed` | 4 x 128, 8 heads | 2.21M |
| `stoch_s1id_trbig` | 6 x 256, 8 heads | 6.24M |

The reference row is `stoch_s1id_scv2` (the BiGRU head) at **7.40** all-MAE,
the campaign's best synthetic-only model.

## Reading the result

- All three near or above the BiGRU: the head is not the bottleneck and the
  earlier transformer failures were about capacity being wasted, not missing.
- Stock poor, heavier better: the earlier failures were an underfitting head on
  a narrow family, and capacity is worth spending here.
- All three poor: the transformer failure is architectural and independent of
  the noise family, which closes the question for this task.

Policy: `conf/online_mix/stoch_s1id_dload.yaml`.
Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Synthetic-only transfer with the stochastic noise family](../../docs/experiments/stochastic-transfer.md).
