---
experiment: r6hb_scv2
training_config: conf/experiment/r6hb_scv2.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# r6hb_scv2 — the stochastic curriculum as stage 1 of the winning mixed recipe

## Motivation

`r4hb_scv2` at **2.67** all-MAE is this project's best result on the frozen
split. It is not a real-only run: it is a comb stage 1 plus a real stage 2. R6
is that recipe with one line changed — the stage-1 checkpoint.

| | R4 (the target) | R6 (this arm) |
|---|---|---|
| stage-1 checkpoint | `m3abl_comb_scv2_s1` | `stoch_s1id_scv2` |
| stage-1 noise family | analytic static comb | stochastic combs |
| real stage-2 stream | `hb_m3s2_dload.yaml` | same |
| trunk / loss / optim / patience | `simple_conv_v2`, `pit_mse` | same |

## The claim being tested

The stochastic campaign's product is a better *initialization*. Before any real
data is seen, on the frozen split:

| stage-1 model | all-MAE | cells at or better than the R4 target |
|---|---|---|
| `stoch_s1id_scv2` | **7.40** | 3 of 6 |
| `m3abl_comb_unigru128_s1` | 8.30 | — |

If a better start survives the fine-tune, R6 beats 2.67. If it does not, the
synthetic stage is fungible past some quality bar and the campaign's gains do
not reach the mixed regime — which is itself worth knowing, because every
stage-1 arm since has been optimizing a quantity that does not transfer.

## Sibling regimes, for the record

- R3 — generator + comb curriculum, stage 2
- R4 — comb-only curriculum, stage 2 (**the target**)
- R5 — mixed ONE-stage: real 45.5%, silence 9.1%, generator 22.7%, comb 22.7%
- R6 — stochastic curriculum, stage 2 (this arm)

The joint counterpart of R6 (R5 with the stochastic family as the synthetic
half) is the natural follow-up if sequential wins, and the more interesting arm
if it does not.

Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Unified baseline evaluation on the frozen validation split](../../docs/experiments/unified-baseline-eval.md).
