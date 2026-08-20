---
experiment: gen_a1_amp_v2
training_config: conf/experiment/gen_a1_amp_v2.yaml
batch: docs/experiments/amplitude-target-training.md
---

# `gen_a1_amp_v2`

## Motivation

`gen_a1_amp` retrained on the linewidth-matched v2 decompositions (fresh experiment name so the v1 checkpoints stay addressable).

## Conclusion

Ran: best val_loss 0.856 at the epoch-13 early stop. Comb-aware offline selection through the render twin pending. See the batch doc.
