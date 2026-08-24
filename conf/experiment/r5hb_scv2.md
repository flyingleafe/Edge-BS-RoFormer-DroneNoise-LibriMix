---
experiment: r5hb_scv2
training_config: conf/experiment/r5hb_scv2.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `r5hb_scv2`

## Motivation

Regime cell R5 of the training-regime taxonomy — mixed one-stage training, no
curriculum — at the fixed original architecture, the plain BiGRU baseline
(`simple_conv_v2`). What differs from the as-run `m3abl_mixed_scv2` is the real
component of the pool: `conf/online_mix/hb_m3mixed_dload.yaml` adds the R2
honest ingredients (the zero-labeled silence arm and the SNR reference floor
`snr_ref_floor_rms: 0.02`) to the same real recordings, and keeps both original
ratios — real : generated : comb = 2 : 1 : 1 and silence : real = 0.4 : 2.0.
The shares become real 45.5%, silence 9.1%, generated 22.7%, comb 22.7%.
The rerun makes the no-curriculum control comparable with R2/R3/R4: all four
cells share one real component, thus a difference between the rows is a
difference in the synthetic ingredient and its schedule, not in the real data.
There is no stage 2 and no warm start. NOTE: the generated source needs a CUDA
producer context, thus this stream does not run on a CPU-only box.
Validation stays the fixed full-envelope real split
`dload:DREGON-LM-V4-michaels-valid-full`. Train:
`python train.py experiment=r5hb_scv2`.

## Conclusion

Pending.
