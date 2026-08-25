---
experiment: hb_fkla
training_config: conf/experiment/hb_fkla.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `hb_fkla`

## Motivation

Plain-KLA row of the unified leaderboard, and the cross-implementation
control for `hb_ckla`. The July CKLA campaign found that rotation is not the
active ingredient: `ckla_norot_fs_v2` (rotation off, else identical) beat
the rotating head on every fs_v2 axis, and the cruise pools put plain KLA at
2.77 (DREGON) / 1.40 (FLY124) next to phase-only 2.79 / 1.29. The Kalman
scan recurrence, not the complex rotation, carries the result.

An in-repo ablation cannot close that question alone, because it disables
rotation inside our own scan and thus inherits any bug of that scan. The
vendored flat-KLA layer answers the same question through an independent
codebase. Agreement between `hb_fkla` and `hb_ckla` validates both
implementations, and disagreement localizes a bug.

The one known cost is speed: the vendored layer ran about 6x slower per step
than our scan, and its fs_v2 run (`fkla_fs_v2`) got 8 epochs in one hour of
wall time and stayed unscored. Give this arm a long slot, not a 1 h slot.

## Setup

The HB recipe of `hb_scv2_mag` — data `e12_real_fullflight` with the R2
honest pool (`conf/online_mix/hb_silence_dload.yaml`), loss `pit_mse`,
metrics `rps`, 200 epochs, patience 20, batch 128, lr 1e-3, weight decay
1e-4, monitor mse, `samples_per_validation` 40000, validation on
`dload:DREGON-LM-V4-michaels-valid-full`.

Model `simple_conv_v2_fkla` — the vendored flat exact-KLA layer
(`src/models/fkla/`, kla-loglinear@11e5a39: real OU state, learned fold
weight, no rotation knob by construction) inside the exact
`SimpleConvV2CKLA` wrapper, with `p_init: 1.0` (the same gain fix as
`hb_ckla`), n_fft 2048, hop 512, 4 rotors. The wrapper keeps its native
`stft_mag_if` front-end, thus no front-end override is applied. Budget
deviations from the HB standard: none. `grad_clip` stays at the 5.0 default;
the flat readout has no `atan2` path, thus the `hb_ckla` NaN risk does not
apply here.

No voicing gate exists on this head, thus the honest comparison cell is
`hb_scv2_mag_nogate`.

Train: `python train.py experiment=hb_fkla`.

## Conclusion

Pending.
