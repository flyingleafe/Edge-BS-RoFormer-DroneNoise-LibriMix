---
experiment: m3abl_mixed_scv2
training_config: conf/experiment/m3abl_mixed_scv2.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3abl_mixed_scv2`

## Motivation

Ablation 3 of the M3 curriculum for the plain BiGRU baseline
(`simple_conv_v2`): **is the curriculum necessary, or does mixed training with
the same augmentations suffice?** M3CUR pays for two runs and a warm start
(synthetic pre-train, then real fine-tune). This run pools everything at once
(`conf/online_mix/m3abl_mixed_dload.yaml`): the two real sources of m3cur stage
2 carry no explicit weight, so they merge into one duration-weighted pool at
weight 2.0, while the generator and the comb carry `weight: 1.0` each — real :
generator : comb = 2 : 1 : 1, that is 50% / 25% / 25%. The policy block is
m3cur's, unchanged: one stage, freq_scale p=1.0 alpha in [0.7, 1.3],
gain/polarity p=0.5, noise time-warp p=0.5, all from sample 1. There is no
stage 2 and no `checkpoint:`, so a single run per architecture reads directly
against `m3cur_scv2_s2` on the same validation curve. Data `m3abl_mixed`, model
`simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128 frames,
`samples_per_validation=40000`, validation on the fixed FULL-envelope real
split `dload:DREGON-LM-V4-michaels-valid-full`. NOTE: the generated source
needs a CUDA producer context, so this stream does not run on a CPU-only box.
Train: `python train.py experiment=m3abl_mixed_scv2`.

## Conclusion

Best val/mse **>=70.6** (mae_frame 4.93, W&B 0n465pdv) vs the real-only
control 52.5 and the curriculum 28.4 — already 34% above the control when
the run crashed at eval 19 (shm race in the generated-noise pool, fixed in
a98416f). Resubmitted with resume as `m3abl-mixed-scv2-r-ccfc72`; the
direction (mixed WORSE than real-only, staging necessary) matches the other
two architectures.
