---
experiment: m3abl_comb_scv2_s1
training_config: conf/experiment/m3abl_comb_scv2_s1.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3abl_comb_scv2_s1`

## Motivation

Ablation 1 of the M3 curriculum for the plain BiGRU baseline
(`simple_conv_v2`): **does the neural generator matter for the pre-training
stage, or does the analytic comb alone suffice?** M3CUR stage 1 pre-trains on
50% `gen_m3_refined_all_perrotor` (ep30) plus 50% analytic static comb; this
run removes the generator half, so the comb is the whole synthetic pool
(`conf/online_mix/m3abl_comb_s1_dload.yaml`). Sample rate, duration, seed,
full-flight excitation, speech source and the single-stage policy (freq_scale
p=1.0 alpha in [0.7, 1.3], gain/polarity p=0.5, noise time-warp p=0.5, all from
sample 1) are m3cur verbatim, so the generator is the only difference. Stage 2
(`m3abl_comb_scv2_s2`) is the SAME real fine-tune as m3cur, which makes the
pair read against `m3cur_scv2_s1/s2` as a controlled test of the generator's
contribution. Data `m3abl_comb_s1`, model `simple_conv_v2`, loss `pit_mse`,
metrics `rps`, batch 128 frames, `samples_per_validation=40000`, validation on
the fixed FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=m3abl_comb_scv2_s1`.

## Conclusion

Pending.
