---
experiment: m3abl_gen_scv2_s1
training_config: conf/experiment/m3abl_gen_scv2_s1.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3abl_gen_scv2_s1`

## Motivation

Ablation 2 of the M3 curriculum for the plain BiGRU baseline
(`simple_conv_v2`): **does the analytic comb add anything on top of the neural
generator?** M3CUR stage 1 keeps the static comb at 50% as a spacing-only
anchor with no timbre model; this run removes it, so
`gen_m3_refined_all_perrotor` (ep30, full-flight RPS, `perrotor_noise: 0.5`,
`rotor_jitter_std: 0.2`) is the whole synthetic pool
(`conf/online_mix/m3abl_gen_s1_dload.yaml`). Sample rate, duration, seed,
speech source and the single-stage policy (freq_scale p=1.0 alpha in [0.7,
1.3], gain/polarity p=0.5, noise time-warp p=0.5, all from sample 1) are m3cur
verbatim, so the comb is the only difference. Stage 2 (`m3abl_gen_scv2_s2`) is
the SAME real fine-tune as m3cur. If the pair matches `m3cur_scv2_s1/s2`, stage
1 can drop the comb half. Data `m3abl_gen_s1`, model `simple_conv_v2`, loss
`pit_mse`, metrics `rps`, batch 128 frames, `samples_per_validation=40000`,
validation on the fixed FULL-envelope real split
`dload:DREGON-LM-V4-michaels-valid-full`. NOTE: the generated source needs a
CUDA producer context, so this stage-1 stream does not run on a CPU-only box.
Train: `python train.py experiment=m3abl_gen_scv2_s1`.

## Conclusion

Pending.
