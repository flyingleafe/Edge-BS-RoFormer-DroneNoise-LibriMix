---
experiment: m3abl_comb_transformer_s2
training_config: conf/experiment/m3abl_comb_transformer_s2.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3abl_comb_transformer_s2`

## Motivation

Ablation 1, stage 2 for the IF front-end transformer
(`simple_conv_v2_transformer_if`): the real fine-tune that closes the **does
the generator matter?** question. Warm-started from
`m3abl_comb_transformer_s1`'s `best.ckpt`, it trains on the SAME real stream as
`m3cur_transformer_s2` (`conf/online_mix/m3cur_s2_dload.yaml`: DREGON
`in_flight_noise` minus `free-flight_nosource_room1` plus Michael's FLY125,
both `min_motor_rps: 0.0`, LibriSpeech train-clean-100, one policy stage with
every augmentation from sample 1). Because only the stage-1 pool differs, the
gap between this run and `m3cur_transformer_s2` is what the neural generator
adds over the comb alone. Data `m3cur_s2`, model
`simple_conv_v2_transformer_if`, loss `pit_mse`, metrics `rps`, renewed
patience 20, batch 128 frames, `samples_per_validation=40000`, same fixed
FULL-envelope real validation split. Train: `python train.py
experiment=m3abl_comb_transformer_s2` (after stage 1 uploads best.ckpt).

## Conclusion

Pending.
