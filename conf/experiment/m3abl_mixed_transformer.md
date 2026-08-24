---
experiment: m3abl_mixed_transformer
training_config: conf/experiment/m3abl_mixed_transformer.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3abl_mixed_transformer`

## Motivation

Ablation 3 of the M3 curriculum for the IF front-end transformer
(`simple_conv_v2_transformer_if`): **is the curriculum necessary, or does mixed
training with the same augmentations suffice?** M3CUR pays for two runs and a
warm start (synthetic pre-train, then real fine-tune). This run pools
everything at once (`conf/online_mix/m3abl_mixed_dload.yaml`): the two real
sources of m3cur stage 2 carry no explicit weight, so they merge into one
duration-weighted pool at weight 2.0, while the generator and the comb carry
`weight: 1.0` each — real : generator : comb = 2 : 1 : 1, that is 50% / 25% /
25%. The policy block is m3cur's, unchanged: one stage, freq_scale p=1.0 alpha
in [0.7, 1.3], gain/polarity p=0.5, noise time-warp p=0.5, all from sample 1.
There is no stage 2 and no `checkpoint:`, so a single run per architecture
reads directly against `m3cur_transformer_s2` on the same validation curve.
Data `m3abl_mixed`, model `simple_conv_v2_transformer_if`, loss `pit_mse`,
metrics `rps`, batch 128 frames, `samples_per_validation=40000`, validation on
the fixed FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
NOTE: the generated source needs a CUDA producer context, so this stream does
not run on a CPU-only box. Train: `python train.py
experiment=m3abl_mixed_transformer`.

## Conclusion

Best val/mse **103.8** (mae_frame 6.37, W&B 6jbs95vr, 25 evals) vs the
real-only control 42.3 and the curriculum 38.6. Mixed one-stage training is
2.5x WORSE than real-only: the curriculum staging is necessary. Verdict
details in the batch doc.
