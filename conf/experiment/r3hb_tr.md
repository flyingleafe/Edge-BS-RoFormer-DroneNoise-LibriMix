---
experiment: r3hb_tr
training_config: conf/experiment/r3hb_tr.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `r3hb_tr`

## Motivation

Regime cell R3 of the training-regime taxonomy — the generated + comb
curriculum — at the fixed original architecture, the IF front-end transformer
(`simple_conv_v2_transformer_if`). Stage 1 is unchanged: this run warm-starts from
`m3cur_transformer_s1`'s `best.ckpt`, the same checkpoint the as-run
`m3cur_transformer_s2` used, so the synthetic pre-training is reused, not repeated.
What differs from the as-run original is the real fine-tune stream: it becomes
the R2 honest-base pool (`conf/online_mix/hb_m3s2_dload.yaml` — the zero-labeled
silence arm at weight 0.4 and the SNR reference floor `snr_ref_floor_rms: 0.02`,
with the 50k warm-up stage removed) in place of the plain fs_v2 pool.
The rerun therefore isolates the synthetic ingredient: R3 and R2
now share one real component, and only m3cur stage 1 (the M3 per-rotor generator plus the analytic static comb) separates them.
Validation stays the fixed full-envelope real split
`dload:DREGON-LM-V4-michaels-valid-full`. Train:
`python train.py experiment=r3hb_tr`.

## Conclusion

Pending.
