---
experiment: ckla_p1_freqscale
training_config: conf/experiment/ckla_p1_freqscale.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p1_freqscale`

## Motivation

Mechanistic lever derived from the activation analysis
(`docs/experiments/ckla-activation-analysis.md` §A6): both trained
architectures ignore a ×1.02 frequency scaling of the input (~0.05%
prediction response vs the ideal 2%) — they are amplitude/timbre-anchored
to the training RPS prior and do not read comb spacing at the margin. This
is a structural gap to VK (which reads spacing by construction) shared by
the whole model family. The freq_scale augmentation (noise+RPS pair
rescaled together, labels ×α) is the one transform that makes
spacing-reading necessary for low loss; G6 only ever tested it inside a
6-transform bundle (net negative). Solo, α ∈ [0.85, 1.18], p = 0.5, on the
CKLA head.

## Setup

Exact clone of `ckla_p1_if` with the train policy →
`conf/online_mix/e12_fullflight_freqscale_dload.yaml` (baked into the
experiment config). Post-training: vk_eval + rerun the §A6 scale-response
probe — success = response moves toward the ideal α line AND cruise MAE
improves.

Train: `python train.py experiment=ckla_p1_freqscale`.

## Conclusion

_Pending run._
