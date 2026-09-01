---
experiment: hft_comb
training_config: conf/experiment/hft_comb.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hft_comb`

## Motivation

ARM A — hFT-Transformer (ported) on the ANALYTIC STATIC-COMB curriculum,
validated on a HELD-OUT DRAW OF THAT SAME FAMILY.

THE QUESTION. `docs/harmonic-ports-design.md` claims that a multi-pitch
architecture whose harmonic organ is replaced by a gather at k*r on a linear
STFT can read rotor rates. This row asks the narrowest version of that: can
the architecture fit the CLEANEST harmonic task the project has — one fixed
amplitude profile per clip, comb spacing the only cue — at all. A real-split
number would confound "cannot learn the task" with "does not transfer", and
the campaign has measured that those call for opposite fixes.

Stream conf/online_mix/m3abl_comb_s1_dload.yaml, validation a fresh base_seed
on the same policy at 8 s and unaugmented (conf/data/sal_comb_synthval.yaml).
That pairing is `sal150_comb`'s, unchanged, so this row and the LateDeep
salience row differ in the MODEL and nothing else.

THIS RUN IS ALSO STAGE 1 OF ARM C. The regime taxonomy's R4 is the comb-only
curriculum, stage 2 (docs/experiments/unified-baseline-eval.md), and its
stage 1 is exactly this policy — `r4hb_gru` warm-starts from
`m3abl_comb_unigru128_s1`, which trained on it. `hft_r4` warm-starts from
this run's best.ckpt for the same reason.

Trunk-independent settings (optimizer, monitor, epoch size, clip length) are
`sal150_comb`'s. batch_size is 8 frames rather than 16: the cross-attention
map is (batch, frame, head, rate, harmonic) and its size is linear in every
one of those, so the transformer pays for batch where a CNN does not.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `sal_comb_synthval` · model `hft_rps` · loss `salience_bce_r150` · metrics `salience_bce_r150`. Train with `python train.py experiment=hft_comb`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
