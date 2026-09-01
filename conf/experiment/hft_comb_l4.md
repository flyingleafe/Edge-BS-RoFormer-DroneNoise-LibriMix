---
experiment: hft_comb_l4
training_config: conf/experiment/hft_comb_l4.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hft_comb_l4`

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

─── THE L4 VARIANT ───────────────────────────────────────────────────────
This row is its `_l4`-less twin with the OUTPUT REPRESENTATION replaced, and
nothing else: same stream, same validation draw, same epoch budget, patience,
batch size, workers and monitor. Three files change together and they must:
model   -> conf/model/*_rps_l4.yaml   (n_maps 4: one salience layer per rotor)
loss    -> conf/loss/salience_layers_r150.yaml     (Gaussian layers, PIT BCE)
metrics -> conf/metrics/salience_layers_r150.yaml  (the same, plus rps_mae)

WHY. `models.salience_crf` encoded real training telemetry into the shared
salience map and decoded it back — a PERFECT target, no model involved — and
got the trajectory back 8.24 rev/s away on average, with 39-45% of frames more
than half a bin off. Gaussian per-rotor layers with the CRF readout return
2.22e-16. The old number is an ORACLE FLOOR: no model reading that
representation could have scored better, so every `_l4`-less row above was
measuring the representation as much as the architecture. This row removes
that ceiling and leaves the architecture question intact.

At eval the decode is `models.harmonic_ports.layer_readout.LayerCRFReadout` —
one CRF best path per layer, NO threshold and NO Hungarian step. A stopped
rotor is the path sitting at bin 0, which is a value; the old decoder had to
call it an absence, which is what forced the threshold.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `sal_comb_synthval` · model `hft_rps_l4` · loss `salience_layers_r150` · metrics `salience_layers_r150`. Train with `python train.py experiment=hft_comb_l4`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
