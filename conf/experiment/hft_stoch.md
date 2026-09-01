---
experiment: hft_stoch
training_config: conf/experiment/hft_stoch.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hft_stoch`

## Motivation

ARM B — hFT-Transformer (ported) on the STOCHASTIC rotor-noise curriculum,
validated on a HELD-OUT DRAW OF THAT SAME FAMILY.

THE QUESTION, and why it is not arm A's. The static comb of `hft_comb` has
lines with no width at all. The stochastic family is the widest synthetic
family the project has: per-harmonic amplitude wander, a broadband floor that
does not follow the rotors, and — the one knob measured to move comb contrast
monotonically — lines that broaden with their own order
(docs/experiments/synthetic-solvability-limits.md). Arm A says whether the
architecture can read a comb; arm B says whether it can still read one when
the lines are not delta functions, which is the property the real recordings
actually have.

Stream conf/online_mix/stoch_s1_dload.yaml, validation a fresh base_seed on
the same policy at 8 s and unaugmented (conf/data/sal_stoch_synthval.yaml) —
`sal150_stoch`'s pairing, unchanged, so this row and the LateDeep salience
row differ in the MODEL and nothing else.

batch_size 8, not 16: the cross-attention map is
(batch, frame, head, rate, harmonic) and is linear in the batch.
`grad_accum_steps: 2` restores the effective batch to sal150_stoch's 16
frames — 8 frames is one chunk's eight microphone rows, i.e. eight views of
one acoustic scene, which is not a batch.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `sal_stoch_synthval` · model `hft_rps` · loss `salience_bce_r150` · metrics `salience_bce_r150`. Train with `python train.py experiment=hft_stoch`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
