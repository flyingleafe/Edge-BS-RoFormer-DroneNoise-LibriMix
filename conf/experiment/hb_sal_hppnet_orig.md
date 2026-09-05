---
experiment: hb_sal_hppnet_orig
training_config: conf/experiment/hb_sal_hppnet_orig.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hb_sal_hppnet_orig`

## Motivation

BLOCK S, LEVEL L0 — HPPNet (Wei et al., ISMIR 2022) AS PUBLISHED, on the
honest-base stream. This row is the bottom rung of the multi-pitch adaptation
ladder of `docs/experiments/paper-regime-matrix.md` § "Block S", and the
control for `conf/experiment/hppnet_*.yaml`, which is the same paper with its
harmonic organ, its front end and its frequency axis all replaced.

The argument is the one in `hb_sal_hf0_orig.md` — read that first. HPPNet
carries a second question HarmoF0 does not: its front end is a CQT, so this arm
also measures whether a true constant-Q analysis rescues the log axis that a
log-interpolated STFT cannot. That has a known cost — a 48-bin-per-octave CQT at
27.5 Hz needs a 2.5 s analysis window at its bottom bin, and the training clips
are 1 s.

Kept whole: the nnAudio CQT, `HarmonicDilatedConv` (eight log-axis dilated
branches at `round(log2(k) * 48)`), `CNNTrunk`, and `FreqGroupLSTM`. Dropped:
the onset, offset and velocity heads, which are note-boundary and MIDI-loudness
events that a rotor trajectory does not have. The two published pools — the
`[1, 4]` frequency pool that takes the grid to a piano roll and the frame
subnet's half-rate time pool — are off, because each would make this arm a
measurement of the pool rather than of the log axis; both stay reachable by a
constructor flag.

Its loss and metrics are the SAME FILES arm 1 uses, because the two models emit
bit-identical 352-bin grids, so the two controls differ only in the trunk and
the front end. The monitor is `bce` for arm 1's reason: HPPNet's frame head is
one shared map, and the rev/s metric reads per-rotor layers.

Full batch context: [The paper regime matrix](../../docs/experiments/paper-regime-matrix.md).

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream overridden to
`conf/online_mix/hb_silence_dload.yaml`) · model `hppnet_orig` · loss
`salience_bce_orig` · metrics `salience_bce_orig`. Train with
`python train.py experiment=hb_sal_hppnet_orig`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [The paper regime matrix](../../docs/experiments/paper-regime-matrix.md).
