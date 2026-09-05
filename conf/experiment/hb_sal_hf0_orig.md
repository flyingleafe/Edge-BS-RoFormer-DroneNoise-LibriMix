---
experiment: hb_sal_hf0_orig
training_config: conf/experiment/hb_sal_hf0_orig.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hb_sal_hf0_orig`

## Motivation

BLOCK S, LEVEL L0 — HarmoF0 (Wei et al., ISMIR 2022) AS PUBLISHED, on the
honest-base stream. This row is the bottom rung of the multi-pitch adaptation
ladder of `docs/experiments/paper-regime-matrix.md` § "Block S" (L0 published,
L1 a finer output grid, L2 per-rotor layers with the CRF readout, L3 the comb
gather on the linear STFT), and the control for `conf/experiment/hf0_*.yaml`,
which is the same paper with its harmonic organ and its frequency axis both
replaced.

The ports rest on one claim: that HarmoF0's log-axis harmonic SHIFT has to
become a gather at `k*r` on a LINEAR STFT, because a log grid's
separation-to-bandwidth ratio for two rotors `D` apart is
`D / (r * (2^(1/B) - 1))`, in which the harmonic index cancels
(`docs/harmonic-ports-design.md`). Every port row so far has been read against
the REGRESSORS, which share neither the architecture nor the representation, so
the substitution and the trunk have never been separated. This row holds the
trunk fixed and puts the paper's own axis back: `WaveformToLogSpecgram`,
`MRDConv`, the octave-dilated blocks 2-4, and a 352-bin map at 48 bins per
octave from 27.5 Hz.

It is `conf/experiment/hb_sal_multif0.yaml` with the model, loss and metrics
swapped for that grid, and nothing else — the same honest-base online stream,
the same frozen real validation split, the same epoch budget, patience, batch
size, workers and monitor. So it is directly comparable with `hb_sal_multif0`
and `hb_sal_bp`, the other shared single-map salience rows.

The monitor is `bce` and that is forced: `metrics.SalienceBCEMetric` is the
whole shared-map metric surface, and the per-rotor PIT metric
`metrics.LayerPeakRPSMetric` needs the four salience layers that a monophonic
HarmoF0 does not emit. Rotor-speed error comes from `eval.py`'s PIT suite over
`predict_rps` afterwards.

Full batch context: [The paper regime matrix](../../docs/experiments/paper-regime-matrix.md).

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream overridden to
`conf/online_mix/hb_silence_dload.yaml`) · model `harmof0_orig` · loss
`salience_bce_orig` · metrics `salience_bce_orig`. Train with
`python train.py experiment=hb_sal_hf0_orig`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [The paper regime matrix](../../docs/experiments/paper-regime-matrix.md).
