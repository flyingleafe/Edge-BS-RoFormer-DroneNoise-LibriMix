---
experiment: hppnet_comb
training_config: conf/experiment/hppnet_comb.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hppnet_comb`

## Motivation

ARM A — hppnet_rps on the analytic STATIC-COMB curriculum, validated on a
held-out draw of that same family.

THE QUESTION. docs/harmonic-ports-design.md argues that the multi-pitch
architectures fail here for a structural reason — they gather harmonic
evidence on a log axis, where the separation-to-bandwidth ratio of two rotors
is independent of the harmonic index, so a close pair is unresolvable at every
harmonic at once. This row asks whether removing that one organ is enough:
the model is HPPNet's trunk and its FreqGroupLSTM with the harmonic gather
moved onto the linear STFT, and the task is the easiest harmonic family the
project has.

THE PAIRING is conf/experiment/sal150_comb.yaml exactly — same stream
(conf/online_mix/m3abl_comb_s1_dload.yaml through conf/data/sal_comb_synthval.yaml),
same held-out-same-family validation, same epoch size, same batch, same
monitor. Three things differ, and all three are the port:
1. The model, from multif0_salience (LateDeep CNN over HCQT) to hppnet_rps.
2. The grid, from 0-150 in 1000 bins to 0-150 in 300 bins (0.5 rev/s — the
design note's error table says the finer grid buys nothing once the
peak is log-parabola-interpolated).
3. The frame rate, from [16000, 256] (the HCQT hop) to [16000, 512] (this
project's STFT hop, which is what the port's front end runs on).
So sal150_comb is the reference this row is read against, and hb_sal_multif0
(flight MAE 4.01 on real data) is the reference sal150_comb was read against.

WHY SAME-FAMILY VALIDATION: conf/data/sal_comb_synthval.yaml's header. The row
asks whether the architecture can learn the task, not whether it transfers,
so halting on a real-split metric would confound the two.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `sal_comb_synthval` · model `hppnet_rps` · loss `salience_bce_r150` · metrics `salience_bce_r150`. Train with `python train.py experiment=hppnet_comb`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
