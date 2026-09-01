---
experiment: hf0_comb
training_config: conf/experiment/hf0_comb.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hf0_comb`

## Motivation

ARM A — harmof0_rps on the analytic STATIC-COMB curriculum, validated on a
held-out draw of the same family.

This is conf/experiment/sal150_comb.yaml with the MODEL swapped and nothing
else. That row is multif0_salience (a LateDeep CNN over an HCQT) on a 0-150
rev/s super-resolution output grid; this row is HarmoF0 with its
log-frequency harmonic shift replaced by a gather at k*r on the linear STFT
(conf/model/harmof0_rps.yaml). Stream, validation draw, epoch budget,
patience, batch size, worker count, monitor and samples_per_validation are
sal150_comb's verbatim, so the pair isolates the ARCHITECTURE.

The output grid differs in resolution only (300 bins at 0.5017 rev/s against
1000 at 0.1502) and the reason is in conf/model/harmof0_rps.yaml: the design
note's error table makes 0.5 rev/s cost 0.013-0.13 rev/s of discretization,
under the campaign's 0.2 rev/s honest floor, and here every bin costs a
gather.

WHY THIS ARM EXISTS. The static comb is the family the gather is exactly
right for: lines at k*r and nothing else. If a harmonic-gather architecture
cannot beat an HCQT CNN here, the substitution buys nothing anywhere.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `sal_comb_synthval` · model `harmof0_rps` · loss `salience_bce_r150` · metrics `salience_bce_r150`. Train with `python train.py experiment=hf0_comb`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
