---
experiment: hppnet_stoch
training_config: conf/experiment/hppnet_stoch.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hppnet_stoch`

## Motivation

ARM B — hppnet_rps on the STOCHASTIC rotor-noise curriculum, validated on a
held-out draw of that same family.

The stochastic twin of conf/experiment/hppnet_comb.yaml, and the same argument
applies verbatim: the row measures whether the port can fit this family, so
the validation draw comes from the family and not from the real split
(conf/data/sal_stoch_synthval.yaml's header). The stream is
conf/online_mix/stoch_s1_dload.yaml — one `stochastic` source at full-flight
excitation plus a 0.2-weight silence arm, the widest synthetic family the
project has, and the family the static comb of arm A is the sharp-line corner
of.

Read against conf/experiment/sal150_stoch.yaml, which is this row with
multif0_salience in place of the port (see hppnet_comb.yaml for the three
things that differ). The A-to-B drop measures what line width costs the port;
the port-to-multif0 difference measures what the gather buys.

Full batch context: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).

## Setup

Hydra wiring — data `sal_stoch_synthval` · model `hppnet_rps` · loss `salience_bce_r150` · metrics `salience_bce_r150`. Train with `python train.py experiment=hppnet_stoch`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
