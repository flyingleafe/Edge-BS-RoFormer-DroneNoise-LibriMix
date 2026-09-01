---
experiment: hppnet_stoch_l4
training_config: conf/experiment/hppnet_stoch_l4.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hppnet_stoch_l4`

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

Hydra wiring — data `sal_stoch_synthval` · model `hppnet_rps_l4` · loss `salience_layers_r150` · metrics `salience_layers_r150`. Train with `python train.py experiment=hppnet_stoch_l4`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
