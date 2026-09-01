---
experiment: hf0_r4_l4
training_config: conf/experiment/hf0_r4_l4.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hf0_r4_l4`

## Motivation

ARM C, faithful R4 — conf/experiment/hf0_real.yaml plus the warm start that
makes it the R4 regime cell rather than a plain real-data row.

R4 in docs/experiments/unified-baseline-eval.md's taxonomy is "comb-only
curriculum, stage 2": pre-train on the analytic static comb alone, then
fine-tune on the honest real pool conf/online_mix/hb_m3s2_dload.yaml. The
regression cells (r4hb_scv2 / r4hb_tr / r4hb_gru) warm-start from
m3abl_comb_<trunk>_s1; those checkpoints are simple_conv_v2-family trunks and
cannot be loaded into a salience model, so the salience stage 1 has to be
built here — and it is exactly arm A (conf/experiment/hf0_comb.yaml), which
trains on the same conf/online_mix/m3abl_comb_s1_dload.yaml policy.

THIS ROW THEREFORE DEPENDS ON ARM A: submit it only after hf0_comb has
written a best.ckpt to the artifact store.

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

Hydra wiring — data `m3cur_s2` · model `harmof0_rps_l4` · loss `salience_layers_r150` · metrics `salience_layers_r150`. Train with `python train.py experiment=hf0_r4_l4`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
