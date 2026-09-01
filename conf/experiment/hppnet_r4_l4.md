---
experiment: hppnet_r4_l4
training_config: conf/experiment/hppnet_r4_l4.yaml
batch: docs/experiments/harmonic-multipitch-ports.md
---

# `hppnet_r4_l4`

## Motivation

ARM C — hppnet_rps on the R4 regime: the comb-only curriculum, stage 2, on
the R2 honest real base. Validated on the frozen REAL split.

WHAT R4 IS. The regime taxonomy of docs/experiments/unified-baseline-eval.md
§ "Regime-matched reruns": R2 real-only honest, R3 gen+comb curriculum, R4
COMB-ONLY CURRICULUM, R5 mixed one-stage. R4 is two stages — a synthetic
static-comb stage 1, then a real fine-tune on the R2 honest pool
(conf/online_mix/hb_m3s2_dload.yaml: the real full-envelope pool with the
zero-labeled silence arm and the SNR reference floor, warm-up stage removed).
The as-run rows are r4hb_scv2 / r4hb_tr / r4hb_gru, and this row is r4hb_gru
with the trunk swapped for the port. It is NOT conf/experiment/ladder_r4_scv2.yaml,
which is rung 4 of the line-width ladder and is synthetic throughout — that
one cannot be "the real-data curriculum".

STAGE 1 IS ARM A. r4hb_* warm-start from `m3abl_comb_unigru128_s1`, the
static-comb stage-1 checkpoint of their own architecture. The port's
equivalent is conf/experiment/hppnet_comb.yaml, which trains on the SAME
policy (conf/online_mix/m3abl_comb_s1_dload.yaml), so this row must be
submitted after that one finishes and its best.ckpt is on R2.

WHAT DIFFERS FROM conf/experiment/r4hb_gru.yaml: the model/loss/metrics
(salience on the port's 0-150 rev/s grid instead of PIT-MSE regression), the
batch size (16 frames, the salience rows' value, against the regression rows'
128 — the trunk is heavier per frame), and samples_per_validation (16000, as
in the port's synthetic arms). The data config is m3cur_s2 with its train
stream replaced by the R2 honest pool, which is exactly what r4hb_gru does;
the validation half is m3cur_s2's own frozen real split
dload:DREGON-LM-V4-michaels-valid-full, untouched.

THE REFERENCES this row is read against, both on that split: the salience
family's best real row hb_sal_multif0 at flight MAE 4.01, and the best
regressor r3hb_gru at 2.79.

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

Hydra wiring — data `m3cur_s2` · model `hppnet_rps_l4` · loss `salience_layers_r150` · metrics `salience_layers_r150`. Train with `python train.py experiment=hppnet_r4_l4`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Harmonic multi-pitch architectures ported to the linear STFT](../../docs/experiments/harmonic-multipitch-ports.md).
