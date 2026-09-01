# SALV2 — rebuilt synthetic streams, the speech A/B, and the CRF objective

**Status:** running (submitted 2026-09-01)
**Configs:** `conf/experiment/salv2_*.yaml` (20 cells)
**Streams:** `conf/online_mix/salv2_{comb,stoch}.yaml` · `conf/data/salv2_*.yaml`

## Why this batch exists

The l4 port arms showed `val/bce` rising while `val/rps_mae` fell, on an ONLINE
training stream that never repeats a sample. Memorization cannot explain that,
so the cause had to be elsewhere. Reading the configs found three defects, all
measured rather than argued:

1. **Train and valid were never the same distribution.** The training policy
   fired `freq_scale` (alpha ~ U(0.7, 1.3), which resamples the audio AND
   multiplies the RPS labels) on EVERY sample and `noise_time_warp` on half of
   them. The fixed validation set was built with `augment: false`, and that flag
   DELETES all three augmentation blocks. The two loss curves were therefore
   never comparable, in level or in direction.

2. **The validation set was one flight.** `FixedSynthFrameDataset`'s `n` counts
   frames AFTER the per-mic `flat_map`, so `n: 96` over 8 microphones is 12
   clips — and 12 is fewer than the policy's `flight_reuse: 32`, so all 12 were
   windows of a SINGLE trajectory, each served 8 times. `comb_bench.py` had
   already recorded this defect for the sibling config.

3. **The static comb had no stopped rotors.** Over 60 clips the zero fraction
   was 0.001 and no clip was majority-zero, because a 4 s window lands in a full
   flight's ground phase only rarely and `synthetic_intermittent` is cruise-only.
   The stochastic family was fine — it draws zeros from its `silence` arm.

## What changed

`conf/online_mix/salv2_{comb,stoch}.yaml` replace the old policies:

- No `freq_scale`, no `noise_time_warp`. Trajectory diversity comes from the
  generator instead, where validation gets it too.
- Two excitations, half and half: `full_flight` (ground → warm-up → takeoff →
  cruise → landing → ground) and `synthetic_intermittent` (the cruise-only OU
  maneuvering model).
- A `silence` arm at weight 0.2 in BOTH families, so the comb arm carries
  stopped rotors: zero fraction 0.001 → 0.183, majority-zero clips 0 → 11/60.
- `rps_scale_range: [0.45, 1.20]` as the exact replacement for `freq_scale`. It
  scales the trajectory at GENERATION time, so the comb is rendered at the
  scaled speed and the label stays exact. 1.20 is the largest factor that cannot
  put a label off the 150 rev/s grid given the 120 rev/s profile ceiling.
  Measured: clip means 32–96 rev/s, max label 105, per-frame slew max 2.8 rev/s.
- Training clips 4 s, validation clips 8 s, `flight_reuse: 1` in validation, so
  32 clips are 32 distinct flights (25–29 of them distinct after the silence arm
  contributes its identical all-zero label).

**Speech is a switch on ONE policy**, not a copied file
(`apply_speech_override`). At a fixed seed the two conditions come back with
byte-identical labels and byte-identical rotor noise — verified — so
`SpeechPairedSynthValidDataset` scores the same 32 flights twice, quiet and with
a talker, and the speech contrast carries no trajectory variance.

## The grid

{HarmoF0, HPPNet, SimpleConvV2} × {static comb, stochastic} × {without speech,
with speech} = 12 cells, plus a CRF-objective arm for the two salience ports
(2 × 2 × 2 = 8). 20 in total.

**Batch size 32**, measured on an A100-80 at 4 s clips (peak allocated):

| model | bce / mse | crf @ 25 |
|---|---|---|
| harmof0 | 6.09 GiB | 12.46 GiB |
| hppnet | 17.18 GiB | 17.18 GiB |
| scv2 | 4.75 GiB | — |

HPPNet binds at every objective: its `FreqGroupLSTM` runs `B*G` sequences and
its peak does not move when the loss changes, so the CRF is invisible against
it. 32 is the largest power of two that keeps the worst cell inside a 24 GB
card. Because the CRF is affordable there, `max_step_rev_s` stays at the
decoder's 25.0 rather than being trimmed to 8.0.

**Monitor is PIT per-frame MAE in rev/s** (`rps_mae` for the ports,
`mae_frame` for the regressor), NOT the training objective — see defect (1):
cross-entropy is unbounded and punishes a confident near-miss that a bounded
argmax error barely notices, so selecting on it discards better models.

## The CRF arm

`losses.LayerPITCRFLoss` is `log Z - score(gold)` under the same emissions
(`log sigmoid(z)`) and the same hinge band `layer_readout.LayerCRFReadout`
decodes with, so training, selection and deployment become one object. The
campaign has already paid for that gap once: the trained head's advantage did
not survive the switch from an argmax decoder to a Viterbi decoder (`close`
0.459 → 0.669). `log Z` is path-independent, so the R × R assignment costs R
forward passes, not R². Its test locks the loss's gold path to the decoder's
Viterbi output on a perfect layer.

## Conclusion

Pending — the cells are running.
