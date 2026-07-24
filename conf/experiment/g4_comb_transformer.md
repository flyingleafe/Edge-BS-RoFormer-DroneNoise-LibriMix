---
experiment: g4_comb_transformer
training_config: conf/experiment/g4_comb_transformer.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g4_comb_transformer`

## Motivation

Phase G4 of the VK-parity push (campaign criterion 2.3). The G2 front-end
arms established two facts on the protocol eval: (a) IF phase evidence is
directionally right — `g2_if` was the first arm to beat the E12 baseline
(DREGON 2.62→2.481 smoothed, 3.186→3.082 raw); (b) constant-Q harmonic
stacking hurts (`g2_hcqt` refuted, 4.20 raw) — CQT bins smear exactly the
sub-rev/s structure the task needs. Conclusion: harmonic AGGREGATION is the
missing ingredient, and it must live on the LINEAR frequency grid.

This arm gives the trunk the blind VK tracker's own scan statistic as a
front-end: for every candidate f0 (30..120 rev/s, step 0.25 → 361 rows),
aggregate whitened-magnitude and IF evidence along the harmonic comb
(teeth k·f0 ≤ 1200 Hz — the band the VK scan proved out; small f0 rows get
more teeth, mean-over-teeth normalised). The trunk then operates in
f0-space where each rotor is a ridge.

## Setup

Identical to `e12_real_fullflight_transformer` (online-mix DREGON
whole-envelope + FLY125 real noise, LibriSpeech speech, 1 s chunks,
time-warp + gain/polarity/channel-drop augmentations, pit_mse, patience 20,
valid = `dload:DREGON-LM-V4-michaels-valid-full`) except the model:

* `simple_conv_v2_transformer_comb` — the SimpleConvV2Transformer trunk on
  the `comb_if` front-end (3 channels × 361 f0 rows, hop-512 time grid):
  1. comb score — mean whitened log-mag over the row's teeth (whitening =
     running median over frequency, 150 Hz window, mirroring
     `data_processing.vk_blind_seeding.whitened_logmag`);
  2. frequency consensus — per-tooth IF deviation converted to rev/s
     (IF·Δf/k), Fisher magnitude·k²-weighted mean, clamped to ±2 rev/s;
  3. occupancy — fraction of teeth above the frame's spectrum median (the
     stage-guard tooth statistic).

## Evaluation

`python scripts/rps_predictor_vk_eval.py` after registering the checkpoint —
per-clip pooled PIT-MAE on the VK-comparison clips, with and without
test-time smoothing arms; compare against E12 (3.186 raw / 2.62 med), g2_if
(3.082 raw / 2.481 chmean-med20) and the blind-VK bars (DREGON cruise
0.68-0.74, FLY124 1.027).

## Conclusion

Pending — training not yet run.
