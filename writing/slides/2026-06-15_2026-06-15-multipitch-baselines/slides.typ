#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [Salience-Map Multi-Pitch Baselines for RPS Prediction],
  subtitle: [DREGON-LM-V4 / valid — 30 clips × 8 channels],
  author: [Dmitrii Mukhutdinov],
  date: [2026-06-15],
)

= Setup

- Dataset: DREGON-LM-V4/valid — 30 clips × 8 channels, 16 kHz
- Evaluation: permutation-invariant (PIT), 24 rotor permutations
- Models: SimpleConvV2, SimpleConv (regression) vs LateDeep, LateDeep-fast, Basic Pitch (salience)

= Basic Pitch (Spotify model)

#figure(
  placement: none,
  image("assets/basic-pitch-illustration.png", height: 80%),
  caption: [Basic Pitch: CQT → harmonic stacking → multi-branch conv → $Y_o, Y_p, Y_n$ (Bittner et al., 2022). Only ~16,000 parameters in total!],
)


= LateDeep (multif0_salience)

#cols(columns: (1fr, 2fr), lazy-layout: true)[

  #figure(
    placement: none,
    image("assets/multif0-illustration-2.png", width: 100%),
    caption: [Peak picking & thresholding → multi-F0 output (Cuesta et al., 2020)],
  )

][

  #figure(
    placement: none,
    image("assets/multif0-illustration-1.png", width: 100%),
    caption: [LateDeep architecture: HCQT magnitude + phase differentials → CNN → salience map. ~1.2M parameters in total],
  )

]

LateDeep-fast is a modification of LateDeep using simplified stacked HCQT (as Basic Pitch) + implementing two legs for magnitude and phase as single grouped convolutional stack.

= Adapting models to our task

*Training*
- Binary salience targets derived from 4 ground-truth RPS values
- BCE loss → sigmoid → continuous salience map $S(t,f) in [0,1]$

*Inference*
+ Peak detection: local maxima above threshold $theta = 0.3$
+ Hungarian tracking: 4 trajectories, max jump 3 bins per frame
+ Resample to STFT grid → PIT metrics (RMSE, MAE, $R^2$)

= Leaderboard

#figure(
  image("assets/leaderboard_metrics.png", width: 100%),
  caption: [RPS prediction leaderboard on DREGON-LM-V4/valid. Left: RMSE (Hz); right: $R^2$.],
)

= Per-rotor MAE

#figure(
  image("assets/per_rotor_mae.png", width: 80%),
  caption: [Per-rotor frame MAE for the three salience models. Basic Pitch shows strong rotor-dependent failure.],
)

= LateDeep

#figure(
  image("assets/sample_00026_multif0_salience_3pane.pdf", height: 80%),
  caption: [LateDeep — spectrogram, salience map, and RPS trajectories.],
)

= LateDeep-fast

#figure(
  image("assets/sample_00026_multif0_salience_fastest_3pane.pdf", height: 80%),
  caption: [LateDeep-fast — spectrogram, salience map, and RPS trajectories.],
)

= Basic Pitch

#figure(
  image("assets/sample_00026_basic_pitch_salience_3pane.pdf", height: 80%),
  caption: [Basic Pitch — spectrogram, salience map, and RPS trajectories.],
)

= LateDeep vs SimpleConvV2

#figure(
  placement: none,
  image("assets/sample_00026_compare_simpleconvv2_vs_multif0_salience.png", width: 95%),
  caption: [Top: SimpleConvV2 (RMSE 1.62 Hz). Bottom: LateDeep (RMSE 6.30 Hz).],
)

= LateDeep-fast vs SimpleConvV2

#figure(
  placement: none,
  image("assets/sample_00026_compare_simpleconvv2_vs_multif0_salience_fastest.png", width: 95%),
  caption: [Top: SimpleConvV2. Bottom: LateDeep-fast (RMSE 6.42 Hz) — essentially same accuracy as LateDeep.],
)

= Basic Pitch vs SimpleConvV2

#figure(
  placement: none,
  image("assets/sample_00026_compare_simpleconvv2_vs_basic_pitch_salience.png", width: 95%),
  caption: [Top: SimpleConvV2. Bottom: Basic Pitch (RMSE 23.24 Hz, $R^2 = -16.21$) — complete failure.],
)

= Take-away

- Task-specific regression (SimpleConvV2) far outperforms off-the-shelf multi-pitch salience maps
- Salience grid resolution floor: ~2.5–3 Hz irreducible error from bin snapping
- LateDeep-fast (stacked HCQT) is nearly free: same accuracy, ~2× faster front-end
- Basic Pitch is unusable for continuous RPS tracking — designed for discrete musical notes

= Narrow-band super-resolution salience — idea

*The failure mode:* rotors cluster in 69–89 Hz; 55% of frames have two rotors < 1 Hz apart → coarse grids merge them → trajectories collapse to the mean.

*The fix — decouple output grid from input CQT:*
- *Narrow input:* HCQT `fmin = 55` Hz, 1 octave, harmonics [1,2,3,4] (multif0); contour CQT `fmin = 55` Hz, 12 semitones (Basic Pitch)
- *Super-resolution output:* `FreqSuperResHead` → linear 55–110 Hz grid, 360 bins (~0.15 Hz/bin), $(5,1)$-conv sharpening, BCE end-to-end
- Tracker reads the fine grid (max-jump auto-scaled to ~1.5 Hz)

= Narrow-SR — leaderboard

#figure(
  image("assets/leaderboard_metrics_narrow_sr.png", width: 100%),
  caption: [With narrow-SR models (last two bars). `multif0_salience_narrow_sr`: RMSE 6.30 → *4.03 Hz*, $R^2$ 0.19 → *0.57* — now the best salience model, 3rd overall.],
)

= Narrow-SR — per-rotor MAE

#figure(
  image("assets/per_rotor_mae_narrow_sr.png", width: 80%),
  caption: [Narrow-SR LateDeep has the lowest and most *even* per-rotor errors (1.8–2.9 Hz) — the near-unison rotors no longer collapse.],
)

= Narrow-SR LateDeep

#figure(
  image("assets/sample_00026_multif0_salience_narrow_sr_3pane.pdf", height: 80%),
  caption: [multif0_salience_narrow_sr — salience now restricted to 55–110 Hz on a fine linear grid; four distinct trajectories tracked.],
)

= Narrow-SR Basic Pitch

#figure(
  image("assets/sample_00026_basic_pitch_narrow_sr_3pane.pdf", height: 80%),
  caption: [basic_pitch_narrow_sr — RMSE 23.24 → 11.66 Hz, but $R^2$ still negative ($-3.24$): diffuse map, mistracks.],
)

= Narrow-SR LateDeep vs SimpleConvV2

#figure(
  placement: none,
  image("assets/sample_00026_compare_simpleconvv2_vs_multif0_salience_narrow_sr.png", width: 95%),
  caption: [Top: SimpleConvV2 (RMSE 1.62 Hz). Bottom: narrow-SR LateDeep (RMSE 4.03 Hz) — gap narrowed from ~4× to ~2.5×.],
)

= Narrow-SR Basic Pitch vs SimpleConvV2

#figure(
  placement: none,
  image("assets/sample_00026_compare_simpleconvv2_vs_basic_pitch_narrow_sr.png", width: 95%),
  caption: [Top: SimpleConvV2. Bottom: narrow-SR Basic Pitch (RMSE 11.66 Hz, $R^2 = -3.24$) — much improved but still unusable.],
)

= Narrow-SR — take-away

- Concentrating the grid in the rotor band + a learned super-resolution head *closes most of the LateDeep ↔ regression gap* and removes the rotor-collapse failure mode
- `multif0_salience_narrow_sr` now beats its round-trip resolution floor (2.5–3 Hz) → finer grid buys real localization, not hallucinated precision
- Salience maps are *not* fundamentally unsuited to closely-spaced rotors — the old baselines were just mis-resolved
- Basic Pitch stays unusable regardless of grid; SimpleConvV2 still leads
- Next lever: longer training clips to lift the 1-second input-resolution wall

= Cross-drone test: Michael's FLY124

DREGON-trained *SimpleConvV2 (8ch)*, evaluated *without retraining* on a different aircraft (Michael's FLY124 8-channel recording, RPS from DJI telemetry).

- Stable in-flight slices only (per-frame mean rotor speed > 45 Hz): 9 slices × 8 channels = 72 rows
- PIT: frame MAE *5.4 Hz*, $R^2$ median *0.52* — vs *0.93–0.96* in-domain on DREGON-LM-V4
- Fixed-order $R^2$ median $approx 0$ → much of the error is rotor *identity* (permutation), not trajectory shape

= FLY124 — example slice

#figure(
  placement: none,
  image("assets/fly124_sample_00004.png", width: 100%),
  caption: [FLY124 `sample_00004` (channel 0): input spectrogram (left) and predicted RPS (solid) over PIT-aligned GT (dotted, right). Tracks the dynamics but *underpredicts the faster rotors* (GT ~80–95 Hz vs prediction ~75–85 Hz).],
)

= FLY124 — per-channel error vs in-domain

#figure(
  placement: none,
  image("assets/fly124_vs_v4_per_channel.png", width: 100%),
  caption: [SimpleConvV2 per-channel PIT error: in-domain DREGON-LM-V4 (uniform ~1 Hz, $R^2$ 0.94–0.96) vs cross-drone FLY124. Most mics rise to 3.4–4 Hz, but channel 1 collapses (12.2 Hz, $R^2 = -2.7$) and ch6–7 degrade — a *channel-dependent* gap, i.e. mic-placement / SNR, not a uniform shift.],
)

= Cross-drone — take-away

- The DREGON-trained model does not transfer well to Michael's recording
- The gap is *channel-dependent* (ch1 collapses, ch6–7 degrade) — mic placement / SNR, not a uniform shift
- Motivates adding Michael's other recording *FLY125* to the training set, to see if the model could generalize across two different drones at least.
- Next: train on DREGON+FLY125, re-test on FLY124 to measure how much the cross-drone gap closes

= Adding FLY125 to training closes the gap

DREGON-trained vs *DREGON + FLY125*-trained SimpleConvV2 (8ch), same two eval sets (PIT):

#table(
  columns: (auto, auto, auto, auto),
  inset: 7pt,
  align: (left + horizon, left + horizon, center + horizon, center + horizon),
  table.header([*Training set*], [*Eval set*], [*RMSE (Hz)*], [*$R^2$ median*]),
  table.hline(),
  [DREGON-only], [DREGON-LM-V4 (in-dom.)], [*1.62*], [*0.955*],
  [DREGON+FLY125], [DREGON-LM-V4 (in-dom.)], [2.77], [0.776],
  table.hline(),
  [DREGON-only], [FLY124 (cross-drone)], [7.96], [0.515],
  [DREGON+FLY125], [FLY124 (cross-drone)], [*1.63*], [*0.961*],
)

RMSE *7.96 #sym.arrow.r 1.63 Hz* ($R^2$ 0.52 #sym.arrow.r 0.96) — model seem to be able to generalize across drones.

= FLY125 — per-channel error

#figure(
  placement: none,
  image("assets/fly125_per_channel.png", width: 100%),
  caption: [The DREGON-only channel-dependent failure on FLY124 (right: ch1 spikes to 12 Hz, ch6–7 degrade) is *erased* — DREGON+FLY125 is uniform ~1.1 Hz on every mic. In-domain (left): a uniform ~1 Hz regression.],
)

= FLY125 — training dynamics

#figure(
  placement: none,
  image("assets/fly125_loss_curves.png", width: 100%),
  caption: [Train/val PIT loss (RMSE Hz). Val sets differ (DREGON-only: V4; DREGON+FLY125: V4+FLY125 mix). The FLY125 run peaks at *epoch 20* then overfits — the model is still not sufficient to generalize without some quality degradation.],
)

= FLY125 — example slice

#figure(
  placement: none,
  image("assets/fly125_sample_00004.png", width: 100%),
  caption: [DREGON+FLY125 on FLY124 `sample_00004` (ch0): predictions are useful now.],
)

= FLY125 — in-domain old vs new

#figure(
  placement: none,
  image("assets/fly125_v4_compare.png", height: 80%),
  caption: [DREGON-LM-V4 `sample_00012` (ch0): spectrogram, then DREGON-only (PIT MAE 1.19 Hz) and DREGON+FLY125 (2.17 Hz) predictions over GT (dotted). We see that model lost precision on DREGON recordings.],
)

= FLY125 — take-away

- *In principle, best model generalizes across drones*: FLY124 RMSE 7.96 #sym.arrow.r 1.63 Hz, $R^2$ 0.52 #sym.arrow.r 0.96.
- But its overall precision degrades from additional data due to _overfitting_ still.
- Next steps: fight overfitting (more randomized, non-repeating mixtures) and iterate on model architecture once more, to get model which is precise on both drones.
