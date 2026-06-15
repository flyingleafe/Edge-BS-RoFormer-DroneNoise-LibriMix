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
