#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [RPS Prediction for Drone Speech Enhancement],
  subtitle: [April 14, 2026],
  author: [Dmitrii Mukhutdinov],
  date: [2026-04-14],
)

= Why This Experiment

- Multi-task training (denoising + RPS prediction) was much worse than baseline.
- Question: is RPS prediction itself broken for DCUNet / DCCRN encoders?
- So we isolated pure RPS prediction and compared against SimpleConv.

= What I Have Been Doing

- RPS prediction with DCUNet / DCCRN encoders vs SimpleConv baseline
- Goal: identify architectural limitations affecting motor-speed prediction
- In parallel: debug and restart multi-task experiments (WIP)

= RPS Predictor Architectures

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 12pt,
  [
    *SimpleConv*
    Log-Mag STFT $arrow$ Real Conv Blocks $arrow$ FPN Head $arrow$ 4-Rotor RPS
  ],
  [
    *DCUNet Encoder + RPS Head*
    Complex STFT $arrow$ DCUNet Encoder $arrow$ FPN Head $arrow$ 4-Rotor RPS
  ],
  [
    *DCCRN Encoder + RPS Head*
    Complex STFT $arrow$ DCCRN Encoder $arrow$ FPN Head $arrow$ 4-Rotor RPS
  ],
)

= Metrics (5 Validation Samples)

#figure(
  image("assets/summary_metrics.png"),
  caption: [Summary metrics for RPS prediction.],
)

#table(
  columns: (2fr, auto, auto, auto),
  inset: 6pt,
  align: (left, center, center, center),
  table.header(
    [*Model*], [*RMSE $arrow$*], [*MAE $arrow$*], [*$R^2$ $arrow$*],
  ),
  strong[SimpleConv], strong[1.63], strong[1.16], strong[0.82],
  [DCUNet], [2.62], [1.69], [0.56],
  [DCCRN], [2.46], [1.54], [0.61],
)

= Sample Comparisons

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  figure(
    image("assets/sample_00000_plot.png"),
    caption: [sample_00000],
  ),
  figure(
    image("assets/sample_00149_plot.png"),
    caption: [sample_00149],
  ),
)

= More Sample Comparisons

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  figure(
    image("assets/sample_00449_plot.png"),
    caption: [sample_00449],
  ),
  figure(
    image("assets/sample_00599_plot.png"),
    caption: [sample_00599],
  ),
)

= Main Result

- DCUNet / DCCRN encoders with attached RPS heads are worse than SimpleConv
- This supports the hypothesis that these encoder setups are not ideal for RPS prediction

= Critical Issues Found

- Ground-truth RPS was injected at the encoder while the encoder also predicted RPS
- This made the auxiliary prediction objective partially meaningless
- RPS output was not normalized
- RPS loss dominated encoder updates and destabilized training dynamics

= Why We Restarted Multi-Task Experiments

- The RPS predictor comparison gave a real signal: architecture matters
- But the first multi-task setup had severe confounders
- We fixed those issues and relaunched experiments (WIP)
- Detailed multi-task results will follow later

= Next Steps

- Progress has been bad last month: still had issues to finish up on the job I'm leaving
- But we have 1.5 months more for focused research due to interruption extension
- Experimental directions before 18 May:
  - Still, diffusion models
  - Transformer architectures replacing convolutional architectures
  - Achieve good results separately on RPS prediction and on noise generation
  - Combine obtained RPS predictor and noise generator into the denoising model

Main issue is lack of continuous worktime spent on research. Expectation of good progress in such a regime was incorrect.
