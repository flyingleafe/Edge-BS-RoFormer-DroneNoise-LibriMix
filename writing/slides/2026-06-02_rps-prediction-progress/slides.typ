#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [RPS Prediction from Drone Audio],
  subtitle: [Three experiments, one story],
  author: [Dmitrii Mukhutdinov],
  date: [2026-06-02],
)

= Comparison with classical Methods

PYIN, cepstral analysis, HPS, matched-filter bank, NMF

#figure(
  image("assets/classical_vs_neural.png", width: 80%),
)

*Result:* All classical methods fail. SimpleConv (blue) tracks ground truth; the rest are noise.

= Second Attempt: Architecture Sweep

10 SimpleConv variants. Same data, same training.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(image("assets/leaderboard.png", width: 100%)),
  figure(image("assets/pareto.png", width: 100%)),
)

*Finding:* BiGRU temporal head is the single most important change. BiGRU-v2 dominates the Pareto frontier.

Overall, using better architectures is obviously fruitful.

= What the Improvement Looks Like

One random DREGON-LM valid sample (8 s, −15 dB SNR)

#figure(
  image("assets/sample_comparison.png", width: 80%),
)

*Top:* noisy mixture spectrogram. *Middle:* SimpleConv baseline. *Bottom:* BiGRU-v2.

= Third Attempt: Build a Harder Dataset

DREGON-LM-V1 was too easy:
- Train/validation overlap (same recordings, even though different chunks of those)
- 1-second clips (≈32 STFT frames)
- Only one microphone channel used -- *maybe models only learned one microphone position*

DREGON-LM-V2 is harder:
- Zero recording overlap
- 3-second clips (≈94 frames)
- All 8 microphone channels
- Also, *added 20% of synthetic constant drone noise* obtained via adding individual motor recordings from DREGON
  - The idea was to push the model more to pitch tracking away from pattern-matching.

= Evaluating on Old Validation

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(image("assets/sample_comparison.png", width: 100%)),
  figure(image("assets/sample_comparison_v3.png", width: 100%)),
)

= Evaluating on New Validation

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(image("assets/sample_comparison_v2_old.png", width: 100%)),
  figure(image("assets/sample_comparison_v2_v3.png", width: 100%)),
)

= Cross-Evaluation Shock

Every model tested on both validation sets

#figure(
  image("assets/cross_eval.png", width: 80%),
)

Old models: stellar on V1, collapse on V2. V3 models: worse on V1, robust on V2.

= Degradation Factors

#figure(
  image("assets/degradation.png", width: 80%),
)

Old checkpoints: *63–123×* degradation. V3 checkpoints: *2.2–4.7×*.

= Does the Model Know Which Rotor Is Which?

PIT-MSE vs standard MSE on V2 valid

#figure(
  image("assets/pit_gap.png", width: 80%),
)

SimpleConv: 38% gap (no temporal head = no stable ordering). BiGRU-v2: 4% gap.

= Conclusion

Expanding the dataset (more samples, more channels, synthetic motor combos) *did not yield generalization* - models start failing to train.

The V2 validation "gains" seem to be from predicting constant speeds better — a shortcut learned from 20% synthetic steady-state clips, which otherwise seem to collapse the training rather than regularizing it.

Today: will remove constant steady-state clips from training / validation sets and re-run; the goal is to observe generalization across channels on free-flight recordings.
