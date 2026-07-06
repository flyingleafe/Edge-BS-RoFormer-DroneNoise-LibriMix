#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [Channel Generalization in RPS Prediction],
  subtitle: [Do ch0-trained models generalize across microphone positions?],
  author: [Dmitrii Mukhutdinov],
  date: [2026-06-09],
)

= SimpleConv

#figure(image("assets/simpleconv_tikz.png", height: 55%), caption: none)

- *5* 2-D conv blocks
- *0.54 M* parameters
- Global average pooling + 1-D conv head
- Trained on *ch0 only* (DREGON-LM, 6000 samples)

= SimpleConvV2

#figure(image("assets/simpleconv_v2_tikz.png", height: 55%), caption: none)

- *6* residual blocks + SE
- *1.50 M* parameters
- Frequency attention pooling + BiGRU head
- Trained on *ch0 only* (same data)

= The Question

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #figure(image("assets/mic_array.png", width: 100%), caption: none)
  ],
  [
    - 8 microphones, 4 rotors
    - Mic 0 (orange) = training channel
    - Mic 4 (green) = same Z face as mic 0
    - Mics 1,3,5,7 = opposite face

    *Do ch0-trained models work on all 8 channels?*
  ],
)

= Dataset: DREGON-LM-V4 / valid

#table(
  columns: 3,
  align: (left, right, left),
  table.header([Recording], [Clips], [Source]),
  [`nosource`], [7], [Pure drone],
  [`speech-low`], [6], [Drone + speech],
  [`whitenoise-low`], [6], [Drone + noise],
)

- *19* non-overlapping 8 s clips
- *8 channels* per clip
- *No synthetic mixing* — raw recordings
- Early takeoff/landing excluded (`RPS > 30`)

= ch0-only Results: MSE

#figure(image("assets/mse_bars.png", height: 60%), caption: none)

- Green = ch0 (training), red = all others
- *3–10×* MSE degradation on edge mics
- SimpleConvV2 is *worse* than SimpleConv

= ch0-only: SimpleConv

#figure(image("assets/slide_ch0only_sc.png", height: 60%), caption: none)

- ch0: MAE = 1.13
- ch1: MAE = 7.06

= ch0-only: SimpleConvV2

#figure(image("assets/slide_ch0only_v2.png", height: 60%), caption: none)

- ch0: MAE = 0.37
- ch3: MAE = 13.20 — *catastrophic collapse*

= PIT on ch0-only

#table(
  columns: 4,
  align: (left, right, right, right),
  table.header([Model], [MSE], [PIT MSE], [Δ]),
  [SimpleConv], [35.49], [34.77], [−2.0%],
  [SimpleConvV2], [40.28], [40.07], [−0.5%],
)

= PIT on ch0 only

#figure(image("assets/mse_bars_pit.png", height: 60%), caption: none)

= 8ch Training

```
Training batch: (B, C, T) → (B·C, T)

Model still sees 1 channel per prediction.
Difference: training batch contains all 8 mic positions.

We also use PIT (Permutation-invariant training) - selecting the best-fitting
indexing of rotor predictions for loss propagation
```

#table(
  columns: 2,
  align: (left, left),
  table.header([Model], [Checkpoint]),
  [SimpleConv 8ch], [`rps_8ch_v4_simple_conv`],
  [SimpleConvV2 8ch], [`rps_8ch_v4_simple_conv_v2`],
)

= 8ch Results: Non-PI evaluation

#figure(image("assets/mse_bars_8ch_v4.png", height: 55%), caption: none)

- SimpleConv: uniform ~25–35 MSE per channel, *R² = 0.57*
- SimpleConvV2: uniform ~55–68 MSE per channel, *R² = −0.78*

= 8ch Results: PI evaluation

#table(
  columns: 4,
  align: (left, right, right, right),
  table.header([Model], [Not PI], [PI], [Δ]),
  [SimpleConv], [29.70], [28.37], [−4.5%],
  [SimpleConvV2], [61.39], [*3.30*], [*−94.6%*],
)

= 8ch Results: PI evaluation

#figure(image("assets/mse_bars_8ch_v4_pit.png", height: 55%), caption: none)

- SimpleConvV2: PIT MSE *3.30*, R² = *0.94*

It managed to learn well across all 8 channels!

= Why PIT is the Right Metric

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    *The model gets the speeds right.*

    - No-PIT MSE = 61.39
    - PIT MSE = 3.30

    But swaps rotor indices; however, this is okay.

    Which rotor is loudest depends on mic position.
    There is no acoustic signature that labels rotors
    independently of where the mic is.
  ],
  [
    Also, we do not really care about order of the rotors. We just need the model to say "here is 4 rotors and these are their RPS traces".
  ],
)

= Results on the sample with higly varying RPS

#figure(image("assets/slide_dynamic_8ch_sc.png", height: 70%), caption: none)

= Results on the sample with higly varying RPS

#figure(image("assets/slide_dynamic_8ch_v2.png", height: 70%), caption: none)

= Conclusion

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    *ch0-trained models:*
    - Easy to get good performance on one channel only
    - This would not generalize

    *8ch-trained models:*
    - Simple models (SimpleConv) already cannot generalize over several mic positions for a single drone
    - More complex model however can do that
  ],
  [
    Next steps:

    - Finish re-alignment of Michael's data, finally
    - Include it in the training set
    - Finish re-implementations of multi-pitch tracking models from literature - test them similarly on 8ch training task
  ],
)
