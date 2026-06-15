#figure(
  placement: none,
  table(
    columns: (2fr, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left + horizon, center + horizon, center + horizon, center + horizon, center + horizon, center + horizon),
    table.header([*Alignment*], [*RMSE (Hz)*], [*MAE frame (Hz)*], [*MAE clip (Hz)*], [*$R^2$ mean*], [*$R^2$ median*]),
    table.hline(),
    [Fixed-order], [9.61], [7.53], [6.89], [-0.855], [-0.086],
    [PIT (oracle rotor match)], [7.96], [5.41], [4.41], [-0.273], [0.515],
  ),
  caption: [DREGON-trained SimpleConvV2 (8ch) on Michael's FLY124, 9 stable in-flight 8 s slices (72 sample×channel rows). PIT = best of 24 rotor permutations per channel.],
) <tab:fly124>