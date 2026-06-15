#figure(
  placement: none,
  table(
    columns: (2fr, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left + horizon, center + horizon, center + horizon, center + horizon, center + horizon, center + horizon),
    table.header([*Model*], [*RMSE (Hz)*], [*MAE frame (Hz)*], [*MAE clip (Hz)*], [*$R^2$*], [*$R^2$ median*]),
    table.hline(),
    [SimpleConvV2 (8ch)], [1.62], [1.08], [0.57], [0.930], [0.955],
    [SimpleConv (8ch)], [3.55], [2.71], [2.48], [0.676], [0.690],
    [multif0_salience], [6.30], [3.40], [2.99], [0.188], [0.330],
    [multif0_salience_fastest], [6.42], [3.58], [3.28], [0.106], [0.358],
    [basic_pitch_salience], [23.24], [16.19], [15.90], [-16.213], [-10.285],
  ),
  caption: [RPS prediction leaderboard on DREGON-LM-V4/valid (30 clips × 8 channels, PIT eval).],
) <tab:leaderboard>