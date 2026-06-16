#figure(
  placement: none,
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left + horizon, left + horizon, center + horizon, center + horizon, center + horizon, center + horizon),
    table.header([*Training set*], [*Eval set*], [*RMSE (Hz)*], [*MAE frame (Hz)*], [*$R^2$ mean*], [*$R^2$ median*]),
    table.hline(),
    [DREGON-only], [DREGON-LM-V4 (in-domain)], [1.62], [1.08], [0.930], [0.955],
    [DREGON+FLY125], [DREGON-LM-V4 (in-domain)], [2.76], [2.18], [0.769], [0.776],
    table.hline(),
    [DREGON-only], [FLY124 (cross-drone)], [7.96], [5.41], [-0.273], [0.515],
    [DREGON+FLY125], [FLY124 (cross-drone)], [1.63], [1.17], [0.949], [0.961],
    table.hline(),
  ),
  caption: [SimpleConvV2 (8ch) PIT metrics (best of 24 rotor permutations per channel). Adding Michael's FLY125 to training collapses the cross-drone FLY124 error (RMSE 7.96 #sym.arrow.r 1.63 Hz) at the cost of a modest in-domain DREGON-LM-V4 regression (1.62 #sym.arrow.r 2.77 Hz). FLY124 = 9 stable in-flight 8 s slices; DREGON-LM-V4 = 30 valid clips, each #sym.times 8 channels.],
) <tab:fly125>