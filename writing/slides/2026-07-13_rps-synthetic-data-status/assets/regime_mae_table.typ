#figure(
  table(
    columns: 4, align: (left, right, right, right), stroke: 0.5pt,
    table.header([Recipe / regime], [Transformer], [Uni-GRU-128], [SimpleConv-v2]),
    table.cell(colspan: 4)[#emph[real-only, cruise-trained] (All: 7.2 / 7.8 / 5.0)],
    [Cruise], [2.5], [2.7], [2.5],
    [Warm-up], [10.0], [12.1], [6.7],
    [Ground], [34.4], [35.7], [19.7],
    table.cell(colspan: 4)[#emph[sim full-flight curriculum] (All: 8.3 / 7.3 / 7.0)],
    [Cruise], [5.2], [2.8], [2.7],
    [Warm-up], [20.2], [10.1], [14.6],
    [Ground], [10.9], [33.4], [25.0],
    table.cell(colspan: 4)[#emph[real full-flight (min_rps=0)] (All: 4.8 / 6.9 / 7.7)],
    [Cruise], [2.8], [3.0], [5.2],
    [Warm-up], [7.7], [18.6], [19.4],
    [Ground], [13.9], [16.0], [7.1],
  ),
  caption: [Per-regime MAE (rev/s, PIT-aligned via `align_rps_to_gt`), all 3 architectures x 3 recipes, full-flight validation set (michaels-valid-full, 27 cruise / 6 warm-up / 4 ground STFT-frame windows; regime classified per-window by mean ground-truth RPS; "All" (in each recipe header, Transformer / Uni-GRU-128 / SimpleConv-v2 order) is the mean over all windows, not a regime average).],
)
