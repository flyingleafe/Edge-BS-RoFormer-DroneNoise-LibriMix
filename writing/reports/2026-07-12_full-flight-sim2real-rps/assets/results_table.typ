#figure(
  table(
    columns: 5, align: (left, right, right, right, right), stroke: 0.5pt,
    table.header([Model], [Cruise], [Warm-up], [Ground], [All]),
    table.cell(colspan: 5)[#emph[Transformer]],
    [real-only + warp], [15.3], [384.9], [2450.0], [338.4],
    [full-flight curriculum], [--], [--], [--], [--],
    table.cell(colspan: 5)[#emph[Uni-GRU-128]],
    [real-only + warp], [14.6], [389.0], [1658.3], [253.0],
    [full-flight curriculum], [--], [--], [--], [--],
    table.cell(colspan: 5)[#emph[SimpleConv-v2]],
    [real-only + warp], [15.7], [241.9], [1227.8], [183.4],
    [full-flight curriculum], [--], [--], [--], [--],
  ),
  caption: [Per-regime and overall PIT-MSE on the full-envelope real validation set. Lower is better.],
) <tab-results>
