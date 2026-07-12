#figure(
  table(
    columns: 5, align: (left, right, right, right, right), stroke: 0.5pt,
    table.header([Training data], [Cruise], [Warm-up], [Ground], [All]),
    table.cell(colspan: 5)[#emph[Transformer]],
    [real-only (cruise)], [15.3], [384.9], [2450.0], [338.4],
    [sim full-flight curriculum], [48.3], [463.7], [198.0], [131.9],
    [real full-flight], [20.4], [149.4], [374.8], [79.6],
    table.cell(colspan: 5)[#emph[Uni-GRU-128]],
    [real-only (cruise)], [14.6], [389.0], [1658.3], [253.0],
    [sim full-flight curriculum], [17.9], [258.2], [1153.4], [179.6],
    [real full-flight], [19.7], [646.1], [655.7], [190.0],
    table.cell(colspan: 5)[#emph[SimpleConv-v2]],
    [real-only (cruise)], [15.7], [241.9], [1227.8], [183.4],
    [sim full-flight curriculum], [20.6], [301.8], [634.4], [132.5],
    [real full-flight], [43.3], [476.5], [335.2], [145.1],
  ),
  caption: [Per-regime and overall PIT-MSE on the full-envelope real validation set (27 cruise / 6 warm-up / 4 ground clips). Lower is better; bold-worthy numbers discussed in text.],
) <tab-results>
