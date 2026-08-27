# r4a_seed2 — seed repeat 2 of `r4a_lr3e4`

One of four runs (two seeds x two configs) that decide whether the 2.61 vs 2.67
all-MAE difference between `r4a_lr3e4` and `r4hb_scv2` survives reseeding.

A 2% gap on one seed each is not a result. If the distributions overlap, the
record claim is withdrawn from the board and the log.

These configs exist as files rather than command-line overrides because
`zoo.load` composes by experiment name — a run launched with
`experiment_name=` alone cannot be scored afterwards.
