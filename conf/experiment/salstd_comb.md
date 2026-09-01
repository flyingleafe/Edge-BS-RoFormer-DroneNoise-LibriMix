---
experiment: salstd_comb
training_config: conf/experiment/salstd_comb.yaml
batch: docs/experiments/salience-map-rps-tracking.md
---

# `salstd_comb`

## Motivation

CONTROL for conf/experiment/sal150_comb.yaml: the same analytic static-comb curriculum
and the same same-family validation, on the STANDARD salience grid (the log
grid of conf/model/multif0_salience.yaml — 32.7 Hz, 6 octaves, over_sample 5,
360 bins), which is what every existing salience row uses.

The pair isolates the grid. sal150_comb changes data AND grid against the
real-data row hb_sal_multif0; this row changes data only, thus the difference
between the two is the grid's contribution and nothing else.

Everything except model/loss/metrics is sal150_comb verbatim.

Full batch context: [Salience-Map Multi-F0 Tracking for RPS Prediction](../../docs/experiments/salience-map-rps-tracking.md).

## Setup

Hydra wiring — data `sal_comb_synthval` · model `multif0_salience` · loss `salience_bce_multif0` · metrics `salience_bce_multif0`. Train with `python train.py experiment=salstd_comb`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Salience-Map Multi-F0 Tracking for RPS Prediction](../../docs/experiments/salience-map-rps-tracking.md).
