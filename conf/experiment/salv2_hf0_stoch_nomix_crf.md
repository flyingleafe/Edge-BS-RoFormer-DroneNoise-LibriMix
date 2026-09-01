---
experiment: salv2_hf0_stoch_nomix_crf
training_config: conf/experiment/salv2_hf0_stoch_nomix_crf.yaml
batch: docs/experiments/salv2-speech-and-objective-grid.md
---

# `salv2_hf0_stoch_nomix_crf`

## Motivation

One cell of the 20-cell SALV2 grid: {HarmoF0, HPPNet, SimpleConvV2} x {static
comb, stochastic} x {without speech, with speech}, plus a CRF-objective arm
for the two salience ports. This cell is **HarmoF0 (MRDConv trunk, 0.36 M params)** on the **stochastic rotor-noise family**,
**without speech**, trained on the CRF negative log-likelihood (`log Z - score(gold)`), the deployed decoder's own objective.

Its BCE twin is [`salv2_hf0_stoch_nomix`](./salv2_hf0_stoch_nomix.md), identical in every other field, so the pair isolates the objective.

The grid exists because the predecessor streams could not support a
train-versus-validation comparison at all: `freq_scale` fired on every training
sample while the fixed validation set deleted all augmentation, and `n: 96` over
8 microphones was 12 clips against `flight_reuse: 32`, so every validation frame
came from ONE trajectory. Full batch context:
[SALV2 — rebuilt streams, the speech A/B, and the CRF objective](../../docs/experiments/salv2-speech-and-objective-grid.md).

## Setup

Hydra wiring — data `salv2_stoch_nomix` · model `harmof0_rps_l4` · loss `salv2_crf_r150` · metrics
`salience_layers_r150`. Train with `python train.py experiment=salv2_hf0_stoch_nomix_crf`.

Training clips are 4 s and validation clips 8 s, from conf/online_mix/salv2_stoch.yaml. Batch size 32
frames (4 chunks x 8 microphones) is measured, not inherited: on an A100-80 at
4 s the worst cell of the grid (HPPNet, any objective) peaks at 17.18 GiB, which
is the largest power of two that stays inside a 24 GB card.

The monitor is `rps_mae` — PIT per-frame absolute error in rev/s — and NOT the
training objective. Selecting on the objective was measured to be wrong here: on
the l4 arms `val/bce` rose while `val/rps_mae` fell, because cross-entropy is
unbounded and punishes a confident near-miss that a bounded argmax error barely
notices.

## Conclusion

Pending — the cell is running.
