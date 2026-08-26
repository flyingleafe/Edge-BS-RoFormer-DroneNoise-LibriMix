---
experiment: stoch_s1s_both
training_config: conf/experiment/stoch_s1s_both.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1s_both`

## Motivation

Both synthetic families in one stream, at equal weight.

The board says the two families each own a different cell and that no single
model has held two at once:

| model | all-MAE | zero | low | flight |
|---|---|---|---|---|
| `r4hb_scv2`, the target | 2.67 | 2.87 | 3.48 | 2.49 |
| `m3abl_comb_unigru128_s1` | 8.30 | **4.73** | 24.24 | 6.00 |
| `stoch_s1h_scv2` | 9.07 | 27.98 | 26.77 | **2.60** |
| `stoch_s1g_scv2` | 8.08 | 20.27 | **16.20** | 4.50 |

The analytic static comb owns the stopped rotors at 4.73 rev/s — better than any
stochastic arm and within 1.6x of the real-trained model. The stochastic family
owns cruise at 2.60, level with real training, where the comb reaches only 6.00.

Every arm of this campaign so far has replaced one family with the other and
inherited its weakness along with its strength. This arm stops choosing: the
stochastic pool and the analytic comb enter at weight 1.0 each, the silence arm
stays at 0.2, and a model sees both textures in the same stream. The two are
different draws behind one interface, and the earlier curriculum work already
mixed the comb with the neural generator inside a single stage-1 stream, so
nothing about the mixture is new machinery.

Data `stoch_s1s`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1s_both`.

## Conclusion

PENDING — the run has not finished.
