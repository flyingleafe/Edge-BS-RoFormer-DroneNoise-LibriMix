# E9 — "Hard" Combined Generated-Noise Task

**Status:** completed — **Date:** 2026-07-12, conclusion backfilled 2026-08-20

## Motivation

Neither generated-noise family transfers to real RPS prediction on its own:

- **E7** (neural generator, vicinal interp): real val PIT MSE ~222–225, R² ≈ −10
  across all three archs — the predictor reverse-engineers the generator's
  amplitude dynamics (a shortcut absent in real data).
- **E8** (analytic static-comb, amplitudes RPS-independent): helped the
  transformer (225→189, R² −10.1→−7.3) but not the smaller heads, and all still
  fail (~10–12 rev/s off vs the real baseline's ~2.7). The synthetic RPS
  distribution roughly matches real (mean ~79), so it is not an RPS-range gap.

So removing the amplitude shortcut is necessary-but-insufficient. E9 makes the
training task **harder** so no single-source shortcut survives:

- **50% neural generator + 50% static-comb** noise (MixedNoisePool, weights
  1:1) — the model must read RPS from both the neural gen's rich, variable
  amplitudes *and* the static comb's amplitude-free combs, so neither source's
  idiosyncratic amplitude→RPS mapping is a reliable shortcut; the common,
  transferable cue is the comb *frequency*.
- LibriSpeech mixtures at −30..0 dB.
- **50% mixture-level augmentation from the start** (`random_gain`,
  `random_polarity`, `channel_drop`) — further discourages amplitude/level
  memorisation.
- **Patience 20** (vs E7/E8's 5/8) — train longer.

Only the transformer is run first (the arch E8 helped). The converged
`last.ckpt` (added to `training.loop` this batch — `best.ckpt` is the early
val-best, wrong for this) is the artifact for the sim→real failure-mode
diagnostic (pred vs GT RPS on a real clip).

## Setup

`e9_hard_transformer` — data `rps_hard_combined`, model
`simple_conv_v2_transformer`, `pit_mse`, `rps`, patience 20,
`samples_per_validation=5000`. Neural-gen source runs a GPU producer.

```bash
omnirun submit --backend colab --gpu-type L4 --gpus 1 --time 4h --yes -- \
  python train.py experiment=e9_hard_transformer \
    data.train.params.path=conf/online_mix/rps_hard_combined_dload.yaml \
    "data.valid.params.data_dir='dload:DREGON-LM-V4-michaels-valid'"
```

## Conclusion

*(Backfilled 2026-08-20 from the W&B run summaries and the
[2026-07-12 report](../../writing/reports/2026-07-12_full-flight-sim2real-rps/).)*

**E9 is where the sim2real "failure" turned out to be a contaminated
validation set.** The first transformer run (2026-07-11) scored best PIT-MSE
176.5 (R² −7.9) on the then-current valid (`min_motor_rps=30`) — better than
E7/E8 but still failure-shaped. The FLY124 ground warm-up clips (~36 rev/s,
`flyCState=AssistedTakeoff`, labelled as flight) were then found leaking into
that valid. The clean free-flight-only split (`min_motor_rps=50`, published as
`DREGON-LM-V4-michaels-valid` pin `b6ece43d`) landed together with the
remaining E9 configs, and every number changed sign:

| Arch | Stage 1 (gen-only) | R² | Stage 2 (real fine-tune) | R² |
|---|---|---|---|---|
| `uni_gru128` | 17.8 | 0.62 | **11.1** | 0.74 |
| `transformer` | 19.2 | 0.59 | 14.1 | 0.70 |
| `scv2` | 25.4 | 0.40 | 13.6 | 0.73 |

(Best-epoch validation PIT-MSE; the same transformer recipe scored 176.5 on
the contaminated valid and 19.2 on the clean one.)

**Verdicts:**

1. **Sim transfer is real.** A predictor trained on generated noise ALONE
   (50% neural generator + 50% static comb, with augmentation) reaches
   PIT-MSE 17.8–25.4 with positive R² on real free-flight validation. The
   earlier "R² ≈ −10, worse than the mean" conclusion was the yardstick, not
   the models.
2. **A short real fine-tune closes most of the residual gap** (11.1–14.1).
   For reference, the best real-data-only number was 7.33 (C10 online
   `uni_gru128`) — but on the *old* cruise-only split, so the two are not
   directly comparable (the 2026-07-13 deck states the split caveat).
3. The hard-mix design (no single-source amplitude shortcut) plus
   augmentation-from-the-start is the recipe E10/E11 inherited.

Follow-up: the E10–E12 coverage chain ([e10-full-flight.md](e10-full-flight.md))
asks the next question — full-flight coverage — and ends with real full-flight
training beating the synthetic curriculum.
