# E10 — Full-flight synthetic training (reduce domain mismatch, not raise difficulty)

**Premise.** The E7–E9 sim→real "failures" were largely a contaminated validation
set (FLY124 ground warm-up leaking in; fixed → E9 sim transfer is real, val MSE
17–25 across archs). The residual gap is domain mismatch, so E10 *reduces the
mismatch* rather than making the task harder: cover the whole flight envelope and
drop augmentation.

**Three synthetic-data upgrades** (commit 4b3619b):
1. `rps_synthesis.generate_full_flight` — RPS trajectories through ground → warm-up
   → takeoff → cruise → landing → ground (not cruise-only).
2. Static-comb amplitude ∝ rps^2.5 (zero RPS → silence); `rps.kind: full_flight`.
3. `balance_rps` generator training data so the generator sees low/zero RPS.

**Pipeline.**
- `e10_noisegen_fullrange` — retrain the per-drone generator on the full RPS range
  (`noise_rps_dregon_michaels_fullrange`, balance_rps) so it learns zero → silence.
- `e10_full_{transformer,unigru128,scv2}` — train the 3 predictors on 50% neural
  gen (full_flight, from the retrained ckpt) + 50% static-comb (full_flight) +
  LibriSpeech, **no augmentation**. Validate on `DREGON-LM-V4-michaels-valid-full`
  (full envelope: warm-up/takeoff/cruise/landing/ground).

Policies: `conf/online_mix/e10_full_flight_{dload,p100}.yaml`; data
`conf/data/e10_full_flight.yaml`.

## Conclusion

*(Backfilled 2026-08-20. Full numbers and figures:
[2026-07-12 report](../../writing/reports/2026-07-12_full-flight-sim2real-rps/).
This is the batch doc for E10–E12.)*

**E10 (no augmentation) failed as predicted-in-reverse:** the models fit the
synthetic stream (train error ≈ 5.6 rev/s) but under-read real cruise badly
(≈ 51 rev/s where the truth is ≈ 80) — overfitting to synthetic texture.
Lesson: augmentation is a domain-gap reducer, not added difficulty.

**E11** restored augmentation, added the emitter-level smoothstep silence gate
(a stopped rotor is now exactly silent), and ran the sim full-flight
curriculum plus real-only time-warp baselines on the full-envelope valid.

**E12 is the punchline:** the same real-only recipe with `min_motor_rps: 0`,
so the real take-off ramp stays in training. Per-regime PIT-MSE on the
full-envelope real valid (27 cruise / 6 warm-up / 4 ground clips), Transformer:

| Training data | Cruise | Warm-up | Ground | All |
|---|---|---|---|---|
| real-only (cruise-trained) | 15.3 | 384.9 | 2450.0 | 338.4 |
| sim full-flight curriculum (E11) | 48.3 | 463.7 | 198.0 | 131.9 |
| **real full-flight (E12)** | 20.4 | **149.4** | 374.8 | **79.6** |

The real full-flight Transformer roughly quarters the cruise-only baseline's
overall error with no synthetic data — the low-speed failure was a training
filter, not sim2real. Mean-collapse was explicitly refuted (all models track
per-regime means). Synthetic keeps two roles: it beats the cruise-only
baseline (any low-speed coverage helps), and it is the only source of true
silence — every model still guesses 10–15 rev/s on a stopped drone. Caveat:
only 4 ground + 6 warm-up valid clips, so per-regime numbers are noisy; the
Transformer row is the defensible headline.
