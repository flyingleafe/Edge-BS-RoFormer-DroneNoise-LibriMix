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

_Pending runs._
