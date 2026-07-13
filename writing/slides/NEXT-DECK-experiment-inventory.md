# Next deck — experiment inventory since the last slides

Prep note for the next slide deck. **Not a deck itself.** Delete or fold into the
deck directory once `writing/slides/<date>_<title>/` is scaffolded.

## Boundary

Last deck: **`writing/slides/2026-07-06_gp-rotor-noise/`** (created 2026-07-06,
substantively revised through commit `3ff865b`, 2026-07-10).

It ended on: *"the deep generator doesn't help as RPS augmentation (actually makes
things worse) → physics/GP rotor noise is the alternative"*. Its section list:
hypothesis → deep generator architecture → real vs generated → not good enough →
loss curves → Part I take-away → how others generate rotor noise → how ours differs
→ does it work (one recording) → what it costs / what's next.

**Everything below is after that deck.**

---

## Experiments

### 1. RPS trajectory refinement, stages A–D — 2026-07-10
- Doc: `docs/experiments/rps-trajectory-refinement.md`
- **Report: `writing/reports/2026-07-10_rps-refinement/`**
- **Result: a reframe, not a win.** Built comb-alignment refinement of telemetry RPS
  labels. DREGON labels turned out **nearly unbiased** (command err 0.633, bias
  −0.057; smoothed-measured 0.484 / +0.017). So the mid-frequency harmonic washout
  is **jitter linewidth**, not label error.
- Stages B+C are actively **biased** on twin rotor pairs (rotors fly ~0.65 rev/s
  apart → low/mid harmonics merge; low-band confounders outvote resolved high k).
  Only stage D (phase-based) can arbitrate. Trust D.
- Fed directly into E6.

### 2. Kalman harmonic tracker — Phase 0 — 2026-07-10
- Docs: `docs/experiments/kalman-harmonic-tracker-phase0.md`,
  bet card `docs/experiments/bets/kalman-harmonic-tracker.md`
- Code: `src/experiments/kalman_harmonic/`
- **BET KILLED at gate K2.** Drift-robustness hypothesis refuted: the tracker
  collapses *faster* than plain least-squares as RPS drift grows
  (0.5% drift: tracker −5.45 dB vs `lstsq_VP` −4.79 dB SI-SDR).
- Salvaged lessons: the **joint per-order update** is required (the as-drafted
  diagonal update fails); learned rotation correction is the reusable idea.

### 3. E5 — Time-warp augmentation — 2026-07-10/11 ✅ POSITIVE
- Doc: `docs/experiments/e5-timewarp.md`
- Warp the noise+RPS pair together (α ≤ 1.12).
- Best val PIT-MSE: transformer 11.76 → **8.74 (−26%)**; scv2 9.71 → **8.85 (−9%)**;
  uni_gru128 10.45 → 10.33 (tie).
- **These checkpoints are the real-only baseline for everything after.**

### 4. E6 — Noise-generator harmonic linewidth (jitter injection) — 2026-07-11 ✅ POSITIVE
- Doc: `docs/experiments/noise-gen-linewidth.md`
- Six arms: `baseline`, `randphase`, `jitter`, `jitter_latreg`, `+perdrone`,
  `+perdrone_michaels`.
- Comb-masked |Δlog-mag| (dB) vs real, DREGON in-flight, k<10:
  baseline 9.00 → randphase 8.29 → jitter 7.26 → **jitter_latreg 7.04**.
  msSTFT 5.67 → **4.85**.
- Caveat: DREGON-calibrated σ does **not** transfer to the M100 at idle (FLY124 =
  four-way tie) → **per-drone σ needed** (dregon σ0.63 / michaels σ0.61).

### 5. E7 — Generated-noise curriculum (sim→real) — 2026-07-11
- Doc: `docs/experiments/e7-gen-curriculum.md` (⚠️ conclusion still says "pending run")
- Gen-only pretrain → real finetune; 3 archs × 2 stages.
- Introduced the vicinal **`interp` mode**: novel drones sampled along the
  DREGON↔Michael's embedding segment.

### 6. E8 — Static-comb noise model — 2026-07-11
- Doc: `docs/experiments/e8-static-comb.md` (⚠️ "pending run")
- Analytic comb instead of the neural generator, to *force* harmonic tracking.
  3 archs, stage 1.

### 7. E9 — "Hard" combined generated-noise task — 2026-07-12
- Doc: `docs/experiments/e9-hard-combined.md` (⚠️ "pending run")
- Neural-gen + static-comb combined; 3 archs + 3 real-finetunes.
- **This is where the big bug surfaced.**

### 8. Validation-set contamination fix — 2026-07-12 🔑 PIVOTAL
- No standalone doc; folded into the E10 batch doc + the final report.
- FLY124's **ground warm-up** (~36 rev/s, DJI `flyCState = AssistedTakeoff`) was
  leaking into validation via `min_motor_rps=30`. Raised to **50**; republished
  `DREGON-LM-V4-michaels-valid` (pin `b6ece43d`) plus a `-valid-full` full-envelope
  split for per-regime eval.
- **This reversed the E7–E9 verdict.** Apparent sim→real failure (MSE ~204, R² −10)
  was an artifact of the yardstick; on the clean valid the same models score ~20.

### 9. E10 — Full-flight synthetic training — 2026-07-12 ❌ NEGATIVE
- Doc: `docs/experiments/e10-full-flight.md` (the batch doc E11/E12 hang off)
- Added: full-flight RPS envelope (ground→warm-up→take-off→cruise→landing→ground),
  physical amplitude scaling (sound power ∝ rps^5 → pressure ∝ rps^2.5), and
  `balance_rps` (flatten the RPS histogram of generator training data).
- Ran deliberately **without** augmentation ("don't make the task harder prematurely").
- **Failed**: overfit the synthetic domain — under-read real cruise (51 vs 80 rev/s).
  Lesson: augmentation is a domain-gap *reducer*, not added difficulty.

### 10. E11 — Silence-gated generator + augmented full-flight + real finetune — 2026-07-12
- Batch doc: `docs/experiments/e10-full-flight.md`
- **Emitter-level smoothstep silence gate**: at rps=0 the oscillator phase freezes and
  sin(φ) becomes a DC constant → the generator could never be silent. Gate multiplies
  each rotor's waveform by smoothstep(rps/10), so a stopped rotor is *exactly* silent.
- Retrained generator `e11_noisegen_silence`; then:
  - `e11_full_aug_{transformer,unigru128,scv2}` — sim full-flight, augmentation restored
  - `e11_full_ft_warp_{...}` — real finetune with time-warp (the curriculum)
  - `e11_real_warp_{...}` — real-only baselines on the full-envelope valid

### 11. E12 — Real full-flight diagnostic — 2026-07-12 🏆 THE WINNER
- Configs + docs: `conf/experiment/e12_real_fullflight_{transformer,unigru128,scv2}.{yaml,md}`
- Same recipe as the real-only baseline, but **`min_motor_rps: 0`** — keep the whole
  powered envelope so the **real** take-off ramp becomes training data.
  (Mechanism: `online_mixing.py:113` `_inflight_window` keeps `[first,last]` where all
  motors > threshold; 0.0 keeps everything from when the motors start spinning.)
- **Best model overall.**

---

## Headline results (per-regime PIT-MSE, full-envelope real valid: 27 cruise / 6 warm-up / 4 ground clips)

| Training data | Cruise | Warm-up | Ground | **All** |
|---|---|---|---|---|
| *Transformer* | | | | |
| real-only (cruise-trained) | 15.3 | 384.9 | 2450.0 | 338.4 |
| sim full-flight curriculum | 48.3 | 463.7 | 198.0 | 131.9 |
| **real full-flight (min_rps=0)** | 20.4 | **149.4** | **374.8** | **79.6** |
| *Uni-GRU-128* | | | | |
| real-only | 14.6 | 389.0 | 1658.3 | 253.0 |
| sim curriculum | 17.9 | 258.2 | 1153.4 | 179.6 |
| real full-flight | 19.7 | 646.1 | 655.7 | 190.0 |
| *SimpleConv-v2* | | | | |
| real-only | 15.7 | 241.9 | 1227.8 | 183.4 |
| sim curriculum | 20.6 | 301.8 | 634.4 | 132.5 |
| real full-flight | 43.3 | 476.5 | 335.2 | 145.1 |

**Mean-collapse refuted:** every model's cruise mean-prediction ≈ 79 rev/s (truth 78.9),
none flat at the ~49 global mean. They *track*; they were only starved of low-speed
training data. (See the tracking figure in the report.)

---

## The through-line for the deck

A single story with a twist:

> generator improvements (E5–E8) → combined task (E9) → **the validation set was lying**
> (FLY124 ground warm-up) → full-flight coverage, synthetic (E10–E11) → **and then the
> punchline (E12): the low-RPS failure was never sim2real at all.** It was a
> `min_motor_rps ≥ 30` filter silently deleting the real take-off ramp from training.
> Removing one threshold beat the whole synthetic pipeline.

Secondary, still-true messages:
- Synthetic full-flight *does* beat the cruise-only baseline — any low-speed coverage helps.
- Synthetic remains the **only** source for **true silence** (0 rev/s), the one regime
  nobody nails (best model still guesses 10–15 rev/s on a stopped drone), because
  near-silent audio carries no comb to read. Complement, not substitute.

---

## Intermediate reports

| Report | Date | Covers |
|---|---|---|
| `writing/reports/2026-07-06_gp-rotor-noise/` | 07-06 (rev 07-10) | companion to the *last* deck — GP/FWH rotor noise |
| `writing/reports/2026-07-10_rps-refinement/` | 07-10 | stages A–D, the label-bias reframe |
| `writing/reports/2026-07-12_full-flight-sim2real-rps/` | 07-12 | **8-page report**: contaminated valid, full-flight envelope, silence gate, augmentation, 3-condition results + mean-collapse refutation |

---

## Caveats to carry onto slides

1. **E7 / E8 / E9 doc conclusions still say `_Pending run._`** — never backfilled after
   the valid-set fix reversed their verdict. Worth correcting before they mislead anyone.
2. **Small-n low-RPS regimes**: only 4 ground + 6 warm-up validation clips, so those
   per-regime numbers are noisy — visible in the arch inconsistency (Uni-GRU's real
   full-flight is *worse* than its curriculum, unlike the other two archs). The
   Transformer result is the clean, defensible headline; the regimes would firm up with
   more full-envelope validation recordings.
3. The E10→E11→E12 work all shares one batch doc (`e10-full-flight.md`); E11/E12 have no
   separate batch write-up beyond their per-experiment `.md` config docs and the report.
