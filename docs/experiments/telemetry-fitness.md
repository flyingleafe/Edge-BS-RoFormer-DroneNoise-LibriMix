# Trajectory goodness of fit — the harness (6a) and the fitter (6b), issue 17

**Status:** harness built and verified (2026-08-06) · GitHub issue #17 §A-D ·
library `src/tracking/fitness.py` · driver `scripts/telemetry_fitness.py` ·
tests `tests/tracking/test_fitness.py` · smoke JSON
`results/telemetry_fitness/{smoke,smoke_b0_0.25}/`.
The **fitter** (phase 6b, issue #17 "Proposed procedure" steps 1-6) is built and
verified on top of it — see "The fitter" below. No campaign has been run: that
is phase 6c.

This note records the DESIGN and the ACCEPTANCE of the harness, plus a smoke
run. It does not interpret the DREGON numbers — that is phase 6c.

## Why

Issue 17 states the difficulty in one sentence: *a fitted trajectory has far
more freedom than fixed telemetry, so it will fit better whether or not it is
more correct.* Every previous choice in the displacement campaign was made by
eye in the comb explorer. The harness replaces the eye with three numbers, at
fixed degrees of freedom, each carrying its own null.

## The statistic

A candidate trajectory `r(t)` is scored by demodulating the audio at the
carriers it implies (`comb_displacement.demod_comb_bank`, which is
`phase_increment_tracker.demod_bank` plus the half-integer trick), then
reporting **three components, never one**:

| component | definition | what a wrong carrier does |
|---|---|---|
| **broadband** | share of demodulated envelope power OUTSIDE the near-DC region `\|f\| <= dc_hz(k)` | the line leaves DC, so in-band residual rises |
| **phase noise** | `k^2`-weighted mean square of `dr_k(t) = arg(z_k[n] conj(z_k[n-1])) fs_env / (2 pi k)`, **about zero**, per mic | a rate error is a constant `dr`; the mean square carries it |
| **magnitude roughness** | share of `\|z_k(t)\|` power above `rough_cut_hz` | the amplitude beats against the line that was missed |

Three points of design, each of which fixes a way of getting this wrong:

- **About zero, not about the mean.** A scale error makes `dr_k` a nonzero
  CONSTANT. A variance about its own mean would be blind to exactly the error
  the campaign is hunting.
- **`k^2` weighting is WP18's, not an assumption.** `docs/experiments/rps-refine-precision.md`
  measured the optimal weight as `1 / v_k ~ k^2.0` (DREGON) with no `|p_k|`
  factor, and measured the common jitter term to be per-MIC, which is why the
  aggregation keeps microphones separate.
- **The centre is the coherent pulse-pair estimate**
  (`comb_displacement.pulse_pair` / `pulse_pair_bank`, promoted here to one
  vectorized implementation, which `scripts/vk_frontend_probe.py` also used).
  Phase 5 established that the median of wrapped increments biases toward zero
  once the phasor is noise dominated.

The coupled VK envelope solve is deliberately NOT used: phase 5 rejected it
(`docs/experiments/vk-frontend-probe.md`, issue 15) — degenerate on contested
harmonics, 15-110x the cost. Fixed-carrier demodulation is the estimator.

## Fixed degrees of freedom

Per (window, rotor), everything except the carrier is pinned to the window's
REFERENCE trajectory:

- the band `B_k = min(b0 k, 0.45 rate_ref)` Hz — `rate_ref` is the reference's
  mean, never the candidate's, so a 0.5 %-scaled candidate does not get a
  0.5 %-different band;
- the envelope rate, block partition, edge trim, harmonic set;
- the **admission mask**. There is deliberately no envelope-SNR gate: an
  admission rule that read the candidate would hand a flexible trajectory an
  easier cell set, which is the failure mode the issue is about.

`test_degrees_of_freedom_are_identical_across_candidates_and_controls` asserts
the cell count is one value across every candidate and every control.

## Held-out scoring (§A)

Every component is computed per **cell** `(channel, harmonic, time block)`.
A `Holdout` is then nothing but a mask over those axes, which is what makes all
three hold-out families one mechanism:

| hold-out | fit side | score side |
|---|---|---|
| `fit_k_even` / `fit_k_odd` | one harmonic parity | the other |
| `fit_ch_0` | mic 0 | mics 1-7 |
| `fit_blk_0+2+4+6` | half the blocks | the gaps |

Phase 6a has no fitter, so the fit side is recorded and excluded. Phase 6b
hands the same object to the fitter.

## Controls (§B)

`apply_control` returns `(carriers, skip, half)` for all four. `skip[i]` names
the reference rotor whose line carrier `i` sits on, so the collision gate
excludes the CARRIER's own line rather than a slot index.

| control | carrier |
|---|---|
| `on` | the candidate |
| `offcomb` | the same trajectory at the half-integer comb `(k + 0.5) g(t)` |
| `mismatch` | another window of the same recording, this window's audio |
| `permute` | the candidate's rotor rows rolled by one |
| FLY124 | not a carrier — the identical procedure on recalibrated labels, `--dataset fly124` |

### The permutation control is invariant, and that is the finding

**The acoustic components cannot see rotor identity.** They are functions of
the carrier SET, and permuting rows does not change that set; the gate follows
the carrier, so it does not change either. The smoke run confirms it to the
last printed digit on both recordings (`on` and `permute` rows are identical).

This is not a defect, it is the precise form of what the issue's own comment
worked out: a permutation null needs a quantity attached to a rotor
*independently of the carrier*. That quantity is the residual pairing against
telemetry, and there the permutation is a violent null — DREGON `d_rms` goes
from 0 to **10.95 rev/s**, FLY124 to **11.06**. So the permutation control is
run against the residual half of the report, and its invariance on the acoustic
half is an assertion the tests check
(`test_rotor_permutation_leaves_the_acoustic_score_alone`).

**Consequence for 6b:** rotor assignment can never be certified by the acoustic
fit. It must be certified by the residual pairing.

## Residual decomposition (§C) and uncertainty (§D)

`residual_decompose(candidate, reference, ft)` runs one joint least squares of
`d = candidate - reference` on `[reference, d(reference)/dt, 1]`, giving a
**scale** in percent, a **lag** in seconds (a period counter reports the
previous revolution, so a lag appears as a derivative term) and an offset. The
residual after that model is tested against the DREGON tachometer's signature —
bounded by half a 0.269 rev/s step, flat-ish to the 24.85 Hz refresh Nyquist,
structure at 49.7 Hz — and every reading is a number, never a verdict.

Read `design_cond` before `scale_pct`. Phase 6b found that on a cruise window the
rate column and the intercept of that design are collinear, so the scale/offset
split is not identified (DREGON w01: -29.5 % against +27 rev/s, `design_cond`
122). `d_mean`, `d_rms` and `lag_s` are unaffected;
`tracking.telemetry_refit.scale_summary` is the well-posed scale. See "Two
defects the verification found" below.

One honest limitation the harness reports rather than hides: the frozen
protocol's frame grid is 0.032 s = **31.25 Hz**, so the 49.7 Hz refresh line is
NOT resolvable (`f_tach_resolved: false`, `tach_line_ratio: null`). Only
`tach_bound_frac` and the flatness below 14.8 Hz are available on that grid. A
finer grid resolves it, which the synthetic test exercises at 125 Hz.

`bootstrap_scores` resamples cells with replacement over microphone subsets,
harmonic subsets and time blocks and returns mean/sd/2.5-97.5 percentiles for
any scalar derived from the score. Resampling is a re-aggregation of cells
already computed, so 200 resamples cost milliseconds, not another demodulation.

## Acceptance tests (the verification of 6a)

`tests/tracking/test_fitness.py`, 21 tests, all passing. The core is a
synthetic 2-rotor comb (70 / 95 rev/s, slow sinusoidal variation, 2 mics,
k = 1..25, additive noise) where the true trajectory is known. Scores on that
window, `b0 = 1 rev/s`, k = 2..20, 4 blocks:

| candidate | control | broadband | phase noise | roughness | pp centre |
|---|---|---|---|---|---|
| **truth** | on | **0.00282** | **0.00517** | **0.3627** | -0.0035 |
| truth x 1.005 | on | 0.82498 | 0.18659 | 0.4213 | **-0.4021** |
| staircase (0.269 rev/s @ 49.7 Hz) | on | 0.06496 | 0.00855 | 0.4356 | +0.0011 |
| truth | offcomb | 0.80164 | 0.80046 | 0.2780 | -0.1191 |
| truth x 1.005 | offcomb | 0.82603 | 0.92890 | 0.3322 | -0.0728 |
| staircase | offcomb | 0.80385 | 0.84711 | 0.2577 | -0.1187 |

Read:

- **Truth outranks both corruptions on all three components.** The required
  ranking holds, and it holds with the hold-out masks on too
  (`test_holdout_scores_still_rank_truth_first`, all three families).
- **The off-comb null is blind.** Every off-comb row sits at the same
  0.78-0.93 noise level regardless of candidate. The on-comb phase-noise ratio
  scaled/truth is **36x**; the same ratio at the null is **1.16x**
  (`test_null_ratio_beats_the_on_comb_ratio` requires the on-comb ratio to be
  more than twice the null's).
- **The pulse-pair centre reads the injected error directly**: -0.402 rev/s
  against an injected -0.005 x 80 = -0.40 rev/s.

Other assertions: hold-out masks partition their axes; permutation invariance
to 1e-6 relative and pairing collapse by >50x; `residual_decompose` recovers a
pure 0.5 % scale to 0.02 pp and a pure 0.05 s lag to 0.01 s; the staircase
residual reads `tach_bound_frac > 0.9` and a resolvable refresh line on a
125 Hz grid; the bootstrap CI brackets the point estimate; the payload is JSON
serializable; `fitness_stage` appends its diagnostics.

## Smoke run (frozen protocol, 2 windows, 4 controls, 3 candidates)

`python scripts/telemetry_fitness.py --smoke --jobs 6` — 24 units, **41 s**
locally, so no remote job was needed. A second arm at `--b0 0.25`. Windows:
DREGON `free-flight_nosource_room1__w01` (the displacement campaign's window,
rotors 86.10 / 75.54 / 85.68 / 74.72 rev/s) and `FLY124__w02`. `k = 2..40`,
`fs_env = 250 Hz`, 8 blocks, 8 mics, 200 bootstrap resamples.

`b0 = 1.0 rev/s` (`admit_frac` 0.0064 DREGON / 0.0264 FLY124):

| dataset | candidate | control | broadband | phase noise | roughness | pp centre |
|---|---|---|---|---|---|---|
| dregon | telemetry | on | 0.7008 | 1.4278 | 0.2051 | -0.0069 |
| dregon | telemetry | offcomb | 0.6861 | 1.5315 | 0.2117 | +0.0327 |
| dregon | telemetry | mismatch | 0.6551 | 1.5387 | 0.1690 | -0.0040 |
| dregon | scale:0.99458 | on | 0.6678 | 1.3077 | 0.2085 | +0.0197 |
| dregon | lp:5 | on | 0.6890 | 1.3614 | 0.2047 | -0.0046 |
| fly124 | telemetry | on | **0.5158** | **1.1572** | 0.1869 | -0.0429 |
| fly124 | telemetry | offcomb | 0.6603 | 1.4279 | 0.1592 | +0.0556 |
| fly124 | telemetry | mismatch | 0.6609 | 1.5705 | 0.1998 | -0.0020 |
| fly124 | scale:0.99458 | on | 0.5953 | 1.1899 | 0.2014 | +0.0096 |
| fly124 | lp:5 | on | 0.5121 | 1.0612 | 0.1784 | -0.0329 |

`b0 = 0.25 rev/s` (`admit_frac` 0.0280 DREGON / 0.0665 FLY124):

| dataset | candidate | control | broadband | phase noise | roughness |
|---|---|---|---|---|---|
| dregon | telemetry | on | 0.3235 | 0.1167 | 0.0267 |
| dregon | telemetry | offcomb | 0.3086 | 0.1446 | 0.0238 |
| dregon | scale:0.99458 | on | 0.2904 | 0.1182 | 0.0244 |
| dregon | lp:5 | on | 0.3210 | 0.1143 | 0.0249 |
| fly124 | telemetry | on | **0.2990** | **0.1082** | 0.0175 |
| fly124 | telemetry | offcomb | 0.3630 | 0.1245 | 0.0220 |
| fly124 | scale:0.99458 | on | 0.3855 | 0.1321 | 0.0225 |
| fly124 | lp:5 | on | 0.3010 | 0.1093 | 0.0203 |

Sanity only, no interpretation:

1. **The FLY124 negative control behaves.** Its telemetry stands clear of both
   of its own nulls at both capture settings, and multiplying it by the
   withdrawn DREGON factor 0.99458 makes it WORSE (0.516 -> 0.595 at
   `b0 = 1`; 0.299 -> 0.386 at `b0 = 0.25`). That is the direction a
   recalibrated label set must give.
2. **DREGON telemetry does not stand clear of its own nulls** at either
   setting — the on-comb value sits at or above the off-comb and the
   mismatched-telemetry level. The harness therefore has no on-comb lock to
   report for DREGON telemetry here.
3. **Coverage is the binding constraint, again.** `admit_frac` is 0.6-6.6 %.
   The DREGON twins are 0.42 and 0.82 rev/s apart, so at a k-scaled capture of
   `b0` rev/s the interferer is inside the band at EVERY harmonic — 64 of
   4992 cells survive at `b0 = 1`. This is the structural degeneracy the issue
   already names ("no bandwidth and no harmonic contains the displaced line
   while excluding the twin"), now with a number on it. It matches WP18's
   independent conclusion that coverage, not precision, binds.
   `FitnessConfig.gate_band_frac` (`--gate-band-frac`) is the knob that trades
   purity for coverage; it was left at the strict 1.0 for this run.
4. The residual block is exact where it should be: `scale:0.99458` reads
   `scale_pct = -0.542`, `lag_s = 0`, `resid_rms = 0` on both recordings.

## API for phase 6b

```python
from tracking.fitness import (
    FitnessConfig, Holdout, apply_control, window_cells, score_cells,
    bootstrap_scores, residual_decompose, score_window, fitness_stage,
)

cfg = FitnessConfig(k_min=2, k_max=40, b0_revs=1.0, fs_env=250.0, n_blocks=8)

# the expensive half — one demodulation per rotor under one control
cells = window_cells(audio, ft, candidate, reference, cfg=cfg,
                     control="none", partner=None)          # -> list[Cells]

# everything below is a re-aggregation of those cells
score  = score_cells(cells, Holdout.harmonics(0), cfg=cfg)  # -> FitnessScore
ci     = bootstrap_scores(cells, Holdout.none(), cfg=cfg, n_boot=200, seed=0)
resid  = residual_decompose(candidate, reference, ft, cfg=cfg)

# or the whole unit at once (every hold-out + bootstrap + residual)
payload = score_window(audio, ft, candidate, reference, cfg=cfg,
                       holdouts=None, control="none", n_boot=200)

# and as a pipeline stage, which scores the frame's rps without changing it
frame = fitness_stage(reference_entry="rps_meas", cfg=cfg)(frame)
```

`FitnessScore` carries `broadband`, `phase_noise`, `roughness`, `pp_dr`,
`pp_abs`, `snr_median`, `n_cells`, `per_rotor`, and `component(name)`.
`Holdout` constructors: `none()`, `harmonics(fit_parity)`, `channels(fit)`,
`blocks(fit)`. A fitter plugs in by consuming `Holdout.score_mask(...)`'s
complement — the fit side is named, not implied.

## Reproduce

```bash
python -m pytest tests/tracking/test_fitness.py -q
python scripts/telemetry_fitness.py --smoke --jobs 6 --out results/telemetry_fitness/smoke
python scripts/telemetry_fitness.py --smoke --jobs 6 --b0 0.25 \
  --out results/telemetry_fitness/smoke_b0_0.25
python scripts/telemetry_fitness.py --dataset all --jobs 8   # 15 windows x 3 x 4
```

---

# The fitter (issue 17, phase 6b)

**Status:** built and verified (2026-08-06) · library `src/tracking/telemetry_refit.py` ·
driver `scripts/telemetry_refit.py` · tests `tests/tracking/test_telemetry_refit.py`
(27 tests) · smoke JSON + candidate `.npz` in `results/telemetry_refit/smoke/`.

Phase 6a built the judge. This is the thing it judges: the procedure of issue 17
§ "Proposed procedure", all six steps. It is written as WIRING — the peel, the
peel seam and the twin rule are the flagship's own, unchanged — plus the three
things the issue says the displacement campaign was missing.

## The six steps and where each one lives

| step | issue 17 says | here |
|---|---|---|
| 1 | pre-smooth the carrier ~5 Hz | `presmooth(r, ft, cut_hz)` — the SAME function the 6a driver's `lp:` candidate now calls |
| 2 | coarse-to-fine in k, never `k_caps=(80,80,80)` | `k_cap_for_error` / `advance_k`; one rung per outer iteration |
| 3 | alternate with the envelope solve and the peel | one outer iteration IS one `pipelines.pi_kalman_arm_stage` application |
| 4 | least-squares-projected subtraction | `peel_mode="ls"`, the existing default; asserted per iteration via `energy_ok` |
| 5 | stop on convergence, not `n_iter=3` | `tol_rev_s` on `max \|dr\|` + an iteration cap, plus a plateau stop |
| 6 | keep `pair_mode`, twins out of each other's peel | `pair_mode="joint"`; the peel rule verified by test, not rebuilt |

## The advance rule (step 2), and why it is not the band

The k-scaled band is the shape the campaign's only identity-preserving arm used,
and its capture is `b0` rev/s at **every** harmonic by construction. So the band
cannot be what makes one rung safer than the next — it is k-free. The quantity
that is k-dependent is the phase wrap: a residual rate error `e` turns into an
envelope phase increment `2 pi k e / fs_env` per sample, and `pi_kalman_refine`
discards increments past `wrap_guard_rad`. The rung is therefore admissible while

    k <= wrap_guard_rad * fs_env / (2 pi e)

which at the defaults (2.8 rad, 62.5 Hz) is `k <= 27.9 / e`. The error estimate
`e` starts at the prior `e0 = 1.5` rev/s (the issue's own 0.2-0.7 rev/s bias plus
the staircase and the flight wander), and afterwards is the 95th percentile of
the last application's `|dr|`. The **maximum** is not usable as an error estimate:
on real audio it stays near 1 rev/s throughout, dominated by isolated tracker
spikes. Growth is capped at `k_growth = 2` per rung and the rung never steps back
down. Measured ladders: DREGON `18 -> 36 -> 69 -> 79 -> 87 -> 96`, FLY124
`18 -> 36 -> 72 -> 96 -> 96`.

## The stop (step 5)

`tol_rev_s = 0.02` on `max |dr|` is the issue's criterion and it is implemented as
written — and on real audio it never fires, because that maximum sits at
0.45-1.3 rev/s while the trajectory has stopped moving in bulk. So there is a
second stop: when the 95th-percentile update improves by less than `plateau_rel`
(5 %) of the previous iteration's, the alternation has plateaued. Only the
tolerance stop is reported as `converged`; `stop_reason` names which one fired,
and every iteration's `delta_max` / `delta_q` / per-rotor `delta_rms` is recorded.
Both smoke windows stop on `plateau`, at 6 and 5 iterations of a cap of 8.

## What the procedure buys, and what it does not

The trajectory residual is **not** the product. On the synthetic window the
fitter's own per-frame noise is 0.05-0.15 rev/s — comparable to the 0.087 rev/s
tachometer staircase it replaces — while the systematic scale is recovered to
0.005 %. This is issue 17's own warning ("the refined tracks carry two distinct
corrections and they must be reported separately") measured rather than repeated,
and `test_residual_rms_is_reported_and_does_not_beat_the_staircase` pins it.

## Two defects the verification found (both fixed)

**`residual_decompose`'s `scale_pct` is not identified on a cruise window.** Its
per-rotor design is `[r, dr/dt, 1]`, and a rotor holding 86 rev/s to about 1 %
makes the rate column and the intercept collinear, so least squares splits the
systematic part between them arbitrarily. DREGON w01 reads `scale_pct -29.5 %`
with `offset +27 rev/s` — one number, not two. `residual_decompose` now reports
`design_cond` (122 on DREGON, 43 on FLY124) so the reading labels itself, and
`telemetry_refit.scale_summary` gives the two well-posed alternatives:

- `per_rotor_pct` = `100 d_mean / mean_rate`, no regression at all;
- `global_pct` = ONE shared scale over all four rotors, `100 sum(d r)/sum(r r)`,
  well conditioned because the rotors sit at genuinely different rates.

**A whole-window FFT brickwall rings on a drifting trajectory.** `brickwall`
treats the series as periodic, and a window whose rate drifts end to end carries
a step at the wrap. On the synthetic staircase the bare filter made the carrier
WORSE — 0.087 -> 0.112 rev/s. `presmooth` now removes the least-squares line
first (0.087 -> 0.075), and since 6a's `lp:` candidate calls `presmooth`, both
halves of the campaign get the fix. **The `lp:5` rows of the 6a smoke tables
above predate it.**

## Acceptance: synthetic recovery (the verification of 6b)

`data_processing.rps_synthesis.synth_comb_window`, 4 s, 2 mics, `k <= 30`,
6 dB comb-to-noise, rotor means 84.8 / 66.1 / 97.3 / 90.7 rev/s (no twin pair, so
this tests the ladder rather than the twin logic). Truth known exactly; the
carrier handed to the fitter is the truth corrupted the two ways DREGON's
telemetry is corrupted. Fitter at `k_top = 24`, 4 iterations, 3 s per case.

| corruption | recovered-vs-TRUE global scale | per-rotor | reported scale vs the carrier | trajectory rms vs truth |
|---|---|---|---|---|
| `x 1.005` | **+0.0012 %** | -0.0002 .. +0.0050 % | **-0.4963 %** (injected -0.4975) | 0.050-0.151 |
| staircase 0.269 rev/s @ 49.7 Hz | **+0.0012 %** | -0.0009 .. +0.0050 % | **+0.0014 %** (correctly ~0) | 0.054-0.154 (carrier: 0.082-0.091) |
| both | **+0.0019 %** | -0.0015 .. +0.0055 % | **-0.5036 %** | 0.053-0.152 |

Read: the scale is recovered two orders of magnitude inside the 0.1 % bar, the
zero-mean staircase is correctly read as no systematic shift, and the per-frame
trajectory is not improved. Rotor order and every inter-rotor gap survive
(gaps within 10 %).

## Acceptance: the two real-data smokes

`python scripts/telemetry_refit.py --smoke --jobs 2`, 6 FFT workers, this laptop:
**~150 s per window**, both in ~4.5 min wall. No remote job was needed.
`arm = main` (the procedure exactly as the issue states it), cap 8 iterations.

| | DREGON `free-flight_nosource_room1__w01` | FLY124 `w02` |
|---|---|---|
| rotor means, rev/s | 86.10 / 75.54 / 85.68 / 74.72 | 90.39 / 74.07 / 79.14 / 75.72 |
| k ladder | 18, 36, 69, 79, 87, 96 | 18, 36, 72, 96, 96 |
| stop | plateau, 6 iterations | plateau, 5 iterations |
| **global scale** | **-0.395 %** | **-0.066 %** |
| per-rotor scale | -0.449 / -0.473 / -0.271 / -0.369 % | -0.143 / -0.072 / +0.044 / -0.062 % |
| `d_rms` (the de-staircasing part) | 0.883 | 0.671 |
| `resid_rms` after the systematic model | 0.720 | 0.663 |
| rotor order kept | yes | yes |
| gap ratios (fit / raw) | **0.70** / 1.01 / 0.87 | 0.99 / 1.02 / 1.00 |
| `design_cond` of the 6a regression | 122 | 43 |

Three readings, none of them a verdict — the verdict needs 6a's controls over
the whole window set, which is 6c:

1. **The FLY124 negative control passes.** The identical procedure returns
   **-0.066 %** where the labels were recalibrated, against the issue's published
   -0.063 % from a completely different estimator. It is 6x smaller than the
   DREGON reading, and the per-rotor values straddle zero. This is the check the
   issue calls "the single most valuable", and the fitter is not simply flexible:
   the flexibility would have shown up here.
2. **DREGON reads -0.395 %**, inside the -0.19..-0.55 % band the other estimators
   span and larger than the campaign's own `B0=1` arm (-0.313 %), which had none
   of steps 1, 2, 3 or 5.
3. **The twin gap is the caveat.** DREGON's r0-r2 gap goes 0.422 -> 0.296 rev/s
   (ratio **0.70**) while every FLY124 gap moves less than 2.4 %. Rotor order
   survives, so this is not the wide-fixed-band collapse (which sign-flipped the
   gap), but it is the same direction and it is **not controlled**: FLY124 has no
   pair closer than 1.65 rev/s, so it has no twin gap to collapse. The comparison
   proves nothing about twins. 6c must separate "the twins genuinely converge" from
   "the fitter pulls them together" — and per 6a that can only be settled on the
   residual pairing, never on the acoustic fit.

## The candidate file format (the 6a <-> 6b seam)

One `.npz` per window, written by the refit driver, read by the fitness driver:

    results/telemetry_refit/<out>/traj/<arm>/<window_key>.npz
      ft      (N,)    frame times, audio-relative seconds
      r_raw   (R, N)  the untouched telemetry
      r_init  (R, N)  the pre-smoothed carrier the fit started from
      r_fit   (R, N)  the refined trajectory, on this window's own ft grid

6a's candidate language already had the hook (`file:PATH:KEY` loads
`np.load(PATH)[KEY]`). The only extension is that **`{key}` in a candidate spec is
replaced by the window key**, so one spec scores a whole directory:

```bash
python scripts/telemetry_refit.py --dataset all --arms main --jobs 4
python scripts/telemetry_fitness.py --dataset all \
  --candidates 'telemetry,file:results/telemetry_refit/traj/main/{key}.npz:r_fit'
```

Verified end to end on the smoke pair.

## Arms (for 6c, not run here)

Each named arm of `scripts/telemetry_refit.py` turns exactly ONE step off, so the
campaign can say what each step bought instead of shipping a bundle: `main`,
`nosmooth` (step 1), `flatk` (step 2 — the old `k_caps=(80,80,80)`), `nopeel`
(step 3), `gate` (step 6), and `b0_3` (the campaign's other identity-preserving
band).

## Reproduce

```bash
python -m pytest tests/tracking/test_telemetry_refit.py -q
TRACKING_FFT_WORKERS=6 python scripts/telemetry_refit.py --smoke --jobs 2 \
  --out results/telemetry_refit/smoke
```
