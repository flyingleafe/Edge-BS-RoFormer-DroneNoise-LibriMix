# Trajectory goodness of fit — the harness (6a), the fitter (6b), the campaign (6c), the sensitivity fix (6d), the time shift (6e)

> **Read 6d first if you are citing a number.** Phase 6d added the component the
> first three lacked (line power against a local floor) and re-scored everything:
> the verdicts hold, the DREGON scale is -0.68 % [-0.88,-0.53], and the refined
> trajectory is measured to lock 2.47 dB better than the labels.
>
> **Phase 6e adds a second correction and reverses a theory.** DREGON's
> telemetry also carries a time offset of **-42 ms [-85,-31]** — it runs EARLY,
> not late, so a tachometer reporting lag is excluded by sign. The
> per-microphone differential the theory predicts is 0.156 ms, which is ~500x
> below what the ridge resolves; the delay itself is real and is confirmed on
> the michaels rig at slope 1.013 by a phase estimator, not by a rate shift.

**Status:** harness built and verified (2026-08-06) · GitHub issue #17 §A-D ·
library `src/tracking/fitness.py` · driver `scripts/telemetry_fitness.py` ·
tests `tests/tracking/test_fitness.py` · smoke JSON
`results/telemetry_fitness/{smoke,smoke_b0_0.25}/`.
The **fitter** (phase 6b, issue #17 "Proposed procedure" steps 1-6) is built and
verified on top of it — see "The fitter" below. The **campaign** (phase 6c) is
run — see "The campaign" at the end, and the decision it produced in
`docs/experiments/dregon-comb-displacement.md`.

This note records the DESIGN and the ACCEPTANCE of the harness, plus a smoke
run. The campaign section holds the measurement.

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
  vectorized implementation, which the since-deleted `vk_frontend_probe`
  driver also used).
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

---

# The campaign (issue 17, phase 6c)

**Status:** run and analyzed (2026-08-06) · jobs `telemetry-6c-701374` and
`telemetry-6c-g025-ca3395` on `uni-cpu`, 16 cores, 36 min + 20 min wall,
2640 units, **zero failures** · driver `scripts/telemetry_campaign.sh` ·
reader `scripts/telemetry_report.py` · JSON `results/telemetry_report.json`
and `results/telemetry_report_g025.json` · unit JSON under
`results/telemetry_{refit,fitness}/` (uncommitted, 55 MB).

The campaign fits all 15 protocol windows under six arms, then judges every
candidate at fixed degrees of freedom against all four controls, then profiles
a one-parameter scale family. DREGON is the measurement. FLY124 is the negative
control, and it runs through the identical procedure.

## Provenance

The prep cache is a gitignored artifact, so the cluster rebuilt the 15 windows
from the pinned dataset `beatvk-valid-raw@54849c13ed3a`. Every unit carries the
window's `prep_sha1`, and all 15 fingerprints are identical to the local pulled
cache. The rebuilt windows are the same windows, byte for byte.

## Coverage decided the settings, and geometry decided coverage

At the smoke's capture (`b0 = 1` rev/s, strict gate) the conditioning gate
admits **0.09 %** of DREGON cells, and **7 of the 9 DREGON windows have no
admitted cell at all**. Every DREGON number of the first job is therefore
noise from two windows. FLY124 has 264-608 cells per window at the same
setting.

The gate reads the REFERENCE only, so coverage is a function of the telemetry
geometry and of `b0 * gate_band_frac` — no audio, no candidate, no score. That
makes the setting selectable without touching a result:

| `b0` x `gate_band_frac` | DREGON admit | windows with cells | FLY124 admit |
|---|---|---|---|
| 1.00 | 0.0009 | 2/9 | 0.048 |
| 0.50 | 0.0077 | 6/9 | 0.075 |
| 0.25 | 0.0235 | 8/9 | 0.107 |
| **0.125** | **0.0655** | **9/9** | 0.158 |
| 0.05 | 0.152 | 9/9 | 0.287 |

The campaign therefore reports two settings: the strict one (`b0 = 1`,
`gate = 1.0`), which has no DREGON coverage, and `b0 = 0.5`,
`gate_band_frac = 0.25`, the loosest product that admits cells in every window
while the band stays wider (0.5 rev/s) than the displacement being measured
(~0.3-0.7 rev/s). The rule is coverage, fixed before any score was read.

## The four controls

`b0 = 0.5`, `gate_band_frac = 0.25`, hold-out `none`, broadband component
(lower is better), pooled over windows:

| candidate | on | offcomb | mismatch |
|---|---|---|---|
| DREGON telemetry | **0.6644** | 0.6671 | 0.6644 |
| DREGON `lp:5` | 0.6627 | 0.6634 | 0.6644 |
| DREGON `scale:0.99458` | 0.6464 | 0.6565 | 0.6644 |
| DREGON `fit:main` | **0.5927** | 0.6607 | 0.6644 |
| FLY124-cruise telemetry | **0.4589** | 0.5712 | 0.5739 |
| FLY124-cruise `scale:0.99458` | 0.6508 | 0.5656 | 0.5739 |
| FLY124-cruise `fit:main` | 0.4397 | 0.5610 | 0.5739 |

Four readings, and the first is the campaign's central result:

1. **DREGON's raw telemetry has no acoustic lock.** Its on-comb score equals
   its own off-comb null and equals a mismatched window's telemetry, to the
   third decimal. FLY124's telemetry clears both of its nulls by 0.11. The
   labels of one drone sit on its comb and the labels of the other do not.
2. **Every fitted DREGON trajectory clears both nulls** (0.593 against 0.661
   and 0.664). This is the discrimination issue 17 asks for, and it passes.
3. **The withdrawn factor 0.99458 destroys the FLY124 lock** — 0.459 becomes
   0.651, which is worse than the null. So the statistic resolves a 0.542 %
   scale error easily where it has coverage, and FLY124 needs no such
   correction.
4. **The permutation null collapses the residual pairing**: `d_rms` goes from
   0.80 to 10.35 rev/s on DREGON and from 0.44 to 11.68 on FLY124. The
   acoustic components are permutation-invariant by construction, as phase 6a
   established, so the permutation is read on the pairing.

## The scale, two estimators that do not agree

**The one-parameter profile (fixed DOF).** The family `lp:5+scale:s` has a
single free parameter, so its minimum cannot be bought with flexibility. 29
values of `s` from -1.20 % to +0.20 %, per window, pooled:

| group | broadband | phase noise | hold-outs (broadband) |
|---|---|---|---|
| **DREGON** | **-0.77 % [-0.95,-0.59]** | -0.73 % [-0.82,-0.56] | -0.66 / -0.72 / -0.80 / -0.81 |
| **FLY124 cruise** | **-0.01 % [-0.05,+0.01]** | -0.06 % [-0.25,+0.02] | -0.00 / -0.02 / -0.04 / -0.01 |
| DREGON off-comb null | -1.08, depth 0.018 | edge, depth 0.03 | — |

The hold-out families (even k, odd k, mic 0, half the blocks) agree inside each
other's intervals on DREGON, the off-comb null has no interior minimum and
one third of the depth, and the FLY124 control returns zero. The campaign's
definition of done is met by this row.

**The fitter's own mean shift.** The refit moves the trajectory and reports
`scale_summary.global_pct`, one shared scale per window:

| group | global scale | per recording |
|---|---|---|
| **DREGON** (9 windows) | **-0.347 % [-0.394,-0.288]** | -0.380 / -0.309 / -0.353 |
| **FLY124 cruise** (4) | **-0.038 % [-0.062,-0.014]** | — |
| FLY124 warmup (2) | +0.142 % [-0.046,+0.331] | out of validity, see below |

Per rotor on DREGON: -0.43 / -0.32 / -0.31 / -0.30 — all four negative and of
one size. **Twin capture predicts alternating signs** (each rotor is pulled
toward its twin, which is below for two rotors and above for the other two),
so the common sign refutes twin capture as the explanation of the shift. The
per-rotor profile minima say the same: rotors 2 and 3, whose twin traps lie at
+0.49 % and +1.10 %, put their minima at -0.71 % and -0.49 %, away from their
traps.

**The two estimates differ by a factor of two, and the difference is not
statistical** — the intervals do not touch. They are also not the same
quantity. The profile asks "what single constant best explains the comb"; the
fitter asks "where does a free trajectory settle". The fitted trajectories beat
every constant (0.593 against the best constant's 0.639), so a constant is
known to be an incomplete model of the label error, and the mean shift of a
shape-changing fit has no reason to equal the best constant. What the campaign
can say is that both are negative, both pass FLY124, and the systematic between
the two methods (0.4 pp) is 4-8x either one's statistical interval.

## The pulse-pair centre is inert on DREGON

`pp_dr` is a rate error in rev/s, so it looks like a third scale estimator, and
on DREGON it reads about zero. It is not evidence. The response gain is
measurable, because `scale:0.99458` is a known injection of -0.542 %:

| dataset | telemetry | `scale:0.99458` | difference | gain |
|---|---|---|---|---|
| FLY124 cruise | -0.010 % | +0.161 % | 0.171 | **0.32** |
| DREGON | -0.025 % | -0.015 % | 0.010 | **0.02** |

DREGON's pulse-pair estimator returns 2 % of a known displacement. Its ~0
reading measures its own inertia. This is the low-SNR shrinkage the
displacement campaign already recorded (pulse-pair -0.231 against the ridge's
-0.424), now with a number on the gain.

## Ablation: what each of the six steps bought

DREGON global scale beside the FLY124-cruise control of the same arm:

| arm | step turned off | DREGON | FLY124 cruise | `d_rms` | order kept | min gap ratio |
|---|---|---|---|---|---|---|
| `main` | — | -0.347 [-0.394,-0.288] | -0.038 | 0.798 | 8/9 | 0.380 |
| `nosmooth` | 1, pre-smoothing | -0.363 [-0.408,-0.311] | -0.023 | **0.531** | 8/9 | 0.530 |
| `flatk` | 2, the k ladder | **-0.261** [-0.318,-0.202] | -0.038 | 0.756 | 8/9 | 0.348 |
| `nopeel` | 3, the peel | **-0.442** [-0.501,-0.358] | -0.065 | 0.853 | 8/9 | **0.138** |
| `gate` | 6, `pair_mode` | -0.299 [-0.356,-0.236] | -0.037 | 0.752 | 8/9 | 0.406 |
| `b0_3` | (wider band) | -0.357 [-0.530,-0.187] | **-0.316** | 1.102 | 6/9 | **0.101** |

Read:

- **`b0_3` fails its control.** It finds -0.32 % on recalibrated FLY124 labels
  and it damages rotor identity on both datasets. The wide band is estimator
  flexibility, exactly as issue 17 predicts, and the arm is disqualified. Note
  that no other arm was chosen by this test: the five remaining arms all pass,
  and they span -0.26 to -0.44 %.
- **The peel is what holds the twins apart.** Without it the shift grows 27 %
  and the smallest gap ratio falls to 0.138. A larger number with collapsed
  gaps is the failure mode of the whole campaign, not a better measurement.
- **The k ladder costs 25 % of the shift** (`flatk` -0.261 against -0.347).
  The old `k_caps=(80,80,80)` puts the decision on out-of-capture harmonics.
- **The pre-smoothing does not move the scale** (-0.363 against -0.347) but it
  halves the per-frame movement (`d_rms` 0.531 against 0.798). It buys the
  separation of the two corrections, not the correction itself.
- The arm spread, -0.26 to -0.44 %, is the fitter's own systematic. It is
  larger than any arm's window-to-window interval.

## The residual, reported separately

`residual_decompose` of `fit - raw telemetry`, DREGON `main`, pooled:

| quantity | DREGON | FLY124 cruise |
|---|---|---|
| systematic mean shift `d_mean` | **-0.261 rev/s** | -0.029 |
| total `d_rms` | 0.798 | 0.441 |
| `resid_rms` after the systematic model | **0.688** | 0.429 |
| `lag_s` | +0.0011 | -0.0013 |
| `tach_bound_frac` (share inside half a step) | 0.19 | 0.36 |
| `tach_flatness` below 14.8 Hz | 0.46 | 0.41 |
| `design_cond` of the 6a regression | 79.8 | 96.9 |

The two corrections are of different sizes and different kinds: the systematic
part is 0.26 rev/s and the part that remains after it is 0.69 rev/s. They are
reported separately here and must stay separate.

**The tachometer-signature test does not pass, and cannot on this grid.** Only
19 % of the DREGON residual falls inside the tachometer's half-step bound
(+-0.135 rev/s), so the residual is about five times larger than pure
de-staircasing. The 49.7 Hz refresh line is **not resolvable** on the frozen
protocol's 31.25 Hz frame grid (`f_tach_resolved: false`,
`tach_line_ratio: null`), so the sharpest form of the test is unavailable
without a finer grid. What the residual carries beyond the staircase is the
fitter's own per-frame noise, which phase 6b measured at 0.05-0.15 rev/s on
synthetic audio and which is clearly larger on real audio.

## Rotor identity

Rotor order survives in 14 of the 15 windows under `main`. The one exception,
`free-flight_whitenoise-low_room1__w02`, has a raw twin gap of 0.120 rev/s —
the two rotors are the same rate inside the label's own quantisation step, so
the order there has no meaning to preserve.

The twin gaps scatter both ways and do not collapse: the r0-r2 gap ratio over
the nine DREGON windows is 1.25, 0.70, 0.86, 1.18, 0.55, 0.94, 1.14, 0.38,
0.97 — mean 0.94, spread +-50 %. Phase 6b's single smoke window read 0.70 and
called it a caveat; over nine windows it is noise around 1, not a trend. The
`nopeel` and `b0_3` arms are the ones that damage gaps (0.138 and 0.101).

## Rate dependence: weak evidence for a tick miscount

The nine DREGON windows sit at two rate clusters (60-68 and 80 rev/s), and the
scale grows with rate:

    scale % = -0.0089 * rate + 0.32,  slope 95 % CI [-0.076, -0.0034]

A pure scale predicts slope 0; the fixed tick miscount of the displacement
campaign (`d = -c r^2`, `c = 6.7e-5`) predicts -0.0067. The interval excludes
zero and contains the tick-miscount prediction. This is suggestive, not
settled: nine windows, two rate clusters, and the low-rate windows are also the
harder ones.

## Reproduce

```bash
bash scripts/telemetry_campaign.sh 16          # the three stages, 15 windows
python scripts/telemetry_report.py             # the strict-gate reading
python scripts/telemetry_report.py \
  --fit results/telemetry_fitness/campaign_g025 \
  --profile results/telemetry_fitness/scale_profile_g025 \
  --out results/telemetry_report_g025.json     # the coverage-adequate reading
```

---

# Phase 6d — the sensitivity fix (issue 17, the challenge to 6c)

**Status:** run and analyzed (2026-08-06) · library `tracking.fitness`
(component 4, `line_power` / `line_masks`, `admit_ridge`) · driver
`bash scripts/telemetry_campaign.sh <jobs> 14` · tests
`tests/tracking/test_fitness.py` (27).

Provenance, because it is not one job:

- `telemetry-6d-0a0540` (`uni-cpu`, 16 workers) produced the **scale profile**
  (870/870 units) and the non-fitted candidates of the control table. Its refit
  stage was OOM-killed, so its 352 fitted-candidate units are `.err`.
- the **fitted-trajectory re-score** (`arm = main`, all 15 windows, 3 controls)
  was run locally against the 6c trajectories in
  `results/telemetry_refit/campaign/traj/main/`, which are the same artifacts
  6c scored — `results/telemetry_fitness/campaign_6d_local/`.
- `telemetry-6d2-e6c967` (12 workers, 96 GB) re-runs the refit and the remaining
  five ablation arms.

Unit JSON: `results/telemetry_fitness/{campaign_6d,campaign_6d_local,scale_profile_6d}/`
(the first two pulled to `omnirun-outputs/telemetry-6d-0a0540/`).

## The challenge

6c reported DREGON raw telemetry at **0.6644** against its own off-comb null of
**0.6671** — no lock, to the third decimal — and concluded from it. The
objection: the telemetry-versus-harmonics mismatch is plainly visible on a
spectrogram, and a few `pi_kalman` iterations visibly improve the fit. A
statistic that cannot separate a visibly-nearly-correct carrier from a
half-integer comb is not measuring what the eye measures, and a decision resting
on it is unsafe.

The objection is right about the statistic and wrong about the conclusion, and
both halves needed measuring rather than arguing.

## The diagnosis, on 6c's own cells

`window_cells` re-run over all 15 protocol windows at the campaign's settings,
per cell rather than pooled: 89,856 DREGON cells (4 rotors x 8 mics x 39
harmonics x 8 blocks) and 39,936 FLY124-cruise cells.

### (a) "The gate excludes the visible low-k cells" — REFUTED as stated

The admitted cells are the LOW harmonics, not the high ones, and their SNR
distribution is the population's:

| at `b0 = 1.0`, `gate_band_frac = 0.25` | DREGON | FLY124 cruise |
|---|---|---|
| conditioning gate admits | 2.35 % | 9.19 % |
| mean `k` of admitted cells (all cells: 21.0) | **6.6** | 5.6 |
| per-cell SNR median, admitted / all | 1.15 / 1.04 | 0.87 / 1.02 |
| share of cells with SNR > 1, admitted / all | 0.56 / 0.52 | 0.48 / 0.52 |

So the gate is not an SNR filter and not a high-`k` filter. At the 6c reading's
own setting (`b0 = 0.5`, `gate 0.25`) it admits 6.55 % of DREGON cells with mean
`k` 9.4 and median SNR 1.09 against the population's 1.04.

### (a') What IS true, and it was not reported: the two sides saw different combs

The quantity the gate destroys is not SNR, it is **line energy**. Per cell, the
floor-subtracted line power (the new component's own numerator) says how much of
the comb a cell set contains:

| share of the comb's total line energy inside the gate | DREGON | FLY124 cruise |
|---|---|---|
| conditioning gate | **5.7 %** | **18.6 %** |
| ridge gate (phase 6d) | 6.9 % | 20.5 % |

The 6c campaign compared a DREGON number computed on 5.7 % of DREGON's comb with
a FLY124 number computed on 18.6 % of FLY124's — a **3.3x asymmetry between the
measurement and its own negative control**, which no reported quantity carried.
`line_share_gated` now travels in every unit so this cannot recur.

### (b) "The shares saturate" — CONFIRMED, and it is not the whole story

All three 6c components are SHARES of the same in-band power, so they are bounded
and they compress toward their noise value as the envelope becomes noise
dominated. On DREGON they are AT that value everywhere — the on-comb and
off-comb readings agree to 0.01 in every SNR band, including the top 5 %:

| DREGON, broadband, conditioning-gate cells | on | off-comb |
|---|---|---|
| SNR 0.04-0.52 | 0.7565 | 0.7566 |
| SNR 0.52-0.91 | 0.7941 | 0.7919 |
| SNR 0.91-1.42 | 0.7814 | 0.7807 |
| SNR 1.42-2.45 | 0.7800 | 0.7706 |
| SNR 2.45-9.69 | 0.6889 | 0.6810 |
| SNR 9.69-258 | 0.4220 | 0.4310 |

But the same table on FLY124 separates at EVERY SNR band, by 0.14 to 0.34
(0.5559/0.7014, 0.5797/0.7368, 0.6506/0.7842, 0.5390/0.7709, 0.2095/0.5751,
0.0557/0.3946). Two recordings, one statistic, one setting: saturation with SNR
is real but it is not what distinguishes them. What distinguishes them is
whether the label carrier has a line on it at all.

### (c) "There is no magnitude-concentration component" — CONFIRMED, and fixed

Nothing in the harness asked the question the eye asks: *is there a line here,
and how far does it stand above the noise around it?* That question is not a
share of anything, so no amount of re-weighting the first three components
produces it.

## The new component: ridge concentration

`FitnessScore.ridge`, dB, and the only component where **more is better**:

    ridge = 10 log10( mean power in |f| <= dc_hz(k) / floor density )

on the demodulated envelope spectrum of one cell, where the floor density is the
MEDIAN of an annulus of the same block's spectrum, divided by `ln 2` (the
median-to-mean factor of an exponential periodogram bin, so a pure-noise cell
reads 0 dB rather than +1.6). It is the phase-7 generator readout — a fixed band
against a local floor, never a peak search, `--self-test` flat to 0.008 dB —
moved into the demodulation domain, and it is literally the same function:
`tracking.fitness.line_power`, which `scripts/gen_label_sensitivity_eval.py` now
calls (a test pins the two readings equal).

Four design points, each fixing a way of getting this wrong:

- **Hann on the ridge spectrum only.** A rectangular block leaks a strong line
  at -13 dB into its own first sidelobe and decays 6 dB/octave, which puts the
  LINE into the floor region and compresses the ratio being reported. Hann is
  -31 dB and 18 dB/octave. Components 1-3 keep the untapered spectrum: they are
  shares of the same total and a taper only reweights it.
- **The annulus never overlaps the line region.** Its inner edge is pushed out
  to `dc_hz + res_hz`. At low `k` the resolution floor makes `dc_hz` comparable
  to the whole band, and the first cut of this component read +2995 dB there —
  a floor estimate made of the line, divided by a clamp. A harmonic with fewer
  than `min_floor_bins` annulus bins has no ridge reading and says so (NaN).
- **Twins are scored, not gated — by a floor definition.**
  `comb_displacement.interloper_offsets_hz` returns every foreign line that
  lands in band, and each is excised from the floor region (option (ii) of the
  two the issue allows; the joint two-carrier fit is the other, and it would add
  the free parameter this harness exists to avoid). The excision positions are
  `(reference fact) - (this carrier)`, so a candidate whose band sits elsewhere
  excises the interferer where it actually is; if nothing survives the excision
  the floor falls back to the full annulus, which can only LOWER the ridge.
- **Its own gate.** `admit_ridge` requires the nearest foreign line to be
  outside `ridge_clear * dc_hz(k)` — resolved away from DC — not absent from the
  band. Both gates still read the REFERENCE only, and
  `test_degrees_of_freedom_are_identical_across_candidates_and_controls` now
  asserts BOTH cell counts are one value across every candidate and control.

One asymmetry of the excision rule, stated because it is not obviously
harmless: a candidate whose error happens to equal a sibling's offset (DREGON's
label error, 0.3-0.8 rev/s, against a twin separation of 0.42) puts its OWN
displaced line inside the excised zone, which removes that line from the floor
and can only RAISE its ridge. The direction is conservative for every claim
below — it inflates raw telemetry's reading, and raw telemetry is the candidate
this note concludes has no lock.

Coverage, honestly: at `b0 = 1` the ridge gate admits 5.48 % of DREGON cells
against the conditioning gate's 2.35 %, and 6.9 % of the line energy against
5.7 %. It is **not** the 96 % a two-rotor geometry would give, because at high
`k` DREGON's four-rotor comb is genuinely unresolvable — the interloper spacing
falls below the line region's own width (`dc_revs = 0.1` rev/s, i.e. `0.1 k` Hz)
somewhere near `k = 20`. That is the structural degeneracy the issue names, not
a gate that can be loosened: `ridge_clear` trades it, and the trade is reported
with the number.

## Acceptance (the same battery as 6a, extended)

`tests/tracking/test_fitness.py`, 27 tests. On the synthetic 2-rotor comb
(70 / 95 rev/s, truth known), `b0 = 1`:

| candidate | broadband | ridge, on | ridge, off-comb null |
|---|---|---|---|
| **truth** | 0.0028 | **+34.3 dB** | +0.67 |
| staircase (0.269 rev/s @ 49.7 Hz) | 0.0650 | +23.1 dB | +0.52 |
| truth x 1.005 | 0.8250 | -1.5 dB | +0.26 |

Truth outranks both corruptions; every off-comb row sits at 0 dB within 0.7,
which is the `ln 2` correction working; and a 0.35 rev/s rate error collapses the
ridge to the null because it puts the line outside a 0.10 rev/s window. A second
synthetic window with a DREGON twin pair (86.10 / 85.68 rev/s) pins the coverage
claim: the conditioning gate admits **0 %** of its cells — the components 1-3
score does not exist there at all — while the ridge gate admits **73.7 %** and
reads **+33.6 dB** on the truth.

## The re-score: controls (15 windows, `b0 = 1.0`, `gate_band_frac = 0.25`)

Hold-out `none`, pooled over windows. `ridge` is dB and **higher is better**;
the other three are residual shares and lower is better.

| candidate | control | broadband | phase noise | **ridge** |
|---|---|---|---|---|
| DREGON telemetry | on | 0.7518 | 1.233 | **-0.60** |
| DREGON telemetry | offcomb | 0.7430 | 1.238 | -0.66 |
| DREGON telemetry | mismatch | 0.7507 | 1.290 | -0.84 |
| DREGON `lp:5` | on | 0.7515 | 1.232 | -0.48 |
| **DREGON `scale:0.99458`** | on | 0.7213 | 1.141 | **+0.14** |
| DREGON `scale:0.99458` | offcomb | 0.7510 | 1.251 | -0.64 |
| **FLY124-cruise telemetry** | on | 0.5239 | 0.987 | **+2.72** |
| FLY124-cruise telemetry | offcomb | 0.7137 | 1.273 | -0.77 |
| FLY124-cruise telemetry | mismatch | 0.7124 | 1.295 | -0.76 |
| FLY124-cruise `scale:0.99458` | on | 0.6820 | 1.072 | **-0.18** |

Read, in the order the challenge demands:

1. **The component is not blind to a correct carrier.** FLY124's recalibrated
   telemetry stands **3.5 dB** clear of both its nulls. The withdrawn DREGON
   factor 0.99458 destroys that lock completely (+2.72 -> -0.18, i.e. down to
   the null), which is a 0.542 % displacement resolved on a component that reads
   zero on noise. The negative control passes in both directions.
2. **DREGON's raw telemetry still does not separate from its own off-comb null**
   (-0.60 against -0.66). This is the crux and the answer is: it does not
   improve, and the reason is now measurable rather than mysterious.
3. **A scale-corrected DREGON carrier DOES separate** (+0.14 against its null
   -0.64, a 0.78 dB clearance) — the same audio, the same cells, the same fixed
   geometry, one constant applied to the carrier. The comb is there. The
   telemetry is not on it.

**Why raw telemetry reads null-like, in one line.** The ridge window is
`dc_revs = 0.10` rev/s wide. The label error is 0.3-0.8 rev/s. So the line sits
three to eight window-widths off the telemetry carrier, and a statistic that
asks "is there a line ON this carrier" correctly answers no. The eye reads a
SPECTROGRAM, which is not conditioned on the carrier at all; what it sees is
"the comb is there and the telemetry does not sit on it" — the same statement.
The 6c reading was right; the inference "the statistic is blind" was the part
that had to be tested, and it is half true: the shares are blind, the ridge is
not, and both give the same verdict here for different reasons.

## The re-score: the one-parameter scale profile `lp:5+scale:s`

15 values of `s` from -1.20 % to +0.20 %, 15 windows, per hold-out family. The
profile has ONE free parameter, so its extremum cannot be bought with
flexibility; and every family is scored on cells the fit never chose.

| group / control | ridge extremum | 95 % CI (windows) | basin depth |
|---|---|---|---|
| **DREGON, on** | **-0.683 %** | **[-0.877, -0.533]** | **1.064 dB** |
| DREGON, on, fit even k | -0.664 % | [-0.957, -0.458] | 1.121 |
| DREGON, on, fit odd k | -0.692 % | [-0.966, -0.435] | 1.078 |
| DREGON, on, fit mic 0 | -0.684 % | [-0.874, -0.538] | 1.136 |
| DREGON, on, fit half the blocks | -0.637 % | [-0.785, -0.278] | 0.953 |
| DREGON, **off-comb null** | +0.069 % | [-1.135, +0.163] | **0.133** |
| **FLY124 cruise, on** | **-0.059 %** | **[-0.098, -0.021]** | 4.642 dB |
| FLY124 cruise, off-comb null | +0.200 % (grid edge) | — | 0.492 |
| FLY124 warmup, on | +0.105 % | [-0.573, +0.105] | 0.538 |

- **The four hold-out families agree to 0.06 pp** (-0.637 to -0.692) and every
  interval contains every other point estimate.
- **The null has 8x less depth** (0.133 against 1.064 dB) and no interior
  extremum. On the 6c broadband profile at this setting the same ratio is 2.4x
  (0.048 against 0.020), so the ridge's discrimination is 3x sharper than the
  statistic the 6c decision rested on.
- **FLY124 returns zero** with a basin 4x deeper than DREGON's, so the null
  result there is a measurement, not an absence of signal.
- FLY124 warmup remains out of validity (its on-comb depth 0.54 is its null's
  0.63), as 6c already recorded.

The same profile read on the 6c components at this setting: broadband
-0.617 % [-0.883, -0.404], phase noise -1.146 % [-0.938, -0.301] (its curve is
noisier — the vertex leaves the bootstrap band), roughness -1.097 % with a
0.010 depth against the null's 0.010, i.e. nothing. So the four components put
the scale between -0.6 and -1.1 %, and the two that carry a real basin (ridge,
broadband) agree at **-0.62 to -0.68 %**.

## The re-score: the fitted trajectories

The candidate the challenge is really about — the `pi_kalman` peeled alternation
of issue 17 steps 1-6 (`arm = main`, the 6c trajectories, unchanged), scored on
the identical cells at the identical fixed geometry:

| group | candidate | on | offcomb | mismatch | gain over telemetry |
|---|---|---|---|---|---|
| **DREGON** | telemetry | -0.60 | -0.66 | -0.84 | — |
| **DREGON** | `scale:0.99458` | +0.14 | -0.64 | -0.84 | +0.74 dB |
| **DREGON** | **`fit:main`** | **+1.88** | -0.56 | -0.84 | **+2.47 dB** |
| FLY124 cruise | telemetry | +2.72 | -0.77 | -0.76 | — |
| FLY124 cruise | **`fit:main`** | **+2.98** | -0.52 | -0.76 | **+0.26 dB** |
| FLY124 warmup | `fit:main` | +0.04 | -0.15 | -0.52 | (out of validity) |

(ridge, dB, hold-out `none`. DREGON hold-out families for `fit:main`: 1.38 even
`k` / 2.35 odd `k` / 1.97 mic 0 / 1.78 half the blocks — all positive, all
clearing both nulls.)

**This is the challenge's own claim, measured.** The refined trajectory locks
onto the comb **2.47 dB** better than the labels do, on a statistic that reads
0 dB on noise, clears both nulls, and holds on every hold-out family. The eye
was right that `pi_kalman` improves the fit; 6c's statistic could only see that
improvement as 0.6644 -> 0.5927 of a saturated share.

And the negative control makes the number mean something: **the same procedure
on FLY124's recalibrated labels buys 0.26 dB**, 9.5x less. A generically
flexible fitter would have bought a comparable amount on both.

**A constant is confirmed to be the wrong model, by a bigger margin than 6c
could see.** The best constant (`scale:0.99458`) buys +0.74 dB; the free
trajectory buys +2.47 dB. Two thirds of the label error is not a scale.

## The re-score: all six ablation arms (`telemetry-6d2-e6c967`, 540/540 units)

The cluster re-ran the refit from the pinned dataset and scored every arm. Its
`main` trajectories reproduce the 6c ones exactly — the ridge reads **+1.87667**
against the local pass's **+1.8767** on the same 15 windows — so the arms below
and the 6c tables describe the same objects.

| arm | step turned off | DREGON ridge | FLY124-cruise ridge | 6c broadband (DREGON) |
|---|---|---|---|---|
| `main` | — | **+1.88** | +2.98 | 0.652 |
| `flatk` | 2, the k ladder | +1.75 | +3.04 | 0.658 |
| `gate` | 6, `pair_mode` | +1.69 | +2.96 | 0.651 |
| `nopeel` | 3, the peel | +1.64 | +3.07 | 0.653 |
| `nosmooth` | 1, the pre-smoothing | +1.43 | +3.20 | 0.661 |
| **`b0_3`** | (the wide band) | **-0.20** | +1.47 | 0.732 |
| — | raw telemetry | -0.60 | +2.72 | 0.752 |

Two things this adds to 6c:

- **The wide-band arm is disqualified on DREGON directly.** 6c could only reject
  `b0_3` through its negative control (it "found" -0.32 % on recalibrated FLY124
  labels). The ridge says the thing itself: `b0_3`'s DREGON trajectory has **no
  line on it** (-0.20 dB, its own off-comb null is -0.56), and it degrades
  FLY124's existing lock from +2.72 to +1.47. A wide capture does not merely
  inflate the estimate, it walks the carrier off the comb.
- **The six steps are ranked by what they buy in lock, not only in shift.**
  Pre-smoothing is worth **+0.45 dB** of ridge, the peel **+0.24**, `pair_mode`
  **+0.18**, the k ladder **+0.13** — every one positive, and in an order the
  6c broadband column (0.651-0.661, a 0.010 spread) could not resolve.
  On FLY124 the same four differences are within +-0.22 dB of each other and of
  the wrong sign as often as not, which is what "the labels are already right"
  should look like.

## What phase 6d changes, and what it does not

| 6c said | 6d says |
|---|---|
| DREGON telemetry has no acoustic lock (0.6644 vs null 0.6671) | **Same verdict, now on a statistic that can detect a lock** (-0.60 vs -0.66, where FLY124's telemetry reads +2.72 vs -0.77). The 6c evidence was weaker than it looked; the conclusion survives. |
| the fitted trajectories clear both nulls | **By 2.4 dB rather than by 0.07 of a share**, with the FLY124 control at one tenth of that |
| the best constant scale is -0.77 % [-0.95,-0.59] | the ridge profile says **-0.68 % [-0.88,-0.53]**, hold-outs agreeing to 0.06 pp, null 8x shallower |
| the two estimators disagree 2x (-0.77 vs -0.347) | **unchanged** — the ridge is a constant-family estimator and lands in the constant cluster |
| coverage is the binding constraint | **quantified in the right units**: the gate saw 5.7 % of DREGON's comb line energy against 18.6 % of FLY124's, an asymmetry nothing in the 6c report carried |

### The twin trap, checked per rotor on the new component

The ridge is the component most exposed to twin capture — it asks "is there a
line at this carrier", and a twin's line IS a line. So the per-rotor profile is
run against each rotor's own trap (the scale at which its carrier lands on a
sibling):

| DREGON rotor | mean rate | ridge maximum | nearest twin trap | depth |
|---|---|---|---|---|
| 0 | 79.71 | **-0.863 %** | **+0.61 %** (opposite side) | 1.28 dB |
| 1 | 70.40 | **-0.718 %** | -1.23 % (0.5 pp away) | 1.34 dB |
| 2 | 79.75 | -0.555 % | **-0.60 % (confounded)** | 0.82 dB |
| 3 | 69.83 | **-0.658 %** | **+1.24 %** (opposite side) | 1.40 dB |

Three of four rotors find a negative maximum with their trap on the other side
or half a percent away; the fourth is confounded and is also the shallowest. The
three unconfounded rotors average **-0.75 %**. Twin capture is refuted for this
measurement a second time, on the component that is most vulnerable to it.
FLY124's four rotors read -0.11 / -0.01 / +0.01 / -0.17 % with no trap nearer
than 2.2 %.

## Reproduce

```bash
python -m pytest tests/tracking/test_fitness.py -q      # 27 tests
bash scripts/telemetry_campaign.sh 12 14                # refit + the 6d re-score
python scripts/telemetry_report.py \
  --fit results/telemetry_fitness/campaign_6d \
  --profile results/telemetry_fitness/scale_profile_6d \
  --profile-component ridge --out results/telemetry_report_6d.json
```

One operational note for the next campaign: the first 6d job
(`telemetry-6d-0a0540`, 16 workers, no `--mem`) was **OOM-killed in the refit
stage** and the fitness stages then ran with no trajectories to read, turning
352 units into `.err` files while the profile's 870 units completed normally.
`gridrun` did exactly what it promises — a unit exception is a file, not a dead
pool — but a stage that depends on a previous stage's artifacts needs that
checked, not assumed. Submit the refit with `--mem` and fewer workers
(`telemetry-6d2-e6c967`, 12 workers, 96 GB).

---

# Phase 6e — the time-shift theory (issue 17)

**Status:** run and analyzed (2026-08-06) · driver `scripts/telemetry_timeshift.py`
(**deleted** in the 2026-08 R2 consolidation; its one library-grade estimator is
now `tracking.fitness.measure_tdoa`, see § "Reproduce") · `shift:` joins the
phase-6a candidate language in
`scripts/telemetry_fitness.py` · job `telemetry-6e-ed2338` on `uni-cpu`,
16 workers, **4320/4320 units, zero failures**, 2 h 19 min · local `tdoa` and
`refit-lag` passes · JSON `results/telemetry_timeshift/report_{ridge,tdoa,refit_lag}.json`,
unit trees pulled to `omnirun-outputs/telemetry-6e-ed2338/`.

**Verdict in one line: partially supported, with the sign reversed.** A common
time offset between `motors_measured` and the audio is real and measurable, but
it is **negative** — the telemetry runs EARLY, not late — and the
per-microphone differential the theory predicts is **500 times below** what this
statistic can resolve, which the campaign states as a number rather than as an
absence.

## The theory, split into the two claims it contains

1. **`motors_measured` is LATE.** A period counter reports the revolution that
   has just finished and the logging hold adds more, so the reading at log time
   `t` describes the shaft some `tau` earlier. The correction reads the
   telemetry at `t + tau`.
2. **It is late by a DIFFERENT amount at each microphone.** Sound from rotor `j`
   reaches microphone `c` after `d_cj / 343`, so the best `tau` should fall as
   distance rises, at slope `-1 / 343` s/m.

Sign convention, once and everywhere: **positive `tau` means the telemetry LAGS
the shaft**, and the candidate `r(t + tau)` removes that lag.

The two claims are three orders of magnitude apart, and that — not their truth —
decided the design.

## What each claim has to move, before any measurement

A time shift is visible only through the trajectory's own slope: `r(t + tau)`
differs from `r(t)` by `tau dr/dt`. The 5 Hz pre-smoothed `|dr/dt|` of the
fifteen protocol windows is **8.5 to 33.5 rev/s^2**. It must be read on the
smoothed trajectory: a raw gradient of the 0.269 rev/s staircase on the 32 ms
grid reads ~8 rev/s^2 of pure quantisation.

DREGON's rig (`data_processing.sources.geometry("DREGON")`, the
180-degree-corrected mic frame) puts the eight microphones 0.22-0.40 m from the
four rotors:

| quantity | value |
|---|---|
| rotor-to-mic delay `d_cj / 343` | 0.64-1.16 ms, mean **0.92** |
| spread across the 8 mics of ONE rotor | **0.482 ms** |
| spread across mics after averaging the 4 rotors | **0.156 ms** |

So the two claims arrive at the ridge as:

| claim | carrier displacement `tau dr/dt` |
|---|---|
| a 20 ms common lag | 0.17-0.67 rev/s |
| the per-(rotor, mic) differential, 0.482 ms | 0.004-0.016 rev/s |
| the per-mic differential, 0.156 ms | **0.0013-0.0052 rev/s** |

The ridge's line region is `dc_revs = 0.10` rev/s wide. The common lag is one to
seven window-widths — inside the instrument. The per-mic differential is a
fiftieth of a window width, on an eighth of the cells. **The per-microphone half
of the theory is below the ridge's resolution by a factor of about 20 before any
noise is considered.** It was measured anyway, so that the statement is a
number.

## The grid, and why it needed two axes

`shift:` is an interpolation on the frame grid, never a roll of samples, so
`tau` is free of the 32 ms grid. 36 values of `tau` from -120 to +405 ms
against **4 rate scales** (0, -0.341, -0.683, -1.024 %, bracketing phase 6d's
-0.683 %), 15 windows, on-comb and off-comb, at 6d's frozen reading settings
(`b0 = 1`, `gate_band_frac = 0.25`, `k = 2..40`, `fs_env = 250 Hz`, 8 blocks).

The second axis is not a precaution, it is the measurement. A scale error is
proportional to `r`; a lag is proportional to `dr/dt`. On a cruise window those
shapes are nearly orthogonal, but a window with a sustained trend makes a lag
LOOK like a scale, and the campaign has a fitted scale already:

| DREGON on-comb, scale arm | best `tau` |
|---|---|
| 0.000 % | **+157.6 ms** |
| -0.341 % | -41.5 |
| **-0.683 %** (6d) | **-41.5** |
| -1.024 % | -62.8 |

**Without the scale correction the shift axis reads +158 ms; with it, -42 ms,
and the sign flips.** The two corrections mask each other exactly as feared, and
a one-axis sweep of either would have produced a confident wrong answer. The
joint maximum of the surface is at `scale = -0.683 %`, `tau = -15 ms` (grid),
`-41.5 ms` by the basin parabola.

## The common shift: measured, and the wrong way round

Pooled over the nine DREGON windows and the four FLY124 cruise windows, ridge in
dB (more is better), 95 % CI over windows:

| group | scale | control | `tau*` ms | CI | depth dB | ridge @ `tau*` | ridge @ 0 |
|---|---|---|---|---|---|---|---|
| **DREGON** | **-0.683 %** | **on** | **-41.5** | **[-85.4, -30.6]** | **1.358** | **+0.529** | +0.320 |
| DREGON | -0.683 % | offcomb | +405 (edge) | — | 0.374 | -0.437 | -0.580 |
| DREGON | 0 % | on | +157.6 | [+114.7, +249.5] | 0.504 | -0.253 | -0.476 |
| **FLY124 cruise** | **0 %** | **on** | **+4.09** | **[+2.3, +6.0]** | **3.232** | **+2.561** | +2.561 |
| FLY124 cruise | 0 % | offcomb | +405 (edge) | — | 0.484 | -0.505 | -0.783 |
| FLY124 warmup | 0 % | on | -120 (edge) | — | 0.205 | -0.107 | -0.220 |

Six readings:

1. **The negative control is exact.** FLY124's recalibrated labels put their
   maximum at **+4.1 ms [+2.3, +6.0]** with a **3.23 dB** basin, and the four
   windows individually read +3, +7, +3, +2 ms. The instrument is unbiased, its
   scatter is milliseconds, and the sharp peak is what a correct carrier looks
   like.
2. **DREGON's maximum is interior and negative.** Its curve rises from +0.14 dB
   at -120 ms to +0.53 at -15 ms and falls to -0.83 by +285. All nine windows
   put `tau*` on the negative side (-8 to -120 ms).
3. **The off-comb null is featureless.** DREGON's null wanders between -0.44 and
   -0.81 dB across the whole grid with its extremum at the edge, depth 0.374
   against the on-comb 1.358 — a **3.6x** ratio. (For comparison, 6d's SCALE
   profile had 8x. The `tau` basin is real but shallower than the scale's.)
4. **Negative `tau` means the telemetry is EARLY**, so the correction reads it
   EARLIER still. This is the opposite of the proposed mechanism: a period
   counter's reporting lag is necessarily POSITIVE (one revolution at 80 rev/s
   is +12 ms, and the hold only adds). The mechanism is excluded by sign. What
   remains is a stream time-alignment offset — precisely the correction the
   other rig already carries (`tracking.protocols.FROZEN_FLY124_ALIGNMENT` is
   `(-20.84 s, 1.001)`) and that DREGON has never been given.
5. **The basin is broad.** Half-depth half-width is **98 ms** on DREGON against
   **38 ms** on FLY124, so "-42 ms" should be read as "somewhere between about
   -85 and -15 ms", not as two significant figures.
6. **It buys very little.** +0.529 dB at the joint optimum against +0.320 at
   `tau = 0` — the shift is worth **+0.21 dB** on top of the best constant
   scale, where 6d measured the free fitted trajectory at **+1.88 dB**. A lag
   plus a scale is still an incomplete model of the label error, and by the same
   margin 6d already reported.

`r(tau*, |dr/dt|)` is **-0.04** across the nine windows at the 6d scale, so the
pooled number is not an artefact of the three high-slope windows carrying it.

## The independent witness: a lag read off the fitter, with no ridge in it

`--mode refit-lag`. Phase 6c fitted a free trajectory that beats the labels by
2.47 dB, and nothing about it was chosen by a `tau` profile. Regressing its own
correction `r_fit - r_init` per window on `[dr/dt, r, 1]` returns the lag and
the scale as COEFFICIENTS:

| group | `tau` (ms) | scale (%) | mean `r2` |
|---|---|---|---|
| **DREGON** (9) | **-2.9 [-4.7, -1.1]** | **-0.811 [-1.188, -0.453]** | 0.034 |
| **FLY124** (6) | **+1.2 [-0.1, +2.5]** | +0.095 [-0.179, +0.419] | 0.013 |

The scale column reproduces the campaign's own -0.68 to -0.77 %, which is what
makes the `tau` column worth reading. And the `tau` column says the same two
things the ridge does — **DREGON negative, FLY124 zero** — while disagreeing on
magnitude by 14x. Read `r2` before either: the systematic model explains 1-7 %
of the correction (the rest is de-staircasing and the fitter's own per-frame
noise), so this BOUNDS the lag rather than measuring it. What it bounds tightly
is the theory as stated: a lag of +12 ms or more would be unmissable in this
coefficient, and it is not there.

Two estimators, both negative, differing 14x. This is the same pattern 6c and 6d
recorded for the scale (2x), and it is reported the same way: the sign is the
finding, the magnitude is not settled.

## The per-microphone differential: refuted as measurable, on the ridge

Per-microphone `tau*` against the geometry prediction (rotor-averaged), DREGON
on-comb:

| scale arm | predicted spread | measured spread | `r` | slope (predicted 1.0) |
|---|---|---|---|---|
| 0 % | 0.156 ms | 352.2 ms | -0.618 | -846 |
| -0.341 % | 0.156 ms | 162.0 ms | **+0.362** | +250 |
| -0.683 % | 0.156 ms | 78.0 ms | -0.656 | -190 |
| -1.024 % | 0.156 ms | 104.7 ms | -0.138 | -55 |

The measured spread is **500 to 2200 times** the predicted one, the regression
slope is off by two to three orders of magnitude, and the correlation **changes
sign between arms of the same measurement**. This is noise with the shape of a
result, and the sensitivity table above said so in advance: 0.156 ms of delay
moves the carrier by 0.0013-0.0052 rev/s against a 0.10 rev/s line region, on
one eighth of the cells. The per-microphone claim is not refuted by this
campaign — it is *unmeasurable by this instrument*, and no amount of averaging
fixes a twentieth of a resolution element.

## The instrument that CAN resolve it, and where it works

A propagation delay is a phase ramp across the comb, not a rate error. Harmonic
`k` of rotor `j` arrives at microphone `c` with phase `-2 pi k rate_j d_cj / 343`
relative to the reference microphone, so the mean harmonic-to-harmonic phase
INCREMENT of the cross-spectrum gives the inter-mic delay directly:

    delay_cj = -mean_k wrap(psi_{k+1} - psi_k) / (2 pi rate_j)

with no unwrapping ambiguity while `|delay| < 1 / (2 rate)` = 6.25 ms, against
delays of at most 0.5 ms. The phases come from the same demodulated envelopes
the ridge reads, at the envelope-spectrum bin the comb's own line sits in —
found once per (rotor, block) by a joint scan over a single rate offset, never
by a peak search per harmonic. `--self-test` injects 0.35 / -0.20 / 0.80 ms into
a synthetic comb and gets back 0.3499 / -0.1951 / 0.8031, **max error 4.9 us**,
which is also what pins the sign.

Rotor slot `j` of the telemetry is not guaranteed to be rotor `j` of the
geometry file, so the fit searches all 24 labellings — and a best-of-24 is a
selected maximum, so it carries its own null (permute the MICROPHONE labels,
which keeps every value and destroys the spatial pattern, then take the same
best-of-24).

Ungated (`--tdoa-gate none`, 38 harmonic pairs), median over windows:

| block | rig | windows | `r` best-of-24 | slope | null p95 | p | window sd |
|---|---|---|---|---|---|---|---|
| **FLY124 cruise, on** | michaels | 4 | **+0.829** | **+1.013** | 0.564 | **0.000** | **0.098 ms** |
| FLY124 all, on | michaels | 6 | +0.804 | +0.953 | 0.551 | 0.000 | 0.152 ms |
| FLY124 cruise, offcomb | michaels | 4 | +0.657 | +0.779 | 0.478 | 0.000 | 0.114 ms |
| **DREGON, on** | DREGON | 9 | +0.300 | +0.816 | 0.512 | **0.532** | **0.861 ms** |
| DREGON, offcomb | DREGON | 9 | +0.121 | +0.222 | 0.423 | 0.800 | 0.759 ms |

Three readings:

1. **The sub-millisecond effect is real and it is measurable — on the rig whose
   comb is resolvable.** On michaels the inter-mic delays track the rig geometry
   at slope **1.013** against a predicted 1.0, with **0.098 ms** window-to-window
   repeatability and `p = 0.000` against the relabelling null. This is the
   physics of the theory's second half, confirmed directly.
2. **On DREGON the same estimator says nothing** (`p = 0.53`), and the reason is
   the campaign's standing constraint: its twin pair is 0.42 rev/s apart, the
   ridge gate leaves **4-8** harmonic pairs against FLY124's 9, and the delay's
   error falls as `1/sqrt(pairs)`. Even ungated the window-to-window scatter
   (0.861 ms) exceeds the whole predicted spread (0.78 ms). Coverage binds here
   too.
3. **The off-comb control is NOT a null for this measurement**, and that is not
   a defect. Off-comb frequencies carry the same rotors' broadband noise from the
   same directions, so a direction estimator finds them (FLY124 offcomb
   `r = +0.657`). The on-comb reading is still the stronger one and the only one
   with unit slope; the half-integer comb remains a null for the RIDGE, which is
   where 6a and 6d use it.

Finally, the propagation term's own contribution to the global `tau`: the mean
rotor-to-mic delay is 0.92 ms, and it enters the best `tau` as **-0.92 ms**
(the audio is delayed, so the fitting carrier is the shaft history delayed). It
is **2 %** of the measured -41.5 ms. The common offset is not propagation.

## What 6e changes

| before 6e | after 6e |
|---|---|
| the label error is a scale of -0.683 %, and two thirds of it is something else | a **time offset of -42 ms [-85, -31]** is part of that something else, worth +0.21 dB of the missing 1.35 dB |
| (untested) the telemetry lags the shaft | **refuted in sign** by two independent estimators; the telemetry runs EARLY, so a counter reporting lag is not the mechanism. A stream alignment offset is, and DREGON has never been given the one FLY124 carries |
| (untested) the best `tau` varies per microphone with `d/c` | **unmeasurable on the ridge** by a factor of ~500, measured scatter 78-352 ms against 0.156 ms predicted, correlation sign-unstable across arms |
| (untested) the per-(rotor, mic) delay itself | **confirmed on michaels** at slope 1.013, `p < 0.001`, 0.098 ms repeatability; **not resolvable on DREGON** (`p = 0.53`), coverage again |

The open question 6e leaves is the same shape as 6c's: two estimators of one
quantity, agreeing on sign and disagreeing 14x on magnitude (-41.5 ms from the
ridge, -2.9 ms from the fitter's own correction). A DREGON alignment refit — the
`(offset, dilation)` pair that FLY124 got, fitted rather than assumed — is the
experiment that would settle it, and it is now worth doing because the sign is
established.

## Reproduce

The driver `scripts/telemetry_timeshift.py` — its four modes (`ridge`, `tdoa`,
`refit-lag`, `report`), its `--self-test` and the `omnirun` recipe that ran the
4320-unit sweep — was **deleted in the 2026-08 R2 consolidation**. Phase 6e is
closed, so the measured findings in this section are the record.

Everything the driver called is still library code. The ridge is
`tracking.fitness.score_cells`, the fitter's own correction is
`tracking.telemetry_refit`, and the one estimator that lived only in the driver
— the cross-channel comb-phase delay — was promoted to
**`tracking.fitness.measure_tdoa`** (with its joint line scan
`tracking.fitness.line_bins`). Its sign and its ~5 us accuracy, which the
`--self-test` used to assert, are now asserted by
`tests/tracking/test_fitness.py::test_measure_tdoa_recovers_an_injected_delay_with_the_right_sign`.

To read the inter-mic delay of one frozen prep window:

```python
from tracking.fitness import FitnessConfig, measure_tdoa
from tracking.protocols import load_prep_window

w = load_prep_window("free-flight_nosource_room1__w03")
out = measure_tdoa(w["audio"], w["ft"], w["r"], w["r"], cfg=FitnessConfig(k_min=2, k_max=40))
```

`out["delay_ms"]` is `(rotor, mic)` relative to `ref_ch`; pass `half=True` for
the off-comb null and `gate=False` for the ungated arm.
