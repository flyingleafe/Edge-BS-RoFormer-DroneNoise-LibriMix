# Coupled VK envelopes as the tracker's front end — measured, and rejected

**Status:** done (2026-08-06, one session) · GitHub issue #15 · driver
`scripts/vk_frontend_probe.py` (**deleted** in the 2026-08 R2 consolidation —
the campaign is closed and this document is the record) · raw JSON
`results/vk_frontend_probe/{b0_1.0,b0_0.35}/` · job `bash-abb74e` (uni-cpu,
succeeded) plus two local runs.

## Question

Issue #15 asks whether `pi_kalman`'s rate observations should come from the
coupled envelope solve `A_k(t)` (`tracking.vk_tracking.vk_envelopes`) instead
of the independently demodulated `z_k(t)`
(`tracking.phase_increment_tracker.demod_bank`). The claim is that coupling
assigns contested energy once, so `A_k` stays clean where `z_k` returns a
blend — and that it removes the demodulation cost at the same time.

**Answer: no.** Coupling is well conditioned only on the harmonics where it
duplicates the demodulation within noise, and it is degenerate on exactly the
contested harmonics it was proposed for. It also costs 15-110x more, not less.

## Method

Both front ends run at the SAME carriers (the window's raw telemetry, the
`tracking_ref` init), on the same envelope grid (`fs_env = 100 Hz`), with
matched capture in rev/s, and both are reduced to the same observation

    dr_k(t) = arg(u_k[n] conj(u_k[n-1])) * fs_env / (2 pi k)     [rev/s]

the shaft-rate ERROR of the carrier. A rate error displaces harmonic `k` by
`k dr`, so this observation is k-INDEPENDENT: every harmonic of one rotor must
report the same value. That is the accuracy criterion (`cons_err`, distance
from the demodulated front end's `k^2`-weighted fused answer for that rotor).
The per-track centre is the coherent pulse-pair estimate, not the median of
wrapped increments, which is biased toward zero once the phasor is noise
dominated.

Band correspondence, the thing the whole experiment rests on:

| | demod (`demod_bank`) | coupled (`vk_envelopes`) |
|---|---|---|
| shape | brickwall, half width `B_k` | VK-2 passband, -3 dB **full** width `bw_k` |
| k-scaled | `B_k = min(k b0, 0.45 fs_env)` | `bw_k = min(k bw_rps, 0.9 fs_env)` |
| match | | `bw_rps = 2 b0` — same capture radius, same clamp knee |

Arms: **wide** = the proposal, k-scaled at `bw_rps = 2 b0`; **peel** = the
flagship's own scalar 1 Hz solve (`pipelines.PEEL_BW_HZ`), as the
smoothness-bias extreme. Two capture settings: `b0 = 1 rev/s` (the setting
the displacement campaign found best) and `b0 = 0.35 rev/s`
(`pi_kalman_refine`'s default). Windows: DREGON `nosource` w00 (the frozen
`tracking_ref` clip) and w01 (the displacement campaign's window), FLY124 w02
— 16 s, 8 mics, `k_max = 80`, `f_max = 7500 Hz`. Harmonics are labelled by the
distance of the nearest real line of another rotor (`tracking.nearest_interloper_hz`,
new): **clean** = at least 2 bands away, **contested** = inside one band.

## Three results the setup produced before any accuracy number

**1. There is no "wide" coupled solve to run without disabling a guard.** The
coupling predicate puts 316-320 of the 320 tracks into ONE group, at every
`couple_hz` from 6 to 50 Hz (measured on w00). That group contains a
near-coincident pair, so `VKConfig.sep_bw_factor` clamps the whole group's band
to the `cfg.bw_hz` floor: a k-scaled request of `2k` Hz comes back as 1.00 Hz
at k = 1, 7, 40 and 70 alike. "Wide" here means raising that floor to
`0.9 fs_env`, i.e. switching the guard off.

**2. With the guard off the normal equations stop being positive definite.**
`cholesky_banded` raised on the big group in **every window of every run**, and
`vk_envelopes` fell back to sparse LU. The cost, per 16 s / 8-mic window at
K = 80:

| | wide (guard off) | peel (1 Hz) | the demodulation it would replace |
|---|---|---|---|
| DREGON w00 | **1106 / 1255 s** (node) | 56 s | `demod_bank` K=80, 4 rotors: **9.8 s** |
| DREGON w01 | 151-198 s | 29-34 s | (2.45 s per rotor, 1 FFT worker) |
| FLY124 w02 | 406-606 s | 41 s | |

Peak RSS 13-16 GB for the wide solve. So the coupled front end costs **15x to
110x** the whole demodulation bank it was supposed to remove. Issue #16's
phase-resampling route remains the cost lever; this is not one.

**3. At the campaign's capture there is almost no interferer-free harmonic.**
Over the three windows, `b0 = 1 rev/s` leaves **8 clean tracks out of 960**
(all FLY124, k <= 5); `b0 = 0.35` leaves 95, of which 93 pass admission.
**No** clean track has k >= 20 in any window at either setting. The "they should agree where nothing is in band"
test only has a sample at all at the narrower capture, and only at low k.

## The comparison

Pooled over the three windows, medians over tracks, rev/s. `med|A-z|` and
`corr` compare the arm against the demodulated observation; `|A|/|z|` is the amplitude ratio (nominal **2**, since `x ~ 2z`).

`b0 = 0.35 rev/s`, 898 admitted tracks:

| set | n | est | cons_err | mad_dr | hp_frac | med\|A-z\| | corr | \|A\|/\|z\| |
|---|---|---|---|---|---|---|---|---|
| clean | 93 | demod | 0.0202 | 0.174 | 0.504 | — | — | 1 |
| | | **wide** | 0.0211 | 0.186 | 0.513 | 0.132 | 0.124 | **2.03** |
| | | peel | 0.0077 | 0.047 | 0.238 | 0.120 | 0.037 | 1.07 |
| contested | 677 | demod | 0.0068 | 0.198 | 0.768 | — | — | 1 |
| | | **wide** | 0.0079 | 0.167 | 0.810 | 0.212 | 0.004 | **179** |
| | | peel | 0.0016 | 0.006 | 0.264 | 0.135 | 0.002 | 0.38 |
| contested, k >= 45 | 382 | demod | 0.0057 | 0.208 | 0.795 | — | — | 1 |
| | | **wide** | 0.0063 | 0.200 | 0.835 | 0.231 | 0.000 | **210** |

`b0 = 1 rev/s`, 365 tracks: clean (8, FLY124) wide `|A|/|z|` **1.93**,
corr **0.179**, cons_err 0.092 -> 0.022; contested (309) `|A|/|z|` **223**,
corr 0.007; contested k >= 45 (26) `|A|/|z|` 282, corr 0.003, cons_err
0.229 -> 0.077. Per window the coupled arm is **worse** on the accuracy
criterion wherever the reference is not itself near-degenerate: w00 at
`b0 = 1` cons_err 0.0735 -> 0.0998 with `|A|/|z|` = **850**, w00 at
`b0 = 0.35` 0.0091 -> 0.0103 with `|A|/|z|` = **1156**.

Read it in this order:

- **The issue's qualitative prediction holds.** On clean harmonics the two
  front ends agree within their own noise (`med|A-z|` 0.132 vs each
  estimator's own MAD 0.174 / 0.186). On contested ones they diverge beyond it
  (0.212 vs 0.167-0.198). Divergence is exactly where an interferer is.
- **The divergence is a cancelling mode, not explaining-away.** Where they
  diverge, the coupled envelope's amplitude is **179x** the demodulated one
  (up to 1156x on w00) and its rate observation is **uncorrelated** with the
  data (corr 0.004, and 0.000 at k >= 45). This is the twin-collapse failure
  `VKConfig.sep_bw_factor`'s own docstring predicts, reproduced by switching
  that clamp off. Issue risk 5, confirmed.
- **It buys no accuracy.** Contested cons_err 0.0068 -> 0.0079 at `b0 = 0.35`,
  and 0.0735 -> 0.0998 on w00 at `b0 = 1`. The apparent win at
  `b0 = 1, k >= 45` (0.229 -> 0.077) is shrinkage: the degenerate solve reports
  `pp_dr ~ 0` for every track, which scores well against a fused reference that
  is itself near 0.
- **The cost on clean harmonics is small but real**: cons_err +4 %
  (0.0202 -> 0.0211), observation noise +7 % (0.174 -> 0.186).
- **Smoothness bias (risk 6) is a property of the narrow band, not the wide
  one.** The wide arm's high-frequency share is unchanged (0.513 vs 0.504)
  because `rho^2` collapses from 4.25e5 at 1 Hz to **0.027** at 90 Hz — the
  prior is inert. The peel band shows the bias plainly: hp_frac 0.238 and MAD
  0.047. Same knob, opposite ends: **the prior that regularises the
  near-singular coupling is the prior that biases the observation.**
- **Risk 1, confirmed numerically.** The peel arm's fused answer is
  0.0001-0.0011 rev/s for every rotor of every window: a 1 Hz band has capture
  `1/(2k)` rev/s = 0.007 rev/s at k = 70, so it cannot represent the error it
  is asked to measure. A peel-band envelope is not an observation.

### The 27 Hz case (w01, r0 and r2 at k = 70)

The issue's headline example, `b0 = 1` (band +-45 Hz; the nearest interferer is
in fact 8.2 Hz away, not 27 — a 4-rotor comb is denser than the r0/r2 pair
alone):

| track | est | pp_dr | mad_dr | cons_err | \|A\|/\|z\| | corr |
|---|---|---|---|---|---|---|
| r0 k=70 | demod | -0.248 | 0.475 | 0.228 | 1 | — |
| | wide | -0.020 | 0.357 | 0.001 | **103** | 0.028 |
| r2 k=70 | demod | +0.181 | 0.545 | 0.239 | 1 | — |
| | wide | +0.055 | 0.357 | 0.114 | **146** | 0.014 |

The coupled observation is smoother and closer to the fused answer, and it is
not a measurement: its amplitude is two orders of magnitude off and it carries
no correlation with the demodulated signal it is derived from.

### Near-coincident pairs

Closest pair on w01, r0 k=7 vs r1 k=8, 0.5 Hz apart. MAD of the rate
observation, individual a / individual b / **pair sum** (formed exactly in the
first track's frame, `x_m + x_n exp(i(k_n phi_n - k_m phi_m))`):

| est | mad_a | mad_b | mad_sum |
|---|---|---|---|
| demod | 0.540 | 0.495 | 1.260 |
| wide | 0.307 | 0.357 | 1.252 |
| peel | 0.043 | 0.031 | 1.105 |

The sum is not the stable quantity at the OBSERVATION level in any arm: it is a
two-tone mixture whose phase advances with the beat and the stronger line, so
merging the pair does not recover a usable rate. Same on every pair measured.

A conditioning gate, on the other hand, is trivially available and clean:
`|A_k| / |z_k| > 5` fires on **0 %** of clean tracks (their whole range is
1.39-3.33) and on **84-90 %** of contested ones. It is a perfect specificity
test — there is simply nothing left to keep after it fires.

## Verdict

**Reject** taking the tracker's rate observations from `vk_envelopes`.

1. Where the coupled solve is well conditioned (`|A|/|z| ~ 2`, clean
   harmonics) it reproduces the demodulated observation within noise and is
   slightly worse on both accuracy (+4 %) and noise (+7 %), at 15-110x the
   cost.
2. Where it was proposed to help (contested, high k) it is degenerate:
   amplitude 179-1156x, correlation with the data 0.000-0.007, and no accuracy
   gain. The mechanism is the documented twin collapse, and the guard that
   prevents it (`sep_bw_factor`) also makes "wide" unreachable, because the
   whole comb is one coupling group.
3. Near-coincident pairs must **stay dropped**. A conditioning gate identifies
   them perfectly, and neither the individual envelopes nor their sum yields a
   rate observation. The existing `pair_mode="joint"` two-tone spectral fit —
   which does not use increments — remains the only treatment that recovers
   anything from a twin pair.

Partial adoption ("coupled at contested high k, demod elsewhere") is the one
combination the data rules out most strongly: contested high k is precisely
where the coupled envelope is degenerate.

If it were adopted anyway, issue risks 3-4 would bite and are not cheap: the
`c_noise = 1 - sinc(2 B dt)` and `c_diff` corrections are brickwall
derivations that do not describe the VK passband, and the coupled solve
deliberately correlates errors across tracks while the WP18 fusion assumes a
diagonal `R`. Doing it properly needs the solve's own covariance — a `g x g`
block per envelope sample for a `g = 316` group. That is not tractable, which
is a third independent reason for the same answer.

## Honest limitation

Neither front end recovers the telemetry scale error the comb-displacement
campaign measured (DREGON expected -0.41..-0.47 rev/s; both arms' fused answers
sit within ~0.1 rev/s of zero, `bias_err` ~ 0.40-0.46). On FLY124, where the
expected error is -0.05 rev/s, `bias_err` is ~0.05 — everything reads about
zero. In-band noise attenuates a coherent phase-increment centre toward zero,
and at these bandwidths both estimators are noise dominated. So this probe
compares the two front ends fairly but certifies neither as accurate; the
DREGON displacement is not visible in raw increments at the tracker's own
bands, only in the campaign's short-time envelope ridge.

## Reproduce

The driver `scripts/vk_frontend_probe.py` was **deleted in the 2026-08 R2
consolidation**, together with the `--rescore` pass and the `omnirun` recipe
that ran it. The probe answered its question and the answer is no, so the
measurements above are the record; there is nothing left to re-run.

The driver held no library-grade code of its own. Both front ends it compared
are library calls (`tracking.vk_tracking.vk_envelopes` and
`tracking.phase_increment_tracker.demod_bank`), and the coherent pulse-pair
centre it read them with is `tracking.comb_displacement.pulse_pair_bank` —
promoted there in issue 17 phase 6a exactly so that the probe and
`tracking.fitness` could not drift apart. The raw per-track JSON stays at
`results/vk_frontend_probe/`.
