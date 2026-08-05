# The DREGON $k = 2$ wiggle: real shaft-rate deviation, or interference?

Data: `traces/*.npz` (written by `retrace.py`), analysis and figure in `make_figs2.py`
(`wiggle_stats`, `fig6`), numbers in `wiggle_stats.json`, figure `figs/F6_wiggle.png`.
Five windows: the three DREGON cruise `w01` windows (one per recording) and FLY124 `w03`,
`w04`. For each window the rotor with the strongest harmonic set is used (rotor 0 in all
five). Traces are peak-pick offsets on a common 2 s / 0.25 s base, **ungated** — the
twin-collision gate rejects 78-89% of the DREGON (harmonic, frame) pairs and would delete the
$k = 2$ trace entirely, which is exactly the trace in question.

---

## VERDICT

**The wiggle is a REAL deviation of the shaft rate from telemetry.** It is not twin
beating, not a fixed-frequency artifact, and not a simple telemetry lag. Riding on top of
it is a **constant** offset of about **-0.5 rev/s (-0.54% of rate)** on DREGON, which the
same evidence shows is also real and is most likely a telemetry **scale** error.

Four tests, all pointing the same way.

### 1. The offset is FLAT in $k$ — so it is a RATE, not a frequency

$\delta_k$ (time mean, rev/s) for the strongest rotor:

| window | $k{=}2$ | $k{=}4$ | $k{=}6$ | $k{=}8$ | $1/k$ would predict at $k{=}4$ |
|---|---|---|---|---|---|
| DREGON nosource w01 | -0.507 | -0.506 | -0.529 | -0.341 | -0.254 |
| DREGON speech-low w01 | -0.575 | -0.546 | -0.278 | -0.189 | -0.288 |
| DREGON whitenoise-low w01 | -0.437 | -0.478 | -0.328 | -0.282 | -0.219 |
| FLY124 w03 | -0.106 | -0.099 | -0.166 | -0.108 | -0.053 |
| FLY124 w04 | -0.060 | -0.056 | -0.050 | -0.023 | -0.030 |

A shared *rate* deviation appears at the same rev/s at every $k$. A shared fixed
*frequency* $f$ appears at $f/k$ rev/s and halves from $k{=}2$ to $k{=}4$. Every window is
flat; none halves. Panel (d) of F6 shows the same thing over all 40 harmonics with marker
size set by prominence.

### 2. The time-varying part correlates ACROSS harmonics, with slope 1, not 1/2

Fixing the pair $(k{=}2, k{=}4)$ a priori for every window (no post-hoc selection), the
regression slope of $\delta_4(t)$ on $\delta_2(t)$ is bracketed by the two one-sided
regressions:

| window | r | slope bracket | shared rate predicts | fixed frequency predicts |
|---|---|---|---|---|
| DREGON nosource w01 | **+0.786** | [1.38, 2.23] | 1.0 | 0.5 |
| DREGON speech-low w01 | +0.265 | [0.47, 6.69] | 1.0 | 0.5 |
| DREGON whitenoise-low w01 | **+0.668** | [1.22, 2.75] | 1.0 | 0.5 |
| FLY124 w03 | **+0.648** | [0.77, 1.84] | 1.0 | 0.5 |
| FLY124 w04 | **+0.826** | [0.97, 1.43] | 1.0 | 0.5 |

Four of five windows exclude the fixed-frequency prediction of 0.5 outright and bracket or
exceed 1.0. (`speech-low` is uninformative — its $k{=}4$ ridge is only 5.2 dB, so the
bracket is useless, not contradictory.) The brackets sit slightly **above** 1.0 on DREGON
because the weaker harmonic carries more peak-pick noise, which inflates its variance.

### 3. The correlation appears exactly where a line exists — and nowhere else

Panel (c) of F6 plots $r(\delta_2, \delta_k)$ against $k$ with each harmonic's prominence
shaded behind it. The correlation tracks the prominence:

- DREGON whitenoise-low w01: $r$ = +0.67 ($k$=4, 9.1 dB), +0.63 ($k$=14, 7.9 dB),
  +0.67 ($k$=13, 5.5 dB), +0.23 ($k$=11, 6.0 dB), +0.08 ($k$=37, 5.2 dB).
- FLY124 w03: +0.65 ($k$=4, 18.9 dB), +0.56 ($k$=6, 14.8 dB), +0.29 ($k$=10, 14.7 dB),
  +0.02 ($k$=8, 11.9 dB).
- Everywhere the prominence falls to the noise floor, $r$ falls into the +-0.3 band that the
  off-comb null also occupies.

That is the signature of one shared deviation seen through per-harmonic estimator noise —
attenuated where the line is weak, strong where it is not. Independent peak-pick noise
could not produce r = +0.79 on 57 samples.

Mean pairwise cross-$k$ correlation over the five plotted harmonics (which include noise
ones, so this is a diluted statistic): DREGON nosource +0.14 (off-comb null +0.09),
speech-low +0.26 (-0.15), whitenoise-low **+0.44** (-0.01), FLY124 w03 +0.23 (+0.14),
w04 **+0.41** (-0.07). Measurement above null in all five.

### 4. The alternatives are ruled out

**Twin beating — RULED OUT.** DREGON's rotor 2 is the near-twin of rotor 0. On the
rescaled axis its line sits at $r_2(t) - r_0(t)$, which is *k*-independent. Over
`nosource w01` that track has mean -0.73 rev/s but swings from **-3.86 to +2.18 rev/s**
(std 1.45) — it crosses rotor 0's rate and leaves the band repeatedly. The measured
$k{=}2$ ridge sits at -0.51 +- 0.21 rev/s and does **not** follow it (panel a1 of F6 draws
both). Same for the other two DREGON windows (twin track std 1.86 and 2.13 rev/s). On
FLY124 the nearest other rotor is 9.7 rev/s away and never enters the band at all, yet
FLY124 shows the same flat-in-$k$, cross-$k$-correlated wiggle — so the effect does not
need a twin.

Beat-frequency check: the dominant frequency of the prominence-weighted mean trace is
0.07 Hz (nosource), 0.35 Hz (speech-low), 0.07 Hz (whitenoise-low), 0.07 Hz (FLY124 w03
and w04) — i.e. the wiggle is a slow drift over the 16 s window, not a beat. The
instantaneous twin split at cruise is 0.3-3.9 rev/s, whose $k{=}2$ beat would be 0.6-7.8 Hz,
one to two orders of magnitude faster.

**A rotor-index permutation — RULED OUT.** DREGON's twin pair is (rotor 0, rotor 2), whose
mean rates differ by -0.42 rev/s in `nosource w01` — suspiciously close to the measured
-0.51. If the telemetry channel called "rotor 0" actually logged the *other* physical
rotor, demodulating at $2\,g_0$ would land on $2\,r_2$ and give exactly that. Two facts
kill it:

- A permutation is **antisymmetric**: rotor 0 would read $r_2 - r_0$ and rotor 2 would read
  $r_0 - r_2$, i.e. opposite signs. Measured, over the 34 bar-clearing low-$k$ units of all
  9 DREGON cruise windows, **all four rotors are negative**: -0.545, -0.256, -0.446, -0.313
  rev/s (rotors 0-3). The same sign on both members of a twin pair cannot be a swap.
- A permutation would make the ridge follow $r_2(t) - r_0(t)$ exactly. It does not (above).

**Telemetry lag — RULED OUT as the explanation, though a small lag may contribute.** If
telemetry were merely late by $\tau$, the measured offset would be $\tau\,dg/dt$. Panel (e)
of F6 cross-correlates $\bar\delta(t)$ with $dg/dt$ over +-2 s:

| window | best lag | r at best lag | $\tau$ needed to explain the amplitude |
|---|---|---|---|
| DREGON nosource w01 | -0.75 s | +0.39 | 0.070 s |
| DREGON speech-low w01 | -1.00 s | +0.39 | 0.095 s |
| DREGON whitenoise-low w01 | +0.50 s | -0.21 | 0.101 s |
| FLY124 w03 | -1.75 s | -0.37 | 0.014 s |
| FLY124 w04 | +0.75 s | -0.27 | 0.014 s |

The peak correlation is weak (|r| <= 0.39, i.e. <= 15% of the variance), it is not at a
consistent lag, and it is not at a physically plausible lag (0.5-1.75 s). A lag also cannot
produce the **constant** -0.5 rev/s offset at all, since $dg/dt$ has zero mean over the
window. Conclusion: a lag explains neither the offset nor most of the wiggle.

**A telemetry SCALE error — SUPPORTED for the constant part, not the wiggle.** The
constant offset is -0.54% of rate on DREGON (per rotor: -0.55%, -0.34%, -0.54%, -0.45%;
34 bar-clearing low-$k$ units, rates 56-86 rev/s) and -0.063% on FLY124. That is the same
form and the same order as the +0.70% multiplicative correction measured for Michael's rig
in WP14 — and FLY124's near-zero residual here is an **independent confirmation** that the
2026-07-31 recalibration worked. Within a window, though, the instantaneous slope
$d\delta_2/dg$ is -0.097 (nosource) and -0.057 (whitenoise-low), 10-16x steeper than the
-0.0054 a pure scale error would give, so the *fluctuation* is not a scale error. It is a
genuine rate deviation that DREGON's telemetry does not resolve.

---

## What this means

1. **DREGON's rotor telemetry is biased by about -0.5% at cruise.** Nothing in this
   project has corrected for it. Every DREGON-derived RPS label, every "blind track vs
   telemetry" error figure, and every generator trained on DREGON labels carries that
   bias. The magnitude (-0.42 rev/s at cruise) is **twice** the 0.2 rev/s "honest 0 dB
   label-jitter floor" the flagship work assumes.
2. **On top of the bias sits a real, slow, ~0.2 rev/s rms deviation** that telemetry does
   not capture. This is a floor on how well any tracker can agree with telemetry, and it
   is measurable only at $k \le 6$, where lines exist.
3. **FLY124 is clean** (-0.06 rev/s, -0.063%), so the effect is a DREGON property, not a
   property of the measurement.
4. Suggested next step: repeat the WP14-style calibration fit (the one that produced
   Michael's `time_offset` / `time_dilation` / `rps_scale`) on DREGON, using only $k \le 6$
   where the lines clear the 6 dB bar. The lever is a single scale constant per recording.
   The acoustic rate is BELOW telemetry, so telemetry over-reports and the correction is
   ~x0.9946 (against Michael's x1.007, which went the other way).
