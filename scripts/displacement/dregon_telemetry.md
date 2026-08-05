# Why the DREGON comb sits 0.54 % below telemetry

Measured 2026-08-05. Scripts in this directory: `bench_rate.py` (static bench + audio
clock), `ladder.py` (regime ladder + regressions), `combscan.py` (window-free
order-space scan), `recentre.py` (two-pass re-centred k sweep), `shortscan.py`
(short-segment high-k scan), `make_f7.py` (figure F7). Numbers:
`bench_rate.json`, `audio_clock.json`, `ladder.json`, `combscan.json`,
`recentre.json`, `shortscan.json`. Figures: `figs/F7_regime_ladder.png`,
`figs/F10_combscan.png`, `figs/F11_shortscan.png`.

---

## VERDICT

**H1 wins: DREGON's rotor telemetry over-reports by a clean multiplicative
0.542 % (95 % CI 0.450-0.618 %).** H2 (commanded setpoint) is refuted twice
over. H3 (aeroacoustic displacement) is unsupported and physically implausible,
but the measurement that was supposed to kill it — the static bench — cannot be
done, because DREGON's bench recordings carry no telemetry at all.

| hypothesis | verdict | why |
|---|---|---|
| **H1** telemetry scale error | **SUPPORTED** | slope 0.99458, intercept consistent with 0, R² 0.999; replicated by a window-free estimator; audio time base verified correct |
| **H2** commanded setpoint | **REFUTED** | the reference channel is a period-counting tachometer, and the command channel agrees with it to 0.04 % anyway |
| **H3** real acoustic displacement | **unsupported** | no Doppler is available (array and rotors are rigidly co-mounted); the offset's form is exactly multiplicative; but the hover-vs-cruise contrast is untestable — see "What is still open" |

---

## 1. What the rotor-speed channel actually is

**The displacement measurement's reference is `motors_measured`.** The frozen
`beatvk-valid-raw` spec pins it for all three DREGON recordings
(`src/data_processing/derivations.py:1305-1307`), and `sources/dregon.py:448-460`
copies the `.mat` `measured` field verbatim with no unit conversion.

`motors_measured` **is a measured tachometer, not a setpoint.** Read directly out
of `data/DREGON/DREGON_free-flight_nosource_room1/..._motors.mat`:

| property | value |
|---|---|
| field name | `motor.measured`, `(M, 4)` float64, one column per rotor |
| units | rev/s |
| lattice | **v = 1 / (n x 42.0 us), n integer** — fitted 1/delta = 23809.53 Hz, residual rms 0.0006 of a step, max 0.0027, n = 258..1188 |
| update rate | value changes at **49.7 Hz** (median 20.1 ms), logged at 1002 Hz with zero-order hold |
| quantisation step | v^2 x delta = **0.269 rev/s at 80 rev/s** |
| availability | only the five `free-flight_*_room1` recordings |

A reciprocal-integer lattice is the signature of a **period counter**: the ESC
times the interval between commutation events in 42 us ticks and reports
`1/(n x tick)`. That measures the shaft. It cannot be a commanded value — a
setpoint has no reason to live on a reciprocal lattice.

`motors_command` is the opposite: a continuous float (94 431 distinct values in
one recording, spacing 1e-4), present in every recording.

**Nothing in the DREGON raw tree documents either channel.** There is no README;
`micPos.txt` and `plot_structure.m` cover geometry only. The "measured
tachometer" reading above is inferred from the lattice, not quoted from
documentation. It is a strong inference, but it is an inference.

### The command channel does not explain anything either

Per-rotor `measured / command` over the cruise part of all five room1 recordings:

| recording | rotor means |
|---|---|
| free-flight_nosource_room1 | 1.00029 1.00015 1.00090 1.00010 |
| free-flight_speech-low_room1 | 1.00065 1.00049 1.00075 1.00012 |
| free-flight_whitenoise-low_room1 | 1.00049 1.00037 1.00111 1.00045 |
| free-flight_speech-high_room1 | 0.99996 1.00013 0.99998 0.99999 |
| free-flight_whitenoise-high_room1 | 0.99981 0.99988 0.99983 0.99959 |

The two channels agree to **0.04 % on average, 0.11 % worst case** — an order of
magnitude smaller than the 0.54 % acoustic gap and of no consistent sign. H2
requires shafts running slower than commanded under load; DREGON's own telemetry
says that gap is essentially zero. Measuring the acoustic offset against
`command` instead of `measured` gives **-0.394 %** (CI 0.315-0.456, n = 65)
against `measured`'s **-0.542 %** — a 0.15 pp difference on different unit sets,
i.e. within the scatter of the estimator, not a channel difference.

**Project-wide note:** `frames.PUBLISHED_RPS_KEYS` prefers `motors_command`
while the beat-VK protocol pins `motors_measured`. Since the channels agree to
0.04 % (~0.03 rev/s), this inconsistency is real but negligible next to the bias
documented here.

---

## 2. The audio time base is correct — the degeneracy is broken

A 0.54 % acoustic-vs-telemetry ratio is degenerate between "telemetry
over-reports" and "the audio clock runs slow" — the same degeneracy WP14 flagged
on Michael's rig. It is broken here directly.

`silent-flight_whitenoise-low_room1` and the two `free-flight_whitenoise-*`
recordings re-record the shipped `emitted_signals/2min_white_noise.wav` through
the onboard 8-mic array. GCC-PHAT (200-3000 Hz) of 8 s blocks against the emitted
file gives the lag drift, i.e. (onboard recorder clock)/(ground playback clock) - 1:

| recording | blocks | span | clock error | 95 % CI | resid rms |
|---|---|---|---|---|---|
| silent-flight_whitenoise-low_room1 | 16 | 60 s | **+0.0023 %** | -0.011 .. +0.024 % | 5.7 ms |
| free-flight_whitenoise-low_room1 | 11 | 52 s | **-0.0081 %** | -0.025 .. -0.003 % | 2.0 ms |
| free-flight_whitenoise-high_room1 | 12 | 52 s | **-0.0074 %** | -0.015 .. -0.001 % | 2.1 ms |

Two independent devices agree to better than **0.03 %**, i.e. **20x smaller than
the effect**. A 0.54 % audio-clock error would have shown as ~420 ms of drift over
78 s; the observed drift is under 5 ms. The audio time base is nominal 44 100 Hz.

So the acoustic shaft rate really is 0.54 % below what the tachometer reports.

---

## 3. The static bench — the requested decisive test cannot be done

`DREGON_individual_motors_recordings/` holds `Motor{1-4}_{50,60,70,80,90}.wav`
and `allMotors_70.wav`, 8-channel 44.1 kHz, 25-45 s each. **None of them has
telemetry.** The only reference is the number in the file name.

Acoustic shaft rate by a comb fit (mean log-magnitude over harmonics 1..40 below
6 kHz, scanned over 0.6-1.4 x nominal, per 5 s block):

| motor | 50 | 60 | 70 | 80 | 90 | mean acoustic/nominal |
|---|---|---|---|---|---|---|
| Motor1 | 49.03 | 58.78 | 68.49 | 78.22 | 87.98 | **0.9788** |
| Motor2 | 48.46 | (unstable) | 67.71 | 77.31 | 86.92 | **0.9671** |
| Motor3 | 49.15 | 59.05 | 68.71 | 78.66 | 88.45 | **0.9829** |
| Motor4 | 49.79 | 59.81 | 69.50 | 79.38 | 89.40 | **0.9942** |

Block-to-block scatter is 0.001-0.04 rev/s, so the acoustic estimate is precise
to ~0.05 %. But **the four motors differ from each other by up to 2.8 percentage
points at the same nominal** (Motor2 -3.3 %, Motor4 -0.6 %). A shared telemetry
constant cannot produce a per-motor spread. The file-name number is therefore an
**open-loop throttle command** that each motor undershoots by its own amount —
which is what the bench is: a motor on a stand with no flight controller closing
the loop.

**Consequence: the bench measures acoustic-vs-command, not acoustic-vs-telemetry,
so it cannot test H1/H3.** It does establish two useful things: the acoustic
estimator is unbiased and precise on high-SNR isolated rotors, and each motor's
acoustic/nominal ratio is nearly flat in rate (Motor1 -1.94 % at 50 to -2.25 % at
90), so nothing rate-dependent and exotic is happening in the acoustics.

`allMotors_70.wav` was dropped: four simultaneous rotors make the single-comb fit
ambiguous, exactly as expected.

---

## 4. The regime ladder

Same estimator as `nullcontrol.py` (same demod bank, same per-harmonic band, same
corrected `carrier_collision_mask`, same >=6 dB prominence bar), extended to every
DREGON recording that has telemetry, built from the local raw tree. Units are
(window, rotor, k) with k in 2..13 that clear the bar. Fits over those units, CIs
by block bootstrap over windows.

| regime | telemetry | n units | windows | rate range | **scale error** | 95 % CI | free-fit slope | free-fit intercept | R² |
|---|---|---|---|---|---|---|---|---|---|
| free flight, cruise | measured | 54 | 13 | 56.5-86.6 | **-0.542 %** | -0.618 .. -0.450 | 0.99403 [0.9875, 1.0001] | **+0.044** [-0.374, +0.563] | 0.9990 |
| free flight, cruise | command | 65 | 13 | 66.3-86.8 | -0.394 % | -0.456 .. -0.315 | 0.99206 | +0.319 | 0.9982 |
| all DREGON cruise | mixed | 59 | 17 | 56.5-86.6 | -0.574 % | -0.675 .. -0.481 | 0.99758 | -0.262 | 0.9987 |
| **hover** (hovering_nosource_room2) | command | **3** | 2 | — | no fit | — | — | — | — |
| **maneuver** (updown/rectangle/spinning) | command | **2** | 5 | — | no fit | — | — | — | — |
| **warm-up** (< 45 rev/s) | either | **0** | — | — | no fit | — | — | — | — |
| static bench | none | — | — | — | **not measurable** | — | — | — | — |

**The intended ladder does not exist in the data.** The room2 recordings — the
only hover and maneuver material — have no `motors_measured` at all, and their
command-referenced comb barely clears the bar anywhere (0-2 units per window
against 3-11 for room1). `free-flight_nosource_room2` gives **zero** bar-clearing
units across four windows, and the window-free scan finds no comb excess over the
half-integer null there either, so this is not a threshold artefact: room2's audio
does not lock to its command telemetry. Whatever the cause (a different
recording session, a different sync, or a laggier command log), **the hover-vs-
cruise contrast that would test H3 cannot be measured.** Do not read the three
hover units as a result.

### The form of the offset

The free fit's **intercept is +0.044 rev/s with a CI spanning zero**
(-0.374 .. +0.563) while the slope is 0.994 — the textbook signature of a
multiplicative error rather than an additive one. R² = 0.999.

A pure scale (`delta = -0.00542 r`) and a fixed tick miscount in the reciprocal
counter (`delta = -c r²`, fitted c = 6.7e-5, i.e. about **1.6 ticks** of
under-count) fit essentially equally well: RSS 3.94 vs 3.99 over the 56-87 rev/s
range. **These two sub-mechanisms are not separable with the available rates.**
Both are H1; they differ only in whether the bias is 0.54 % everywhere or grows
from 0.38 % at 56 rev/s to 0.58 % at 86.

### Independent replication with no search window at all

`combscan.py` resamples the audio uniformly in telemetry rotor phase and scores a
whole comb at once, `S(s) = mean over k of dB(order = s k)`, scanning s over
+-2.5 %. There is no peak-search window, no band, no per-harmonic gate, and the
null is the same scan on a half-integer comb. On the seven rotor-units whose
on-comb peak exceeds the null by >= 0.9 dB:

| unit | rate | s - 1 | excess over null |
|---|---|---|---|
| nosource_w01 r0 | 86.11 | **-0.580 %** | 2.53 dB |
| whitenoise-low_w01 r0 | 86.14 | **-0.568 %** | 2.31 dB |
| speech-low_w01 r0 | 86.17 | **-0.488 %** | 2.06 dB |
| whitenoise-low_w01 r3 | 74.17 | **-0.492 %** | 1.98 dB |
| nosource_w01 r1 | 75.56 | **-0.582 %** | 1.69 dB |
| whitenoise-low_w01 r1 | 75.14 | **-0.538 %** | 1.67 dB |
| whitenoise-low_w01 r2 | 86.22 | **-0.640 %** | 0.95 dB |

Mean **-0.555 %**, median -0.568 %. Units below the null threshold scatter over
the whole +-2.5 % grid, as they should. A completely different estimator, the same
answer.

---

## 5. The peak-search-window bug, and what survives it

The search half-width is `min(1.5k, 8)` Hz = `min(1.5, 8/k)` rev/s, so it shrinks
as 1/k and a constant -0.42 rev/s displacement leaves it at k ~ 19. The bug is
real. `recentre.py` re-runs the full sweep with the carrier at `k s_r g_r(t)`,
`s_r` estimated per (window, rotor) from the bar-clearing k <= 13 units, and
re-runs the half-integer null re-centred identically.

DREGON cruise, 9 windows x 4 rotors, counts over the 6 dB bar:

| band | re-centred on | re-centred null | uncentred on | n | residual |
|---|---|---|---|---|---|
| k2-13 | **57** | 3 | 57 | 432 | +0.087 % |
| k14-25 | **7** | 4 | 4 | 432 | +0.001 % |
| k26-40 | 3 | 5 | 2 | 540 | -0.119 % |
| k41-60 | 1 | 0 | 1 | 720 | -0.176 % |
| k61-100 | 2 | 0 | 1 | 1297 | +0.062 % |

FLY124 cruise, same protocol: k2-13 **95** over the bar (null 36), k14-25 **21**
(null 2), k26-40 1 (null 0), nothing above.

**Re-centring changes almost nothing.** It lifts DREGON's k14-25 count from 4 to 7
out of 432 against a null of 4, and does nothing elsewhere. So the DREGON/FLY124
asymmetry is **not** the bug's signature: after the fix FLY124 still has 11 % of
its k14-25 units over the bar against DREGON's 1.6 %.

### But the high-k comb IS there — the limit is coherence time, not the window

The user is right about 6 kHz. Spectral autocorrelation of a DREGON cruise slice
(`free-flight_nosource_room1` at t = 30.5 s, rotors at 87.5/74.0/84.1/74.4 rev/s):

| segment | 1-2 kHz | 3-4 kHz | 5.5-6.5 kHz |
|---|---|---|---|
| 0.10 s | **170 Hz** | **170 Hz** | **170 Hz** |
| 0.25 s | 232 Hz | 172 Hz | 44 Hz |
| 1.0 s | 172 Hz | 172 Hz | 41 Hz |
| 4.0 s | 79 Hz | 42 Hz | 40 Hz |

172 Hz is the blade-passage rate of the 86 rev/s rotor (2 x shaft). The comb is
alive at 6 kHz — that is k ~ 75 in shaft units — on 0.1 s segments and gone by
1 s. `measure_displacement.seg_len_env` floors its STFT segment at **1 s**, and
the order spectrum in `combscan.py` uses 16 s, so **both "no high-k line" readings
are coherence-limited, not evidence of absence.** The physical cause is the
shaft's own ~0.2 rev/s rms wander: at k = 75 that is 15 Hz of frequency wander,
which smears the line across a 1 s segment.

`shortscan.py` re-runs the window-free comb-scale scan on 0.25 s segments (about
20 revolutions, order resolution 0.05, against a 0.4-order effect at k = 75), with
the half-integer null; results in `shortscan.json` and `figs/F11_shortscan.png`.
SHORTSCAN_RESULT

None of this changes the low-k measurement or the verdict: the 0.54 % is measured
at k = 2..13, where the search window is 0.6-1.5 rev/s wide, well outside the
displacement, and it is confirmed by an estimator that has no window at all.

---

## 6. CONSEQUENCE

**DREGON's rotor telemetry over-reports rotor speed by 0.542 % (CI 0.450-0.618 %).
At cruise (56-87 rev/s) that is a systematic bias of 0.31-0.47 rev/s, mean
0.424 rev/s. Every DREGON RPS number this project and the literature report is
measured against biased labels.**

- It is **twice** the 0.2 rev/s "honest 0 dB label-jitter floor" the flagship work
  assumes.
- It is **23 %** of the blind-VK DREGON cruise bar (1.807) and **41 %** of the
  steady bar (1.030). A tracker that found the true shaft rate exactly would still
  be scored at ~0.42 rev/s error on DREGON.
- **Both telemetry channels carry it** (they agree to 0.04 %), so switching from
  `motors_measured` to `motors_command` does not help.
- It is a **scale**, so it is removable by one constant per dataset — the same
  form and the same order as the +0.70 % correction measured for Michael's rig in
  WP14, in the opposite direction. The correction is x0.99458.
- Separately, DREGON labels carry a **0.269 rev/s quantisation step at 80 rev/s,
  refreshed at only 49.7 Hz**. That is not a bias, but it is the same size as the
  bias and it is the floor on per-frame label accuracy.
- FLY124's residual is **-0.063 %**, so this is a DREGON property, and FLY124's
  near-zero value is independent confirmation that the 2026-07-31 Michael's
  recalibration worked.

Do not over-read the precision. The 0.542 % rests on 54 units from 13 windows of
5 recordings of one flight session in one room, all at 56-87 rev/s.

## 7. What is still open

1. **Scale vs fixed tick miscount is unresolved.** Both fit equally well over
   56-87 rev/s (RSS 3.94 vs 3.99). Separating them needs bar-clearing units at
   低 rate — warm-up and spin-up windows, where the current estimator finds **zero**
   units over the bar. A low-rate-capable estimator would settle it and would say
   whether the correction is one constant or a rate-dependent curve.
2. **The hover/maneuver rungs are missing.** room2 has no tachometer and does not
   lock to its command telemetry, so the regime contrast that would formally
   exclude H3 was not measured. H3 is rejected on physical grounds (co-mounted
   source and receiver, hence no Doppler) plus the exactly-multiplicative form,
   not on a hover-vs-cruise comparison.
3. **No external ground truth.** An optical tachometer trace, or the ESC firmware's
   tick constant, would close this outright. Neither is in the dataset, and the
   DREGON distribution ships no documentation of the channel.
4. **The high-k comb has not been checked for the same displacement** beyond the
   short-segment scan above; a properly short-segment tracker over k = 40..100
   would test whether the whole comb, not just k <= 13, sits at -0.54 %.
5. **Do not "fix" the labels yet.** The correction is a single constant, but it
   should be applied the way WP13/WP14 applied Michael's — measured per recording,
   validated by a post-shipped residual scan — not inferred from 13 windows.
