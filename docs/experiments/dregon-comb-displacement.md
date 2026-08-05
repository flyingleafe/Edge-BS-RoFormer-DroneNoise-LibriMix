# DREGON comb displacement — the telemetry over-reports by 0.54 %

**Status:** in progress (2026-08-05, one day) · narrative record in
`docs/experiments/beat-vk.md` (sections dated 2026-08-05) · code, raw
reports and JSON in `scripts/displacement/`.

> Three claims were made here and two were withdrawn within the same day.
> Read the "what died" section before citing anything.

## Motivation

The blind RPS tracker degrades slightly on DREGON cruise while clearly
helping FLY124. The standing explanation was the "displaced comb": the low
harmonics sit below `k * telemetry` in translating flight, so any
audio-locked refiner is charged a bias by a telemetry-referenced metric.
The property had been inferred from probes but never seen per harmonic.
The user asked for it to be measured and plotted — where the telemetry
line stops agreeing with the comb, which harmonics agree and which do not
— and, when the first answers looked wrong, kept pushing until the
measurement was right.

## Method

Heterodyne harmonic `k` around `k * g_r(t)` with the tracker's own demod
bank, take the short-time envelope spectrum, read the ridge as a
shaft-rate offset in rev/s, gate out frames where a neighbouring rotor's
line enters the search window. 15 frozen-protocol windows, pin
`beatvk-valid-raw@54849c13ed3a`. Plus, crucially, **three independent
estimators** built to be immune to each other's failure modes: an
off-comb-carrier null, a window-free order-space comb scan
(`combscan.py`, resamples audio uniformly in telemetry rotor phase and
scores a whole comb at once), and a coherent pulse-pair estimator.

## THE RESULT

**DREGON's rotor telemetry over-reports rotor speed by 0.542 %
(95 % CI 0.450-0.618 %).** At cruise (56-87 rev/s) that is a systematic
label bias of 0.31-0.47 rev/s, mean **0.424 rev/s**. Every DREGON RPS
number this project and the literature report is measured against biased
labels.

- **Twice** the 0.2 rev/s "honest label-jitter floor" the flagship work
  assumes.
- **23 %** of the blind-VK DREGON cruise bar (1.807) and **41 %** of the
  steady bar (1.030). A tracker that recovered the true shaft rate
  exactly would still score ~0.42 rev/s on DREGON.
- **Both telemetry channels carry it** (they agree to 0.04 %), so
  switching `motors_measured` <-> `motors_command` does not help.
- It is a **scale**, removable by one constant per dataset: **x0.99458**.
  Same form and order as the +0.70 % correction measured for Michael's
  rig in WP14, in the opposite direction.
- FLY124's residual is **-0.063 %**, so this is a DREGON property — and
  independent confirmation that the 2026-07-31 Michael's recalibration
  worked.
- Separately: DREGON labels carry a **0.269 rev/s quantisation step at
  80 rev/s, refreshed at only 49.7 Hz**. Not a bias, but the same size as
  one, and the floor on per-frame label accuracy.

Evidence, in the order that makes it hard to escape:

1. **The reference channel is a real tachometer, not a setpoint.**
   `motors_measured` values lie on a reciprocal-integer lattice
   `v = 1/(n * 42.0 us)` (fitted 1/delta = 23809.53 Hz, residual rms
   0.0006 of a step) — the signature of a period counter timing
   commutation events. A commanded value has no reason to live on such a
   lattice. (Inferred from the lattice; the DREGON tree ships no
   documentation of either channel.)
2. **The command channel does not explain it.** Per-rotor
   `measured/command` over five room1 recordings agrees to 0.04 % mean,
   0.11 % worst case, with no consistent sign — an order of magnitude
   below the 0.54 % acoustic gap.
3. **The audio clock is not the cause.** The 0.54 % is degenerate between
   "telemetry over-reports" and "the audio clock runs slow" — the same
   degeneracy WP14 flagged on Michael's rig. Broken directly here:
   GCC-PHAT of the onboard re-recording against the shipped
   `emitted_signals/2min_white_noise.wav` gives a clock error of
   +0.0023 % / -0.0081 % / -0.0074 % on three recordings, two independent
   devices agreeing to better than 0.03 % — **20x smaller than the
   effect**. A 0.54 % clock error would have shown ~420 ms of drift over
   78 s; observed drift is under 5 ms.
4. **A completely different estimator agrees.** The window-free
   order-space scan (no peak-search window, no per-harmonic gate,
   half-integer-comb null) gives **-0.555 %** mean / -0.568 % median over
   the seven rotor-units that clear the null by >= 0.9 dB.
5. **The form is multiplicative.** Free fit: slope 0.99403, intercept
   +0.044 rev/s with a CI spanning zero (-0.374..+0.563), R^2 = 0.999.

Unresolved sub-mechanism: a pure scale (`d = -0.00542 r`) and a fixed tick
miscount in the reciprocal counter (`d = -c r^2`, c = 6.7e-5, about
**1.6 ticks** of under-count) fit equally well (RSS 3.94 vs 3.99) over the
available 56-87 rev/s range. Both are the same verdict; they differ only
in whether the bias is 0.54 % everywhere or grows from 0.38 % at 56 rev/s
to 0.58 % at 86.

Do not over-read the precision: 54 units from 13 windows of 5 recordings
of one flight session in one room, all at 56-87 rev/s.

## The displacement itself

Real, one-sided, and larger than first reported. Pooled SNR-weighted
values are DILUTED by harmonics with no measurable line (they contribute
noise centred on zero). Restricted to the 34 DREGON low-k units clearing
6 dB over the off-comb null: **-0.424 rev/s** (pulse-pair -0.231, biased
toward zero at low SNR); FLY124 -0.051 over 74 units. This vindicates the
original 4-probe estimate of 0.3-0.5 rev/s.

**Lesson: report the bar-restricted statistic, never the pooled
SNR-weighted one.**

The wiggle the user spotted on the k=2 ridge is a genuine shaft-rate
deviation from telemetry, not interference: `delta_k` is FLAT in k
(-0.51,-0.51,-0.53,-0.34 at k=2,4,6,8, where a fixed-frequency artifact
would give -0.51,-0.25,-0.17,-0.13); with the (k=2,k=4) pair fixed a
priori r = +0.65..+0.83 over four windows with slope brackets excluding
0.5 and bracketing 1.0; twin beating, rotor permutation and telemetry lag
are all ruled out. Detail: `scripts/displacement/wiggle.md`.

## What died, and why (methodology worth keeping)

### WITHDRAWN — "the high-k comb tracks telemetry to 0.086 rev/s"

Killed the same day by null controls. Running the identical pipeline at a
carrier where no rotor line can exist returns **0.0857** against the
measured **0.0856** — ratio 1.00 (FLY124 1.01).

The trap: a peak-pick inside a search window of half-width W returns ~W/2
on PURE NOISE, and the half-width here is `min(1.5k, 8)` Hz = `<= 8/k`
rev/s, shrinking as 1/k, so high-k noise peak-picks average to a small,
impressive-looking number. A pulse-pair estimator does not rescue it — it
returns ~0 on symmetric in-band noise too, so agreement between the two
estimators was never evidence.

The derived reading — "20x headroom, the residual blind error is estimator
error not physics" — is **withdrawn**, and figure F3 must not be used. The
seed-first neural program keeps its historical justification (every neural
failure was a capture/assignment failure) but loses this headroom number.

**Lesson: run an off-comb-carrier null before believing any demod-derived
number.**

### REFUTED — "the search-window bug explains the DREGON/FLY124 asymmetry"

Proposed (by me) when the window bug was found, and tested rather than
assumed. `recentre.py` re-runs the full sweep with the carrier at
`k * s_r * g_r(t)`, `s_r` fitted per (window, rotor) from bar-clearing
k<=13 units, with the null re-centred identically. **Re-centring changes
almost nothing**: DREGON k14-25 goes from 4 to 7 units over the bar out of
432, against a null of 4. After the fix FLY124 still has 11 % of its
k14-25 units over the bar against DREGON's 1.6 %. The window bug is real
but it is not the explanation.

### RESOLVED — the high-k comb IS there; the limit is coherence time

The user's observation that DREGON clearly has comb structure near 6 kHz
is correct, and it explains the residual. Spectral autocorrelation of a
DREGON cruise slice (rotors at 87.5/74.0/84.1/74.4 rev/s):

| segment | 1-2 kHz | 3-4 kHz | 5.5-6.5 kHz |
|---|---|---|---|
| 0.10 s | **170 Hz** | **170 Hz** | **170 Hz** |
| 0.25 s | 232 Hz | 172 Hz | 44 Hz |
| 1.0 s | 172 Hz | 172 Hz | 41 Hz |
| 4.0 s | 79 Hz | 42 Hz | 40 Hz |

172 Hz is the blade-passage rate of the 86 rev/s rotor. The comb is alive
at 6 kHz — k ~ 75 in shaft units — on 0.1 s segments and gone by 1 s.
`measure_displacement.py` floors its STFT segment at **1 s** and
`combscan.py` uses 16 s, so **both "no high-k line" readings are
coherence-limited, not evidence of absence.** Physical cause: the shaft's
own ~0.2 rev/s rms wander is 15 Hz of frequency wander at k=75, which
smears the line across a 1 s segment. Consistent with the previously
measured decoherence budget (tau_k ~ 0.4-1.7 s at k=8-40).

**Actionable consequence: high harmonics are usable, but only with
segments short enough to beat the wander.** The short-segment scan
(`shortscan.py`, 0.25 s ~ 20 revolutions) was still running when the
agent stalled; `dregon_telemetry.md` carries a `SHORTSCAN_RESULT`
placeholder to be filled.

### NEEDS RE-CHECK — Arm A (high-k anchor)

Arm A ran with a k floor of 16 and `band_hz=12`; WP18 independently
measured 0 usable DREGON harmonics at B=12, and the comb above k~14 is
coherence-limited at the segment lengths in use. So "the high-k anchor
removes 95 % of the degradation" may mean the stage was made nearly
inert. It beat init on two windows, so not fully inert. Hypothesis, not
verdict — but the T1 note in `beat-vk.md` must not be read as "the k>=16
comb is on-grid and usable".

## What is still open

1. **Scale vs fixed tick miscount** — not separable over 56-87 rev/s.
2. **H3 (a real aeroacoustic displacement) is unsupported but not
   excluded.** The test that would kill it cannot be run: DREGON's
   `individual_motors_recordings/` carry **no telemetry at all** (only a
   number in the file name, which the bench shows is an open-loop
   throttle command each motor undershoots by its own amount, -0.6 % to
   -3.3 % — a per-motor spread no shared telemetry constant can produce).
   And the room2 hover/maneuver recordings have no `motors_measured`,
   with audio that does not lock to their command channel at all
   (`free-flight_nosource_room2`: zero bar-clearing units in four
   windows, and no comb excess over the null either). **The hover-vs-
   cruise contrast that would test H3 does not exist in this dataset.**
3. Whether to apply the x0.99458 correction to DREGON labels, and how to
   report every historical DREGON number if we do.
