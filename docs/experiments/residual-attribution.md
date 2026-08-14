# Per-rotor attribution of the VK broadband residual: REFUTED on both arrays

**Date**: 2026-08-12 · **Status**: DONE — verdict: supervise per-MIC floors.
Code: `src/experiments/residual_attribution/` + `scripts/residual_attribution.py`;
results + figures in `results/residual_attribution/` (gitignored; reproduce in
minutes on CPU, commands in the script docstring).

## Verdict

Per-rotor attribution of the broadband residual is NOT trustworthy on either
array. Amplitude-target training must supervise **per-microphone floors
D_m(f)** (fitted, stored in the result JSONs), not per-rotor PSDs.

1. Geometry is good enough: above 339 Hz (DREGON) / 630 Hz (Michael's) the
   four rotor steering signatures are near-orthogonal (VIF ≈ 1), and synthetic
   data through the repo's own `propagate` recovers known per-rotor PSDs to
   0.5 dB (≈20 dB dynamic range above 500 Hz).
2. The residual is almost all incoherent: the per-mic diagonal term carries
   90.0 % (DREGON) / 69.7 / 73.4 % (FLY124/125) of residual power.
3. The rotor hypothesis is not earned: a WRONG geometry of equal dimension
   (rotor square turned 45°, or random positions) fits the measured
   cross-spectra as well or better, while the same control separates cleanly
   on ideal data. At 1-2 kHz the 45°-rotated geometry beats the truth.

**Epistemics lesson (extends the wind-channel one):** non-degenerate shares
with tight bootstrap intervals (±0.01) can still be UNEARNED. Only a null
control on the geometry itself separates a real spatial fit from a flexible
parameterization absorbing structure. Run geometry nulls before believing any
array-attribution result.

## The number that binds the generator design

Measured per-mic residual energy spread on DREGON: **8.54 dB**. Four equal
rotors radiating 1/r into this array can produce **1.59 dB**. The generator's
propagated per-rotor noise branch therefore CANNOT reproduce the observed
per-mic pattern — an incoherent per-mic term (or per-mic calibration with
enough range) is mandatory in any noise-branch supervision.

Why the free-field four-rotor model fails (not separated): extended/wake
sources rather than hub point sources, reverberation (room1 indoors),
airframe scattering, and diaphragm flow noise (DREGON mics sit in the wake).

Caveats: one DREGON + two Michael's recordings; Michael's geometry is an
unvalidated ring model; residual defined by the 1 Hz-bandwidth VK solve
(leftover tonal energy shows as narrow coherence peaks).

---

# Second pass: band power, a measured basis, and the modulation lever

**Date**: 2026-08-14 · **Status**: DONE — the verdict above holds, and it now
has a mechanism and a bound. Code: `src/experiments/residual_attribution/power.py`
+ `scripts/residual_attribution_power.py`; tests
`tests/test_residual_attribution_power.py`; results in
`results/residual_attribution_power/` (gitignored, minutes on one core).

## Why a second instrument

The first pass fitted four COHERENT point sources to the array cross-spectrum.
A refutation by a null control tells you that a model failed. It does not tell
you which assumption broke. This pass drops coherence and phase, and uses the
two levers that a fully incoherent field still gives:

1. A MEASURED per-rotor per-microphone transfer. DREGON ships single-motor
   bench recordings (`Motor{1-4}_{50..90}.wav`, 8 channels, one rotor running).
2. The time modulation of the four rotor speeds, with no geometry at all.

Rejected alternatives, and why. **Beamforming and steered covariance fitting**:
they need a coherent field, and the measurement below shows there is none.
**Multichannel NMF**: it learns the spatial basis from the data, so it cannot
be falsified by a geometry null control, which is the one control this question
needs. **The physics gate of `wind_wake_gen`**: it predicts one microphone
floor from geometry, but it is a forward model with free constants, and the
bench recordings measure the same quantity directly.

## 1. The residual carries almost no coherent field

Mean magnitude-squared coherence over the 28 microphone pairs, 20 s of cruise:

| Band (Hz) | DREGON | FLY124 | FLY125 | diffuse-field model |
|---|---|---|---|---|
| 50-200 | **0.014** | 0.468 | 0.448 | 0.98 |
| 200-500 | 0.111 | 0.788 | 0.714 | 0.83 |
| 500-1000 | 0.373 | 0.711 | 0.690 | 0.39 |
| 1000-2000 | 0.278 | 0.471 | 0.410 | 0.04 |
| 2000-4000 | 0.171 | 0.203 | 0.229 | 0.02 |
| 4000-8000 | 0.100 | 0.081 | 0.120 | 0.00 |

A diffuse field is the most incoherent SOUND field there is. Below 500 Hz the
DREGON residual is far under it. Thus the DREGON low band is not a sound field
at the array at all. It is flow noise on each diaphragm, and it is
uncorrelated between microphones by construction. Michael's array sits on a
boom, 0.33 m above and forward of the body, and it stays acoustic down to
50 Hz. This is the mechanism behind the first pass: on DREGON the model asked
for a rank-one term where the physics gives a diagonal one.

## 2. The per-rotor basis EXISTS, and the bench measures it

Each single-motor recording gives the per-microphone pattern of one rotor,
with airframe shadowing and wake included. The pattern is stable: the cosine
between the patterns of two neighbouring throttle settings is 0.95 to 1.00
over 50-90 % throttle, in every band. Per band (mean over rotors):

| Band (Hz) | cond | max cos | mic spread | additivity excess |
|---|---|---|---|---|
| 100-250 | **5.0** | **0.244** | 23.1 dB | +10.90 dB |
| 250-500 | 6.0 | 0.378 | 15.7 dB | +10.46 dB |
| 500-1000 | 10.5 | 0.924 | 7.6 dB | +7.49 dB |
| 1000-2000 | 28.7 | 0.969 | 6.9 dB | +3.99 dB |
| 2000-4000 | 10.4 | 0.821 | 7.2 dB | +2.04 dB |
| 4000-8000 | 12.4 | 0.736 | 10.6 dB | **+1.46 dB** |
| 8000-14000 | 10.2 | 0.883 | 14.3 dB | **+0.62 dB** |

The free-field `1/d^2` basis reads cond 14.8 and max cos 0.919 at every band.
So the measured basis is better conditioned than the model below 500 Hz and
above 2 kHz, and it is worse at 1-2 kHz. Its columns match the free-field
columns rotor for rotor (the nearest free-field column is the diagonal one in
every band), which confirms the `Motor{n}` to rotor `n-1` map.

**But powers do not add.** The last column compares `allMotors_70` against the
sum of the four single-motor clips at the same throttle. Four rotors together
make 10.9 dB more low-band power than the four rotors one at a time. The excess
falls with frequency and reaches +0.62 dB at 8-14 kHz. Above 4 kHz the
incoherent-sum assumption holds. Below 1 kHz it fails, and no method that
apportions power there can be correct.

## 3. In flight the four rotors are one degree of freedom

The identifiability test. Project the four rotor speeds onto the quadrotor
control modes (common, roll, pitch, yaw) and measure how much band power the
three DIFFERENTIAL modes explain over the common one. Only that part can ever
tell one rotor from another. The null is the same fit after a block shuffle of
the three differential columns against the data.

| Recording | r2 of the common mode | delta r2 of roll/pitch/yaw | null q95 |
|---|---|---|---|
| DREGON free-flight | 0.001 - 0.072 | 0.040 - 0.087 | 0.090 - 0.146 |
| FLY124 | 0.059 - 0.581 | 0.004 - 0.050 | 0.049 - 0.098 |
| FLY125 | 0.019 - 0.413 | 0.004 - 0.021 | 0.010 - 0.031 |

In 20 of the 21 recording-band cells the differential information is BELOW its
own shuffled null. The one exception (FLY125, 250-500 Hz, 0.0180 against
0.0166) is a coin toss. The cause is in the speeds: the variance-inflation
factor of the four rotor-speed regressors is 21 to 118. After the common mode
is removed, the per-rotor speeds keep a standard deviation of 1.4-1.9 rev/s
(DREGON) and 2.2-4.6 rev/s (Michael's) against a mean near 70 rev/s.

The common mode, on the other hand, explains up to 0.58 of the band power on
FLY124 at 8-14 kHz. The residual IS rotor noise. It moves as ONE source.

Two consequences follow, and both are measured:

- The per-microphone rotor shares from the modulation fit have a 90 % bootstrap
  interval 0.35 to 0.83 wide, on a quantity that lives in [0, 1]. They are
  unconstrained.
- The basis-constrained fit does not beat its own rotor permutation. With the
  measured bench basis on DREGON the true assignment reads r2 0.5035 at
  8-14 kHz against 0.4988-0.4999 for the three rotations. At three of the seven
  bands a rotation WINS, and at a fourth the two tie. The free-field basis
  behaves the same way on all three recordings.

The estimator itself is not the problem.
`tests/test_residual_attribution_power.py` gives it four rotors with
independent speeds and a known basis, and it recovers the shares to 0.1, picks
the true assignment over every permutation, and finds the differential modes
above their null. Under collinear speeds the same code fails, exactly as the
flight data does.

## Verdict, second pass

Per-rotor attribution of the broadband residual is not identifiable in flight,
for THREE independent reasons, one per lever:

1. **No steerable field.** Below 500 Hz on DREGON the residual is under the
   diffuse-field floor, so phase carries nothing.
2. **No differential excitation.** A quadrotor in cruise holds its four rotors
   within a few percent of one speed, so the four sources have one common
   modulation and the design has a VIF of 21-118.
3. **No additivity where the patterns are sharpest.** The bench patterns
   separate best below 500 Hz (max cos 0.244), and that is exactly the band
   where four rotors together make 10.9 dB more than the sum of their parts.

What IS attributable: the rotor SYSTEM against the ambient, and each
microphone's own floor. The bench basis is a real, stable, measured object, and
it stays useful as a RENDERING basis even though it cannot be inverted.

## What the generator should consume

1. **Per-microphone broadband floors `D_m(f)`**, as the first pass concluded.
   The measured per-microphone spread of the residual is 4.0-11.4 dB (DREGON),
   7.7-12.0 dB (FLY124) and 6.0-13.4 dB (FLY125), per band. Four equal 1/r
   sources into the DREGON array give 1.59 dB. A propagated per-rotor noise
   branch alone cannot make that pattern.
2. **One common rotor-system driver**, not four. The common mode explains up to
   0.58 of the band power. Four independent per-rotor broadband levels are not
   supervisable from any recording in hand.
3. **The bench basis as a fixed rendering kernel** (`bench.json` -> `basis`,
   `(mic, rotor, band)`), if per-rotor broadband sources are wanted for
   structural reasons. It is measured, it is throttle-stable, and above 4 kHz
   it composes to within 1.5 dB. Do not fit it to flight audio, and do not
   use it below 1 kHz.

## Caveats

- One DREGON recording and two Michael's recordings, as in the first pass.
- The bench recordings hold the airframe in a different state from flight. The
  additivity column measures that difference and does not remove it.
- Michael's rig has no bench recordings, so its basis stays the unvalidated
  free-field ring model.
- DREGON's flight is nearly constant-speed after its ramp, so its modulation
  lever is weak by construction, not by accident. A recording with deliberate
  per-rotor speed steps would settle question 2 directly, and no such recording
  exists in this project.
