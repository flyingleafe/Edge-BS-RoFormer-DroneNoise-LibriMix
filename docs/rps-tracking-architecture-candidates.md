# Which architecture can learn to track variable frequencies from partially observed harmonics? (2026-09-03)

Status: synthesis of three months of experiment logs (docs/experiments/, the CKLA,
comb-salience, comb-slot, HG-CKLA and classical-tracking campaigns), plus five
preliminary probes run on 2026-09-03. Sections 1 and 2 are the systematized
knowledge; section 3 derives the design requirements; section 4 is the
candidate shortlist with the evidence behind each; section 5 records the probes.
Fact sheets with per-experiment numbers: the review bundle under the session
scratchpad (A regressors, B synthetic and signal, C classical, D neural-in-
classical); the durable numbers are quoted here with their source campaign.

## 1. The signal: what real drone ego-noise looks like to an estimator

Units rev/s unless stated. "DREGON" = MikroKopter, indoor, 8 mics in the wake;
"FLY124/125" = DJI M100, outdoor, boom ring above the body.

**Harmonic content by order.** Track energy along refined trajectories sits
97.5% in orders k = 1-9, 1.6% in 10-24, 0.7% in 25-49, 0.3% in 50-80. Tooth
contrast over the local floor on DREGON cruise: 6.76 / 1.36 / 0.13 / 0.01 dB
in the same bands; the peak-to-floor null is -0.78 dB, so above k ~ 25 a line
is not separable in a 128 ms STFT although comb-locked energy exists there
(the Vold-Kalman decomposition solves to k_hi 62 on DREGON, 57 on FLY124/125).
On the frozen real cruise clips in a 0.25 s window, the count of harmonics
above twice the local floor is 22 of 80, but that is the noise count (a
periodogram bin exceeds twice its own running median with probability 0.25,
and a decoy rate at 1.37x reads the same 0.24-0.25). What stands out is at
LOW order only: FLY124 has 5 of its first 9 harmonics above 2x the floor and 3
of 9 above 8x, with a band level of 1.80 against the decoy's 0.85; DREGON has
about 3 of 9 at 2x and a band level of 0.89 against a decoy's 0.79. Above
k = 24 both rigs read exactly the decoy value in every column (probe P2). The
comb, as a set of resolvable lines, ends near k = 10 on DREGON and near k =
10-24 on FLY124; at 1 s integration DREGON has ONE clearly usable harmonic
(k = 2, +6.1 dB over the null) and a weak group k = 4-13.

**Blade-pass structure.** Two-bladed rotors put the blade-pass frequency at
2 x shaft rate. On FLY124 the odd-to-even level ratio at the TRUE rate is 0.64
(k <= 24) and at HALF the true rate 0.63; on DREGON 0.87 against 0.86 (P2).
The odd/even test that settles the octave on the synthetic comb cannot tell
the truth from its half on real audio. Odd teeth exist only because the two
blades differ and vanish first as throttle falls. There is no r/2
sub-harmonic comb on either rig.

**Line width and phase.** Shaft wander sigma ~ 0.6, tau ~ 16 ms broadens
harmonic k by ~0.6 k Hz and gives ~0.24 k rad of phase wander per 2048-sample
window (12 rad at k = 50). Real harmonics flicker 4-5 dB frame to frame at
every order. Decoherence times tau_k ~ 0.4-1.7 s at k = 8-40; k ~ 75 is
coherent at 0.10 s and gone by 1 s. Phase lock at the telemetry trajectory is
~0.1 at k = 1-2 and 0.03 at k >= 5; single motors lock at 0.72-0.88 and four
motors on a static bench collapse it to 0.02-0.09: the other rotors, not
aerodynamics, destroy the coherence. Because the refiner demodulates ALONG the
trajectory, 0.5-7 rev/s of jitter still captures the whole comb; line width
bites a fixed STFT, not a tracker.

**Four near-unison rotors.** DREGON cruise is two tight pairs at 0.42 and
0.82 apart (median minimum split 0.42; Michael's 1.05); 71.6% of DREGON cruise
frames have two rotors within 1 rev/s, against 17-25% in the synthetic
families. Per-rotor deviation from the common mode: sd 1.4-1.9 (DREGON),
2.2-4.6 (Michael's). At k = 70 a rotor has 32 other-rotor lines within +-6
rev/s and a clean window of only -0.40 to +0.24. All four rotors of one
airframe lie within 1.25x (at most 1.45x). Ramp excursions are overwhelmingly
common-mode; DREGON ramps are swept at 22-35 rev/s per second, while 85% of
Michael's low-speed frames are a stationary hold at 0.38-0.52 of hover.

**The floor.** 60-68% of the energy is residual after the comb. The residual
is 90% per-mic incoherent on DREGON (70-73% on Michael's) with an 8.5 dB
per-mic spread; DREGON's low band is flow noise on each diaphragm (mean
pairwise coherence 0.014 at 50-200 Hz), the M100 boom array stays acoustic to
50 Hz. Median harmonic level over the local floor: 2.7 dB at cruise, 5.5 dB
on ramps. Level is a cue: zero-regime clips sit at 0.175 of cruise RMS, ramps
at 0.37, and a common rotor-speed mode explains up to 0.58 of band power.
Per-rotor attribution of the residual is not identifiable in flight (VIF
21-118, four rotors make +10.9 dB more low-band power than the sum of four
single-motor clips).

**Labels.** DREGON telemetry is a period counter on a 0.269 rev/s lattice at
49.7 Hz, over-reporting by 0.35-0.85% (0.28-0.68 at cruise) and running 42 ms
early; the DREGON labels are deliberately NOT corrected. Michael's telemetry
had a clock dilation and a +0.70% scale, both corrected. Every good model
over-estimates real cruise by +1-2% (today's error profile), which is the
size of this bias. Speech is mixed at -30 to 0 dB; the recorder rolls off
above 6 kHz.

**What the score surfaces look like.** The margin between the truth and the
best decoy of the Whittle comb score is 1.65 nats on the static comb, 0.105
on the stochastic family, and ~0.03 on real DREGON cruise, where the whole
700-candidate surface lives inside 0.05 nats. Per 0.25 s window, the argmax
of the magnitude score over 20-150 rev/s lands within 2 rev/s of a true rate
in 0-0.3% of DREGON windows and 46-68% of FLY124 windows at k_max 10-40,
then 0% on FLY124 at k_max 80, where the half rate wins 58%: the logarithm
protects the true rate only while k_max stays inside the observed comb. With
the classical seeder's floor (P2) the DREGON argmax runs to the bottom of the
grid (median 20.8 on a 20-150 grid). Probe P5 traced that to the FLOOR: with
a running median of +-30 STFT bins (about +-120 Hz, which sits in the
valleys of DREGON's ~80 Hz tooth spacing) the same Whittle score finds a true
rate on 41-53% of DREGON frames at K = 20 from one microphone, and a wide
(301-bin) or global floor reproduces the 0% exactly. Eight microphones,
power-averaged, take DREGON to 47% and are the only configuration in the
probe whose label-robust margin reaches zero. On the DECODED output (P1c)
the optimum floor is narrower still, 15-31 bins (60-120 Hz): DREGON cruise
4.3 at 31 bins, 9.4 at 61, 41 at 307 bins with 42% of rotor-frames at half. A plain harmonic SUM picks the
half rate in ~50% of windows on both rigs at every k_max (P2). Both the
magnitude and the phase profiles peak 0.4-0.6 rev/s BELOW the DREGON labels
(-0.5 to -0.75%), reproducing the telemetry bias from acoustics alone. Real
audio sits with the stochastic family, not the static one, in every one of
these measures.

## 2. What each model family does with that signal, and the walls

**Regressors** (conv trunk on the linear STFT, GRU or Transformer head,
PIT-MSE). Best real all-frame MAE 2.53-2.68 (`r4a_lr3e4`, `r4hb_scv2`).
Trained on real data alone they read loudness and the speed prior:
frequency-scaling slope 0.03 without label-transforming augmentation, 0.14
with it. Trained on the comb they DO read spacing (slope 0.94 on real audio,
an unseen synthetic family read within 5%), and what remains is a 0.836
scale factor from the level anchor: synthetic pools fix every chunk at RMS
0.1 where real clips sit at 0.041, and the prediction ratio swings 0.12 to
0.90 over a hundredfold level change. Level is nevertheless a cue the task
needs below cruise (forcing one level takes the real-trained zero cell from
2.9 to 61). The pool is exactly permutation-invariant over frequency and
spends comb spacing to 0.04-0.19 bins before the head; frequency aggregation
is the one axis the architecture search never touched. Their cruise error is a per-clip offset the size of the
label bias plus a lag through maneuvers (residual decorrelation 0.46 s), with
no aliases and no outliers. Structural limits: the squared-error optimum is
the conditional mean, so unresolved rotors collapse to a ~10 rev/s fan while
true spread runs 0.2-43; a comb at r is a dilation of a comb at r', which
convolution does not share weights across. They do not cross the rig boundary
(cruise 45 on an unseen rig).

**Salience ports** (CombGather on the linear STFT, CNN trunk, per-bin LSTM,
four Gaussian per-rotor layers, CRF readout). Near-exact on the static comb
(median 0.37) with two fast failure modes: maneuvers (2.5x the steady error,
26% of the total, because the LSTM integrates along a fixed rate bin) and
octave jumps (9% HPPNet, 51% HarmoF0; only where 2r is inside the grid). On
the stochastic family they lock onto 2x for whole clips. On real audio,
zero-shot, they read cruise at about HALF the true speed (median relative
bias -48%); after real fine-tuning HPPNet matches the regressors at cruise
(2.86 vs 2.28) and loses on ramps (16.5), ground (8.5) and below 40 rev/s.
Probe P2b names the mechanism: on the stochastic family the trained port
scores 2.05 with the full comb, 16.2 when the audio is low-passed at
harmonic 40, 39.5 at harmonic 20, 44.8 at harmonic 10, with the half-rate
fraction rising to 0.21. The port learned to read the orders above 40 that
the synthetic family renders cleanly and real rotors never expose.

**The comb-salience / slot family** (Whittle emission from CombGather,
spectral peel or slot allocation, hinge-Viterbi or chain CRF). With ZERO
trained parameters it beats the classical scan on 6 of 7 static cells
(training-matched cell 0.043 against 1.254 classical and 4.37 regression)
and reaches geomean 0.487 (0.432 trained through the CRF); on the stochastic
family 2.80 against 8.67 for trained regressors. On real audio (today): DREGON cruise
4.30 and FLY124 cruise 20.6 on the frozen 8 s clips from ONE microphone
(P1); with the eight microphones power-averaged and a 15-bin floor the
same zero-parameter decoder reads DREGON cruise at 1.49 rev/s, 97.4% of
rotor-frames within 10% of the truth and 1.6% at half (P1c). That is better
than every trained neural model on this protocol (best regressor 2.28) and
inside the classical blind bar (1.83 on the beat-VK windows). FLY124 stays at
19-22 with either mic count: its limit is the harmonic count and the octave,
not variance. On the beat-VK windows the earlier single-mic rows read 8.88 /
5.97 against the bars 1.83 / 3.99. The CRF loss gains 15% then diverges
where the margin is thin.

**CKLA** (pooled features, complex Kalman linear-attention recurrence). The
recurrence degenerates into a multi-horizon accumulator because its
measurements are state-independent (the pool collapses frequency before the
recurrence). The phase-only readout reads frequency (probe slope 1.10) at a
cruise precision of 3.5; cross-drone FLY124 1.10-1.36 is the best neural
number there. Rotation buys nothing measurable.

**HG-CKLA** (state-conditioned complex gather at k f_r, innovation phasors,
per-order weights initialized to the k^2 law, twin gate, CKLA recurrence, a
bounded +-5 rev/s residual around the seed). Stage A refiner: on 37 clips at
identical corrupted inits it removes 18% of the corruption MSE against 5% for
one classical pi_kalman pass; its learned half did not move (flat from
epoch 1). Gates G1 (synthetic capture) and G2 (cruise parity) were never run.
It is the only built component with a per-order reliability model: a masked,
decohered or collided order drops out by weight.

**The classical stack** (multichannel comb-gram + hinge-Viterbi seed, peel,
Vold-Kalman / phase-increment refinement with weights 1/v_k ~ k^2.0 (DREGON)
or k^1.5 (Michael's), floor and crossing gates, coarse-to-fine harmonic caps,
RTS smoothing). Blind cruise 0.68-0.74 (DREGON, against biased telemetry) and
1.03 (FLY124); 0.97-1.22 against a real tachometer. Its limits are all
capture: refinement is inert outside the demod band B/(2k), its fixed point is
~0.15 b0 rather than the truth, band annealing cannot hand off because
capture is set by the error tail and precision by the bulk, seeding costs
53-205 s per 16 s window and is channel-hungry (1.81 at 8 mics, 18.8 at 1),
and the interleaving regime needs line-phase continuity that the seed
discards. Its answer to twins is peeling under delta < bw/k, coupling, or a
joint two-tone fit; increment observations on a pair yield nothing.

**The walls, named.**

- W1 Capture and assignment, not precision. Every neural failure of the
  campaign was a seeding or assignment failure; with an exact init the
  estimator floor is 0.055, the honest 0 dB floor ~0.2.
- W2 The octave is exactly nested (the f0/2 basis contains f0's), so it
  cannot be annealed away; it needs a two-sided term that charges predicted
  lines that land on no energy, and on real audio the odd/even level ratio
  is blind (0.64 vs 0.63).
- W3 Partial observation. Only ~22 of 80 orders are visible, the informative
  ones are k <= ~25, and a model that reads higher orders collapses on real
  audio (P2b). A mean over harmonics lets a coincidence beat a quiet rotor;
  quantile pooling raised worst-rotor acquisition 0.26 to 0.45.
- W4 Twins need phase. Magnitude cannot arbitrate 0.4-0.8 rev/s pairs
  (bias toward the pair mean), and which ridge belongs to which rotor is not
  in the magnitude surface at crossings.
- W5 Microphones and the floor. Eight mics raise the DREGON true-rate hit
  rate 3.7-6.3x for either score and take the zero-parameter decoder from
  8.0 to 1.49 on DREGON cruise, more than any score choice; they cost FLY124
  (18.8 to 21.6), whose margin is a harmonic count, not a variance. The floor
  normalization is a first-order design choice on real audio (0% to 53% hit
  rate, and 4.3 to 41 rev/s on the decode, from the floor width alone) and
  belongs to the learned part of the emission.
- W6 Selection. Synthetic fit and real transfer are anti-correlated within a
  run; monitors that are not the task metric have inverted rankings three
  times; effect sizes below +-0.27 (DREGON cruise) need seeds.
- W7 Labels. DREGON precision below ~0.5 rev/s cannot be scored against
  raw telemetry; Michael's is the calibrated rig.

## 3. Design requirements that follow

1. Gather at the hypothesis (read the spectrum at k r), never convolve
   along linear frequency: the problem's symmetry is dilation. Linear axes
   only; every warped front end failed to train.
2. Evidence normalized per harmonic and per channel (a floor-relative
   ratio), so the model cannot read level or timbre; level is then supplied
   separately as a feature for the zero decision, where it is a real cue.
3. A learned reliability over (order, channel, time) with a soft order
   statistic in place of the mean; low orders first, with a coarse-to-fine
   cap whose admissible rung is set by the phase wrap and the envelope band
   (the basin law 1/(K T)), not by the spectrogram.
4. An explicit octave term: the cost of predicted teeth on empty bins,
   and discrete moves that halve the rate and double the order count.
5. Phase increments as the precision channel only, inside the basin a
   magnitude seed provides (about +-1 rev/s), with weights ~ k^1.5-2 and no
   amplitude factor there; uniform weights and magnitude for detection over
   the grid.
6. Explain-away across rotors (peel or slots) and a temporal MAP with a
   physical hinge slew; twins handled by coupling or peeling, not by more
   evidence.
7. Eight microphones, pooled per channel, for capture (the largest single
   lever measured on DREGON); refinement works at one or two. The floor
   normalization is learned, not fixed: its width decides capture on DREGON.
8. A zero decision from contrast, not from level alone.
9. Train through the deployed decoder (CRF), on real audio plus a synthetic
   family whose harmonics are partially observed the way real ones are, and
   select on the task metric.

## 4. The candidates

**C1. Slot-comb CRF with a learned partial-observation emission.** The
comb-salience / slot family as the seed and coverage model, with three
additions the requirements demand: a reliability head over (order, channel)
that replaces the mean over harmonics, the explicit empty-tooth octave term,
and per-channel pooling of the eight mics. Trained through the CRF on the
honest real pool plus a partial-comb synthetic family, selected on PIT MAE.
Why it can transfer: with zero parameters and eight microphones it already
reads real DREGON cruise at 1.49 on the frozen 8 s clips (P1c), ahead of
every trained model; it cannot read level or timbre by construction; and its
own log predicts that a learned head pays exactly where real audio departs
from the Whittle model. What it still gets wrong is exactly the list of
learnable parts: the octave on FLY124 (a harmonic-count limit that needs the
empty-tooth term and a learned harmonic cap), ramps (the slew and a per-rotor
deviation model) and ground (a zero decision). What must be true: the learned emission must
widen the 0.03-nat margin on real DREGON without the CRF descent diverging.
Cost to test: the trainer exists (`scripts/train_comb_slots.py`); a real-
window data path and the head are one day on one GPU.

**C2. HG-CKLA as the precision stage, seeded by C1, with the three KalmanNet
fixes.** The cell is the classical pi_kalman pass made differentiable: the
measurement is a phase increment at the state-predicted order positions,
each order weighted by a learned reliability, fused and smoothed. It is the
only built piece that handles partial observation natively, and the endorsed
thesis shape (seed, annealed refinement, heads). Fixes: run the scan both
ways (filter to smoother), feed the cell the state differences it currently
cannot see, and derive the gain from a carried variance instead of a
predicted sigmoid. What P4 showed: the v1 refiner improves a realistic
neural initialization on real cruise by 8% in one pass and never hurts, but
its fixed point is not the truth and its own floor is 0.4 rev/s. So C2 is not
a precision stage as built. It stays on the list as the only per-order
reliability model, conditional on the three fixes producing a fixed point at
the truth on the corrupted-label task (a one-day test, retraining the flat
run), and it is the natural cell for a phase term inside C1 rather than a
separate stage after it.
Cost: the seam `SlotCombNet.decode -> HGCKLARefiner.forward` is a script;
retraining the refiner is cheap (its run was flat at 21 epochs).

**C3, tested and withdrawn as a seed: a phase-coherent comb emission.**
The innovation-phasor score over the rate grid (HG-CKLA's measurement used
as a scorer, coarse-to-fine in K) was measured on the real cruise clips
(P5). It does not expose true rates where the magnitude score fails: at lag
one, 87.5% frame overlap makes the gathered band's centroid coincide with the
gather position for any smooth spectrum, so line-free noise scores 0.59 at
K = 10, which is the real off-rate background; its true-rate contrast is
3.6x smaller than the magnitude score's on identical frames; its misses are
unstructured decoys; and the k^2 weight law, right for the estimator inside
the basin, is wrong for a detector over the grid. Phase is a local 2:1
signal within about 1 rev/s of the truth. So phase increments belong in the
refinement stage (C2), inside the basin that a magnitude seed provides, and
the seed is built on the locally floored, mic-averaged magnitude comb. The
whole model is then still a differentiable copy of the classical pipeline;
what P5 fixes is the order of its two evidence channels.

**The shared skeleton, made concrete.** All three candidates are one model
with parts switched on or off, and every part maps to a classical object:

```
X_c(f,t)  complex STFT per channel c (n_fft 4096, hop 512, linear axis)
P_c       |X_c|^2 ; floor_c = running median over frequency (detached)
z_{k,c}(r,t) = log1p( P_c(k r, t) / floor_c(k r, t) )      gather at the hypothesis
u_{k,c}(r,t) = X_c(k r,t) conj X_c(k r,t-1) e^{-i 2 pi k r H/fs}   innovation phasor

emission  S(r,t) = sum_{k,c} g_{k,c} phi(z_{k,c}) / sum g   (C1: learned g = reliability;
                                                            g = 1 is the Whittle score)
        - lambda * sum_k hinge(tau - z_k)                     (C1: empty-tooth octave term)
        + mu * | sum_{k<=K} w_k u_{k,c}/|u_{k,c}| | / sum w   (C3: phase coherence, K annealed)
        channels combined by a learned soft-max, not a mean   (per-channel pooling)
decoder   R slots claim bins by comb templates (explain-away), chain CRF with the
          physical hinge slew; zero decision from grid contrast
refiner   HG-CKLA cells at the decoded tracks: gather u at k f_r(t), fuse angles with
          learned per-order weights, Kalman gain from a carried variance   (C2)
loss      CRF NLL through the decoder (+ MSE after assignment for the refiner);
          real honest pool + partial-comb synthetic; select on PIT MAE
```

The reliability g is a small network over (z_{k,c}, its two rate neighbours,
the floor level, k/K, and a short time context); it is what turns the mean
over harmonics into a learned order statistic and what lets a masked or
decohered order drop out by weight. The empty-tooth hinge charges predicted
lines that land on the floor, which the sub-harmonic r/2 does at every odd
order; it is the two-sided term the octave analysis asks for, learnable in
lambda and tau. The phase term is the seed-stage twin of the refiner's
measurement, with the basin law 1/(K T) giving the anneal K = 3, 5, 10, 20.

**Control: the CNN port on a partially observed synthetic family.** The
data-side alternative: keep HPPNet-l4 and train it on the stochastic family
with a per-clip random harmonic cutoff (10-80) and tooth dropout, warm-started
from the trained port, then score it zero-shot on real audio. If the
half-speed reading disappears, partial observation is a training-distribution
problem and the port stays a candidate; if it does not, the explicit
reliability model of C1/C2 is required. Running on vast at the time of
writing.

## 5. Preliminary probes (2026-09-03)

Sets: the frozen real split (37 clips x 8 mics, 8 s), the salv2 stochastic
validation part. All CPU unless stated.

**P1. Zero-parameter comb family on real clips (channel 0).** Ground /
ramp / cruise / DREGON cruise / FLY124 cruise PIT MAE, `CombSalienceNet` with
the peel + Viterbi decoder and the octave move off: 75.8 / 36.1 / 10.6 /
4.30 / 20.6. On DREGON cruise 76% of rotor-frames are within 10% of the
truth (clip 16 correct to 0.7 rev/s); on FLY124 42% at the truth, 24% at
half, 15% at double. The classical `seed_from_gram` gives the inverse:
DREGON cruise 44 (it lands at the low edge of its 30-100 range, which is
half of 75-86) and FLY124 cruise 6.7. The `ratio` octave gate makes the
half rate the typical answer on DREGON (55% of rotor-frames) and must not
be used on real audio. Reading: the half-rate failure is a property of the
score and its rate range, not of the neural family; the gather + peel +
Viterbi skeleton transfers to DREGON cruise at zero parameters; FLY124 is
the octave wall; ground needs a zero decision.

**P2. Visible harmonics and the aggregation test on real cruise.** About 5
resolvable harmonics per 0.25 s window on FLY124 and 3 on DREGON, all at
k <= 9; above k = 24 the comb reads as a decoy. The odd/even ratio at the
truth equals the ratio at the half rate (0.87 vs 0.86 DREGON, 0.64 vs 0.63
FLY124), because real audio has no empty bins between the lines. The Whittle
argmax finds the truth on FLY124 while k_max stays within the comb (62% at
k_max 20) and never on DREGON, where it runs to the grid edge; the plain sum
picks the half rate in 50% everywhere. Reading: on DREGON, capture cannot
come from a per-window magnitude score at all, and the octave cannot be
settled by an odd/even test; the harmonic cap must follow the observed comb
and a decoy must be charged for the bins it claims.

**P2b. The trained port under comb truncation (stochastic clips).** PIT MAE
2.05 / 16.2 / 39.5 / 44.8 at cutoff orders 80 / 40 / 20 / 10; half-rate
fraction 0.00 / 0.21 / 0.21 / 0.18; true-rate fraction 0.92 / 0.42 / 0.05 /
0.00. Reading: the port's real-audio failure is the partial comb.

**P1c. Floor width and microphone count on the decoded output (37 clips).**
`CombSalienceNet`, channel 0, floor 15 / 31 / 61 / 123 / 307 bins: DREGON
cruise 6.5 / 4.3 / 9.4 / 17.8 / 41.3, FLY124 cruise 19.5 / 20.6 / 26.2 /
26.4 / 24.9. `SlotCombNet` (n_iter 0, the same decoder) with one mic versus
eight mics power-averaged, floor 15 bins: DREGON cruise 8.03 -> 1.49 (97.4%
of rotor-frames within 10%, 1.6% at half), FLY124 18.8 -> 21.6, all 37
clips 22.3 -> 19.3; at 31 bins 6.11 -> 1.65 with the half fraction at 0.000;
at 307 bins averaging does not help (42.7 -> 44.5). The decoder's own octave
move helps DREGON at one mic (6.1 -> 4.4), does nothing at eight, and costs
FLY124 heavily (22.4 -> 31.1). Reading: the seed skeleton is right and needs
eight mics and a narrow local floor; FLY124's remaining error is the octave
under a short comb, which no averaging touches.

**P4. HG-CKLA v1 refiner on realistic initializations (real clips, one
pass).** Corrupted labels (its training condition): 1.164 -> 1.002 overall,
cruise 1.33 -> 1.11 (-16%). Realistic inits, cruise: `r4hb_scv2` 2.28 -> 2.09
(-8.3%, no frame hurt), mixed uni-GRU 2.77 -> 2.58 (-6.9%), HPPNet r4_l4
2.86 -> 2.81 (-2%); the synthetic-only port at 35 is untouched (outside
capture). Oracle init (true labels): the refiner moves away by 0.41 at cruise
(median 0.32, p90 0.84, signed mean +0.02), its own noise floor, of the size
of the label bias. Capture: a constant offset of 0.5 is left as is; 1.0 ->
0.68, 2.0 -> 1.21, 4.0 -> 3.13, 8.0 -> 7.67 (negative offsets alike).
Iteration (M5): one pass is the optimum. `r4hb_scv2` cruise 2.28 -> 2.09 ->
2.12 -> 2.17, with 71% of cruise frames worse at pass 2 than at pass 1; the
corrupted labels 1.16 -> 1.00 -> 1.04 -> 1.10; a pure +2 offset 2.00 -> 1.21
-> 1.01 -> 1.01, parking 1 rev/s from the truth. Reading: the
state-conditioned phase-increment measurement transfers to real audio and
improves the best regressor where it matters, at three times the one-pass
gain of the classical refiner (-5%) and without ever hurting; but its fixed
point is not the truth, its own floor is 0.4, and its pull is a fixed ~40%
of the offset inside +-2 rev/s. As built, HG-CKLA v1 is a one-shot coarse
pull, not a precision stage. The design still holds the only per-order
reliability model in the codebase, and the three untested fixes (state
differences in the cell, a tracked variance for the gain, a bidirectional
scan) are exactly the parts that decide whether a fixed point at the truth
exists; the classical refiner's attractor at 0.15 b0 says the same wall is
there for the hand-built version.

**P5. Phase-coherent comb score on real cruise windows (26 clips, 8 mics,
1586 frame pairs per channel).** Gather verified on synthetic combs to
0.005 rad. True-rate argmax hit rate, single mic, best K: phase 0.090
(DREGON) / 0.260 (FLY124) against magnitude 0.409 / 0.391; all four rotors
in the top-8 peaks 1.1% vs 11.4% (DREGON). Basin contrast at K = 10 on
DREGON: phase 0.040 (uniform weights), 0.012 (k^2), magnitude 0.144. Line-
free noise scores 0.587 at lag 1 (the frame-overlap artifact); lag 2-4
removes it and doubles the contrast, but the basin narrows in proportion and
the hit rate stays at 0.15 at best. Eight mics: phase 0.077 -> 0.287,
magnitude 0.075 -> 0.469 on DREGON; FLY124 gains little. The floor control:
the earlier 0% DREGON magnitude result reappears with a 301-bin or global
floor and becomes 0.528 (K = 20, channel 0) with the 61-bin running median.
Reading: the seed reads magnitude with a local floor and eight mics; phase
is the refinement channel.

**E. HPPNet-l4 trained on the partial-comb family, zero-shot on real.**
Pending (vast job).

## 6. Shortlist and what decides between the candidates

Two candidates survive the day, in this order, plus one control.

**1. The slot-comb CRF with a learned emission (C1).** The only family that
transfers to real audio without training, and the only one whose remaining
errors are exactly the parts a learned emission can hold: a reliability
over (order, channel) in place of the mean, the empty-tooth octave charge,
the floor width, a learned harmonic cap that follows the observed comb
(FLY124 loses the truth at k_max 80 and keeps it at 20), and the CRF
transitions. Eight microphones and a narrow local floor are not options but
requirements. First test: train the emission through the CRF on 8-mic real
windows of the honest pool (with the ramp and ground clips in) plus a
partial-comb synthetic family, select on PIT MAE, and read the frozen split
per phase against 1.49 / 21 / 36 / 76 (DREGON cruise / FLY124 cruise / ramp
/ ground) of the zero-parameter decoder. A day of one GPU. Kill criterion:
the trained emission loses DREGON cruise while fixing FLY124, which would say
the two rigs need different harmonic caps and the cap must be state-driven.

**2. A phase-increment refiner inside the basin (C2).** HG-CKLA's
measurement transfers (one pass removes 8% of the best regressor's cruise
error and never hurts) and its per-order reliability is the right object
for partial observation; but as built its fixed point is not the truth and
its floor is 0.4. Test: retrain the refiner with the state-difference
features, a carried variance for the gain and a bidirectional scan, on the
corrupted-label task, and require a fixed point at the truth (iteration must
not drift out) before it is put behind C1. A day of one GPU. If it fails,
the classical pi_kalman pass stays as the precision stage behind C1, which
already reaches 0.97-1.22 against a tachometer from a good seed.

**Control: the CNN port on partial-comb data (E).** If the port trained on
a randomly truncated comb reads real cruise near the truth zero-shot, the
port family stays a candidate for the seed and the question becomes which of
the two seeds trains better on real windows; if it still halves, the
explicit reliability of C1 is required, not optional.

What is closed by today's measurements: phase as a seed signal (P5); the
odd/even ratio as an octave test on real audio (P2); the k^2 weight law for
detection over a grid (P5); the plain harmonic sum (P2); iteration of the v1
refiner (M5); a mean over harmonics at k_max beyond the observed comb (P2,
P2b); a wide floor (P1c, P5).

