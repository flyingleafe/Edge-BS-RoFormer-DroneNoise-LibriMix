# Slot-comb CRF v2: more learnable parameters, the same mechanism (2026-09-04)

Status: ARCHIVED, unsuccessful (user decision, 2026-09-06). Every group
below is implemented as an opt-in flag of `SlotCombNet`
(`src/models/comb_slots.py`, `comb_slots_emission_v2.py`,
`comb_slots_prior.py`; trainer `scripts/train_slot_v2.py`) and was trained
on the paper's splits: the arms lose to the trained neural rows by an order
of magnitude on every split (static comb 5.09 vs 0.46, stochastic 15.6 vs
2.48 on selection, real all-frame 12.7 vs 2.74), the OFF state never fires,
and the arms drift away from their early best while the CRF likelihood
keeps falling. The record, the numbers and the reading are in
`docs/experiments/paper-regime-matrix.md` § "Slot-comb v2 test". The
papers do not mention v2. The design text below is kept as written; the
measurements it starts from are in
`docs/experiments/candidate-tests-2026-09-04.md` (the C1 campaign) and the
mechanism it keeps is `src/models/comb_slots.py` (`SlotCombNet`,
`PartialEmission`).

## 1. What "the same mechanism" means

C1 is a conditional random field (CRF) over a grid of rotor rates. Three
pieces make it what it is, and v2 keeps all three:

1. **Gather at the hypothesis.** The score of rate `r` at frame `t` is a
   function of the spectrum read at the harmonics `k * r`, and of nothing
   else. One set of weights serves every rate, because the read positions are
   proportional to the hypothesis. This is the property a convolution on a
   log-frequency axis cannot have, and it is why the family beats the
   harmonic-stacking CNNs on the static comb with zero trained parameters.
2. **Explain-away by allocation.** A slot that claims a rate claims a share of
   every bin under its comb. Other slots read the floor in that share. Two
   slots on one rotor each see half of it and together score less than one
   slot there plus one on an uncovered rotor.
3. **A chain over time.** Each slot's rate is decoded by Viterbi over the
   grid with a transition cost, and trained by the CRF negative
   log-likelihood (NLL) of the true trajectory, slot-matched. Selection and
   deployment are the same object.

Everything else is a *parameterization* of those three pieces. Today most of
it is hand-set: the running-median floor, the delta-thin read, the equal
weight of the forty orders (relaxed by the 135-parameter emission), the hinge
transition, the Gaussian claim width, the share rule, the discrete octave and
relocate moves, the contrast threshold for silence, and the grid itself. The
design principle of v2 is the one the campaign already used: **each new
parameter group is a family that contains the current setting at
initialization**, so the zero-parameter corner stays inside v2, and training
can only leave it by lowering the CRF NLL.

## 2. What the campaign says v2 must fix

The regime table (per mono frame, PIT MAE in rev/s) of the best trained arm
A6b, decoded at eight microphones, against the best single-microphone
regressor `r4hb_scv2`:

| regime | A6b, 8 mics | regressor, 1 mic | what limits C1 there |
|---|---|---|---|
| DREGON cruise | 0.76 | 2.92 | the label floor (signed bias −0.43 %) |
| FLY124 cruise | 11.6 | 1.24 | two 36 rev/s hover clips read at exactly 2× (7.9 of the 11.6); one rotor of four on a decoy on the 80 rev/s clips |
| ramp, frames in 30-100 rev/s | 19.5 | 3.67 | the hinge transition cannot follow a real ramp; the grid has no state below 30 |
| ramp, all frames | 31.1 | 5.11 | frames below 30 rev/s are outside the grid and the loss |
| ground / zero frames | 60.9 | 1.60 | the model has no "no rotor" state; the zero decision is a hand-set contrast threshold, off |
| DREGON cruise, corner at ONE mic | 8.0 | 2.92 | the read variance: a single mic has 8 times the per-bin variance the mean has |

So the failures are, in order of size: no zero state, no grid below 30, a
transition that cannot ramp, a multiple (double-rate) that nothing in the
unary penalizes, an assignment error that only a hand-written coverage move
repairs, and a single-microphone precision gap of about 3×. Each maps to one
parameter group below.

## 3. The parameter groups

The order is by expected gain per unit of cost. The count after each title is
the number of new parameters.

### 3.1 An OFF state in the chain (4 to 6 parameters)

Add one state `OFF` to each slot's chain. Its unary is a learned affine
function of the frame's contrast statistic (`SlotCombNet.contrast`, the
max-minus-median of the score over the grid) and of nothing that reads level:

    u_off(t) = theta_0 - theta_1 * contrast(t)

The transitions `ON -> OFF` and `OFF -> ON` cost `c_1` and `c_2`, `OFF -> OFF`
costs zero. Viterbi and forward-backward keep the banded recursion for the ON
states and add one scalar per frame:

    alpha_off(t) = logsumexp(alpha_off(t-1), logsumexp_g alpha_g(t-1) - c_1) + u_off(t)
    alpha_g(t)   = logsumexp(banded(alpha(t-1))_g, alpha_off(t-1) - c_2) + s_g(t)

The decode emits 0 rev/s in OFF frames. The loss covers zero frames: the gold
state is OFF where the true rate is below 0.5 rev/s. Initialization
`theta_0 = -1e4` makes OFF unreachable, which is the current model.

Why this and not a threshold: the threshold is calibrated once, off, and
cannot see the neighbors. In the chain, a frame whose contrast is ambiguous
inherits its neighbors' state, which is what the warm-up frames need (their
contrast, 0.294, overlaps the no-rotor band, 0.240 to 0.305). This is the
"silence means zero" convention of the project, made a state of the same
CRF.

### 3.2 The grid below 30 rev/s, and the loss over every frame (0 parameters)

Extend the grid to 10-100 rev/s at the same 0.1 step (900 points, 1.3× the
cost). Frames whose true rate is between 0.5 and 10 rev/s are masked out of
the loss and counted as errors at evaluation. Nothing else changes, but two
things become necessary:

- The 15-bin running median is corrupted below about 20 rev/s, where the
  harmonics are 5 bins apart or closer and the floor IS the comb. Section
  3.4 gives the floor that does not have this problem.
- The transition must let a rotor cross the grid quickly. Section 3.3.

### 3.3 A learned transition (about 30 parameters)

The hinge `pen(offs) = stiff * (max(|offs| - step_free, 0) / step_free)^2`
is built once from `slew = 12 rev/s^2` and `stiff = 40`. A real DREGON ramp
runs at 20 to 40 rev/s^2, which is 6 to 10 grid steps per frame against a
free band of 3.8 steps, so the hinge charges about 90 nats per frame to
follow it, against a salience difference of about 1 nat. The chain therefore
lags and then jumps.

Replace the vector by a learned one. Parameterize it as a symmetric,
non-negative, non-decreasing function of `|offs|`, zero at the origin:

    pen(j) = sum_{i<=j} softplus(d_i),  j = 0..span

with `d_i` initialized so that `pen` equals the hinge. Build the band with
`span` from `slew = 30` (about 40 offsets) so that the learned penalty has
room, while the initial values keep the old hinge inside it. The CRF NLL
trains `d` directly through the log-partition; no other code changes.

### 3.4 The gap gather: a multiple discriminator and a comb-conditioned floor (about 50 parameters)

The one read the current emission never makes is the spectrum **between**
the teeth. A second `CombGather` at the half-integer orders `k + 1/2` gives
`z_gap(k, g, t) = log1p(P((k + 1/2) r_g) / floor)`. It serves two purposes.

**Multiple discriminator.** A hypothesis at `2 r` (the FLY124 double-rate
failure) has every one of its teeth present, so no test on the teeth rejects
it. Its gaps are the odd harmonics of the true comb, and they are NOT empty.
Charge the unary for full gaps:

    s(g, t) -= softplus(mu) * mean_k w'_k * phi'(z_gap(k, g, t))

with `mu` initialized at −8 (off), `w'_k` per order and `phi'` a warp like the
existing one. This is the symmetric partner of the empty-tooth charge, which
rejects a SUBHARMONIC (empty teeth) but not a multiple. With both charges in
the unary, the octave move of the decoder (which made FLY124 worse, 13.3 to
25.9, because its odd-even threshold 0.6 is hand-set) is not needed.

**Comb-conditioned floor.** The running median measures the floor from the
neighborhood of a bin, and below 20 rev/s that neighborhood is full of
teeth. The gaps of the hypothesis are a floor estimate that is correct for
the hypothesis by construction: at the true rate the gaps hold the floor, at
a multiple they hold lines (so the contrast collapses, which is the
discriminator again). Mix the two in log domain:

    log floor_k(g, t) = alpha * log median + (1 - alpha) * log P_gap(k, g, t)

with `alpha` (one parameter, or one per order) initialized at 1. The median
stays detached; the gap level is a read of the data, not a fit, so it needs no
detachment either.

### 3.5 A cross-order emission network (2,000 to 5,000 parameters)

This is the capacity lever, and the only group that raises the parameter
count past a few hundred. The current emission is a weighted mean over
orders, with a weight that depends on a reading and its four neighbors. What
it cannot express is the SHAPE of the harmonic vector: which orders are loud
relative to which, at this rate, on this rig. A two-blade rotor puts its
power on the even shaft orders, a coaxial pair on different ones, and the
informative orders shift with the rate (order 40 sits at 1.4 kHz at 36 rev/s
and at 3.6 kHz at 90). The regressor learns that shape; C1 has no place to
put it.

Give it one, inside the gather: a small 1-D convolutional network **over
the order axis**, applied independently at every `(g, t)`:

    inputs per (k, g, t):  phi(z_k), z_gap_k, lfn_k, k / K, log r_g, is_mean
    net:  conv1d(k, kernel 5, 6 -> 16) -> GELU -> conv1d(k, kernel 5, 16 -> 16) -> GELU -> conv1d(k, kernel 1, 16 -> 2)
    outputs per (k, g, t):  a weight logit  l_k  and an evidence  e_k
    s(g, t) = sum_k softplus(l_k) * (phi(z_k) + e_k) / sum_k softplus(l_k)

The last layer is zero-initialized, so at initialization `l_k = 0`, `e_k = 0`
and the score is the classical mean. The weights are shared across rates
(the hypothesis enters only as the feature `log r_g`), so the read stays
proportional to the hypothesis and one network serves every rate. This is
the harmonic convolution of HarmoF0 and HPPNet, moved from the
log-frequency axis onto the rate grid where the gather already put the
harmonics in register.

Two rules from the campaign carry over. The weights are **bounded below by
zero** (softplus, not `1 + MLP`), which removes the non-finite steps A3 and
A6 met. And the level must not be readable: `lfn` stays the normalized log
floor, and no raw power enters.

Cost: the activations live on the `(B, C, K, G, T)` grid, 40 × 900 × 63 =
2.3 M cells per channel per crop, times 16 hidden channels = 145 MB per crop
per layer in float32. At one microphone this is affordable at batch 4 with
the existing per-channel checkpoint. At eight microphones it is not, and
that is the right order anyway: the fair comparison is at one microphone.

### 3.6 A learned pairwise rate prior across slots (about 16 parameters)

The 80 rev/s clips lose one rotor of four to a decoy. The decoder repairs
this with the relocate move, which is coverage ascent with a hand-written
acceptance rule. The joint model has no term that says "four rotors of one
airframe are near one another and are not multiples of one another".

In the sequential peel, slot `i` is decoded after slots `j < i` are fixed,
so its unary can read their paths:

    s_i(g, t) += sum_{j<i} psi(r_g - r_j(t)),   psi(d) = sum_m v_m * exp(-(d - c_m)^2 / (2 w^2))

with 16 centers `c_m` over −70 to 70 rev/s and `v_m` initialized at zero.
This is explain-away on the RATE axis, next to the existing explain-away on
the frequency axis, and it is trained by the same NLL because the earlier
paths are constants in slot `i`'s chain. The risk is that it learns the
rate spread of the training rigs; the ramp windows, where all rotors move
together, keep it honest, and the ablation must include an unseen rig.

### 3.7 Learned read kernels and claim widths (2 to 80 parameters)

Two constants of the front end are family-dependent, and the docstrings say
so. The read is a single interpolated bin, right for a delta-thin line and
wrong for a Lorentzian one whose width grows with the order. Replace it by a
Gaussian kernel of learned width `sigma_k = softplus(s_0 + s_1 * k)` per
order, applied as K smoothed copies of the spectrogram before the gather (K
× F × T = 5 M cells, cheap), initialized narrow. The claim mask width of the
`CombMaskBank` (1.5 bins) becomes learnable in the same way by rebuilding
the bank each forward (one chunked pass over 250 harmonics, about 0.4
GFLOP). Both start at the current values.

### 3.8 What stays hand-set, and why

- The share rule `pw * (1 - other) + floor * other`. A learned share depth
  makes a slot able to keep evidence it should give up; the identical-rotor
  cell needs the exact share to stay stable.
- The union evidence and the relocate move. They are search, not model, and
  they act after the chain. With 3.4 and 3.6 in the unary the decoder can
  run with relocate only, as it does now.
- The STFT (4096 / 512 at 16 kHz). A second resolution is a different
  design.
- The parabolic sub-grid step.

## 4. What v2 is, in one line

The same CRF, with every hand-set constant on its path replaced by a family
that contains it: an OFF state, a grid from 10 rev/s, a learned transition,
a read between the teeth, a cross-order network at the hypothesis, a rate
prior across slots, and learned read and claim widths. About 3,000 to 6,000
parameters, against 135 today.

## 5. How to test it

The protocol is the campaign's: the frozen real split, per mono frame, PIT
MAE by regime (zero / below-grid / ramp / cruise, per rig), the static and
stochastic comb parts, the single-microphone corner and the
single-microphone regressor as the two references, and **one part at a
time** from the corner, each at three seeds (the campaign's single seed
moved FLY124 by 6 rev/s between arms that differ by an inactive term).

The order of the ablation follows the order of the sections, because each
group's gain is expected on a different regime: 3.1 and 3.2 on the zero and
ramp frames, 3.3 on the in-grid ramp frames, 3.4 on FLY124 cruise, 3.5 on
the single-microphone cruise precision, 3.6 on the 80 rev/s clips, 3.7 on
the stochastic comb. A group that does not move its regime is dropped, not
kept.

Two gates before any of it: the mono arm A8 must land (the fair baseline of
the current emission at one microphone), and the corner-plus-one-part
training must run at batch 4 on one gpushort GPU with the cross-order
network in place, since 3.5 is the only group whose cost is not negligible.
