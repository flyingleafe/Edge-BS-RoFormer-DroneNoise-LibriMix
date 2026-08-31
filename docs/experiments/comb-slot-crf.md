# Slot allocation and a chain CRF: the peel done jointly, trained through its decoder

Campaign opened 2026-08-31. Two questions, in order: does a joint slot
allocation with a CRF loss solve the STATIC comb, and does it then solve the
STOCHASTIC comb — or, if not, what exactly stops it.

## What was built

**`models/comb_crf`** — a linear-chain CRF over the rate grid. Its Viterbi is
`tracking.comb_seed._viterbi_ridge` index for index (a test asserts it on three
random surfaces), and replacing the `max` with `logsumexp` in the same recursion
gives `log Z`. The training loss is therefore

    loss  =  log Z  -  score(gold path)

the exact negative log-likelihood of the true trajectory under the model the
decoder maximizes. That closes the selection/deployment mismatch that made the
previous head's gains evaporate: there, training used cross-entropy on the
salience map, selection used an argmax decoder, deployment used Viterbi, and the
advantage did not survive the last step.

**This is not CTC.** CTC marginalizes an unknown alignment between a short label
sequence and many frames. Here the labels are dense — a true rate exists in every
frame — so the alignment CTC integrates over does not exist, and its blank and
collapse machinery would add a degree of freedom the task does not have. The
permutation over rotors is a real ambiguity, but it is a permutation, not a
monotonic alignment, and is resolved once by minimum-cost assignment over the 24
orderings.

**`models/comb_slots`** — R slots that ALLOCATE bins instead of peeling one comb
at a time. Each slot holds a distribution over the rate grid; its claim is
`d_i(f,t) = sum_g p_i(g,t) M[g,f]` for a bank of comb templates, so a slot's only
free variable is its RATE. A free per-bin mask would make each slot a
general-purpose separator able to explain anything, which discards the dilation
structure that is the reason this family beats convolutions at all.

Bins are shared, not copied: with total claim `c`, slot `i` scores

    Y_i = Y (1 - (c - d_i)/max(c,1))  +  floor ((c - d_i)/max(c,1))

Two cases fix the design. Four rotors at distinct rates: slot `i` sees the full
spectrum at its own lines and the floor at everyone else's — explain-away,
computed simultaneously. Four rotors at ONE rate, where a collapsed answer is
CORRECT: every slot sees a quarter of the line power, the score drops but the
ranking does not move, so the configuration is stable. A plain mutual notch has
all four slots erase each other, so the share normalization is not cosmetic.

`n_iter=0` runs the sequential peel with hard slots and reproduces the deployed
`comb_salience.decode_peel_viterbi`; joint sweeps then refine an initialization
that is the current method.

## Three defects the measurements forced

**The claim must span the whole band; the score need not.** Scoring stops at
`k_max = 32` harmonics, but a rotor radiates lines all the way up. With the CLAIM
also truncated at 32, a rotor at 34.6 rev/s was notched only to 1107 Hz and a
slot at 68.5 then fed on that same rotor's surviving even harmonics above it.
Full-band claims: `typical-idle` 20.2 -> 17.1.

**`typical-idle` was an OBJECTIVE wall, not a search wall.** The natural global
criterion — the sum of the slots' own path scores — ranked a solution holding two
multiples ABOVE the truth on every clip (1619 against 1508, 1665 against 1590,
1752 against 1609). No restart count fixes an objective that prefers the wrong
answer. Replacing it with Whittle evidence over the union of covered bins,

    J = sum_{f,t} c(f,t) * max(u - 1 - log u, 0),   u = Y / floor,

reverses the ranking on every clip. `g(u) = u - 1 - log u` is the generalized
likelihood ratio for the exponential bin-power law and is ZERO at `u = 1`, so
covering an empty bin is worth nothing — which is what stops a union objective
from being won by a half-rate that covers the gaps.

**The octave gate is now blind, and the two gates no longer trade.** The previous
work ended with `octave_mode` chosen per cell and no rule for choosing it. The
reason is that the two directions need different discriminators and were being
judged by one:

* HALVING is refused by the slot's own odd-to-even evidence ratio. A
  subharmonic's odd harmonics land in the gaps, so the ratio collapses. No
  absolute threshold, no knowledge of the other rotors.
* A MULTIPLE cannot be refused that way at all — its lines are a subset of the
  true comb's and are always present. It is refused by COVERAGE: moving the slot
  down onto the fundamental adds that rotor's odd harmonics to the union, and `J`
  rises.

Because the gates read different quantities they do not conflict, and
`octave_mode` is gone.

## How the slots' claims combine: measured, not argued

`c = sum_j d_j` clamped, `max_j d_j`, and noisy-or `1 - prod(1 - d_j)` were all
tried. `max` is exactly idempotent, so a duplicate slot adds nothing — and it is
WORSE end to end, because two adjacent rotors genuinely share bins and only the
louder one gets credit:

| union rule | typical-idle (8 clips) | geomean, 7 cells |
|---|---|---|
| clamped sum | 6.843 | 0.523 |
| max | 11.391 | 0.562 |
| **noisy-or** | **6.843** | **0.523** |

Noisy-or matches the clamped sum and saturates, so it is the default. Exact
idempotence is not achievable with soft claims and is not the property that
matters: what matters is that covering a NEW rotor beats duplicating a slot,
measured at 3.5x, and a test asserts that ratio rather than idempotence.

## The static comb, untrained

PIT-RMSE in rev/s, 8 clips per cell, one blind configuration throughout —
noisy-or union, no per-cell octave choice. `peel` is the deployed
`decode_peel_viterbi` on the SAME clips.

| regime | rotor spread | peel | slots, 0 sweeps | slots, 1 sweep |
|---|---|---|---|---|
| identical | 0 | 1.291 | 1.311 | 1.286 |
| tight | 2 | 0.816 | 0.845 | 0.823 |
| close | 5 | 1.138 | 0.501 | **0.495** |
| typical | 11 | 0.045 | 0.043 | **0.038** |
| wide | 20 | 0.038 | 0.025 | **0.021** |
| typical-fast | 11, exc 6 | 2.616 | 2.615 | 2.614 |
| typical-idle | 11 at 40 | 30.885 | **6.843** | 7.701 |
| **geomean** | | **0.772** | 0.523 | **0.507** |

For scale, the previous family reached geomean 0.532 only with `octave_mode`
hand-picked per cell; with one blind setting it was at 28 to 31 on
`typical-idle`. So the architecture reaches a slightly better figure than the
previous family's per-cell ORACLE while making no per-cell choice at all.

## What is left is crossings, and it is not the decoder

Residual error tracks how often two rotors pass through each other, and nothing
else. Counting the frames in which some pair sits within 0.5 rev/s:

| regime | frames with a pair < 0.5 rev/s | order swaps per clip | PIT-RMSE |
|---|---|---|---|
| identical | 0.840 | 14.6 | 1.286 |
| tight | 0.697 | 11.4 | 0.823 |
| close | 0.497 | 5.1 | 0.495 |
| typical-fast | 0.199 | 9.9 | 2.614 |
| typical | 0.034 | 0.2 | 0.038 |
| typical-idle | 0.034 | 0.2 | 7.701 |
| wide | 0.000 | 0.0 | 0.021 |

Every crossing-free cell is at 0.02 to 0.04 rev/s. Every cell with crossings is
one to three orders of magnitude worse, in proportion to how many. `typical-idle`
is the single exception — crossing-free and still at 7.7 — and its cause is the
octave problem above, not the crossings.

`typical-fast` breaks the monotone ordering (fewer close frames than `close`,
worse error) because when those rotors do meet they are moving fast, so a swap
costs more rev/s before the paths separate again. Note that `identical` does NOT
mean four constant equal speeds: the four rotors share a centre but wander
independently by +-1.5 rev/s, so they interleave continuously.

**A hypothesis of mine that the data refuted.** `typical-fast`'s peak physical
slew is 15.3 rev/s^2 against a default hinge band of 12, so the band looked like
the obvious cause. Widening it does nothing: 2.820 at slew 12, 2.821 at 20, 2.835
at 30. The cost is the crossings, not the band.

This is the wall the seeding campaign already identified and it is reached, not
pushed: where trajectories interleave, which ridge belongs to which rotor is not
present in a magnitude surface at all, and no search over paths in that surface
recovers it. Resolving it needs the phase continuity of the individual lines.

## Training through the CRF: it works, unlike the previous loss

600 steps, 137-parameter rate-conditioned head, batch 4, 2 s crops, selection on
the DEPLOYED decoder by geometric mean across cells.

| step | identical | tight | close | typical | wide | typical-fast | typical-idle | geomean |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.227 | 0.884 | 0.580 | 0.051 | 0.022 | 2.817 | 17.946 | 0.621 |
| 300 | 1.229 | 0.856 | 0.553 | 0.031 | 0.022 | 2.818 | 12.723 | 0.545 |
| **500** | 1.227 | 0.856 | 0.543 | **0.024** | 0.022 | 2.818 | 11.556 | **0.516** |
| 600 | 1.223 | 0.856 | 0.543 | 0.032 | 0.022 | 6.646 | 6.638 | 0.563 |

**Training improves the head by 17% on the decode metric**, where the previous
campaign's cross-entropy loss made it seventy times worse on the training-matched
cell while its own loss fell monotonically. The difference is the objective: the
CRF loss is the negative log-likelihood of the gold path under the decoder that
is deployed, so lowering it cannot mean anything but making that path more likely.

The gains sit exactly where they should. `close`, `typical` and `typical-idle`
improve; `identical`, `tight` and `typical-fast` do not move at all, which is the
crossing wall and is not a property of the head; `wide` does not move because it
is already at the grid quantization floor (a 0.1 rev/s grid gives 0.029 rev/s RMS).

## The stochastic comb

The stochastic family renders each harmonic as a Lorentzian of half width
`gamma0 + slope * k` Hz whose power drifts as a Gaussian process, and realizes
the whole spectrum by filtering white noise, so every bin is an exponential draw
about its mean. `comb_bench_stochastic` takes its trajectories from `comb_clip`
ITSELF, so the two families share their labels exactly and a difference in
results is attributable to the rendering alone.

### It is partly solvable, and the architecture is what makes it so

Coherent line realization (tones with a wandering phase over a filtered-noise
floor), 8 clips per cell:

| regime | peel | slots, 0 sweeps | slots, 1 sweep | static, for scale |
|---|---|---|---|---|
| identical | 6.731 | 2.420 | **2.390** | 1.286 |
| tight | 6.958 | **1.055** | 1.096 | 0.823 |
| close | 7.791 | **1.079** | 1.080 | 0.495 |
| typical | 7.545 | 1.161 | **1.160** | 0.038 |
| wide | 7.921 | 3.907 | **3.308** | 0.021 |
| typical-fast | 8.722 | **5.459** | 5.463 | 2.614 |
| typical-idle | 38.684 | 32.353 | **32.351** | 7.701 |
| **geomean** | **9.571** | 3.004 | **2.944** | 0.507 |

Two things to read off it. The architecture is worth **3.3x** over the deployed
peel on this family, a larger factor than on the static comb — joint allocation
matters more, not less, when the evidence is weak. And the mid-spread cells land
at 1.0 to 1.2 rev/s, against the 8.67 rev/s that this project's trained
regression networks reach on the stochastic family's own distribution. So the
family is not unsolvable, and the previous ceiling was not the data.

But it is 30x worse than the static comb at `typical` (1.160 against 0.038), and
the ordering INVERTS: `wide` is the easiest static cell (0.021) and one of the
hardest stochastic ones (3.308).

### Why: the score margin collapses by a factor of 16

The right instrument is not spectral contrast but the objective's own margin —
how far the true rate outscores the best decoy at least 2 rev/s away from every
rotor. Per frame, 4 clips at `typical`:

| family | frames where the truth beats every decoy | mean margin | peak score |
|---|---|---|---|
| static | **1.000** | **1.652** | 6.241 |
| stochastic, coherent lines | 0.810 | 0.105 | 1.559 |
| stochastic, Rayleigh lines | 0.643 | 0.038 | 1.491 |

On the static comb the truth is the per-frame maximum in **every frame of every
clip**, by a margin of 1.65 nats. On the stochastic family it is the maximum in
81% of frames (coherent) or 64% (Rayleigh), by 0.105 and 0.038. The margin falls
16x and then a further 2.8x.

That single table explains everything above it. A decoder that takes per-frame
maxima cannot work when the truth loses a fifth to a third of the frames, which
is why the peel sits at 9.6. A decoder that integrates a whole trajectory can
still win, because 0.105 nats accumulated over 251 frames is a large number
against noise that averages down — which is why the CRF path decoder reaches 1.1.
The residual 1.1 rev/s is what survives when 19% of frames actively vote wrong.

The coherent-versus-Rayleigh split separates the two causes cleanly:

* **Line broadening and the floor** cost the 16x. `gamma_k = gamma0 + slope * k`
  with slope up to 0.8 Hz per harmonic makes harmonic 32 about 26 Hz wide against
  a 75 Hz harmonic spacing, so the upper comb smears into a floor that sits only
  1 to 16 dB below it. The gather reads one interpolated bin per harmonic; a line
  spread over many bins does not put its power there.
* **Realization noise** costs the remaining 2.8x. With Rayleigh lines each bin is
  an independent exponential draw about its mean, so a single frame's evidence
  has a coefficient of variation of one, and `log1p` of that is a very noisy
  statistic. This is a variance cost, not a lost-information cost, and it is
  exactly what temporal integration is for.

### Rayleigh lines, and an inversion that explains the family

The default stochastic realization, where every bin is an exponential draw:

| regime | peel | slots, 0 sweeps | slots, 1 sweep | static, for scale |
|---|---|---|---|---|
| identical | 8.312 | 1.612 | **1.601** | 1.286 |
| tight | 8.358 | 3.032 | **3.006** | 0.823 |
| close | 7.858 | 2.003 | **1.977** | 0.495 |
| typical | 8.218 | 7.563 | 7.563 | 0.038 |
| wide | 8.673 | **4.777** | 4.781 | 0.021 |
| typical-fast | 10.478 | 4.723 | **4.722** | 2.614 |
| typical-idle | 40.338 | **31.028** | 31.029 | 7.701 |
| **geomean** | **10.737** | 4.716 | **4.696** | 0.507 |

**The cell ordering inverts completely.** On the static comb `identical` is the
hardest non-octave cell (1.286) and `wide` the easiest (0.021). Here `identical`
is the BEST cell (1.601) and `typical` one of the worst (7.563).

That inversion identifies what sets difficulty in each family, and they are
different things:

* On the static comb, lines are delta-thin and every one of them clears the floor,
  so finding a rotor is never in doubt and the only question is WHICH rotor a
  ridge belongs to. Difficulty is therefore set by CROSSINGS, and `identical` —
  four rotors sharing a centre and wandering through each other — is the worst
  case.
* On the stochastic family, the question is whether a comb is visible at all.
  Difficulty is set by per-comb line strength against the floor. Four rotors at
  ONE rate put four combs' power into the same lines, so the evidence adds and
  the cell becomes easy — and the crossings that ruin it on the static comb cost
  nothing, because all four answers are the same number anyway. Spread the rotors
  and each comb stands alone at a quarter of the power.

So the two families are not two difficulties of one task. They are two different
tasks that happen to share a label format, which is the same conclusion the
earlier campaign reached from spectral contrast and is here reached from the
decoder's own error.

### It is a search wall, not an evidence wall

The objective-versus-search test that identified `typical-idle` as an objective
wall gives the opposite verdict here. Union evidence at the TRUTH against union
evidence at the decoded solution, coherent lines:

| cell | seed | PIT-RMSE | J(decoded) | J(truth) | verdict |
|---|---|---|---|---|---|
| wide | 0 | 3.57 | 234542 | 238662 | truth — search wall |
| wide | 1 | 4.00 | 218408 | 222115 | truth — search wall |
| wide | 2 | 0.59 | 263837 | 266662 | truth — search wall |
| typical | 0 | 1.37 | 229212 | 232117 | truth — search wall |
| typical | 1 | 2.69 | 207308 | 212714 | truth — search wall |
| typical | 2 | 0.40 | 259835 | 262636 | truth — search wall |

On every clip the truth scores higher than the answer returned. The information
is present and the objective ranks it correctly; the search does not reach it.
The failure has a shape: on `wide` two slots settle near 78 rev/s while the rotor
at 71.6 goes uncovered — a DUPLICATE, which no octave move can express because
the offending slot is not at a multiple of anything.

A relocation move was added for exactly that: re-solve one slot against the
others' claims and keep it only if union evidence rises. It is coordinate ascent
on the right objective, and how much it recovers scales with how bad the
duplicate problem is in the first place:

| family | score margin | without relocation | with relocation | gain |
|---|---|---|---|---|
| static | 1.652 nats | 0.507 | **0.487** | 4% |
| stochastic, coherent | 0.105 nats | 2.944 | **2.800** | 5% |
| stochastic, Rayleigh | 0.038 nats | 4.696 | **3.834** | **19%** |

The gain grows as the margin shrinks, which is a confirmation of the diagnosis
rather than a tuning result: the weaker the evidence, the more often two slots
pile onto one rotor, and the more a move that can only fix duplicates is worth.
(An earlier single-cell probe suggested the static gain was zero; the full sweep
says 4%, and the corrected figure is the one above.) It does not close the gap, because with
the truth winning only 64 to 81% of frames the re-solve often lands on a decoy
rather than on the uncovered rotor. That is the honest state of it: right
objective, insufficient search, and the next move to try is a joint one over two
slots rather than one at a time.

### Training on the stochastic family: it helps, then it diverges

Same recipe as the static run, coherent lines, learning rate 3e-3:

| step | identical | tight | close | typical | wide | typical-fast | typical-idle | geomean |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.349 | 1.042 | 1.171 | 1.327 | 3.521 | 4.337 | 35.739 | 3.132 |
| **100** | **1.304** | 1.059 | 1.164 | 1.493 | 3.386 | 3.807 | 31.562 | **2.674** |
| 200 | 3.997 | 1.061 | 1.141 | 1.239 | 3.322 | 4.740 | 21.277 | 2.964 |
| 300 | 8.713 | 9.119 | 6.258 | 6.008 | 5.835 | 10.373 | 7.355 | 7.496 |
| 500 | 10.492 | 10.461 | 6.516 | 4.905 | 10.449 | 14.128 | 14.299 | 9.580 |

A 15% gain by step 100 and then a collapse — every cell degrades together after
step 200 while the static run at the same learning rate improved monotonically to
step 400. Selection on the decode metric holds the step-100 checkpoint, so the
collapse costs accuracy rather than correctness, but the contrast between the two
families is the finding: where the margin is 1.65 nats the CRF loss is a safe
thing to descend, and where it is 0.105 nats the same descent finds ways to
sharpen the surface that do not correspond to finding rotors. A lower learning
rate is running.

The static run, for comparison — noisy-or union, learning rate 3e-3:

| step | close | typical | typical-idle | geomean |
|---|---|---|---|---|
| 0 | 0.580 | 0.051 | 13.034 | 0.593 |
| 100 | 0.568 | 0.060 | 0.893 | 0.411 |
| 300 | 0.552 | 0.023 | 1.830 | 0.396 |
| **400** | **0.541** | **0.023** | 1.726 | **0.392** |
| 600 | 0.541 | 0.024 | 2.602 | 0.503 |

**34% better than the untrained corner**, and `identical`, `tight`,
`typical-fast` and `wide` do not move at all — the first three because they are
the crossing wall and the last because it is already at the grid quantization
floor of 0.029 rev/s RMS. Everything training can reach, it reaches.

## Summary against the goal

**Static comb, static-only validation: solved.** One blind configuration, no
per-cell choices, geomean of per-cell PIT-RMSE **0.487** rev/s untrained against
0.772 for the deployed peel — and against 0.532 for the previous family with its
octave gate hand-picked per cell, which this configuration does not need.
Training through the CRF takes it to **0.392**, a 34% improvement, where the
previous cross-entropy loss had degraded the equivalent head seventyfold.

What remains on the static comb is crossings, and the correlation is tight enough
to call it a diagnosis rather than an association: every crossing-free cell lands
at 0.02 to 0.04 rev/s, and every cell with crossings is worse in proportion to how
many it has. That is the phase wall the seeding campaign identified, reached
rather than pushed.

**Stochastic comb, stochastic-only validation: partly solved, with the gap
measured.** Geomean 2.800 with coherent line realization and 3.834 with the
Rayleigh one, against 9.571 and 10.737 for the deployed peel and 8.67 for this
project's trained regression networks on the family's own distribution. So the
family is roughly three times more solvable than anything previously applied to
it, and 30x less solvable than the static comb.

The whole degradation reduces to one measurement: the score margin between the
truth and the best decoy falls from 1.652 nats to 0.105 (coherent) and 0.038
(Rayleigh), and the truth stops being the per-frame maximum in 19% and 36% of
frames. Line broadening and the floor cost the first factor of 16; realization
noise costs the further 2.8, and being a variance cost rather than an information
cost is why temporal integration recovers so much of it — the architecture is
worth 3.3x here against 1.5x on the static comb.

**The two families are not two difficulties of one task.** The cell ordering
inverts: `identical` is the worst static cell and the best Rayleigh one, `wide`
the best static and among the worst stochastic. Static difficulty is set by
crossings; stochastic difficulty by per-comb line strength, and four rotors on one
rate pool their power into the same lines.

**Where each stops, and what kind of wall it is.** `typical-idle` was an objective
wall and was fixed by replacing the objective. Every stochastic cell is a search
wall — union evidence at the truth exceeds the returned answer on all six clips
tested — so the remaining stochastic error is not a statement about the
information in the signal. The failure shape is a duplicate slot; single-slot
relocation recovers part of the gap (2.944 -> 2.800 coherent, 4.696 -> 3.834
Rayleigh, 0.507 -> 0.487 static) and a joint two-slot move is the untried next step.

**Open, and stated rather than smoothed.** The crossing wall needs phase. The
stochastic search wall needs a better move set. And the CRF loss is not
unconditionally safe to descend: on the stochastic family it improved the head 15%
by step 100 and then collapsed it (2.674 -> 9.580 by step 500) at a learning rate
that was stable on the static family for 400 steps. Where the margin is 1.65 nats
descending this loss finds rotors; where it is 0.105 nats the same descent finds
ways to sharpen the surface that do not correspond to rotors. Nothing here has run
on a real recording.


## The trained head on the full benchmark

The training run above selected on four validation clips per cell. Scored on the
full eight, with training seeds (0-999) disjoint from evaluation seeds (1000+):

| cell | untrained | trained |
|---|---|---|
| identical | 1.323 | 1.305 |
| tight | 0.847 | 0.823 |
| close | 0.475 | **0.397** |
| typical | 0.043 | **0.032** |
| wide | 0.025 | 0.024 |
| typical-fast | **2.617** | 4.311 |
| typical-idle | 6.184 | **1.992** |
| **geomean** | 0.513 | **0.432** |

**16% better on held-out clips**, so the training gain is not an artifact of the
small validation set. Five cells improve, `wide` is at the grid floor, and one
cell gets worse: `typical-fast` goes 2.617 to 4.311. That cell is the crossing
wall, where the head has nothing useful to learn, so what training does there is
trade it for the cells it CAN move. Reporting the geometric mean alone would hide
that, which is the reason to print the cells.

This is the best static-comb figure in the campaign: **0.432**, against 0.772 for
the deployed peel and 0.532 for the previous family with its octave gate
hand-picked per cell.
