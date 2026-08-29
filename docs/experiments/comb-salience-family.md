# A neural family that contains the classical comb scan

## Why the regression family cannot solve this, by construction

The `rps_predictor` family maps a log-magnitude spectrogram to R rotor rates and
trains with PIT-MSE. Two things bound it that width and depth cannot move.

**The head is a regressor and the loss is squared error**, so the optimal output
is the conditional MEAN. Where the model cannot resolve individual rotors it
shrinks each estimate toward their average. That is the measured "fan": on
cruise clips spanning 20 rev/s, `comb_floor_deep` places the centre almost
perfectly (75.38 against a true 75.00) and returns a spread of 10.5.

**The features cannot be gathered at a hypothesis.** A comb at rate `r` is a
DILATION of a comb at rate `r'`, not a translation of it. Convolutions share
weights across translations, so on a linear-frequency spectrogram a
convolutional stack has no weight sharing across the one symmetry the problem
has, and must learn a separate detector per rate. The classical scan indexes the
spectrum at `k r` — it reads the bins a hypothesis predicts — and there is no
convolutional equivalent of that.

## The benchmark the campaign was using cannot measure this

Before any of the above could be tested, the instrument had to be checked. The
comb-floor task is not a static-comb task:

* Its fixed validation set carries `flight_reuse: 32`. Ninety-six clips contain
  **twelve distinct rotor trajectories**, each repeated eight times: the audio
  differs per clip, the labels do not.
* **A third of the clips have all four rotors at literally identical speeds** —
  a configuration the training stream never produces (0 of 40 sampled). On those
  there is no spread to resolve, so a collapsed fan is the CORRECT answer, and
  `best.ckpt` was selected by validation MSE on that set.
* Training and validation overlap on neither axis: training spans rotor spread
  8.7-13.9 rev/s at centres 57-92, validation is 33% at spread 0 with centres
  37.4 and 74.
* Speech is mixed in at -30 to 0 dB SNR — the right task for enhancement, the
  wrong one for "given a static comb, restore the rotor speeds".

So the reported comb floor (base 2.535, deep 2.155, wide 2.968) measures
something else, and the conclusion drawn from it that "depth helps, width hurts"
rests on twelve flights, four of them degenerate. Splitting the set confirms the
distortion: `comb_floor_deep` scores 1.084 on the degenerate clips and 2.753 on
the rest.

## The replacement benchmark

`data_processing.comb_bench` generates the task the question names: a pure static
comb, four rotors sharing one harmonic profile, nothing but a small white floor.
Rotor spread, centre and excursion are explicit, so results are a FUNCTION of how
far apart the rotors are. PIT-RMSE, eight clips per cell:

| regime | spread | classical | neural (deep) | oracle centre | prior |
|---|---|---|---|---|---|
| identical | 0 | 1.132 | 4.643 | 0.977 | 1.10 |
| tight | 2 | 1.066 | 4.284 | 1.226 | 1.33 |
| close | 5 | 1.056 | 4.361 | 2.099 | 2.16 |
| **typical** | **11** | **1.254** | **4.374** | **4.207** | 4.24 |
| **wide** | **20** | **0.374** | **4.127** | 7.510 | 7.53 |
| typical-fast | 11 | 3.399 | 5.542 | 5.650 | 6.00 |
| typical-idle | 11 @ 40 | 8.885 | 6.428 | 4.207 | 35.30 |

The fan is unambiguous here: the neural model is **flat at 4.1-4.6 whatever the
rotors do**, and at the training-matched `typical` cell its 4.374 is no better
than predicting the per-frame centre (4.207). That is the parity target — 1.254
at `typical`, 0.374 at `wide`.

## The family, and the corner case

`models.comb_salience` scores a grid of rate hypotheses instead of regressing a
number.

* `CombGather` reads the spectrum at every harmonic of every candidate rate. Its
  gathered offsets are PROPORTIONAL to the hypothesis, so one set of weights
  serves every rate — the operation convolution cannot express.
* `CombScoreHead` turns those readings into a score per (rate, frame). In
  `classical` mode it computes `mean_k log1p(power / floor)`, with no
  parameters.

**The corner case is verified exactly.** On the same periodogram and the same
grid, `CombGather` + the classical head reproduces `tracking.comb_seed.comb_score`
to **4.4e-15** over six clips and 3500 grid points — float64 round-off, not an
approximation. The grid must be passed verbatim: `np.arange` and
`torch.linspace` differ by ~1e-12 rev/s over this range, which at the 40th
harmonic is enough to show as a 1e-11 mismatch. The learned head is initialized
to zero effect, so it too starts as the exact classical score.

Locked in by `tests/models/test_comb_salience.py`.

## A local trap worth recording

This worktree has no `.venv`; it uses the main checkout's, whose editable `.pth`
points at the MAIN repo's `src`. A bare `pytest` therefore tests the main repo,
not the worktree. Every test run and script here needs `PYTHONPATH=src` — the
same lesson already recorded for omnirun jobs, hitting locally.

## The peel is the missing piece, and it was predicted

The end-to-end corner case first scored 17.4 / 16.4 / 12.1 rev/s on the
coincident regimes while doing fine on the separated ones. The cause was a
0.6 rev/s rate-space exclusion hardcoded in the decoder — the SAME mistake this
campaign already measured and documented for the classical peel, where an
exclusion wide enough to suppress a duplicate also forbids the second rotor of a
close pair. Sweeping it reproduced that trade-off exactly:

| min_sep | identical | close | typical | wide |
|---|---|---|---|---|
| 0.02-0.10 | 1.15 | 1.23 | 3.14 | 7.49 |
| 0.30 | 5.19 | 2.48 | **0.76** | 2.15 |
| 0.60 | 17.42 | 11.78 | 2.26 | **1.80** |

Replacing the rate exclusion with a RELATIVE salience test ("does a second peak
stand at `rel` of the strongest?") did not help, and that is the informative
part — it has no rate scale in it and still trades the same way:

| rel | identical | close | typical | wide |
|---|---|---|---|---|
| 0.70 | 10.65 | 5.98 | **0.96** | 2.39 |
| 0.85 | 1.19 | 0.84 | 2.30 | 7.81 |
| 0.92 | 1.13 | 1.69 | 4.64 | 10.72 |

Salience MAGNITUDE does not separate a real rotor from an alias peak, so no
threshold on it can supply model order. The classical method never needed one
because it PEELS: notch the found comb out and the salience at its rate
collapses, so the next peak is genuinely another rotor. That is the
explain-away loop listed as the fourth constructional gap, and adding it as an
unrolled `score -> argmax -> notch -> rescore` fixes the coincident and the
separated regimes AT THE SAME TIME, which no threshold could:

| regime | spread | peel w=1.0 | peel w=1.5 | classical | regression net |
|---|---|---|---|---|---|
| identical | 0 | 1.226 | 1.420 | **1.132** | 4.643 |
| tight | 2 | **0.935** | 0.946 | 1.066 | 4.284 |
| close | 5 | **0.475** | 0.542 | 1.056 | 4.361 |
| **typical** | **11** | 0.259 | **0.209** | 1.254 | 4.374 |
| wide | 20 | 1.334 | 0.691 | **0.374** | 4.127 |
| typical-fast | 11 | 3.865 | 3.581 | **3.399** | 5.542 |
| typical-idle | 11 @ 40 | 25.795 | 28.054 | **8.885** | 6.428 |

**With no trained parameters at all**, the family beats the classical scan on
`tight`, `close` and `typical` — 0.209 against 1.254 on the training-matched
cell, a factor of 6, and a factor of 21 against the regression family's 4.374.
It is comparable on `identical` and `typical-fast`.

Two cells remain behind. `wide` (0.691 against 0.374) is the cell where the
classical pipeline's Viterbi supplies temporal continuity the frame-independent
decoder does not. `typical-idle` was 25.8 against 8.9 for a diagnosed reason:
at a 40 rev/s centre the MULTIPLE at 80 sits inside the 30-100 search grid, and
the net had no octave test. The odd-to-even ratio test is now ported from
`comb_seed`.

## Octave handling: two gates that trade, and one cell still open

The 40 rev/s centre was 28.0 rev/s against the classical 8.9. The salience was
never at fault — dumping it shows the true rates ARE the strongest candidates
(40.92, 44.72, 34.31, 43.12 against truths 34.4 / 38.0 / 42.0 / 45.3). The peel
is what fails: it picks 41.07 correctly and then 52.35, 76.57, 90.39, which are
multiples of the remaining rotors.

The cause is that at a 40 rev/s centre the multiples (65-94) fall INSIDE the
30-100 search grid, while at 75 they fall outside it (139-161). That is why only
this cell needs a downward octave test, and the ported test could not supply one:
it walks upward only, which catches a subharmonic and can never catch a multiple,
whose harmonics are a subset of the true comb's and so are always present.

Adding the downward walk gave two gates that trade, and neither dominates:

| cell | up only | + down (ratio) | + down (score-gated) | classical |
|---|---|---|---|---|
| identical | 1.420 | 1.569 | 1.420 | 1.132 |
| tight | 0.946 | 1.324 | 0.946 | 1.066 |
| close | 0.542 | 0.992 | 0.542 | 1.056 |
| typical | **0.209** | 0.740 | **0.209** | 1.254 |
| wide | 0.691 | 0.909 | 0.691 | 0.374 |
| typical-fast | 3.486 | 3.604 | 3.486 | 3.399 |
| typical-idle | 28.056 | **6.877** | 28.008 | 8.885 |

Score-gating (demote only if the half actually scores better) is exactly right in
principle — a subharmonic scores below the truth by construction, so a true
fundamental can never be demoted — and it restores every 75 rev/s cell to the
digit. It does not rescue the 40 rev/s centre, and the reason is honest rather
than a bug: by the third peel the notch has already removed the neighbours' low
harmonics, so on THAT spectrum the half really does not score better. The damage
happened earlier, in the peel.

Notch width does not fix it either; it is its own trade-off, and in the opposite
direction from what this cell wants:

| cell | w=0.25 | w=0.5 | w=1.0 | w=1.5 |
|---|---|---|---|---|
| typical | 1.946 | 0.543 | 0.259 | **0.209** |
| wide | 6.495 | 4.126 | 1.334 | **0.691** |
| typical-idle | **22.660** | 23.409 | 25.800 | 28.056 |

So `octave_mode` is an explicit choice rather than a tuned constant, and the
40 rev/s centre is recorded as OPEN. Three hypotheses about it have been wrong so
far — that it was a multiple the up-walk could catch, that the ported test would
fix it, and that notch width was the cause.

## Where the family stands, untrained

Best configuration per cell, against both baselines:

| regime | this family | classical | PIT-MSE regression |
|---|---|---|---|
| identical | 1.201 | **1.132** | 4.643 |
| tight | **0.946** | 1.066 | 4.284 |
| close | **0.542** | 1.056 | 4.361 |
| **typical** | **0.209** | 1.254 | 4.374 |
| wide | 0.691 | **0.374** | 4.127 |
| typical-fast | 3.486 | **3.399** | 5.542 |
| typical-idle | **6.877** | 8.885 | 6.428 |

Four cells better than the classical scan, one tied, two behind — **with no
trained parameters at all**. Every number is the classical scan's own score
function re-plumbed into a hypothesis-scoring architecture with an unrolled peel.
Against the regression family the training-matched cell improves by a factor of
21 (0.209 against 4.374).

## Temporal continuity: the last structural piece

`decode_peel` chooses independently in every frame, discarding the one thing a
rotor trajectory certainly has. `decode_peel_viterbi` takes each rotor as a
smooth path through the salience instead, reusing `comb_seed._viterbi_ridge`
so the temporal model is literally the classical one — the hinge cost that is
free up to the airframe's physical slew and steep past it.

| regime | argmax peel | Viterbi peel | classical |
|---|---|---|---|
| identical | 1.420 | 1.216 | **1.132** |
| tight | 0.946 | **0.907** | 1.066 |
| close | 0.542 | **0.420** | 1.056 |
| typical | 0.209 | **0.043** | 1.254 |
| wide | 0.691 | **0.038** | 0.374 |
| typical-fast | 3.486 | **3.062** | 3.399 |
| typical-idle | 28.008 | 25.778 | **8.885** |

It helps everywhere and helps most exactly where it was predicted to: `wide`
went 0.691 -> 0.038, an eighteen-fold improvement, because separated rotors are
where a frame-independent decoder throws away the most. That cell was the one
place the classical pipeline had a structural advantage, and it no longer does.

With the octave gate chosen per cell, best figures:

| regime | this family | classical | PIT-MSE regression |
|---|---|---|---|
| identical | 1.216 | **1.132** | 4.643 |
| tight | **0.907** | 1.066 | 4.284 |
| close | **0.420** | 1.056 | 4.361 |
| **typical** | **0.043** | 1.254 | 4.374 |
| **wide** | **0.038** | 0.374 | 4.127 |
| typical-fast | **3.062** | 3.399 | 5.542 |
| typical-idle | **5.200** | 8.885 | 6.428 |

Six of seven cells beat the classical scan and the seventh is within 7%. On the
training-matched cell the improvement is 29x over the classical scan and 102x
over the regression family (0.043 against 1.254 and 4.374).

**Caveat that is not a footnote.** The octave gate is chosen per cell —
`scored` for the 75 rev/s centres, `ratio` for the 40 rev/s one — and no blind
rule for choosing between them has been found. The obvious candidate (is a
multiple inside the search grid?) does not discriminate, because halving is
admissible at both centres. Running both and selecting by the union-of-bins
joint score from the classical seeding work would decide it blind at twice the
cost, and is untested. Until then the honest claim is per-cell-best, not a
single configuration that wins everywhere.

**Still zero trained parameters.** Everything above is the classical score
function re-plumbed into hypothesis scoring with an unrolled peel and a Viterbi
path. Training the head is running separately.
