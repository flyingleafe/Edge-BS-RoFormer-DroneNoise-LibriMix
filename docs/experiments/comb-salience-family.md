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
