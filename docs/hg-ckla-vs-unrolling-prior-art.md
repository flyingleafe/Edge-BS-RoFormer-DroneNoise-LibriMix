# HG-CKLA against the algorithm-unrolling prior art

HG-CKLA is an unrolled algorithm in the standard sense: `src/models/hg_ckla.py`
states "one cell = one `pi_kalman` outer iteration", and each cell re-reads the
spectrogram through a differentiable soft gather at the harmonic positions its
own current state predicts. That last property is the one the design was built
around, and it is worth noting that it goes BEYOND the usual unrolling recipe —
in most unrolled filters the observation is fixed and only the update is learned.

Prior art consulted, both in the library at fulltext:

- Monga, Li and Eldar, *Algorithm Unrolling: Interpretable, Efficient Deep
  Learning for Signal and Image Processing* (arXiv 1912.10557) — the survey.
- Revach, Shlezinger, Ni, Lopez Escoriza, van Sloun and Eldar, *KalmanNet:
  Neural Network Aided Kalman Filtering for Partially Known Dynamics*
  (arXiv 2107.10043, IEEE TSP 2022) — the closest architecture: it keeps the
  Kalman recursion and learns ONLY the Kalman gain. Already tagged to the
  `complex-ou-layer` project, so this line was engaged before.

## Where HG-CKLA already agrees with the prior art

| principle | KalmanNet | HG-CKLA |
|---|---|---|
| unroll the classical iteration | EKF flow with an RNN inside it | one cell per `pi_kalman` outer iteration |
| learn the gain, not the state | RNN computes the Kalman gain | `sigmoid` gain decides how much enters the state |
| keep the physics | SS model retained, not linearized | WP18 weight law `w_k ~ k^2`, learnable |
| train end to end on the target | state MSE, no ground-truth gain needed | residual loss on the track |
| coarse-to-fine | — | classical `k_caps` anneal per cell |

## Three discrepancies that can cost us

### 1. The temporal block is CAUSAL, but the algorithm it mirrors is a SMOOTHER

The module docstring calls the CKLA block "the random-walk smoother". It is not
one. `models/ckla.py` contains no `bidirectional`, `reverse` or `flip`: the scan
runs forward only, so the cell implements a **filter**. The classical algorithm
it mirrors runs `_rw_kalman_rts` — a forward Kalman pass AND a backward
Rauch-Tung-Striebel pass, so that every frame's estimate uses the whole clip.

This is the most costly of the three. The clips are 8 s and offline, so there is
no causality requirement to honour, and the backward pass is exactly what lets
the classical filter propagate a confident late frame back over an uncertain
early one. KalmanNet's own paper flags smoothing as the extension it had not yet
made ("as the MB KF constitutes the first part of the Rauch-Tung-Striebel
smoother, one can extend KalmanNet to implement high-performance smoothing...
we leave the exploration... for future work"). HG-CKLA inherited the filter half
and named it after the whole.

**Fix**: run the scan in both directions and combine, or add an explicit
RTS-style backward recursion over the cell's own posterior.

### 2. The cell sees the observation difference but NOT the state differences

`HGCKLACell.measure` builds `feats` from `u_re`, `u_im`, `log_mag` and two
scalars (`df_phys`, mask occupancy). The innovation phasor `u` IS an observation
difference — KalmanNet's `Δy` feature — so that much is right.

Missing are the STATE differences KalmanNet feeds alongside it:
`Δx̂ = x̂_{t|t} − x̂_{t|t−1}` (posterior minus prior) and
`Δx̃ = x̂_{t|t} − x̂_{t−1|t−1}` (successive posteriors). The paper's ablation is
explicit that these matter: "Using the sequences of differences as input notably
improves the convergence rate of the MB RNN, indicating the benefits of using
the differences as features."

In HG-CKLA the analogue is available for free — cell `j` knows the increment
cell `j−1` applied, and the state's own time difference. Neither is fed in.

**Fix**: concatenate the previous cell's applied increment and the state's
temporal difference to `feats`. Cheap, and with direct empirical support.

### 3. No uncertainty recurrence: the gain is a function, not a tracked covariance

HG-CKLA's gain is `sigmoid(out[..., 0])` — a learned scalar per (rotor, frame)
read straight off the features. The classical `pi_kalman` instead carries a
posterior variance `p_j` forward and the gain FALLS OUT of it, which is what
makes a frame with no surviving measurements degrade gracefully into a pure
prediction step.

KalmanNet's second architecture is precisely this point: it gives the tracked
second-order moments (`Q`, `Σ`, `S`) their own interconnected GRUs, wired to
follow the gain computation of the model-based filter. The paper reports that
this structured version needs about **2.5e4 parameters against 5e5** for the
generic single-RNN architecture — a 20x reduction — because the structure does
work the parameters would otherwise have to learn.

**Fix**: carry a scalar uncertainty per (rotor, frame) through the cells and let
the gain be computed from it, rather than predicted independently at each frame.

## What HG-CKLA has that the prior art does not

The soft gather makes the OBSERVATION state-dependent. KalmanNet's `y_t` is
given; HG-CKLA re-forms its measurement from the raw spectrogram at positions
its own estimate predicts, every cell. That is the property that makes an
unrolled architecture appropriate here at all — the measurement model is where
this problem's difficulty lives, not the update rule — and it is the reason the
cell exists rather than a CKLA head on a pooled spectrogram, whose frequency
pool collapses the spectral axis before the recurrence can look at it.
