# v4: the unified Gaussian model — design

Date: 2026-08-17. Status: design frozen for implementation; v3 paths must
stay bitwise intact behind default-off flags.

## The model (one sentence)

A window of drone audio is a zero-mean Gaussian process whose power
spectral density is a smooth broadband floor plus a comb of Lorentzian
lines riding the rotor trajectories:

    M_c(f, t) = S_c(f, t) + sum_{i,k} H_{c,i,k}(t) * L_{gamma_k}(f - k r_i(t))

with L_gamma a unit-peak Lorentzian of half-width gamma_k = max(0.6 k, gamma_min) Hz
(the measured linewidth law), S_c smooth in f (and piecewise/interpolated
in t), and H >= 0 the per-line powers. The phase-noise structure of v3
(theta shaft corrections, psi per-track corrections, both integrated OU
-> D2 penalties) is unchanged and enters through the line positions and
the coherent posterior.

Two deliberate consequences:

- The objective is the MARGINAL Whittle likelihood — the line processes
  are integrated out, so J has no envelope term at all:

      J_v4 = sum_{c,f,t} [ P_c(f,t)/M_c(f,t) + log M_c(f,t) ]
             + lambda_theta ||D2 theta||^2 + sum lambda_psi(k) ||D2 psi||^2
             (+ R(log S) as an explicit penalty, below)

  This is what the H-aware `total_h` bolt-on approximated; in v4 it IS
  the objective. `rent` and `data` are no longer separate stories.

- The decomposition channels are POSTERIOR estimates under the fitted
  model, not products of a separate split stage:
  comb channel = posterior mean of the line processes (block A with a
  ridge — see below); broadband = x − comb (exact identity preserved).
  Per-line powers H are first-class outputs — the generator's amplitude
  targets, by construction.

## The two fixes, concretely

### F1 — the floor inside the likelihood

Replace `masked_smooth_psd`'s projection with a penalized-Whittle fit of
log S jointly with H, per (channel, time-block), NO mask:

    minimize over (g = log S, H >= 0):
      sum_f [ P~(f)/(e^g + sum H L) + log(e^g + sum H L) ]
      + lambda_f || D2_f g ||^2

alternating two exactly-solvable-ish steps, 2–3 rounds:
  (a) H-step: given S, nonnegative fit of the Lorentzian amplitudes to
      the excess — the existing `_lorentzian_design` + NNLS machinery,
      per block (frames pooled by the block's median periodogram P~).
  (b) S-step: given H, damped Newton/IRLS on g with the pentadiagonal
      D2 penalty — banded Hessian (diag + D2^T D2), `solveh_banded`.
      Init g from the CURRENT masked fit (a good warm start and the
      fallback if Newton diverges: keep the iterate with lower objective).

lambda_f is a new hyperparameter and it must be a length scale in hertz
on the frequency axis (the "floor may not vary faster than B_f" knob),
calibrated once on the synthetic fixture to reproduce the current
cepstral-family smoothness (~400 Hz at 32 kHz / n_fft 4096). Time
smoothness stays structural (blocks + the existing time interpolation).

### F2 — proper (amplitude) priors on the envelopes

Block A keeps the banded solver; each track's prior precision gains a
diagonal ridge:

    prior_track = rho_k^2 D2^T D2 + beta_k I,   beta_k = c0 / H_k

with H_k the fitted line power of that (c, i, k) (block-piecewise; use
the block value at each envelope frame). Bandwidths open to the physical
law: b_A(k) = max(b0, 0.6 k) Hz with NO spacing cap — the ridge is what
makes the overlapping system positive-definite (the v3 spacing cap and
the coherent/stochastic split existed only because the prior was
improper). c0 is calibrated ONCE on synthetic: an OU line of known
(H, gamma) in known noise must come back with posterior power within
±20% of the Wiener target across k in {5, 20, 60}; record the
calibration in the tests.

Shrinkage behaves correctly by itself: strong coherent low-k lines have
H >> S·b so the ridge is negligible (no bias); floor-level tracks are
shrunk toward zero (no absorption). The WOLA stochastic split stage is
NOT run in the v4 arm — the comb channel already carries the line flanks.

Phase reading (block B) is unchanged, but reads only tracks whose
posterior is phase-informative: keep the existing concentration gate —
it is exactly the right filter and needs no new machinery.

## What v4 deletes (the simplification ledger)

- the stochastic WOLA split stage (v4 arm) — regions, per-bin gain,
  union bookkeeping;
- the mask in the floor fit (the whole "how wide is the mask" axis and
  its measured pathologies);
- the H-aware / marginal / adaptive-floor / h-lorentzian MEASURE bolt-ons
  (their content becomes the objective itself; keep the flags for v3
  comparison runs, freeze development on them);
- the coherent/stochastic boundary as an estimation concept (survives
  only as a presentation split by coherence time).

## Alternation (per window)

  0. init: carrier from labels/hypothesis; S from the current masked fit
     (warm start); H = 0.
  1. (S, H) step — F1 fit on the ORIGINAL periodogram (not the residual:
     the model explains lines through H, so no subtraction is needed and
     hard-EM bias through the residual is avoided).
  2. A step — banded solve with whitening 1/S and ridge c0/H (F2).
  3. B step — phase corrections from gated tracks (unchanged), under the
     existing k-trust ladder.
  4. repeat 1–3 (3 iterations as today); J_v4 read at convergence.

Note step 1 fits powers on the original: the coherent reconstruction is
NOT subtracted before the fit; H at low k then includes the coherent
line's power, which is correct (H is the line's total power; the
posterior split between "waveform captured" and "power only" is block
A's business, not the fit's).

## Seams and drivers

- `tracking.joint_decompose`: new `fit_floor_powers()` (F1), ridge
  support threaded into the block-A path (F2; `vk_envelopes` gains an
  optional per-track diagonal `ridge` argument), `map_objective_v4()`
  (or `map_objective(v4=True)`) computing J_v4; `JointConfig.v4: bool
  = False` master switch selecting the alternation above.
- `scripts/vk_decompose.py`: `--v4` arm (implies no stochastic stage;
  writes the same envelopes/residual/report products plus an `h_powers`
  table in the npz and a `v4` block in the report).
- `scripts/joint_rescore.py`: `--v4` (ranks by J_v4/cell).
- Everything default-off; the pinned v3b regression and all current
  tests must pass untouched.

## Acceptance gates (in order)

1. Synthetic (the regime-3 fixture + a blanket-band variant): fitted S
   within ±0.5 dB of truth INSIDE dense bands (the v3 failure); comb
   channel takes the lines, broadband PSD ≈ S (no carve, no dents);
   ridge calibration test passes.
2. w01 probe (local): the rotor-0-style carve numbers — comb take in a
   sole-owner band vs the track-free control band — must be comparable
   (no >1 dB extra stripping).
3. w01 rescore, 4 hypotheses at k=65: J_v4 ranking; the fan's
   floor-artifact channel is closed by construction — measure whether
   refined <= telemetry < fan finally holds.
4. Full recordings (cluster): gates >= v3e on all three recordings;
   FLY rotor-0 band carve gone; visual pass.

## Success criterion for the paper

If gates 1–4 pass, v4 is the centerpiece: one Gaussian model, one
marginal-likelihood objective, decomposition = posterior inference,
measure = the objective, amplitude targets = model parameters. The v3
machinery becomes the ablation story (what each simplification breaks).
