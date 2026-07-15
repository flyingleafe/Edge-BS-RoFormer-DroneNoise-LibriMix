# Coupled Vold–Kalman Order Tracking — Design

**Status:** design approved, implementation in progress · **Date:** 2026-07-14
**Goal (session):** (1) re-annotate the current valid set (DREGON + Michael's)
convincingly better than the RPS predictors alone — or decisively prove the
method cannot; (2) produce SPCup annotations whose per-rotor tracks visibly
ride the harmonic peaks on spectrogram overlays.

**Predecessor post-mortem:** `docs/experiments/rps-trajectory-refinement.md`.
Stages B+C failed because their objective (comb-sampled log-magnitude,
per-rotor, uniformly weighted) is non-attributive — rotors never compete for
spectral energy — so tight rotor pairs (DREGON: ~0.65 rev/s apart) bias the
argmax toward the pair mean (twin capture, −0.44 rev/s). Stage D (phase-slope
demodulation, Fisher weights k²|z|²) was the only unbiased stage. This design
replaces the heuristic comb score with a generative residual functional whose
minimizer is the Vold–Kalman (VK) filter, and keeps stage D's phase-slope idea
as the frequency update.

**Off-the-shelf survey:** PyVKF (github.com/CyprienHoelzl/PyVKF, port of
van der Seijs' MATLAB) is a faithful 2nd-gen VK but (a) GPL-3 — cannot be
vendored, (b) solves the full `T·M` complex sparse system at audio rate —
intractable at 16 kHz × 25 s × (4 rotors × ~40 harmonics) ≈ 16M unknowns,
(c) frequencies are *inputs*; no tracking loop. MATLAB Order Tracking Toolbox
rejected for obvious reasons. We therefore implement from the published math
(Vold & Leuridan 1993; Tuma 2005 bandwidth formula) with the three changes
below. PyVKF is used only as a numerical cross-check oracle in scratchpad.

---

## 1 · The functional

Signal model per channel c:

    y_c(t) = Σ_{i,k} Re[ a_{c,i,k}(t) · exp(j φ_{i,k}(t)) ] + noise,
    φ_{i,k}(t) = 2π k ∫ r_i,   i = rotor (R=4), k = harmonic (k_min..k_max)

Minimize over envelopes `a` (given trajectories r) the discrete functional

    J[a] = Σ_t | y(t) − Σ_m Re[a_m(t) c_m(t)] |²  +  Σ_m ρ_m² ‖Δ^p a_m‖²   (VK-2)

with `c_m(t) = exp(j 2π Σ_{τ≤t} k_m r_{i_m}(τ)/fs)` the phasor, `p = 2`
(2nd-order difference prior), and selectivity `ρ` set from the target −3 dB
bandwidth via the Tuma relation. Normal equations: block system, diagonal
blocks banded (bandwidth 2p+1), off-diagonal blocks diagonal with entries
`conj(c_m) c_n` — the **coupling** that makes tracks compete (explaining-away).

## 2 · Three deviations from textbook VK (the actual design)

### 2.1 Demodulate + decimate (tractability)

Envelopes are narrowband (≤ tens of Hz) by construction. So instead of solving
at audio rate: demodulate `z_m = LP[ y · conj(c_m) ]` (zero-phase filtfilt),
decimate to an envelope grid `fs_env` (default 100–200 Hz), and solve VK-2 on
the decimated grid. The coupling coefficient between tracks m, n on the
decimated grid is `LP[conj(c_m) c_n]` — it survives decimation **iff** the
instantaneous frequency separation |f_m − f_n| is below ~fs_env/2, i.e.
exactly the pairs that genuinely interfere. Unknowns drop from T·M to
(T·fs_env/fs)·M ≈ 200k → seconds, not hours.

### 2.2 Sparse coupling groups (structure)

Coupling only matters where tracks are close. Partition the M = R·K tracks by
union-find on the predicate `min_t |f_m(t) − f_n(t)| < couple_hz` (default
couple_hz ≈ fs_env/2). Each group solves its own coupled banded system;
singleton groups are plain single-order VK (banded, `scipy.linalg.solve_banded`
on the real/imag stacked system or complex spsolve). For DREGON this yields
K groups of the 4 rotors per harmonic index (twin coupling — the killed
Kalman bet's "joint per-order update is mandatory" lesson, now structural);
for SPCup it automatically couples cross-harmonic crossings
(e.g. 4·r₁ ≈ 3·r₂). Grouping is recomputed each outer iteration.

### 2.3 Outer frequency loop (VK gives envelopes, not frequencies)

VK assumes known frequency tracks. We alternate:

1. **Envelope solve** (VK-2 above) given current `r_i(t)`.
2. **Frequency update:** per rotor, per envelope sample, the phase slope
   `δ̂_ik(t) = angle(x_ik(t+1) · conj(x_ik(t))) · fs_env / (2π k)` estimates
   the trajectory error in rev/s; fuse across harmonics and channels by a
   weighted, smoothness-regularized 1-D solve

       min_δ Σ_{c,k,t} w_ck(t) (δ̂_ck(t) − δ(t))² + λ ‖Δ²δ‖²,
       w_ck = k² |x_ck|²  (Fisher weights)

   — a tridiagonal-plus-banded solve, per-sample resolution, **no splines**;
   λ maps to a trajectory bandwidth. Clip per-iteration |δ| ≤ max_step
   (default 0.5 rev/s), add to r, iterate.
3. **Annealing schedule:** start with low k_max (capture: init error δ₀ is
   trackable while k·δ₀ < fs_env/2 → k_max grows as the track locks) and a
   smooth trajectory prior; widen k_max and relax λ over n_outer ≈ 4–8 rounds.

Convergence metric per round: joint residual ratio ‖y − ŷ‖²/‖y‖² (the same
number `harmonic_lsq_residual` reports, now from the VK reconstruction) +
max |δ| update.

## 3 · Practical constraints

- **Track validity mask:** drop (weight-zero) track samples where
  `k·r < f_min` (60 Hz default) or `k·r > min(f_max, 0.45·fs)`, and rotors
  with `r < 5 rev/s` (near-silence carries no comb — known residual regime).
- **Multichannel:** envelope solve per channel independently (embarrassingly
  parallel); frequency update fuses all channels via the weights. Cap at 8 ch.
- **Confidence:** reuse the detuned-comb contrast idea, but VK-native: per
  window, ratio of track envelope energy to the residual energy in its band;
  plus the existing `comb_confidence` gate for triage compatibility.
- **Numerics:** float64 in the solver; p = 2 only (higher orders
  ill-conditioned, cf. Springer high-order VKF paper); assert the Tuma
  denominator positivity (bandwidth too small ⇒ raise with the actual limit).
- **No torch dependency** in the core (numpy + scipy sparse/banded); it is an
  offline annotation tool. Torch allowed only in optional adapters.

## 4 · Module layout

`src/data_processing/vk_tracking.py`:

- `VKConfig` (dataclass): fs, fs_env=100.0, p=2, bw_hz=1.0 (envelope −3 dB),
  k_min=1, k_max=40, f_min=60.0, f_max=6000.0, couple_hz=None (→ fs_env/2),
  n_outer=6, traj_lambda (trajectory smoothness), max_step=0.5,
  k_schedule ("grow" | "fixed"), min_rps=5.0.
- `demodulate(audio, phase, cfg) -> z` — vectorized demod + decimate.
- `vk_envelopes(audio, r, cfg) -> Envelopes` — one coupled solve (all groups).
- `vk_track(audio, r_init, frame_times, cfg) -> VKResult` — the outer loop.
  `r_init` on any time grid; result returned on the same grid plus the dense
  envelope grid. `VKResult`: r_refined, envelopes, per-round residual ratios,
  confidence, coupling-group log.
- `vk_reconstruct(envelopes, ...) -> waveform` (diagnostics/overlays).

Tests `tests/test_vk_tracking.py` (synthetic, all must pass before any real
data):
1. Single rotor, known wobble ±2 rev/s @ 0.3 Hz, 20 harmonics, SNR 10 dB:
   recover r within 0.05 rev/s RMS from a +1.5 rev/s biased constant init.
2. **Twin pair 0.65 rev/s apart** (the B+C killer): init both at the pair
   mean; both recovered with |bias| < 0.1 rev/s.
3. Crossing tracks (two rotors crossing in base speed): tracked through the
   crossing without identity swap (check by continuity).
4. Capture basin: init offsets {0.5, 1, 2, 3} rev/s → report basin edge;
   with k_schedule="grow" basin must extend ≥ 2 rev/s.
5. White noise, no comb: confidence ≈ 0, r stays at init (no hallucination).
6. PyVKF cross-check (scratchpad-only script, not a repo test): envelope MSE
   on their demo signal within 2× of PyVKF's.

## 5 · Evaluation plan (separate scripts, after tests pass)

1. **DREGON natural experiment** (`scripts/vk_validation.py`, mirrors
   `rps_refinement_validation.py`): 5 room1 recordings, init = cleaned
   command, truth = 0.25 s-smoothed measured. Report unsigned err / bias
   pooled + per twin pair; bandwidth (λ) U-curve sweep; perturbed-init capture
   basin. Success gate vs predecessors: bias |·| ≤ 0.1 (B+C was −0.44) and
   unsigned err ≤ command's 0.633 (i.e. refinement must not damage).
2. **Valid-set re-annotation** (DREGON-LM-V4-michaels-valid-full clips):
   init from (a) telemetry, (b) SimpleConvV2 predictor outputs; compare
   PIT-MAE/MSE vs predictor alone on clips where ground truth exists; report
   per-regime (cruise/warm-up/ground). VK(predictor-init) − predictor is the
   headline number.
3. **SPCup annotation** (`scripts/vk_spcup.py`, reuses the blind-init triage
   from `rps_refinement_spcup.py`): operating-point init → VK track →
   spectrogram overlays (zoomed panels at low/mid k, incl. the KU Leuven
   maneuvering recording that defeated stage B/C). Deliverable: overlay PNGs
   where tracks visibly ride the peaks through maneuvers + confidence gates.

## 6 · Honest expectations

- VK will not recover DREGON's 0.63 rev/s zero-mean fast jitter (bandwidth vs
  noise trap — measure where the U-curve bottoms out, don't promise).
- True silence (r ≈ 0) remains unannotatable from audio; the mask + confidence
  gate must say so explicitly rather than hallucinate.
- If valid-set re-annotation cannot beat predictors because telemetry is
  already near-truth (DREGON audit result), the decisive comparison is on
  *Michael's* 29 Hz telemetry (genuine headroom between samples) and SPCup
  (no labels at all) — frame conclusions accordingly.
