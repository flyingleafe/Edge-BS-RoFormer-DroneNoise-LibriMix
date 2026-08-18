# AGENTS.md — `src/tracking`

The rotor-speed tracking stack: given drone audio, give each rotor's speed over time.
Vold–Kalman order tracking, phase-increment refinement, blind seeding, beam/DP search, and the
goodness-of-fit machinery that judges a trajectory against the audio. Pure array code (see
"Purity rule"). Extracted from `data_processing` in the 2026-08 refactor
(`docs/refactor-2026-08-plan.md` §4, Phase 2), then consolidated in three passes: the
primitives, the front door, and the retirement of the campaign scripts.

## How to read this package

Four layers. Each knows only about the layer below it:

| Layer | Module(s) | What it is |
|-------|-----------|------------|
| **Front door** | `top.py` | every stage, and every shipped variant as a named composition |
| **Ladder cores + frozen configs** | `pipelines.py` | the calibrated multi-step algorithms the stages call |
| **Algorithm cores** | the thirteen modules of §3 | one algorithm each — arrays in, arrays out |
| **Primitives** | `dsp.py` | the transforms every core shares |

`protocols.py` sits beside all four: it declares WHICH recordings, windows and grids the
campaigns score on, as data.

A new reader opens `top.py`, reads the two tables in §1, and then opens only the core whose
paragraph in §3 matches the question. Two invariants make that possible: **no module outside
`top.py` defines a stage**, and **no transform lives outside `dsp.py`**.

## 1. The front door: `top.py`

**`top.py` holds three things and nothing else**: the frame plumbing, EVERY stage (each with
its config dataclass beside it), and EVERY shipped variant of the algorithm as a named
few-line composition. Read it, look at the configs, and you can rebuild any variant by
composing stages in a `for` loop. Everything else in this package is an array core that
`top.py` wires; it wires, it never implements.

Concretely: the frame plumbing (`Stage`, `tracking_frame` — `dtype=` keeps a float64 signal
exact —, `get_audio`/`get_rps`/`with_rps`/`with_meta`, `pipeline`), then the stages, then the
recipes. The frame contract itself is §"The Stage API" below.

The stage vocabulary:

| Stage | Config | What it does |
|-------|--------|--------------|
| `blind_seed_stage` | `SeedConfig` | blind comb scan → constant `rps` init |
| `coarse_init_stage` | `CoarseConfig` | full-range Viterbi c(t) → time-varying init |
| `vit2dsp_stage` | `Vit2dspConfig` | the calibrated blind-annotation ladder |
| `vk_stage` | `VKConfig` | coupled Vold–Kalman order tracking |
| `peel_stage` | `PeelConfig` | subtract the other rotors' combs → a seam in `meta` |
| `pi_kalman_stage` | `PiConfig` | phase-increment Kalman refinement (eats the seam) |
| `warp_stage` | kwargs | iterated time-warp IF refinement |
| `refine_coherent_stage` | `RefineConfig` | coherent phase-slope refinement |
| `presmooth_stage` / `scale_stage` / `shift_stage` | scalars | trajectory candidates for the judge |
| `fitness_stage` | `FitnessConfig` | score the trajectory (does not change it) |
| `fvk_stage` | `FVKConfig` | score it by F_VK, the profiled coupled-VK residual |
| `fvk_refine_stage` | `FVKConfig` | L-BFGS on F_VK under a `k_max` annealing schedule |
| `decompose_stage` | `FVKConfig` | split the audio into per-harmonic tracks + a residual (seam in `meta`) |
| `joint_init_stage` | `FVKConfig` + `JointConfig` | seed the v3 alternation's state (seam in `meta`) |
| `vk_solve_stage` | `JointConfig` | v3 **block A** — one whitened VK solve (`objective=True` adds the MAP readout to its log entry) |
| `phase_split_stage` | `JointConfig` | v3 **block B** — the shaft / per-rotor / per-track phase split |
| `floor_stage` | `JointConfig` | v3 **block C** — the masked smooth log floor |
| `stochastic_stage` | `JointConfig` | v3 **regime 3** — the residual's comb-locked energy into its own channel |
| `refit_stage` | `RefitConfig` | the whole telemetry refit as one stage |
| `guarded` | `SeedConfig` | wrap a stage with the blind per-track guard |

The combinators — a stage in, a stage out:

| Combinator | What it does |
|------------|--------------|
| `pipeline(a, b, ...)` | left-to-right composition |
| `iterate(stage, n)` | run `stage` `n` times; `n <= 0` is the identity |
| `windowed(inner, window_s=, hop_s=)` | run `inner` per overlapping window and stitch the banks |
| `guarded(inner)` | run `inner`, then revert the rotors the blind guard vetoes |

The recipes:

| Recipe | Composition |
|--------|-------------|
| `vit2dsp` | blind seed → the calibrated ladder |
| `blind_fullrange` | blind seed (K, R) → coarse full-range init → the ladder |
| `flagship(n_apps)` | `n_apps` × (peel → pi_kalman) |
| `peel_alternation` | `flagship` one application at a time, every frame kept |
| `refit_stage` | presmooth → coarse-to-fine (peel → pi_kalman) to convergence |
| `judge(candidate)` | a candidate stage → `fitness_stage` under one control |
| `joint_solve_window` | floor → (iters − 1) × (solve → split → floor) → solve, the last one reading the MAP objective (`stochastic=True` appends regime 3). `JointConfig.v4` selects the unified model through this same shape — v4 changes the BLOCKS, not the alternation |

Every campaign driver calls a recipe. A script must not assemble a ladder of its own — if a
variant is worth running, it is worth a named recipe here.

## 2. The primitives: `dsp.py`

**`dsp.py`** — THE signal-processing primitives, one each: `zoom_bands` (the zoom-IFFT band-select kernel behind every demodulation), `demod` (the one demodulation driver — carrier synthesis, chunked flush, band select), `boxcar` (the one moving average), plus the selection seam `dsp_config` / `resolve` (device, pad) and the thread knob `thread_pool` / `threads`. Torch only; numpy at every seam. Leaf module — imports numpy and torch, nothing else.

Everything expensive in this package is one of these three calls, which is why one section
of knobs ("Performance knobs" below) covers the whole stack.

## 3. The algorithm cores

### Tracking — turn an init into a trajectory

**`vk_tracking.py`** — Coupled Vold–Kalman order tracker: `VKConfig`, `vk_track`, `vk_envelopes`, `vk_reconstruct`, `demodulate`, `ls_project_envelopes` (per-harmonic per-block least-squares re-fit of the envelopes onto the audio — the peel subtraction that cannot inject energy; ONE core, `_ls_project`, tiled on CPU and clip-long on any other device), plus the schedule helpers (`k_schedule`, `bw_schedule`, `env_stride`, `second_diff`). Every demodulation here is a `dsp.demod` call.

**`phase_increment_tracker.py`** — Phase-increment ML instantaneous-frequency tracker: `pi_kalman_refine`, plus the two named calls of the `dsp` primitives — `zoom_lp_decimate` (`zoom_bands` in this module's parameterization) and `demod_bank` (the one-rotor naming of `demod`; `_demod_bank` stays as an alias for the tests and `scripts/tracking_ref.py`). `pi_kalman_refine(peel_audio=, pair_audio=)` is the PEEL SEAM — per-rotor / per-pair replacement audio for that pass only (what the flagship used to inject by monkeypatching two private functions).

**`warp_refinement.py`** — Iterated time-warp (generalized-demodulation) IF refiner: `iter_warp_refine`. Smooths with `dsp.boxcar`.

**`rps_refinement.py`** — Comb-spectral trajectory refinement: `RefineConfig`, `compute_logmag`, `refine_trajectories`, `refine_coherent`, `estimate_clock_offset`, `comb_confidence`.

### Seeding — find the rotors with no init at all

**`vk_blind_seeding.py`** — Blind seeding of initial trajectories: `SeedConfig`, `blind_seed`, `stage_guard`, `residual_rescan`, plus `logmag_spectrogram` — THE whitened-spectrogram core (per-channel white + raw, optional `n_fft`/`hop_length`) that `whitened_logmag`, `pipelines.whitened_logmag_multi` and the scripts' coarse pass all read through.

**`rotor_dp.py`** — Exact single-rotor Viterbi lattice: `viterbi_path`, `greedy_peel`, `track_masked`.

**`joint_beam_tracker.py`** — Joint 4-rotor beam search over comb emissions: `joint_beam_track`, `build_objective`, `comb_tables`, `comb_scores`.

### Judging and correcting a trajectory

**`fitness.py`** — Goodness of fit of a CANDIDATE trajectory against the audio (issue 17 phase 6a): `FitnessConfig`, `window_cells` (the one demodulation), `score_cells` -> `FitnessScore` (FOUR components — broadband residual, `k^2`-weighted phase-increment mean square, magnitude roughness, and the phase-6d **ridge concentration**: `line_power` / `line_masks`, dB of line density on the carrier over a LOCAL floor, the one component where more is better, the one that does not saturate once the envelope is noise dominated, and the one with its own `admit_ridge` gate — it needs the interferer resolved away from DC and excised from the floor, not a clean band, which is 96 % of DREGON's cells against 6.6 %. It is the phase-7 generator readout promoted, so `scripts/gen_label_sensitivity_eval.py` calls the same function), `Holdout` (held-out harmonics / channels / time blocks as a mask over `(channel, harmonic, block)` cells), `apply_control` (the four §B controls), `bootstrap_scores`, `residual_decompose` (scale/lag + the DREGON tachometer signature) FIXED degrees of freedom: the band, the block grid and the admission gate are pinned to the window's REFERENCE trajectory, so the carrier is the only input that changes. The acoustic components are permutation-invariant by construction — rotor identity is certified by the residual pairing, never by the fit. Driver: `scripts/telemetry_fitness.py`; design + acceptance: `docs/experiments/telemetry-fitness.md`.

**`fitness_vk.py`** — F_VK: the same goodness of fit as a DIFFERENTIABLE objective over trajectories (`docs/trajectory-fitness-design.md` §1 Fact 3, §2). `FVKConfig` / `FVKStage` / `DEFAULT_SCHEDULE`, `fvk_score` (numpy scorer: profiled residual, R², per-harmonic captured energy, config echo), `fvk_loss` (the torch scalar), `solve_envelopes` (THE thin wrapper over `vk_envelopes` — nothing numerical of its own), `alias_charge` (the order/alias counter-term, off by default, reading through `fitness.line_power`), `optimize_trajectory` (L-BFGS). The VK cost is quadratic in the envelopes at a fixed trajectory, so substituting the closed-form envelope solution back leaves a function of the trajectory alone; the VK literature takes the speed as tachometer-given and never optimizes it, which is the gap this fills.

Three things a caller must know:

- **The envelope theorem is the whole trick.** `fvk_loss` solves `a*` with the existing numpy/scipy solver under `no_grad`, then rebuilds the carriers in torch (`phase_k = 2 pi k cumsum(r) / sr`) and evaluates the objective with `a*` DETACHED. Since `a*` is stationary in `a`, that gradient IS the profiled objective's — no autograd through the banded Cholesky. Two details are load-bearing, not cosmetic, because both decide whether `dL/da = 0` really holds: the prior weight is `(stride/2) rho^2`, not `rho^2` (the VK normal equations live on the decimated grid), and the data term carries the solver's own `edge_taper` (extracted into `vk_tracking.edge_taper` so there is one implementation). Measured on a 1 s single-rotor window: the chain rule matches finite differences to 1.7e-5, the profiled objective to 3.2 % of the gradient (0.7 % at 2 s); with either detail wrong it is 39-65 %.
- **Fixed degrees of freedom by construction.** `FVKConfig.vk_config` disables the VK validity mask (`f_min = 0`, `f_max = inf`, `min_rps = 0`) — the mask is the one part of the solver that would react to the candidate — and the harmonic set is capped from a pinned REFERENCE trajectory instead (`k_cap`). Every candidate is scored on the identical `(channel, rotor, harmonic)` cells; the score reports `n_cells`.
- **The smoothness weight is not portable, so ask for `"auto"`.** `optimize_trajectory(smooth_lambda=1.0)` is calibrated for a CRUISE window (log-domain prior ~0.8 against a data term normalized to order 1). A takeoff ramp reads 244 and the window then cannot move at all. `smooth_lambda="auto"` calls `auto_smooth_lambda`, which measures the prior of the init itself and holds it to half the data term; the weight and the init prior come back in the diagnostics (`smooth_lambda` / `prior_init`). Any driver that sees whole recordings wants it.
- **The basin knob is `bw_rps`, not `k_max`.** The `1/(K T)` law is for a coherent harmonic sum; here every harmonic has its own VK envelope with a `k`-scaled band, so the capture radius is `bw_rps / 2` rev/s at EVERY harmonic and the gradient at a 0.5 rev/s error still points at truth at `k_max` = 80. What `k_max` moves is the depth and the curvature of the well (objective at truth 0.587 -> 0.073 from `k_max` 5 to 80, neighbours barely moving), which is the precision half of the same law and why the schedule still starts coarse. Open the band to a non-capture 2.0 rev/s and `k_max` = 80 does break into 7 local minima inside ±1 rev/s.

**`decompose.py`** and **`joint_decompose.py`** — the WINDOWED decomposition, v2 and v3. They
have their own catalog below (§3.1), because they are the one part of this package a reader is
expected to REBUILD rather than call.

**`telemetry_refit.py`** — The FITTER phase 6a's harness was built to judge (issue 17 phase 6b): `RefitConfig`, `refit_window` -> `RefitResult`, `presmooth` (THE 5 Hz detrended low pass of the campaign — the 6a driver's `lp:` candidate calls it), `k_cap_for_error` / `advance_k` (the coarse-to-fine ladder, from the phase-wrap capture rule `k <= wrap_guard_rad fs_env / (2 pi e)` — never a flat `k_caps`), `scale_summary` (the well-posed scale, per-rotor mean shift plus one joint global LS scale), and `order_and_gaps` (THE identity test). One outer iteration IS one `top.pi_kalman_arm_stage` application, so the LS peel, the peel seam and the twin rule are the flagship's, wired not rebuilt. Driver: `scripts/telemetry_refit.py`; design + acceptance: `docs/experiments/telemetry-fitness.md` § "The fitter".

**`comb_displacement.py`** — Where the acoustic comb sits relative to a rotor-speed CARRIER, with the nulls that make the answer meaningful: `DisplacementConfig` (band / search window / gate geometry), `demod_comb_bank` (integer or half-integer carrier), `ridge_from_envelope`, `profile_prominence`, `pulse_pair`, `carrier_collision_mask` (the twin rule re-derived against the TRUE rotor lines, so an arbitrary carrier can be gated), `nearest_interloper_hz` (the same geometry as a DISTANCE, so a caller with its own band gets its own collision rule and can grade a contested harmonic by how close the interferer is), `weighted_stats`, `combine_k`, `measure_variant`. The carrier is one trajectory row, never "rotor i of the array" — that is what makes an off-comb, mismatched or fitted carrier expressible. Driver: `scripts/displacement/nullcontrol.py`.

**`order_domain.py`** — The order domain: resample the audio uniformly in rotor PHASE, then FFT. `order_spectrum`, `comb_scan` (a whole comb scored at once over a scale grid, `half=True` for the half-integer null), `segment_comb_scan` (the same on short segments, which is the only reading that survives the sub-second high-k coherence time), `scan_summary`, `peak_orders`. No peak-search window anywhere — this is the estimator built to be immune to the other one's failure mode. Driver: `scripts/displacement/combscan.py`. Also the machinery issue #16 Tier 2 wants for removing the `K` factor from the demod cost.

**`phase_noise.py`** — WP18 rank-one-plus-diagonal covariance of the per-harmonic rate opinions: `Arm`, `demod_rotor`, `arm_covariance`, `fit_rank_one`, `channel_coherence`, plus `brickwall` — THE whole-window FFT filter of the package, which `fitness` and `telemetry_refit.presmooth` both read through. Measures the harmonic-common jitter term `sigma_J^2` against the per-harmonic terms `v_k` — the evidence behind the `VKConfig.freq_weight` shape. Its data side (recordings + window selection) is injected by the caller — the WP18 campaign's own window builder was retired with the campaign, so a new caller must supply its own windows. Record: `docs/experiments/rps-refine-precision.md`.

### Constants

**`rotors.py`** — Quadrotor control-allocation constants: `MIXER`, `NUM_ROTORS`, `MODE_NAMES`, `modes_from_rps`, `rps_from_modes`. `data_processing.rps_synthesis` re-exports them.

## 3.1 The decomposition: the primitive inventory

This is the catalog of `decompose.py` + `joint_decompose.py` + their stages. Every entry is
importable as `tracking.<name>`. Read it beside the model it implements:

```
y_c(t) = sum_{r,k} Re[ g_{r,k,c}(t) e^{j(k phi_r(t) + k theta_r(t) + psi_{r,k}(t))} ] + n_c(t)
```

`phi` is the annotated shaft phase, `theta` a slow coherent shaft correction (rig-common plus a
small per-rotor part), `psi` a slow per-track phase correction, `g` the residual envelope — which
then needs AMPLITUDE bandwidth only — and `n` colored noise with a smooth log spectrum `S`. The
sum is EXACT by construction, because the residual is DEFINED as the unexplained part. Only the
SPLIT is estimated, and it is the MAP estimate under the VK prior. Design and measured results:
`docs/vk-decompose-v3-design.md`; v1/v2 record: `docs/experiments/vk-decomposition.md`.

**The state.** `JointState` is what the alternation accumulates and the one seam the stages pass
along, `meta["joint"]`: the carrier, `theta`, `psi`, the floor model `psd`, the last solve's
`env` / `x_eff` / `residual` / `track_energy` / `n_solves`, and — once regime 3 has run — the
`stochastic` channel beside the residual. It is frozen — every block returns a
NEW state. The carrier is HELD and never re-derived, because the alternation is conditioned on
one carrier for the whole window and `theta` is a correction on top of it.

### Setting up a solve

| Primitive | Purpose |
|---|---|
| `solve_config(k_max, *, sr, mics, bw_rps=1.0, f_max=6000.0) -> FVKConfig` | THE measurement geometry — one construction, so every solve agrees |
| `k_cap(cfg, reference) -> int` | the harmonic set, from the RECORDING's reference trajectory (never the window's) |
| `solve_audio(audio, cfg, mics=None) -> (C, T)` | the one channel-selection rule, float64 and contiguous |
| `to_audio_grid(r, frame_times, n_t, sr) -> (R, T)` | the trajectory on the audio grid — THE carrier every solve is built from |
| `shaft_phase(r_audio, sr) -> (R, T)` | `2 pi cumsum(r) / sr`, the fundamental phase `phi` |
| `BandwidthSchedule(bw0_hz, slope_hz_per_k, cap_frac_of_sep, bw_abs_max)` | the v2 linewidth-matched band law, with its CLI spelling (`.parse` / `.text`) |
| `base_bandwidths(r_audio, k_hi, cfg) -> (M,)` | the band the solver would use with no schedule — the reference the gain is taken against |
| `line_separations(r_audio, rotor, k) -> (M,)` | Hz from each track's line to the nearest other line |
| `track_rho2_gain(r_audio, k_hi, cfg, sched, rho_scale=1.0) -> (M,) \| None` | THE bandwidth law in the solver's own currency; both solve paths read it |
| `group_plan(r_audio, k_hi, cfg) -> dict` | THE memory model — coupling is transitive, `~1e-4 k_hi^2 window_s` GB per worker. Read it before sizing a job |

### Block A — the whitened solve

| Primitive | Purpose |
|---|---|
| `vk_envelopes(audio, r, vk, *, k_hi=, rho2_gain=, phase_offset=, env_rotation=, data_weight=)` | THE solver. The last three arguments are the whole joint seam, and all three `None` is the v2 arithmetic bit for bit |
| `whiten_weights(psd, k, rotor, r_env, t_env, *, clamp_db=15.0) -> (M, J)` | `1 / sqrt(S(k r(t), t))` — the Whittle weighting, collapsed to one scalar per track and frame |
| `solve_block(state, audio) -> JointState` | block A: weights → hooks → solve → `x_eff = x e^{j psi}` → reconstruct → residual |
| `vk_solve_stage(cfg=None, *, profile=None)` | its Frame adapter |
| `solve_window(audio, r_audio, cfg, *, k_hi, mics=, rho_scale=, bw_schedule=) -> Envelopes` | the v2 solve — block A with no corrections and no whitening |
| `reconstruct(x, k, rotor, phase, stride) -> (recon, track_energy)` | the bank back to audio against a GLOBAL phase, chunked in time |

### Block B — the phase split

| Primitive | Purpose |
|---|---|
| `split_phases(x, k, rotor, valid, fs_env, *, k_trust, ...) -> PhaseSplit` | the `k`-weighted shaft estimate (rig-common, then per rotor), then a per-track `psi` |
| `split_block(state) -> (JointState, PhaseSplit)` | block B: the split folded into the accumulated corrections |
| `phase_split_stage(cfg=None)` | its Frame adapter |
| `wh_smooth(y, lam, weight=None) -> ndarray` | THE Whittaker-Henderson smoother — a one-dimensional VK, one banded solve |
| `wh_lambda(bw_hz, fs) -> float` | its weight from a bandwidth, through the solver's OWN Tuma relation |
| `bw_psi_hz(k, slope=0.6, cap=8.0, floor=1.5)` | the per-track correction band, `clip(slope k, floor, cap)` |
| `theta_rate(theta, fs_env) -> ndarray` | `d theta / dt / 2 pi` in rev/s — the GAUGE-FREE form, the only one that crosses a window |
| `upsample_env(vals, n_out, stride)` | the envelope grid back to audio rate, tail held |

### Block C — the floor

| Primitive | Purpose |
|---|---|
| `masked_smooth_psd(audio, sr, r_audio, k_hi, ...) -> SmoothPSD` | Welch with every predicted line masked per frame, then a moving median and a cepstral lift |
| `floor_block(state, audio) -> JointState` | block C: the floor of what is left — the audio before the first solve, the residual after |
| `floor_stage(cfg=None)` | its Frame adapter |
| `SmoothPSD.pooled() -> (B, F)` | the geometric mean over microphones — what the whitening weight reads |
| `stft_power(audio, starts, n_fft, frames_per_chunk=64)` | THE framed Hann power spectrogram of this module, shared with the order-cell probe |
| `frame_starts(n_t, n_fft, hop)` | its frame grid |

### Regime 3 — the stochastic comb channel

Regimes 1 and 2 are the coherent envelope at the annotated carrier and at the CORRECTED one.
Both need a band narrow against the local line spacing, because two coherent envelopes whose
passbands overlap are not identifiable — a cluster run at a cap of 1.5x the line separation went
singular (SuperLU "Not enough memory to perform factorization", `r2 = -1`). Identifiability caps
a coherent band at about 0.4x the local spacing, and above about `k` 10 the measured linewidth
`0.6 k` Hz is wider than that, so the flanks of every line are comb-locked energy that no
coherent envelope can carry. Regime 3 carries it, as a POWER split with no phase model.

| Primitive | Purpose |
|---|---|
| `stochastic_split(residual, sr, r_audio, k_hi, *, psd=None, n_fft=4096, ...) -> StochasticSplit` | THE split: a PER-BIN amplitude gain `a = clip(sqrt(S / P~), 0, 1)` inside the UNION of the comb search regions, `a = 1` outside. Broadband `= a Y`, stochastic `= (1 - a) Y` |
| `comb_lines(rate, k_hi) -> (lines Hz, k)` | one frame's whole comb, `k` beside it because the band law is written in `k` |
| `line_half_widths(lines, k, *, slope_hz_per_k=0.6, min_half_hz=0)` | the COHERENT law: `min(0.6 k, local spacing)`, floored at one bin — the spacing cap is what stops one band reaching over its neighbour |
| `stochastic_half_widths(k, *, width_factor=2.0, min_half_hz=0)` | regime 3's OWN law: `2 x 0.6 k` Hz, NO spacing cap |
| `stochastic_block(state) -> (JointState, StochasticSplit)` | the block, on the state the last block-A solve left |
| `stochastic_stage(cfg=None)` | its Frame adapter (`top.py`) |
| `_wola_plan(n_t, n_fft, hop)` | the padding and frame grid that make the overlap-add an EXACT identity at any window and any hop |

Six things a caller must know:

- **The gain is per BIN and it is an AMPLITUDE.** Both halves are the fix for a measured failure
  of the flat per-band Wiener gain this replaced (full-scale DREGON, `results/vk_decompose_v3c`).
  A power gain `S / P` is the conditional mean, which is not a typical floor realization: its
  power `S^2 / P` is DENTED below the floor at every strong line, and the acceptance gate — the
  order-cell excess of the BROADBAND channel — cannot tell a dent from a line (k1-9 went UP,
  4.3 % -> 7.8 % retained, the profile peak moving to +/- 0.5 orders). `sqrt(S / P~)` leaves the
  broadband channel at power `S` in expectation instead. And ONE gain over a union that spans many
  lines scales the region uniformly, so the comb PATTERN survives at reduced amplitude (depth
  0.386 -> 0.380 dB at k10-24); the smoothed periodogram carries the line SHAPE, so a per-bin gain
  concentrates the removal at the line cores with no line model at all.
- **The floor is a STEP in time, and that is what the seams are.** `S` is one spectrum per block
  (`~4 s`), so the gain steps at every block boundary — and because much of the comb band sits
  within ~1 dB of the floor, the clip at `a = 1` toggles whole bands between "nothing taken" and
  "something taken" across one boundary. Measured: the rectangular on/off patches of the FLY124
  demo spectrograms have vertical edges at 59.5 s and 63.4 s, which are boundaries 15 and 16 of
  that run's 3.96 s block grid, exactly. `floor_time_interp=True`
  (`JointConfig.stochastic_floor_interp`, `scripts/vk_decompose.py --stochastic-floor-interp`)
  reads `log S` per FRAME instead, linearly interpolated between the block centers
  `SmoothPSD.t_block`. It is OFF by default and the default path is bitwise unchanged — the block
  floor is what every published number was produced with.
- **The two smoothing widths are fixed, and the number that fixes them is chi-square variance.**
  `P_SMOOTH_FRAMES = 5`, `P_SMOOTH_BINS = 3`: one periodogram bin has 100 % relative standard
  deviation, 15 averaged bins bring the power estimate to ~26 % and the amplitude gain, its square
  root, to ~13 %. They are NOT knobs. Measured alternative: dropping the frequency boxcar
  (`bins = 1`) moves DREGON's retained numbers to 6.75 / 8.88 / 15.65 / 7.35 % — better in three
  bands, worse in the one (k1-9) the estimator is weakest in — so the shipped pair stands.
- **The bands are UNIONED, and that is not cosmetic.** Two lines whose bands touch — at
  `2 x 0.6 k` Hz nearly all of them do above `k` 35 — become ONE region, so their shared energy is
  never taken twice. `n_bands_per_frame` in the diagnostics is the count after the union. The
  union now only delimits WHERE the gain may differ from one; inside it the gain is per bin, so a
  bin that sits at floor level passes through whether or not a region claims it.
- **The broadband channel is a SUBTRACTION.** `residual - stochastic`, exactly as the residual
  itself is `audio - recon`, so `coherent + stochastic + broadband = original` holds to float
  roundoff and no consumer has to trust an overlap-add. The state's `residual` is never
  rewritten; `JointState.stochastic` lands beside it.
- **`P` and `S` are both power spectral DENSITIES on `masked_smooth_psd`'s own normalization**
  (`1 / (sr * sum(w^2))`), so the ratio is scale free whatever `n_fft` the split runs at.
- **The floor is per WINDOW on the state and per RECORDING in the driver.** `stochastic_block`
  scores against the alternation's own floor; `stochastic_split(psd=None)` fits a fresh one with
  the same block C, which is what `scripts/vk_decompose.py --stochastic` does on the STITCHED
  residual, because one window's floor does not describe a minute.
- **Do NOT widen block C's comb mask to feed this.** The search regions blanket 91-94 % of
  1-4 kHz on DREGON, so it is tempting to give the floor fit a wider mask and let the cepstral lift
  bridge. Measured: the shipped mask (3.0, 0.45) leaves the between-region residual within
  +0.84 / +0.11 / +0.03 dB of `S` at 1-2 / 2-4 / 4-8 kHz and -0.04 dB above the comb, so the
  extrapolation is already sane; widening it to (5.0, 0.48) biases `S` HIGH by 1.9 dB at 1-2 kHz
  (`band_floor_share` 1.16, the region seeing no excess to remove at all) and costs the gate
  10.25 / 20.03 / 10.85 % against 9.61 / 16.98 / 8.04 %.

**What limits it now.** On the synthetic fixture the per-bin gain is exact where it is applied —
`a^2 |Y|^2` measured on the transform lands -0.14 dB from `S` — while the RESYNTHESIZED broadband
reads +3.3 dB against the same floor. The gap is the analysis-modify-synthesis error of the
weighted overlap-add, not the gain: a 20 dB deep, one-bin-wide notch has an impulse response as
long as the frame, and it wraps. More overlap does not help (hop `n_fft / 8` reads +3.25 dB), a
zero-padded analysis window is worse (+9.3 dB), and smoothing the GAIN in frequency fills the
notch it is supposed to cut (+10 dB at three bins). It matters least where the method is used:
DREGON's residual sits 0.5 to 1.7 dB over its floor inside the regions, not 20 dB.

### The window layer

| Primitive | Purpose |
|---|---|
| `frame_grid(n_t, sr, hop_s)` | the recording's own frame grid |
| `interp_rps(vals, stamps, ft)` | telemetry onto that grid, in float64 |
| `window_bounds(n_frames, window_s, hop_s, hop_frame_s)` | the window tiling, last window right-aligned, every frame covered |
| `window_span(ft, i0, i1, n_t, stride, sr, hop_frame_s)` | one window's sample range, SNAPPED to the envelope stride |
| `window_geometry(sr, window_s, hop_s, fs_env=100.0) -> (stride, ramp)` | the envelope stride and the cross-fade length |
| `fade_weights(n_win, ramp)` | the linear cross-fade, floored so a singly-covered frame still resolves |
| `stitch_bank(windows, phi, stride, ramp) -> dict` | the phase re-reference `exp(-j k Phi(a0-1))` plus the cross-fade |
| `stitch_windows(windows, phi, stride, ramp, *, r_audio=, sr=) -> dict` | THE stitch: `stitch_bank` for v2, and for v3 the rate stitch that puts windows carrying their own `theta` on ONE carrier |
| `global_rate_correction(windows, stride, a_min, a_max, ramp)` | the per-window `dr` cross-faded onto one global envelope grid |
| `corrected_phase(r_audio, dr_env, sr, stride, a_min, a_max)` | `r + dr` and its integral — the carrier the stitched bank belongs to |
| `window_extra_phase(theta_w, phi_hat, phi_tilde, a0, stride, n_env_w)` | the rotation that moves one window onto that carrier |
| `phase_reference_deviation(r_audio, phi, a0, sr)` | the stitch's assumption MEASURED, not assumed |
| `windowed(inner, *, window_s, hop_s)` | the combinator: tile, run `inner`, stitch |

### The readings, and the instruments that judge a decomposition

| Primitive | Purpose |
|---|---|
| `map_objective(residual, sr, psd, *, x, k, bw_track, theta, psi, fs_env, ..., logdet_posterior=None, h_carrier=None)` | THE converged MAP objective, term by term: `data` + `rent` (the Whittle pair on the floor block's own STFT grid) + `phase_priors` + `envelope_prior`, plus `n_cells`. A pure OBSERVER — switching it on cannot move a product. The weights are the ones the run used: `wh_lambda` for the two phase priors, and `rho^2` read back out of `Envelopes.bw_track` through the solver's own Tuma relation. `logdet_posterior` adds the MARGINAL readout, `h_carrier` the H-AWARE one (both below) |
| `prior_logdet(bw_track, n_env, fs_env, p=2)` | `log det'` of the improper envelope prior `blkdiag(rho_m^2 D2^T D2)` — the pseudo-determinant, because `D2` kills a constant and a ramp per track |
| `d2_pseudo_logdet(n_env)` | its length-only part, one banded Cholesky of the pentadiagonal `D2 D2^T` and cached — `O(n)`, not an eigendecomposition |
| `joint_objective(state)` | the same, with every argument taken off a `JointState` (the last solve's residual against the last floor) |
| `energy_ledger(audio, recon, track_energy, k)` | total / tracks / residual / CROSS TERM — the tracks are not orthogonal, and the cross term is the honest statement of that |
| `phase_model_report(...)` | per-rotor drift against `k` plus `rank_one_share` — shaft jitter against per-harmonic drift |
| `solve_report(state, audio, *, profile)` | the reading of one block-A solve: the shares, the flatness, the order cell |
| `order_cell_profile(audio, sr, r_audio, ...) -> dict` | THE probe: the spectrum re-expressed in ORDERS, folded cell by cell. Read `excess_db` (absolute comb power left, comparable across signals) BEFORE `depth_db` (a ratio, which can rise as the residual falls toward the floor) |
| `order_cell_bands(audio, sr, r_audio, **kw)` | the same table without the plot arrays — what a report carries |
| `cell_profile(profile, grid, lo, hi, order_step, ...)` | one band's fold, with the in-cell trend removal that killed a published half-order verdict |
| `whitened_flatness(residual, sr, psd)` | flatness of `|N|^2 / S` beside flatness of `|N|^2` — a correct floor leaves a flat residual |
| `residual_tones(residual, sr, r_audio, ...)` | the NON-comb tonal peaks left, with their distance to the nearest rotor order. Measurement only |
| `welch_psd(audio, sr, nperseg=4096)` | THE Welch of this module |

### The MARGINAL objective — what profiling does not charge for

`J` as shipped is PROFILED: the envelopes' best value is substituted back, which pays no rent
for their freedom. So a hypothesis whose bands cover more of the spectrum can win by ABSORPTION
alone — measured on 5 frozen windows (`results/joint_rescore/`), the profiled `J` ranks
adversarial coverage fans above the telemetry on 3 of them. `JointConfig.marginal` switches on
the exact Gaussian correction that charges for it:

```
J_marg = J + 0.5 * n_channels * (n_fft / hop) * (log det M - log det' R)
```

- `M` is the whitened banded posterior precision block A already factorized. `Envelopes.logdet`
  carries it — read off the Cholesky diagonal inside `_solve_group_banded` (and off SuperLU's
  `U` on the fallback path), so the readout costs no second factorization. It is ONE channel's
  worth, because every channel is a right-hand side against that same system.
- `R` is IMPROPER — `D2` kills a constant and a ramp, two null directions per track — so it is
  the PSEUDO-determinant `prior_logdet`: `(T - 2) sum_m log rho_m^2 + M log det'(D2^T D2)`.
- The `n_fft / hop` factor is not a knob. `data` + `rent` are summed over frames that overlap,
  so they are that many times ONE likelihood while the correction is one likelihood's worth.
  Without it the correction is out-scaled two to one on the shipped grid and the Occam property
  fails; it is reported as `marginal_redundancy` so a caller can divide it back out.

Four hypothesis-INDEPENDENT constants are dropped, so the readout is valid only for comparing
hypotheses on ONE window with the SAME cells and the SAME track count — which is exactly what
`scripts/joint_rescore.py` pins (`k_hi` comes from the telemetry, never from the candidate):
the Gaussian volume factors; the improper prior's null-space volume (two directions per track);
the real-versus-complex parameterization factor (a complex Gaussian carries 1 rather than 0.5,
which scales the correction and cannot flip its sign); and the solver's own scaling convention
(the data term enters as `w` with a right-hand side of `2 w z`) plus the `1e-8` ridge and any
`diag_scale` PD repair that live inside `M`.

The acceptance property is Occam's, and it is a test (`tests/tracking/test_marginal_objective.py`):
add a whole spurious rotor whose every line sits on pure floor, and the PROFILED objective
improves while the MARGINAL one gets worse.

### The H-AWARE data term — the stochastic comb, in the likelihood

Absorption is only half of what lets a coverage fan win. The other half is in the DATA term:
regime 3 exists because no coherent envelope can carry the `0.6 k` Hz flanks of a line, so `J`
charges EVERY hypothesis for that flank energy alike and the true trajectory has no advantage
over a fan that misses the humps entirely. Measured on the same five frozen windows, the exact
envelope marginalization (`--marginal`) moved no ranking at all. `JointConfig.h_aware` puts the
stochastic comb where it belongs, in the noise model:

```
data_h = sum_{c,f,t} [ P / (S + H) + log(S + H) - log S ] ,   total_h = total - data + data_h
H(f, frame) = max(0, P~(f, frame) - S(f))   inside the hypothesis's own comb SEARCH REGIONS
H(f, frame) = 0                             everywhere else
```

- The REGIONS are the hypothesis's only degree of freedom: per frame of the objective's own STFT
  grid, `k r_r(t)` for every `k` the pinned track set names, half widths from
  `stochastic_half_widths` (the 3.0-linewidth law, floored at one bin), unioned by `_line_mask`.
  Regime 3's law, not the coherent one — same code, so the two readouts cannot drift apart.
- `H` inside them is the PROFILED nuisance, bounded by the estimator the split already uses:
  `P~` is the measured power on the same grid, smoothed with the split's own fixed boxcar
  (`P_SMOOTH_FRAMES` x `P_SMOOTH_BINS`, edge mode nearest). The numerator `P` stays the
  UNSMOOTHED measured power — the smoothing belongs to the nuisance, not to the likelihood.
- `data_h` folds the `log(S + H) - log S` half in, so `rent` keeps its meaning, no logarithm is
  counted twice, and `total_h = total - data + data_h` is exact. At `H = 0` it IS `data`.

The asymmetry is honest and it is the mechanism: `H` is fitted from the same data it explains,
so inside a hypothesis's regions the term is nearly hypothesis independent (at floor level `H`
is zero up to the small positive bias of clipping a noisy `P~ - S`). The discrimination is where
a real hump exists and a hypothesis's regions MISS it — those cells pay `P / S + log S` in full.
A fan that opens regions on empty floor buys nothing; a trajectory whose regions sit on the humps
stops paying for them.

Acceptance is `tests/tracking/test_h_aware_objective.py`, on the regime-3 fixture at a PINNED
floor: the true rates take `data_h` to 0.19-0.23 of `data`, rates shifted by 5 rev/s (regions
disjoint from the lines, measured) leave it at 0.78-0.97, and the profiled total cannot separate
the two AT ALL. On pure floor the term charges under 2 % whatever the carrier is.

### v4 — the unified model (`JointConfig.v4`, default off)

Everything above this line is a correction bolted onto a measure whose noise model does not
contain the comb. v4 puts the comb IN the model — one Gaussian process whose power spectral
density is a smooth floor plus a comb of Lorentzians riding the trajectories:

```
M_c(f, t) = S_c(f, t) + sum_{i,k} H_{c,i,k}(t) L_{gamma_k}(f - k r_i(t)),  gamma_k = max(0.6 k, one bin)
```

Design and gates: `docs/v4-unified-model-design.md`. One switch selects it, because its four
parts only work together, and off is the v3 arm call for call.

| Primitive | Purpose |
|---|---|
| `fit_floor_powers(audio, sr, r_audio, k_hi, ...) -> (SmoothPSD, HPowers)` | **F1**: `S` and the line powers `H`, fitted JOINTLY with NO mask on the ORIGINAL signal, per (microphone, time block) |
| `HPowers` | the fitted powers: `(C, B, L)` peak power spectral density on `masked_smooth_psd`'s own units, plus the line table. `.pooled()` (arithmetic mean over microphones), `.block_of(t)` |
| `floor_lambda(b_f_hz, sr, n_fft)` / `floor_penalty(psd, b_f_hz)` | the floor's smoothness weight from a LENGTH SCALE in hertz, and the penalty term itself |
| `whittle_floor_objective(p, hump, g, lam)` | the F1 cost of one cell — what GUARDS every step and chooses between the starts |
| `v4_rho2_gain(r_audio, k_hi, cfg, ...)` | **F2a**: the band law `max(b0, 0.6 k)` Hz with NO spacing cap, in the solver's own currency |
| `v4_ridge(psd, hp, k, rotor, r_env, t_env, weight, c0=)` | **F2b**: the amplitude prior `beta = c0 S / H`, fed to `vk_envelopes(ridge=)` |
| `map_objective(..., v4_powers=, v4_carrier=)` | **J_v4**: `sum [P/M + log M]` + the two phase priors + the floor penalty. No envelope term, no separate rent |
| `joint_objective(state, audio)` | the same off a state — and under v4 the `audio` is REQUIRED |

Five things a caller must know, and the first two are the ones that will bite:

- **`J_v4` scores the ORIGINAL signal, not the residual.** The line processes are integrated out
  rather than conditioned on, so the thing the model describes is the audio; scoring the residual
  would count the comb twice, once by subtracting it and once by modelling it. `joint_objective`
  raises rather than guess. Its column is therefore NOT comparable with `total` — different model,
  different signal — which is why `scripts/joint_rescore.py --v4` ranks on it alone.
- **The `(S, H)` alternation is BISTABLE, and the two starts are a guard and not a knob.** Where
  the comb blankets a band, "floor on the blanket with `H = 0`" and "lines claim the blanket" are
  both honest stationary points; from the masked warm start the first H-step finds no excess and
  nothing ever moves, so the v3 failure survives INSIDE the v4 fit. The objective ranks them
  correctly, so the fit screens `FLOOR_START_DB = (0, -12)` dB on `FLOOR_SCREEN_ROUNDS` and
  refines the winner. Measured on the dense fixture the two differ by 13 dB of fitted floor.
- **The band law changes `rho^2` and NOTHING else**, so the coupling partition, `group_plan` and
  the banded memory are identical to v3's — measured on the smoke window, 0.149 GB either way.
  What moves is wall time: 1.42 s to 2.53 s there, the difference being the three `(S, H)` fits.
- **Regime 3 does not run.** The comb channel already carries the line flanks, so the
  decomposition is two channels and a subtraction, and `joint_solve_window(stochastic=True)` is
  REFUSED under v4 rather than silently ignored.
- **`H` is a first-class product**, not a diagnostic: it is the generator's amplitude targets by
  construction, and `scripts/vk_decompose.py --v4` writes it into the unit `.npz`
  (`h_rotor` / `h_k` / `h_t` / `h_lines` / `h_half` / `h_power`).
- **A v4 group that will not factorize FAILS — it never falls into `splu`.** The automatic
  fallback is right for a v2-sized group and is the OOM bomb for a v4 one (SuperLU's fill-in on
  300+ tracks does not fit in memory; the measured field failure was a spin-up window that took
  its worker and then the pool with it). Under `ridge` the banded path keeps its two `diag_scale`
  repair retries and then raises a `MemoryError` naming the group, the harmonics and the knob, so
  `gridrun` writes one `.err` and the other units keep running. An EXPLICIT `solver="splu"` is a
  different statement and is still honoured.

**Conditioning, and the limit of the fix.** With the bands uncapped, two rotors whose lines nearly
coincide have passbands that nearly coincide, so the difference direction of that pair has almost
no data curvature; `rho^2` is small because the band is wide, and `beta = c0 S / H` is small for
exactly the STRONG lines — so the loudest comb in a window is the one that breaks the
factorization. `RIDGE_FLOOR_FRAC = 0.03` holds the prior above 3 % of the group's own mean data
curvature, and the constant is squeezed from both sides: below it nothing factorizes on the
four-rotor spin-up fixture (120 tracks, rotors fanning 2 rev/s and crossing), and above it the
estimator starts to move — at 0.3 the Wiener calibration breaks (0.72-0.76, outside ±20 %) and a
strong line keeps 0.61 of its power. At the chosen value the calibration is unmoved to three
decimals and a strong line keeps 0.945 instead of 0.980.

The deficiency is NOT float rounding — it is the decimated cross term's own approximation error,
which is percent-relative — so the floor has a reach and past it there is nothing to do. Four
rotors within ~1 rev/s would need 0.1, costing a strong line 17 %; those windows are genuinely
unidentifiable at these bands (four combs 0.3 Hz apart at `k` 1 are not four combs to a 3-second
window). `tests/tracking/test_v4_conditioning.py` pins both halves: the floor factorizes a group
that fails without it, and past its reach the failure is clean and names its own mechanism.

**Past that reach, the band law steps aside — `JointConfig.v4_band_law`.** DREGON is a twin rig
(its pairs sit 0.43 and 0.81 rev/s apart), so at `k_hi` 83 six of its seven windows are in the
unidentifiable regime and fail cleanly. `scripts/vk_decompose.py` catches that one exception per
window and retries ONCE with `v4_band_law=False`: the envelope bands come from the `--bw-schedule`
(the v3 spacing-capped law) and **everything else stays v4** — the joint `(S, H)` fit, the
amplitude prior with its floor, `J_v4`. That is not a degradation of what the model estimates.
The amplitude targets ARE `(S, H)`, they come from the F1 fit, and that fit never looks at a band;
the uncapped bands refine the WAVEFORM channel, which is only identifiable where the rotor spreads
allow it. The row carries `v4_band_fallback: true` and the report's `v4` block carries
`n_band_fallback` / `band_law_mixed`.

Mixed sets stitch: the stitch's only compatibility check is the HARMONIC SET, and `k_hi` comes
from the recording's reference trajectory rather than from any window's bands, so windows solved
under different band laws stitch exactly as windows that were not. `bw_track` is carried from the
first window and never enters the arithmetic — which is why `band_law_mixed` has to be reported,
because `bw_track_hz_by_band` is then one window's law and not the recording's.

**The fallback derives its own bands — `v4_fallback_rho2_gain`, not `track_rho2_gain`.** This is
the one place where reusing the v2 seam is wrong, and it cost a second field failure (FLY124,
`k_hi` 83: 13 of 21 units died in the objective READOUT with `bw_hz=100.0 exceeds fs_env=100.0`).
The chain, all four links measured:

1. `bw_rps` is 1.0, so the solver's own band for track `k` is `min(k, 0.9 fs_env)` capped by the
   group's minimum line separation. On DREGON everything is in one dense group and the cap holds
   every track near 1 Hz; on a rig whose rotors are tens of hertz apart an isolated high-`k` track
   is never capped, and its band is 60 to 90 Hz on a **100 Hz** envelope grid.
2. `schedule_bandwidths` FLOORS at that band — "a schedule never NARROWS a track below v1" is
   right for v2 and wrong here — so a schedule asking for 3 Hz achieves 30 to 90.
3. `bandwidth_neutral` multiplies `rho^2` by the track's own `mean(u^2)`, which is BELOW one for a
   loud track (the weight is normalized over all cells, not per track, and the clamp bottoms at
   `10^-1.5`), so it WIDENS those bands further.
4. `_tuma_bw` then saturates at `fs_env` exactly, and `_tuma_rho` refuses that value.

So the fallback asks for the same schedule with the floor at the smallest numerically usable band
instead of the solver's, and `JointState` keeps `bw_schedule` / `rho_scale` so the retry can
re-derive rather than reuse a compiled gain. `solve_block` additionally holds the v4 arm's
achieved band inside `0.9 fs_env` after the neutral gain, and `map_objective` reads a saturated
band at the widest CONVERTIBLE one — an observer must not be the thing that raises. Measured on
the FLY fixture at a schedule absmax of 3 Hz: the solver's clamp and the v2 gain both give 30.0 Hz,
the fallback law gives 3.0, and with no schedule at all it gives 1.0.

The two calibrated constants, both frozen with their measurement in the test that made it:

- `FLOOR_LENGTH_HZ = 600` — the floor's smoothness length in hertz. On the dense fixture (one
  comb whose density `gamma / Delta` runs 0.06 to 0.48) the fitted floor lands at
  0.36 / 0.53 / 0.70 / 0.46 dB rms in the four bands, against 0.37 / 0.63 / 1.03 / 0.73 at 400 Hz
  (the v3 cepstral lift's own scale) and 0.33 / 0.61 / 0.49 / 0.31 at 800 Hz.
- `V4_RIDGE_C0 = 1` — and it is 1 for a derivable reason, not a fitted one: the band is `0.6 k` Hz
  and the line's own half width is the same `0.6 k`, so line and noise contribute in proportion to
  the SAME noise-equivalent bandwidth and the ratio of the two powers the band admits is exactly
  the ratio of the two densities, `S / H`. Measured against the Wiener target over three seeds:
  0.97 / 1.02 / 1.01 at `k` 5 / 20 / 60 for a line 10 dB over the floor.

Acceptance (gate 1 of the design, plus the carve):
`tests/tracking/test_v4_floor.py` reads the fitted floor at 0.33 / 0.45 / 0.58 / 0.40 dB rms
against the masked fit's 2.00 / 4.83 / 8.12 / 10.46 on the same fixture — and against 0.33 / 2.05
/ 6.69 / 5.75 for the v4 fit given only the warm start, which is the bistability measured.
`tests/tracking/test_v4_carve.py` gives one rotor a band it owns with lines on the low harmonics
only: the line-free tracks take 3.9 % of what the same wide bands take with the prior switched
off, while the owned harmonics keep 84 %. `tests/tracking/test_v4_ridge.py` holds the `c0`
calibration; `tests/tracking/test_v4_objective.py` holds `J_v4`'s discrimination.

The bar on the floor gate is 0.9 dB rms and not the design's 0.5, for a measured reason that is
in the fit's favour: the Hann main lobe smears each line's skirts into the bins the floor is read
from, so a CORRECT fit reads a few tenths of a decibel high on this grid whatever it does. The
residual error is that bias and almost nothing else.

### Three things that will bite a caller

- **The annealing ladder starts at `k` 3**, and the reason is the ENVELOPE BAND, not the phase
  unwrap: harmonic `k` of a shaft wandering `sigma_r` rev/s is a modulation of bandwidth about
  `k sigma_r` Hz, and a `B` Hz band distorts its phase once `k sigma_r > B / 2`.
- **Whitening is bandwidth-neutral by default** (`JointConfig.bandwidth_neutral`). Without it a
  down-weighted track keeps its curvature prior, so its achieved band narrows by the same factor
  and its envelope is over-smoothed.
- **The floor mask must be about three linewidths wide, and capped.** Too wide is as bad as too
  narrow, because the fit then bridges gaps instead of reading the floor.

### Names this pass changed

| Old | New | Why |
|---|---|---|
| `joint_decompose.track_rho2_gain` | `decompose.track_rho2_gain` | it was the v2 construction copied; now both solve paths read the one |
| the loop body of `joint_solve_window` | `solve_block` / `split_block` / `floor_block` | the three blocks are separately callable and separately composable |
| `joint_decompose.joint_solve_window` | `top.joint_solve_window` (re-exported as `tracking.joint_solve_window`) | it is a composition of stages now, and stages live only in `top.py` |
| `scripts/vk_decompose.stitch_envelopes` (the joint half) | `joint_decompose.stitch_windows` | numerics out of the driver; the driver keeps the file I/O |
| `scripts/vk_decompose._order_cell_bands` | `joint_decompose.order_cell_bands` | it was written twice |
| — (new) | `JointState`, `joint_state`, `joint_result`, `solve_report`, `joint_iterations` | the state, its seed, its read-out and the report view of the log |
| — (new) | `iterate`, `windowed`, `window_geometry`, `stft_power`, `frame_starts`, `solve_audio` | the combinators and the shared kernels they and the driver both need |

## 4. The ladder cores and the frozen registry: `pipelines.py`

**`pipelines.py`** — The ladder ARRAY CORES and the FROZEN config registry — everything here computes, nothing here is a Stage. (1) The blind-annotation ladder: `CAPTURE_CFG`, `REFINE_CFG`, `TRACK_CFG`, `MIDBAND_CFG(S)`, `SEED_CFG` (calibrated values; changing them invalidates published annotations) plus `vit2dsp_pipeline`, `vit_stage1`, `tooth_cube`, `pair_score_2d_spatial`, `joint_viterbi`, `apply_guard`, `whitened_logmag_multi`, `viterbi_lattice`/`viterbi_ridge`, and `Segment` (the minimal `LadderInput`). (2) The `blind_fullrange` coarse pass: `CoarseConfig` + `coarse_init` (`bpf_octave_ratio`, `coarse_spectrogram`, `coarse_frame_scores`, `energy_bridge`) — moved out of `scripts/beatvk_vk_arms.py`, block comment and measured numbers carried across. (3) The FLAGSHIP peel: `PEEL_*` / `PI_*` / `PI_VARIANTS` / `ARMS` and `make_peels`. ONE tooth sampler, `comb_teeth`, is behind every comb reading in the module.

## 5. Protocols and prepared segments: `protocols.py`

**`protocols.py`** — Evaluation-protocol window specs as DATA (audio loaders injected by scripts; the frozen prep-cache reader lives here): `ProtocolSpec`/`WindowSpec`/`PoolSpec`, the `beatvk` + `vk37` registries (`BEATVK`, `VK37`, `PROTOCOLS`, `BEATVK_REPORT_POOLS`), `iter_windows`, `regime_of`, `to_frame` (frame builder via `stages.tracking_frame`), `FROZEN_FLY124_ALIGNMENT`. Also the three protocol operations that must exist exactly once: `slice_window` (recording -> one window's audio/grid/telemetry/edge mask), `pit_align` (THE Hungarian rotor assignment — `cost="mse"|"mae"` and an optional `edge_mask`, one implementation covering both the losses' and the campaign's conventions; `losses.pit.align_rps_to_gt` delegates to it), `pool_means`, the frozen-window reader `resolve_prep_dir` / `load_prep_window`, and the prepared-segment vocabulary `Prepared` / `smooth_frames` (the record a loader fills and the protocols' GT boxcar — declared here so a live script never has to import a campaign driver to name what it holds; FILLING it stays outside the package, `scripts/vk_validation.prepare_recording` being the vk37 loader). Consumed by `scripts/beatvk_*.py`, `scripts/tracking_ref.py`, `scripts/vk_validation.py`, `scripts/rps_eval.py`.

Tests live in `tests/tracking/`.

## The drivers, and which ones are left

Every script here injects data and calls a recipe. After the 2026-08 consolidation the live
set is small enough to list:

| Script | What it drives |
|--------|----------------|
| `scripts/tracking_ref.py` | THE guard — `--compare` / `--self-check --device cuda` / `--bench` on one frozen clip |
| `scripts/beatvk_vk_arms.py`, `beatvk_flagship.py`, `beatvk_eval.py`, `beatvk_rescore.py` | the beat-VK protocol: prep, the flagship alternation, scoring, re-score |
| `scripts/rps_eval.py` | protocol x prediction x refinement, the generic evaluator |
| `scripts/vk_validation.py` | the vk37 protocol and its DREGON loader (`prepare_recording`) |
| `scripts/vk_phase_validation.py` | the phase-recovery ladder S0–S4; its synth helpers back two tests |
| `scripts/telemetry_fitness.py`, `telemetry_refit.py`, `telemetry_report.py` | issue 17: the judge, the fitter, the reader |
| `scripts/displacement/{nullcontrol,combscan,refine_kscaled,comb_explorer}.py` | the comb-displacement campaign |
| `scripts/refine_dregon_rps.py` | the windowed L-BFGS telemetry refit of the GENERATOR's DREGON recordings -> the committed `src/data_processing/refined_labels/` sidecars |
| `scripts/vk_decompose.py` | the windowed VK decomposition -> per-recording `envelopes.npz` / `residual.npz` / `report.json` (the pooled MAP objective is `report.json` -> `objective`; `--stochastic` adds regime 3 and the gate reading `order_cell.residual_final`; `--v4` selects the unified model, implies `--joint`, refuses `--stochastic`, and adds the `h_power` line table to the unit `.npz` plus a `v4` block to the report) |
| `scripts/joint_rescore.py` | the decomposition AS A MEASURE: one window x one trajectory hypothesis (telemetry, the committed `refined` sidecar of `scripts/refine_dregon_rps.py`, or a step-5 arm of `scripts/fvk_arms.py`) -> the converged MAP objective, at a `k_hi` pinned by the telemetry so the hypotheses share their cells. `--marginal` ranks by the marginal total and prints both orders; `--h-aware` ranks by `total_h` (the stochastic comb in the data term) and prints every column; `--v4` ranks by `total_v4` and REPLACES both, being a different model rather than a correction |
| `scripts/rps_refine_lab.py` | the blind-seed arm ladder (M1/M2/M3, the oracle floor) — the one research surface not yet promoted |
| `scripts/jb_probe.py`, `sr_dp_probe.py` | the joint-tracker and single-rotor-DP probes (WP19/WP20 closed; WP21 open, so both are held) |

Retired in the consolidation, with their campaigns closed and recorded:
`vk_blind_annotation.py`, `vk_blind_sweep.py` (blind annotation —
`docs/vk-order-tracking-design.md` §7.5), `neural_reanchor.py` (negative, recorded in
`docs/experiments/beat-vk.md`), `phase_noise_cov/` (WP18 — `rps-refine-precision.md`), and the
`cd_iter` chain plus `cd_iter_sweep.sh` (built, never run — `beat-vk.md` § wide-anneal). A
retired driver's RESULTS stay on disk and stay scorable; what goes is the code that produced
them once.

## Performance knobs (issue #16)

The demodulation transforms dominate `pi_kalman_refine` and `vk_envelopes`. Every one of them is the same kernel — `dsp.zoom_bands`, reached through the one driver `dsp.demod` — and these knobs control it, all with measured defaults:

| Knob | Default | What it does |
|------|---------|--------------|
| `TRACKING_FFT_WORKERS` (env), `thread_pool(n)` (context manager), `pi_kalman_refine(threads=n)` | 1 (or `OMP_NUM_THREADS`) | Torch CPU threads, clamped to the process's CPU affinity. The name is historical — it is the tracking stack's ONE thread knob. The default stays 1 because oversubscribing on a restricted Slurm allocation thrashes, so offline and interactive callers must opt in. Bit-identical. |
| `TRACKING_DEMOD_BUDGET_MB` (env), `dsp.DEMOD_BUDGET_BYTES` / `dsp.DEVICE_BUDGET_BYTES` | 64 MB on CPU, 512 MB elsewhere | Working set of one demodulation flush, i.e. how many harmonics share a transform. On CPU this is a **cache** knob, not a memory-headroom knob: channels are already batched jointly, so bigger flushes amortize nothing and only leave cache. On a device it is the memory bound instead (the full `(8, 40, 256000)` complex64 bank is 655 MB). Bit-identical. |
| `TRACKING_DEVICE` (env), `dsp_config(device=...)` | `cpu` | Torch device — `cuda` moves the carriers, the products and the transforms onto the GPU, and switches the peel to its clip-long tiling. |
| `vk_tracking.LS_TILE_BYTES` | 256 KB | Working set of one gain-fit tile in the peel. A CPU cache knob: the tile is streamed six times, so it must stay resident. Tiles are whole 0.25 s blocks, so the floor is one block; off CPU the tile is the whole clip. Bit-identical. |
| `TRACKING_PAD` (env), `dsp_config(pad=...)` | `exact` | `fast` grows the envelope grid to the next 5-smooth length. NOT bit-identical (a zero tail lengthens the circular convolution), so it is opt-in. |

There is **no backend knob**. The scipy/pocketfft transform and the numpy peel core were deleted in the 2026-08 consolidation: torch runs the same arithmetic on CPU and on CUDA, so the second implementation bought a summation order and a maintenance surface, nothing else.

Three properties the optimization campaign relies on:

- The off-comb noise probe is sliced out of the on-comb spectrum (a constant frequency offset is a pure bin shift), so a demodulation costs one forward FFT, not two. The probe offset is snapped to the bin grid.
- Envelopes are **complex64** — the transform's own precision. Code that needs float64 (variances, gates, the Kalman) uses `_abs2` / `_increment_phase`, which compute in float64 from the components rather than widening the bank.
- The `band_env = 0.45` band always fits inside the decimated Nyquist range (`floor(0.45 n) <= (n - 1) // 2` for every `n`), so the zoom identity holds on degenerate grids too and the kernel needs no short-clip special case.

### Which kernel path

`zoom_bands` has two ways to lift a band out of the spectrum, and the choice is automatic:

- **Uniform** (one half-width and one shift for every row): two slices — `narrow` at DC, a cached `index_select` when the band is shifted. No index tensor built per call, which is what keeps the many small demods of `pi_kalman_refine` cheap.
- **Per row** (the `band_mode="k_scaled"` / `probe_mode="clean"` paths): one `gather` with a cached `(rows, n_envp)` wrapped index plus a keep mask.

A caller that hands over one entry per track when every entry is the same (`demod_bank`'s fixed probe offset is the common case) is collapsed onto the uniform path by `_collapse` — same bins, cheaper kernel.

### CPU, before and after the consolidation

Measured on the frozen 16 s / 8-mic clip, this laptop, `scripts/tracking_ref.py --bench --bench-vk`. "before" is the scipy transform plus the numpy peel at the same thread count:

| Stage | before, 1 thread | after, 1 thread | before, 4 threads | after, 4 threads |
|-------|------------------|-----------------|-------------------|------------------|
| `zoom_lp_decimate` (8, T) | 11.2 ms | 12.2 ms (+9 %) | 7.0 ms | 5.9 ms (**-16 %**) |
| `demod_bank` K=40 | 541 ms | 630 ms (+16 %) | 395 ms | 307 ms (**-22 %**) |
| `vk_envelopes` | 20.4 s | 18.7 s (**-8 %**) | 15.4 s | 13.9 s (**-10 %**) |
| `ls_project_envelopes` | 2.41 s | 2.59 s (+7 %) | 2.72 s | 2.58 s (**-5 %**) |
| `pi_kalman_refine` (full) | 3.88 s | 4.30 s (+11 %) | 2.91 s | 2.31 s (**-21 %**) |
| one peeled application, end to end | 24.0 s | 20.1 s (**-16 %**) | — | — |

The one-thread column is where torch was expected to lose — its per-call tensor/gather/copy overhead against pocketfft's single-thread SIMD, which the pre-consolidation table below measured at +46 % on `pi_kalman_refine`. Caching the band indices and taking the uniform-band slice path bought most of that back (+11 %). At four threads the consolidated path wins across the board. `vk_envelopes` improves at both counts because its coupling cross-phasors now go through the same kernel as everything else.

`pad="fast"` still earns its keep only when the bad factor is in `n_env` (1009 -> 1024) and is worth **nothing** when it is in `stride` — `n_pad` is a multiple of `stride` by construction, so a stride like `round(44100 / 62.5) = 706 = 2 * 353` poisons every admissible length. The 44.1 kHz Bluestein trap is fixed by choosing an `fs_env` whose stride factorizes, not by padding (the torch transform, which used to be the other fix, is now the only transform).

### On a GPU

```bash
TRACKING_DEVICE=cuda python <whatever>
```

Re-measured **after** the consolidation on `uni-gpushort` (job `r3-gpu-fdbf49`, 2026-08-06), same
frozen clip, `--bench --bench-devices cpu,cuda --bench-workers 1,4 --bench-vk`. All four columns
come from ONE job on ONE node, so read across a row and never against the laptop table above — the
node's CPU is the slower machine.

| Stage | cpu, 1 thr | cpu, 4 thr | cuda, 1 thr | cuda, 4 thr | cuda vs best cpu |
|-------|-----------|-----------|-------------|-------------|------------------|
| `zoom_lp_decimate` (8, T) | 17.3 ms | 5.7 ms | 3.4 ms | 3.3 ms | 1.7x |
| `_demod_bank` K=40 | 830.6 ms | 299.0 ms | 54.6 ms | 16.4 ms | **18x** |
| `vk_envelopes` | 23.5 s | 13.1 s | 10.7 s | 10.8 s | 1.2x |
| `ls_project_envelopes` | 5.85 s | 5.83 s | 264 ms | 183 ms | **32x** |
| `pi_kalman_refine` (full) | 6.14 s | 2.26 s | 415 ms | 396 ms | **5.7x** |

The consolidation did not cost the GPU anything. Against the pre-consolidation cuda leg (job
`bash-01dc95`, same node class) every stage held or improved: `_demod_bank` 24.4 -> 16.4 ms,
`pi_kalman_refine` 0.47 -> 0.40 s, `ls_project_envelopes` 0.44 -> 0.18 s, `vk_envelopes`
11.1 -> 10.7 s. That job's scipy/CPU column also lands on top of this job's torch/CPU column
(16.8 / 849 ms / 5.85 / 26.0 / 4.76 s against 17.3 / 831 ms / 6.14 / 23.5 / 5.85 s), which is the
laptop table's conclusion measured a second time: torch on CPU is the scipy path's equal, so
deleting the second implementation cost nothing.

Two things this run measured that the old one could not, both from the thread columns:

- **The peel is memory bound, and now it is proven.** `ls_project_envelopes` is the one stage whose
  CPU time does not move with thread count at all (5.85 s -> 5.83 s). Its 32x on a GPU is bandwidth,
  not arithmetic — exactly what the peel section below claims, and the reason no demod knob touches it.
- **`vk_envelopes` is now the whole cost, and the transform is not why.** On cuda it too ignores the
  thread count (10.7 s -> 10.8 s) and it is **82 %** of one end-to-end application (11.3 s of 13.8 s
  in the self-check leg). What is left in it is the scipy banded Cholesky plus the numpy cross-phasor
  host work — the two terms named under "What is left after the demod", neither of which is a
  transform. Any further GPU work on this stack starts there.

The same job ran `--self-check --device cuda`, which is how the CUDA path's CORRECTNESS is verified
(the cpu/exact and cuda legs in one process). It **passed**, so the R1/R2 consolidation introduced no
device bug at the numpy in/out seams or in the device-keyed caches:

| array | max abs | of scale | verdict |
|-------|---------|----------|---------|
| `env_z` | 3.44e-8 | 2.21e-7 | close |
| `env_x` | 1.40e-6 | 3.37e-6 | close |
| `env_x_ls` | 1.07e-6 | 9.05e-6 | close |
| `r_next` | 5.10e-6 rev/s | 5.52e-8 | close |
| `env_valid` | 0 flips | — | identical |
| `env_bw_track`, `env_t_env`, `r0` | 0 | — | identical |

End to end in that leg: 28.05 s on cpu/exact against 13.79 s on cuda (`vk_envelopes` 16.5 -> 11.3 s,
`ls_project_envelopes` 5.88 -> 0.37 s).

### The peel (`ls_project_envelopes`)

It has no transform in it at all, so no demod knob touches it, and after the demod work it was the largest remaining term: 6.9-11 s per application, a Python loop over 160 tracks. The fix is not to break the loop — tracks are fitted **sequentially against a running residual** and reordering them into independent fits is measurably worse (see the docstring) — but to see that everything *inside* one iteration is independent: 64 blocks x 8 channels.

`_ls_project` is the one core. It runs the carrier recursion on the device, keeps every host sync out of the track loop (the "track is all zero" test and the clip counter are precomputed or accumulated on the device), and sweeps tiles of `LS_TILE_BYTES` so the five block sums and the residual update stay cache-local — the tiling insight the deleted numpy core was built on, carried across. The naive form streamed six clip-long float64 arrays through DRAM per track, ~40 GB of traffic for the frozen clip. Off CPU the tile is the whole clip, because a GPU has no cache to block for and pays for kernel launches instead.

The peel's own floor is the residual traffic itself: it is read twice and written once per track, and no reordering that keeps the sequential guarantee can avoid that.

### What is left after the demod (profiled before the consolidation, 4 threads)

| Call | Total | Transform | Not the transform |
|------|-------|-----------|-------------------|
| `pi_kalman_refine` | 3.36 s | 2.19 s FFT + 0.52 s carrier `mul` | **0.15 s** — gating, observations, Kalman/RTS all together |
| `vk_envelopes` | 13.35 s | 6.00 s FFT | 3.98 s banded Cholesky, 2.04 s cross-pair phasors + bookkeeping, 0.66 s seam conversions |
| `ls_project_envelopes` | 2.5 s | none | 0.62 s residual update, 0.53 s `<resid, p/q>`, 0.48 s basis, 0.44 s Gram, 0.22 s Re/Im split |

So the "Python-side gating becomes the bottleneck" worry from issue #16 does **not** materialize: `_rw_kalman_rts` is 22 ms and the whole gating/observation layer is 4 % of `pi_kalman_refine`. What is left is the coupled-group **banded Cholesky**, the dominant term of `vk_envelopes` once the FFT is on a GPU. The banded solve stays on scipy on purpose: `torch.linalg` has no banded Cholesky, the systems are `g * n_env` up to ~64000 unknowns so dense is out, and a block-tridiagonal solver of our own is exactly the bespoke code this project does not write for a 4 s term.

One smaller lever, measured but not taken: the coupling cross-phasors are still built in numpy and shipped to the device per flush (2.0 s of host work plus ~10 GB of transfer on the GPU run — fusing them into `demod`'s recursion is the same pattern).

### The guard

`scripts/tracking_ref.py` diffs a frozen 16 s DREGON cruise window (`results/tracking_ref/`) array by array:

- `--compare [--exact]` against the stored `.npz`. Tolerance mode uses the per-array `TOL` bar (scale-relative for the envelopes, an absolute 1e-4 rev/s for `r_next`, **zero flips** for the gate masks).
- `--self-check --device cuda` runs the cpu/exact leg and the selected device in ONE process and diffs them — no 100 MB `.npz` to ship, which is how the GPU is verified.
- `--bench [--bench-devices cpu,cuda] [--bench-workers 1,4] [--bench-vk] [--bench-json PATH]` reports per-stage wall times.

The stored `.npz` was captured on the scipy transform and the numpy peel, so **`--exact` no longer passes and is not expected to**: the consolidation changed the summation order of every transform. Tolerance mode does pass, and the measured moves against that reference are

| array | max abs | of scale | bar |
|-------|---------|----------|-----|
| `env_z` | 2.36e-8 | 1.5e-7 | 1e-5 |
| `env_x` | 3.03e-7 | 7.3e-7 | 1e-3 |
| `env_x_ls` | 1.88e-7 | 1.6e-6 | 1e-3 |
| `r_next` | 6.57e-6 rev/s | — | 1e-4 rev/s |
| `env_valid` | 0 flips | — | 0 flips |

which is the same order the old scipy->torch backend swap showed — the arithmetic did not change, only where it runs.

`env_x` carries a looser bar than `env_z` for a reason: at `bw_hz = 1` the VK normal equations have `rho^2 ~ 4e5` and a condition number ~1e7, so the solve amplifies the demod's complex64 rounding by one to three orders depending on the clip. `r_next` moves 6.6e-6 rev/s and no gate flips — and `r_next` plus the gates are what the tracker consumes.

## Purity rule

This package imports only `numpy`, `scipy`, `torch`, `tdseries`, and `utils`. It must NOT import `data_processing`, `models`, or `training`. The permitted direction is `data_processing` → `tracking` (for example, `rps_synthesis` imports `tracking.rotors`).

## The Stage API (`top.py`)

Every tracking stage is a callable `Stage = Callable[[td.Frame], td.Frame]`. The frame contract:

- `"audio"`: `(mic, time)` float32 Series on a `GridIndex` at the audio rate (`tracking_frame` accepts `(T,)` and stores `(1, T)`).
- `"rps"`: `(rotor, time)` float64 Series on a `StampIndex` at the trajectory frame times — the current candidate trajectories. A stage replaces this entry (via `with_rps`) and appends one `{"stage": name, ...}` diagnostics dict to the `"tracking"` list inside the invariant `"meta"` sub-Frame (append-only; frames are never mutated).
- `"rps_meas"`: optional reference trajectories, never touched.

The adapters are thin: the array cores (`vk_track`, `blind_seed`, `pi_kalman_refine`, `iter_warp_refine`, `refine_coherent`, `stage_guard`) are unchanged; all cores accept `(T,)` or `(C, T)` audio, and frame times are re-based to the audio entry's `t_start`, so time-sliced frames work. `guarded(inner)` is the blind-annotation campaign's `_apply_guard`, promoted: run `inner`, then `stage_guard` on the before/after trajectories against the whitened spectrogram, reverting vetoed rotors.

**Seams.** A stage that does NOT change the trajectory leaves its product in `meta` and logs nothing; the stage that consumes it records what it consumed. `peel_stage` and `decompose_stage` are two such stages — `peel_stage` leaves `meta["peel_seam"]`, `pi_kalman_stage` eats it, clears it and reports the peel diagnostics in its own entry; `decompose_stage` leaves `meta["decompose"]` (the envelope bank, the global phase, the track sum and the per-track energies) for a caller rather than for a later stage, and reports the energy ledger. That is what keeps one application of the flagship one log entry however it is composed, and it is why `pipeline(peel_stage(...), pi_kalman_stage(...))` reproduces the fused `pi_kalman_arm_stage` bit for bit (`tests/tracking/test_top.py`). The annealed `pi` variants carry their trust region the same way — the last `band_b0_final` in the log — so the flagship is a plain composition and not a driver.

```python
import tracking as trk

frame = trk.tracking_frame(audio, 16000, meta={"recording_id": rid})
run = trk.pipeline(trk.blind_seed_stage(4), trk.guarded(trk.vk_stage(trk.VKConfig())))
out = run(frame)
r, ft = trk.get_rps(out)               # (4, N) rev/s + frame times
print([e["stage"] for e in out["meta"]["tracking"]])  # ['blind_seed', 'vk', 'guard']
```

The recipes are in `top.py`; their array cores are in `pipelines.py`. The flagship alternation, one frame per application:

```python
frames = trk.peel_alternation(frame, n_apps=4, arm="peeled")   # [init, app1, ..., appN]
r, ft = trk.get_rps(frames[-1])
diag = [f["meta"]["tracking"][-1] for f in frames[1:]]         # peel + step + wall per app
```

The third seam is `meta["joint"]`, and it is the one a stage both READS and REWRITES: the
`JointState` of the v3 alternation (§3.1). The three blocks are separate stages, so the whole
method is hand-composable and every round leaves its own log entry:

```python
import tracking as trk

state = trk.joint_state(r_audio, cfg, k_hi=80, n_t=n_t)     # or joint_init_stage, from a frame
run = trk.pipeline(
    trk.floor_stage(),
    trk.iterate(trk.pipeline(trk.vk_solve_stage(), trk.phase_split_stage(), trk.floor_stage()), 2),
    trk.vk_solve_stage(profile=True),
)
out = run(trk.with_meta(frame, joint=state))
res = trk.joint_result(trk.joint_state_of(out), trk.joint_iterations(out))
```

That composition IS `joint_solve_window`, which is why there is no second path. A whole
recording is `trk.windowed(run, window_s=12, hop_s=9)`.

Each application of the flagship is `flagship(1)` = `peel_stage` then `pi_kalman_stage` — `make_peels` at the current track, then one `pi_kalman_refine` pass on the peeled residuals through the tracker's `peel_audio`/`pair_audio` seam. `peel_alternation` is a driver only because it returns every intermediate frame; the alternation itself is a composition. `tracking_frame(..., dtype=np.float64)` keeps a float64 signal exact (the frame stores float32 by default and `get_audio` returns whichever it holds).

`scripts/vk_blind_annotation.py` and `scripts/vk_blind_sweep.py` are GONE — every calibrated
config and every ladder core they held is in `pipelines.py`, their PIT scorer is
`protocols.pit_align`, and their campaigns are closed in `docs/vk-order-tracking-design.md`
§7.5. What has NOT moved is the blind-seed arm ladder (M1 / M2 / M3 and the oracle floor),
which is still the sole implementation in `scripts/rps_refine_lab.py`; it is a research
surface, not a wiring layer, so it stays a script until its campaign closes.

## The comb-displacement campaign

`comb_displacement.py` and `order_domain.py` are the two INDEPENDENT estimators of the same quantity — "does the acoustic comb sit on the telemetry" — and they were built to fail differently. The first demodulates at a carrier and peak-picks inside a search window, which is precise and biased: a peak-pick inside a half-width `W` window returns about `W / 2` on pure noise, and that alone produced (and then withdrew) a published claim. The second scores a whole comb in the order domain with no window at all. Neither is trusted without the half-integer null, which both provide (`half=True`). Read `docs/experiments/dregon-comb-displacement.md` before you use either: it records what the campaign measured, what it withdrew, and why one number is still uncertain by 2x. The three drivers are `scripts/displacement/{nullcontrol,combscan,refine_kscaled}.py`.

Phase 6a of the same campaign added the JUDGE: `fitness.py` scores a candidate trajectory at fixed
degrees of freedom, with held-out harmonics/channels/time and all four section-B controls
(`docs/experiments/telemetry-fitness.md`). Phase 6b added the FITTER it judges:
`telemetry_refit.py`. The two never call each other's verdict — the fitter's only use of `fitness`
is `residual_decompose`, which is a reading of `fit - telemetry`, not a score.

Phase 6e added a second correction on top of the scale — a TIME offset. Its driver was
deleted in the R2 consolidation and its estimator promoted to `fitness.measure_tdoa`;
`docs/experiments/telemetry-fitness.md` § 6e is the record. DREGON's telemetry runs EARLY by
-42 ms [-85,-31], which excludes a tachometer reporting lag by sign; the shift and the scale
mask each other, so a one-axis sweep reads +158 ms instead. The per-microphone differential the
propagation predicts (0.156 ms) is ~500x below what the ridge resolves, and the measured per-mic
scatter says so. The inter-mic delay IS measurable — by the cross-channel phase of the comb
rather than by a rate shift — and reads slope 1.013 against the michaels rig geometry, while
DREGON's twin pair leaves too few harmonics for the same estimator to say anything.

Three things about the fitter that will bite a caller who does not read them:

- **`residual_decompose`'s `scale_pct` is not identified on a cruise window.** The rate column and
  the intercept of its `[r, dr/dt, 1]` design are collinear when a rotor holds 85 rev/s to 1 %, so
  the systematic part is split arbitrarily (DREGON w01: `scale_pct -31.9 %` with `offset +27.1
  rev/s`). `design_cond` says when. Read `telemetry_refit.scale_summary` instead.
- **`presmooth` detrends before it filters.** `brickwall` is a whole-window FFT filter and a drifting
  trajectory carries a step at the wrap; the bare filter made the synthetic carrier WORSE
  (0.087 -> 0.112 rev/s). Both `presmooth` callers get the fix.
- **The trajectory residual is not what the procedure buys.** The fitter's own per-frame noise is
  comparable to the tachometer staircase it replaces. The systematic scale and the de-staircasing
  are two corrections and are reported separately, exactly as issue 17 requires.


