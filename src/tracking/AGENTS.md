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
| `refit_stage` | `RefitConfig` | the whole telemetry refit as one stage |
| `guarded` | `SeedConfig` | wrap a stage with the blind per-track guard |

The recipes:

| Recipe | Composition |
|--------|-------------|
| `vit2dsp` | blind seed → the calibrated ladder |
| `blind_fullrange` | blind seed (K, R) → coarse full-range init → the ladder |
| `flagship(n_apps)` | `n_apps` × (peel → pi_kalman) |
| `peel_alternation` | `flagship` one application at a time, every frame kept |
| `refit_stage` | presmooth → coarse-to-fine (peel → pi_kalman) to convergence |
| `judge(candidate)` | a candidate stage → `fitness_stage` under one control |

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

**`decompose.py`** — The WINDOWED decomposition: `x(t) = sum_{rotor,k} Re[a e^{jk phi}] + residual`, one coupled-VK solve per window, stitched. Three sections. (1) The **windowed-application primitives** — `frame_grid`, `interp_rps`, `window_bounds`, `window_span`, `fade_weights`, `to_audio_grid` (re-exported from `fitness_vk`, so a window is decomposed at the carrier it was scored at) and `shaft_phase`. Tiling a recording and cross-fading the per-window results back is the same operation whatever the windows produce, so `scripts/refine_dregon_rps.py` reads through the same functions instead of holding a second copy. (2) The **solve** — `solve_config` (the one measurement geometry), `solve_window`, `group_plan` (THE memory model: coupling is transitive, the whole comb is ONE banded system, `~1e-4 k_hi^2 window_s` GB per worker — read it before sizing a job), `reconstruct` (`vk_reconstruct`'s interpolation rule against a GLOBAL phase, plus per-track energies; the two are diffed against each other in the tests), `stitch_bank` (the phase re-reference `exp(-j k Phi(a0-1))` — without it two overlapping windows hold one physical track at two phase origins and the cross-fade cancels them — and the cross-fade), `phase_reference_deviation` (that assumption MEASURED, not assumed). (3) The **readings** — `energy_ledger` (total / tracks / residual / cross term; the tracks are not orthogonal, so the cross term is the honest statement of inter-track interference) and `phase_model_report` (per-rotor drift against `k` plus the `rank_one_share` — the shaft-jitter vs per-harmonic-drift discrimination). The sum is EXACT by construction because the residual is DEFINED as the unexplained part; only the SPLIT is estimated, and it is the MAP estimate under the VK prior. Driver: `scripts/vk_decompose.py` (data, units and file formats only).

**`telemetry_refit.py`** — The FITTER phase 6a's harness was built to judge (issue 17 phase 6b): `RefitConfig`, `refit_window` -> `RefitResult`, `presmooth` (THE 5 Hz detrended low pass of the campaign — the 6a driver's `lp:` candidate calls it), `k_cap_for_error` / `advance_k` (the coarse-to-fine ladder, from the phase-wrap capture rule `k <= wrap_guard_rad fs_env / (2 pi e)` — never a flat `k_caps`), `scale_summary` (the well-posed scale, per-rotor mean shift plus one joint global LS scale), and `order_and_gaps` (THE identity test). One outer iteration IS one `top.pi_kalman_arm_stage` application, so the LS peel, the peel seam and the twin rule are the flagship's, wired not rebuilt. Driver: `scripts/telemetry_refit.py`; design + acceptance: `docs/experiments/telemetry-fitness.md` § "The fitter".

**`comb_displacement.py`** — Where the acoustic comb sits relative to a rotor-speed CARRIER, with the nulls that make the answer meaningful: `DisplacementConfig` (band / search window / gate geometry), `demod_comb_bank` (integer or half-integer carrier), `ridge_from_envelope`, `profile_prominence`, `pulse_pair`, `carrier_collision_mask` (the twin rule re-derived against the TRUE rotor lines, so an arbitrary carrier can be gated), `nearest_interloper_hz` (the same geometry as a DISTANCE, so a caller with its own band gets its own collision rule and can grade a contested harmonic by how close the interferer is), `weighted_stats`, `combine_k`, `measure_variant`. The carrier is one trajectory row, never "rotor i of the array" — that is what makes an off-comb, mismatched or fitted carrier expressible. Driver: `scripts/displacement/nullcontrol.py`.

**`order_domain.py`** — The order domain: resample the audio uniformly in rotor PHASE, then FFT. `order_spectrum`, `comb_scan` (a whole comb scored at once over a scale grid, `half=True` for the half-integer null), `segment_comb_scan` (the same on short segments, which is the only reading that survives the sub-second high-k coherence time), `scan_summary`, `peak_orders`. No peak-search window anywhere — this is the estimator built to be immune to the other one's failure mode. Driver: `scripts/displacement/combscan.py`. Also the machinery issue #16 Tier 2 wants for removing the `K` factor from the demod cost.

**`phase_noise.py`** — WP18 rank-one-plus-diagonal covariance of the per-harmonic rate opinions: `Arm`, `demod_rotor`, `arm_covariance`, `fit_rank_one`, `channel_coherence`, plus `brickwall` — THE whole-window FFT filter of the package, which `fitness` and `telemetry_refit.presmooth` both read through. Measures the harmonic-common jitter term `sigma_J^2` against the per-harmonic terms `v_k` — the evidence behind the `VKConfig.freq_weight` shape. Its data side (recordings + window selection) is injected by the caller — the WP18 campaign's own window builder was retired with the campaign, so a new caller must supply its own windows. Record: `docs/experiments/rps-refine-precision.md`.

### Constants

**`rotors.py`** — Quadrotor control-allocation constants: `MIXER`, `NUM_ROTORS`, `MODE_NAMES`, `modes_from_rps`, `rps_from_modes`. `data_processing.rps_synthesis` re-exports them.

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
| `scripts/vk_decompose.py` | the windowed VK decomposition -> per-recording `envelopes.npz` / `residual.npz` / `report.json` |
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

**Seams.** A stage that does NOT change the trajectory leaves its product in `meta` and logs nothing; the stage that consumes it records what it consumed. `peel_stage` and `decompose_stage` are the two such stages — `peel_stage` leaves `meta["peel_seam"]`, `pi_kalman_stage` eats it, clears it and reports the peel diagnostics in its own entry; `decompose_stage` leaves `meta["decompose"]` (the envelope bank, the global phase, the track sum and the per-track energies) for a caller rather than for a later stage, and reports the energy ledger. That is what keeps one application of the flagship one log entry however it is composed, and it is why `pipeline(peel_stage(...), pi_kalman_stage(...))` reproduces the fused `pi_kalman_arm_stage` bit for bit (`tests/tracking/test_top.py`). The annealed `pi` variants carry their trust region the same way — the last `band_b0_final` in the log — so the flagship is a plain composition and not a driver.

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

Each application is `flagship(1)` = `peel_stage` then `pi_kalman_stage` — `make_peels` at the current track, then one `pi_kalman_refine` pass on the peeled residuals through the tracker's `peel_audio`/`pair_audio` seam. `peel_alternation` is a driver only because it returns every intermediate frame; the alternation itself is a composition. `tracking_frame(..., dtype=np.float64)` keeps a float64 signal exact (the frame stores float32 by default and `get_audio` returns whichever it holds).

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


