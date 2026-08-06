# AGENTS.md — `src/tracking`

The rotor-speed tracking stack: Vold–Kalman order tracking, trajectory refinement, blind seeding, and beam/DP search. Pure array code, extracted from `data_processing` in the 2026-08 refactor (`docs/refactor-2026-08-plan.md` §4, Phase 2).

## Modules

| Module | Purpose |
|--------|---------|
| `vk_tracking.py` | Coupled Vold–Kalman order tracker: `VKConfig`, `vk_track`, `vk_envelopes`, `vk_reconstruct`, `demodulate`, `ls_project_envelopes` (per-harmonic per-block least-squares re-fit of the envelopes onto the audio — the peel subtraction that cannot inject energy; two cores, `_ls_project_np` cache-blocked and `_ls_project_torch` clip-long), plus the schedule helpers (`k_schedule`, `bw_schedule`, `env_stride`, `second_diff`, `fft_workers`). |
| `rps_refinement.py` | Comb-spectral trajectory refinement: `RefineConfig`, `compute_logmag`, `refine_trajectories`, `refine_coherent`, `estimate_clock_offset`, `comb_confidence`. |
| `vk_blind_seeding.py` | Blind seeding of initial trajectories: `SeedConfig`, `blind_seed`, `stage_guard`, `residual_rescan`, plus `logmag_spectrogram` — THE whitened-spectrogram core (per-channel white + raw, optional `n_fft`/`hop_length`) that `whitened_logmag`, `pipelines.whitened_logmag_multi` and the scripts' coarse pass all read through. |
| `phase_increment_tracker.py` | Phase-increment ML instantaneous-frequency tracker: `pi_kalman_refine`, `zoom_lp_decimate`, `demod_bank` (the harmonic bank; public since phase 4b, `_demod_bank` stays as an alias for the tests and `scripts/tracking_ref.py`). `pi_kalman_refine(peel_audio=, pair_audio=)` is the PEEL SEAM — per-rotor / per-pair replacement audio for that pass only (what the flagship used to inject by monkeypatching two private functions). |
| `comb_displacement.py` | Where the acoustic comb sits relative to a rotor-speed CARRIER, with the nulls that make the answer meaningful: `DisplacementConfig` (band / search window / gate geometry), `demod_comb_bank` (integer or half-integer carrier), `ridge_from_envelope`, `profile_prominence`, `pulse_pair`, `carrier_collision_mask` (the twin rule re-derived against the TRUE rotor lines, so an arbitrary carrier can be gated), `nearest_interloper_hz` (the same geometry as a DISTANCE, so a caller with its own band gets its own collision rule and can grade a contested harmonic by how close the interferer is), `weighted_stats`, `combine_k`, `measure_variant`. The carrier is one trajectory row, never "rotor i of the array" — that is what makes an off-comb, mismatched or fitted carrier expressible. Driver: `scripts/displacement/nullcontrol.py`. |
| `fitness.py` | Goodness of fit of a CANDIDATE trajectory against the audio (issue 17 phase 6a): `FitnessConfig`, `window_cells` (the one demodulation), `score_cells` -> `FitnessScore` (three components — broadband residual, `k^2`-weighted phase-increment mean square, magnitude roughness), `Holdout` (held-out harmonics / channels / time blocks as a mask over `(channel, harmonic, block)` cells), `apply_control` (the four §B controls), `bootstrap_scores`, `residual_decompose` (scale/lag + the DREGON tachometer signature) and the `fitness_stage` adapter. FIXED degrees of freedom: the band, the block grid and the admission gate are pinned to the window's REFERENCE trajectory, so the carrier is the only input that changes. The acoustic components are permutation-invariant by construction — rotor identity is certified by the residual pairing, never by the fit. Driver: `scripts/telemetry_fitness.py`; design + acceptance: `docs/experiments/telemetry-fitness.md`. |
| `order_domain.py` | The order domain: resample the audio uniformly in rotor PHASE, then FFT. `order_spectrum`, `comb_scan` (a whole comb scored at once over a scale grid, `half=True` for the half-integer null), `segment_comb_scan` (the same on short segments, which is the only reading that survives the sub-second high-k coherence time), `scan_summary`, `peak_orders`. No peak-search window anywhere — this is the estimator built to be immune to the other one's failure mode. Driver: `scripts/displacement/combscan.py`. Also the machinery issue #16 Tier 2 wants for removing the `K` factor from the demod cost. |
| `demod_backend.py` | The one zoom-IFFT band-select kernel behind every demodulation, per backend: `zoom_bands` (generic), `demod_comb` (fused carrier + transform on device), `demod_backend` / `resolve` (selection). Leaf module — imports only numpy/scipy/torch. |
| `phase_noise.py` | WP18 rank-one-plus-diagonal covariance of the per-harmonic rate opinions: `Arm`, `demod_rotor`, `arm_covariance`, `fit_rank_one`, `channel_coherence`. Measures the harmonic-common jitter term `sigma_J^2` against the per-harmonic terms `v_k` — the evidence behind the `VKConfig.freq_weight` shape. Its data side (recordings + window selection) is injected by `scripts/phase_noise_cov/windows.py`. |
| `joint_beam_tracker.py` | Joint 4-rotor beam search over comb emissions: `joint_beam_track`, `build_objective`, `comb_tables`, `comb_scores`. |
| `rotor_dp.py` | Exact single-rotor Viterbi lattice: `viterbi_path`, `greedy_peel`, `track_masked`. |
| `warp_refinement.py` | Iterated time-warp (generalized-demodulation) IF refiner: `iter_warp_refine`. |
| `rotors.py` | Quadrotor control-allocation constants: `MIXER`, `NUM_ROTORS`, `MODE_NAMES`, `modes_from_rps`, `rps_from_modes`. `data_processing.rps_synthesis` re-exports them. |
| `stages.py` | The TimeFrame stage API (plan §3.2): `Stage`, `tracking_frame` (`dtype=` keeps a float64 signal exact), `get_audio`/`get_rps`/`with_rps`, `pipeline`, and the adapters `blind_seed_stage`, `vk_stage`, `pi_kalman_stage`, `warp_stage`, `refine_coherent_stage`, `guarded`. |
| `protocols.py` | Evaluation-protocol window specs as DATA (loaders injected by scripts): `ProtocolSpec`/`WindowSpec`/`PoolSpec`, the `beatvk` + `vk37` registries (`BEATVK`, `VK37`, `PROTOCOLS`, `BEATVK_REPORT_POOLS`), `iter_windows`, `regime_of`, `to_frame` (frame builder via `stages.tracking_frame`), `FROZEN_FLY124_ALIGNMENT`. Also the three protocol operations that must exist exactly once: `slice_window` (recording -> one window's audio/grid/telemetry/edge mask), `pit_align` (THE Hungarian-on-MSE rotor assignment; `losses.pit.align_rps_to_gt` delegates to it) and `pool_means`. Consumed by `scripts/beatvk_*.py`, `scripts/tracking_ref.py`, `scripts/vk_validation.py`, `scripts/rps_eval.py`. |
| `pipelines.py` | The canonical LADDERS. (1) The blind-annotation ladder: the FROZEN config registry (`CAPTURE_CFG`, `REFINE_CFG`, `TRACK_CFG`, `MIDBAND_CFG(S)`, `SEED_CFG` — calibrated values; changing them invalidates published annotations), the vit2dsp core (`vit2dsp_pipeline`, `vit_stage1`, `tooth_cube`, `pair_score_2d_spatial`, `joint_viterbi`, `apply_guard`, `whitened_logmag_multi`, `viterbi_lattice`/`viterbi_ridge`), and the Stage adapter `vit2dsp_stage` (self-seeding via `blind_seed_stage`). (2) The FLAGSHIP peeled alternation: its frozen settings (`PEEL_*`, `PI_*`, `PI_VARIANTS`, `ARMS`), the peel `make_peels`, one application as the Stage `pi_kalman_arm_stage`, and the driver `peel_alternation`. |

Tests live in `tests/tracking/`.

## Performance knobs (issue #16)

The demodulation transforms dominate `pi_kalman_refine` and `vk_envelopes`. Every one of them is the same kernel — `demod_backend.zoom_bands` — and these knobs control it, all with measured defaults:

| Knob | Default | What it does |
|------|---------|--------------|
| `TRACKING_FFT_WORKERS` (env), `fft_worker_pool(n)` (context manager), `pi_kalman_refine(fft_workers=n)` | 1 (or `OMP_NUM_THREADS`) | FFT worker threads of the **scipy** backend, clamped to the process's CPU affinity. The default stays 1 because oversubscribing on a restricted Slurm allocation thrashes — offline and interactive callers must opt in. Bit-identical. |
| `TRACKING_DEMOD_BUDGET_MB` (env), `DEMOD_BUDGET_BYTES` | 64 MB | Working set of one **scipy** demodulation flush, i.e. how many harmonics share a transform. This is a **cache** knob, not a memory-headroom knob: channels are already batched jointly, so bigger flushes amortize nothing and only leave cache. Bit-identical. |
| `TRACKING_BACKEND` (env), `demod_backend(backend=...)` | `scipy` | `scipy` (bit-identical, the CPU default) or `torch` (device-agnostic; the GPU path). Selects the core of `ls_project_envelopes` as well, which has no transform in it — the seam is the selection, not the kernel. |
| `TRACKING_DEVICE` (env), `demod_backend(device=...)` | `cpu` | Torch device — `cuda` moves the carriers, the products and the transforms onto the GPU. |
| `vk_tracking.LS_TILE_BYTES` | 256 KB | Working set of one gain-fit tile in the **numpy** peel core. A cache knob: the tile is streamed six times, so it must stay resident. Tiles are whole 0.25 s blocks, so the floor is one block. Bit-identical. |
| `TRACKING_PAD` (env), `demod_backend(pad=...)` | `exact` | `fast` grows the envelope grid to `next_fast_len(n_env)`. NOT bit-identical (a zero tail lengthens the circular convolution), so it is opt-in. |
| `TRACKING_TORCH_BUDGET_MB` (env), `TORCH_BUDGET_BYTES` | 512 MB | Working set of one **torch** flush — the device-memory knob (the full `(8, 40, 256000)` complex64 bank is 655 MB). |

Two properties the optimization campaign relies on, both proven by `tests/tracking/test_phase_increment_tracker.py`:

- The off-comb noise probe is sliced out of the on-comb spectrum (a constant frequency offset is a pure bin shift), so a demodulation costs one forward FFT, not two. The probe offset is snapped to the bin grid.
- Envelopes are **complex64** — the transform's own precision. Code that needs float64 (variances, gates, the Kalman) uses `_abs2` / `_increment_phase`, which compute in float64 from the components rather than widening the bank.

### Which backend

Measured on the frozen 16 s / 8-mic clip (this laptop, torch CPU threads matched to the FFT worker count):

| Stage | scipy w=1 | torch w=1 | scipy w=4 | torch w=4 |
|-------|-----------|-----------|-----------|-----------|
| `_demod_bank` K=40 | 884 ms | 757 ms | 420 ms | 481 ms |
| `vk_envelopes` | 19.3 s | 18.4 s | 15.2 s | 14.2 s |
| `pi_kalman_refine` | 3.82 s | 5.58 s | 3.07 s | 2.22 s |

So **on CPU the default stays `scipy`**: torch wins nothing consistently and loses 46 % on `pi_kalman_refine` at the Slurm-safe one-thread default (its per-call tensor/gather/copy overhead dominates the many small demods). `torch` earns its keep on a GPU, and on one CPU case: a transform length with a large prime factor. Measured on an 8-channel zoom (4 threads):

| Case | `n_pad` | scipy exact | scipy fast | torch exact | torch fast |
|------|---------|-------------|------------|-------------|------------|
| 16 kHz frozen (stride 256, n_env 1000) | 256000 | 9.7 ms | 8.8 ms | 8.2 ms | 6.8 ms |
| bad `n_env` (stride 256, n_env 1009) | 258304 | 53.4 ms | 13.2 ms | 10.7 ms | 7.3 ms |
| 44.1 kHz stride (stride 706, n_env 1000) | 706000 | 156.5 ms | 159.1 ms | 40.8 ms | 44.1 ms |

Read: `pad="fast"` is worth 4x when the bad factor is in `n_env` (1009 -> 1024) and worth **nothing** when it is in `stride` — `n_pad` is a multiple of `stride` by construction, so `706 = 2 * 353` poisons every admissible length. The 44.1 kHz Bluestein trap is fixed by the **torch backend** (3.8x), or by choosing an `fs_env` whose stride factorizes; not by padding.

### On a GPU

```bash
TRACKING_BACKEND=torch TRACKING_DEVICE=cuda python <whatever>
```

Verified on `uni-gpushort` (job `bash-01dc95`, one FFT worker for the scipy leg), same frozen clip:

| Stage | scipy / CPU | torch / cuda | speedup |
|-------|-------------|--------------|---------|
| `zoom_lp_decimate` (8, T) | 16.8 ms | 3.8 ms | 4.4x |
| `_demod_bank` K=40 | 849 ms | 24.4 ms | **35x** |
| `pi_kalman_refine` (full) | 5.85 s | 0.47 s | **12.3x** |
| `vk_envelopes` | 26.0 s | 11.1 s | 2.3x |
| `ls_project_envelopes` | 4.76 s | 0.44 s | **10.9x** |
| one peeled application, end to end | 32.5 s | 13.9 s | 2.3x |

The demod itself is done: 35x on the bank, and `pi_kalman_refine` — the 16-min-per-arm call issue #16 opened on — is now under half a second. The peel is done too: the same clip's `ls_project_envelopes` read 11.2 s / 11.0 s / 1.0x on the previous job (`bash-aaaee7`), and the application 36.0 s / 24.6 s. What is left below 3x is a stage the transform never dominated.

### The peel (`ls_project_envelopes`)

It has no transform in it at all, so neither knob above touched it, and after the demod work it was the largest remaining term: 6.9-11 s per application, a Python loop over 160 tracks. The fix is not to break the loop — tracks are fitted **sequentially against a running residual** and reordering them into independent fits is measurably worse (see the docstring) — but to see that everything *inside* one iteration is independent: 64 blocks x 8 channels. Two cores now:

- `_ls_project_np` (default). The naive form streamed six clip-long float64 arrays through DRAM per track, ~40 GB of traffic for the frozen clip, with a fresh 16 MB allocation per temporary. It now sweeps tiles of `LS_TILE_BYTES`, building the basis into preallocated buffers and doing all five block sums plus the residual update tile-local, and it upsamples the envelope by broadcast over a `(C, n_knots, stride)` view instead of a gather. **Bit-identical** — same products, same `reduceat` segments, same order of subtraction — which `scripts/tracking_ref.py --compare` proves at `env_x_ls` 0.000e+00. 7.22 s -> 2.59 s on this laptop, 4.76 s on the compute node.
- `_ls_project_torch`. No tiling (a GPU has no cache to block for and pays for launches), the carrier recursion on the device, and no host sync inside the track loop. 0.44 s on `cuda`; on CPU it loses to numpy 5.66 s to 2.6 s, exactly as the transforms do, so the default stays numpy.

Two numpy traps found while proving bit-identity, both worth remembering: `np.multiply(f32, f32, out=f64)` selects the **float32** loop and rounds the product before storing it (use float64 inputs or `dtype=np.float64`), and splitting numpy's complex64 multiply into real multiplies moves the result by one float32 ulp — so the basis is built with a genuine complex multiply.

### What is left after the demod (re-profiled, torch backend, 4 threads)

| Call | Total | Transform | Not the transform |
|------|-------|-----------|-------------------|
| `pi_kalman_refine` | 3.36 s | 2.19 s FFT + 0.52 s carrier `mul` | **0.15 s** — gating, observations, Kalman/RTS all together |
| `vk_envelopes` | 13.35 s | 6.00 s FFT | 3.98 s banded Cholesky, 2.04 s cross-pair phasors + bookkeeping, 0.66 s seam conversions |
| `ls_project_envelopes` | 2.5 s | none | 0.62 s residual update, 0.53 s `<resid, p/q>`, 0.48 s basis, 0.44 s Gram, 0.22 s Re/Im split |

So the "Python-side gating becomes the bottleneck" worry from issue #16 does **not** materialize: `_rw_kalman_rts` is 22 ms and the whole gating/observation layer is 4 % of `pi_kalman_refine`. What is left is the coupled-group **banded Cholesky**, the dominant term of `vk_envelopes` once the FFT is on a GPU. The banded solve stays on scipy on purpose: `torch.linalg` has no banded Cholesky, the systems are `g * n_env` up to ~64000 unknowns so dense is out, and a block-tridiagonal solver of our own is exactly the bespoke code this project does not write for a 4 s term.

One smaller lever, measured but not taken: the coupling cross-phasors are still built in numpy and shipped to the device per flush (2.0 s of host work plus ~10 GB of transfer on the GPU run — fusing them into `demod_comb`'s recursion is the same pattern). The peel's own floor is now the residual traffic itself: it is read twice and written once per track, and no reordering that keeps the sequential guarantee can avoid that.

### The guard

`scripts/tracking_ref.py` diffs a frozen 16 s DREGON cruise window (`results/tracking_ref/`) array by array:

- `--compare [--exact]` against the stored `.npz`. Tolerance mode uses the per-array `TOL` bar (scale-relative for the envelopes, an absolute 1e-4 rev/s for `r_next`, **zero flips** for the gate masks).
- `--self-check --backend torch --device cuda` runs the scipy/exact leg and the selected backend in ONE process and diffs them — no 100 MB `.npz` to ship, which is how the GPU is verified.
- `--bench [--bench-backends scipy,torch] [--bench-workers 1,4] [--bench-vk] [--bench-json PATH]` reports per-stage wall times.

Both the peel cores are covered: `--compare` holds `_ls_project_np` to bit-identity, and `--self-check --backend torch --device cuda` diffs `_ls_project_torch` against it (job `bash-01dc95`: `env_x_ls` 8.5e-6 of scale, `r_next` 5.6e-6 rev/s, 0 gate flips).

`env_x` carries a looser bar than `env_z` for a reason: at `bw_hz = 1` the VK normal equations have `rho^2 ~ 4e5` and a condition number ~1e7, so the solve amplifies the demod's complex64 rounding by one to three orders depending on the clip (a scipy→torch swap: `env_z` 1.5e-7 of scale, `env_x` 7.3e-7 on the full 16 s clip but 3.7e-5 on a 4 s cut). `r_next` moves 4.5e-6 rev/s and no gate flips — and `r_next` plus the gates are what the tracker consumes.

## Purity rule

This package imports only `numpy`, `scipy`, `torch`, `tdseries`, and `utils`. It must NOT import `data_processing`, `models`, or `training`. The permitted direction is `data_processing` → `tracking` (for example, `rps_synthesis` imports `tracking.rotors`).

## The Stage API (`stages.py`)

Every tracking stage is a callable `Stage = Callable[[td.Frame], td.Frame]`. The frame contract:

- `"audio"`: `(mic, time)` float32 Series on a `GridIndex` at the audio rate (`tracking_frame` accepts `(T,)` and stores `(1, T)`).
- `"rps"`: `(rotor, time)` float64 Series on a `StampIndex` at the trajectory frame times — the current candidate trajectories. A stage replaces this entry (via `with_rps`) and appends one `{"stage": name, ...}` diagnostics dict to the `"tracking"` list inside the invariant `"meta"` sub-Frame (append-only; frames are never mutated).
- `"rps_meas"`: optional reference trajectories, never touched.

The adapters are thin: the array cores (`vk_track`, `blind_seed`, `pi_kalman_refine`, `iter_warp_refine`, `refine_coherent`, `stage_guard`) are unchanged; all cores accept `(T,)` or `(C, T)` audio, and frame times are re-based to the audio entry's `t_start`, so time-sliced frames work. `guarded(inner)` mirrors `scripts/vk_blind_annotation.py`'s `_apply_guard`: run `inner`, then `stage_guard` on the before/after trajectories against the whitened spectrogram, reverting vetoed rotors.

```python
import tracking as trk

frame = trk.tracking_frame(audio, 16000, meta={"recording_id": rid})
run = trk.pipeline(trk.blind_seed_stage(4), trk.guarded(trk.vk_stage(trk.VKConfig())))
out = run(frame)
r, ft = trk.get_rps(out)               # (4, N) rev/s + frame times
print([e["stage"] for e in out["meta"]["tracking"]])  # ['blind_seed', 'vk', 'guard']
```

Two ladders live in `pipelines.py`. The vit2dsp one (`vit2dsp_stage` for frames, `vit2dsp_pipeline` for arrays), and the flagship alternation:

```python
frames = trk.peel_alternation(frame, n_apps=4, arm="peeled")   # [init, app1, ..., appN]
r, ft = trk.get_rps(frames[-1])
diag = [f["meta"]["tracking"][-1] for f in frames[1:]]         # peel + step + wall per app
```

Each application is `pi_kalman_arm_stage` — `make_peels` at the current track, then one `pi_kalman_refine` pass on the peeled residuals through the tracker's `peel_audio`/`pair_audio` seam. It is a driver rather than a plain `pipeline(...)` composition because the annealed variants carry each application's `band_b0` posterior into the next. `tracking_frame(..., dtype=np.float64)` keeps a float64 signal exact (the frame stores float32 by default and `get_audio` returns whichever it holds).

`scripts/vk_blind_annotation.py` keeps thin back-compat aliases (`_SEED_CFG`, `_tooth_cube`, ...) plus everything data- or GT-bound (recording prep, mic-geometry weights, PIT scoring, superseded arms). Remaining ladders (blind-seed arms, cd_iter) stay in `scripts/rps_refine_lab.py` for now.

## The comb-displacement campaign

`comb_displacement.py` and `order_domain.py` are the two INDEPENDENT estimators of the same quantity — "does the acoustic comb sit on the telemetry" — and they were built to fail differently. The first demodulates at a carrier and peak-picks inside a search window, which is precise and biased: a peak-pick inside a half-width `W` window returns about `W / 2` on pure noise, and that alone produced (and then withdrew) a published claim. The second scores a whole comb in the order domain with no window at all. Neither is trusted without the half-integer null, which both provide (`half=True`). Read `docs/experiments/dregon-comb-displacement.md` before you use either: it records what the campaign measured, what it withdrew, and why one number is still uncertain by 2x. The three drivers are `scripts/displacement/{nullcontrol,combscan,refine_kscaled}.py`.

Phase 6a of the same campaign added the JUDGE: `fitness.py` scores a candidate trajectory at fixed
degrees of freedom, with held-out harmonics/channels/time and all four section-B controls
(`docs/experiments/telemetry-fitness.md`). The fitter it is built for arrives in phase 6b.


