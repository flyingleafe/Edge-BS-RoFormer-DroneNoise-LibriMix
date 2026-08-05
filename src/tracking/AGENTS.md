# AGENTS.md — `src/tracking`

The rotor-speed tracking stack: Vold–Kalman order tracking, trajectory refinement, blind seeding, and beam/DP search. Pure array code, extracted from `data_processing` in the 2026-08 refactor (`docs/refactor-2026-08-plan.md` §4, Phase 2).

## Modules

| Module | Purpose |
|--------|---------|
| `vk_tracking.py` | Coupled Vold–Kalman order tracker: `VKConfig`, `vk_track`, `vk_envelopes`, `vk_reconstruct`, `demodulate`, `ls_project_envelopes` (per-harmonic per-block least-squares re-fit of the envelopes onto the audio — the peel subtraction that cannot inject energy), plus the schedule helpers (`k_schedule`, `bw_schedule`, `env_stride`, `second_diff`, `fft_workers`). |
| `rps_refinement.py` | Comb-spectral trajectory refinement: `RefineConfig`, `compute_logmag`, `refine_trajectories`, `refine_coherent`, `estimate_clock_offset`, `comb_confidence`. |
| `vk_blind_seeding.py` | Blind seeding of initial trajectories: `SeedConfig`, `blind_seed`, `whitened_logmag`, `stage_guard`, `residual_rescan`. |
| `phase_increment_tracker.py` | Phase-increment ML instantaneous-frequency tracker: `pi_kalman_refine`, `zoom_lp_decimate`. |
| `demod_backend.py` | The one zoom-IFFT band-select kernel behind every demodulation, per backend: `zoom_bands` (generic), `demod_comb` (fused carrier + transform on device), `demod_backend` / `resolve` (selection). Leaf module — imports only numpy/scipy/torch. |
| `phase_noise.py` | WP18 rank-one-plus-diagonal covariance of the per-harmonic rate opinions: `Arm`, `demod_rotor`, `arm_covariance`, `fit_rank_one`, `channel_coherence`. Measures the harmonic-common jitter term `sigma_J^2` against the per-harmonic terms `v_k` — the evidence behind the `VKConfig.freq_weight` shape. Its data side (recordings + window selection) is injected by `scripts/phase_noise_cov/windows.py`. |
| `joint_beam_tracker.py` | Joint 4-rotor beam search over comb emissions: `joint_beam_track`, `build_objective`, `comb_tables`, `comb_scores`. |
| `rotor_dp.py` | Exact single-rotor Viterbi lattice: `viterbi_path`, `greedy_peel`, `track_masked`. |
| `warp_refinement.py` | Iterated time-warp (generalized-demodulation) IF refiner: `iter_warp_refine`. |
| `rotors.py` | Quadrotor control-allocation constants: `MIXER`, `NUM_ROTORS`, `MODE_NAMES`, `modes_from_rps`, `rps_from_modes`. `data_processing.rps_synthesis` re-exports them. |
| `stages.py` | The TimeFrame stage API (plan §3.2): `Stage`, `tracking_frame`, `get_audio`/`get_rps`/`with_rps`, `pipeline`, and the adapters `blind_seed_stage`, `vk_stage`, `pi_kalman_stage`, `warp_stage`, `refine_coherent_stage`, `guarded`. |
| `protocols.py` | Evaluation-protocol window specs as DATA (loaders injected by scripts): `ProtocolSpec`/`WindowSpec`/`PoolSpec`, the `beatvk` + `vk37` registries (`BEATVK`, `VK37`, `PROTOCOLS`), `iter_windows`, `regime_of`, `to_frame` (frame builder via `stages.tracking_frame`), `FROZEN_FLY124_ALIGNMENT`. Consumed by `scripts/beatvk_eval.py`, `scripts/beatvk_vk_arms.py`, `scripts/vk_validation.py`, `scripts/rps_eval.py`. |
| `pipelines.py` | The canonical blind-annotation ladder: the FROZEN config registry (`CAPTURE_CFG`, `REFINE_CFG`, `TRACK_CFG`, `MIDBAND_CFG(S)`, `SEED_CFG` — calibrated values; changing them invalidates published annotations), the vit2dsp core (`vit2dsp_pipeline`, `vit_stage1`, `tooth_cube`, `pair_score_2d_spatial`, `joint_viterbi`, `apply_guard`, `whitened_logmag_multi`), and the Stage adapter `vit2dsp_stage` (self-seeding via `blind_seed_stage`). |

Tests live in `tests/tracking/`.

## Performance knobs (issue #16)

The demodulation transforms dominate `pi_kalman_refine` and `vk_envelopes`. Every one of them is the same kernel — `demod_backend.zoom_bands` — and these knobs control it, all with measured defaults:

| Knob | Default | What it does |
|------|---------|--------------|
| `TRACKING_FFT_WORKERS` (env), `fft_worker_pool(n)` (context manager), `pi_kalman_refine(fft_workers=n)` | 1 (or `OMP_NUM_THREADS`) | FFT worker threads of the **scipy** backend, clamped to the process's CPU affinity. The default stays 1 because oversubscribing on a restricted Slurm allocation thrashes — offline and interactive callers must opt in. Bit-identical. |
| `TRACKING_DEMOD_BUDGET_MB` (env), `DEMOD_BUDGET_BYTES` | 64 MB | Working set of one **scipy** demodulation flush, i.e. how many harmonics share a transform. This is a **cache** knob, not a memory-headroom knob: channels are already batched jointly, so bigger flushes amortize nothing and only leave cache. Bit-identical. |
| `TRACKING_BACKEND` (env), `demod_backend(backend=...)` | `scipy` | `scipy` (bit-identical, the CPU default) or `torch` (device-agnostic; the GPU path). |
| `TRACKING_DEVICE` (env), `demod_backend(device=...)` | `cpu` | Torch device — `cuda` moves the carriers, the products and the transforms onto the GPU. |
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

### The guard

`scripts/tracking_ref.py` diffs a frozen 16 s DREGON cruise window (`results/tracking_ref/`) array by array:

- `--compare [--exact]` against the stored `.npz`. Tolerance mode uses the per-array `TOL` bar (scale-relative for the envelopes, an absolute 1e-4 rev/s for `r_next`, **zero flips** for the gate masks).
- `--self-check --backend torch --device cuda` runs the scipy/exact leg and the selected backend in ONE process and diffs them — no 100 MB `.npz` to ship, which is how the GPU is verified.
- `--bench [--bench-backends scipy,torch] [--bench-workers 1,4] [--bench-vk] [--bench-json PATH]` reports per-stage wall times.

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

The vit2dsp ladder lives in `pipelines.py` (`vit2dsp_stage` for frames, `vit2dsp_pipeline` for arrays). `scripts/vk_blind_annotation.py` keeps thin back-compat aliases (`_SEED_CFG`, `_tooth_cube`, ...) plus everything data- or GT-bound (recording prep, mic-geometry weights, PIT scoring, superseded arms). Remaining ladders (blind-seed arms, cd_iter) stay in `scripts/rps_refine_lab.py` for now.
