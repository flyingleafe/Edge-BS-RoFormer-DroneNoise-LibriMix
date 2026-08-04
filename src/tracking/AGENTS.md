# AGENTS.md — `src/tracking`

The rotor-speed tracking stack: Vold–Kalman order tracking, trajectory refinement, blind seeding, and beam/DP search. Pure array code, extracted from `data_processing` in the 2026-08 refactor (`docs/refactor-2026-08-plan.md` §4, Phase 2).

## Modules

| Module | Purpose |
|--------|---------|
| `vk_tracking.py` | Coupled Vold–Kalman order tracker: `VKConfig`, `vk_track`, `vk_envelopes`, `vk_reconstruct`, `demodulate`, plus the schedule helpers (`k_schedule`, `bw_schedule`, `env_stride`, `second_diff`, `fft_workers`). |
| `rps_refinement.py` | Comb-spectral trajectory refinement: `RefineConfig`, `compute_logmag`, `refine_trajectories`, `refine_coherent`, `estimate_clock_offset`, `comb_confidence`. |
| `vk_blind_seeding.py` | Blind seeding of initial trajectories: `SeedConfig`, `blind_seed`, `whitened_logmag`, `stage_guard`, `residual_rescan`. |
| `phase_increment_tracker.py` | Phase-increment ML instantaneous-frequency tracker: `pi_kalman_refine`, `zoom_lp_decimate`. |
| `joint_beam_tracker.py` | Joint 4-rotor beam search over comb emissions: `joint_beam_track`, `build_objective`, `comb_tables`, `comb_scores`. |
| `rotor_dp.py` | Exact single-rotor Viterbi lattice: `viterbi_path`, `greedy_peel`, `track_masked`. |
| `warp_refinement.py` | Iterated time-warp (generalized-demodulation) IF refiner: `iter_warp_refine`. |
| `rotors.py` | Quadrotor control-allocation constants: `MIXER`, `NUM_ROTORS`, `MODE_NAMES`, `modes_from_rps`, `rps_from_modes`. `data_processing.rps_synthesis` re-exports them. |
| `stages.py` | The TimeFrame stage API (plan §3.2): `Stage`, `tracking_frame`, `get_audio`/`get_rps`/`with_rps`, `pipeline`, and the adapters `blind_seed_stage`, `vk_stage`, `pi_kalman_stage`, `warp_stage`, `refine_coherent_stage`, `guarded`. |
| `pipelines.py` | The canonical blind-annotation ladder: the FROZEN config registry (`CAPTURE_CFG`, `REFINE_CFG`, `TRACK_CFG`, `MIDBAND_CFG(S)`, `SEED_CFG` — calibrated values; changing them invalidates published annotations), the vit2dsp core (`vit2dsp_pipeline`, `vit_stage1`, `tooth_cube`, `pair_score_2d_spatial`, `joint_viterbi`, `apply_guard`, `whitened_logmag_multi`), and the Stage adapter `vit2dsp_stage` (self-seeding via `blind_seed_stage`). |

Tests live in `tests/tracking/`.

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
