# Checkpoint: Noise-Gen — swapped split + random phases + smoothness (GPU handoff)

**Date**: 2026-07-01 · **Branch**: `main` · **Status**: config committed, full
pipeline CPU-smoke verified end-to-end; **needs a real GPU run**

Supersedes the launch command in
[`noise-generation-online-dregon-michaels.md`](noise-generation-online-dregon-michaels.md)
(read that first for the model/data background — this doc only changes three
things on top of it).

## What's new vs. the previous handoff

1. **Swapped (correct) DREGON split.** The old config trained on *one* DREGON
   recording (room1) and validated on *five* (room2) — backwards. The run that
   produced the slide checkpoint (`results/noise_gen_dregon_michaels_swapped/`)
   swapped them. That split is now committed as
   **`configs/noise_gen_online_dregon_michaels_swapped.yaml`**:
   - **train** = DREGON room2 nosource (5 recs) + Michael's **FLY125**
   - **valid** = DREGON room1 nosource (1 rec) + Michael's **FLY124**
2. **Random initial harmonic phases during training** — *automatic, nothing to
   set.* `HarmonicNoiseGenNew.forward` draws a fresh random phase per harmonic
   while `model.training` is True, and uses **zero** phase at eval. The training
   loop already does `model.train()`/`model.eval()`, so this "just happens." (For
   inference/rendering always call `model.eval()`, as the notebook and slides
   `prepare.py` already do — that gives deterministic, zero-phase synthesis.)
3. **Smoothness regularisers** (the Stage-2 report's squared-2nd-difference
   penalties), opt-in via two flags:
   - `--harm_smooth_weight` — harmonic amplitudes over **time**
   - `--noise_smooth_weight` — diffuse-noise filter shape over **time + freq**
   Default 0 (off). Validation stays pure multi-scale-STFT so best-checkpoint
   selection is comparable to prior runs.

## TL;DR for the colleague — copy-paste to launch

On a Slurm login node, repo root, `main` checked out:

```bash
dvc pull data/DREGON data/new-drone-noises   # noise recordings (see Prereqs)

./sbatch.sh -J noisegen_swap --partition=sae --time=8:00:00 -- \
  python train_noise_generation.py \
    --online_config configs/noise_gen_online_dregon_michaels_swapped.yaml \
    --cond_dim 16 --device cuda:0 \
    --epochs 200 --patience 20 --batch_size 32 \
    --duration_s 1.0 --n_harmonics 100 \
    --samples_per_epoch 6000 --num_valid 256 --num_workers 8 \
    --harm_smooth_weight 1e-2 --noise_smooth_weight 1e-2 \
    --save_path /gpfs/scratch/acw592/results/noise_gen_swapped_smooth
```

That's the whole experiment. Random phases are on automatically (train mode).
WandB logs to project `noise-generation` (`WANDB_API_KEY` from `.env`/env; no key
⇒ `mode=disabled`). It logs `train/spectral` and `train/smoothness` separately
whenever a smoothness weight is > 0.

> **Path note:** the committed config uses `root: data` (repo-relative). If the
> recordings live under `/gpfs/scratch/acw592/data`, either symlink `data/` →
> there or copy the config and edit the two `root:` fields (as the old GPFS run
> did — see `results/noise_gen_dregon_michaels/*.gpfs.swapped.yaml`).

## Tuning the smoothness weights (please read — likely needs a small sweep)

`1e-2` matches the report/CLI-help nominal, but it is **probably too weak**. In a
CPU smoke the raw (unweighted) penalty was ~0.26 while the spectral loss was in
the hundreds, so at `1e-2` smoothness contributes ~0.003 — negligible. The
amplitude 2nd-difference is naturally small, so the weight has to be large to
bite. Suggested short sweep once the baseline trains cleanly:

- `--harm_smooth_weight` ∈ {1e-2, 1e-1, 1, 10}
- `--noise_smooth_weight` ∈ {1e-2, 1e-1, 1, 10}

Pick the largest weights that don't start hurting the **val** spectral loss
(over-smoothing kills transients). The two are independent — the harmonic term
smooths amplitude flicker, the noise term smooths the broadband filter shape.
`--noise_smooth_weight` is ignored under `--no_diff_noise` (no noise branch).

## Verification already done (CPU smoke, this exact config)

1-epoch, `--samples_per_epoch 24 --num_valid 8 --batch_size 4 --n_harmonics 64
--device cpu` with both smoothness weights at `1e-2`:
```
Online | train recs [...room2 x5..., michaels_FLY125]
       | valid recs ['free-flight_nosource_room1', 'michaels_FLY124']
       | drones ['dregon', 'michaels']
Smoothness: harm 0.01 (time) | noise 0.01 (time+freq)
    1   675.1905   246.5817    1.0e-03
train/smoothness 0.00257   train/spectral 675.188
```
Confirms: swapped split resolves, smoothness path runs + backprops + logs,
random-phase train path runs. This is plumbing only — **not** quality (random
init, tiny counts). The GPU run is the real thing.

## Checkpoint format (unchanged)

`--save_path/best_positional_harmonic_gen.pt` is a bundle
`{"model","codebook","cond_dim","drone_names"}`. Load exactly as in the previous
handoff doc / `notebooks/noise_gen_real_vs_generated.ipynb`. Render with
`model.eval()` (⇒ zero-phase, deterministic).

## Prereqs / gotchas (in addition to the previous doc)

- `min_motor_rps: 30.0` on Michael's is load-bearing (FLY124 has idle/zero-RPS
  regions); DREGON sources need `download: false`. Both already set in the config.
- Do a `--partition=gpushort --time=1:00:00` GPU smoke first if unsure.
- Compare against the slide checkpoint
  (`results/noise_gen_dregon_michaels_swapped/best_positional_harmonic_gen.pt`),
  which is the *same split without* random phases / smoothness — so any A/B is
  clean: does random-phase + smoothness improve the rendered noise (esp. DREGON
  mid-frequency harmonics, the known weak spot)?

## Code pointers

- Config (new): `configs/noise_gen_online_dregon_michaels_swapped.yaml`
- Training + smoothness loss: `train_noise_generation.py`
  (`_smoothness_loss`, `--harm_smooth_weight`/`--noise_smooth_weight`)
- Random-phase logic: `src/models/generative/harmonic_gen_new.py`
  (`HarmonicNoiseGenNew.forward`, `initial_phases`), threaded through
  `src/models/generative/positional_harmonic_gen.py` (`emit`/`forward`)
- Smoothness penalty: `src/models/generative/losses.py` (`smoothness_penalty`)
- Tests: `tests/train/test_noise_generation.py`,
  `tests/models/test_positional_harmonic_gen.py`
