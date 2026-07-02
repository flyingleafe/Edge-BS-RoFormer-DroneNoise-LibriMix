# Checkpoint: RPS training with generated-noise augmentation (GPU handoff)

**Date**: 2026-07-02 · **Branch**: `main` (uncommitted — see "Status") · **Status**:
code done + CPU-smoke verified end-to-end; **needs a real GPU run**

## TL;DR — the experiment

Train an **RPS predictor** on the online-mixing stream, but add a **generated
noise source** on top of the real DREGON + Michael's recordings: a trained
`PositionalHarmonicNoiseGen` is rendered **live on the GPU** by one background
producer process and fed in like any other noise recording. The synthetic
(intermittent) RPS trajectory that drives each generated chunk is also its
**exact RPS label** — so this is unlimited-variety, perfectly-labelled
augmentation.

**Hypothesis:** synthetic-trajectory augmentation via the generator improves RPS
prediction (especially generalization), because the model sees far more distinct
RPS trajectories than the handful of real recordings provide. (This is the
"train RPS predictors with synthetic-trajectory augmentation" bet from the
2026-06-30 supervisor slides.)

## Copy-paste to launch

On a Slurm login node, repo root, `main` checked out, with the noise-gen
checkpoint present on the cluster (see Prereqs):

```bash
dvc pull data/DREGON data/new-drone-noises data/librispeech   # noise + speech

./sbatch.sh -J rps_genaug --partition=sae --time=12:00:00 -- \
  python train_rps_predictor.py \
    --model simple_conv_v2 --device cuda:0 \
    --epochs 200 --patience 50 --batch_size 16 --num_workers 6 \
    --data_root datasets/DREGON-LM-V4-michaels \
    --online_mix \
    --mix_config configs/online_mix_generated_augment_example.yaml \
    --samples_per_validation 5000 --pit_loss
```

Validation is the **fixed real** set from `--data_root .../valid` (the generated
source is training-only). WandB → your usual RPS project (`WANDB_API_KEY` from
`.env`; no key ⇒ disabled).

## The clean A/B

The treatment config is **exactly** the established no-augmentation baseline plus
one `kind: generated` source — so the comparison is clean:

| Arm | `--mix_config` |
|-----|----------------|
| **baseline** (real noise only) | `configs/online_mix_v4_michaels_train_no_room1_no_aug.yaml` |
| **treatment** (+ generated) | `configs/online_mix_generated_augment_example.yaml` |

Run both with identical model/flags; compare PIT RMSE / R² on the fixed valid set
(FLY124 + DREGON room-mismatch), i.e. the same numbers as the
[FLY125 cross-drone](../../.pi/checkpoints) work. Everything else — DREGON room1
exclusion, FLY125-train-only, LibriSpeech source, SNR `[-30,0]` — is identical.

## What the generated source is (config block)

```yaml
- kind: generated
  weight: 2.0            # ~1/3 of noise batches (real pool weight = 2 real sources)
  checkpoint: results/noise_gen_dregon_michaels_swapped/best_positional_harmonic_gen.pt
  drone: michaels        # codebook key + geometry (michaels reproduces best)
  n_harmonics: 100       # MUST equal the checkpoint's training value
  device: cuda:0         # the ONE extra CUDA context (the producer)
  gen_batch: 32
  random_phase: true     # per-chunk harmonic phase => extra texture variety
  refresh: true          # live stream (set false for a reproducible fixed bank)
  rps: {kind: synthetic_intermittent, aggressiveness: 1.0}
  buffer: {slots: 512, warmup: 32}   # ring ≈ 384 MB host RAM
```

Architecture (why a process + shared buffer): the mixer runs in **forked**
DataLoader workers, which can't init CUDA. So one **spawn** producer owns the
single generation CUDA context and renders batches into a **shared-memory ring
buffer**; the fork workers only read finished chunks (lock-free seqlock).
Generation rate is decoupled from consumption — a slow GPU just means more chunk
reuse. Full write-up: `data_processing/AGENTS.md` § "Generated noise source".

## Prerequisites

1. **A trained noise-gen checkpoint on the cluster** at the `checkpoint:` path.
   Either:
   - sync the local `results/noise_gen_dregon_michaels_swapped/best_positional_harmonic_gen.pt`, or
   - use the output of the noise-gen run in
     [`noise-gen-swapped-smoothness-random-phase.md`](noise-gen-swapped-smoothness-random-phase.md)
     (`<save_path>/best_positional_harmonic_gen.pt`) and point `checkpoint:` at it.
   **`n_harmonics` in the config must match how that checkpoint was trained (100
   for the standard runs).** It is not stored in the bundle, so a mismatch is a
   silent `load_state_dict` shape error.
2. **Data** under repo-relative `data/` (config uses `root: data`): DREGON,
   Michael's (`new-drone-noises`), LibriSpeech. Same as other online-mix jobs.
3. **A GPU** — the producer renders on `cuda:0`, the same GPU training uses (the
   producer model is tiny, ~236k params; it shares the card fine).

## CPU smoke already done (plumbing proof)

End-to-end on CPU (tiny random-init checkpoint): `kind: generated` flows through
`OnlineMixIterableDataset` and yields `(mixture (8,T), rps (4,F))`; the producer
process, shared buffer, seqlock reads, and TimeFrame wrapping all work. Tests:
`tests/data_processing/test_generated_noise.py` (+ existing online-mix tests) —
**42 pass**. This verifies plumbing only, **not** whether augmentation helps.

## Gotchas (the "without thinking twice" list)

- **`n_harmonics` must match the checkpoint** (see Prereqs) — the single most
  likely footgun.
- **`device: cuda:0` on the generated source** must be the GPU your job gets.
  Under Slurm this is the allocated card; leave it `cuda:0`.
- **Keep `--num_workers` modest (≈4–8).** Workers fork from a CUDA-initialised
  main; they don't touch CUDA (only read shared memory), which is fine, but a
  huge worker count multiplies the fork warning noise and host-RAM for the ring.
- **The generated stream is NOT seed-reproducible** (buffer contents depend on
  timing). That's fine for training; validation stays on the fixed real
  `--data_root .../valid`. For a reproducible generated set use `refresh: false`.
- **`weight` tunes the generated share.** Default 1.0 per source item; with two
  real sources + the generated at `weight: 2.0`, generated is ~½. Lower it to
  lean on real noise, raise it to lean on synthetic variety — worth a small sweep
  (e.g. generated ∈ {0.5, 1, 2, 4} × the real total).
- **Try both drones.** Add a second `kind: generated` with `drone: dregon`
  (and `dregon_dir: data/DREGON`) for more trajectory/timbre variety — but note
  the generator reproduces DREGON worse (wind), so start michaels-only.
- **Producer teardown** is automatic (`atexit` / `close()`); if a job is killed
  hard, the OS reclaims the shared memory. No manual cleanup.

## Output / next steps

- Standard RPS-predictor checkpoints + WandB curves; compare treatment vs baseline
  PIT RMSE / R² on the fixed valid set.
- If it helps: sweep `weight` and `aggressiveness`; add generated `dregon`; try a
  smoothness-trained noise-gen checkpoint (should give cleaner harmonics).
- If it hurts: the generator quality is the bottleneck (mid-freq harmonics / DREGON
  wind) — feed that back into noise-gen training, not the RPS model.

## Code pointers

- Source pool + producer: `data_processing/generated_noise.py` (`GeneratedNoisePool`)
- Dispatch / mixing: `data_processing/online_mixing.py` (`build_noise_pool`,
  `MixedNoisePool`, `NoisePool`)
- Config (treatment): `configs/online_mix_generated_augment_example.yaml`
- RPS trajectories: `data_processing/rps_synthesis.py` (`generate_intermittent_batch`)
- Tests: `tests/data_processing/test_generated_noise.py`
- Docs: `data_processing/AGENTS.md` § "Generated noise source", `configs/AGENTS.md`
- Noise-gen checkpoint it depends on:
  [`noise-gen-swapped-smoothness-random-phase.md`](noise-gen-swapped-smoothness-random-phase.md)
