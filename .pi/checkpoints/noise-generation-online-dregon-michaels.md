# Checkpoint: Noise-Generation Online Training (DREGON + Michael's, GPU handoff)

**Date**: 2026-06-26 · **Branch**: `noise-generation-task` · **Status**: code done + CPU-smoke verified; **needs a real GPU run**

## TL;DR for the colleague — copy-paste to launch

On a Slurm login node, from the repo root, with the branch checked out:

```bash
git checkout noise-generation-task
dvc pull data/DREGON data/new-drone-noises        # ~needed: noise recordings (see Prereqs)

./sbatch.sh -J noisegen_dm --partition=sae --time=8:00:00 -- \
  python train_noise_generation.py \
    --online_config configs/noise_gen_online_dregon_michaels.yaml \
    --cond_dim 16 --device cuda:0 \
    --epochs 200 --patience 20 --batch_size 32 \
    --duration_s 1.0 --n_harmonics 100 \
    --samples_per_epoch 6000 --num_valid 256 --num_workers 8 \
    --save_path /gpfs/scratch/acw592/results/noise_gen_dregon_michaels
```

That's the whole experiment. WandB logs to project `noise-generation` (uses
`WANDB_API_KEY` from `.env`/env; without a key it runs `mode=disabled`).
Logs land in `/gpfs/scratch/acw592/logs/%x.o%j`.

## What this experiment is

Trains the **position-aware drone-noise generator** (inverse of RPS prediction:
RPS + array geometry → multichannel noise) jointly on **two drones** with a
learned per-drone code.

- **Model**: `PositionalHarmonicNoiseGen` (`src/models/generative/positional_harmonic_gen.py`).
  Single-rotor harmonic+filtered-noise emitter synthesises each rotor; `propagate`
  renders to all 8 mics (1/r attenuation + exact fractional delay, rotors summed
  in the rfft domain). Differentiable w.r.t. position.
- **Per-drone conditioning is EXTERNAL** (`cond_dim=16`): the model takes a code
  `z (B,16)` as a forward input (like geometry); the `name→z` table is a
  separate `DroneCodebook` (`src/tasks/noise_generation.py`). Model params never
  resize with drone count. FiLM-modulates the emitter backbone.
- **Online data**: reuses the RPS online-mixing slicer
  (`data_processing.online_mixing.TimeFrameNoisePool`) to cut clean noise + RPS +
  **per-frame geometry** on the fly from long recordings — no precomputed chunks,
  no speech mixing. Each sliced frame carries its own geometry (`global_data`)
  and drone identity, so DREGON and Michael's stream together.

## Data split (set in `configs/noise_gen_online_dregon_michaels.yaml`)

| Split | DREGON (in_flight_noise) | Michael's |
|-------|--------------------------|-----------|
| **train** | `free-flight_nosource_room1` (the only non-room2 nosource) | `FLY125` |
| **valid** | room2 nosource ×5: `free-flight`, `hovering`, `updown`, `rectangle`, `spinning` | `FLY124` |

Both drones: 8-channel audio, 4 rotors → `rel_pos (8,4,3)`, fully batchable
together. `drone_name` = `michaels` if `recording_id` startswith `michaels`,
else `dregon` → codebook gets 2 codes `['dregon','michaels']`.

## Prerequisites on the cluster

1. **Data** under repo-relative `data/` (the config uses `root: data`):
   - `dvc pull data/DREGON data/new-drone-noises`
   - Michael's loader honours `DATA_ROOT` env (defaults to `<repo>/data`); if the
     recordings live under `/gpfs/scratch/acw592/data`, either symlink `data/` →
     there or `export DATA_ROOT=/gpfs/scratch/acw592/data` **and** edit the two
     `root:` fields in the YAML to match.
   - No geometry photos needed at runtime — geometry is computed from constants
     in `data_processing/michaels.py` / `dregon.py`.
2. **Env**: the project venv/flake (torch, torchaudio, wandb, dvc). Same as RPS
   training jobs.

## CPU smoke already done (proof it runs end-to-end)

`--samples_per_epoch 128 --num_valid 32 --batch_size 8 --n_harmonics 64 --device cpu`,
1 epoch, ~0.4 min:
```
Online | drones ['dregon', 'michaels']
Train: 128 stream | Valid: 32 fixed | params 236,572 | codebook dim 16
  e1  train 338.97  val 71.74
```
Bundle saved OK. This only verifies plumbing/gradient flow — **not** model
quality (random init, tiny counts). The GPU run is the real thing.

## Output / checkpoint format

`--save_path/best_positional_harmonic_gen.pt` is a **bundle** (not a bare
state_dict):
```python
import torch
from train_noise_generation import get_model
from tasks.noise_generation import DroneCodebook
b = torch.load(path, map_location="cpu", weights_only=False)
# b: {"model", "codebook", "cond_dim", "drone_names"}
model = get_model("positional_harmonic_gen", sample_rate=16000, n_harmonics=100, cond_dim=b["cond_dim"])
model.load_state_dict(b["model"])
cb = DroneCodebook(b["cond_dim"], names=b["drone_names"]); cb.load_state_dict(b["codebook"])
# render: z = cb(["dregon"]);  audio = model(rps, rel_pos, z)
```

## Gotchas (the "without thinking twice" list)

- **Michael's `min_motor_rps: 30`** in the YAML is load-bearing — FLY124 has
  idle/zero-RPS regions; `0.0` lets all-silent slices into validation. Don't
  lower it.
- **DREGON source needs `download: false`** (already in the YAML). The loader's
  default tries an SSL download that fails on the cluster.
- **`--samples_per_epoch` defines the "epoch"** (the stream is infinite);
  `--num_valid` validation slices are fixed/deterministic (seed = base_seed+1).
  Early-stopping/scheduler key off the fixed valid set.
- **Loss is multi-scale STFT magnitude** (mic axis folded into batch). It is blind
  to a common delay but sees inter-rotor delay differences (the geometric signal).
  Train vs val numbers aren't directly comparable (val recordings differ from
  train); watch the **val curve trend**, not the absolute gap.
- **Few-shot adaptation** to an unseen drone is wired: train a base model, then
  `--init_checkpoint <bundle> --freeze_emitter --cond_dim 16` with a single-drone
  online config fits just that drone's code.
- Bump `--time`/`--num_workers` as needed; `sae` max wall is 10 days. Use
  `--partition=gpushort --time=1:00:00` for a quick GPU smoke first if unsure.

## Code pointers

- Model: `src/models/generative/positional_harmonic_gen.py`
- Codebook + task: `src/tasks/noise_generation.py` (`DroneCodebook`)
- Training + online datasets: `train_noise_generation.py`
  (`OnlineNoiseGenDataset`, `FixedNoiseGenDataset`, `_noise_item`,
  `build_noise_pools`)
- Config: `configs/noise_gen_online_dregon_michaels.yaml`
- Tests: `tests/train/test_noise_generation.py`, `tests/models/test_positional_harmonic_gen.py`
- Task doc: `src/tasks/noise-generation/AGENTS.md`
- Commits: `a253d23` (task+model+geometry), `dd11881` (external codebook),
  `50ca9a4` (geometry edge-offset fix), `250e7a2` (online streaming).

## What's next (after the GPU run)

- Listen to rendered noise per drone (`model(rps, rel_pos, cb([name]))`) vs the
  held-out targets; check whether the two codes diverge meaningfully.
- Sweep `cond_dim` (8/16/32) and `n_harmonics`; try `--no_diff_noise` ablation.
- Add more drones / recordings to the YAML — no model change needed (codebook
  grows by name).
- Evaluate the few-shot path: freeze emitter, adapt a new drone's code from a few
  minutes of audio.
