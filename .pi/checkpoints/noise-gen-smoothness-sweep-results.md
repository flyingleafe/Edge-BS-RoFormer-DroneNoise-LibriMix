# Checkpoint: Noise-Gen Smoothness Sweep — Results & Colleague Handoff

**Date**: 2026-07-02 · **Branch**: `main` · **Status**: sweep complete (9 runs); one combined run in progress

_Read [`noise-gen-swapped-smoothness-random-phase.md`](noise-gen-swapped-smoothness-random-phase.md) first for the model/data/random-phase background._

## What we did

Ran a 1D sweep over `--harm_smooth_weight` and `--noise_smooth_weight` independently (the two regularisers act on orthogonal quantities — harmonic amplitude curves over time vs. diffuse-noise filter shape over time+freq — so independent sweeps are valid).

**Common args**: `--online_config configs/noise_gen_online_dregon_michaels_swapped.yaml --cond_dim 16 --epochs 50 --patience 10 --batch_size 32 --n_harmonics 100 --samples_per_epoch 500 --num_valid 128 --num_workers 8`

**TL;DR**: `harm_smooth_weight=1e-1` is the nominal winner (val: 5.3506), but the top 3 are within ~0.008 of each other — essentially noise-level. Smoothness doesn't meaningfully move the needle on raw spectral val loss.

## Results

| Rank | `--harm_smooth_weight` | `--noise_smooth_weight` | **Best Val** | Epochs | Time |
|------|------------------------|--------------------------|-------------|--------|------|
| 🥇 | **1e-1** | 0 | **5.3506** | 50 | 21.5 min |
| 🥈 | 0 (baseline) | 0 | 5.3554 | 50 | 21.1 min |
| 🥉 | 0 | **10** | 5.3581 | 49 | 21.1 min |
| 4 | 0 | 1e-1 | 5.3882 | 37 | 15.5 min |
| 5 | 1e-2 | 0 | 5.4302 | 49 | 20.5 min |
| 6 | 10 | 0 | 5.4465 | 39 | 16.5 min |
| 7 | 0 | 1 | 5.4812 | 27 | 11.3 min |
| 8 | 1 | 0 | 5.5171 | 30 | 12.6 min |
| 9 | 0 | 1e-2 | 5.5988 | 24 | 10.4 min |

**Combined run in progress**: `harm_smooth_weight=1e-1 --noise_smooth_weight=10` at `/gpfs/scratch/acw592/results/noise_gen_sweep/harm_1e-1_noise_10/`.

### Takeaways

- **Harm smoothness** peaks at 1e-1. Lower (1e-2) does little; higher (1, 10) over-smooths and hurts.
- **Noise smoothness** needs large weights (>1) to help. 1e-2 *actively hurts* (5.60 vs baseline 5.36). At weight=10 it nearly matches baseline.
- Neither regulariser provides a *meaningful* val-loss improvement. But the real question is **qualitative**: does smoothness make the harmonic/noise outputs "cleaner" or more physically plausible? That's what the colleague analysis below is for.

## For the colleague — what to analyse

### The question

The raw spectral val loss doesn't tell the full story. Smoothness might not improve the summed output spectrally, but it could make the **harmonic** and **noise** components individually more reasonable:
- Are harmonic amplitudes smoother over time (less flicker)?
- Is the filtered-noise shape more temporally/frequency-stable?
- Do the rendered harmonic-only and noise-only outputs sound more natural?

### How to separate harmonic and noise components

The model emits both components but sums them for the final output. To get them separately, call the emitter with `return_dict=True`:

```python
import torch
from src.models.generative.positional_harmonic_gen import PositionalHarmonicNoiseGen
from tasks.noise_generation import DroneCodebook

CKPT = "/gpfs/scratch/acw592/results/noise_gen_sweep/baseline/best_positional_harmonic_gen.pt"
bundle = torch.load(CKPT, map_location="cpu", weights_only=False)

model = PositionalHarmonicNoiseGen(
    cond_dim=bundle["cond_dim"],
    n_harmonics=...,  # was 100 for these runs
    sample_rate=16000,
)
model.load_state_dict(bundle["model"])
model.eval()

codebook = DroneCodebook(cond_dim=bundle["cond_dim"], drone_names=bundle["drone_names"])
codebook.load_state_dict(bundle["codebook"])

# Get per-rotor emitted signals (harmonic + noise summed)
# rps: [B, R, T]  — from the online mixer / TimeFrameNoisePool
# rel_pos: [B, M, R, 3] — mic-to-rotor vectors
# z: [B, cond_dim] — drone code from codebook

with torch.no_grad():
    out = model(rps, rel_pos, z=z, return_dict=True, initial_phases=torch.zeros(...))
# out["audio"]     — [B, M, T] summed at each mic
# out["sources"]   — [B, R, T] per-rotor (harmonic + noise summed)
# out["harm_amps"] — [B, R, O, H, t_a] harmonic amplitude curves (the smoothness target)
# out["noise_amps"]— [B, R, F, t_n] filter curves (the smoothness target)

# To get harmonic-only and noise-only audio separately,
# call the inner emitter directly:
emitted = model.emit(rps, z=z, return_dict=True, initial_phases=torch.zeros(...))
# emitted["sources"]  — [B, R, T] per-rotor (harmonic + noise summed)

# For the raw per-rotor components before propagation:
inner_out = model.emitter(ms_rps, z=z_folded, return_dict=True)
# inner_out["harm_noise"] — [B*R, O, H, T] harmonic oscillator bank output
# inner_out["diff_noise"] — [B*R, T] filtered noise output
# inner_out["harm_amps"]  — [B*R, O, H, t_a] amplitude curves
# inner_out["noise_amps"] — [B*R, F, t_n] filter curves
# inner_out["audio"]      — [B*R, T] harmonic + noise summed
```

**Key**: `model.emitter.forward(rps_folded, return_dict=True)` gives `harm_noise` (oscillator bank) and `diff_noise` (filtered white noise) **as separate waveforms**. These are the raw per-rotor components before 1/r propagation.

See the existing analysis notebook for the full loading/data pipeline:
`notebooks/noise_gen_real_vs_generated.ipynb` — already loads checkpoints, extracts RPS from TimeFrames, renders audio, computes spectrograms.

### What to look for

1. **Harmonic amplitudes over time** (`harm_amps`): plot a few harmonics from baseline vs. harm=1e-1 — is the envelope smoother (less high-frequency flicker)?
2. **Noise filter shape** (`noise_amps`): plot the filter magnitude over time for a few frequency bands — baseline vs. noise=10 — is it more temporally stable?
3. **Harmonic-only audio**: extract `harm_noise.sum(dim=-2)` (sum over oscillators and harmonics) and listen — does it sound cleaner?
4. **Noise-only audio**: extract `diff_noise` and listen — is the broadband residual less "gurgly"?
5. **Compare baseline vs harm=1e-1 vs noise=10 vs combined (harm=1e-1+noise=10)** on the same test slices from DREGON room1 + FLY124.

### Checkpoint locations

All at `/gpfs/scratch/acw592/results/noise_gen_sweep/`:

| Directory | Smoothness | Best Val |
|-----------|------------|----------|
| `baseline/` | none | 5.3554 |
| `harm_1e-1/` | harm=1e-1 | 5.3506 |
| `harm_1e-2/` | harm=1e-2 | 5.4302 |
| `harm_1/` | harm=1 | 5.5171 |
| `harm_10/` | harm=10 | 5.4465 |
| `noise_1e-2/` | noise=1e-2 | 5.5988 |
| `noise_1e-1/` | noise=1e-1 | 5.3882 |
| `noise_1/` | noise=1 | 5.4812 |
| `noise_10/` | noise=10 | 5.3581 |
| `harm_1e-1_noise_10/` | combined ⏳ | (running) |

Each has `best_positional_harmonic_gen.pt` (bundle: `model`, `codebook`, `cond_dim`, `drone_names`).

## Code pointers

- Sweep results + training logs: `/gpfs/scratch/acw592/results/noise_gen_sweep/`
- Slurm logs: `/gpfs/scratch/acw592/logs/ng_*.o*`
- WandB project: `flyingleafe/noise-generation` (runs named `positional_harmonic_gen_DREGON-LM-V4`)
- Config: `configs/noise_gen_online_dregon_michaels_swapped.yaml`
- Training + smoothness loss: `train_noise_generation.py`
- Emitter (harmonic/noise synthesis): `src/models/generative/harmonic_gen_new.py`
- Positional propagation model: `src/models/generative/positional_harmonic_gen.py`
- Smoothness penalty: `src/models/generative/losses.py` (`smoothness_penalty`)
- Analysis notebook: `notebooks/noise_gen_real_vs_generated.ipynb`

## Gotchas

- `data/recording_with_motor_speed` → `/gpfs/scratch/acw592/data/recording_with_motor_speed` (symlink created 2026-07-02; was missing on first submit attempt)
- Checkpoints are PyTorch bundles: `{"model": state_dict, "codebook": state_dict, "cond_dim": int, "drone_names": list}`
- `model.eval()` → zero-phase deterministic synthesis; `model.train()` → random phases
- `min_motor_rps: 30.0` in config is load-bearing for Michael's data
- `--noise_smooth_weight` is ignored if `--no_diff_noise` is set (no noise branch)
