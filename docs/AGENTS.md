# docs/ — Design Documentation

Contains design specs, debugging guides, and research notes. Not auto-generated — maintained manually.

## Why this directory exists

Long-form documentation that doesn't belong in code comments or AGENTS.md files. Architecture specs, paper notes, and debugging guides live here.

## Contents

| File/Dir | Purpose |
|----------|---------|
| `debug-training-loop.md` | Training loop debugging guide — read this before debugging training failures |
| `data-and-artifacts.md` | dload + wandb Artifacts workflow (datasets → R2, checkpoints → wandb), per-machine setup |
| `dcunet-refactored.md` | Design notes for DCUNet/DCCRN refactoring (Paper 2) |
| `diffusion-buffer-paper.md` | Notes on the diffusion buffer paper |
| `diffusion-prompt.md` | Prompt used to implement the diffusion buffer model |
| `fwh_rotor_acoustic_simulator_plan.md` | FWH rotor acoustic simulator design + implementation status |
| `koopman-and-order-tracking-ideas.md` | Literature survey: Koopman operators + modal-synthesis for bidirectional audio↔RPS latent states; Vold–Kalman aeroacoustics literature and "VK in reverse" (blind IF estimation) — cross-references `vk-order-tracking-design.md`'s outer loop |
| `experiments/` | One doc per past experiment (motivation/results/conclusion); live PhD-bet detail cards under `experiments/bets/` — see `docs/experiments/AGENTS.md` |

## Gotchas

- When debugging training issues, read `debug-training-loop.md` first, then the source in `src/training/`