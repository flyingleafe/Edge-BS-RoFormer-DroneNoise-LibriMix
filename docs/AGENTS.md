# docs/ — Design Documentation

Contains design specs, debugging guides, and research notes. Not auto-generated — maintained manually.

## Why this directory exists

Long-form documentation that doesn't belong in code comments or AGENTS.md files. Architecture specs, paper notes, and debugging guides live here.

## Contents

| File/Dir | Purpose |
|----------|---------|
| `debug-training-loop.md` | Training loop debugging guide — read this before debugging training failures |
| `data-and-artifacts.md` | DVC + wandb Artifacts workflow (datasets → R2, checkpoints → wandb), per-machine setup |
| `dcunet-refactored.md` | Design notes for DCUNet/DCCRN refactoring (Paper 2) |
| `diffusion-buffer-paper.md` | Notes on the diffusion buffer paper |
| `diffusion-prompt.md` | Prompt used to implement the diffusion buffer model |
| `fwh_rotor_acoustic_simulator_plan.md` | FWH rotor acoustic simulator design + implementation status |
| `experiments/` | One doc per past experiment (motivation/results/conclusion); live PhD-bet detail cards under `experiments/bets/` — see `docs/experiments/AGENTS.md` |
| `superpowers/` | Postdoc platform architecture specs and implementation plans |

## Gotchas

- `superpowers/specs/` contains design documents that may describe features not yet implemented
- When in doubt about postdoc behavior, read source code in `src/postdoc/` first, then check `debug-training-loop.md` for training issues