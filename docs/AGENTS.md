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
| `koopman-and-order-tracking-ideas.md` | Literature survey: Koopman operators + modal-synthesis for bidirectional audio↔RPS latent states; Vold–Kalman aeroacoustics literature and "VK in reverse" (blind IF estimation) — cross-references `vk-order-tracking-design.md`'s outer loop |
| `rps-tracking-architecture-candidates.md` | 2026-09-03 synthesis: the measured structure of drone ego-noise, what each model family does with it, the seven walls, the design requirements, and the candidate architectures for tracking variable frequencies from partially observed harmonics, with the day's probes |
| `experiments/` | One doc per past experiment (motivation/results/conclusion); live PhD-bet detail cards under `experiments/bets/` — see `docs/experiments/AGENTS.md` |

## Removed components

- `src/fwh_rotor_sim` (FWH rotor acoustic simulator, BEMT + Farassat 1A) was removed on 2026-08-04 during the repo refactor, together with its plan doc (`fwh_rotor_acoustic_simulator_plan.md`), its notebook, and the `load-real-propeller-geometry` skill. The external repo <https://github.com/flyingleafe/auraflow> replaces it. The last in-repo version is available in the git history.

## Gotchas

- When debugging training issues, read `debug-training-loop.md` first, then the source in `src/training/`