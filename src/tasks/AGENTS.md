# src/tasks/ — Task Definitions

Each subdirectory describes one ML task this project addresses: the model
interface, training integration, code placement, and existing implementations.

Use these when reimplementing a paper model — the task description defines
the target contract the model must satisfy.

## Available tasks

| Task | Directory | Training script | Model interface |
|------|-----------|-----------------|-----------------|
| RPS Prediction | `rps-prediction/` | `train_rps_predictor.py` | `forward(audio) → (B, 4, T_stft)` |
| Noise Generation | `noise-generation/` | `train_noise_generation.py` | `forward(rps, rel_pos) → (B, M, T)` |

## Adding a task

1. Create `src/tasks/<task-name>/AGENTS.md`.
2. Document the model interface, training integration, code placement,
   and front-end conventions.
3. Add an entry to the table above.
