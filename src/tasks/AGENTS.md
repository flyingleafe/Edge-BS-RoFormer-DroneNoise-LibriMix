# src/tasks/ — Task Definitions

Each subdirectory describes one ML task this project addresses: the model
interface, training integration, code placement, and existing implementations.

Use these when reimplementing a paper model — the task description defines
the target contract the model must satisfy.

## Available tasks

| Task | Directory | Training entry point | Model interface |
|------|-----------|-----------------------|-----------------|
| RPS Prediction | `rps-prediction/` | `train.py` (Hydra; models `src/models/registry.py::RPS_MODEL_REGISTRY`) | `forward(audio) → (B, 4, T_stft)` |
| Noise Generation | `noise-generation/` | `train.py` (Hydra; models `src/models/registry.py::build_noise_gen_model`; `conf/experiment/e2_noise_gen_dregon_michaels.yaml`/`e3_noise_gen_swapped_smoothness.yaml`, see REPLICATION.md § E2/E3) | `forward(rps, rel_pos) → (B, M, T)` |

## Adding a task

1. Create `src/tasks/<task-name>/AGENTS.md`.
2. Document the model interface, training integration, code placement,
   and front-end conventions.
3. Add an entry to the table above.
