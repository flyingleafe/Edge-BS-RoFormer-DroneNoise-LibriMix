# src/tasks/ — Task Definitions

A **task** is the function type of a model over `tdseries.Frame`s: which
entries it consumes, which it produces, and at which rates. `tasks/task.py`
declares them as `Task(name, input_spec, output_spec)` built by parameterized
factories (`TASK_FACTORIES`); `tasks/codecs.py` gives each one a `Codec` — the
`to_inputs` / `call_model` / `to_frame` seam between the Frame world and a
plain `nn.Module`. `training.config.build_task_and_codec` builds both from the
one `task` + `task_params` block of `conf/model/<name>.yaml`, so the spec and
the adapter can never drift apart.

The per-task subdirectories describe the model interface, training
integration, code placement, and existing implementations. Read the relevant
one when reimplementing a paper model — it defines the contract the model must
satisfy.

## Available tasks

Every task name below is a key in **both** `tasks.task.TASK_FACTORIES` and
`tasks.codecs.CODEC_FACTORIES`, and is what a model config's `task:` field
names.

| Task | Directory | Model interface | Notes |
|------|-----------|-----------------|-------|
| `speech_enhancement` | `speech-enhancement/` | `forward(x, rps=None) → (B, 1, T)` or `(enhanced, rps_pred)` | `mixture` (+ optional `rps`) in, `enhanced` (+ optional `rps_pred`) out. Paper 1 + Paper 2 families |
| `rps_prediction` | `rps-prediction/` | `forward(audio) → (B, 4, T_stft)` | Models in `src/models/registry.py::RPS_MODEL_REGISTRY`. `use_cond=True` is the conditional-refiner variant: `forward(audio, rps_cond)`, plain (non-PIT) loss |
| `salience_rps` | — (see below) | `forward(audio) → (B, F, T)` logits | Mixture in, frequency-salience logits out; RPS is derived at eval time by Hungarian tracking. BCE-on-salience training. Configs `conf/model/{multif0,basic_pitch}_salience*.yaml` |
| `noise_generation` | `noise-generation/` | `forward(rps, rel_pos) → (B, M, T)` | RPS + array geometry in, per-mic drone noise out. `distributional`/`spatial` widen the output contract (mean + PSD envelopes) for the likelihood objectives |

`salience_rps` has no subdirectory of its own: it is a *readout* variant of
RPS prediction (same input, same data, different output head and loss), so
`rps-prediction/AGENTS.md` covers it — see its "Existing implementations"
table and `src/models/salience_rps.py`.

Training entry point for all four: `python train.py experiment=<name>`.

## Files

| File | Purpose |
|------|---------|
| `task.py` | `Task` + the four factories + `TASK_FACTORIES` |
| `codecs.py` | One `Codec` per task + `CODEC_FACTORIES` / `build_codec` |

| `checkpoints.py` | `load_model("Type@/path/ckpt.pt")` — bare `state_dict` loading against `RPS_MODEL_REGISTRY`. For experiment-level loading (config + R2 resolution + codec) use `zoo.load` instead |
| `rps_prediction.py` | The RPS evaluation pipeline `evaluate-rps` calls: `load_predictor`, `load_input_set`, `evaluate`, `align_rps_to_gt` |
| `classical_rps_predictors.py` | Non-learned RPS baselines (`cepstral`, `hps`, `pyin`, `nmf`, `matched_filter`) |
| `noise_generation.py` | Noise-generation task helpers (`geometry_to_rel_pos`, …) |
| `cli.py` | The `evaluate-rps` typer CLI (`pyproject.toml` `[project.scripts]`) |

## Adding a task

1. Add a factory to `tasks/task.py` + `TASK_FACTORIES`, and its `Codec` to
   `tasks/codecs.py` + `CODEC_FACTORIES` — with the **same** keyword
   parameters, so one `task_params` dict builds both.
2. Create `src/tasks/<task-name>/AGENTS.md` documenting the model interface,
   training integration, code placement, and front-end conventions.
3. Add an entry to the table above.
