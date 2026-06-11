---
name: reimplement-model
description: Reimplement a paper model into the project framework. Routes through task-specific interface descriptions before proceeding to paper analysis, source-code study, model design, and implementation.
---

# Reimplement a Paper Model

End-to-end workflow for porting a model from a research paper into the
project's training/evaluation framework.

## When to use

- User asks to implement / reproduce / port a model from a paper
- User mentions a model by paper name or author+year
- User wants a new baseline for an existing task

## Workflow

### 1. Identify the task

Ask the user or infer from context which **task** this model serves:
- `rps-prediction` — rotor speed from drone noise
- `speech-enhancement` — clean speech from noisy mixture
- `multi-f0` — multi-pitch estimation (original paper task)

Read the task description at `src/tasks/<task-name>/AGENTS.md`.  This defines:
- The model interface (input/output shapes, required constructor params)
- How the model plugs into training (`train_rps_predictor.py`, `train.py`, …)
- Where to place code (`src/models/<subdir>/`)
- Existing implementations for reference

### 2. Study the paper

1. **Find the paper**. Search arXiv / Google Scholar / Semantic Scholar.
   If the user provided a title, use it directly.
2. **Read the architecture section**. Focus on:
   - Input representation (raw audio? spectrogram? CQT? MFCC?)
   - Model architecture (CNN? RNN? Transformer? hybrid?)
   - Output format (frame-level? sequence? single vector?)
   - Any special training procedures (pre-training, losses, augmentation)
3. **Identify the front-end**. Does the model compute its own TF transform,
   or accept pre-computed features?  Map to a `SpectralFrontEnd` if it's a
   standard transform (STFT, CQT, mel).  If it's novel, plan a new front-end.

### 3. Find and study source code

1. Search GitHub / paper website for official implementation.
2. If none exists, check Papers With Code, TensorFlow Hub, HuggingFace.
3. Read the model definition (usually `model.py` or `models.py`).
   Key things to extract:
   - Exact layer dimensions, kernel sizes, strides, padding
   - Activation functions, normalization types
   - Any non-obvious details (custom padding, weight tying, initialization)
4. Note the **exact input preprocessing** — normalization, feature scaling,
   frame centering.  This is the #1 source of silent degradation.

### 4. Design the implementation

Map the paper model onto the task interface from step 1.

**Front-end decision tree:**
```
Does the model use a standard TF transform inline?
├─ STFT → use existing STFTMag / STFTMagPhase front-end
├─ CQT / HCQT → use existing HCQTFrontEnd, or add variant
├─ Mel spectrogram → add MelFrontEnd (new, subclass SpectralFrontEnd)
└─ Novel / learned → inline in model, document why
```

**Model structure:**
- Core model: `src/models/<descriptive_name>/model.py`
- RPS/task wrapper (if needed): `src/models/<descriptive_name>/rps_predictor.py`
- Tests: root-level `test_<name>.py` or inline in the model file
- Import path: `from models.<descriptive_name> import YourModel`

**Checklist before coding:**
- [ ] Input shape matches what the dataset provides
- [ ] Output shape matches the task interface
- [ ] Front-end choice documented and justified
- [ ] Non-standard dependencies identified (nnAudio, torchaudio, …)
- [ ] Pre-trained weights available? (plan checkpoint loading)

### 5. Implement

**Order of implementation (test at each step):**

1. **Minimum forward pass** — inputs → outputs with correct shapes.
   No training, just instantiate and forward.
2. **Front-end parity** (if applicable) — compare against paper's reference
   implementation or librosa.  Bit-exact or within numerical tolerance.
3. **Registration** — add to `MODEL_REGISTRY` in the appropriate training
   script.  Test: `model = get_model("key"); out = model(audio)`.
4. **Gradient flow** — one optimizer step, verify loss decreases.
5. **Full training** — one epoch, verify loss curve, no NaN/exploding.
6. **Baseline comparison** — evaluate against existing models on the task.

**Coding conventions:**
- Follow existing patterns in `src/models/rps_predictor.py` (constructor
  params, front-end default construction, forward signature).
- Use `nn.Sequential` for simple blocks; `nn.ModuleList` only when iterating
  with different operations per layer.
- Document paper reference in module docstring: title, authors, venue, year.
- Add GPU-compatible alternatives for any CPU-only bottleneck (e.g. librosa
  → nnAudio for CQT).

### 6. Verify & document

- [ ] `import` sweep: all new modules importable via `from models.X`
- [ ] Unit tests pass (at minimum: shape test, grad flow test)
- [ ] Model appears in registry and can be selected via CLI
- [ ] Task description updated with the new model entry
- [ ] `src/models/AGENTS.md` updated if adding a new subdirectory

## Key Principles

- **Front-end first.** Before writing the model, settle the representation.
  A model that bakes in STFT can't be compared against one that uses HCQT.
- **Test smallest unit first.** Forward pass shape → grad flow → training step
  → full epoch.  Don't debug a training loop to find a shape bug.
- **Duplicate exactly, then adapt.** Implement the paper model *verbatim*
  first (same layers, same dims, same activations).  Verify against reference
  output.  Only then adapt to the task interface.
- **Surface assumptions.** If the paper assumes 44.1 kHz and we use 16 kHz,
  note it.  If the paper's dataset has 10s clips and ours has 3s, note it.
  These are the places where silent performance gaps hide.

## References

- Task descriptions: `src/tasks/<task-name>/AGENTS.md` (index at `src/tasks/AGENTS.md`)
- Front-end system: `src/models/frontends/`
- Existing RPS models: `src/models/rps_predictor.py`, `src/models/multif0/`
- Front-end contract: see `src/models/frontends/__init__.py` (SpectralFrontEnd ABC)
