# Task: RPS Prediction

Predict rotor speed (RPS, in Hz) from drone noise audio on a per-STFT-frame
grid.  Input is raw audio, output is a 4-rotor RPS trajectory resampled to the
STFT hop grid.

## Model interface

```python
class RPSPredictor(nn.Module):
    def forward(self, audio: Tensor) -> Tensor:
        """
        audio: (B, N) or (B, 1, N) — raw waveform at 16 kHz
        returns: (B, 4, T_stft) — RPS (Hz) for 4 rotors per STFT frame
        """
```

- **Input rate**: 16 kHz (DREGON-LM standard).
- **Output grid**: STFT time frames — `T_stft = N // hop_length + 1`, default
  `hop_length=512`.
- **Output channels**: 4 rotors.  Broadcasting from single prediction to 4
  channels is acceptable; per-rotor prediction is better.  Loss is MSE or
  PIT-MSE (respects permutation invariance of rotor labels).
- **Required constructor params**: `n_fft`, `hop_length`, `num_rotors` (the
  factory `get_model(name, n_fft, hop_length, num_rotors)` passes these).

## Training integration

### Registry
Add the model class to `RPS_MODEL_REGISTRY` in `src/models/rps_predictor.py`
(or wherever the class lives), then re-export it from
`src/models/registry.py::RPS_MODEL_REGISTRY` — the single registry the
unified `train.py`/`eval.py` (Hydra) and `tasks.checkpoints.load_model` both
resolve model names against:

```python
RPS_MODEL_REGISTRY = {
    ...
    "your_model_key": YourModelClass,
}
```

`src/models/registry.py::build_model(name, **params)` constructs
`YourModelClass(**params)`; `get_rps_model(model_name, n_fft, hop_length,
num_rotors, ...)` is the richer factory that also handles the salience-model
narrow-input/super-resolution config overrides.

### Dataset
`DREGONRPSDataset` loads `mixture.wav` + `rps.npy` from DREGON-LM chunks.
Dataloaders for this task should expose the common `(audio, rps_target)` format:
audio is `(T,)` or `(C, T)`, and `rps_target` is `(4, T_stft)` on the model's
STFT output grid.  Salience-map models do **not** require a special dataset item;
the training loop derives their BCE salience targets on the fly from `rps_target`.
For online mixing, use an infinite `IterableDataset`, not a finite map-style
`Dataset`: the training loop defines an arbitrary validation cadence such as
`samples_per_validation`, consumes that many online samples, then evaluates on a
fixed validation set. The public interface is config-in/stream-out: do not add
separate source-cache preparation scripts or cache-specific CLI flags; if a
cache/memmap is needed and missing, create it under the hood from the same YAML
config. Do not invent per-segment data containers: use existing
`TimeFrame`/`TimeSeries` objects for aligned audio+telemetry and plain NumPy
arrays/memmaps/tensors for simple unaligned audio; prefer OmegaConf YAML nodes
or plain mappings over custom config dataclasses.

### Loss
Training loop calls `model(audio)` to get `rps_pred: (B, 4, T)` and computes
MSE against `rps_gt: (B, 4, T)`.  PIT-MSE variant also supported
(`--loss pit_mse`).

## Code placement

- **Core model code**: `src/models/<subdir>/` (e.g. `src/models/multif0/`).
  Importable as `from models.<subdir> import YourModel`.
- **RPS wrapper** (if model needs adaptation to RPSPredictor interface):
  `src/models/<subdir>/rps_predictor.py`.
- **Front-end** (if new TF transform needed): `src/models/frontends/<name>.py`,
  subclass `SpectralFrontEnd`, decorate with `@register_frontend`.
  Existing: `stft_mag`, `stft_magphase`, `hcqt`.

## Front-end integration

If the model needs a specific TF representation, use the pluggable front-end
system:

```python
from models.frontends import build_frontend

class YourModel(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, num_rotors=4, frontend=None):
        ...
        if frontend is None:
            frontend = build_frontend("stft_mag", n_fft=n_fft, hop_length=hop_length)
        self.frontend = frontend

    def forward(self, audio):
        x = self.frontend(audio)  # (B, C, F, T)
        ...
```

Existing models use this pattern (see `src/models/rps_predictor.py` SimpleConv*
family).  The front-end handles STFT/HCQT/whatever; the model only sees a
`(B, C, F, T)` tensor.

## Existing implementations

| Model | Key | Front-end | Approach |
|-------|-----|-----------|----------|
| SimpleConv family | `simple_conv` … | STFT mag (1ch) or mag+phase (3ch) | 2D CNN encoder → frequency pool → Conv1d/BiGRU/TCN head |
| DCUNetEncRPS | `dcunet_enc_rps` | inline complex STFT | DCUNet complex-conv encoder → mean pool → Conv1d head |
| MultiF0RPSPredictor | `multif0_rps` | HCQT (10ch) | LateDeep CNN → sigmoid salience → soft-centroid → RPS |

## Evaluation

Use `notebooks/eval_rps_predictor.ipynb` (or generate model comparisons via
the `generate-model-comparisons` skill).  Metrics: MAE, RMSE of predicted
RPS vs ground truth.

---

## Checklist for a new RPS predictor implementation

1. [ ] Read the paper and find official source code.
2. [ ] Understand the input → output mapping: raw audio → prediction.
3. [ ] If the model does TF analysis internally, decide: extract as front-end
       or keep inline (prefer front-end if it's a standard transform).
4. [ ] Implement core model in `src/models/<subdir>/`.
5. [ ] Wrap to `RPSPredictor` interface if needed.
6. [ ] Register in `src/models/registry.py::RPS_MODEL_REGISTRY`.
7. [ ] Smoke test: `model = build_model("key"); out = model(audio); assert out.shape == (B, 4, T_stft)`.
8. [ ] One-epoch training run to verify gradient flow and loss decrease.
9. [ ] Evaluation against baseline metrics.
