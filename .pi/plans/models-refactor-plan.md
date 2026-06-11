# Models Refactoring Plan — `src/models/` + pluggable spectral front-ends

**Status:** proposed
**Author:** agent (with user)
**Scope:** relocate all model code under `src/models/`; introduce a swappable
TF-domain feature front-end shared by the SimpleConv (STFT) family and the
multi-F0 (HCQT) baseline; keep SimpleConv checkpoints loadable.

---

## 1. Goals (from the request)

1. **One home for models.** All model code lives under `src/models/`; every
   consumer imports `from models.…` (resolved to `src/models/`), not
   `from src.models.…` and not a root-level `models/`.
2. **Pluggable spectral front-end.** The TF-domain feature extractor (STFT now;
   CQT/HCQT increasingly) is a *component*, not hardcoded in each `forward()`.
   We must be able to hold the model fixed and swap the representation, to
   measure how the representation alone moves performance.
   **Hard constraint: SimpleConv variants must keep loading their old
   checkpoints** (bit-identical features by default).
3. **Multi-F0 baseline as a first-class RPS predictor.** Already trainable via
   `train_rps_predictor.py`. Make its HCQT parameters configurable and let its
   front-end be swapped for STFT / another TF extractor, same as (2).

---

## 2. Current state (interfaces first)

### 2.1 Two model trees
| Tree | Contents | Imported as |
|------|----------|-------------|
| root `models/` | all enhancement models (`dcunet`, `dccrn`, `edge_bs_rof`, …), `rps_predictor.py` (10 SimpleConv variants), `generative/` | `from models.X` — works only because cwd is on `sys.path` (namespace pkg, `__file__ is None`) |
| `src/models/multif0/` | HCQT, 4 CNN variants, `MultiF0RPSPredictor`, nnAudio CQT | `from src.models.multif0.X` |

`src/utils`, `src/postdoc`, `src/tasks`, `data_processing` are **editable-installed
packages** (`[tool.hatch.build.targets.wheel].packages`). `src/models` is *not* in
that list, hence the asymmetric import path.

### 2.2 Feature extraction is inlined in every model
Every SimpleConv `forward()` repeats:
```python
X = torch.stft(audio, n_fft, hop, window=self.window,
               return_complex=True, normalized=True)
mag = torch.log1p(X.abs()).unsqueeze(1)        # (B,1,F,T)
```
`SimpleConvMagPhaseBiGRU` instead stacks `[mag, cos, sin]` → `(B,3,F,T)`.
`MultiF0RPSPredictor._hcqt_features` calls **librosa per-sample on CPU** and the
CNN takes a **pair** `(mag, dphase)` each `(B, H, F, T)`.

State that lands in a SimpleConv checkpoint: `encoder.*`, `head.*` (learned) +
`window` (a registered, non-learned hann buffer).

### 2.3 Consumers (blast radius)
- `src/utils/__init__.py::get_model_from_config` — ~20 `from models.X import`
- `train.py` → `utils.get_model_from_config`
- `train_rps_predictor.py` → `from models.rps_predictor import …` **and**
  `from src.models.multif0.rps_predictor import MultiF0RPSPredictor`
- `src/tasks/checkpoints.py` → `from train_rps_predictor import MODEL_REGISTRY`,
  `from models import MODEL_TYPES`
- `slides/2026-06-02-rps-progress/*.py` → `from models…`

---

## 3. Target architecture

### 3.1 Directory layout
```
src/models/
  __init__.py            # re-exports get_model_from_config, MODEL_TYPES, registries
  frontends/
    __init__.py          # FRONTEND_REGISTRY + build_frontend(name, **kw)
    base.py              # SpectralFrontEnd ABC
    stft.py              # STFTMag, STFTMagPhase
    hcqt.py              # HCQTFrontEnd (librosa ref + nnAudio GPU path)
  rps/
    __init__.py          # RPS_MODEL_REGISTRY, get_rps_model
    simple_conv.py       # the 10 SimpleConv* variants (moved verbatim, then re-wired)
    multif0.py           # MultiF0RPSPredictor (re-wired onto FrontEnd)
  multif0/               # the multi-F0 *base* CNN + nnAudio CQT (unchanged science)
    model.py  hcqt.py  nnaudio_cqt.py  gpu_cqt.py
  enhancement/           # dcunet.py dccrn.py edge_bs_rof/ dptnet/ … (moved verbatim)
  generative/            # moved verbatim
```
The exact sub-foldering of `enhancement/` is mechanical; the **import name stays
`models`** so no `from models.dcunet import …` line changes.

### 3.2 Make `src/models` importable as `models`
1. Add `"src/models"` to `[tool.hatch.build.targets.wheel].packages`.
2. **Move** root `models/*` → `src/models/` (a move, not a copy — two dirs both
   answering to `models` is the one thing we must not ship).
3. `uv sync` / editable reinstall so the `.pth` maps `src/models` → `models`.
4. Net import churn: only the multif0 call-sites that said `from src.models.…`
   become `from models.…`. The ~20 `from models.X` in `src/utils` are untouched.

### 3.3 The front-end contract
```python
class SpectralFrontEnd(nn.Module):
    out_channels: int                      # C the model will receive
    def forward(self, audio: Tensor) -> Tensor:   # (B, N) -> (B, C, F, T)
        ...
    def num_frames(self, n_samples: int) -> int:  # time-grid length, for alignment
        ...
```
A front-end returns **one** `(B, C, F, T)` tensor. Channel semantics are the
front-end's contract, not the model's:

| Front-end | `out_channels` | F | Reproduces |
|-----------|----------------|---|------------|
| `STFTMag` | 1 | `n_fft//2+1` | current SimpleConv input (log1p mag) |
| `STFTMagPhase` | 3 | `n_fft//2+1` | `SimpleConvMagPhaseBiGRU` (mag,cos,sin) |
| `HCQTFrontEnd(phase=False)` | `H` | 360 | multif0 mag-only |
| `HCQTFrontEnd(phase=True)` | `2H` | 360 | multif0 mag+dphase, stacked on C |

The multi-F0 CNN currently takes a **pair** `(mag, dphase)`. We adapt
`MultiF0RPSPredictor` (not the base `LateDeep`) to split the `2H` channel axis
back into `(mag, dphase)` before calling the CNN — the base science model is
left byte-identical, the splitting lives in the RPS wrapper.

### 3.4 Model contract
Models stop owning `torch.stft`. They accept a built front-end:
```python
class SimpleConv(nn.Module):
    def __init__(self, frontend: SpectralFrontEnd | None = None,
                 n_fft=2048, hop_length=512, num_rotors=4):
        self.frontend = frontend or STFTMag(n_fft, hop_length)
        # encoder first conv in_channels = self.frontend.out_channels
    def forward(self, audio):
        x = self.frontend(audio)      # (B, C, F, T)
        h = self.encoder(x); …
```
Because the encoder pools frequency with `mean(dim=2)` at the end, a SimpleConv
can ingest **any** F (1025 from STFT, 360 from HCQT) unchanged — this is exactly
the representation-swap experiment requirement (2) asks for.

---

## 4. Checkpoint compatibility (the critical bit)

Old SimpleConv state-dict keys: `encoder.*`, `head.*`, `window`.
After refactor: `encoder.*`, `head.*`, `frontend.window`. **Only one key moves.**

Strategy:
1. Default `frontend` for every SimpleConv* is an `STFTMag`/`STFTMagPhase` whose
   `forward` calls `torch.stft(... normalized=True)` then `log1p` — **the same
   op the old `forward` ran**, so features are bit-identical.
2. Provide a load shim in `src/models/rps/simple_conv.py`:
   ```python
   def load_simpleconv_state_dict(model, sd):
       if "window" in sd and "frontend.window" not in sd:
           sd = {**sd, "frontend.window": sd.pop("window")}
       return model.load_state_dict(sd, strict=False)
   ```
   `window` is a deterministic hann window, so even `strict=False` with a missing
   buffer is safe; the remap keeps it exact.
3. **Acceptance test:** load a real pre-refactor SimpleConv checkpoint, run on a
   fixed noise buffer, assert output equals the pre-refactor model's output to
   `atol=0` (same STFT) — gate the merge on this.

`encoder.0.*` first-conv weights are unchanged because default `out_channels`
(1 or 3) matches the old hardcoded input channel count.

---

## 5. Migration steps (each independently testable)

1. **Relocate, no behaviour change.** Move root `models/*` → `src/models/`,
   register in `pyproject`, reinstall. Fix the handful of `from src.models.multif0`
   → `from models.multif0`. Run existing tests + a smoke import of every
   `get_model_from_config` key. *Gate: full import sweep green.*
2. **Introduce `frontends/`.** Add `SpectralFrontEnd`, `STFTMag`,
   `STFTMagPhase`, `HCQTFrontEnd`, `FRONTEND_REGISTRY`. Unit-test each against
   the inlined expression it replaces (numeric equality). *Gate: front-end
   parity tests.*
3. **Re-wire SimpleConv family onto front-ends** with default = exact-STFT.
   Add the load shim. *Gate: checkpoint-equality test from §4.3.*
4. **Re-wire `MultiF0RPSPredictor`** to take a front-end (default `HCQTFrontEnd`,
   params configurable), splitting `2H` → `(mag,dphase)`. Keep the librosa path
   as `HCQTFrontEnd(backend="librosa")`; wire `backend="nnaudio"` for GPU.
   *Gate: existing `test_multif0_rps.py` passes; one training step runs.*
5. **Expose front-end selection** to `train_rps_predictor.py`: `--frontend
   {stft_mag,stft_magphase,hcqt}` plus passthrough params (`--hcqt-bins-per-oct`,
   `--n-harmonics`, …). Model construction reads it. *Gate: a 1-epoch run with
   each (model × frontend) pair that makes sense.*
6. **Docs.** Rewrite `src/models/AGENTS.md` (registry table, front-end section,
   checkpoint-compat note); update root `AGENTS.md` directory map; delete the old
   root `models/AGENTS.md` or leave a stub pointer.

---

## 6. Consumer updates
- `src/utils/__init__.py`: unchanged (`from models.X` still resolves).
- `train_rps_predictor.py`: import `from models.rps import …`; add `--frontend`.
- `src/tasks/checkpoints.py`: unchanged keys; verify `MODEL_TYPES` export still
  satisfied by new `src/models/__init__.py`.
- slides scripts: unchanged.

---

## 7. Config / CLI surface (new)
```
--frontend stft_mag | stft_magphase | hcqt        # default per model
# STFT passthrough:   --n-fft --hop-length
# HCQT passthrough:   --hcqt-sr --hcqt-hop --hcqt-fmin
                      --hcqt-bins-per-oct --n-harmonics --hcqt-backend
```
For enhancement models (`train.py`/`get_model_from_config`) the front-end stays
implicit for now — those models embed STFT in their own encoders and are out of
scope for the swap (see §9).

---

## 8. Risks & mitigations
| Risk | Mitigation |
|------|------------|
| Two `models` packages shadow each other mid-migration | Step 1 is a **move**; verify `import models; models.__file__` points into `src/models` before proceeding |
| SimpleConv checkpoint silently degrades | §4.3 bit-equality gate blocks merge |
| HCQT into SimpleConv changes encoder F-dim → shape bug | encoder ends in `mean(dim=2)`, F-agnostic; covered by a shape test per (model×frontend) |
| nnAudio vs librosa drift | front-end parity test asserts 100% peak-bin match (already established) |
| `.ipynb_checkpoints/` stale copies get moved too | exclude them in the move |

---

## 9. Out of scope (named, not done)
- Enhancement models (`dcunet`, `dccrn`, roformers) keep their internal STFT;
  retrofitting them onto `SpectralFrontEnd` is a separate effort — their encoders
  are complex-valued and frequency-shape-dependent, so the swap is not free.
- Learnable front-ends (SincNet, trainable filterbanks): the ABC permits them
  later; none built now.
- A unified front-end for the *generative* models (they invert, not just analyze).
```
