# src/models/ — Model Implementations

All neural network models live here.  This directory is an editable-installed
package (mapped to ``models`` in ``pyproject.toml`` hatch packages), so
``from models.X import Y`` works from anywhere in the project.

## Directory layout

```
src/models/
  __init__.py           (none — namespace package)
  frontends/            Pluggable spectral front-ends
  multif0/              Multi-F0 HCQT + CNN (Cuesta et al. ISMIR 2020)
  rps_predictor.py      SimpleConv* family + DCUNet/DCCRN encoders (RPS)
  dcunet.py             DCUNet (speech enhancement)
  dccrn.py              DCCRN
  dcunet_refactored.py  DCUNetRefactored, DCCRNRefactored
  demucs4ht.py          HTDemucs
  diffusion_buffer.py   DiffusionBufferModel
  dptnet/               DPTNet
  edge_bs_rof/          BSRoformer, MelBandRoformer
  generative/           RPS → noise generative models (DroneNoiseGen, …)
```

## Spectral front-ends

Located in `frontends/`.  Every front-end implements:

```python
class SpectralFrontEnd(nn.Module):
    out_channels: int
    def forward(self, audio: Tensor) -> Tensor:   # (B, N) → (B, C, F, T)
    def num_frames(self, n_samples: int) -> int:
```

Build via: `from models.frontends import build_frontend; fe = build_frontend(name, **kw)`

| Key | Class | `out_channels` | F (typical) | Description |
|-----|-------|----------------|-------------|-------------|
| `stft_mag` | STFTMag | 1 | n_fft//2+1 | log₁₊ magnitude |
| `stft_magphase` | STFTMagPhase | 3 | n_fft//2+1 | log mag + cos(θ) + sin(θ) |
| `hcqt` | HCQTFrontEnd | H or 2H | 360·bpo/60 | Harmonic CQT (librosa). `phase=True`→2H |

### Using a front-end

```python
from models.frontends import build_frontend

fe = build_frontend("stft_mag", n_fft=2048, hop_length=512)
features = fe(audio)  # (B, 1, F, T)

# HCQT with custom resolution (120 bins/octave = 10¢)
fe = build_frontend("hcqt", phase=True, over_sample=10, harmonics=[1,2,3,4,5])
features = fe(audio)  # (B, 10, 720, T)
```

### Adding a new front-end

1. Subclass `SpectralFrontEnd`, set `key` and `out_channels`.
2. Decorate with `@register_frontend`.
3. Import the module from `frontends/__init__.py::_ensure_imported`.
4. It's instantly available via `build_frontend(key)`.

## Model type registry (enhancement models)

From `utils.get_model_from_config()` — see `src/utils/AGENTS.md`.

| Key | Model | File | RPS support |
|-----|-------|------|-------------|
| `edge_bs_rof` | Edge-BS-RoFormer | `edge_bs_rof/` | No |
| `mel_band_roformer` | MelBandRoformer | `edge_bs_rof/` | No |
| `dcunet` | DCUNet | `dcunet.py` | Yes |
| `dcunet_refactored` | DCUNetRefactored | `dcunet_refactored.py` | Yes |
| `dccrn` | DCCRN | `dccrn.py` | Yes |
| `dccrn_refactored` | DCCRNRefactored | `dcunet_refactored.py` | Yes |
| `dptnet` | DPTNet | `dptnet/` | No |
| `htdemucs` | HTDemucs | `demucs4ht.py` | No |
| `diffusion_buffer` | DiffusionBufferModel | `diffusion_buffer.py` | No |

## RPS prediction models

Registered in `train_rps_predictor.py::MODEL_REGISTRY` (see `rps_predictor.py::RPS_MODEL_REGISTRY`).

| Key | Class / description |
|-----|---------------------|
| `simple_conv` | Baseline — 5-block encoder + Conv1d head |
| `simple_conv_v2` | Residual + SE + attention pool + BiGRU |
| `simple_conv_wide` | Wider/deeper baseline |
| `simple_conv_tcn` | TCN head (dilated convs) |
| `simple_conv_multiscale` | FPN-style multi-scale fusion |
| `simple_conv_bigru` | Baseline encoder + BiGRU head |
| `simple_conv_bigru_v2` | Deeper + BiGRU head |
| `simple_conv_magphase_bigru` | 3-channel (mag+phase) + BiGRU |
| `simple_conv_attn_pool` | Attention frequency pooling |
| `simple_conv_se_next` | SE + residual + deeper |
| `dcunet_enc_rps` | DCUNet complex-conv encoder (in `train_rps_predictor.py`) |
| `dccrn_enc_rps` | DCCRN complex-conv encoder (in `train_rps_predictor.py`) |
| `multif0_rps` | Multi-F0 LateDeep CNN + soft-centroid RPS |

All SimpleConv* models now accept a `frontend=` kwarg.  Old checkpoints are
loadable via automatic `window` → `frontend.window` remap.

## Multi-F0 (Cuesta et al. ISMIR 2020)

Located in `multif0/`.  Pure PyTorch reimplementation.

| Class | Description |
|-------|-------------|
| `EarlyShallow` | Early-fusion, shallow CNN |
| `EarlyDeep` | Early-fusion, deep CNN |
| `LateDeep` | Late-fusion, deep CNN (best reported) |
| `LateDeepNoPhase` | LateDeep without phase input |
| `MultiF0RPSPredictor` | Wraps `LateDeep` for RPS prediction (HCQT→CNN→soft-centroid→RPS) |

HCQT is in `multif0/hcqt.py` (librosa-based, reference implementation).
A GPU-compatible version via nnAudio is in `multif0/nnaudio_cqt.py`.

## Generative models (RPS → noise)

In `generative/`.  Not registered in `get_model_from_config`.  Used by
`train_noise_gen.py`.

| Class | Purpose |
|-------|---------|
| `DroneNoiseGen` | Per-rotor harmonic oscillator bank |
| `DroneNoisePlusFilterGen` | DroneNoiseGen + RPS-conditioned filtered-noise residual |
| `HarmonicTransformModule` | VP-transform harmonic analysis/synthesis |

## Checkpoint compatibility

SimpleConv* models produced before the front-end refactor (pre-0.13) stored
the Hann window as ``"window"``.  After the refactor it is ``"frontend.window"``.
All model classes define ``load_state_dict`` that automatically remaps.
Existing checkpoints load without user intervention.

To verify a legacy checkpoint loads correctly:

```python
from models.rps_predictor import SimpleConv
model = SimpleConv(n_fft=2048, hop_length=512, num_rotors=4)
ckpt = torch.load("old_checkpoint.pt")
model.load_state_dict(ckpt["state_dict"], strict=True)  # remap is automatic
```
