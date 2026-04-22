# models/ — Model Implementations

Contains all neural network model implementations for speech enhancement, plus RPS conditioning components.

## Why this directory exists

Central place for model architecture code. Each model is either a single file or a subdirectory. All models are registered in `utils.py:get_model_from_config()` for use by `train.py`.

## Model Type Registry

| Key | Model | File | RPS support |
|-----|-------|------|-------------|
| `edge_bs_rof` | Edge-BS-RoFormer (BSRoformer) | `edge_bs_rof/` | No |
| `mel_band_roformer` | MelBandRoformer | `edge_bs_rof/` | No |
| `dcunet` | DCUNet | `dcunet.py` | Yes (bottleneck, gru, hierarchical) |
| `dcunet_refactored` | DCUNetRefactored | `dcunet_refactored.py` | Yes (decoder-only: bottleneck, hierarchical) |
| `dccrn` | DCCRN | `dccrn.py` | Yes (bottleneck, gru) |
| `dccrn_refactored` | DCCRNRefactored | `dcunet_refactored.py` | Yes (decoder-only: bottleneck, hierarchical) |
| `dptnet` | DPTNet | `dptnet/` | No |
| `htdemucs` | HTDemucs | `demucs4ht.py` | No |
| `diffusion_buffer` | DiffusionBufferModel | `diffusion_buffer.py` | No |

## RPS Conditioning Architecture

### RotorEncoder (shared, in `dcunet.py`)
- Input: raw RPS `(4, n_motor_samples)` at motor sampling rate (~929 Hz)
- 2-layer 1D conv: `4 → 32 → 64` channels
- Resamples to match STFT frame count
- Output: 64-dim per-frame RPS features

### Fusion Strategies
- **Bottleneck**: RPS features projected to bottleneck dim → added at bottleneck layer
- **GRU**: RPS features concatenated with flattened features before GRU
- **Hierarchical**: RPS features injected at multiple encoder/decoder levels

### Auxiliary RPS Prediction Head (`RPSPredictionHead`, in `dcunet.py`)
- FPN-style multi-scale head on encoder features
- Predicts 4 rotor speeds from encoder output
- Multi-task loss: `total_loss = main_loss + lambda_rps * rps_mse_loss`
- Enabled via `predict_rps: true` in config

### Refactored Models (in `dcunet_refactored.py`)
- Separate `EncoderModule` and `DecoderModule` classes
- Encoder: clean feature extraction (no RPS)
- Decoder: receives RPS data for conditioning
- Config key: `decoder_rps_fusion: bottleneck | hierarchical`

## Adding a New Model

1. Implement in `models/` — single file for simple models, subdirectory for complex ones
2. Import and register in `utils.py:get_model_from_config()` with a unique key
3. Create config YAML in `configs/` — see `configs/AGENTS.md`
4. Create experiment YAML in `experiments/` — see `experiments/AGENTS.md`

## Adding RPS to a Model

1. Add `use_rps` flag to config
2. Import `RotorEncoder` from `models.dcunet`
3. Encode RPS: `rps_features = rotor_encoder(rps, target_length=n_stft_frames)`
4. Choose fusion strategy (see above)
5. Optionally add `RPSPredictionHead` for auxiliary prediction
6. Dataset auto-loads `rps.npy` when `load_rps: true` in config

## Gotchas

- Model type keys must match **exactly** in `utils.py:get_model_from_config()` — typos cause silent failures
- `dcunet_refactored.py` contains both DCUNetRefactored and DCCRNRefactored — they share the encoder/decoder pattern
- RPS fusion happens at different places depending on strategy — check model code, not just config