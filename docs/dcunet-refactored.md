# DCUNet/DCCRN Refactored - Separate Encoder/Decoder Modules

This document describes the refactored DCUNet and DCCRN implementations with separate encoder and decoder modules, where RPS conditioning is a decoder-side feature.

## Overview

The refactored implementation (`models/dcunet_refactored.py`) provides:

1. **Separate Encoder and Decoder Modules**: Clean, modular architecture where encoder and decoder can be accessed independently.

2. **Decoder-side RPS Conditioning**: RPS information is only used in the decoder, not the encoder. This allows:
   - Training encoder-only baselines
   - Analyzing where RPS helps most (encoder vs decoder)
   - Decoupled architecture for research flexibility

## Architecture

### EncoderModule (`EncoderModule`)
- Clean feature extraction from STFT input
- No RPS input - pure signal processing
- Returns: `(bottleneck_features, encoder_features)` where `encoder_features` is a list of skip connections

### DecoderModule (`DecoderModule`)
- Receives bottleneck and skip connections from encoder
- Optionally receives RPS data for conditioning
- Two fusion strategies:
  - **bottleneck**: RPS injected at start of decoder (after first transposed conv)
  - **hierarchical**: RPS injected at multiple decoder levels

## Decoder RPS Fusion Strategies

### 1. Decoder Bottleneck Fusion
- RPS features are injected at the beginning of the decoder path
- Mirrors encoder bottleneck fusion conceptually
- Single injection point at bottleneck level

```python
# Architecture:
# Encoder -> Bottleneck -> [RPS Injection] -> Decoder Layers -> Output
```

### 2. Decoder Hierarchical Fusion
- RPS features are injected at multiple decoder levels
- Mirrors encoder hierarchical fusion
- Each level gets RPS info aligned to that level's time dimension

```python
# Architecture:
# Encoder -> Bottleneck -> Decoder[0] -> [RPS at L1] -> Decoder[1] -> ... -> Output
#                        ↓
#                   [RPS at L0]
```

## RPS Fusion Implementation

### DecoderBottleneckRPSFusion
- `RotorEncoder`: Encodes rotor RPS time series to 64-dim features
- `Conv1d`: Projects from 64 channels to `C*2` (C = bottleneck channels)
- Reshapes to `(B, C, F, T, 2)` complex format
- Broadcasts across frequency dimension

### DecoderHierarchicalRPSFusion
- Shared `RotorEncoder` for all levels
- Per-level `Conv1d` projections to match input channels of each decoder level
- Level 0: matches bottleneck channels
- Level i (i > 0): matches `dec_channels[i-1] * 2` (after skip concat)

## Usage

### Loading via Config

```python
from utils import get_model_from_config

# Baseline (no RPS)
model, config = get_model_from_config('dcunet_refactored', 'configs/12a_DCUNetRefactored_baseline.yaml')

# Decoder bottleneck RPS fusion
model, config = get_model_from_config('dcunet_refactored', 'configs/12b_DCUNetRefactored_decoder_bottleneck.yaml')

# Decoder hierarchical RPS fusion
model, config = get_model_from_config('dcunet_refactored', 'configs/12c_DCUNetRefactored_decoder_hierarchical.yaml')
```

### Direct Python Usage

```python
import torch
from models.dcunet_refactored import DCUNetRefactored, DCCRNRefactored

config = {
    'audio': {
        'chunk_size': 131584,
        'dim_f': 1024,
        'hop_length': 512,
        'n_fft': 2048,
        'num_channels': 1,
        'sample_rate': 16000,
    },
}

# Baseline
model = DCUNetRefactored(config)

# With decoder RPS fusion
config['use_rps'] = True
config['decoder_rps_fusion'] = 'hierarchical'
config['num_rotors'] = 4
model = DCUNetRefactored(config)

x = torch.randn(2, 1, 8192)
rps = torch.randn(2, 4, 100)

output = model(x, rps=rps)  # RPS is only used in decoder
```

### Accessing Encoder/Decoder Modules

```python
model = DCUNetRefactored(config)

# Access encoder
bottleneck, skip_features = model.encoder(x_stft)

# Access decoder
output = model.decoder(bottleneck, skip_features, rps=rps)
```

## Model Variants

| Config File | Model | RPS Fusion |
|-------------|-------|-----------|
| `12a_DCUNetRefactored_baseline.yaml` | DCUNetRefactored | None (baseline) |
| `12b_DCUNetRefactored_decoder_bottleneck.yaml` | DCUNetRefactored | decoder_bottleneck |
| `12c_DCUNetRefactored_decoder_hierarchical.yaml` | DCUNetRefactored | decoder_hierarchical |

## Key Differences from Original Implementation

| Aspect | Original | Refactored |
|--------|----------|------------|
| Encoder RPS | Yes | No (clean) |
| Decoder RPS | No | Yes |
| `use_rps` flag | Model-wide | Decoder-only |
| Encoder Module | Part of model | Standalone `EncoderModule` |
| Decoder Module | Part of model | Standalone `DecoderModule` |

## Parameter Count

| Model | Parameters |
|-------|------------|
| DCUNetRefactored (baseline) | ~2.8M |
| DCUNetRefactored (decoder bottleneck) | ~2.8M |
| DCUNetRefactored (decoder hierarchical) | ~2.9M |

## Training

Use the standard training pipeline with the new model type:

```bash
python train.py \
    --model_type dcunet_refactored \
    --config_path configs/12c_DCUNetRefactored_decoder_hierarchical.yaml \
    --results_path results/dcunet_refactored_hierarchical \
    --data_path datasets/DREGON-LM/train \
    --valid_path datasets/DREGON-LM/valid \
    --device_ids 0
```

## Evaluation

```bash
python final_valid.py \
    --model_type dcunet_refactored \
    --config_path configs/12c_DCUNetRefactored_decoder_hierarchical.yaml \
    --start_check_point results/dcunet_refactored_hierarchical/best_model.ckpt \
    --valid_path datasets/DREGON-LM/valid \
    --store_dir results/evaluation \
    --device_ids 0 \
    --metrics si_sdr sdr pesq stoi
```
