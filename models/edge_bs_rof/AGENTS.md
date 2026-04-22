# models/edge_bs_rof/ — Edge-BS-RoFormer

Band-split RoPE Transformer model for ultra-low SNR UAV speech enhancement. This is the proposed method from Paper 1.

## Why this directory exists

The Edge-BS-RoFormer is a complex model with multiple component files, warranting its own subdirectory rather than a single file.

## Key Concepts

- **Band-split strategy**: Partitions the speech spectrum into non-uniform sub-bands
- **Dual-dimension RoPE**: Rotary Position Encoding for joint time-frequency modeling
- **FlashAttention**: Computational efficiency optimization
- **Edge deployment**: Designed for <500MB memory and <1 RTF on NVIDIA Jetson

## Model Type

- `edge_bs_rof` → Edge-BS-RoFormer (BSRoformer variant)
- `mel_band_roformer` → MelBandRoformer variant

## Files

See individual file comments for implementation details.