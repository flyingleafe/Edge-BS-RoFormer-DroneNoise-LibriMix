# Edge-BS-RoFormer: Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement

[![Read in Chinese](https://img.shields.io/badge/中文版-README-blue.svg)](README_CN.md)

This repository contains the official implementation and the DroneNoise-LibriMix (DN-LM) dataset for the paper "Edge-BS-RoFormer: Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement".

## Abstract

Addressing the significant challenge of speech enhancement in ultra-low Signal-to-Noise Ratio (SNR) scenarios in Unmanned Aerial Vehicle (UAV) voice communication, this study proposes an edge-deployed Band-Split Rotary Position Encoding Transformer (Edge-BS-RoFormer). Existing deep learning methods show significant limitations in suppressing dynamic UAV noise under edge computing constraints. These limitations mainly include insufficient modeling of harmonic features and high computational complexity. The proposed method employs a band-split strategy to partition the speech spectrum into non-uniform sub-bands, integrates a dual-dimension Rotary Position Encoding (RoPE) mechanism for joint time-frequency modeling, and adopts FlashAttention to optimize computational efficiency. Experiments on a self-constructed DroneNoise-LibriMix (DN-LM) dataset demonstrate that the proposed method achieves Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) improvements of 2.2 dB and 2.2 dB, and Perceptual Evaluation of Speech Quality (PESQ) enhancements of 0.15 and 0.11, respectively, compared to Deep Complex U-Net (DCUNet) and HTDemucs under -15 dB SNR conditions. Edge deployment tests reveal the model's memory footprint is under 500MB with a Real-Time Factor (RTF) of 0.33, fulfilling real-time processing requirements. This study provides a lightweight solution for speech enhancement in complex acoustic environments. Furthermore, the open-source dataset facilitates the establishment of standardized evaluation frameworks in the field.

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- ~10GB disk space for datasets

### One-Command Replication

To replicate all paper results with a single command:

```bash
./replicate_paper.sh all
```

Or run individual steps:

```bash
./replicate_paper.sh setup     # Set up environment
./replicate_paper.sh download  # Download source datasets
./replicate_paper.sh dataset   # Create DN-LM dataset
./replicate_paper.sh train     # Train all models
./replicate_paper.sh eval      # Evaluate models
```

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that ensures reproducible environments.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/NonTrivialLiu/Edge-BS-RoFormer-DroneNoise-LibriMix.git
cd Edge-BS-RoFormer-DroneNoise-LibriMix

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/NonTrivialLiu/Edge-BS-RoFormer-DroneNoise-LibriMix.git
cd Edge-BS-RoFormer-DroneNoise-LibriMix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset: DroneNoise-LibriMix (DN-LM)

The DN-LM dataset is synthesized from two source datasets:

1. **LibriSpeech** - Clean speech samples
   - URL: https://www.openslr.org/12/
   - Subset used: `train-clean-100` (~6GB)

2. **DroneAudioDataset** - UAV noise samples
   - URL: https://github.com/saraalemadi/DroneAudioDataset

### Dataset Creation

The dataset is created by mixing speech and UAV noise at various SNR levels:

- **Total duration**: 2 hours
- **Sample duration**: 1 second each
- **Sample rate**: 16 kHz (mono)
- **SNR range**: -30 dB to 0 dB
- **Train/Valid split**: 9:1 (6480 / 720 samples)

To create the dataset manually:

```bash
# Download source datasets
mkdir -p data/librispeech data/drone_audio

# Download LibriSpeech
cd data/librispeech
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
cd ../..

# Clone DroneAudioDataset
git clone https://github.com/saraalemadi/DroneAudioDataset.git data/drone_audio

# Create DN-LM dataset
python create_dataset.py \
    --speech_dir data/librispeech/LibriSpeech/train-clean-100 \
    --noise_dir data/drone_audio \
    --output_dir datasets/DN-LM \
    --train_samples 6480 \
    --valid_samples 720 \
    --sample_rate 16000 \
    --snr_min -30 \
    --snr_max 0
```

### Dataset Structure

```
datasets/DN-LM/
├── train/
│   ├── sample_00000/
│   │   ├── vocals.wav      # Clean speech
│   │   ├── noise.wav       # UAV noise
│   │   └── mixture.wav     # Mixed audio
│   ├── sample_00001/
│   │   └── ...
│   └── metadata.json
└── valid/
    ├── sample_00000/
    │   └── ...
    └── metadata.json
```

## Training

### Train Edge-BS-RoFormer (Proposed Method)

```bash
python train.py \
    --model_type edge_bs_rof \
    --config_path configs/3_FA_RoPE\(64\).yaml \
    --results_path results/edge_bs_roformer \
    --data_path datasets/DN-LM/train \
    --valid_path datasets/DN-LM/valid \
    --dataset_type 1 \
    --device_ids 0 \
    --num_workers 4 \
    --metrics si_sdr sdr \
    --metric_for_scheduler si_sdr
```

### Train Diffusion Buffer (BBED)

```bash
python train.py \
    --model_type diffusion_buffer \
    --config_path configs/9_Diffusion_Buffer_BBED.yaml \
    --results_path results/diffusion_buffer_bbed \
    --data_path datasets/DN-LM/train \
    --valid_path datasets/DN-LM/valid \
    --dataset_type 1 \
    --device_ids 0 \
    --num_workers 4 \
    --metrics si_sdr sdr \
    --metric_for_scheduler si_sdr
```

### Train Baseline Models

```bash
# DCUNet
python train.py \
    --model_type dcunet \
    --config_path configs/5_Baseline_dcunet.yaml \
    --results_path results/dcunet \
    --data_path datasets/DN-LM/train \
    --valid_path datasets/DN-LM/valid \
    --dataset_type 1 \
    --device_ids 0

# DPTNet
python train.py \
    --model_type dptnet \
    --config_path configs/7_Baseline_dptnet.yaml \
    --results_path results/dptnet \
    --data_path datasets/DN-LM/train \
    --valid_path datasets/DN-LM/valid \
    --dataset_type 1 \
    --device_ids 0

# HTDemucs
python train.py \
    --model_type htdemucs \
    --config_path configs/8_Baseline_htdemucs.yaml \
    --results_path results/htdemucs \
    --data_path datasets/DN-LM/train \
    --valid_path datasets/DN-LM/valid \
    --dataset_type 1 \
    --device_ids 0
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Initial LR | 5.0×10⁻⁴ |
| LR Schedule | ReduceLROnPlateau (factor=0.95, patience=2) |
| Batch Size | 12 |
| Steps/Epoch | 200 |
| Early Stopping | 30 epochs patience |
| Precision | FP32 |

### Training Configuration (Diffusion Buffer, BBED)

Config file: `configs/9_Diffusion_Buffer_BBED.yaml`

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial LR | 1.0×10⁻⁴ |
| Batch Size | 32 |
| EMA Decay | 0.999 |
| Epochs | 250 |
| Chunk Frames (K) | 128 |
| STFT Window/Hop | 510 / 256 (periodic Hann) |

## Evaluation

```bash
python final_valid.py \
    --model_type edge_bs_rof \
    --config_path configs/3_FA_RoPE\(64\).yaml \
    --start_check_point results/edge_bs_roformer/best_model.ckpt \
    --valid_path datasets/DN-LM/valid \
    --store_dir results/evaluation \
    --device_ids 0 \
    --metrics si_sdr sdr pesq stoi
```

## Results

### Performance Comparison at -15 dB SNR

| Model | SI-SDR (dB) | PESQ | STOI | GFLOPs | Storage (MB) |
|-------|-------------|------|------|--------|--------------|
| **Edge-BS-RoFormer** | **Best** | **Best** | **Best** | **11.6** | **8.5** |
| DCUNet | -2.2 dB | -0.11 | ≈ | 112.1 | 10.8 |
| DPTNet | -25.0 dB | -0.18 | Lower | 41.8 | 187.3 |
| HTDemucs | -2.3 dB | -0.15 | ≈ | 48.4 | 160.3 |

### Edge Deployment (NVIDIA Jetson AGX Xavier)

| Metric | Edge-BS-RoFormer |
|--------|------------------|
| FLOPs | 11.617 GFLOPs |
| Model Storage | 8.534 MB |
| Runtime Memory | < 500 MB |
| RTF | 0.325 |
| Latency | 330.83 ms |
| Power | 6.536 W |

## Project Structure

```
Edge-BS-RoFormer-DroneNoise-LibriMix/
├── configs/                    # Model configuration files
│   ├── 3_FA_RoPE(64).yaml     # Edge-BS-RoFormer config
│   ├── 9_Diffusion_Buffer_BBED.yaml # Diffusion Buffer (BBED) config
│   ├── 5_Baseline_dcunet.yaml # DCUNet config
│   ├── 7_Baseline_dptnet.yaml # DPTNet config
│   └── 8_Baseline_htdemucs.yaml # HTDemucs config
├── models/                     # Model implementations
│   ├── edge_bs_rof/           # Edge-BS-RoFormer model
│   ├── dcunet.py              # DCUNet baseline
│   ├── dptnet/                # DPTNet baseline
│   └── demucs4ht.py           # HTDemucs baseline
├── create_dataset.py          # DN-LM dataset creation script
├── train.py                   # Training script
├── valid.py                   # Validation during training
├── final_valid.py             # Final evaluation script
├── dataset.py                 # Dataset loader
├── metrics.py                 # Evaluation metrics
├── utils.py                   # Utility functions
├── replicate_paper.sh         # One-click replication script
└── requirements.txt           # Python dependencies
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liu2025edgebsroformer,
  title={Edge-Deployed Band-Split Rotary Position Encoding Transformer for Ultra-Low SNR UAV Speech Enhancement},
  author={Liu, Feifan and Li, Muying and Guo, Luming and Guo, Hao and Cao, Jie and Zhao, Wei and Wang, Jun},
  journal={Drones},
  volume={9},
  number={6},
  pages={386},
  year={2025},
  publisher={MDPI},
  doi={10.3390/drones9060386}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Training framework based on [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) by Roman Solovyev
- LibriSpeech dataset from [OpenSLR](https://www.openslr.org/12/)
- DroneAudioDataset from [saraalemadi](https://github.com/saraalemadi/DroneAudioDataset)
