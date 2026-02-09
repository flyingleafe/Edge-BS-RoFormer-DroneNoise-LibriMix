# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.4'

# Import necessary libraries
import json
import random
import argparse
from tqdm.auto import tqdm
import os
from dotenv import load_dotenv
import torch
import wandb

# Load environment variables from .env file
load_dotenv()
import numpy as np
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ml_collections import ConfigDict
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Callable, Optional
import shutil
import soundfile as sf

# Import custom modules
from dataset import MSSDataset  # Music source separation dataset class
from utils import get_model_from_config  # Load model from config file
from valid import valid_multi_gpu, valid  # Validation functions

from utils import bind_lora_to_model, load_start_checkpoint
import loralib as lora  # LoRA (Low-Rank Adaptation)

import warnings

warnings.filterwarnings("ignore")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command line arguments for configuring model, dataset, and training parameters.

    Main arguments include:
    - model_type: Model type to use (mdx23c/htdemucs, etc.)
    - config_path: Path to configuration file
    - data_path: Training data path
    - dataset_type: Dataset type (1-4)
    - device_ids: GPU device IDs
    - metrics: List of evaluation metrics
    - train_lora: Whether to use LoRA training
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, edge_bs_rof, swin_upernet, bandit, diffusion_buffer")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str,
                        help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str,
                        help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--use_multistft_loss", action='store_true', help="Use MultiSTFT Loss (from auraloss package)")
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora", action='store_true', help="Train with LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    return args


def manual_seed(seed: int) -> None:
    """
    Set random seed to ensure experiment reproducibility.

    Includes:
    - Python random library
    - NumPy
    - PyTorch CPU and GPU
    - CUDA backend
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """
    Initialize training environment.

    Includes:
    - Setting random seed
    - Configuring PyTorch settings
    - Creating results directory
    """
    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(results_path, exist_ok=True)

def wandb_init(args: argparse.Namespace, config: Dict, device_ids: List[int], batch_size: int) -> None:
    """
    Initialize wandb logging system.

    Used for:
    - Recording training process
    - Visualizing training metrics
    - Saving experiment configuration

    API key is taken from --wandb_key argument or WANDB_API_KEY environment variable.
    """
    wandb_key = args.wandb_key if args.wandb_key else os.environ.get('WANDB_API_KEY', '')
    if wandb_key is None or wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=wandb_key)
        wandb.init(entity='flyingleafe', project='msst', config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size })


def prepare_data(config: Dict, args: argparse.Namespace, batch_size: int) -> DataLoader:
    """
    Prepare training data.

    Main steps:
    1. Create MSSDataset instance
    2. Configure DataLoader parameters
    3. Return training data loader
    """
    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, f'metadata_{args.dataset_type}.pkl'),
        dataset_type=args.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return train_loader


def prepare_valid_data(config: Dict, args: argparse.Namespace, batch_size: int = 2) -> Union[DataLoader, None]:
    """
    Prepare validation data loader for audio sample logging.

    Returns None if args.valid_path is not set. Otherwise returns a DataLoader
    with the same dataset format as training (batch, mixes).
    """
    if not getattr(args, 'valid_path', None):
        return None
    validset = MSSDataset(
        config,
        args.valid_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, f'valid_metadata_{args.dataset_type}.pkl'),
        dataset_type=args.dataset_type,
    )
    return DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.pin_memory,
    )


def initialize_model_and_device(model: torch.nn.Module, device_ids: List[int]) -> Tuple[Union[torch.device, str], torch.nn.Module]:
    """
    Initialize model and assign to appropriate device.

    Handles:
    1. Single GPU/multi-GPU configuration
    2. CPU fallback support
    3. DataParallel processing
    """
    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        model = model.to(device)
        print("CUDA is not available. Running on CPU.")

    return device, model


def get_optimizer(config: ConfigDict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Initialize optimizer based on configuration.

    Supported optimizers:
    - Adam: Adaptive moment estimation
    - AdamW: Adam with weight decay
    - RAdam: Rectified Adam
    - RMSprop: Root mean square propagation
    - Prodigy: Novel optimizer
    - SGD: Stochastic gradient descent
    """
    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        print(f'Optimizer params from config:\n{optim_params}')

    name_optimizer = getattr(config.training, 'optimizer',
                             'No optimizer in config')

    if name_optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'prodigy':
        from prodigyopt import Prodigy
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'sgd':
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        print(f'Unknown optimizer: {name_optimizer}')
        exit()
    return optimizer


def masked_loss(y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True) -> torch.Tensor:
    """
    Compute masked loss function.

    Implementation:
    1. Compute MSE loss
    2. Generate mask based on quantile
    3. Apply mask to get final loss

    Shapes:
    - y_: [num_stems, batch_size, channels, audio_length]
    - y: same as y_
    """
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()


def multistft_loss(y: torch.Tensor, y_: torch.Tensor, loss_multistft: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Compute multi-resolution STFT loss.

    Used for:
    - Joint optimization in frequency and time domains
    - Improving audio reconstruction quality

    Supports:
    - 4D tensors (standard models)
    - 3D tensors (Apollo and similar models)
    """
    if len(y_.shape) == 4:
        y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
        y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
    elif len(y_.shape) == 3:
        y1_ = y_
        y1 = y
    else:
        raise ValueError(f"Invalid shape for predicted array: {y_.shape}. Expected 3 or 4 dimensions.")

    return loss_multistft(y1_, y1)


def choice_loss(args: argparse.Namespace, config: ConfigDict) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Select appropriate loss function.

    Supported loss function combinations:
    1. MultiSTFT + MSE + L1
    2. MultiSTFT + MSE
    3. MultiSTFT + L1
    4. MultiSTFT only
    5. MSE + L1
    6. MSE only
    7. L1 only
    8. Masked Loss
    """
    if args.use_multistft_loss:
        loss_options = dict(getattr(config, 'loss_multistft', {}))
        print(f'Loss options: {loss_options}')
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)

        if args.use_mse_loss and args.use_l1_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + nn.MSELoss()(y_, y) + F.l1_loss(y_, y)
        elif args.use_mse_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + nn.MSELoss()(y_, y)
        elif args.use_l1_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + F.l1_loss(y_, y)
        else:
            def multi_loss(y_, y):
                return multistft_loss(y_, y, loss_multistft) / 1000
    elif args.use_mse_loss:
        if args.use_l1_loss:
            def multi_loss(y_, y):
                return nn.MSELoss()(y_, y) + F.l1_loss(y_, y)
        else:
            multi_loss = nn.MSELoss()
    elif args.use_l1_loss:
        multi_loss = F.l1_loss
    else:
        def multi_loss(y_, y):
            return masked_loss(y_, y, q=config.training.q, coarse=config.training.coarse_loss_clip)
    return multi_loss


def normalize_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize batch data.

    Steps:
    1. Compute mean
    2. Compute standard deviation
    3. Apply normalization
    """
    mean = x.mean()
    std = x.std()
    if std != 0:
        x = (x - mean) / std
        y = (y - mean) / std
    return x, y


def get_sample_rate(config: ConfigDict) -> int:
    """Get audio sample rate from config (audio.sample_rate, audio.samplerate, or 16000)."""
    if hasattr(config, 'audio'):
        sr = getattr(config.audio, 'sample_rate', None) or getattr(config.audio, 'samplerate', None)
        if sr is not None:
            return int(sr)
    return 16000


def get_model_output(model: torch.nn.Module, x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Run model inference (no target) and return predicted audio. Handles edge_bs_rof/diffusion_buffer vs others."""
    if args.model_type in ['mel_band_roformer', 'edge_bs_rof', 'diffusion_buffer']:
        return model(x)
    return model(x)


def _tensor_to_audio_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a single audio tensor to 1D numpy (squeeze batch/channel/source dims)."""
    a = t.detach().cpu().float().numpy()
    while a.ndim > 1:
        a = a[0]
    return np.asarray(a, dtype=np.float32)


# SNR categories for wandb audio sample selection (DN-LM metadata: input_snr in dB)
# light: 0 or -5 dB, medium: -10 or -15 dB, heavy: -25 or -30 dB
SNR_CATEGORIES = {
    'light': (-5, 0),    # inclusive [min, max] dB
    'medium': (-15, -10),
    'heavy': (-30, -25),
}


def load_snr_metadata(data_path: Union[str, List[str]]) -> Optional[Dict[str, float]]:
    """
    Load sample_id -> input_snr from DN-LM metadata.json if present.
    data_path: training or validation root (e.g. 'datasets/DN-LM/train' or list of one).
    Returns dict sample_id -> input_snr, or None if no metadata.
    """
    roots = [data_path] if isinstance(data_path, str) else list(data_path)
    for root in roots:
        meta_path = os.path.join(root, 'metadata.json')
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                data = json.load(f)
            # Format: { "train": [ {"id": "...", "input_snr": ...}, ... ] } or { "valid": ... }
            for key in ('train', 'valid'):
                if key in data and isinstance(data[key], list):
                    return {item['id']: item['input_snr'] for item in data[key] if 'id' in item and 'input_snr' in item}
            return None
        except Exception:
            return None
    return None


def select_sample_ids_by_snr(
    metadata: Dict[str, float],
    seed: int = 0,
) -> List[Tuple[str, str, float]]:
    """
    Select one sample_id per SNR category (light, medium, heavy).
    Returns list of (sample_id, category_name, snr_value); length 0--3.
    """
    rng = random.Random(seed)
    selected: List[Tuple[str, str, float]] = []
    for cat_name, (min_db, max_db) in SNR_CATEGORIES.items():
        candidates = [(sid, snr) for sid, snr in metadata.items() if min_db <= snr <= max_db and np.isfinite(snr)]
        if not candidates:
            continue
        sid, snr = rng.choice(candidates)
        selected.append((sid, cat_name, float(snr)))
    return selected


def load_sample_audio_from_folder(
    root: str,
    sample_id: str,
    target_file: str = 'vocals',
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mixture and target for one DN-LM sample from root/sample_id/.
    Returns (mix, target) as float32 numpy (samples,) or (channels, samples) to match model.
    """
    folder = os.path.join(root, sample_id)
    mix_path = os.path.join(folder, 'mixture.wav')
    tgt_path = os.path.join(folder, f'{target_file}.wav')
    if not os.path.isfile(mix_path) or not os.path.isfile(tgt_path):
        raise FileNotFoundError(f"Missing {mix_path} or {tgt_path}")
    mix, sr_m = sf.read(mix_path, dtype='float32')
    tgt, sr_t = sf.read(tgt_path, dtype='float32')
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]
    if tgt.ndim == 1:
        tgt = tgt[np.newaxis, :]
    # (samples, channels) -> (channels, samples)
    mix = np.ascontiguousarray(mix.T)
    tgt = np.ascontiguousarray(tgt.T)
    if target_sr and (sr_m != target_sr or sr_t != target_sr):
        import librosa
        # librosa.resample expects (n_channels, n_samples) or (n_samples,); resamples along last axis
        if sr_m != target_sr:
            mix = librosa.resample(mix, orig_sr=sr_m, target_sr=target_sr, res_type='kaiser_best')
            if mix.ndim == 1:
                mix = mix[np.newaxis, :]
        if sr_t != target_sr:
            tgt = librosa.resample(tgt, orig_sr=sr_t, target_sr=target_sr, res_type='kaiser_best')
            if tgt.ndim == 1:
                tgt = tgt[np.newaxis, :]
    return mix, tgt


def collect_audio_triples_by_snr(
    model: torch.nn.Module,
    args: argparse.Namespace,
    config: ConfigDict,
    device: torch.device,
    data_path: Union[str, List[str]],
    epoch: int,
    target_file: str = 'vocals',
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[str], int]:
    """
    Select 3 samples by SNR category (light, medium, heavy), run model, return triples and captions.
    Returns (triples, captions, sample_rate). If metadata or samples missing, returns ([], [], sr).
    """
    sample_rate = get_sample_rate(config)
    roots = [data_path] if isinstance(data_path, str) else list(data_path)
    if not roots:
        return [], [], sample_rate
    root = roots[0]
    metadata = load_snr_metadata(data_path)
    if not metadata:
        return [], [], sample_rate
    # Fixed seed so the same 3 samples are selected every epoch (stable evolution in wandb)
    selected = select_sample_ids_by_snr(metadata, seed=0)
    if not selected:
        return [], [], sample_rate
    triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    captions: List[str] = []
    normalize = getattr(config.training, 'normalize', False)
    model.eval()
    with torch.no_grad():
        for sample_id, cat_name, snr_val in selected:
            try:
                mix, tgt = load_sample_audio_from_folder(root, sample_id, target_file=target_file, target_sr=sample_rate)
            except FileNotFoundError:
                continue
            # (C, T) -> (1, C, T)
            x = torch.from_numpy(mix).float().unsqueeze(0).to(device)
            y_np = tgt
            if mix.shape[0] == 1:
                y_np_1d = np.squeeze(y_np)
            else:
                y_np_1d = np.mean(y_np, axis=0)
            if normalize:
                mean, std = x.mean().item(), x.std().item()
                if std != 0:
                    x = (x - mean) / std
            out = get_model_output(model, x, args)
            inp_1d = np.squeeze(mix) if mix.shape[0] == 1 else np.mean(mix, axis=0)
            out_1d = _tensor_to_audio_numpy(out[0])
            triples.append((np.asarray(inp_1d, dtype=np.float32), np.asarray(y_np_1d, dtype=np.float32), out_1d))
            captions.append(f'{cat_name} {snr_val:+.1f} dB')
    model.train()
    return triples, captions, sample_rate


def log_audio_triples_to_wandb(
    prefix: str,
    triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    sample_rate: int,
    epoch: int,
    captions: Optional[List[str]] = None,
    output_only: bool = False,
) -> None:
    """
    Log audio to wandb.
    - output_only=False (epoch 0): log input, target, and output for each sample.
    - output_only=True (epoch > 0): log only output for each sample (same keys so evolution is comparable).
    """
    if not triples:
        return
    log_dict = {'epoch': epoch}
    for idx, (inp, tgt, out) in enumerate(triples):
        cap = captions[idx] if captions and idx < len(captions) else f'sample_{idx}'
        if not output_only:
            log_dict[f'{prefix}/sample_{idx}/input'] = wandb.Audio(inp, sample_rate=sample_rate, caption=f'input {cap}')
            log_dict[f'{prefix}/sample_{idx}/target'] = wandb.Audio(tgt, sample_rate=sample_rate, caption=f'target {cap}')
        log_dict[f'{prefix}/sample_{idx}/output'] = wandb.Audio(out, sample_rate=sample_rate, caption=f'output {cap}')
    wandb.log(log_dict)


def train_one_epoch(
    model: torch.nn.Module,
    config: ConfigDict,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    device_ids: List[int],
    epoch: int,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    gradient_accumulation_steps: int,
    train_loader: torch.utils.data.DataLoader,
    multi_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    audio_logging_cache: Optional[Dict[str, Optional[Tuple[List[np.ndarray], List[np.ndarray], int]]]] = None,
) -> None:
    """
    Train model for one epoch.

    Main steps:
    1. Data preprocessing and normalization
    2. Forward pass
    3. Loss computation
    4. Backward pass
    5. Gradient accumulation and optimizer update
    6. Metric logging

    Special handling:
    - Automatic mixed precision training
    - Gradient clipping
    - Special handling for different model architectures
    """
    model.train().to(device)
    print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')
    loss_val = 0.
    total = 0

    normalize = getattr(config.training, 'normalize', False)
    sample_rate = get_sample_rate(config)
    train_audio_triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    train_audio_captions: Optional[List[str]] = None

    # Prefer 3 samples by SNR (light / medium / heavy) when DN-LM metadata exists (stable: same samples every epoch)
    target_file = getattr(config.training, 'target_instrument', None) or 'vocals'
    train_triples_snr, train_captions_snr, _ = collect_audio_triples_by_snr(
        model, args, config, device, args.data_path, epoch, target_file=target_file
    )
    if train_triples_snr:
        train_audio_triples = train_triples_snr
        train_audio_captions = train_captions_snr
    # Epoch > 0 and fallback was used at epoch 0: we'll run model on cached inputs at end of epoch (after training)

    pbar = tqdm(train_loader)
    for i, (batch, mixes) in enumerate(pbar):
        x = mixes.to(device)  # mixture
        y = batch.to(device)

        if normalize:
            x, y = normalize_batch(x, y)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            if args.model_type in ['mel_band_roformer', 'edge_bs_rof', 'diffusion_buffer']:
                # loss is computed in forward pass
                loss = model(x, y)
                if isinstance(device_ids, (list, tuple)):
                    # If it's multiple GPUs sum partial loss
                    loss = loss.mean()
            else:
                y_ = model(x)
                loss = multi_loss(y_, y)

        # Fallback: collect 3 training sample triples from first batch when no SNR metadata (epoch 0 only)
        if not train_audio_triples and epoch == 0 and i == 0 and x.shape[0] >= 3:
            with torch.no_grad():
                model.eval()
                x_slice = x[:3]
                out_slice = get_model_output(model, x_slice, args)
                model.train()
            for k in range(min(3, out_slice.shape[0])):
                inp_k = _tensor_to_audio_numpy(x_slice[k])
                tgt_k = _tensor_to_audio_numpy(y[k])
                out_k = _tensor_to_audio_numpy(out_slice[k])
                train_audio_triples.append((inp_k, tgt_k, out_k))
            # Cache inputs/targets so same samples are used for output-only logging in later epochs
            if audio_logging_cache is not None:
                audio_logging_cache['train'] = (
                    [t[0] for t in train_audio_triples],
                    [t[1] for t in train_audio_triples],
                    sample_rate,
                )

        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()
        if config.training.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        li = loss.item() * gradient_accumulation_steps
        loss_val += li
        total += 1
        pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
        wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})
        loss.detach()

    print(f'Training loss: {loss_val / total}')
    wandb.log({'train_loss': loss_val / total, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})

    # Epoch > 0 and fallback was used at epoch 0: run model on cached inputs (after training) and log outputs only
    if epoch > 0 and audio_logging_cache and audio_logging_cache.get('train') is not None and not train_audio_triples:
        cached_inp, cached_tgt, sample_rate = audio_logging_cache['train']
        model.eval()
        train_audio_triples = []
        with torch.no_grad():
            for inp_k, tgt_k in zip(cached_inp, cached_tgt):
                x = torch.from_numpy(inp_k).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
                out = get_model_output(model, x, args)
                out_k = _tensor_to_audio_numpy(out[0])
                train_audio_triples.append((inp_k, tgt_k, out_k))
        model.train()
        train_audio_captions = None

    if train_audio_triples:
        log_audio_triples_to_wandb(
            'audio_samples/train', train_audio_triples, sample_rate, epoch,
            captions=train_audio_captions, output_only=(epoch > 0),
        )


def save_weights(args: argparse.Namespace, store_path_prefix: str, model: torch.nn.Module, device_ids: List[int],
                 train_lora: bool, epoch: int, metric_value: float, is_early_stop: bool) -> None:
    """
    Save model weights.
    Supports:
    - Standard model weight saving
    - LoRA weight saving
    - Multi-GPU model weight saving
    """
    if train_lora:
        suffix = f'_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_value:.4f}.ckpt'
        if is_early_stop:
            store_path = os.path.join(store_path_prefix, f'early_stop{suffix}')
        else:
            store_path = os.path.join(store_path_prefix, f'model{suffix}')
        torch.save(lora.lora_state_dict(model), store_path)
        best_model_path = os.path.join(store_path_prefix, "best_model.ckpt")
        shutil.copy(store_path, best_model_path)
    else:
        state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
        suffix = f'_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_value:.4f}.ckpt'
        if is_early_stop:
            store_path = os.path.join(store_path_prefix, f'early_stop{suffix}')
        else:
            store_path = os.path.join(store_path_prefix, f'model{suffix}')
        torch.save(state_dict, store_path)
        best_model_path = os.path.join(store_path_prefix, "best_model.ckpt")
        shutil.copy(store_path, best_model_path)


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], epoch: int,
                          scheduler: torch.optim.lr_scheduler._LRScheduler, best_metric: float) -> Tuple[float, float]:
    """
    Compute and log evaluation metrics for current epoch.

    Main functions:
    1. Validate model performance
    2. Adjust learning rate
    3. Log to wandb
    """
    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics_avg = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
    else:
        metrics_avg = valid(model, args, config, device, verbose=False)

    # Check if model needs to be saved
    current_metric = metrics_avg[args.metric_for_scheduler]
    if current_metric > best_metric:
        best_metric = current_metric
        save_weights(args, args.results_path, model, device_ids, args.train_lora, epoch, current_metric, is_early_stop=False)

    scheduler.step(current_metric)
    wandb.log({'metric_main': current_metric})
    for metric_name in metrics_avg:
        wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})
    return current_metric, best_metric


def collect_and_log_valid_audio(
    model: torch.nn.Module,
    args: argparse.Namespace,
    config: ConfigDict,
    device: torch.device,
    valid_loader: Optional[torch.utils.data.DataLoader],
    epoch: int,
    audio_logging_cache: Optional[Dict[str, Optional[Tuple[List[np.ndarray], List[np.ndarray], int]]]] = None,
) -> None:
    """Get 3 validation sample triples by SNR (light/medium/heavy) or from loader; log to wandb. Stable selection; output-only after epoch 0."""
    sample_rate = get_sample_rate(config)
    target_file = getattr(config.training, 'target_instrument', None) or 'vocals'
    triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    captions: Optional[List[str]] = None

    # Epoch > 0 and fallback was used at epoch 0: run model on cached inputs, log outputs only
    if epoch > 0 and audio_logging_cache and audio_logging_cache.get('valid') is not None:
        cached_inp, cached_tgt, sample_rate = audio_logging_cache['valid']
        model.eval()
        with torch.no_grad():
            for inp_k, tgt_k in zip(cached_inp, cached_tgt):
                x = torch.from_numpy(inp_k).float().unsqueeze(0).unsqueeze(0).to(device)
                out = get_model_output(model, x, args)
                out_k = _tensor_to_audio_numpy(out[0])
                triples.append((inp_k, tgt_k, out_k))
        model.train()
    # Prefer 3 samples by SNR when DN-LM metadata exists (stable: same samples every epoch)
    elif getattr(args, 'valid_path', None):
        triples, captions, sample_rate = collect_audio_triples_by_snr(
            model, args, config, device, args.valid_path, epoch, target_file=target_file
        )
    # Fallback: take up to 3 from first batch of valid_loader (epoch 0 only)
    elif valid_loader is not None and epoch == 0:
        model.eval()
        with torch.no_grad():
            for batch, mixes in valid_loader:
                x = mixes.to(device)
                y = batch.to(device)
                if getattr(config.training, 'normalize', False):
                    x, y = normalize_batch(x, y)
                out = get_model_output(model, x, args)
                for k in range(min(3, x.shape[0])):
                    inp_k = _tensor_to_audio_numpy(x[k])
                    tgt_k = _tensor_to_audio_numpy(y[k])
                    out_k = _tensor_to_audio_numpy(out[k]) if out.shape[0] > k else _tensor_to_audio_numpy(out[0])
                    triples.append((inp_k, tgt_k, out_k))
                if audio_logging_cache is not None and triples:
                    audio_logging_cache['valid'] = (
                        [t[0] for t in triples],
                        [t[1] for t in triples],
                        sample_rate,
                    )
                break
        model.train()

    if triples:
        log_audio_triples_to_wandb(
            'audio_samples/valid', triples, sample_rate, epoch,
            captions=captions, output_only=(epoch > 0),
        )


def train_model(args: argparse.Namespace) -> None:
    """
    Main function for model training.

    Complete training pipeline:
    1. Argument parsing and environment initialization
    2. Model and data preparation
    3. Optimizer and loss function configuration
    4. Training loop
       - Training for each epoch
       - Validation and metric computation
       - Model saving
       - Learning rate adjustment
    5. Wandb logging

    Supported features:
    - Multi-GPU training
    - Mixed precision training
    - LoRA fine-tuning
    - Resume from checkpoint
    - Multiple evaluation metrics
    """
    args = parse_args(args)

    initialize_environment(args.seed, args.results_path)
    model, config = get_model_from_config(args.model_type, args.config_path)
    use_amp = getattr(config.training, 'use_amp', True)
    device_ids = args.device_ids
    batch_size = config.training.batch_size * len(device_ids)

    wandb_init(args, config, device_ids, batch_size)

    train_loader = prepare_data(config, args, batch_size)
    valid_loader = prepare_valid_data(config, args, batch_size=3)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='train')

    if args.train_lora:
        model = bind_lora_to_model(config, model)
        lora.mark_only_lora_as_trainable(model)

    device, model = initialize_model_and_device(model, args.device_ids)

    if args.pre_valid:
        if torch.cuda.is_available() and len(device_ids) > 1:
            valid_multi_gpu(model, args, config, args.device_ids, verbose=True)
        else:
            valid(model, args, config, device, verbose=True)

    optimizer = get_optimizer(config, model)
    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    # Reduce LR if no metric improvements for several epochs
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience,
                                  factor=config.training.reduce_factor)

    multi_loss = choice_loss(args, config)
    scaler = GradScaler()

    # Read early stopping configuration
    early_stop = getattr(config.training, 'early_stop', {})
    early_stop_enabled = early_stop.get('enabled', False)
    early_stop_patience = early_stop.get('patience', 5)
    metric_for_early_stop = early_stop.get('metric', args.metric_for_scheduler)
    best_metric = float('-inf')
    no_improvement_count = 0

    print(
        f"Instruments: {config.training.instruments}\n"
        f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
        f"Patience: {config.training.patience} "
        f"Reduce factor: {config.training.reduce_factor}\n"
        f"Batch size: {batch_size} "
        f"Grad accum steps: {gradient_accumulation_steps} "
        f"Effective batch size: {batch_size * gradient_accumulation_steps}\n"
        f"Dataset type: {args.dataset_type}\n"
        f"Optimizer: {config.training.optimizer}"
    )

    print(f'Train for: {config.training.num_epochs} epochs')

    # Cache (input, target) for train/valid when using fallback (no SNR metadata) so same samples are logged every epoch
    audio_logging_cache: Dict[str, Optional[Tuple[List[np.ndarray], List[np.ndarray], int]]] = {'train': None, 'valid': None}

    for epoch in range(config.training.num_epochs):
        train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, gradient_accumulation_steps, train_loader, multi_loss, audio_logging_cache)
        current_metric, best_metric = compute_epoch_metrics(model, args, config, device, device_ids, epoch, scheduler, best_metric)
        if valid_loader is not None:
            collect_and_log_valid_audio(model, args, config, device, valid_loader, epoch, audio_logging_cache)
        if early_stop_enabled:
            # If this epoch's metric doesn't improve best_metric, increment counter; otherwise reset
            if current_metric < best_metric:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            # If no improvement for consecutive epochs reaches threshold, stop training early
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping: {metric_for_early_stop} has not improved for {early_stop_patience} consecutive epochs.")
                save_weights(args, args.results_path, model, device_ids, args.train_lora, epoch, current_metric, is_early_stop=True)
                break

if __name__ == "__main__":
    train_model(None)
