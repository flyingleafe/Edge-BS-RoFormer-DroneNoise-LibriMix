# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

# Import necessary libraries
import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import soundfile as sf
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Union
import loralib as lora


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """
    Load configuration file from specified path based on model type.

    Args:
    ----------
    model_type : str
        Model type (e.g., 'htdemucs', 'mdx23c', etc.)
    config_path : str
        Path to YAML or OmegaConf configuration file

    Returns:
    -------
    config : Any
        Loaded configuration object, can be OmegaConf or ConfigDict format

    Raises:
    ------
    FileNotFoundError: When configuration file does not exist
    ValueError: When error occurs loading configuration file
    """
    try:
        with open(config_path, 'r') as f:
            # htdemucs model uses OmegaConf format configuration
            if model_type == 'htdemucs':
                config = OmegaConf.load(config_path)
            # Other models use yaml format configuration
            else:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple:
    """
    Load corresponding model based on model type and configuration file.

    Args:
    ----------
    model_type : str
        Model type (e.g., 'mdx23c', 'htdemucs', 'scnet', etc.)
    config_path : str
        Path to configuration file (YAML or OmegaConf format)

    Returns:
    -------
    model : nn.Module or None
        Model instance initialized based on model_type
    config : Any
        Configuration object used to initialize model

    Raises:
    ------
    ValueError: When unknown model_type or model initialization error
    """

    config = load_config(model_type, config_path)

    # Load corresponding model architecture based on model type
    if model_type == 'mdx23c':
        # MDX23C model - uses TFC-TDF network architecture
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        # HTDemucs model - high-quality source separation model based on Demucs
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        # Segmentation model - for audio segmentation tasks
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        # TorchSeg model - PyTorch implementation of segmentation model
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        # Mel band-based Roformer model
        from models.edge_bs_rof import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'edge_bs_rof':
        # Base Roformer model
        from models.edge_bs_rof import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'swin_upernet':
        # Swin Transformer + UperNet architecture
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        # Bandit model - Multi-mask multi-source band split RNN
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'bandit_v2':
        # Bandit V2 model - improved version
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == 'scnet_unofficial':
        # Unofficial SCNet implementation
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet':
        # Official SCNet implementation
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'apollo':
        # Apollo model - model from Look2Hear framework
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == 'bs_mamba2':
        # BS-Mamba2 model - separator based on Mamba architecture
        from models.ts_bs_mamba2 import Separator
        model = Separator(**config.model)
    elif model_type == 'experimental_mdx23c_stht':
        # Experimental MDX23C model - TFC-TDF network with STHT
        from models.mdx23c_tfc_tdf_v3_with_STHT import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'dcunet':
        # DCUNet model
        from models.dcunet import DCUNet
        model = DCUNet(config)
    elif model_type == 'dprnn':
        # DPRNN model - source separation model based on deep recurrent neural network
        from models.dprnn.dprnn import DPRNN
        model = DPRNN(config)
    elif model_type == 'dptnet':
        # DPTNet model - source separation model based on dual-path transformer network
        from models.dptnet.dpt_net import DPTNet
        model = DPTNet(config)


    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def read_audio_transposed(path: str, instr: str = None, skip_err: bool = False) -> Tuple[np.ndarray, int]:
    """
    Read audio file and transpose it.

    Args:
    ----------
    path : str
        Path to audio file
    skip_err: bool
        Whether to skip errors
    instr: str
        Instrument name

    Returns:
    -------
    Tuple[np.ndarray, int]
        - Transposed audio data with shape (channels, length)
        - Sample rate (e.g., 44100)
    """

    try:
        mix, sr = sf.read(path)
    except Exception as e:
        if skip_err:
            print(f"No stem {instr}: skip!")
            return None, None
        else:
            raise RuntimeError(f"Error reading the file at {path}: {e}")
    else:
        # Convert mono audio to 2D array
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=-1)
        return mix.T, sr


def normalize_audio(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize audio signal.

    Args:
    ----------
    audio : np.ndarray
        Input audio array with shape (channels, time) or (time,)

    Returns:
    -------
    tuple[np.ndarray, dict[str, float]]
        - Normalized audio array
        - Dictionary containing mean and standard deviation
    """

    # Compute mono signal
    mono = audio.mean(0)
    # Compute mean and standard deviation
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Denormalize a normalized audio signal.

    Args:
    ----------
    audio : np.ndarray
        Normalized audio array
    norm_params : dict[str, float]
        Dictionary containing mean and standard deviation

    Returns:
    -------
    np.ndarray
        Denormalized audio array
    """

    return audio * norm_params["std"] + norm_params["mean"]


def apply_tta(
        config,
        model: torch.nn.Module,
        mix: torch.Tensor,
        waveforms_orig: Dict[str, torch.Tensor],
        device: torch.device,
        model_type: str
) -> Dict[str, torch.Tensor]:
    """
    Apply test-time augmentation (TTA) for source separation.

    Improves separation quality by applying augmentations like channel inversion
    and polarity inversion to the input mixture, then averaging all augmented results.

    Args:
    ----------
    config : Any
        Model configuration object
    model : torch.nn.Module
        Trained model
    mix : torch.Tensor
        Mixture audio tensor (channels, time)
    waveforms_orig : Dict[str, torch.Tensor]
        Original separated waveforms dictionary
    device : torch.device
        Computing device (CPU/CUDA)
    model_type : str
        Model type

    Returns:
    -------
    Dict[str, torch.Tensor]
        Updated separated waveforms dictionary after applying TTA
    """
    # Create augmentations: channel inversion and polarity inversion
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    # Process each augmented mixture
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    # Average all augmented results
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate window array with linear fade-in and fade-out.

    Args:
    ----------
    window_size : int
        Total window size
    fade_size : int
        Size of fade-in/fade-out region

    Returns:
    -------
    torch.Tensor
        Generated window array with shape (window_size,)
    """

    # Generate fade-in and fade-out sequences
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    # Create window and apply fade-in/fade-out
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(
        config: ConfigDict,
        model: torch.nn.Module,
        mix: torch.Tensor,
        device: torch.device,
        model_type: str,
        pbar: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    Unified source separation function supporting multiple processing modes.

    Uses overlapping window chunk processing for efficient artifact-free separation.

    Args:
    ----------
    config : ConfigDict
        Audio and inference settings configuration
    model : torch.nn.Module
        Trained separation model
    mix : torch.Tensor
        Input mixture tensor (channels, time)
    device : torch.device
        Computing device
    model_type : str
        Model type, e.g., "demucs"
    pbar : bool
        Whether to show progress bar

    Returns:
    -------
    Union[Dict[str, np.ndarray], np.ndarray]
        - Dictionary mapping instruments to separated audio when multiple instruments
        - Separated audio array when single instrument
    """

    mix = torch.tensor(mix, dtype=torch.float32)

    # Select processing mode based on model type
    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'

    # Set processing parameters based on mode
    if mode == 'demucs':
        # Demucs mode parameters
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        # Generic mode parameters
        chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        # Add boundary padding
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(
                total=mix.shape[1], desc="Processing audio chunks", leave=False
            ) if pbar else None

            # Process audio in chunks
            while i < mix.shape[1]:
                # Extract and pad audio chunk
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch when batch_size is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    if mode == "generic":
                        window = windowing_array.clone()
                        if i - step == 0:  # First chunk doesn't need fade-in
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last chunk doesn't need fade-out
                            window[-fade_size:] = 1

                    # Add processed results to total result
                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return results
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    """
    Return target instrument list based on configuration.

    Args:
    ----------
    config : ConfigDict
        Configuration object containing instrument list or target instrument

    Returns:
    -------
    List[str]
        Target instrument list
    """
    if getattr(config.training, 'target_instrument', None):
        return [config.training.target_instrument]
    else:
        return config.training.instruments


def load_not_compatible_weights(model: torch.nn.Module, weights: str, verbose: bool = False) -> None:
    """
    Load partially compatible weights into model.

    Args:
    ----------
    model: Target PyTorch model
    weights: Path to weights file
    verbose: Whether to print detailed information
    """

    new_model = model.state_dict()
    old_model = torch.load(weights)
    if 'state' in old_model:
        # htdemucs weight loading fix
        old_model = old_model['state']
    if 'state_dict' in old_model:
        # apollo weight loading fix
        old_model = old_model['state_dict']

    # Iterate through each layer of new model
    for el in new_model:
        if el in old_model:
            if verbose:
                print(f'Match found for {el}!')
            if new_model[el].shape == old_model[el].shape:
                # Same shape, directly copy
                if verbose:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                # Handle different shape case
                if len(new_model[el].shape) != len(old_model[el].shape):
                    if verbose:
                        print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if verbose:
                        print(f'Shape is different: {tuple(new_model[el].shape)} != {tuple(old_model[el].shape)}')
                    # Handle weights with different shapes
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if verbose:
                print(f'Match not found for {el}!')
    model.load_state_dict(
        new_model
    )


def load_lora_weights(model: torch.nn.Module, lora_path: str, device: str = 'cpu') -> None:
    """
    Load LoRA weights into model.

    Args:
    ----------
    model : Module
        Target PyTorch model
    lora_path : str
        Path to LoRA checkpoint file
    device : str
        Device to load weights to
    """
    lora_state_dict = torch.load(lora_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)


def load_start_checkpoint(args: argparse.Namespace, model: torch.nn.Module, type_='train') -> None:
    """
    Load starting checkpoint for model.

    Args:
    ----------
    args: Command line arguments containing checkpoint path
    model: PyTorch model to load checkpoint into
    type_: Method of loading weights
    """

    print(f'Start from checkpoint: {args.start_check_point}')
    if type_ in ['train']:
        if 1:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            model.load_state_dict(torch.load(args.start_check_point))
    else:
        device='cpu'
        if args.model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=False)
            # htdemucs pretrained model fix
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # apollo pretrained model fix
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    if args.lora_checkpoint:
        print(f"Loading LoRA weights from: {args.lora_checkpoint}")
        load_lora_weights(model, args.lora_checkpoint)


def bind_lora_to_model(config: Dict[str, Any], model: nn.Module) -> nn.Module:
    """
    Replace specific layers in model with LoRA extended versions.

    Args:
    ----------
    config : Dict[str, Any]
        Configuration containing LoRA parameters
    model : nn.Module
        Original model with layers to replace

    Returns:
    -------
    nn.Module
        Model with replaced layers
    """

    if 'lora' not in config:
        raise ValueError("Configuration must contain the 'lora' key with parameters for LoRA.")

    replaced_layers = 0  # Replaced layer counter

    # Iterate through all model modules
    for name, module in model.named_modules():
        hierarchy = name.split('.')
        layer_name = hierarchy[-1]

        # Check if this is a target layer for replacement
        if isinstance(module, nn.Linear):
            try:
                # Get parent module
                parent_module = model
                for submodule_name in hierarchy[:-1]:
                    parent_module = getattr(parent_module, submodule_name)

                # Replace original layer with LoRA layer
                setattr(
                    parent_module,
                    layer_name,
                    lora.MergedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        **config['lora']
                    )
                )
                replaced_layers += 1

            except Exception as e:
                print(f"Error replacing layer {name}: {e}")

    if replaced_layers == 0:
        print("Warning: No layers were replaced. Check the model structure and configuration.")
    else:
        print(f"Number of layers replaced with LoRA: {replaced_layers}")

    return model


def draw_spectrogram(waveform, sample_rate, length, output_file):
    """
    Draw spectrogram of audio waveform.

    Args:
    ----------
    waveform: Audio waveform data
    sample_rate: Sample rate
    length: Length to draw
    output_file: Output file path
    """
    import librosa.display

    # Extract required portion of spectrogram
    x = waveform[:int(length * sample_rate), :]
    # Apply short-time Fourier transform to mono signal
    X = librosa.stft(x.mean(axis=-1))
    # Convert amplitude spectrum to dB-scaled spectrogram
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    fig, ax = plt.subplots()
    # Display spectrogram
    img = librosa.display.specshow(
        Xdb,
        cmap='plasma',
        sr=sample_rate,
        x_axis='time',
        y_axis='linear',
        ax=ax
    )
    ax.set(title='File: ' + os.path.basename(output_file))
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    if output_file is not None:
        plt.savefig(output_file)
