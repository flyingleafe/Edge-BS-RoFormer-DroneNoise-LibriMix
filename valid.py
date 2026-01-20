# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

# Import required libraries
import argparse
import time
import os
import glob
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from ml_collections import ConfigDict
from typing import Tuple, Dict, List, Union
from utils import demix, get_model_from_config, prefer_target_instrument, draw_spectrogram
from utils import normalize_audio, denormalize_audio, apply_tta, read_audio_transposed, load_start_checkpoint
from metrics import get_metrics
import warnings

warnings.filterwarnings("ignore")


def logging(logs: List[str], text: str, verbose_logging: bool = False) -> None:
    """
    Log information during validation process.

    Args:
    ----------
    store_dir : str
        Directory to store logs, if empty then don't store
    logs : List[str]
        List to store logs
    text : str
        Text information to log
    """

    print(text)
    if verbose_logging:
        logs.append(text)


def write_results_in_file(store_dir: str, logs: List[str]) -> None:
    """
    Write validation results to file.

    Args:
    ----------
    store_dir : str
        Directory to store result file
    results : List[str]
        List of results to write to file
    """
    with open(f'{store_dir}/results.txt', 'w') as out:
        for item in logs:
            out.write(item + "\n")


def get_mixture_paths(
    args,
    verbose: bool,
    config: ConfigDict,
    extension: str
) -> List[str]:
    """
    Get paths to mixture audio files for validation.

    Args:
    ----------
    valid_path : List[str]
        List of validation dataset directories
    verbose : bool
        Whether to print detailed information
    config : ConfigDict
        Configuration object containing inference parameters like overlap count and batch size
    extension : str
        Audio file extension

    Returns:
    -------
    List[str]
        List of mixture audio file paths
    """
    try:
        valid_path = args.valid_path
    except Exception as e:
        print('No valid path in args')
        raise e

    all_mixtures_path = []
    for path in valid_path:
        part = sorted(glob.glob(f"{path}/*/mixture.{extension}"))
        if len(part) == 0:
            if verbose:
                print(f'No validation data found in: {path}')
        all_mixtures_path += part
    if verbose:
        print(f'Total mixtures: {len(all_mixtures_path)}')
        print(f'Overlap: {config.inference.num_overlap} Batch size: {config.inference.batch_size}')

    return all_mixtures_path


def update_metrics_and_pbar(
        track_metrics: Dict,
        all_metrics: Dict,
        instr: str,
        pbar_dict: Dict,
        mixture_paths: Union[List[str], tqdm],
        verbose: bool = False
) -> None:
    """
    Update evaluation metrics and progress bar.

    Args:
    ----------
    track_metrics : Dict
        Evaluation metrics dictionary for current track
    all_metrics : Dict
        Aggregated evaluation metrics dictionary for all tracks
    instr : str
        Name of current instrument being processed
    pbar_dict : Dict
        Dictionary of metrics to display in progress bar
    mixture_paths : tqdm
        Progress bar object
    verbose : bool
        Whether to print detailed information
    """
    for metric_name, metric_value in track_metrics.items():
        if verbose:
            print(f"Metric {metric_name:11s} value: {metric_value:.4f}")
        all_metrics[metric_name][instr].append(metric_value)
        pbar_dict[f'{metric_name}_{instr}'] = metric_value

    if mixture_paths is not None:
        try:
            mixture_paths.set_postfix(pbar_dict)
        except Exception:
            pass


def process_audio_files(
    mixture_paths: List[str],
    model: torch.nn.Module,
    args,
    config,
    device: torch.device,
    verbose: bool = False,
    is_tqdm: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    Process audio files and perform source separation evaluation.

    Args:
    ----------
    mixture_paths : List[str]
        List of mixture audio file paths
    model : torch.nn.Module
        Trained source separation model
    args : Any
        Argument object containing user-specified options
    config : Any
        Configuration object containing model and processing parameters
    device : torch.device
        Computing device (CPU or CUDA)
    verbose : bool
        Whether to print detailed logs
    is_tqdm : bool
        Whether to show progress bar

    Returns:
    -------
    Dict[str, Dict[str, List[float]]]
        Nested dictionary of evaluation metrics, outer key is metric name, inner key is instrument name
    """
    # Get target instrument list
    instruments = prefer_target_instrument(config)

    # Get test-time augmentation (TTA) settings
    use_tta = getattr(args, 'use_tta', False)
    # Get file storage directory
    store_dir = getattr(args, 'store_dir', '')
    # Get audio encoding format
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # Initialize evaluation metrics dictionary
    all_metrics = {
        metric: {instr: [] for instr in config.training.instruments}
        for metric in args.metrics
    }

    if is_tqdm:
        mixture_paths = tqdm(mixture_paths)

    # Process each mixture audio file
    for path in mixture_paths:
        start_time = time.time()
        # Read mixture audio
        mix, sr = read_audio_transposed(path)
        mix_orig = mix.copy()
        folder = os.path.dirname(path)

        # Resample to target sample rate
        if 'sample_rate' in config.audio:
            if sr != config.audio['sample_rate']:
                orig_length = mix.shape[-1]
                if verbose:
                    print(f'Warning: sample rate is different. In config: {config.audio["sample_rate"]} in file {path}: {sr}')
                mix = librosa.resample(mix, orig_sr=sr, target_sr=config.audio['sample_rate'], res_type='kaiser_best')

        if verbose:
            folder_name = os.path.abspath(folder)
            print(f'Song: {folder_name} Shape: {mix.shape}')

        # Audio normalization
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        # Perform source separation using model
        waveforms_orig = demix(config, model, mix.copy(), device, model_type=args.model_type)

        # Apply test-time augmentation
        if use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        pbar_dict = {}

        # Calculate evaluation metrics for each instrument
        for instr in instruments:
            if verbose:
                print(f"Instr: {instr}")

            # Read original instrument track as reference
            if instr != 'other' or config.training.other_fix is False:
                track, sr1 = read_audio_transposed(f"{folder}/{instr}.{extension}", instr, skip_err=True)
                if track is None:
                    continue
            else:
                # For 'other' track, compute from vocals track
                track, sr1 = read_audio_transposed(f"{folder}/vocals.{extension}")
                track = mix_orig - track

            estimates = waveforms_orig[instr]

            # Resample to original sample rate
            if 'sample_rate' in config.audio:
                if sr != config.audio['sample_rate']:
                    estimates = librosa.resample(estimates, orig_sr=config.audio['sample_rate'], target_sr=sr,
                                                 res_type='kaiser_best')
                    estimates = librosa.util.fix_length(estimates, size=orig_length)

            # Denormalize
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            # Save separation results
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
                out_wav_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.wav"
                sf.write(out_wav_name, estimates.T, sr, subtype='FLOAT')
                if args.draw_spectro > 0:
                    out_img_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.jpg"
                    draw_spectrogram(estimates.T, sr, args.draw_spectro, out_img_name)
                    out_img_name_orig = f"{store_dir}/{os.path.basename(folder)}_{instr}_orig.jpg"
                    draw_spectrogram(track.T, sr, args.draw_spectro, out_img_name_orig)

            # Calculate evaluation metrics
            track_metrics = get_metrics(
                args.metrics,
                track,
                estimates,
                mix_orig,
                device=device,
            )

            # Update evaluation metrics and progress bar
            update_metrics_and_pbar(
                track_metrics,
                all_metrics,
                instr, pbar_dict,
                mixture_paths=mixture_paths,
                verbose=verbose
            )

        if verbose:
            print(f"Time for song: {time.time() - start_time:.2f} sec")

    return all_metrics


def compute_metric_avg(
    store_dir: str,
    args,
    instruments: List[str],
    config: ConfigDict,
    all_metrics: Dict[str, Dict[str, List[float]]],
    start_time: float
) -> Dict[str, float]:
    """
    Compute and log average evaluation metrics for each instrument.

    Args:
    ----------
    store_dir : str
        Log storage directory
    args : dict
        Arguments dictionary
    instruments : List[str]
        List of instruments
    config : ConfigDict
        Configuration dictionary
    all_metrics : Dict[str, Dict[str, List[float]]]
        Dictionary of all evaluation metrics
    start_time : float
        Start time

    Returns:
    -------
    Dict[str, float]
        Average evaluation metrics for all instruments
    """

    logs = []
    if store_dir:
        logs.append(str(args))
        verbose_logging = True
    else:
        verbose_logging = False

    logging(logs, text=f"Num overlap: {config.inference.num_overlap}", verbose_logging=verbose_logging)

    metric_avg = {}
    # Compute mean and standard deviation of metrics for each instrument
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])

            mean_val = metric_values.mean()
            std_val = metric_values.std()

            logging(logs, text=f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})", verbose_logging=verbose_logging)
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val

    # Compute average metrics across all instruments
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            logging(logs, text=f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}', verbose_logging=verbose_logging)
    logging(logs, text=f"Elapsed time: {time.time() - start_time:.2f} sec", verbose_logging=verbose_logging)

    if store_dir:
        write_results_in_file(store_dir, logs)

    return metric_avg


def valid(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: torch.device,
    verbose: bool = False
) -> dict:
    """
    Validate model on a single device.

    Args:
    ----------
    model : torch.nn.Module
        Source separation model
    args : Namespace
        Command line arguments
    config : dict
        Configuration dictionary
    device : torch.device
        Computing device
    verbose : bool
        Whether to print detailed information

    Returns:
    -------
    dict
        Average evaluation metrics for all instruments
    """

    start_time = time.time()
    model.eval().to(device)

    # Get storage directory
    store_dir = getattr(args, 'store_dir', '')
    # Get audio encoding format
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # Get all mixture audio file paths
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)
    # Process audio files and compute evaluation metrics
    all_metrics = process_audio_files(all_mixtures_path, model, args, config, device, verbose, not verbose)
    instruments = prefer_target_instrument(config)

    # Compute average evaluation metrics
    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time)


def validate_in_subprocess(
    proc_id: int,
    queue: torch.multiprocessing.Queue,
    all_mixtures_path: List[str],
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: str,
    return_dict
) -> None:
    """
    Execute validation in subprocess, supporting multi-process parallel processing.

    Args:
    ----------
    proc_id : int
        Process ID
    queue : torch.multiprocessing.Queue
        Queue for receiving mixture audio file paths
    all_mixtures_path : List[str]
        All mixture audio file paths
    model : torch.nn.Module
        Source separation model
    args : dict
        Arguments dictionary
    config : ConfigDict
        Configuration object
    device : str
        Computing device
    return_dict : torch.multiprocessing.Manager().dict
        Shared dictionary for storing results from each process
    """

    m1 = model.eval().to(device)
    if proc_id == 0:
        progress_bar = tqdm(total=len(all_mixtures_path))

    # Initialize evaluation metrics dictionary
    all_metrics = {
        metric: {instr: [] for instr in config.training.instruments}
        for metric in args.metrics
    }

    while True:
        current_step, path = queue.get()
        if path is None:  # Check for end marker
            break
        single_metrics = process_audio_files([path], m1, args, config, device, False, False)
        pbar_dict = {}
        for instr in config.training.instruments:
            for metric_name in all_metrics:
                all_metrics[metric_name][instr] += single_metrics[metric_name][instr]
                if len(single_metrics[metric_name][instr]) > 0:
                    pbar_dict[f"{metric_name}_{instr}"] = f"{single_metrics[metric_name][instr][0]:.4f}"
        if proc_id == 0:
            progress_bar.update(current_step - progress_bar.n)
            progress_bar.set_postfix(pbar_dict)
    return_dict[proc_id] = all_metrics
    return


def run_parallel_validation(
    verbose: bool,
    all_mixtures_path: List[str],
    config: ConfigDict,
    model: torch.nn.Module,
    device_ids: List[int],
    args,
    return_dict
) -> None:
    """
    Run multi-process parallel validation.

    Args:
    ----------
    verbose : bool
        Whether to print detailed information
    all_mixtures_path : List[str]
        All mixture audio file paths
    config : ConfigDict
        Configuration object
    model : torch.nn.Module
        Source separation model
    device_ids : List[int]
        List of GPU device IDs
    args : dict
        Arguments dictionary
    return_dict
        Shared dictionary for storing results from all processes
    """

    model = model.to('cpu')
    try:
        # Extract single model for multi-GPU training
        model = model.module
    except:
        pass

    queue = torch.multiprocessing.Queue()
    processes = []

    # Create a process for each device
    for i, device in enumerate(device_ids):
        if torch.cuda.is_available():
            device = f'cuda:{device}'
        else:
            device = 'cpu'
        p = torch.multiprocessing.Process(
            target=validate_in_subprocess,
            args=(i, queue, all_mixtures_path, model, args, config, device, return_dict)
        )
        p.start()
        processes.append(p)

    # Add tasks to queue
    for i, path in enumerate(all_mixtures_path):
        queue.put((i, path))
    # Add end markers
    for _ in range(len(device_ids)):
        queue.put((None, None))
    # Wait for all processes to complete
    for p in processes:
        p.join()

    return


def valid_multi_gpu(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device_ids: List[int],
    verbose: bool = False
) -> Dict[str, float]:
    """
    Execute validation on multiple GPUs.

    Args:
    ----------
    model : torch.nn.Module
        Source separation model
    args : dict
        Arguments dictionary
    config : ConfigDict
        Configuration object
    device_ids : List[int]
        List of GPU device IDs
    verbose : bool
        Whether to print detailed information

    Returns:
    -------
    Dict[str, float]
        Average value for each evaluation metric
    """

    start_time = time.time()

    # Get storage directory
    store_dir = getattr(args, 'store_dir', '')
    # Get audio encoding format
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # Get all mixture audio file paths
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)

    # Create shared dictionary to store results
    return_dict = torch.multiprocessing.Manager().dict()

    # Run parallel validation
    run_parallel_validation(verbose, all_mixtures_path, config, model, device_ids, args, return_dict)

    # Merge results from all processes
    all_metrics = dict()
    for metric in args.metrics:
        all_metrics[metric] = dict()
        for instr in config.training.instruments:
            all_metrics[metric][instr] = []
            for i in range(len(device_ids)):
                all_metrics[metric][instr] += return_dict[i][metric][instr]

    instruments = prefer_target_instrument(config)

    # Compute average evaluation metrics
    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time)


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
    ----------
    dict_args: Dict
        Command line arguments dictionary, if None then parse from sys.argv

    Returns:
    -------
    argparse.Namespace
        Parsed arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer,"
                             " edge_bs_rof, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint"
                                                                          " to valid weights")
    parser.add_argument("--valid_path", nargs="+", type=str, help="Validate path")
    parser.add_argument("--store_dir", type=str, default="", help="Path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="If --store_dir is set then code will generate spectrograms for resulted stems as well."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='List of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="Dataloader pin_memory")
    parser.add_argument("--extension", type=str, default='wav', help="Choose extension for validation")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def check_validation(dict_args):
    """
    Main function for executing validation.

    Args:
    ----------
    dict_args
        Command line arguments dictionary
    """
    args = parse_args(dict_args)
    torch.backends.cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass

    # Get model and configuration
    model, config = get_model_from_config(args.model_type, args.config_path)

    # Load checkpoint
    if args.start_check_point:
        load_start_checkpoint(args, model, type_='valid')

    print(f"Instruments: {config.training.instruments}")

    # Set computing device
    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = 'cpu'
        print('CUDA is not available. Run validation on CPU. It will be very slow...')

    # Choose validation method based on device count
    if torch.cuda.is_available() and len(device_ids) > 1:
        valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        valid(model, args, config, device, verbose=True)


if __name__ == "__main__":
    check_validation(None)
