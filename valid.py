__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

# Import required libraries
import argparse
import glob
import os
import time
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from metrics import get_metrics
from utils import (
    apply_tta,
    demix,
    denormalize_audio,
    draw_spectrogram,
    get_model_from_config,
    load_start_checkpoint,
    normalize_audio,
    prefer_target_instrument,
    read_audio_transposed,
)

warnings.filterwarnings("ignore")


def logging(logs: list[str], text: str, verbose_logging: bool = False) -> None:
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


def write_results_in_file(store_dir: str, logs: list[str]) -> None:
    """
    Write validation results to file.

    Args:
    ----------
    store_dir : str
        Directory to store result file
    results : List[str]
        List of results to write to file
    """
    with open(f"{store_dir}/results.txt", "w") as out:
        for item in logs:
            out.write(item + "\n")


def get_mixture_paths(args, verbose: bool, config: DictConfig, extension: str) -> list[str]:
    """
    Get paths to mixture audio files for validation.

    Args:
    ----------
    valid_path : List[str]
        List of validation dataset directories
    verbose : bool
        Whether to print detailed information
    config : DictConfig
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
        print("No valid path in args")
        raise e

    all_mixtures_path = []
    for path in valid_path:
        part = sorted(glob.glob(f"{path}/*/mixture.{extension}"))
        if len(part) == 0:
            if verbose:
                print(f"No validation data found in: {path}")
        all_mixtures_path += part
    if verbose:
        print(f"Total mixtures: {len(all_mixtures_path)}")
        print(f"Overlap: {config.inference.num_overlap} Batch size: {config.inference.batch_size}")

    return all_mixtures_path


def update_metrics_and_pbar(
    track_metrics: dict,
    all_metrics: dict,
    instr: str,
    pbar_dict: dict,
    mixture_paths: list[str] | tqdm,
    verbose: bool = False,
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
        pbar_dict[f"{metric_name}_{instr}"] = metric_value

    if mixture_paths is not None:
        try:
            mixture_paths.set_postfix(pbar_dict)  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            pass


def process_audio_files(
    mixture_paths: list[str],
    model: torch.nn.Module,
    args,
    config,
    device: torch.device,
    verbose: bool = False,
    is_tqdm: bool = True,
) -> dict[str, dict[str, list[float]]]:
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
    use_tta = getattr(args, "use_tta", False)
    # Get file storage directory
    store_dir = getattr(args, "store_dir", "")
    # Get audio encoding format
    if "extension" in config["inference"]:
        extension = config["inference"]["extension"]
    else:
        extension = getattr(args, "extension", "wav")

    # Initialize evaluation metrics dictionary
    all_metrics = {
        metric: {instr: [] for instr in config.training.instruments} for metric in args.metrics
    }

    if is_tqdm:
        path_iter = tqdm(mixture_paths, desc="Processing")
    else:
        path_iter = mixture_paths

    # Process each mixture audio file
    for path in path_iter:
        start_time = time.time()
        # Read mixture audio
        mix, sr = read_audio_transposed(path)
        if mix is None or sr is None:
            continue
        mix_orig: np.ndarray = mix.copy()
        folder = os.path.dirname(path)

        # Resample to target sample rate
        if "sample_rate" in config.audio:
            if sr != config.audio["sample_rate"]:
                orig_length = mix.shape[-1]
                if verbose:
                    print(
                        f"Warning: sample rate is different. In config: {config.audio['sample_rate']} in file {path}: {sr}"
                    )
                mix = librosa.resample(
                    mix,
                    orig_sr=sr,
                    target_sr=config.audio["sample_rate"],
                    res_type="kaiser_best",
                )

        if verbose:
            folder_name = os.path.abspath(folder)
            print(f"Song: {folder_name} Shape: {mix.shape}")

        # Audio normalization
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                mix, norm_params = normalize_audio(mix)

        # Load RPS data if model uses rotor conditioning
        rps = None
        if getattr(model, "use_rps", False):
            rps_path = os.path.join(folder, "rps.npy")
            if os.path.exists(rps_path):
                rps = np.load(rps_path)

        # Perform source separation using model
        mix_tensor = torch.from_numpy(mix.copy())
        waveforms_orig = demix(
            config, model, mix_tensor, device, model_type=args.model_type, rps=rps
        )

        # Apply test-time augmentation
        if use_tta and isinstance(waveforms_orig, dict):
            waveforms_orig = apply_tta(
                config, model, mix_tensor, waveforms_orig, device, args.model_type
            )

        pbar_dict = {}

        # Calculate evaluation metrics for each instrument
        for instr in instruments:
            if verbose:
                print(f"Instr: {instr}")

            # Read original instrument track as reference
            if instr != "other" or config.training.other_fix is False:
                track, sr1 = read_audio_transposed(
                    f"{folder}/{instr}.{extension}", instr, skip_err=True
                )
                if track is None:
                    continue
            else:
                # For 'other' track, compute from vocals track
                track, sr1 = read_audio_transposed(f"{folder}/vocals.{extension}")
                track = mix_orig - track

            estimates = waveforms_orig[instr]

            # Resample to original sample rate
            if "sample_rate" in config.audio:
                if sr != config.audio["sample_rate"]:
                    estimates = librosa.resample(
                        estimates,
                        orig_sr=config.audio["sample_rate"],
                        target_sr=sr,
                        res_type="kaiser_best",
                    )
                    estimates = librosa.util.fix_length(estimates, size=orig_length)

            # Denormalize
            if "normalize" in config.inference:
                if config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            # Save separation results
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
                out_wav_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.wav"
                sf.write(out_wav_name, estimates.T, sr, subtype="FLOAT")
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
                instr,
                pbar_dict,
                mixture_paths=mixture_paths,
                verbose=verbose,
            )

        if verbose:
            print(f"Time for song: {time.time() - start_time:.2f} sec")

    return all_metrics


def compute_metric_avg(
    store_dir: str,
    args,
    instruments: list[str],
    config: DictConfig,
    all_metrics: dict[str, dict[str, list[float]]],
    start_time: float,
) -> dict[str, float]:
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
    config : DictConfig
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

    logging(
        logs,
        text=f"Num overlap: {config.inference.num_overlap}",
        verbose_logging=verbose_logging,
    )

    metric_avg = {}
    # Compute mean and standard deviation of metrics for each instrument
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])

            mean_val = metric_values.mean()
            std_val = metric_values.std()

            logging(
                logs,
                text=f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})",
                verbose_logging=verbose_logging,
            )
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val

    # Compute average metrics across all instruments
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            logging(
                logs,
                text=f"Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}",
                verbose_logging=verbose_logging,
            )
    logging(
        logs,
        text=f"Elapsed time: {time.time() - start_time:.2f} sec",
        verbose_logging=verbose_logging,
    )

    if store_dir:
        write_results_in_file(store_dir, logs)

    return metric_avg


def valid_rps_only(
    model: torch.nn.Module,
    args,
    config: DictConfig,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """
    Validate RPS-only model: compute RPS prediction metrics (MSE, neg_mse, R²).

    Instead of speech enhancement metrics, this evaluates how well the model's
    RPSPredictionHead predicts rotor speeds from noisy audio features.

    Returns dict with keys: 'rps_mse', 'neg_mse', 'rps_mae', 'rps_r2'
    The 'neg_mse' key (= -MSE) is compatible with ReduceLROnPlateau('max') scheduler.
    """
    import torch.nn.functional as F

    start_time = time.time()
    model.eval().to(device)

    all_mixtures_path = get_mixture_paths(
        args,
        verbose,
        config,
        getattr(config.inference, "extension", "wav") if hasattr(config, "inference") else "wav",
    )

    chunk_size = config.audio.chunk_size
    rps_length = getattr(config, "rps_length", None)
    sample_rate = config.audio.sample_rate

    all_mse = []
    all_mae = []

    # Collect all target RPS to compute global mean for R²
    all_rps_preds = []
    all_rps_targets = []

    with torch.no_grad():
        pbar = tqdm(all_mixtures_path, disable=not verbose)
        for path in pbar:
            folder = os.path.dirname(path)

            # Load mixture audio
            mix, sr = read_audio_transposed(path)
            if mix is None or sr is None:
                continue
            if sr != sample_rate:
                mix = librosa.resample(
                    mix, orig_sr=sr, target_sr=sample_rate, res_type="kaiser_best"
                )

            # Load RPS
            rps_path = os.path.join(folder, "rps.npy")
            if not os.path.exists(rps_path):
                if verbose:
                    print(f"Skipping {folder}: no rps.npy")
                continue
            rps_np = np.load(rps_path)  # (4, rps_samples)

            # Truncate/pad mix to chunk_size
            if mix.shape[-1] > chunk_size:
                mix = mix[..., :chunk_size]
            elif mix.shape[-1] < chunk_size:
                mix = np.pad(mix, ((0, 0), (0, chunk_size - mix.shape[-1])))

            # Truncate/pad RPS to rps_length
            if rps_length is not None:
                if rps_np.shape[-1] > rps_length:
                    rps_np = rps_np[..., :rps_length]
                elif rps_np.shape[-1] < rps_length:
                    rps_np = np.pad(rps_np, ((0, 0), (0, rps_length - rps_np.shape[-1])))

            # To tensors
            mix_t = torch.from_numpy(mix).float().unsqueeze(0).to(device)  # (1, C, T)
            rps_t = torch.from_numpy(rps_np).float().unsqueeze(0).to(device)  # (1, 4, rps_len)

            # Forward pass
            if args.model_type == "rps_predictor":
                # RPSPredictor: input (B, T) mono audio, output (B, 4, T_stft)
                rps_pred = model(mix_t.squeeze(1))
            else:
                model_out = model(mix_t, rps=rps_t)
                if isinstance(model_out, tuple):
                    _, rps_pred = model_out
                else:
                    continue  # No RPS prediction output

            if rps_pred is None:
                continue

            # Interpolate target RPS to match prediction length
            rps_target = F.interpolate(
                rps_t.float(),
                size=rps_pred.shape[-1],
                mode="linear",
                align_corners=False,
            )

            # Compute per-sample metrics
            mse = F.mse_loss(rps_pred, rps_target).item()
            mae = torch.mean(torch.abs(rps_pred - rps_target)).item()

            all_mse.append(mse)
            all_mae.append(mae)
            all_rps_preds.append(rps_pred.cpu())
            all_rps_targets.append(rps_target.cpu())

    # Aggregate metrics
    if len(all_mse) == 0:
        print("WARNING: No valid RPS samples found for validation")
        return {"rps_mse": 999.0, "neg_mse": -999.0, "rps_mae": 999.0, "rps_r2": -999.0}

    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)

    # Global R²: pool all predictions and targets
    all_preds_cat = torch.cat(all_rps_preds, dim=0)  # (N, 4, T)
    all_targets_cat = torch.cat(all_rps_targets, dim=0)
    ss_res = torch.sum((all_targets_cat - all_preds_cat) ** 2).item()
    ss_tot = torch.sum((all_targets_cat - all_targets_cat.mean()) ** 2).item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    elapsed = time.time() - start_time
    if verbose:
        print(f"RPS Validation: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, R²={r2:.4f} ({elapsed:.1f}s)")

    model.train()
    return {
        "rps_mse": avg_mse,
        "neg_mse": -avg_mse,  # Higher is better — compatible with scheduler 'max' mode
        "rps_mae": avg_mae,
        "rps_r2": r2,
    }


def valid(
    model: torch.nn.Module,
    args,
    config: DictConfig,
    device: torch.device,
    verbose: bool = False,
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
    store_dir = getattr(args, "store_dir", "")
    # Get audio encoding format
    if "extension" in config["inference"]:
        extension = config["inference"]["extension"]
    else:
        extension = getattr(args, "extension", "wav")

    # Get all mixture audio file paths
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)
    # Process audio files and compute evaluation metrics
    all_metrics = process_audio_files(
        all_mixtures_path, model, args, config, device, verbose, not verbose
    )
    instruments = prefer_target_instrument(config)

    # Compute average evaluation metrics
    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time)


def validate_in_subprocess(
    proc_id: int,
    queue: torch.multiprocessing.Queue,
    all_mixtures_path: list[str],
    model: torch.nn.Module,
    args,
    config: DictConfig,
    device: torch.device,
    return_dict,
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
    config : DictConfig
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
        metric: {instr: [] for instr in config.training.instruments} for metric in args.metrics
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
                    pbar_dict[f"{metric_name}_{instr}"] = (
                        f"{single_metrics[metric_name][instr][0]:.4f}"
                    )
        if proc_id == 0:
            progress_bar.update(current_step - progress_bar.n)
            progress_bar.set_postfix(pbar_dict)
    return_dict[proc_id] = all_metrics
    return


def run_parallel_validation(
    verbose: bool,
    all_mixtures_path: list[str],
    config: DictConfig,
    model: torch.nn.Module,
    device_ids: list[int],
    args,
    return_dict,
) -> None:
    """
    Run multi-process parallel validation.

    Args:
    ----------
    verbose : bool
        Whether to print detailed information
    all_mixtures_path : List[str]
        All mixture audio file paths
    config : DictConfig
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

    model = model.to("cpu")
    try:
        # Extract single model for multi-GPU training
        model = model.module  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType]
    except:
        pass

    queue = torch.multiprocessing.Queue()
    processes = []

    # Create a process for each device
    for i, device in enumerate(device_ids):
        if torch.cuda.is_available():
            device = f"cuda:{device}"
        else:
            device = torch.device("cpu")
        p = torch.multiprocessing.Process(
            target=validate_in_subprocess,
            args=(
                i,
                queue,
                all_mixtures_path,
                model,
                args,
                config,
                device,
                return_dict,
            ),
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
    config: DictConfig,
    device_ids: list[int],
    verbose: bool = False,
) -> dict[str, float]:
    """
    Execute validation on multiple GPUs.

    Args:
    ----------
    model : torch.nn.Module
        Source separation model
    args : dict
        Arguments dictionary
    config : DictConfig
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
    store_dir = getattr(args, "store_dir", "")
    # Get audio encoding format
    if "extension" in config["inference"]:
        extension = config["inference"]["extension"]
    else:
        extension = getattr(args, "extension", "wav")

    # Get all mixture audio file paths
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)

    # Create shared dictionary to store results
    return_dict = torch.multiprocessing.Manager().dict()

    # Run parallel validation
    run_parallel_validation(
        verbose, all_mixtures_path, config, model, device_ids, args, return_dict
    )

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


def parse_args(dict_args: dict | None) -> argparse.Namespace:
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
    parser.add_argument(
        "--model_type",
        type=str,
        default="mdx23c",
        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer,"
        " edge_bs_rof, swin_upernet, bandit, diffusion_buffer",
    )
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument(
        "--start_check_point",
        type=str,
        default="",
        help="Initial checkpoint to valid weights",
    )
    parser.add_argument("--valid_path", nargs="+", type=str, help="Validate path")
    parser.add_argument(
        "--store_dir", type=str, default="", help="Path to store results as wav file"
    )
    parser.add_argument(
        "--draw_spectro",
        type=float,
        default=0,
        help="If --store_dir is set then code will generate spectrograms for resulted stems as well."
        " Value defines for how many seconds os track spectrogram will be generated.",
    )
    parser.add_argument("--device_ids", nargs="+", type=int, default=0, help="List of gpu ids")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num_workers")
    parser.add_argument("--pin_memory", action="store_true", help="Dataloader pin_memory")
    parser.add_argument(
        "--extension", type=str, default="wav", help="Choose extension for validation"
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
        "While this triples the runtime, it reduces noise and slightly improves prediction quality.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=["sdr"],
        choices=[
            "sdr",
            "l1_freq",
            "si_sdr",
            "neg_log_wmse",
            "aura_stft",
            "aura_mrstft",
            "bleedless",
            "fullness",
        ],
        help="List of metrics to use.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="",
        help="Initial checkpoint to LoRA weights",
    )

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
        torch.multiprocessing.set_start_method("spawn")
    except Exception:
        pass

    # Get model and configuration
    model, config = get_model_from_config(args.model_type, args.config_path)

    # Load checkpoint
    if args.start_check_point:
        load_start_checkpoint(args, model, type_="valid")

    print(f"Instruments: {config.training.instruments}")

    # Set computing device
    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Run validation on CPU. It will be very slow...")

    # Choose validation method based on device count
    if torch.cuda.is_available() and len(device_ids) > 1:
        valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        valid(model, args, config, device, verbose=True)


if __name__ == "__main__":
    check_validation(None)
