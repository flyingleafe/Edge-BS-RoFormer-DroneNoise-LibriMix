__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

# Import required libraries
import argparse
import glob
import json  # For loading metadata files
import os
import time
import warnings

import librosa
import numpy as np
import pandas as pd  # For saving Excel files
import soundfile as sf
import torch
from omegaconf import DictConfig
from pesq import pesq  # For PESQ metric
from pystoi import stoi  # For STOI metric
from thop import profile  # For FLOPs calculation

# for model params calculation
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

    Parameters:
    ----------
    store_dir : str
        Directory to store logs, if empty logs are not stored
    logs : List[str]
        List to store log messages
    text : str
        Text message to log
    """

    print(text)
    if verbose_logging:
        logs.append(text)


def calculate_pesq(ref, est, orig_sr):
    # print(f"[DEBUG] ref shape: {ref.shape}, est shape: {est.shape}")
    # If 2D array, determine whether to convert to mono
    if ref.ndim == 2:
        if ref.shape[0] > 1:
            ref = librosa.to_mono(ref)
        else:
            ref = ref.squeeze(0)
    if est.ndim == 2:
        if est.shape[0] > 1:
            est = librosa.to_mono(est)
        else:
            est = est.squeeze(0)

    # Adjust sample rate
    target_sr = 16000 if orig_sr >= 16000 else 8000
    ref = librosa.resample(ref, orig_sr=orig_sr, target_sr=target_sr)
    est = librosa.resample(est, orig_sr=orig_sr, target_sr=target_sr)

    try:
        score = pesq(target_sr, ref, est, "wb" if target_sr == 16000 else "nb")
    except Exception as e:
        print(f"[DEBUG] PESQ calculation failed: {e}")
        score = np.nan  # Return NaN instead of None
    # print(f"[DEBUG] pesq score: {score}")
    return score


# Calculate STOI metric function
def calculate_stoi(ref, est, orig_sr, extended=False):
    """
    Calculate Short-Time Objective Intelligibility (STOI or ESTOI) metric.

    Parameters:
    ----------
    ref : numpy.ndarray
        Reference audio signal
    est : numpy.ndarray
        Estimated audio signal
    orig_sr : int
        Original sample rate
    extended : bool
        If True, compute ESTOI instead of STOI

    Returns:
    -------
    float
        STOI/ESTOI score, range [0, 1], higher is better
    """
    # If 2D array, convert to mono
    if ref.ndim == 2:
        if ref.shape[0] > 1:
            ref = librosa.to_mono(ref)
        else:
            ref = ref.squeeze(0)
    if est.ndim == 2:
        if est.shape[0] > 1:
            est = librosa.to_mono(est)
        else:
            est = est.squeeze(0)

    # STOI requires 10000Hz sample rate, need to resample
    target_sr = 10000
    ref = librosa.resample(ref, orig_sr=orig_sr, target_sr=target_sr)
    est = librosa.resample(est, orig_sr=orig_sr, target_sr=target_sr)

    try:
        score = stoi(ref, est, target_sr, extended=extended)
    except Exception as e:
        metric_name = "ESTOI" if extended else "STOI"
        print(f"[DEBUG] {metric_name} calculation failed: {e}")
        score = np.nan
    return score


def write_results_in_file(store_dir: str, logs: list[str]) -> None:
    """
    Write validation results to file.

    Parameters:
    ----------
    store_dir : str
        Directory to store result files
    results : List[str]
        List of results to write to file
    """
    with open(f"{store_dir}/results.txt", "w") as out:
        for item in logs:
            out.write(item + "\n")


def get_mixture_paths(args, verbose: bool, config: DictConfig, extension: str) -> list[str]:
    """
    Get mixture audio file paths for validation.

    Parameters:
    ----------
    valid_path : List[str]
        List of validation dataset directories
    verbose : bool
        Whether to print detailed information
    config : DictConfig
        Configuration object containing inference parameters like overlap and batch size
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
        part = sorted(glob.glob(f"{path}/**/mixture.{extension}", recursive=True))
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

    Parameters:
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
    metadata: dict,
    verbose: bool = False,
    is_tqdm: bool = True,
) -> tuple[dict[str, dict[str, list[float]]], list[dict]]:
    """
    Process audio files and perform source separation evaluation.

    Parameters:
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
        Running device (CPU or CUDA)
    metadata : dict
        Metadata information
    verbose : bool
        Whether to print detailed logs
    is_tqdm : bool
        Whether to display progress bar

    Returns:
    -------
    Tuple[Dict[str, Dict[str, List[float]]], List[dict]]
        Nested dictionary of evaluation metrics, outer key is metric name, inner key is instrument name
        results_data : List[dict]
            Result data for each audio file
    """
    # Calculate model FLOPs
    model = model.module  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType] if hasattr(model, 'module') else model

    # Determine if stereo based on model config, default to mono (stereo=False) if not set
    if hasattr(config, "model") and isinstance(config.model, dict):
        stereo = config.model.get("stereo", False)  # Default mono
    else:
        stereo = False

    # Generate corresponding input dimensions
    segment_length = config.audio.get("segment_length", 131584)
    channels = 2 if stereo else 1
    dummy_input = torch.randn(1, channels, segment_length).to(device)

    # Build dummy RPS input for RPS-conditioned models so profile() doesn't crash
    profile_inputs = (dummy_input,)
    use_rps = config.get("use_rps", False) or config.get("load_rps", False)
    if use_rps:
        rps_length = config.get("rps_length", 8500)
        num_rotors = config.get("num_rotors", 4)
        dummy_rps = torch.randn(1, num_rotors, rps_length).to(device)
        profile_inputs = (dummy_input, dummy_rps)

    try:
        model_profile = profile(model, inputs=profile_inputs, verbose=False)
        model_flops = model_profile[0] / 1e9  # Giga FLOPs
    except Exception:
        model_flops = 0.0  # Skip FLOPs counting if profile fails
    model.to(device)

    # Initialize results collection list
    results_data = []

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
    # If stoi not in metrics, add to all_metrics
    if "stoi" not in all_metrics:
        all_metrics["stoi"] = {instr: [] for instr in config.training.instruments}
    if "estoi" not in all_metrics and "estoi" in args.metrics:
        all_metrics["estoi"] = {instr: [] for instr in config.training.instruments}

    if is_tqdm:
        path_iter = tqdm(mixture_paths)

    # Process each mixture audio file
    for path in path_iter:
        start_time = time.time()
        # Read mixture audio
        mix, sr = read_audio_transposed(path)
        if mix is None or sr is None:
            continue
        mix_orig: np.ndarray = mix.copy()
        folder = os.path.dirname(path)
        # Extract sample_id and input_snr
        sample_id = os.path.basename(folder)
        input_snr = metadata.get(sample_id, {}).get("input_snr", None)

        # Resample to target sample rate
        if "sample_rate" in config.audio:
            if sr != config.audio["sample_rate"]:
                orig_length = mix.shape[-1]
                if verbose:
                    print(
                        f"Warning: sample rate is different. In config: {config.audio['sample_rate']} in file {path}: {sr}"
                    )
                mix = librosa.resample(
                    mix, orig_sr=sr, target_sr=config.audio["sample_rate"], res_type="kaiser_best"
                )

        if verbose:
            folder_name = os.path.abspath(folder)
            print(f"Song: {folder_name} Shape: {mix.shape}")

        # Audio normalization
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                mix, norm_params = normalize_audio(mix)

        # Reset memory statistics
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Load RPS data if model uses rotor conditioning
        rps = None
        if getattr(model, "use_rps", False):
            rps_path = os.path.join(folder, "rps.npy")
            if os.path.exists(rps_path):
                rps = np.load(rps_path)

        # Record inference start time
        inference_start = time.time()
        mix_tensor = torch.from_numpy(mix.copy())
        waveforms_orig = demix(
            config, model, mix_tensor, device, model_type=args.model_type, rps=rps
        )
        inference_end = time.time()
        duration = mix.shape[-1] / sr  # Calculate audio duration (seconds)
        rtf = (
            inference_end - inference_start
        ) / duration  # Calculate RTF = inference time / audio duration
        mem_usage = (
            torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
        )  # GPU memory usage

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
                # If other track, need to calculate from vocals track
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
            # If pesq in metrics, calculate PESQ metric additionally
            if "pesq" in args.metrics:
                track_metrics["pesq"] = calculate_pesq(track, estimates, sr)

            # Calculate STOI metric
            stoi_score = calculate_stoi(track, estimates, sr)
            track_metrics["stoi"] = stoi_score
            all_metrics["stoi"][instr].append(stoi_score)

            if "estoi" in args.metrics:
                estoi_score = calculate_stoi(track, estimates, sr, extended=True)
                track_metrics["estoi"] = estoi_score
                all_metrics["estoi"][instr].append(estoi_score)

            # Collect validation metrics data for each instrument to results_data
            row = {
                "ID": sample_id,
                "Input_SNR": input_snr,
                "sdr": track_metrics.get("sdr", None),
                "si_sdr": track_metrics.get("si_sdr", None),
                "l1_freq": track_metrics.get("l1_freq", None),
                "pesq": track_metrics.get("pesq", None),
                "stoi": stoi_score,
                "estoi": track_metrics.get("estoi", None),
                "RTF": rtf,
                "FLOPs_G": model_flops,
                "GPU_Mem_MB": mem_usage,
            }
            results_data.append(row)

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

    return all_metrics, results_data


def compute_metric_avg(
    store_dir: str,
    args,
    instruments: list[str],
    config: DictConfig,
    all_metrics: dict[str, dict[str, list[float]]],
    start_time: float,
) -> dict[str, float]:
    """
    Calculate and log average evaluation metrics for each instrument.

    Parameters:
    ----------
    store_dir : str
        Log storage directory
    args : dict
        Argument dictionary
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
        Average evaluation metrics across all instruments
    """

    logs = []
    if store_dir:
        logs.append(str(args))
        verbose_logging = True
    else:
        verbose_logging = False

    logging(
        logs, text=f"Num overlap: {config.inference.num_overlap}", verbose_logging=verbose_logging
    )

    metric_avg = {}
    # Calculate mean and standard deviation for each instrument's evaluation metrics
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

    # Calculate average metrics across all instruments
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


_SNR_LEVELS = [-30, -25, -20, -15, -10, -5, 0]


def _nearest_snr_level(snr):
    """Round SNR to nearest discrete level."""
    return min(_SNR_LEVELS, key=lambda l: abs(snr - l))


def compute_per_snr_summary(results_data, metrics, store_dir=None):
    """Compute and print per-SNR-bin mean metrics, and optionally save CSV."""
    if not results_data or not any(r.get("Input_SNR") is not None for r in results_data):
        return

    # Group by nearest SNR level
    from collections import defaultdict

    bins = defaultdict(list)
    for row in results_data:
        if row.get("Input_SNR") is None:
            continue
        level = _nearest_snr_level(row["Input_SNR"])
        bins[level].append(row)

    metric_names = [m for m in metrics if m not in ("sdr",)]  # skip sdr, keep the requested ones
    if not metric_names:
        metric_names = ["si_sdr"]  # fallback

    # Print header
    header = f"{'SNR':>6}"
    for m in metric_names:
        header += f" | {m:>10}"
    header += f" | {'N':>4}"
    print("\n--- Per-SNR Summary ---")
    print(header)
    print("-" * len(header))

    rows_for_csv = []
    for level in sorted(bins.keys()):
        samples = bins[level]
        row_str = f"{level:>6}"
        csv_row = {"snr_level": level, "n_samples": len(samples)}
        for m in metric_names:
            vals = [r[m] for r in samples if r.get(m) is not None and not np.isnan(r[m])]
            mean_val = np.mean(vals) if vals else float("nan")
            row_str += f" | {mean_val:>10.4f}"
            csv_row[m] = round(float(mean_val), 4) if vals else None
        row_str += f" | {len(samples):>4}"
        print(row_str)
        rows_for_csv.append(csv_row)

    # Save CSV if store_dir provided
    if store_dir:
        import csv as csv_mod

        parent_dir = os.path.dirname(store_dir)
        csv_path = os.path.join(parent_dir, os.path.basename(store_dir) + "_per_snr.csv")
        os.makedirs(parent_dir, exist_ok=True)
        fieldnames = ["snr_level"] + metric_names + ["n_samples"]
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"Per-SNR summary saved to {csv_path}")


def valid(
    model: torch.nn.Module, args, config: DictConfig, device: torch.device, verbose: bool = False
) -> dict:
    """
    Validate model on a single device.

    Parameters:
    ----------
    model : torch.nn.Module
        Source separation model
    args : Namespace
        Command line arguments
    config : dict
        Configuration dictionary
    device : torch.device
        Running device
    verbose : bool
        Whether to print detailed information

    Returns:
    -------
    dict
        Average evaluation metrics across all instruments
    """

    start_time = time.time()
    model.eval().to(device)

    # Load metadata and convert to dictionary keyed by id
    metadata_path = os.path.join(args.valid_path[0], "metadata.json")
    if not os.path.exists(metadata_path):
        # metadata.json may be at the dataset root (parent of valid/)
        metadata_path = os.path.join(os.path.dirname(args.valid_path[0]), "metadata.json")
    with open(metadata_path) as f:
        metadata_json = json.load(f)
    # If json contains "valid" key, convert to dictionary
    if "valid" in metadata_json:
        metadata = {entry["id"]: entry for entry in metadata_json["valid"]}
    else:
        metadata = metadata_json

    # Get storage directory and config extension, create subfolder related to config file as output directory
    store_dir = getattr(args, "store_dir", "")
    if store_dir:
        subfolder = os.path.splitext(os.path.basename(args.config_path))[0]
        store_dir = os.path.join(store_dir, subfolder)
        args.store_dir = store_dir
    if "extension" in config["inference"]:
        extension = config["inference"]["extension"]
    else:
        extension = getattr(args, "extension", "wav")

    # Get all mixture audio file paths
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)
    # Call with metadata parameter and receive results_data
    all_metrics, results_data = process_audio_files(
        all_mixtures_path, model, args, config, device, metadata, verbose, not verbose
    )

    # Save results to Excel file and remove "instrument" column
    if store_dir:
        parent_dir = os.path.dirname(store_dir)
        os.makedirs(parent_dir, exist_ok=True)
        excel_filename = (
            os.path.splitext(os.path.basename(args.config_path))[0] + "_validation.xlsx"
        )
        excel_path = os.path.join(parent_dir, excel_filename)
        df = pd.DataFrame(results_data)
        # Remove instrument column if exists
        if "instrument" in df.columns:
            df.drop(columns=["instrument"], inplace=True)
        df.to_excel(excel_path, index=False)
        print(f"Validation results saved to {excel_path}")

    # Per-SNR summary
    compute_per_snr_summary(results_data, args.metrics, store_dir)

    # Continue with existing code
    instruments = prefer_target_instrument(config)
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
    Execute validation in subprocess, supports multi-process parallel processing.

    Parameters:
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
        Argument dictionary
    config : DictConfig
        Configuration object
    device : str
        Running device
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
        single_metrics = process_audio_files([path], m1, args, config, device, False, False)  # pyright: ignore[reportArgumentType]
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

    Parameters:
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
        Argument dictionary
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
            device = "cpu"
        p = torch.multiprocessing.Process(
            target=validate_in_subprocess,
            args=(i, queue, all_mixtures_path, model, args, config, device, return_dict),
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
    model: torch.nn.Module, args, config: DictConfig, device_ids: list[int], verbose: bool = False
) -> dict[str, float]:
    """
    Execute validation on multiple GPUs.

    Parameters:
    ----------
    model : torch.nn.Module
        Source separation model
    args : dict
        Argument dictionary
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

    # Calculate average evaluation metrics
    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time)


def parse_args(dict_args: dict | None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters:
    ----------
    dict_args: Dict
        Command line argument dictionary, if None parses from sys.argv

    Returns:
    -------
    argparse.Namespace
        Parsed argument object
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
        "--start_check_point", type=str, default="", help="Initial checkpoint to valid weights"
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
            "pesq",
            "stoi",
            "estoi",
        ],
        help="List of metrics to use.",
    )
    parser.add_argument(
        "--lora_checkpoint", type=str, default="", help="Initial checkpoint to LoRA weights"
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

    Parameters:
    ----------
    dict_args
        Command line argument dictionary
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

    # Set running device
    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Run validation on CPU. It will be very slow...")

    # Select validation method based on number of devices
    if torch.cuda.is_available() and len(device_ids) > 1:
        valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        valid(model, args, config, device, verbose=True)


if __name__ == "__main__":
    check_validation(None)
