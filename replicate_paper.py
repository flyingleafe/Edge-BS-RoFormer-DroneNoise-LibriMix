#!/usr/bin/env python3
"""
Replication script for Edge-BS-RoFormer paper.

"Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement"

Provides idempotent steps to replicate the paper: setup, download, dataset,
train, and evaluate. Each step skips work if already done.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Callable, NoReturn

import yaml

from create_dataset import create_dataset as create_dataset_impl
from final_valid import check_validation
from train import train_model

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATASETS_DIR = PROJECT_DIR / "datasets"
RESULTS_DIR = PROJECT_DIR / "results"

LIBRISPEECH_DIR = DATA_DIR / "librispeech"
DRONE_AUDIO_DIR = DATA_DIR / "drone_audio"
DNLM_DIR = DATASETS_DIR / "DN-LM"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_ARCHIVE_NAME = "train-clean-100.tar.gz"
LIBRISPEECH_EXTRACTED = LIBRISPEECH_DIR / "LibriSpeech" / "train-clean-100"
LIBRISPEECH_MIN_ARCHIVE_BYTES = 1_000_000_000  # ~1 GB, incomplete otherwise

DRONE_AUDIO_REPO = "https://github.com/saraalemadi/DroneAudioDataset.git"

TRAIN_PATH = DNLM_DIR / "train"
VALID_PATH = DNLM_DIR / "valid"
EVAL_STORE_DIR = RESULTS_DIR / "evaluation"

# Dataset creation defaults (paper: 2h total, 9:1 split, 1s samples)
DATASET_TRAIN_SAMPLES = 6480
DATASET_VALID_SAMPLES = 720
DATASET_SAMPLE_DURATION = 1.0
DATASET_SAMPLE_RATE = 16000
DATASET_SNR_MIN = -30
DATASET_SNR_MAX = 0
DATASET_SEED = 42

# Minimum samples to consider dataset complete
MIN_TRAIN_SAMPLES = 6000
MIN_VALID_SAMPLES = 500

# Model definitions: (model_type, config_path, results_subdir)
MODELS = {
    "rope": ("edge_bs_rof", "configs/3_FA_RoPE(64).yaml", "edge_bs_roformer"),
    "rope_smaller": ("edge_bs_rof", "configs/3_FA_RoPE(48).yaml", "edge_bs_roformer_smaller"),
    "dcunet": ("dcunet", "configs/5_Baseline_dcunet.yaml", "dcunet"),
    "dptnet": ("dptnet", "configs/7_Baseline_dptnet.yaml", "dptnet"),
    "htdemucs": ("htdemucs", "configs/8_Baseline_htdemucs.yaml", "htdemucs"),
}
EVAL_ONLY_MODELS = {
    "diffusion": ("diffusion_buffer", "configs/9_Diffusion_Buffer_BBED.yaml", "diffusion_buffer_bbed"),
}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

class _Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"


def _log(prefix: str, color: str, msg: str) -> None:
    print(f"{color}{prefix}{_Colors.NC} {msg}")


def log_info(msg: str) -> None:
    _log("[INFO]", _Colors.BLUE, msg)


def log_success(msg: str) -> None:
    _log("[SUCCESS]", _Colors.GREEN, msg)


def log_warning(msg: str) -> None:
    _log("[WARNING]", _Colors.YELLOW, msg)


def log_error(msg: str) -> None:
    _log("[ERROR]", _Colors.RED, msg)


def fail(msg: str, code: int = 1) -> NoReturn:
    log_error(msg)
    sys.exit(code)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def check_command(name: str) -> None:
    if shutil.which(name) is None:
        fail(f"Required command '{name}' not found. Please install it first.")


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    cwd = cwd or PROJECT_DIR
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        capture_output=capture,
        text=True,
    )


def count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.rglob(pattern)))


def count_sample_dirs(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_dir() and p.name.startswith("sample_"))


def get_git_status() -> subprocess.CompletedProcess:
    """Check git status to determine if tree is dirty."""
    try:
        return run(["git", "status", "--porcelain"], cwd=PROJECT_DIR, capture=True)
    except subprocess.CalledProcessError:
        fail("Could not check git status. Make sure this is a git repository.")
    except FileNotFoundError:
        fail("git command not found. Please install git.")


def get_git_short_hash() -> str:
    """Get short SHA of current Git commit."""
    try:
        result = run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR, capture=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        fail("Could not get git commit hash. Make sure this is a git repository.")
    except FileNotFoundError:
        fail("git command not found. Please install git.")


def ensure_clean_git_tree() -> None:
    """Ensure the Git working tree is clean (no uncommitted changes)."""
    result = get_git_status()
    if result.stdout.strip():
        log_error("Git working tree is dirty. Uncommitted changes detected:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                log_error(f"  {line.strip()}")
        fail("Please commit or stash your changes before proceeding.")
    log_info("Git working tree is clean.")


def get_num_gpus_from_config(config_path: Path) -> int:
    """Read num_gpus from training config; default to 1, max 2."""
    if not config_path.exists():
        fail(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader) or {}
    except Exception as exc:
        fail(f"Failed to read config {config_path}: {exc}")
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    num_gpus = training_cfg.get("num_gpus", 1)
    try:
        num_gpus = int(num_gpus)
    except (TypeError, ValueError) as exc:
        fail(f"Invalid num_gpus in {config_path}: {exc}")
    if num_gpus < 1 or num_gpus > 2:
        fail(f"num_gpus in {config_path} must be 1 or 2 (got {num_gpus})")
    return num_gpus


# -----------------------------------------------------------------------------
# Step 0: Setup
# -----------------------------------------------------------------------------

def step_setup() -> None:
    log_info("=== Step 0: Setting up environment ===")
    check_command("git")
    check_command("tar")

    venv_uv = PROJECT_DIR / ".venv"
    venv_pip = PROJECT_DIR / "venv"

    if shutil.which("uv") is not None:
        log_info("Using uv for package management")
        if not venv_uv.exists():
            log_info("Creating virtual environment with uv...")
            run(["uv", "venv"], cwd=PROJECT_DIR)
        else:
            log_info("Virtual environment already exists")
        log_info("Installing dependencies with uv...")
        run(["uv", "sync"], cwd=PROJECT_DIR)
        log_success("Environment setup complete with uv")
    else:
        log_warning("uv not found, falling back to pip")
        check_command("python")
        if not venv_pip.exists():
            log_info("Creating virtual environment...")
            run([sys.executable, "-m", "venv", "venv"], cwd=PROJECT_DIR)
        pip = venv_pip / "bin" / "pip"
        if not pip.exists():
            pip = venv_pip / "Scripts" / "pip.exe"
        req = PROJECT_DIR / "requirements.txt"
        if not req.exists():
            fail("requirements.txt not found")
        run([str(pip), "install", "-r", str(req)], cwd=PROJECT_DIR)
        log_success("Environment setup complete with pip")


# -----------------------------------------------------------------------------
# Step 1: Download
# -----------------------------------------------------------------------------

def step_download() -> None:
    log_info("=== Step 1: Downloading source datasets ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # LibriSpeech
    log_info("Checking LibriSpeech dataset...")
    flac_count = count_files(LIBRISPEECH_EXTRACTED, "*.flac")
    if LIBRISPEECH_EXTRACTED.exists() and flac_count > 0:
        log_success(f"LibriSpeech already downloaded and extracted ({flac_count} FLAC files found)")
    else:
        LIBRISPEECH_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = LIBRISPEECH_DIR / LIBRISPEECH_ARCHIVE_NAME
        if not archive_path.exists() or archive_path.stat().st_size < LIBRISPEECH_MIN_ARCHIVE_BYTES:
            log_info("Downloading LibriSpeech train-clean-100 (~6GB)...")
            check_command("wget")
            run(
                ["wget", "-c", "--progress=bar:force", LIBRISPEECH_URL, "-O", str(archive_path)],
                cwd=LIBRISPEECH_DIR,
            )
            log_success("LibriSpeech download complete")
        else:
            log_info("LibriSpeech archive already exists, skipping download")

        extract_dir = LIBRISPEECH_DIR / "LibriSpeech" / "train-clean-100"
        if not extract_dir.exists():
            log_info("Extracting LibriSpeech (~6GB archive)...")
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(LIBRISPEECH_DIR)
            log_success("LibriSpeech extraction complete")
        else:
            log_success("LibriSpeech already extracted")

    # DroneAudioDataset
    log_info("Checking DroneAudioDataset...")
    wav_count = count_files(DRONE_AUDIO_DIR, "*.wav") + count_files(DRONE_AUDIO_DIR, "*.WAV")
    if DRONE_AUDIO_DIR.exists() and wav_count > 0:
        log_success(f"DroneAudioDataset already downloaded ({wav_count} WAV files found)")
    else:
        log_info("Cloning DroneAudioDataset from GitHub...")
        check_command("git")
        if DRONE_AUDIO_DIR.exists():
            shutil.rmtree(DRONE_AUDIO_DIR)
        run(["git", "clone", DRONE_AUDIO_REPO, str(DRONE_AUDIO_DIR)])
        log_success("DroneAudioDataset cloned successfully")

    log_success("All source datasets ready")


# -----------------------------------------------------------------------------
# Step 2: Dataset
# -----------------------------------------------------------------------------

def step_dataset() -> None:
    log_info("=== Step 2: Creating DroneNoise-LibriMix dataset ===")
    train_count = count_sample_dirs(TRAIN_PATH)
    valid_count = count_sample_dirs(VALID_PATH)
    train_meta = TRAIN_PATH / "metadata.json"
    valid_meta = VALID_PATH / "metadata.json"

    if (
        TRAIN_PATH.exists()
        and VALID_PATH.exists()
        and train_meta.exists()
        and valid_meta.exists()
        and train_count >= MIN_TRAIN_SAMPLES
        and valid_count >= MIN_VALID_SAMPLES
    ):
        log_success("DN-LM dataset already exists")
        log_info(f"  Training samples: {train_count}")
        log_info(f"  Validation samples: {valid_count}")
        return

    if (TRAIN_PATH.exists() or VALID_PATH.exists()) and (train_count > 0 or valid_count > 0):
        log_warning(f"Partial DN-LM dataset found (train: {train_count}, valid: {valid_count})")
        log_warning("Removing and recreating for consistency...")
        if DNLM_DIR.exists():
            shutil.rmtree(DNLM_DIR)

    if not LIBRISPEECH_EXTRACTED.exists():
        fail(
            f"LibriSpeech not found at {LIBRISPEECH_EXTRACTED}. Run: python replicate_paper.py download"
        )
    if not DRONE_AUDIO_DIR.exists():
        fail(
            f"DroneAudioDataset not found at {DRONE_AUDIO_DIR}. Run: python replicate_paper.py download"
        )

    speech_count = count_files(LIBRISPEECH_EXTRACTED, "*.flac")
    noise_count = count_files(DRONE_AUDIO_DIR, "*.wav") + count_files(DRONE_AUDIO_DIR, "*.WAV")
    if noise_count == 0:
        noise_count = count_files(DRONE_AUDIO_DIR, "*.mp3") + count_files(DRONE_AUDIO_DIR, "*.ogg")
    if speech_count == 0:
        fail(f"No speech files (*.flac) in {LIBRISPEECH_EXTRACTED}")
    if noise_count == 0:
        fail(f"No noise/audio files in {DRONE_AUDIO_DIR}")

    log_info(f"Found {speech_count} speech files and {noise_count} noise files")
    log_info("Creating DN-LM dataset (this may take a while)...")
    log_info(f"  Training samples: {DATASET_TRAIN_SAMPLES}")
    log_info(f"  Validation samples: {DATASET_VALID_SAMPLES}")
    log_info("  SNR range: -30 to 0 dB")

    speech_dir = str(LIBRISPEECH_EXTRACTED)
    noise_dir = str(DRONE_AUDIO_DIR)
    output_dir = str(DNLM_DIR)
    snr_range = (DATASET_SNR_MIN, DATASET_SNR_MAX)

    print("\n=== Creating Training Set ===")
    create_dataset_impl(
        speech_dir=speech_dir,
        noise_dir=noise_dir,
        output_dir=output_dir,
        num_samples=DATASET_TRAIN_SAMPLES,
        sample_duration=DATASET_SAMPLE_DURATION,
        sample_rate=DATASET_SAMPLE_RATE,
        target_snr_range=snr_range,
        split="train",
        seed=DATASET_SEED,
    )
    print("\n=== Creating Validation Set ===")
    create_dataset_impl(
        speech_dir=speech_dir,
        noise_dir=noise_dir,
        output_dir=output_dir,
        num_samples=DATASET_VALID_SAMPLES,
        sample_duration=DATASET_SAMPLE_DURATION,
        sample_rate=DATASET_SAMPLE_RATE,
        target_snr_range=snr_range,
        split="valid",
        seed=DATASET_SEED + 1,
    )
    print("\n=== Dataset Creation Complete ===")
    print(f"Dataset saved to: {output_dir}")
    log_success("DN-LM dataset creation complete")


# -----------------------------------------------------------------------------
# Step 3: Train
# -----------------------------------------------------------------------------

def _train_setup() -> None:
    if not TRAIN_PATH.exists() or not VALID_PATH.exists():
        fail("DN-LM dataset not found. Run: python replicate_paper.py dataset")

    # Check that git tree is clean before training
    ensure_clean_git_tree()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _train_one(model_key: str, device_ids: list[int] | None) -> None:
    if model_key not in MODELS:
        fail(f"Unknown model: {model_key}")
    model_type, config_rel, results_subdir = MODELS[model_key]
    config_path = PROJECT_DIR / config_rel

    # Get current git commit hash and use it as suffix for results directory
    commit_hash = get_git_short_hash()
    results_subdir_with_hash = f"{results_subdir}_{commit_hash}"
    results_path = RESULTS_DIR / results_subdir_with_hash

    num_gpus = get_num_gpus_from_config(config_path)
    if device_ids is None:
        device_ids = [0, 1] if num_gpus == 2 else [0]
    elif num_gpus == 2:
        if len(device_ids) < 2:
            fail(f"{results_subdir} requires 2 GPUs (num_gpus=2 in config)")
        device_ids = device_ids[:2]
    else:
        device_ids = device_ids[:1]

    ckpt = results_path / "best_model.ckpt"
    if ckpt.exists():
        log_success(f"{results_subdir_with_hash} already trained")
        return
    log_info(
        f"Training {results_subdir_with_hash} (commit: {commit_hash}, gpus: {num_gpus})..."
    )
    train_model({
        "model_type": model_type,
        "config_path": str(config_path),
        "results_path": str(results_path),
        "data_path": [str(TRAIN_PATH)],
        "valid_path": [str(VALID_PATH)],
        "dataset_type": 1,
        "device_ids": device_ids,
        "num_workers": 4,
        "metrics": ["si_sdr", "sdr"],
        "metric_for_scheduler": "si_sdr",
    })
    log_success(f"{results_subdir_with_hash} training complete")


def step_train_rope(device_ids: list[int] | None = None) -> None:
    log_info("=== Training Edge-BS-RoFormer (RoPE) ===")
    _train_setup()
    _train_one("rope", device_ids)


def step_train_rope_smaller(device_ids: list[int] | None = None) -> None:
    log_info("=== Training Edge-BS-RoFormer (RoPE 48) ===")
    _train_setup()
    _train_one("rope_smaller", device_ids)


def step_train_dcunet(device_ids: list[int] | None = None) -> None:
    log_info("=== Training DCUNet ===")
    _train_setup()
    _train_one("dcunet", device_ids)


def step_train_dptnet(device_ids: list[int] | None = None) -> None:
    log_info("=== Training DPTNet ===")
    _train_setup()
    _train_one("dptnet", device_ids)


def step_train_htdemucs(device_ids: list[int] | None = None) -> None:
    log_info("=== Training HTDemucs ===")
    _train_setup()
    _train_one("htdemucs", device_ids)


def step_train(device_ids: list[int] | None = None) -> None:
    log_info("=== Step 3: Training all models sequentially ===")
    log_info("For parallel training, run each model in a separate terminal:")
    log_info("  python replicate_paper.py train_rope")
    log_info("  python replicate_paper.py train_dcunet")
    log_info("  python replicate_paper.py train_dptnet")
    log_info("  python replicate_paper.py train_htdemucs")
    print()
    step_train_rope(device_ids)
    step_train_dcunet(device_ids)
    step_train_dptnet(device_ids)
    step_train_htdemucs(device_ids)
    log_success("All model training complete")


# -----------------------------------------------------------------------------
# Step 4: Evaluate
# -----------------------------------------------------------------------------

def _eval_setup() -> None:
    # Check that git tree is clean before evaluation
    ensure_clean_git_tree()
    EVAL_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _eval_one(
    model_key: str,
    models_map: dict[str, tuple[str, str, str]],
    device_id: int | None = None,
) -> None:
    if model_key not in models_map:
        fail(f"Unknown model: {model_key}")
    model_type, config_rel, results_subdir = models_map[model_key]
    config_path = PROJECT_DIR / config_rel

    # Use current git commit hash as suffix for results directory
    commit_hash = get_git_short_hash()
    results_subdir_with_hash = f"{results_subdir}_{commit_hash}"
    ckpt = RESULTS_DIR / results_subdir_with_hash / "best_model.ckpt"
    if not ckpt.exists():
        log_warning(
            f"{results_subdir_with_hash} checkpoint not found, skipping evaluation"
        )
        return
    device_id = 0 if device_id is None else device_id
    log_info(
        f"Evaluating {results_subdir_with_hash} (commit: {commit_hash}, gpu: {device_id})..."
    )
    check_validation({
        "model_type": model_type,
        "config_path": str(config_path),
        "start_check_point": str(ckpt),
        "valid_path": [str(VALID_PATH)],
        "store_dir": str(EVAL_STORE_DIR),
        "device_ids": [device_id],
        "metrics": ["si_sdr", "sdr", "pesq", "stoi"],
    })
    log_success(f"{results_subdir_with_hash} evaluation complete")


def step_eval_rope() -> None:
    log_info("=== Evaluating Edge-BS-RoFormer (RoPE) ===")
    _eval_setup()
    _eval_one("rope", MODELS)


def step_eval_dcunet() -> None:
    log_info("=== Evaluating DCUNet ===")
    _eval_setup()
    _eval_one("dcunet", MODELS)


def step_eval_dptnet() -> None:
    log_info("=== Evaluating DPTNet ===")
    _eval_setup()
    _eval_one("dptnet", MODELS)


def step_eval_htdemucs() -> None:
    log_info("=== Evaluating HTDemucs ===")
    _eval_setup()
    _eval_one("htdemucs", MODELS)


def step_eval_diffusion() -> None:
    log_info("=== Evaluating Diffusion Buffer ===")
    _eval_setup()
    _eval_one("diffusion", EVAL_ONLY_MODELS)


def step_eval() -> None:
    log_info("=== Step 4: Evaluating all models sequentially ===")
    log_info("For parallel evaluation, run each model in a separate terminal:")
    log_info("  python replicate_paper.py eval_rope")
    log_info("  python replicate_paper.py eval_dcunet")
    log_info("  python replicate_paper.py eval_dptnet")
    log_info("  python replicate_paper.py eval_htdemucs")
    log_info("  python replicate_paper.py eval_diffusion")
    print()
    step_eval_rope()
    step_eval_dcunet()
    step_eval_dptnet()
    step_eval_htdemucs()
    step_eval_diffusion()
    log_success("All model evaluations complete")
    log_info(f"Results saved to {EVAL_STORE_DIR}/")


# -----------------------------------------------------------------------------
# Step 5: Train + Eval Scheduler
# -----------------------------------------------------------------------------


def _normalize_model_list(models: list[str] | None) -> list[str]:
    all_models = list(MODELS.keys()) + list(EVAL_ONLY_MODELS.keys())
    if not models:
        return all_models
    normalized = []
    for model in models:
        if model not in all_models:
            fail(f"Unknown model: {model}")
        if model not in normalized:
            normalized.append(model)
    return normalized


def _run_train_eval_for_model(model_key: str, device_id: int) -> None:
    _train_setup()
    _train_one(model_key, [device_id])
    _eval_setup()
    _eval_one(model_key, MODELS, device_id=device_id)


def _run_eval_only_for_model(model_key: str, device_id: int) -> None:
    _eval_setup()
    _eval_one(model_key, EVAL_ONLY_MODELS, device_id=device_id)


def step_train_eval_all(models: list[str] | None = None) -> None:
    log_info("=== Step 5: Train + Eval scheduler ===")
    selected = _normalize_model_list(models)

    two_gpu_train_models: list[str] = []
    one_gpu_tasks: list[tuple[str, str]] = []

    for model_key in selected:
        if model_key in MODELS:
            _, config_rel, _ = MODELS[model_key]
            num_gpus = get_num_gpus_from_config(PROJECT_DIR / config_rel)
            if num_gpus == 2:
                two_gpu_train_models.append(model_key)
            else:
                one_gpu_tasks.append(("train_eval", model_key))
        else:
            one_gpu_tasks.append(("eval_only", model_key))

    if two_gpu_train_models:
        log_info("Running all 2-GPU training tasks sequentially...")
        for model_key in two_gpu_train_models:
            _train_setup()
            _train_one(model_key, [0, 1])
        log_success("2-GPU training tasks complete")

    for model_key in two_gpu_train_models:
        one_gpu_tasks.append(("eval_only_trained", model_key))

    if not one_gpu_tasks:
        log_success("No 1-GPU tasks to run")
        return

    log_info("Running 1-GPU tasks in parallel across GPU 0 and GPU 1...")
    stream_tasks = [[], []]
    for idx, task in enumerate(one_gpu_tasks):
        stream_tasks[idx % 2].append(task)

    def _run_stream(gpu_id: int, tasks: list[tuple[str, str]]) -> None:
        for task_type, model_key in tasks:
            if task_type == "train_eval":
                _run_train_eval_for_model(model_key, gpu_id)
            elif task_type == "eval_only":
                _run_eval_only_for_model(model_key, gpu_id)
            else:
                _eval_setup()
                _eval_one(model_key, MODELS, device_id=gpu_id)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_run_stream, 0, stream_tasks[0]),
            executor.submit(_run_stream, 1, stream_tasks[1]),
        ]
        for future in futures:
            future.result()

    log_success("Scheduled train/eval tasks complete")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

STEPS: dict[str, Callable[[], None]] = {
    "setup": step_setup,
    "download": step_download,
    "dataset": step_dataset,
    "train": step_train,
    "train_rope": step_train_rope,
    "train_rope_smaller": step_train_rope_smaller,
    "train_dcunet": step_train_dcunet,
    "train_dptnet": step_train_dptnet,
    "train_htdemucs": step_train_htdemucs,
    "eval": step_eval,
    "eval_rope": step_eval_rope,
    "eval_dcunet": step_eval_dcunet,
    "eval_dptnet": step_eval_dptnet,
    "eval_htdemucs": step_eval_htdemucs,
    "eval_diffusion": step_eval_diffusion,
    "train_eval_all": step_train_eval_all,
}


def _usage(prog: str) -> str:
    return f"""Usage: {prog} [step]

Steps:
  setup              Set up Python environment (uv or pip)
  download           Download source datasets (LibriSpeech + DroneAudioDataset)
  dataset            Create DN-LM dataset from source datasets
  train              Train all models sequentially
  train_rope         Train Edge-BS-RoFormer only (proposed method)
  train_rope_smaller Train Edge-BS-RoFormer smaller (48 dim)
  train_dcunet       Train DCUNet baseline only
  train_dptnet       Train DPTNet baseline only
  train_htdemucs     Train HTDemucs baseline only
  eval               Evaluate all trained models sequentially
  eval_rope          Evaluate Edge-BS-RoFormer only
  eval_dcunet        Evaluate DCUNet only
  eval_dptnet        Evaluate DPTNet only
  eval_htdemucs      Evaluate HTDemucs only
  eval_diffusion     Evaluate Diffusion Buffer only
  train_eval_all     Schedule train+eval across selected models
  all                Run all steps (default)

Examples:
  {prog}                  # Run all steps
  {prog} download         # Only download datasets
  {prog} train            # Train all models
  {prog} eval             # Evaluate all models
  {prog} train_eval_all --models rope dcunet  # Train+eval subset
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate Edge-BS-RoFormer paper results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_usage(f"python replicate_paper.py"),
    )
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=list(STEPS.keys()) + ["all"],
        help="Replication step to run",
    )
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="+",
        default=None,
        help="GPU device IDs for training (e.g. 0 1). Default: 0 1 for rope, 0 for others.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODELS.keys()) + list(EVAL_ONLY_MODELS.keys()),
        help=(
            "Models for train_eval_all (default: all). "
            "Eval-only models are allowed (e.g. diffusion)."
        ),
    )
    args = parser.parse_args()

    print("==============================================")
    print(" Edge-BS-RoFormer Paper Replication Script")
    print("==============================================")
    print()

    device_ids = args.device_ids

    if args.step == "all":
        step_setup()
        step_download()
        step_dataset()
        step_train(device_ids)
        step_eval()
        print()
        log_success("=== All steps completed successfully ===")
    else:
        step_fn = STEPS[args.step]
        if args.step == "train_eval_all":
            step_train_eval_all(args.models)
        elif args.step == "train" or args.step.startswith("train_"):
            step_fn(device_ids)
        else:
            step_fn()


if __name__ == "__main__":
    main()
