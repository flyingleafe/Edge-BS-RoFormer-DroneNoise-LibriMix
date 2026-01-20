#!/usr/bin/env bash
#
# Replication Script for Edge-BS-RoFormer Paper
# "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement"
#
# This script provides commands to replicate the paper's results.
# The script is idempotent - it checks if each step has been completed before running.
#
# Usage: ./replicate_paper.sh [step]
#   step: Optional. One of: setup, download, dataset, train, eval, all
#         Default: all

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
DATASETS_DIR="${PROJECT_DIR}/datasets"
RESULTS_DIR="${PROJECT_DIR}/results"

# Dataset paths
LIBRISPEECH_DIR="${DATA_DIR}/librispeech"
DRONE_AUDIO_DIR="${DATA_DIR}/drone_audio"
DNLM_DIR="${DATASETS_DIR}/DN-LM"

# LibriSpeech download URL (train-clean-100 subset, ~6GB)
LIBRISPEECH_URL="https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_ARCHIVE="train-clean-100.tar.gz"

# DroneAudioDataset GitHub repo
DRONE_AUDIO_REPO="https://github.com/saraalemadi/DroneAudioDataset.git"

# Training/validation data paths
TRAIN_PATH="${DNLM_DIR}/train"
VALID_PATH="${DNLM_DIR}/valid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command '$1' not found. Please install it first."
        exit 1
    fi
}

# ============================================================================
# STEP 0: Environment Setup
# ============================================================================
step_setup() {
    log_info "=== Step 0: Setting up environment ==="

    cd "${PROJECT_DIR}"

    # Check for required commands
    check_command "git"
    check_command "wget"
    check_command "tar"

    # Check for uv
    if command -v uv &> /dev/null; then
        log_info "Using uv for package management"

        # Check if venv exists
        if [ ! -d ".venv" ]; then
            log_info "Creating virtual environment with uv..."
            uv venv
        else
            log_info "Virtual environment already exists"
        fi

        # Activate virtual environment
        source .venv/bin/activate

        # Install dependencies
        log_info "Installing dependencies with uv..."
        uv pip install -r requirements.txt

        log_success "Environment setup complete with uv"
    else
        log_warning "uv not found, falling back to pip"
        check_command "python"

        if [ ! -d "venv" ]; then
            log_info "Creating virtual environment..."
            python -m venv venv
        fi

        source venv/bin/activate
        pip install -r requirements.txt

        log_success "Environment setup complete with pip"
    fi
}

# Activate virtual environment if exists (for use in other steps)
activate_venv() {
    if [ -d "${PROJECT_DIR}/.venv" ]; then
        source "${PROJECT_DIR}/.venv/bin/activate"
    elif [ -d "${PROJECT_DIR}/venv" ]; then
        source "${PROJECT_DIR}/venv/bin/activate"
    fi
}

# ============================================================================
# STEP 1: Download Source Datasets
# ============================================================================
step_download() {
    log_info "=== Step 1: Downloading source datasets ==="

    mkdir -p "${DATA_DIR}"

    # --- Download LibriSpeech ---
    log_info "Checking LibriSpeech dataset..."

    LIBRISPEECH_EXTRACTED="${LIBRISPEECH_DIR}/LibriSpeech/train-clean-100"

    # Check if we have FLAC files (LibriSpeech uses FLAC format)
    LIBRI_FLAC_COUNT=$(find "${LIBRISPEECH_EXTRACTED}" -name "*.flac" 2>/dev/null | wc -l)

    if [ -d "${LIBRISPEECH_EXTRACTED}" ] && [ "${LIBRI_FLAC_COUNT}" -gt 0 ]; then
        log_success "LibriSpeech already downloaded and extracted (${LIBRI_FLAC_COUNT} FLAC files found)"
    else
        mkdir -p "${LIBRISPEECH_DIR}"
        cd "${LIBRISPEECH_DIR}"

        # Download if archive doesn't exist or is incomplete
        if [ ! -f "${LIBRISPEECH_ARCHIVE}" ] || [ "$(stat -c%s "${LIBRISPEECH_ARCHIVE}" 2>/dev/null || echo 0)" -lt 1000000000 ]; then
            log_info "Downloading LibriSpeech train-clean-100 (~6GB)..."
            log_info "This may take a while depending on your internet connection..."
            wget -c --progress=bar:force "${LIBRISPEECH_URL}" -O "${LIBRISPEECH_ARCHIVE}"
            if [ $? -ne 0 ]; then
                log_error "Failed to download LibriSpeech"
                exit 1
            fi
            log_success "LibriSpeech download complete"
        else
            log_info "LibriSpeech archive already exists, skipping download"
        fi

        # Extract
        if [ ! -d "LibriSpeech/train-clean-100" ]; then
            log_info "Extracting LibriSpeech (~6GB archive)..."
            tar -xzf "${LIBRISPEECH_ARCHIVE}"
            if [ $? -ne 0 ]; then
                log_error "Failed to extract LibriSpeech"
                exit 1
            fi
            log_success "LibriSpeech extraction complete"
        fi

        cd "${PROJECT_DIR}"
    fi

    # --- Download DroneAudioDataset ---
    log_info "Checking DroneAudioDataset..."

    # Check if we have WAV files (they might be in subdirectories)
    DRONE_WAV_COUNT=$(find "${DRONE_AUDIO_DIR}" -name "*.wav" 2>/dev/null | wc -l)

    if [ -d "${DRONE_AUDIO_DIR}" ] && [ "${DRONE_WAV_COUNT}" -gt 0 ]; then
        log_success "DroneAudioDataset already downloaded (${DRONE_WAV_COUNT} WAV files found)"
    else
        log_info "Cloning DroneAudioDataset from GitHub..."

        if [ -d "${DRONE_AUDIO_DIR}" ]; then
            rm -rf "${DRONE_AUDIO_DIR}"
        fi

        if git clone "${DRONE_AUDIO_REPO}" "${DRONE_AUDIO_DIR}"; then
            log_success "DroneAudioDataset cloned successfully"
        else
            log_error "Failed to clone DroneAudioDataset"
            log_error "Please manually download from: https://github.com/saraalemadi/DroneAudioDataset"
            exit 1
        fi
    fi

    log_success "All source datasets ready"
}

# ============================================================================
# STEP 2: Create DN-LM Dataset
# ============================================================================
step_dataset() {
    log_info "=== Step 2: Creating DroneNoise-LibriMix dataset ==="

    activate_venv

    # Check if dataset already exists with expected number of samples
    TRAIN_SAMPLES_EXIST=$(find "${TRAIN_PATH}" -maxdepth 1 -type d -name "sample_*" 2>/dev/null | wc -l)
    VALID_SAMPLES_EXIST=$(find "${VALID_PATH}" -maxdepth 1 -type d -name "sample_*" 2>/dev/null | wc -l)

    if [ -d "${TRAIN_PATH}" ] && [ -d "${VALID_PATH}" ] && \
       [ -f "${VALID_PATH}/metadata.json" ] && [ -f "${TRAIN_PATH}/metadata.json" ] && \
       [ "${TRAIN_SAMPLES_EXIST}" -ge 6000 ] && [ "${VALID_SAMPLES_EXIST}" -ge 500 ]; then
        log_success "DN-LM dataset already exists"
        log_info "  Training samples: ${TRAIN_SAMPLES_EXIST}"
        log_info "  Validation samples: ${VALID_SAMPLES_EXIST}"
        return 0
    fi

    # Partial dataset exists - remove and recreate for consistency
    if [ -d "${DNLM_DIR}" ] && { [ "${TRAIN_SAMPLES_EXIST}" -gt 0 ] || [ "${VALID_SAMPLES_EXIST}" -gt 0 ]; }; then
        log_warning "Partial DN-LM dataset found (train: ${TRAIN_SAMPLES_EXIST}, valid: ${VALID_SAMPLES_EXIST})"
        log_warning "Removing and recreating for consistency..."
        rm -rf "${DNLM_DIR}"
    fi

    # Verify source datasets exist
    LIBRISPEECH_EXTRACTED="${LIBRISPEECH_DIR}/LibriSpeech/train-clean-100"

    if [ ! -d "${LIBRISPEECH_EXTRACTED}" ]; then
        log_error "LibriSpeech not found at ${LIBRISPEECH_EXTRACTED}"
        log_error "Please run: ./replicate_paper.sh download"
        exit 1
    fi

    if [ ! -d "${DRONE_AUDIO_DIR}" ]; then
        log_error "DroneAudioDataset not found at ${DRONE_AUDIO_DIR}"
        log_error "Please run: ./replicate_paper.sh download"
        exit 1
    fi

    # Verify we have actual audio files
    SPEECH_FILE_COUNT=$(find "${LIBRISPEECH_EXTRACTED}" -name "*.flac" 2>/dev/null | wc -l)
    NOISE_FILE_COUNT=$(find "${DRONE_AUDIO_DIR}" \( -name "*.wav" -o -name "*.WAV" \) 2>/dev/null | wc -l)

    log_info "Found ${SPEECH_FILE_COUNT} speech files and ${NOISE_FILE_COUNT} noise files"

    if [ "${SPEECH_FILE_COUNT}" -eq 0 ]; then
        log_error "No speech files (*.flac) found in ${LIBRISPEECH_EXTRACTED}"
        exit 1
    fi

    if [ "${NOISE_FILE_COUNT}" -eq 0 ]; then
        log_error "No noise files (*.wav) found in ${DRONE_AUDIO_DIR}"
        log_error "Checking alternative audio formats..."
        NOISE_FILE_COUNT=$(find "${DRONE_AUDIO_DIR}" \( -name "*.mp3" -o -name "*.ogg" -o -name "*.flac" \) 2>/dev/null | wc -l)
        if [ "${NOISE_FILE_COUNT}" -eq 0 ]; then
            log_error "No audio files found in DroneAudioDataset"
            exit 1
        fi
        log_info "Found ${NOISE_FILE_COUNT} audio files with alternative formats"
    fi

    cd "${PROJECT_DIR}"

    # According to paper: 2 hours total, 9:1 train/valid split, 1-second samples
    # 2 hours = 7200 samples at 1 second each
    # Training: 6480 samples (1.8 hours)
    # Validation: 720 samples (0.2 hours)

    log_info "Creating DN-LM dataset (this may take a while)..."
    log_info "  - Training samples: 6480"
    log_info "  - Validation samples: 720"
    log_info "  - SNR range: -30 to 0 dB"

    python create_dataset.py \
        --speech_dir "${LIBRISPEECH_EXTRACTED}" \
        --noise_dir "${DRONE_AUDIO_DIR}" \
        --output_dir "${DNLM_DIR}" \
        --train_samples 6480 \
        --valid_samples 720 \
        --sample_duration 1.0 \
        --sample_rate 16000 \
        --snr_min -30 \
        --snr_max 0 \
        --seed 42

    log_success "DN-LM dataset creation complete"
}

# ============================================================================
# STEP 3: Train Models
# ============================================================================

# Get number of available NVIDIA GPUs
get_num_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Generate training command for a model
# Arguments: model_name model_type config_path gpu_id
generate_train_cmd() {
    local model_name="$1"
    local model_type="$2"
    local config_path="$3"
    local gpu_id="$4"

    echo "cd ${PROJECT_DIR} && \
source ${PROJECT_DIR}/.venv/bin/activate 2>/dev/null || source ${PROJECT_DIR}/venv/bin/activate 2>/dev/null || true && \
python train.py \
    --model_type ${model_type} \
    --config_path \"${config_path}\" \
    --results_path \"${RESULTS_DIR}/${model_name}\" \
    --data_path \"${TRAIN_PATH}\" \
    --valid_path \"${VALID_PATH}\" \
    --dataset_type 1 \
    --device_ids ${gpu_id} \
    --num_workers 4 \
    --metrics si_sdr sdr \
    --metric_for_scheduler si_sdr && \
echo 'Training complete for ${model_name}' || echo 'Training FAILED for ${model_name}'"
}

# Run training in a tmux session
# Arguments: session_name command
run_in_tmux() {
    local session_name="$1"
    local cmd="$2"

    # Kill existing session if it exists
    tmux kill-session -t "${session_name}" 2>/dev/null || true

    # Create new detached session and run command
    tmux new-session -d -s "${session_name}" "${cmd}"
    log_info "Started tmux session: ${session_name}"
}

# Wait for tmux sessions to complete
# Arguments: session_names (space-separated)
wait_for_tmux_sessions() {
    local sessions=("$@")
    local all_done=false

    log_info "Waiting for training sessions to complete..."
    log_info "You can attach to any session with: tmux attach -t <session_name>"
    log_info "Sessions: ${sessions[*]}"

    while [ "$all_done" = false ]; do
        all_done=true
        for session in "${sessions[@]}"; do
            if tmux has-session -t "${session}" 2>/dev/null; then
                all_done=false
            fi
        done
        if [ "$all_done" = false ]; then
            sleep 30
        fi
    done
}

step_train() {
    log_info "=== Step 3: Training models ==="

    cd "${PROJECT_DIR}"
    activate_venv

    # Verify dataset exists
    if [ ! -d "${TRAIN_PATH}" ] || [ ! -d "${VALID_PATH}" ]; then
        log_error "DN-LM dataset not found"
        log_error "Please run: ./replicate_paper.sh dataset"
        exit 1
    fi

    mkdir -p "${RESULTS_DIR}"

    # Common training parameters from paper (Section 4.1.2):
    # - Optimizer: AdamW
    # - Initial LR: 5.0e-4
    # - LR Schedule: ReduceLROnPlateau (factor=0.95, patience=2)
    # - Batch size: 12
    # - Early stopping patience: 30 epochs
    # - 200 steps per epoch

    # Detect available GPUs
    NUM_GPUS=$(get_num_gpus)
    log_info "Detected ${NUM_GPUS} GPU(s)"

    # Define training jobs: name|model_type|config_path|checkpoint_path
    declare -a TRAINING_JOBS=()

    if [ ! -f "${RESULTS_DIR}/edge_bs_roformer/best_model.ckpt" ]; then
        TRAINING_JOBS+=("edge_bs_roformer|edge_bs_rof|configs/3_FA_RoPE(64).yaml")
    else
        log_success "Edge-BS-RoFormer already trained"
    fi

    if [ ! -f "${RESULTS_DIR}/dcunet/best_model.ckpt" ]; then
        TRAINING_JOBS+=("dcunet|dcunet|configs/5_Baseline_dcunet.yaml")
    else
        log_success "DCUNet already trained"
    fi

    if [ ! -f "${RESULTS_DIR}/dptnet/best_model.ckpt" ]; then
        TRAINING_JOBS+=("dptnet|dptnet|configs/7_Baseline_dptnet.yaml")
    else
        log_success "DPTNet already trained"
    fi

    if [ ! -f "${RESULTS_DIR}/htdemucs/best_model.ckpt" ]; then
        TRAINING_JOBS+=("htdemucs|htdemucs|configs/8_Baseline_htdemucs.yaml")
    else
        log_success "HTDemucs already trained"
    fi

    # Check if there are any jobs to run
    if [ ${#TRAINING_JOBS[@]} -eq 0 ]; then
        log_success "All models already trained"
        return 0
    fi

    log_info "Jobs to run: ${#TRAINING_JOBS[@]}"

    # Single GPU or no GPU: run sequentially
    if [ "${NUM_GPUS}" -le 1 ]; then
        log_info "Running training jobs sequentially on GPU 0..."

        for job in "${TRAINING_JOBS[@]}"; do
            IFS='|' read -r model_name model_type config_path <<< "$job"
            log_info "Training ${model_name}..."

            python train.py \
                --model_type "${model_type}" \
                --config_path "${config_path}" \
                --results_path "${RESULTS_DIR}/${model_name}" \
                --data_path "${TRAIN_PATH}" \
                --valid_path "${VALID_PATH}" \
                --dataset_type 1 \
                --device_ids 0 \
                --num_workers 4 \
                --metrics si_sdr sdr \
                --metric_for_scheduler si_sdr

            log_success "${model_name} training complete"
        done

    # Multiple GPUs: run in parallel using tmux
    else
        log_info "Running training jobs in parallel across ${NUM_GPUS} GPUs using tmux..."

        # Check if tmux is available
        if ! command -v tmux &> /dev/null; then
            log_warning "tmux not found, falling back to sequential training"
            log_warning "Install tmux for parallel training: sudo apt install tmux"

            for job in "${TRAINING_JOBS[@]}"; do
                IFS='|' read -r model_name model_type config_path <<< "$job"
                log_info "Training ${model_name}..."

                python train.py \
                    --model_type "${model_type}" \
                    --config_path "${config_path}" \
                    --results_path "${RESULTS_DIR}/${model_name}" \
                    --data_path "${TRAIN_PATH}" \
                    --valid_path "${VALID_PATH}" \
                    --dataset_type 1 \
                    --device_ids 0 \
                    --num_workers 4 \
                    --metrics si_sdr sdr \
                    --metric_for_scheduler si_sdr

                log_success "${model_name} training complete"
            done
        else
            # Distribute jobs across GPUs
            declare -a TMUX_SESSIONS=()
            local gpu_idx=0

            for job in "${TRAINING_JOBS[@]}"; do
                IFS='|' read -r model_name model_type config_path <<< "$job"

                local session_name="train_${model_name}"
                local cmd=$(generate_train_cmd "${model_name}" "${model_type}" "${config_path}" "${gpu_idx}")

                run_in_tmux "${session_name}" "${cmd}"
                TMUX_SESSIONS+=("${session_name}")

                # Cycle through available GPUs
                gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
            done

            # Wait for all training sessions to complete
            wait_for_tmux_sessions "${TMUX_SESSIONS[@]}"

            # Verify training completed successfully
            for job in "${TRAINING_JOBS[@]}"; do
                IFS='|' read -r model_name model_type config_path <<< "$job"
                if [ -f "${RESULTS_DIR}/${model_name}/best_model.ckpt" ]; then
                    log_success "${model_name} training complete"
                else
                    log_warning "${model_name} training may have failed - checkpoint not found"
                fi
            done
        fi
    fi

    log_success "All model training complete"
}

# ============================================================================
# STEP 4: Evaluate Models
# ============================================================================
step_eval() {
    log_info "=== Step 4: Evaluating models ==="

    cd "${PROJECT_DIR}"
    activate_venv

    mkdir -p "${RESULTS_DIR}/evaluation"

    # --- Evaluate Edge-BS-RoFormer ---
    if [ -f "${RESULTS_DIR}/edge_bs_roformer/best_model.ckpt" ]; then
        log_info "Evaluating Edge-BS-RoFormer..."
        python final_valid.py \
            --model_type edge_bs_rof \
            --config_path "configs/3_FA_RoPE(64).yaml" \
            --start_check_point "${RESULTS_DIR}/edge_bs_roformer/best_model.ckpt" \
            --valid_path "${VALID_PATH}" \
            --store_dir "${RESULTS_DIR}/evaluation" \
            --device_ids 0 \
            --metrics si_sdr sdr pesq stoi
        log_success "Edge-BS-RoFormer evaluation complete"
    else
        log_warning "Edge-BS-RoFormer checkpoint not found, skipping evaluation"
    fi

    # --- Evaluate DCUNet ---
    if [ -f "${RESULTS_DIR}/dcunet/best_model.ckpt" ]; then
        log_info "Evaluating DCUNet..."
        python final_valid.py \
            --model_type dcunet \
            --config_path "configs/5_Baseline_dcunet.yaml" \
            --start_check_point "${RESULTS_DIR}/dcunet/best_model.ckpt" \
            --valid_path "${VALID_PATH}" \
            --store_dir "${RESULTS_DIR}/evaluation" \
            --device_ids 0 \
            --metrics si_sdr sdr pesq stoi
        log_success "DCUNet evaluation complete"
    else
        log_warning "DCUNet checkpoint not found, skipping evaluation"
    fi

    # --- Evaluate DPTNet ---
    if [ -f "${RESULTS_DIR}/dptnet/best_model.ckpt" ]; then
        log_info "Evaluating DPTNet..."
        python final_valid.py \
            --model_type dptnet \
            --config_path "configs/7_Baseline_dptnet.yaml" \
            --start_check_point "${RESULTS_DIR}/dptnet/best_model.ckpt" \
            --valid_path "${VALID_PATH}" \
            --store_dir "${RESULTS_DIR}/evaluation" \
            --device_ids 0 \
            --metrics si_sdr sdr pesq stoi
        log_success "DPTNet evaluation complete"
    else
        log_warning "DPTNet checkpoint not found, skipping evaluation"
    fi

    # --- Evaluate HTDemucs ---
    if [ -f "${RESULTS_DIR}/htdemucs/best_model.ckpt" ]; then
        log_info "Evaluating HTDemucs..."
        python final_valid.py \
            --model_type htdemucs \
            --config_path "configs/8_Baseline_htdemucs.yaml" \
            --start_check_point "${RESULTS_DIR}/htdemucs/best_model.ckpt" \
            --valid_path "${VALID_PATH}" \
            --store_dir "${RESULTS_DIR}/evaluation" \
            --device_ids 0 \
            --metrics si_sdr sdr pesq stoi
        log_success "HTDemucs evaluation complete"
    else
        log_warning "HTDemucs checkpoint not found, skipping evaluation"
    fi

    log_success "Evaluation complete"
    log_info "Results saved to ${RESULTS_DIR}/evaluation/"
}

# ============================================================================
# Main
# ============================================================================
print_usage() {
    echo "Usage: $0 [step]"
    echo ""
    echo "Steps:"
    echo "  setup    - Set up Python environment with uv"
    echo "  download - Download source datasets (LibriSpeech + DroneAudioDataset)"
    echo "  dataset  - Create DN-LM dataset from source datasets"
    echo "  train    - Train all models (Edge-BS-RoFormer + baselines)"
    echo "  eval     - Evaluate trained models"
    echo "  all      - Run all steps (default)"
    echo ""
    echo "Examples:"
    echo "  $0           # Run all steps"
    echo "  $0 download  # Only download datasets"
    echo "  $0 train     # Only train models"
}

main() {
    local step="${1:-all}"

    echo "=============================================="
    echo " Edge-BS-RoFormer Paper Replication Script"
    echo "=============================================="
    echo ""

    case "${step}" in
        setup)
            step_setup
            ;;
        download)
            step_download
            ;;
        dataset)
            step_dataset
            ;;
        train)
            step_train
            ;;
        eval)
            step_eval
            ;;
        all)
            step_setup
            step_download
            step_dataset
            step_train
            step_eval
            echo ""
            log_success "=== All steps completed successfully ==="
            ;;
        -h|--help|help)
            print_usage
            ;;
        *)
            log_error "Unknown step: ${step}"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
