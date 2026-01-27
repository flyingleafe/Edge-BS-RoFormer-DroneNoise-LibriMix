#!/usr/bin/env bash
#
# Sync Evaluation Results from Remote Server
#
# This script syncs evaluation results from vast-server to the local machine.
# Results are stored on vast-server at: Edge-BS-RoFormer-DroneNoise-LibriMix/results/evaluation
# and synced to local: results/evaluation
#
# Usage: ./sync_results.sh

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="vast-server"
REMOTE_PATH="Edge-BS-RoFormer-DroneNoise-LibriMix/results/evaluation"
LOCAL_PATH="${PROJECT_DIR}/results/evaluation"

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

# ============================================================================
# Main Script
# ============================================================================
main() {
    log_info "Syncing evaluation results from ${REMOTE_HOST}..."
    log_info "Remote path: ${REMOTE_HOST}:${REMOTE_PATH}"
    log_info "Local path: ${LOCAL_PATH}"
    
    # Create local directory if it doesn't exist
    mkdir -p "${LOCAL_PATH}"
    
    # Test SSH connection
    log_info "Testing SSH connection to ${REMOTE_HOST}..."
    if ! ssh -o ConnectTimeout=5 "${REMOTE_HOST}" "echo 'Connection successful'" > /dev/null 2>&1; then
        log_error "Cannot connect to ${REMOTE_HOST}. Please check your SSH configuration."
        exit 1
    fi
    
    # Check if remote path exists
    log_info "Checking if remote path exists..."
    if ! ssh "${REMOTE_HOST}" "test -d ${REMOTE_PATH}" 2>/dev/null; then
        log_error "Remote path does not exist: ${REMOTE_HOST}:${REMOTE_PATH}"
        exit 1
    fi
    
    # Perform rsync
    log_info "Starting rsync..."
    rsync -avz --progress \
        "${REMOTE_HOST}:${REMOTE_PATH}/" \
        "${LOCAL_PATH}/"
    
    if [ $? -eq 0 ]; then
        log_success "Results synced successfully!"
        log_info "Local results available at: ${LOCAL_PATH}"
    else
        log_error "rsync failed. Please check the error messages above."
        exit 1
    fi
}

# Run main function
main "$@"
