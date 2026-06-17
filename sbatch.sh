#!/usr/bin/env bash
# Generic Slurm submission wrapper for the gpushort partition.
#
# Usage:
#   ./sbatch.sh [slurm_params] -- python train_rps_predictor.py [...training params]
#
# Examples:
#   ./sbatch.sh -J rps_scv2 -- python train_rps_predictor.py \
#     --device cuda:0 \
#     --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4 \
#     --save_path /gpfs/scratch/acw592/results/rps_scv2
#
#   ./sbatch.sh -J debug --cpus-per-gpu=4 --mem-per-cpu=8G -- python -c 'import torch; print(torch.cuda.is_available())'
#
# Notes:
#   - gpushort maximum wall time is 1:00:00; this wrapper rejects longer times.
#   - All persistent data/results/logs should live under /gpfs/scratch/acw592.

set -euo pipefail

SCRATCH=/gpfs/scratch/acw592
PARTITION=gpushort
MAX_SECONDS=3600
DEFAULT_TIME=1:00:00

usage() {
  cat <<'EOF'
Usage:
  ./sbatch.sh [slurm_params] -- command [args...]

Defaults added by this wrapper:
  --partition=gpushort
  --time=1:00:00
  --job-name=hns_job
  --output=/gpfs/scratch/acw592/logs/%x.o%j
  --gres=gpu:1
  --cpus-per-gpu=8
  --mem-per-cpu=11G

Examples:
  ./sbatch.sh -J rps_test -- python train_rps_predictor.py --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4 --save_path /gpfs/scratch/acw592/results/rps_test
  ./sbatch.sh -J smoke --time=00:10:00 -- python -c 'import torch; print(torch.cuda.is_available())'
EOF
}

die() {
  echo "error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

# Convert common Slurm time formats to seconds.
# Supported: minutes, minutes:seconds, hours:minutes:seconds, days-hours,
# days-hours:minutes, days-hours:minutes:seconds.
time_to_seconds() {
  local t=$1 days=0 rest h=0 m=0 s=0 IFS=:

  if [[ $t == *-* ]]; then
    days=${t%%-*}
    rest=${t#*-}
  else
    rest=$t
  fi

  read -r -a parts <<< "$rest"
  case ${#parts[@]} in
    1) m=${parts[0]} ;;
    2) m=${parts[0]}; s=${parts[1]} ;;
    3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
    *) return 1 ;;
  esac

  [[ $days =~ ^[0-9]+$ && $h =~ ^[0-9]+$ && $m =~ ^[0-9]+$ && $s =~ ^[0-9]+$ ]] || return 1
  echo $((10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s))
}

validate_time() {
  local value=$1 seconds
  seconds=$(time_to_seconds "$value") || die "could not parse Slurm time '$value'"
  (( seconds <= MAX_SECONDS )) || die "gpushort maximum time is ${DEFAULT_TIME}; got '$value'"
}

slurm_args=()
cmd=()
seen_separator=0

while (($#)); do
  if (( seen_separator )); then
    cmd+=("$1")
    shift
    continue
  fi

  case "$1" in
    --)
      seen_separator=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -p|--partition)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      [[ "$2" == "$PARTITION" ]] || die "this wrapper is only for partition '$PARTITION'; got '$2'"
      slurm_args+=("$1" "$2")
      shift 2
      ;;
    -p*)
      value=${1#-p}
      [[ "$value" == "$PARTITION" ]] || die "this wrapper is only for partition '$PARTITION'; got '$value'"
      slurm_args+=("$1")
      shift
      ;;
    --partition=*)
      value=${1#--partition=}
      [[ "$value" == "$PARTITION" ]] || die "this wrapper is only for partition '$PARTITION'; got '$value'"
      slurm_args+=("$1")
      shift
      ;;
    -t|--time)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      validate_time "$2"
      slurm_args+=("$1" "$2")
      shift 2
      ;;
    -t*)
      value=${1#-t}
      validate_time "$value"
      slurm_args+=("$1")
      shift
      ;;
    --time=*)
      value=${1#--time=}
      validate_time "$value"
      slurm_args+=("$1")
      shift
      ;;
    *)
      slurm_args+=("$1")
      shift
      ;;
  esac
done

(( seen_separator )) || die "missing '--' separator before command"
((${#cmd[@]} > 0)) || die "missing command after '--'"

mkdir -p "$SCRATCH/logs" "$SCRATCH/results"

# Quote the user command once, preserving exact argv boundaries in the job script.
printf -v quoted_cmd '%q ' "${cmd[@]}"

job_script=$(mktemp "${TMPDIR:-/tmp}/hns-gpushort-XXXXXX.sbatch")
cleanup() {
  rm -f "$job_script"
}
trap cleanup EXIT

cat > "$job_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=hns_job
#SBATCH --output=${SCRATCH}/logs/%x.o%j
#SBATCH --partition=${PARTITION}
#SBATCH --time=${DEFAULT_TIME}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G

set -euo pipefail

export SCRATCH=${SCRATCH}
export HNS_DATA_ROOT="\${SCRATCH}/datasets"
export HNS_RESULTS_ROOT="\${SCRATCH}/results"

cd "\${SLURM_SUBMIT_DIR}"
mkdir -p "\${SCRATCH}/logs" "\${SCRATCH}/results"

echo "Job ID: \${SLURM_JOB_ID}"
echo "Job name: \${SLURM_JOB_NAME}"
echo "Node: \$(hostname)"
echo "Submit dir: \${SLURM_SUBMIT_DIR}"
echo "SCRATCH: \${SCRATCH}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: ${quoted_cmd}"
date

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "warning: .venv/bin/activate not found; using current Python environment" >&2
fi

echo "Python: \$(command -v python || true)"
python -V

echo "GPU check:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "warning: nvidia-smi not found" >&2
fi

python - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cudnn version:", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check skipped/failed:", exc)
PY

${quoted_cmd}

date
EOF

chmod 700 "$job_script"

# Command-line Slurm args override the defaults embedded above, after validation.
command sbatch "${slurm_args[@]}" "$job_script"
