#!/usr/bin/env bash
# Cluster driver: per-clip eval for the F2 noise-pool ladder.
#
#   omnirun submit --backend uni-gpushort --gpus 1 --time 55m \
#       --env PYTHONPATH=src -- bash scripts/run_f2_perclip_eval.sh [arm ...]
#
# Two questions, two sets of scorings:
#
# 1. Does step 1 reproduce the survey's DCUNet? -> every arm on the batch's own
#    fixed valid set, SE-valid-avq-survey, plus the `noisy` anchor for the same
#    build (the anchor CSV in results/f1_perclip/ was measured on the previous
#    build of that set, so it is re-measured here rather than reused).
#
# 2. Do the broad-pool arms actually COLLAPSE, or do they merely specialise away
#    from AVQ? Scoring them only on AVQ cannot tell those apart. So each broad
#    arm is ALSO scored on the valid set matching its own training pool
#    (alldrone -> SE-valid-drone, allharmonic -> SE-valid-harmonic). A model
#    that merely specialised elsewhere should look healthy there; a collapsed
#    one stays bad everywhere, with gain_db/corr showing why.
#
# All arms share one architecture config (the ladder's single manipulated
# variable is the training noise pool), so --model-cfg is the same throughout.
set -uo pipefail

MODEL_CFG=conf/model/f2_dcunet_survey.yaml
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(f2_dcunet_avq_survey f2_dcunet_alldrone f2_dcunet_allharmonic)
fi

# Valid set matching each arm's own training pool (empty = AVQ only, which every
# arm is scored on anyway).
own_valid() {
  case "$1" in
    f2_dcunet_alldrone) echo SE-valid-drone ;;
    f2_dcunet_allharmonic) echo SE-valid-harmonic ;;
    *) echo "" ;;
  esac
}

run() {
  local method=$1 valid=$2
  shift 2
  echo "===== eval $method on $valid ====="
  python scripts/eval_se_perclip.py --method "$method" --valid "$valid" \
    --batch 8 --r2-upload "$@" || echo "!!!!! FAILED $method $valid"
}

# Anchors on the F2 valid set: `noisy` is the bar every arm must clear.
run noisy SE-valid-avq-survey

for arm in "${ARMS[@]}"; do
  run "$arm" SE-valid-avq-survey --model-cfg "$MODEL_CFG"
  own=$(own_valid "$arm")
  if [ -n "$own" ]; then
    run "$arm" "$own" --model-cfg "$MODEL_CFG"
  fi
done

echo "F2 per-clip evals done"
