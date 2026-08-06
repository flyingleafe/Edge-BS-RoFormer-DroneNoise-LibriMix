#!/usr/bin/env bash
# The issue-17 campaign (phase 6c): fit every protocol window, then judge every
# fit at fixed degrees of freedom against all four controls.
#
# Three stages, in order, on the SAME checkout (stage 2 and 3 read stage 1's
# trajectories):
#
#   1. refit    15 windows x 6 arms. Each arm turns exactly ONE step of the
#               issue's procedure off, so the campaign can attribute the
#               movement to a step instead of shipping a bundle.
#   2. score    9 candidates (raw telemetry, the pre-smoothed carrier, the
#               withdrawn 0.99458 factor, and the six arms' fits) x 4 controls.
#   3. profile  the ONE-PARAMETER scale family lp:5+scale:s over s, plus a
#               coarse raw-telemetry family. A profile has a single degree of
#               freedom, so its minimum is a scale estimate that flexibility
#               cannot buy — the fitted trajectories' weakness by construction.
#
# DREGON is the measurement; FLY124 is the negative control (its labels were
# recalibrated and read -0.063 %) and runs through the identical procedure.
#
# Usage:  bash scripts/telemetry_campaign.sh [JOBS]
set -uo pipefail
export PYTHONPATH=src
# One FFT worker per unit — the parallelism is gridrun's, and nested thread
# pools only fight over the same cores.
export TRACKING_FFT_WORKERS="${TRACKING_FFT_WORKERS:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

JOBS="${1:-16}"
REFIT=results/telemetry_refit/campaign
FIT=results/telemetry_fitness
ARMS=main,nosmooth,flatk,nopeel,gate,b0_3

CANDS=telemetry,lp:5,scale:0.99458
for arm in main nosmooth flatk nopeel gate b0_3; do
  CANDS="$CANDS,file:$REFIT/traj/$arm/{key}.npz:r_fit"
done

# s = 0.9880 .. 1.0020 step 0.0005 (-1.20 % .. +0.20 %), the span of every
# published estimate plus margin on both sides.
PROFILE=$(python -c "
import numpy as np
print(','.join(f'lp:5+scale:{s:.4f}' for s in np.arange(0.9880, 1.00201, 0.0005)))")
RAW_PROFILE=$(python -c "
import numpy as np
print(','.join(f'scale:{s:.4f}' for s in np.arange(0.9880, 1.00201, 0.0020)))")

echo "== 1/3 refit  (15 windows x 6 arms)"
python scripts/telemetry_refit.py --dataset all --arms "$ARMS" \
  --jobs "$JOBS" --build-preps --out "$REFIT"

echo "== 2/3 score  (9 candidates x 4 controls)"
python scripts/telemetry_fitness.py --dataset all --candidates "$CANDS" \
  --boot 500 --jobs "$JOBS" --out "$FIT/campaign"

echo "== 3/3 profile (the one-parameter scale family)"
python scripts/telemetry_fitness.py --dataset all --candidates "$PROFILE" \
  --controls on,offcomb --boot 500 --jobs "$JOBS" --out "$FIT/scale_profile"
python scripts/telemetry_fitness.py --dataset all --candidates "$RAW_PROFILE" \
  --controls on,offcomb --boot 500 --jobs "$JOBS" --out "$FIT/scale_profile_raw"

echo "campaign done: $REFIT $FIT/campaign $FIT/scale_profile $FIT/scale_profile_raw"
