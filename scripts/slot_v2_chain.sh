#!/usr/bin/env bash
# Chain one `scripts/train_slot_v2.py` arm over 55-minute gpushort segments.
#
#   scripts/slot_v2_chain.sh <arm> [max-segments] -- <train_slot_v2 flags...>
#
#   scripts/slot_v2_chain.sh B1 10 -- --data real --off-state
#   scripts/slot_v2_chain.sh B2 20 -- --data comb --off-state --grid-lo 10 --n-grid 900
#   scripts/slot_v2_chain.sh B3 20 -- --data real --off-state \
#       --init /data/scratch/acw592/results/slot_v2/B2/best.pt
#
# WHY A WRAPPER. `scripts/chain_cmd.sh` already drives a resumable command as a
# chain of short GPU jobs, and it is not replaced here. What this file adds is
# the three settings every v2 arm must share and that are easy to get wrong:
#
#   --out under $RESULTS_ROOT   The trainer's state.pt must live OUTSIDE the
#                               per-revision worktree omnirun creates, or the
#                               next segment starts from step 0.
#   --max-minutes 55            The trainer saves its state and prints
#                               `CHAIN: continue` before the 58 m wall clock,
#                               so a segment never dies inside a validation.
#   --name <arm>                The arm names its directory and its dump column.
#
# The C1 arms were chained exactly this way (`docs/experiments/
# candidate-tests-2026-09-04.md` § C1): 2 s crops, batch 2, lr 1e-3, 1500 steps,
# mono, selection on 48 windows, the non-finite step guard.
set -uo pipefail

ARM=${1:?usage: slot_v2_chain.sh <arm> [max-segments] -- <train_slot_v2 flags...>}
shift
MAX_SEGMENTS=20
if [ "${1:-}" != "--" ]; then MAX_SEGMENTS=$1; shift; fi
[ "${1:-}" = "--" ] && shift

RESULTS_ROOT=${RESULTS_ROOT:-/data/scratch/acw592/results}
BACKEND=${BACKEND:-uni-gpushort}
TIME_LIMIT=${TIME_LIMIT:-58m}
BUDGET=${BUDGET:-55}
OUT="$RESULTS_ROOT/slot_v2/$ARM"

HERE=$(cd "$(dirname "$0")" && pwd)
echo "CHAIN $ARM: out=$OUT backend=$BACKEND segments<=$MAX_SEGMENTS budget=${BUDGET}m"
exec "$HERE/chain_cmd.sh" "v2-$ARM" "$MAX_SEGMENTS" "$BACKEND" "$TIME_LIMIT" -- \
  python scripts/train_slot_v2.py \
  --name "$ARM" --out "$OUT" --max-minutes "$BUDGET" "$@"
