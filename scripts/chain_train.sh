#!/usr/bin/env bash
# Drive one training run as a chain of short GPU jobs.
#
#   scripts/chain_train.sh <experiment> <job-prefix> [max-segments] [backend] [time]
#
# Why: the only long-GPU Slurm partition here (omnirun backend `uni` = sae) is
# ~160 jobs deep with multi-day start estimates, while `uni-gpushort` has idle
# nodes but a hard 1 h wall clock. Since `resume=true` now genuinely restores
# optimizer/scheduler/epoch state (src/training/loop.py), a run can be carried
# across many short jobs instead of waiting for one long one.
#
# omnirun's own `--after` dependency flag takes a daemonless code path that
# cannot see the daemon's backend config ("unknown backend 'uni-gpushort'"), so
# the chaining is done here: submit one segment, wait for it to finish, submit
# the next. The loop stops early when the training loop reports it has nothing
# left to resume (early stop or epoch budget reached), so an over-long chain
# costs nothing.
#
# RESULTS_ROOT is passed so the run dir lives OUTSIDE the per-revision worktree
# that omnirun creates; otherwise any commit made mid-chain would send the next
# segment to an empty directory and silently restart training at epoch 0.
set -uo pipefail

EXPERIMENT=${1:?usage: chain_train.sh <experiment> <job-prefix> [max-segments] [backend] [time]}
PREFIX=${2:?job-prefix required}
MAX_SEGMENTS=${3:-10}
BACKEND=${4:-uni-gpushort}
TIME_LIMIT=${5:-55m}
RESULTS_ROOT=${RESULTS_ROOT:-/data/scratch/acw592/results}

export SSH_ASKPASS=${SSH_ASKPASS:-/bin/false}
OR="uvx -p 3.12 --from $HOME/Projects/omnirun omnirun"

wait_for() {
  local job=$1 st
  while true; do
    sleep 90
    st=$($OR status "$job" 2>/dev/null | grep -E '^status:' | awk '{print $2}')
    case "$st" in
      succeeded | failed | cancelled | timeout)
        echo "CHAIN $PREFIX: $job -> $st"
        LAST_STATE=$st
        return 0
        ;;
    esac
  done
}

submit_segment() {
  local name=$1 out
  # The daemon's SSH ControlMaster expires and every call in the following
  # window fails with "unknown backend"; that window can outlast a handful of
  # quick retries, so back off generously rather than dropping the chain.
  for attempt in $(seq 1 10); do
    out=$($OR submit --backend "$BACKEND" --gpus 1 --time "$TIME_LIMIT" --name "$name" \
      --env PYTHONPATH=src --env "RESULTS_ROOT=$RESULTS_ROOT" \
      -- python train.py "experiment=$EXPERIMENT" resume=true 2>&1)
    local id
    id=$(printf '%s\n' "$out" | grep -oE "${name}-[a-f0-9]+" | head -1)
    if [ -n "$id" ]; then printf '%s' "$id"; return 0; fi
    # The daemon's SSH ControlMaster expires and the first call after that
    # fails with "unknown backend"; `backends check` revives it.
    $OR backends check >/dev/null 2>&1
    sleep 60
  done
  return 1
}

LAST_STATE=""
START_AT=1

# Attach to a segment already in flight (e.g. one submitted by hand) instead of
# starting a second, concurrent one against the same run dir.
if [ -n "${CHAIN_WAIT_FOR:-}" ]; then
  echo "CHAIN $PREFIX: waiting on in-flight segment $CHAIN_WAIT_FOR"
  wait_for "$CHAIN_WAIT_FOR"
  if $OR logs "$CHAIN_WAIT_FOR" 2>/dev/null | grep -qa "nothing to resume"; then
    echo "CHAIN $PREFIX: run already complete"
    exit 0
  fi
  START_AT=2
fi

for i in $(seq "$START_AT" "$MAX_SEGMENTS"); do
  job=$(submit_segment "${PREFIX}-${i}")
  if [ -z "$job" ]; then echo "CHAIN $PREFIX: submit failed at segment $i"; exit 1; fi
  echo "CHAIN $PREFIX: segment $i = $job"
  wait_for "$job"

  # The loop prints this and exits immediately when the restored state says the
  # run is over; no point queueing the rest of the chain.
  if $OR logs "$job" 2>/dev/null | grep -qa "nothing to resume"; then
    echo "CHAIN $PREFIX: run complete (nothing to resume) after segment $i"
    exit 0
  fi
  if [ "$LAST_STATE" = cancelled ]; then
    echo "CHAIN $PREFIX: segment cancelled — stopping chain"
    exit 0
  fi
done

echo "CHAIN $PREFIX: exhausted $MAX_SEGMENTS segments without converging"
