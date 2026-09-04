#!/usr/bin/env bash
# Drive one RESUMABLE shell command as a chain of short GPU jobs.
#
#   scripts/chain_cmd.sh <job-prefix> [max-segments] [backend] [time] -- <command...>
#
# Why: scripts/chain_train.sh chains `python train.py experiment=...` only. The
# comb-slot trainer (scripts/train_slot_real.py) is a standalone script with
# its own CRF loss and its own resume file, so it needs the same chaining with
# a different command. The contract with the command is two marker lines on
# stdout: `CHAIN: continue` (the budget ran out, state is saved, submit the
# next segment) and `CHAIN: done` (the run is complete). A segment that prints
# neither is treated as a crash, and the chain stops.
#
# The command must save its state OUTSIDE the per-revision worktree that
# omnirun creates, so pass an absolute `--out` under $RESULTS_ROOT; a relative
# path would land in the worktree of the segment's commit.
#
# The submit and wait logic mirrors scripts/chain_train.sh (read its comments
# for the two omnirun races it guards against).
set -uo pipefail

PREFIX=${1:?usage: chain_cmd.sh <job-prefix> [max-segments] [backend] [time] -- <command...>}
shift
MAX_SEGMENTS=10
BACKEND=uni-gpushort
TIME_LIMIT=58m
if [ "${1:-}" != "--" ]; then MAX_SEGMENTS=$1; shift; fi
if [ "${1:-}" != "--" ]; then BACKEND=$1; shift; fi
if [ "${1:-}" != "--" ]; then TIME_LIMIT=$1; shift; fi
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "chain_cmd.sh: no command after --" >&2; exit 2; }

export SSH_ASKPASS=${SSH_ASKPASS:-/bin/false}
OR="omnirun"

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
  local name=$1 out id
  shift
  for attempt in $(seq 1 10); do
    out=$($OR submit --backend "$BACKEND" --gpus 1 --time "$TIME_LIMIT" --name "$name" \
      --env PYTHONPATH=src -- "$@" 2>&1)
    id=$(printf '%s\n' "$out" | grep -oE "submitted [^ ]+" | head -1 | awk '{print $2}')
    if [ -n "$id" ]; then printf '%s' "$id"; return 0; fi
    sleep 30
    id=$(COLUMNS=220 $OR queue 2>/dev/null | grep -oE "${name}-[a-f0-9]{6}" | head -1)
    if [ -n "$id" ]; then
      echo "CHAIN $PREFIX: recovered in-flight $id after an unreadable submit" >&2
      printf '%s' "$id"; return 0
    fi
    $OR backends check >/dev/null 2>&1
    sleep 60
  done
  return 1
}

LAST_STATE=""
for i in $(seq 1 "$MAX_SEGMENTS"); do
  job=$(submit_segment "${PREFIX}-${i}" "$@")
  if [ -z "$job" ]; then echo "CHAIN $PREFIX: submit failed at segment $i"; exit 1; fi
  echo "CHAIN $PREFIX: segment $i = $job"
  wait_for "$job"
  logs=$($OR logs "$job" 2>/dev/null)
  if printf '%s' "$logs" | grep -qa "CHAIN: done"; then
    echo "CHAIN $PREFIX: run complete after segment $i"
    exit 0
  fi
  if [ "$LAST_STATE" = cancelled ]; then
    echo "CHAIN $PREFIX: segment cancelled — stopping chain"
    exit 0
  fi
  if ! printf '%s' "$logs" | grep -qa "CHAIN: continue"; then
    echo "CHAIN $PREFIX: segment $i ended without a CHAIN marker ($LAST_STATE) — stopping"
    exit 1
  fi
done
echo "CHAIN $PREFIX: exhausted $MAX_SEGMENTS segments without finishing"
