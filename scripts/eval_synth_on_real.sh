#!/usr/bin/env bash
# Score synthetic-only predictors on the REAL frozen split.
#
# THE QUESTION. The stochastic capacity ladder was validated on real data and
# stopped 9-34 epochs in, every arm still descending. The `stoch_long_*` reruns
# swapped in a half-real/half-synthetic validation set and reached 158-228
# epochs on a plateau. Neither generation ever recorded a REAL-ONLY number for
# the converged weights, so "does converging on synthetic help or hurt real
# transfer" has never been measured.
#
# `data=e12_real_fullflight` supplies the frozen real split as `data.valid`;
# eval.py builds `cfg.data.valid` only, so the training stream is never touched.
# `best` vs `last` is the second question: the codebase already records that
# fitting synthetic harder can COST real accuracy (stoch_s1id_scv2, 8.63 -> 3.70
# synthetic all-MAE against 7.40 -> 14.19 real), and `last` is the most
# synthetic-fitted weights each run produced.
set -uo pipefail
R2=r2://ml-data/artifacts
for cell in \
  "stoch_long_scv2:best"   "stoch_long_scv2:last" \
  "stoch_s1id_scv2:best"   "stoch_s1id_scv2:last" \
  "stoch_long_trxxl:best"  "stoch_s1id_trxxl:best"
do
  exp="${cell%%:*}"; which="${cell##*:}"
  echo "=== $exp / $which ==="
  python eval.py experiment="$exp" data=e12_real_fullflight \
    checkpoint="$R2/$exp/checkpoints/$which.ckpt" \
    experiment_name="xr_${exp}_${which}" || echo "FAILED: $exp/$which"
done
echo "=== SUMMARY (real frozen split) ==="
for d in results/xr_*/eval/metrics.json; do
  [ -f "$d" ] && echo "$d $(cat "$d")"
done
