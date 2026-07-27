#!/usr/bin/env bash
# Cluster driver: per-clip SE eval for every F1 discriminative baseline on both
# valid sets, uploading each result CSV to R2 (artifacts/<method>/perclip/) so
# it survives the unreliable `omnirun pull`. Checkpoints are self-fetched from
# R2 by eval_se_perclip.py. SGMSE is scored by a separate (slower) job.
#
#   omnirun submit --backend uni-gpushort --gpus 1 --time 1h --yes \
#       --env PYTHONPATH=src -- bash scripts/run_se_perclip_eval.sh
set -uo pipefail

METHODS=(
  f1_dcunet_a f1_dcunet_b
  f1_edge_bs_rof_a f1_edge_bs_rof_b
  f1_mpsenet_a f1_mpsenet_b
  f1_tfgridnet_a f1_tfgridnet_b
)
VALIDS=(SE-valid-drone SE-valid-harmonic)

for m in "${METHODS[@]}"; do
  for v in "${VALIDS[@]}"; do
    echo "===== eval $m on $v ====="
    python scripts/eval_se_perclip.py --method "$m" --valid "$v" --batch 16 --r2-upload \
      || echo "!!!!! FAILED $m $v"
  done
done
echo "all per-clip evals done"
