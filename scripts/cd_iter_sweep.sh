#!/usr/bin/env bash
# Stage-C/D iteration sweep (no pi_kalman): does the VK refine loop become
# live with a k-scaled band (bw_rps), the IAVKF bandwidth adaptation
# (bw_adapt), and the WP18 weight law (freq_weight=k_beta)?
#
# Arms (all n_outer=8, everything else = the frozen REFINE_CFG):
#   a0_control            fixed 1.5 Hz band            (the known-inert stage)
#   a1_krps05             bw_rps 0.5  (capture 0.5 rev/s at every harmonic)
#   a1_krps15             bw_rps 1.5  (capture 1.5 rev/s)
#   a2_bwadapt            fixed band + IAVKF adaptation (the never-run B arm)
#   a3_krps15_adapt       k-scaled 1.5 + adaptation
#   a4_krps15_adapt_kbeta a3 + measured weight law (k^2, no amplitude factor)
#
# Offsets probe capture range: 0 (precision), +0.3, +1.0 rev/s.
# Windows: the physical synthetic battery (6) + the two real lab windows.
set -euo pipefail
export PYTHONPATH=src

# The two real lab windows read results/beatvk_vk_arms/prep_cache/*.npz;
# materialize them from the frozen dataset when missing (idempotent).
python - <<'PY'
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from pathlib import Path

import beatvk_vk_arms as arms

out = Path("results/beatvk_vk_arms")
arms.load_manifest(out, None, None)
arms.build_preps(
    out, {"free-flight_nosource_room1": [0], "FLY124": [3]}, None, "dload:DREGON"
)
PY

OUT=results/cd_iter_sweep
mkdir -p "$OUT"

ARM_NAMES=(a0_control a1_krps05 a1_krps15 a2_bwadapt a3_krps15_adapt a4_krps15_adapt_kbeta)
ARM_KWARGS=(
  '{"n_outer": 8}'
  '{"n_outer": 8, "bw_rps": 0.5}'
  '{"n_outer": 8, "bw_rps": 1.5}'
  '{"n_outer": 8, "bw_adapt": true}'
  '{"n_outer": 8, "bw_rps": 1.5, "bw_adapt": true}'
  '{"n_outer": 8, "bw_rps": 1.5, "bw_adapt": true, "freq_weight": "k_beta"}'
)

for i in "${!ARM_NAMES[@]}"; do
  for off in 0.0 0.3 1.0; do
    tag="${ARM_NAMES[$i]}_off${off}"
    if [ -s "$OUT/$tag.json" ]; then
      echo "== $tag: exists, skip"
      continue
    fi
    echo "== $tag"
    python scripts/rps_refine_lab.py \
      --chain cd_iter \
      --windows synthbl,dregon_ramp,fly124_cruise \
      --entry-offset "$off" \
      --cd-kwargs "${ARM_KWARGS[$i]}" \
      --out "$OUT/$tag.json"
  done
done
echo "sweep done: $OUT"
