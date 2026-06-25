#!/usr/bin/env bash
set -uo pipefail

MODEL=${MODEL:-simple_conv_v2_uni_gru128}
DATA=${DATA:-/gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels}
ROOT=${ROOT:-/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/grad_clip_calibration/${MODEL}_offline_16ep}
CLIPS=${CLIPS:-"0.5 1.0 2.0 5.0"}

mkdir -p "$ROOT"
cat <<EOF
Gradient-clip calibration
model=$MODEL
data=$DATA
root=$ROOT
clips=$CLIPS
budget=offline fixed train, 16 epochs, patience 16, batch 32, lr 1e-3, AMP, PIT-MSE
EOF

: > "$ROOT/status.txt"
for CLIP in $CLIPS; do
  SAFE_CLIP=${CLIP//./p}
  SAVE="$ROOT/clip_${SAFE_CLIP}"
  mkdir -p "$SAVE"
  echo
  echo "===== clip=$CLIP save=$SAVE start=$(date -Is) ====="
  set +e
  python train_rps_predictor.py \
    --model "$MODEL" \
    --device cuda:0 \
    --data_root "$DATA" \
    --save_path "$SAVE" \
    --epochs 16 \
    --patience 16 \
    --batch_size 32 --lr 1e-3 --weight_decay 1e-4 \
    --loss pit_mse \
    --grad_clip "$CLIP" \
    2>&1 | tee "$SAVE/train.log"
  STATUS=${PIPESTATUS[0]}
  set -e
  echo "===== clip=$CLIP exit=$STATUS end=$(date -Is) =====" | tee -a "$SAVE/train.log"
  echo "$CLIP $STATUS" >> "$ROOT/status.txt"
done

python - "$ROOT" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for log in sorted(root.glob("clip_*/train.log")):
    text = log.read_text(errors="replace")
    clip = log.parent.name.replace("clip_", "").replace("p", ".")
    status_match = re.search(r"===== clip=.* exit=(\d+)", text)
    epoch_rows = re.findall(
        r"^\s*(\d+)\s+([\-+0-9.eE]+|nan)\s+([\-+0-9.eE]+|nan)\s+([\-+0-9.eE]+|nan)\s+([\-+0-9.eE]+|nan)\s+([\-+0-9.eE]+|nan)\s+([\-+0-9.eE]+|nan)\s+([\-+0-9.eE]+)",
        text,
        flags=re.M | re.I,
    )
    final = re.search(
        r"Per-frame: PIT MSE=([\-+0-9.eE]+|nan).*?MAE=([\-+0-9.eE]+|nan), R²=([\-+0-9.eE]+|nan)",
        text,
        flags=re.S | re.I,
    )

    def to_float(value):
        try:
            return float(value)
        except Exception:
            return None

    finite_val_rows = [
        (int(e[0]), to_float(e[2]), to_float(e[6]))
        for e in epoch_rows
        if e[2].lower() != "nan"
    ]
    best = min(finite_val_rows, key=lambda x: x[1]) if finite_val_rows else (None, None, None)
    rows.append(
        {
            "clip": clip,
            "exit": int(status_match.group(1)) if status_match else None,
            "has_nan": bool(re.search(r"\bnan\b", text, flags=re.I)),
            "epochs_logged": len(epoch_rows),
            "best_epoch_logged": best[0],
            "best_val_pit_logged": best[1],
            "best_r2_logged": best[2],
            "final_pit_mse": to_float(final.group(1)) if final else None,
            "final_mae_frame": to_float(final.group(2)) if final else None,
            "final_r2": to_float(final.group(3)) if final else None,
            "log": str(log),
        }
    )

(root / "summary.json").write_text(json.dumps(rows, indent=2))
print("\nSUMMARY")
print(json.dumps(rows, indent=2))
PY
