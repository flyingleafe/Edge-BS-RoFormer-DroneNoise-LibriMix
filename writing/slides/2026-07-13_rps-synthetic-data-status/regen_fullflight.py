"""Standalone re-render of ONLY the full-flight overlay (transformer, 3
conditions), reusing prepare_regime_eval.py's helpers and the already-cached
R2 checkpoints. Does not touch the MAE table (already fresh) or the
per-regime overlay PNGs (unaffected by the fullflight bug).
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare_regime_eval import (
    CONDITIONS,
    DATA_CFG,
    MODEL_CFG,
    download_ckpt,
    model_cfg,
)

from data_processing.collate import batch_size as fbatch_size
from data_processing.collate import frame_collate, slice_sample
from data_processing.frames import get_meta
from metrics._common import get_array
from tasks.rps_prediction import align_rps_to_gt
from training.config import build_dataset, build_task_and_codec, instantiate_model

device = torch.device("cpu")
_task, codec = build_task_and_codec(MODEL_CFG)

print("Loading validation set (dload:DREGON-LM-V4-michaels-valid-full)...")
valid_ds = build_dataset(DATA_CFG)
loader = torch.utils.data.DataLoader(
    valid_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=frame_collate
)

all_gt = []
all_channel = []
for batch in loader:
    for i in range(fbatch_size(batch)):
        s = slice_sample(batch, i)
        all_gt.append(get_array(s, "rps"))
        all_channel.append(get_meta(s, "channel", None))
print(f"  {len(all_gt)} validation clips loaded")
print("  channel meta sample:", all_channel[:10])

idxs = [i for i, c in enumerate(all_channel) if c == 0]
print(f"  {len(idxs)} channel-0 windows selected for the fullflight walk")
assert idxs, "channel metadata missing -- cannot build the fullflight overlay"

mcfg = model_cfg("simple_conv_v2_transformer")
results = {}
for cond_key, cond_label, r2keys in CONDITIONS:
    print(f"\n=== {cond_key} (transformer, {cond_label}) ===")
    ckpt_path = download_ckpt(r2keys["transformer"])
    if ckpt_path is None:
        results[cond_key] = None
        continue
    model = instantiate_model(mcfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = codec.to_inputs(batch)
            outputs = codec.call_model(model, inputs)
            pred_frame = codec.to_frame(outputs, batch)
            for i in range(fbatch_size(pred_frame)):
                p = slice_sample(pred_frame, i)
                preds.append(get_array(p, "rps_pred"))
    assert len(preds) == len(all_gt), (len(preds), len(all_gt))
    results[cond_key] = {"preds": preds}
    print(f"  {len(preds)} preds computed")

gt_cat = np.concatenate([all_gt[i] for i in idxs], axis=1)
print("GT concatenated shape:", gt_cat.shape, "min/max:", gt_cat.min(), gt_cat.max())
n_rows = sum(1 for k, _, _ in CONDITIONS if results.get(k))
fig, axs = plt.subplots(n_rows, 1, figsize=(9.0, 1.5 * n_rows), sharex=True)
if n_rows == 1:
    axs = [axs]
row = 0
for key, label, _ in CONDITIONS:
    rkey = results.get(key)
    if rkey is None:
        continue
    ax = axs[row]
    p_cat = np.concatenate([align_rps_to_gt(rkey["preds"][i], all_gt[i]) for i in idxs], axis=1)
    t = np.arange(gt_cat.shape[1]) * 512 / 16000
    for r in range(gt_cat.shape[0]):
        ax.plot(t, gt_cat[r], ":", color="grey", lw=0.9, label="GT" if r == 0 else None)
        ax.plot(t, p_cat[r], "-", lw=0.9, label="pred" if r == 0 else None)
    ax.set_ylabel("rev/s", fontsize=8)
    ax.set_title(label, fontsize=9)
    if row == 0:
        ax.legend(
            frameon=False,
            fontsize=7,
            loc="lower left",
            bbox_to_anchor=(1.0, 0.0),
            borderaxespad=0.0,
        )
    row += 1
axs[-1].set_xlabel("time (s)", fontsize=8)
fig.suptitle(
    f"Whole valid-full set, channel 0, time order ({len(idxs)} windows concatenated)",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0, 0.9, 0.95))
outp = HERE / "assets" / "regime_overlay_fullflight.png"
fig.savefig(outp, dpi=140)
plt.close(fig)
print("saved", outp)
