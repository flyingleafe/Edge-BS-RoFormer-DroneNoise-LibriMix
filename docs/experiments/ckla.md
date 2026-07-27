# CKLA — complex Kalman linear attention for RPS prediction

Batch doc for the CKLA architecture campaign (design:
`docs/ckla-design.md`; exploration + prior art:
`docs/complex-ou-layer-exploration.md`). Goal: either beat the neural
floor **2.481** (g2_if, dregon_cruise PIT-MAE on the vk_valid_comparison
protocol) by more than seed noise without regressing FLY124, or produce a
quantified definitive negative naming the mechanism (design §6 kit).

## Ladder

| stage | experiment | question | gate |
|---|---|---|---|
| P0 | `ckla_p0_staticcomb` | can the CKLA head track combs at all, at matched budget vs the E8 transformer arm? | train-dist PIT-MSE ≤ E8 transformer at common epoch; stable fp32 training; rotation path used (§6 diagnostics) |
| P0b | capture-boundary eval | where does lock break vs drift rate × SNR? | boundary at or beyond K2's collapse point |
| P1 | `ckla_p1_*` (E12 schedule, v4-michaels stream) | does it beat 2.481 on real cruise? | > seed-noise margin (~0.15), FLY124 ≤ 2.33 |
| P1a–e | ablation ladder (design §5) | which ingredient carries/fails? | — |

## P0 protocol

Identical to `e8_staticcomb_s1_transformer` in every field except
`model` → `simple_conv_v2_ckla_mag` (stft_mag front-end — isolates the
head; the E8 arms all ran stft_mag). Comparison numbers from the E8/E9
batch (post valid-cleanup, [[sim2real-rps-transfer-findings]]): the
on-distribution comparison is the *train* PIT-MSE trajectory at common
epochs (wandb), the transfer read is the fixed real valid.

## Results

### P0 — `ckla_p0_staticcomb` (kaggle `python-9d450c`, wandb `jcrr4tqe`, 2026-07-25)

Trained stably (fp32 scan under amp, lr 1e-3, no divergence); early-stopped
ep 15, ~2 min/epoch on P100.

**On-distribution (train PIT-MSE, same static-comb stream — the clean
comparison):** CKLA reaches train ≈ 3.4 by **epoch 3**; the E8 transformer
(`2sabeq2g`) needs **~23 epochs** to reach the same level (ep 3–7 window:
CKLA 3.3–4.8 vs transformer 8.4–12.7; E8 uni_gru128 never got below ~9).
≈7× faster epoch-convergence to an equal-or-lower floor → gates (a)+(b)
of design §4 passed decisively.

**Real-valid transfer (caveat — valid sets differ):** CKLA best val/mse
**21.7** (rmse 4.48, R² at last ep −0.52) on the CLEAN
`DREGON-LM-V4-michaels-valid` (pin b6ece43d) vs E8 transformer's recorded
188.7 on the *contaminated* pre-cleanup valid — NOT directly comparable.
For scale: the E9 hard-combined transformer (50% neural gen + 50%
static-comb + augs) scored ~20.7 on the clean valid; CKLA matches that
from static-comb-only training with no augmentation.

**Fair rescore (uni-cpu `bash-653c8d`, eval.py, identical clean valid):**
E8 transformer best.ckpt → MSE **85.4**, rmse 9.01, mae_clip 7.90,
R² −1.37. CKLA P0 best (train-time val) → MSE **21.7**, rmse 4.48,
mae_clip 4.54 — **~4× lower MSE / 2× lower RMSE** at identical training
data and protocol. (E8 has no last.ckpt — predates the feature. Symmetric
eval.py scoring of the CKLA ckpt: `bash-94cdbf`.) Gate (c) supported by
the partial capture table (job `python-20c95f` OOM'd after the CKLA
block): graceful degradation — MAE 1.3–1.8 rev/s locked at aggressiveness
≤ 1, 3.3 at 2, capture lost only at 4; interference axis monotone and
mild. Rerun with `--mem 16`: `python-91cd13`.

### P0b — capture boundary + rotation ablation (uni-cpu `python-91cd13`, DONE)

`scripts/ckla_capture_boundary.py`, drift (aggressiveness 0.25–4) × SNR
(+10…−20 dB speech-rel), 16×4 s clips/cell, outputs in
`results/ckla_capture_boundary/` (pull via `omnirun pull python-91cd13`).

1. **CKLA locks, the transformer never does.** CKLA: lock fraction
   0.38–0.69 / MAE 1.3–1.8 rev/s at aggressiveness ≤ 1, degrading
   gracefully (3.3 at agg 2) to capture loss at agg 4 (15.6). E8
   transformer: lock fraction **0.00 in every cell**, MAE 3.3–5.2 — never
   a sustained lock even at the easiest cell. Gate (c) passed.
2. **Rotation ablation is NULL.** Zeroing s/ω0/W_ω on the trained model:
   ΔMAE within ±0.03 everywhere. Gate (d) FAILED — the trained model does
   not use the complex path.
3. **Parameter forensics** (best.ckpt): ω0 at ring init, s ≈ init 0.1,
   W_ω grew only to ~0.03 from zero-init, OU decay/noise params ≈ init.
   The model trained its projections, not its dynamics.

**Symmetric-harness confirmation** (`bash-94cdbf`, eval.py both models,
identical clean valid): CKLA best MSE **21.74** / rmse 4.48 / mae_clip
2.76 / **R² +0.50** vs E8 transformer best 85.4 / 9.01 / 7.90 / −1.37.

**Interpretation:** the P0 win belongs (so far) to the KLA
uncertainty-gated recurrence, not the complex extension — plausibly
because 1 s clips (T≈32) give phase accumulation nothing to do.

### `ckla_p0_norot` — train-time rotation-off control (kaggle `python-3c4ae9`, wandb `08k0ct9x`, DONE)

Best val/mse **21.51** (ep 4) vs rotation-on **21.70** (ep 7); train
convergence identical (~3.4 by ep 3–4 both arms). **At 1 s context the
complex rotation contributes exactly nothing** — proven from both
directions (eval-time ablation null + train-time control identical). The
1 s P0 win is entirely the real-KLA uncertainty-gated recurrence. The
complex hypothesis now rests on the 4 s pair `ckla_p0_4s` /
`ckla_p0_4s_norot` (kaggle `python-72ff01` / `python-761e7f`).

### P1 — `ckla_p1_if` (kaggle `python-3fd926`, wandb `s4u1tb7w`, DONE + vk_eval'd)

Trained 42 ep (best full-envelope val/mse 85.2 @ ep 22, then the familiar
overfit drift). vk_eval (uni-cpu `python-9d2f30`, results
`results/ckla_p1_vk_eval` on the job worktree; pull blocked by local
omnirun tmp disk-full — read via ssh):

| pool | g2_if floor | ckla_p1_best | VK bars |
|---|---|---|---|
| dregon_cruise | **2.481** | 2.87 (chmean+ma2; 3.02 stitch) | 0.68–0.74 |
| fly124_cruise | 2.33 | **1.36–1.39** (stitch/stitchmed) | 1.027 blind / 0.282 telem |

Pure head swap at identical front-end: **DREGON cruise +0.39 worse than
the floor; FLY124 cruise 40% better — the best neural cross-drone score
of the campaign**, within 0.35 of the blind-VK bar on that pool. The
Kalman-structured head generalizes across drones far better than
attention. `ckla_p1_last` confirms the pattern (dregon 3.03–3.06, fly124
1.36–1.44).

Levers launched next: `ckla_p1_4s` (4 s native context — refuted for the
transformer, untested for a recurrent tracker), `ckla_p1_norot`
(complex-path attribution on real data).

### `ckla_p0_4s` — 4 s synthetic context arm (kaggle `python-72ff01`, wandb `my6d6emg`, DONE)

Best val/mse **23.97 at epoch 0**, then monotone overfit (110 by ep 8,
run cut by the 9 h kaggle cap at ~1 h/epoch). vs the 1 s arms' 21.5–21.7:
**4 s native context bought nothing on static-comb** — consistent with the
data's nature (near-stationary RPS within clips). The 4 s rotation-off
twin was descoped after the queue stall (below); the rotation question at
4 s is subsumed by the real-protocol twins.

**4 s rotation attribution (eval-time, 2026-07-27, laptop light run):**
ablating the rotation path on the 4 s-trained `last.ckpt` (ep 8): ΔMAE
±0.05 across agg {0.5, 1, 2} × SNR {0, −10} — **null at 4 s as well**
(rotation params ≈ init: |s| 0.09–0.10, |W_ω| 0.04–0.05, ω0 drift ≤0.06).
The rotation-attribution matrix on synthetic is now closed at both
contexts; the only measured causal rotation effect remains real DREGON
(+0.31, activation analysis §A4). The descoped `ckla_p0_4s_norot`
training twin is fully redundant.

**Infra postmortem (2026-07-27):** the kaggle job finished but omnirun kept
it "running" — a ghost that starved the whole queue for ~1.5 days; on top,
the hetzner omnirun daemon's disk hit 99% (18 G of accumulated
`/var/lib/omnirun/artifacts`), failing every placement with ENOSPC (also
the earlier `omnirun pull` failure). Ghost + stale queue cancelled;
`ckla_p1_pnoise` / `ckla_p1_freqscale` resubmitted (colab T4) and place as
soon as the disk is freed (cleanup needs user authorization — pre-fix F1
artifacts ≤ 07-22 are the safe 8 G).

### Mechanistic activation analysis (2026-07-26, DONE)

Full §6-kit instrumentation of the trained `ckla_p1_if` head on 12 real
valid clips — see **[`ckla-activation-analysis.md`](./ckla-activation-analysis.md)**
(tooling: `scripts/ckla_activation_analysis.py` + `return_state`/
`capture_state` taps in `src/models/ckla.py`). Headlines: the head is a
fixed multi-horizon accumulator bank with a regime-level gate (gain φ/λ →
10⁻⁶, static slot mix, λ_v constant within cruise); rotation IS weakly
load-bearing on real DREGON (+0.31 PIT-MAE when zeroed, ω-excursions
track GT RPS at r = 0.82 in layer 2) unlike the P0 null, but null on
FLY124; and BOTH ckla_p1 and g2_if ignore a ×1.02 frequency scaling
(≈0.05% response vs ideal 2%) while CKLA is *more* gain/recoloring
sensitive — the "CKLA = comb-reader" cross-drone hypothesis is refuted.

### Mechanistic levers — full-envelope AND cruise-pool results (2026-07-27)

Both levers, trained on gpushort (Slurm 20927842/20927843 via legacy
sbatch after a first-run cache race; wandb `smwulrhf` / `hilihk2v`):

**Full-envelope val (fixed valid-full, PIT-MSE, best epoch)** — the levers
work exactly as designed: base `ckla_p1_if` 85.2 → `ckla_p1_freqscale`
**63.0** (ep 11) → `ckla_p1_pnoise` **44.8** (ep 39). Matched-protocol
transformer baselines: uni_gru128 172.3 / transformer-mag 72.7 /
transformer-IF (g2_if) 63.7. **pnoise 44.8 vs 63.7 = −30% at fully
matched conditions** (same stream/front-end/schedule; only the head + one
init constant differ) — the first clean architecture win of the campaign.

**Cruise pools (vk_eval, Slurm 20928550) — the ledger metric INVERTS:**

| arm | dregon_cruise best | fly124_cruise best |
|---|---|---|
| g2_if floor | **2.481** | 2.33 |
| CKLA base | 2.87 | **1.36** |
| CKLA pnoise | 3.94 (all arms ≥3.94) | 2.38 |
| CKLA freqscale | 2.60 (raw/none!) | 2.19 (ma5) |

Mechanistic synthesis: the cruise pools reward maximal within-clip
averaging (near-constant RPS) — the base model's accumulator degeneration
IS the fly124 1.36 win, and both levers destroy it by restoring
bandwidth. Full-envelope val rewards tracking — the levers win there.
The two metrics pull the gain knob in opposite directions. Notably
freqscale's best dregon cell is the RAW (unsmoothed) arm at 2.60 — closest
CKLA has come to the 2.481 floor, and consistent with spacing-reading
producing intrinsically stabler per-frame estimates. freqscale gives back
the fly124 accumulator win (2.19 vs 1.36).

Probe stage of the eval job crashed on a script dispatch bug
(`transformer_forward_taps` applied to a CKLA model) — λ-gain and
scale-response verification pending a fixed re-run.

### ⚠️ THE STAGING BUG (2026-07-27, found via the pnfs bit-identity anomaly)

`ckla_p1_pnfs` (freqscale policy, config-verified loaded, provenance-print
verified in-process) trained **bit-identical to plain-policy
`ckla_p1_pnoise` for 40 epochs** — twice (pnfs and the instrumented
pnfs2). Root cause: `OnlineMixFrameDataset(flatten_channels=True)` expands
each generated chunk into **C = 8 mono frames** (measured: 8.0 exactly on
the E12 stream), so the training loop consumes 8 frames per global sample
id — the policy's `until: 50000` boundary sits at **epoch ~80**, not 10.
No E12-family run ever exceeded 57 epochs. **Stage 2 (augmentations,
noise_augmentations, noise_time_warp) has never fired in ANY E12-recipe
experiment.**

Blast radius:
1. **Every "weak-aug + time-warp" E12 description is wrong** — all arms
   trained on plain mixtures throughout (incl. base e12, g1 4s/8s, g2,
   g3, g6-if-staged, freqscale, pnoise, pnfs, norot arms).
2. **The freqscale attribution collapses**: `ckla_p1_freqscale` (63.0,
   dregon 2.60) trained UNAUGMENTED — its differences vs base
   `ckla_p1_if` (85.2, dregon 2.87) are pure cross-hardware run variance
   (A100 vs P100, same effective config, seed 0).
3. **Which yields the campaign's first honest same-config variance
   estimate**: full-envelope ±22 MSE, dregon_cruise ±0.27, fly124_cruise
   ±0.8 — LARGER than several ledger margins (IF-channel Δ0.14; the
   fly124 1.36-vs-2.33 "win" Δ1.0 borderline; pnoise-vs-norot rotation
   gap Δ11.9 full-envelope — all within or near the variance floor).
4. Same-hardware same-seed runs are bit-deterministic (pnfs ≡ pnoise ≡
   pnfs2) — which masked the variance until the accidental replicas.
5. What stands: P0 synthetic margins (7×/4×/lock-vs-never — far above any
   noise floor), all activation-analysis mechanistic measurements, the
   pnoise gain-alive verification, G5's stage-1-augs-diverge observation.

Consequences: (a) stage boundaries need frame-accurate semantics (or
until values divided by the channel count); (b) NO ledger claim below the
variance floor is admissible without seed replicates; (c) the aug-lever
conclusions (G6 "refuted") need re-examination — G6's own staging must be
checked.

## Conclusion

_Being rewritten in light of the staging bug. Open: probes2 (checkpoint
properties, still valid), g2_if_freqscale (now a transformer variance
replicate), ebsrof debugging (failed to learn: val ~1150 flat)._
