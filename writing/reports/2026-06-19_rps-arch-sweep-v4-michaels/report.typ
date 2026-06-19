#import "/writing/templates/typst/report.typ": report, author-meta

#let render-table(rows) = text(size: 8pt)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (center, left, right, right, right, right, right),
    inset: 5pt,
    stroke: 0.4pt + luma(180),
    table.header(..rows.first().map(c => strong(c))),
    ..rows.slice(1).flatten()
  )
]

#show: report.with(
  title: [RPS-Predictor Architecture Sweep on DREGON-LM-V4-michaels],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    This report documents an autonomous architecture-search session
    (`20260617-012233-dregon-lm-v4-michaels-simple-conv-v2`) that swept 26
    rotations-per-second (RPS) predictor variants around the `simple_conv_v2`
    baseline. The same 26 model keys were trained under two regimes and
    evaluated on a single fixed validation set
    (`DREGON-LM-V4-michaels/valid`): (1) an offline fixed-train sweep
    (50 epochs, patience 10), and (2) an online-mixed rerun (200 max epochs,
    patience 50, augmentations after 50k samples). A subsequent export job
    saved raw validation predictions for all 52 resulting checkpoints. This
    document describes the experiment sets and presents the measured results
    only; interpretation is deferred.
  ],
  keywords: ("RPS prediction", "architecture sweep", "DREGON-LM", "online mixing", "autoresearch"),
)

= Experimental Setup

== Task, dataset, and baseline

The task is multi-rotor RPS estimation: predict four per-rotor rotation-rate
trajectories from drone audio. All runs in this session use:

- *Dataset:* `DREGON-LM-V4-michaels` (training source varies by regime; see
  below). All evaluation uses the single fixed validation set
  `DREGON-LM-V4-michaels/valid`.
- *Baseline:* `simple_conv_v2` — STFT log-magnitude front-end
  ($n_"fft" = 2048$, hop $512$), a residual + squeeze-excitation 2-D encoder,
  attention frequency pooling, and a bidirectional GRU temporal head.
- *Optimizer flags (shared by every run):*
  `--batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress`.

== Metrics

All variants are scored with permutation-invariant (PIT) matching between the
four predicted and four ground-truth rotor tracks:

- *PIT MSE* — primary metric, lower is better.
- *RMSE* — square root of the PIT MSE.
- *MAE/f*, *MAE/c* — per-frame and per-clip mean absolute error.
- *$R^2$* — coefficient of determination, higher is better ($1.0$ is the max).

== The 26 variants

The candidates were organised into hypothesis families spanning temporal-head
replacements, input-feature changes, frequency pooling, causal/streaming
stacks, and SMoLnet-style frequency-dilated backbones.

All keys below share the `simple_conv_v2` (`scv2`) prefix unless noted;
`smolnet_rps` is abbreviated `smol`.

#text(size: 9.5pt)[
  / Baseline: `scv2` — reference architecture.
  / Temporal head: `..._transformer` (global attention), `..._local_attn`
    (windowed attention), `..._gru96` (wider BiGRU), `..._tcn`
    (symmetric dilated TCN).
  / Input features: `..._multires` (two STFT resolutions), `..._dwt`
    (Haar-like temporal branch), `..._magphase` (log-mag + cos/sin phase).
  / Pooling: `..._dual_pool` (attention + mean frequency pooling).
  / Causal / streaming: ten variants in the `uni_gru` / `causal_gru` /
    `causal_tcn` families — unidirectional or fully time-causal heads at
    hidden sizes 64–128, with optional GroupNorm and dropout 0.2–0.3
    (full key list in the result tables).
  / SMoLnet / freq-dilated: seven variants combining SMoLnet-style
    frequency-dilated bodies with TCN, BiGRU, or simple heads, both as
    standalone `smol` models and as `scv2` + `smol` refinements
    (e.g. `..._smol_tcn`, `..._smol_causal_tcn`).
]

= Experiment Set 1 — Offline Fixed-Train Sweep

Each variant was trained for up to 50 epochs (early-stopping patience 10) on
the fixed offline `DREGON-LM-V4-michaels` train set, using the shared optimizer
flags. Every candidate passed a single-forward-pass smoke test
(output shape `(2, 4, 94)`) before submission. All 26 runs completed on the
Slurm `gpushort` partition.

#figure(
  image("assets/fig_offline_leaderboard.png", width: 88%),
  caption: [
    Offline fixed-train sweep: validation PIT MSE for all 26 variants
    (log scale, sorted best-on-top). Colour denotes hypothesis family; the
    dashed red line marks the `simple_conv_v2` baseline.
  ],
)

#figure(
  render-table(csv("assets/offline.csv")),
  caption: [Offline fixed-train leaderboard, ranked by PIT MSE.],
)

= Experiment Set 2 — Online-Mixed Rerun

The same 26 model keys were retrained with online mixture generation rather
than a fixed offline train set, keeping identical optimizer flags. Setup
(2026-06-18):

- *Train stream:* `configs/online_mix_v4_michaels_train_no_room1_gpfs.yaml`.
- *Noise sources:* DREGON in-flight noise (excluding
  `free-flight_nosource_room1`) plus Michael's `FLY125`.
- *Speech:* LibriSpeech `train-clean-100` (one unreadable FLAC excluded).
- *SNR:* uniform $[-30, 0]$ dB; independent speech per channel; SNR not
  per-channel.
- *Augmentation:* enabled after 50,000 global samples ($p = 0.5$): random gain,
  polarity flip, or single-channel drop.
- *Budget:* 200 max epochs, patience 50, 5,000 samples per validation.
- *Validation:* identical fixed set `DREGON-LM-V4-michaels/valid`.

Of the 26 runs, 25 completed and 1 (`smolnet_rps_tcn`) hit the 1-hour
`gpushort` wall clock before its final-checkpoint evaluation; its row uses the
best logged validation metrics (marked `*`).

#figure(
  image("assets/fig_online_leaderboard.png", width: 88%),
  caption: [
    Online-mixed rerun: validation PIT MSE for all 26 variants (log scale,
    sorted best-on-top). `*` = timed-out run scored on best logged row.
  ],
)

#figure(
  render-table(csv("assets/online.csv")),
  caption: [
    Online-mixed leaderboard, ranked by PIT MSE. `*` marks the timed-out
    `smolnet_rps_tcn` run, scored on its best logged validation row rather than
    a loaded-best final evaluation.
  ],
)

= Cross-Series Comparison

Both series share the same model keys and the same fixed validation set, so the
per-model scores can be paired directly. Of the 26 variants, 21 scored a lower
(better) PIT MSE under online-mixed training (points below the diagonal in
Figure 3); the five exceptions were `simple_conv_v2`, `scv2_gru96`, `scv2_tcn`,
`scv2_smol_tcn`, and `scv2_smol_causal_tcn`. The largest improvements belong to
the variants that scored worst offline, while the high-$R^2$ region is densely
populated in both regimes (Figure 4).

#figure(
  image("assets/fig_offline_vs_online.png", width: 72%),
  caption: [
    Per-model PIT MSE under offline vs. online-mixed training (log–log). Points
    below the diagonal scored lower (better) PIT MSE under online mixing; the
    most extreme movers are labelled.
  ],
)

#figure(
  image("assets/fig_pit_vs_r2.png", width: 100%),
  caption: [
    PIT MSE (log) vs. $R^2$ for all 26 variants, offline (left) and online-mix
    (right). Colour denotes hypothesis family.
  ],
)

= Discussion

== Online mixing as the default regime — but not a uniform lift

The two series share model keys and a single validation set, so the per-model
deltas are directly comparable (Figure 3). The dominant effect is that online
mixing with augmentation rescues the models that were broken or overfit
offline: the global transformer ($43.52 arrow 8.46$), and essentially the
entire unidirectional-GRU family (e.g. `scv2_uni_gru` $228.67 arrow 8.73$,
`scv2_uni_gru128` $39.81 arrow 7.33$). The mechanism is the standard one —
an effectively unbounded training stream plus per-sample augmentation removes
the fixed-set overfitting that the 50-epoch offline runs were exposed to.

It is *not*, however, a uniform improvement. Several models that were already
strong offline are slightly *worse* online: `scv2_smol_tcn`
($9.08 arrow 12.61$), `scv2_smol_causal_tcn` ($8.38 arrow 8.99$), `scv2_gru96`
($8.66 arrow 10.79$), and even the plain baseline drifts ($7.89 arrow 8.54$).
Online mixing therefore compresses the field by lifting the failures rather
than raising the ceiling, and in doing so it *reshuffles the architecture
ranking*: the best offline family (v2 + SMoLnet refinement) is no longer best
online, where the wide unidirectional GRU takes the top PIT MSE. The practical
takeaway is that online mixing should be the default training regime, but model
selection must be done *under that regime* — offline rankings do not transfer.

== Residual temporal overfitting

Online mixing diversifies the acoustic *dressing* of each rotor-speed
trajectory (speech, SNR, augmentation) but not the underlying *set* of RPS
sequences, which remains small. Models with explicit temporal dependence can
still overfit this limited menu of trajectories. The residue is visible:
`scv2_causal_gru` remains the weakest GRU online ($14.64$, $R^2 = 0.77$, below
baseline), and the BiGRU baseline reaches its best epoch only at epoch 42 —
a slow grind on limited sequence structure. Addressing this would require
augmenting the RPS trajectories themselves (e.g. time-warping / speed
perturbation of the curves, or synthetic trajectory generation), which acoustic
mixing cannot provide.

== Why the causal-RNN offline$arrow$online swing is so large

The unidirectional / causal recurrent heads show by far the largest regime
sensitivity. Three effects stack, and only the second is the substantive
mechanism; the other two inflate the apparent magnitude.

+ *Half the offline causal numbers measure a broken run, not an architecture.*
  A unidirectional GRU with fixed LR $1e^{-3}$ and AMP sits on a stability
  knife-edge: `scv2_uni_gru` went to NaN after epoch 10/11 (final eval loaded
  the epoch-4 checkpoint), `scv2_uni_gru128` logged a NaN train row, and
  `scv2_uni_gru64_norm_do03`'s best checkpoint (epoch 6) itself had a NaN train
  loss. Early-stopping then loads garbage. So the offline causal leaderboard
  largely ranks *which configuration happened not to diverge*. The evidence
  that this is artifactual: offline, the ordering is dominated by stability
  (the finite-staying `scv2_uni_gru96_norm_do03` lands on top); online, the
  ordering flips to the architecturally-sensible one — *capacity*
  (`scv2_uni_gru128` $7.33$ > `..._uni_gru128_norm` $7.99$ > the 96-wide
  variants $approx 8.3$). The online run is the first clean read of these
  models.

+ *Fresh-sample gradients fix both pathologies the GRU uniquely suffers from.*
  An unbounded online stream replaces memorized-sample gradients with
  fresh-sample estimates, smoothing the expected loss surface — the classic
  stabilizer for recurrent training, which is why the NaNs mostly vanish.
  Independently, it breaks the autoregressive shortcut: a causal GRU can
  memorize "given this prefix, the RPS curve continues like $X$" from a handful
  of repeated sequences instead of reading harmonics; online, the same
  trajectory recurs under many SNRs / speakers / augmentations, so memorizing
  the continuation stops being reliable and the model is pushed onto the real
  spectral feature. The GRU swung most because it is the one family hit by
  *both* recurrent instability *and* autoregressive memorization; conv/TCN heads
  have neither acutely, and the BiGRU baseline was already in a stable basin
  (and so barely moved).

+ *A training-budget confound.* The online runs used 200 epochs / patience 50
  versus the offline 50 / 10. Some gain could be a stable-but-slow GRU finally
  converging under a forgiving early-stop. This does not explain the NaN cases
  and the online winners early-stop early (best epoch 17), so it is not the main
  story, but it means offline-vs-online is confounded by
  (data regime) $times$ (epoch budget) $times$ (patience) rather than a clean
  A/B.

The clinching counterexample for a *genuine* stability floor underneath the
data-regime effect is `scv2_uni_gru64_norm_do03`, which fails in *both* regimes
(offline $95.48$, online $88.81$, NaN in both): online mixing cures the
instability at hidden size 128/96 but not at the too-narrow 64 head.

== Follow-up: unidirectional RNNs with gradient clipping

To disambiguate the three effects above, the cheapest isolating experiment is
to rerun the unidirectional-GRU family with gradient clipping (and otherwise
matched settings), testing whether the offline instability — and hence much of
the apparent swing — is removed once divergence is controlled:

- If the clipped heads train stably *offline* and recover competitive scores,
  the large offline$arrow$online swing was mostly artifactual (instability +
  early-stop-on-garbage, possibly compounded by the epoch budget).
- If they remain poor offline despite stable training, the swing is genuinely
  driven by training-data diversity (residual temporal overfitting), confirming
  the mechanism in the previous subsection.

#block(fill: luma(238), inset: 9pt, radius: 4pt, width: 100%, stroke: 0.5pt + luma(170))[
  *Results pending.* The gradient-clipped unidirectional-RNN runs are in
  progress; this subsection will be filled in with their PIT MSE / $R^2$ and the
  resulting disambiguation once the jobs complete.
]

= Validation-Prediction Export

Training logs retained only checkpoints, W&B IDs, and printed aggregate
metrics — not raw predictions. A separate `gpushort` job
(`12642388`, `save_validation_predictions.py`) re-evaluated all 52 best
checkpoints (26 offline + 26 online) on `DREGON-LM-V4-michaels/valid` and saved
per-checkpoint arrays for downstream analysis and plotting.

Each checkpoint folder now contains a `validation_rps_predictions/` subdirectory
with:

- `pred_raw.npy` — raw model output before PIT rotor-order matching, shape
  `(rows, 4, frames)`.
- `target.npy` — validation target on the prediction frame grid.
- `target_pit_matched_to_pred.npy` — target reordered by the PIT-optimal rotor
  assignment for metric/plot overlays.
- `sample_ids.npy`, `channels.npy` — per-row metadata.
- `metadata.json` — checkpoint path, shapes, and quick metrics recomputed from
  the saved arrays.

Series roots holding the per-model `validation_rps_predictions/` folders:

- Offline:
  `.../autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/<model>/`
- Online-mix:
  `.../autoresearch/20260618-v4-michaels-online-mix-200ep-aug50k-gpushort/<model>/`
