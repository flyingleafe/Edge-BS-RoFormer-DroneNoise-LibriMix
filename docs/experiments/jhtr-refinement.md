# Joint Harmonic Trajectory Refinement (JHTR)

**Status:** CPU engineering and full synthetic data-parity gates passed, 2026-09-06; GPU profiling and training pending. No learned JHTR result is recorded yet.

## Motivation

Test whether rich phase-product observations, freshly read around jointly updated rotor trajectories, can improve corrupted guesses without moving correct guesses. Conditional precision is the first claim; success does not establish audio-only acquisition or cross-rig transfer.

Historical context only: the September 4 candidate campaign reports HG-CKLA-v2 regressor-seeded cruise refinement 2.28 → 1.85 → 1.79 → 1.73 rev/s and oracle displacement 0.55 → 0.75 → 0.88 over whole-operator passes. These are historical real-benchmark values, not matched S1/S2 controls or new JHTR results. See [candidate tests](candidate-tests-2026-09-04.md). The published mono comb→R4 result of 2.74 all-frame PIT-MAE is also historical and cannot be compared directly to a privileged-input synthetic refiner.

**User scope decision:** do not train or evaluate new HG-CKLA controls. The compute campaign contains JHTR and its power-only, frozen-read and independent-slot ablations. Historical HG-CKLA results motivate the stability question, but removal of its matched compute also removes any supported new head-to-head conditional architecture claim against it.

## Frozen recipes and experiment order

Start fresh, separate conditional S1 and S2 pilots:

- `jhtr_cond_s1_nomix` inherits `salv2_scv2_comb_nomix`.
- `jhtr_cond_s2_nomix` inherits `salv2_scv2_stoch_nomix`.
- Each has `_power`, `_frozen`, `_independent` twins. The first zeros product features, the second reads once at initialization, the third removes cross-slot interactions and competing-track descriptors. These are matched architectures, not a different dataset or optimization regime.
- Speech checks are `jhtr_cond_s{1,2}_mix`, conditional real is `jhtr_cond_real`. Audio-only rows and the real ladder/C1 curriculum are later, gated experiments, not a commitment to spend compute after a failed mechanism gate.

The only synthetic conditional task bridge is `use_cond: true`, unchanged `mse_cond`, and the existing corruption (train seed 20260729, fixed validation seed 777). The parent waveform pipeline, source mixture, augmentation, crop, labels apart from existing whole-row alignment, optimizer, final-only objective, monitor and exposure are unchanged. Standard guesses must come from the fixed Frames' actual `rps_cond`, **not** the legacy real benchmark's simplified index-based corruption.

S1/S2 inherit 4 s training, 8 s fixed validation, batch 32, 16,000 mono examples per validation interval, 200 epochs maximum, patience 20, AdamW 1e-3/1e-4, plateau factor .5/patience 5, `mae_frame` selection, AMP, gradient clip 5, and seed 0. Eight microphones are separate mono examples. For the inherited loop that is 500 batches per complete interval. Record completed epochs, actual examples/updates from logs, wall-time truncations and replayed partial epochs; nominal ceilings are not observed exposure. Do not silently change effective batch/accumulation to accommodate memory.

C1 must use its own matched stage-1 lineage and as-run stage-2 recipe, not SALV2 weights substituted for its different pretraining. Real-only rungs do not receive a synthetic warm start. Preserve the real parent's monitor differences and actual warm-up choices. See [paper regime matrix](paper-regime-matrix.md).

## Measurement implementation

`experiments.refiner_bench.run()` is unchanged, including its historical real split, resampling and MAE-optimal PIT conventions. New `evaluate_jhtr()` consumes supplied fixed Frames and an optional loaded FrameModel. `scripts/jhtr_campaign.py` is thin CLI glue around existing Hydra, dataset, trainer model/codec, and checkpoint resolver seams; it is not a trainer, scheduler, renderer or selector.

A local `results/<experiment>/best.ckpt` can be evaluated before zoo publication. Prefer `--config` pointing to that run's saved resolved config; using a current YAML to load historical weights is not recipe-fidelity evidence. `best.ckpt` and `last.ckpt` are bare state dictionaries; `train_state.pt` contains separate optimizer/scheduler/epoch state. Never choose an epoch using oracle drift or capture tests.

### Saved evidence

Each evaluation directory holds `resolved_config.yaml`, `results.json`, and one compressed NPZ per case. JSON records the experiment/checkpoint, local checkpoint hash when available, dload-lock hash, full metadata, original group IDs, complete frame fingerprints and explicit missing-coverage status. Each NPZ contains:

- Original conditioning at block zero, all six block trajectories, three full-operator outputs with hidden state reset at each public call, conditioning-ordered targets, and every output timestamp.
- Per-example ordered MSE/MAE; existing MAE-optimal PIT-MAE; existing MSE-optimal PIT-MAE and PIT-MSE; existing `mae_clip` (MAE between time-averaged rotor speeds, not mean per-clip `mae_frame`). No replacement metric definitions.
- Applicability and natural regime/separation masks, original flight/recording IDs, immutable frame fingerprints, recovery success/start/longest interval and identity recovery.
- Boundary frames remain in ordinary metrics. Phase-invalid boundary observations never delete benchmark frames.

The fixed-set scalar/block summaries include **all** examples. Applicability-filtered strata/bootstrap summaries are separately identified by their counts; offset clipping, impossible doubles, missing active cases and empty natural separation bins never disappear silently. All-four recovery uses one whole-crop MSE-optimal permutation, then requires all four errors ≤1 rev/s for timestamps spanning at least .5 s. At the 32 ms grid, 17 points span .512 s; 16 points span only .480 s and fail. This is an offline successful interval, not causal latency or physical identity recovery at an unidentifiable crossing.

Guesses include existing standard corruption, oracle, all signed ±.5/1/2/4 offsets, half/double where legal, collapse onto an existing active track, duplicates, missing-active, false-active stopped slots, and a .5 s wrong-track interval. Only guesses change; no new examples or corruptions enter training. `locator` calls the model with no telemetry and saves initializer versus refinement. A conditional-only checkpoint's locator was not trained as an audio-only initializer; its diagnostic cannot establish audio-only training performance.

Natural separations are exact coincidence and (0,.1), [.1,.25), [.25,.5), [.5,1), [1,2), [2,∞) rev/s, with crossing sign changes reported separately. No invented examples populate empty bins. Activity rates use the existing .5 rev/s threshold after whole-crop matching; the legacy corruption preserves OFF and therefore ordinary conditional validation alone cannot establish activity recognition. False-active/missing-active probes test non-oracle guesses explicitly. Octave failure fraction is within 1 rev/s of half/double truth, evaluated on truth above 4 rev/s to avoid a trivial near-OFF overlap.

**Source-known conditional symmetry limitation:** identical supplied seed rows
remain identical under JHTR's shared row-equivariant initialization and updates.
There is no conditional slot identity/bias to break this symmetry. Thus exact
duplicates cannot split, and all-collapsed seeds cannot recover four distinct
tracks, regardless of training. Report these probes as a structural acquisition
limit, not a learned precision failure. Near-duplicate inputs and already
distinct rough guesses remain different questions. Retain the exact-collapse
diagnostics and strict all-four scoring; do not add slot biases or an objective
to rescue this architecture comparison.

### Original-group uncertainty

`paired_group_bootstrap` draws 10,000 paired bootstrap samples of **original** recordings/flights. Microphones, speech twins and corruption variants travel together; they never count as independent confidence samples. Sample-weighted point means are preserved using resampled group sums/counts even with unequal group sizes. One independent group yields `unestablished`, not a confidence interval.

Synthetic `sample_id` is accepted as flight identity only when the composed fixed dataset declares `flight_reuse: 1`; otherwise provide independently established `--groups` JSON in exact frame order. Real crops must be grouped by original recording, not clip or microphone. Verify any external group mapping against source provenance. A trajectory bootstrap at model seed 0 does not measure initialization variability. `compare` rejects different frame fingerprints, targets, groups, applicability or conditional guesses, then saves paired overall and natural-stratum intervals. Stratum comparisons weight each covered example equally and label that convention; they are not relabeled whole-dataset metrics.

### Observable-subset limitation

Ordinary fixed synthetic Frames retain mixture, RPS, sample ID and channel, but not sufficient per-source order powers and local-floor metadata to certify three orders per rotor ≥6 dB above floor and ≥16 Hz from foreign lines for .5 s. Mixture peaks alone cannot prove source attribution. Therefore the default precision-subset status is **unestablished**, with full-set results still measured. Do not replace the parent set with favorable tones or a new renderer to manufacture coverage.

An externally certified `--observable` NPZ may contain a boolean `mask` of shape (N,4,T), accompanied by `--observable-provenance` identifying independent existing-source evidence and the frozen criterion. Freeze that evidence before inspecting model errors; do not select easy frames after observing results. An arbitrary boolean mask is not itself proof. Unsupported/empty coverage limits the precision claim even if full-set MAE is favorable.

## Verification and scientific gates

Main owns all validation and launches. No worker test/build/lint/formatter or experiment execution is claimed in this record.

1. **Before training:** validate the tensor reader with tone sign/rate, exact chirp timestamps, phase gauge, noise/silence, 1/4/8 s boundary/frame counts, autograd/finite difference and doubled-padding checks. Require isolated-tone inferred error <.02 rev/s and normalized phase-gauge agreement <1e-4. Run real CPU model forward/backward; profile GPU matched batch and duration. A failed numerical gate blocks training.
2. **Recipe/data parity:** strict fully resolved parent-child comparison; all 256 fixed or paired512 examples instantiated independently with/without conditioning. Require non-target bytes, every series' timestamps/dims, metadata, IDs/order/count unchanged; target rows may only permute. Verify repeat access and independent rebuild of guesses, plus identical speech-twin target alignment/guesses. Do not treat a smoke subset as parity proof.
3. **Conditional precision:** at each signed offset require ≥20% mean error reduction versus identity on certified observable coverage. At **every** block oracle displacement mean ≤.10 and p95 ≤.25 rev/s. Identity passes preservation but fails correction. Publish full-set drift, not just subset drift. All eight signed offsets must be measured for the combined gate.
4. **Fixed-point claim:** three complete six-block operator applications, hidden reset each time, must preserve the oracle tolerance. Six-block preservation alone is a narrower claim. Repeat summaries are not a convergence guarantee.
5. **Mechanism:** full versus power/frozen must each reduce conditional MAE by ≥10%, with positive paired 95% improvement interval. Independent-slot comparisons focus on naturally close/crossing cases. A tie supports the simpler architecture. Frequency-scaling and harmonic-cutoff sensitivity reuse the existing paper campaign probes on fixed selected checkpoints, never new augmentation or checkpoint selection.
6. **Acquisition:** all-four offline recovery success must improve over identity and power-only on certified observable cases, with positive paired interval; report full set and unresolved coincidence separately. Audio-only adoption additionally requires ≥10% lower all-frame PIT-MAE versus strongest genuinely matched mono reference, positive paired interval, and no named regime worsening by >max(.1 rev/s, 5% of reference). Do not infer this from a conditional checkpoint's privileged input.

### First integration finding: conditioning changed waveform RNG

The first union gate exposed a data-contract defect, not an allowable parity
tolerance: appending a conditioning `pipe.map` made it the dload pipeline root.
Dload's preorder node IDs salt `Random`/`Shuffle`/`Mix` RNGs, so the additional
root changed upstream waveform/nuisance draws even though corruption itself
used a separate seed. The correction applies the existing `_corrupt_frame` at
the `OnlineMixFrameDataset` iteration boundary, after the unchanged audio and
channel-flattening adapter. Both online training and fixed validation then use
the parent's unmodified dload topology.

Corruption seeds, `sample_id * 256 + channel`, row swaps/twins, epoch handling
and worker sharding remain unchanged. Historical conditional streams built
with the extra root do not replay byte-for-byte under the corrected seam;
that old behavior is not a matched-parent reference. The full256/paired512
checker remains strict about waveform bytes, timestamps, metadata, sample
order and target-only row permutations. Main must rerun the union gate and
the unchanged commands below; source correction alone is not parity proof.

### Commands for Main

Run from the dedicated worktree with its environment (`PYTHONPATH=src` if using a shared environment). These do not launch training:

```bash
python scripts/jhtr_campaign.py --help
pytest tests/experiments/test_jhtr_bench.py -q
python scripts/jhtr_campaign.py check --parent salv2_scv2_comb_nomix --experiment jhtr_cond_s1_nomix --conditional-bridge --data --out results/jhtr/check-s1.json
python scripts/jhtr_campaign.py check --parent salv2_scv2_stoch_nomix --experiment jhtr_cond_s2_nomix --conditional-bridge --data --out results/jhtr/check-s2.json
python scripts/jhtr_campaign.py check --parent salv2_scv2_comb_mix --experiment jhtr_cond_s1_mix --conditional-bridge --data --out results/jhtr/check-s1-mix.json
python scripts/jhtr_campaign.py check --parent salv2_scv2_stoch_mix --experiment jhtr_cond_s2_mix --conditional-bridge --data --out results/jhtr/check-s2-mix.json
python scripts/jhtr_campaign.py check --parent jhtr_cond_s1_nomix --experiment jhtr_cond_s1_nomix_power --out results/jhtr/parity-s1-power.json
python scripts/jhtr_campaign.py profile --experiment jhtr_cond_s1_nomix --device cpu --smoke --out results/jhtr/cpu-smoke.json
python scripts/jhtr_campaign.py profile --experiment jhtr_cond_s1_nomix --device cuda --out results/jhtr/gpu-profile.json
```

Repeat no-bridge config parity for frozen/independent and S2 variants. Then, **only after** the parent's monitor has selected checkpoints:

```bash
python scripts/jhtr_campaign.py evaluate --experiment jhtr_cond_s1_nomix --checkpoint results/jhtr_cond_s1_nomix/best.ckpt --device cuda --n 1 --cases standard oracle --out results/jhtr/s1-smoke
python scripts/jhtr_campaign.py evaluate --experiment jhtr_cond_s1_nomix --checkpoint results/jhtr_cond_s1_nomix/best.ckpt --device cuda --out results/jhtr/s1-full
python scripts/jhtr_campaign.py evaluate --experiment jhtr_cond_s1_nomix_power --checkpoint results/jhtr_cond_s1_nomix_power/best.ckpt --device cuda --out results/jhtr/s1-power
python scripts/jhtr_campaign.py compare --reference results/jhtr/s1-power --candidate results/jhtr/s1-full --out results/jhtr/s1-phase-comparison.json
```

Pass `--config <saved-resolved-config.yaml>` for as-run fidelity. Repeat for frozen/independent/S2 selected checkpoints with unchanged cases and fixed data. Use existing `python train.py experiment=...` and omnirun (`uni`, `uni-gpushort` on the current host) for authorized training; this measurement CLI does not orchestrate it.

## Results

Engineering evidence from the dedicated worktree:

- New targeted regressions: **48 passed, 1 CUDA-only skip**; tone/chirp, gauge,
  padding, masks, gradients, model/ablation behavior, conditioning and scoring.
- Affected existing DSP/online-mixing/corruption regressions: **64 passed**.
  Import-linter: **3 contracts kept, 0 broken** (241 files, 544 dependencies).
- Full six-block, 32-order, width-128 model: **5,824,729 parameters**. CPU smoke,
  batch one / 1 s / 32 frames: cold forward **2.179 s**, backward **3.288 s**;
  finite outputs and parameter gradients. This is not GPU throughput.
- Strict independently rebuilt validation parity: **256/256 S1**, **256/256 S2**,
  **512/512 speech-paired S1**, **512/512 speech-paired S2**. Non-target bytes,
  timestamps, metadata, sample order and deterministic conditioning passed.
- All **21 composed experiment recipes** checked. C1 stage two differs additionally
  only by its required own-stage-one checkpoint; stage-one recipe parity passed,
  but checkpoint existence/training is not established.
- Actual trainer preflight passed for `jhtr_cond_s1_nomix`; no optimizer training
  was requested by that `validate_only=true` invocation. Touched-file Ruff passed;
  Pyright reported zero errors after the integration corrections.

Raw evidence is under `results/jhtr/`: `cpu-smoke.json`, `check-s1.json`,
`check-s2.json`, `check-s1-mix.json`, `check-s2-mix.json`, and
`all-recipe-parity.json`. GPU memory, learning curves and selected-checkpoint
precision/capture/oracle measurements remain pending.

The real checkpoint-loading/evaluator CLI also passed an explicitly **untrained**
one-example smoke: seven saved block states `(1,7,4,251)`, four complete-operator
states `(1,4,4,251)`, exact timestamps, NPZ/JSON output, and correctly unestablished
one-group uncertainty. `results/jhtr/evaluator-smoke.json` records the integration
proof; disposable untrained weights/output arrays were removed. Smoke output
explicitly makes no parent-monitor selection or learned-performance claim.

Measured unchanged-parent coverage (`results/jhtr/parent-coverage.json`): S1 spans
0–103.688 rev/s with 12.50% stopped rotor-frames; S2 spans 0–107.462 rev/s with
25.57% stopped rotor-frames. All four truth rows coincide at 28.77% of S1 and
35.42% of S2 timestamps (including stopped states); entire coincident clips are
21.875% and 31.25%, respectively. These are retained, not filtered out. The first
smoke example happens to have coincident rows; it is not representative evidence
of four-track separation.

GPU gate `jhtr-gpu-gate-068355` failed before tests: the shared cluster editable
environment imported a different checkout. All subsequent jobs explicitly set
`PYTHONPATH=src`. Source-pinned gate `jhtr-gpu-gate-src-b55d6c` passed **37 tests**
(including the CUDA reader check), then failed in full-batch CUDA SDPA with
`invalid configuration argument`. `_AttentionFF` now chunks its **independent
sequence batch axis** to bound CUDA launch dimensions; neither the training
batch, trajectory/time attention domain, loss nor schedule changes. A large-batch
value/gradient equivalence regression covers the boundary. GPU re-verification
is still required; this is not a fit claim.

**User-authorized compute:** at most **$30 combined** for paid S1/S2 pilots, after
the free GPU gate passes. Paid ablations require further approval. Current offers
are roughly $0.95–$1.03/A100-80-hour; no paid jobs have been launched yet.

## Conclusion and failure strategy

No effectiveness conclusion yet. If a numerical gate fails, inspect interpolation, integration, normalization, masking, padding and gradients before training. If optimization fails, inspect actual processed exposure, gradient finiteness and final-only inherited loss; do not add rescue objectives/curricula. If full-set learning improves but conditional precision drifts, examine noisy local phase, interference, collisions, identity ambiguity and synthetic/real mismatch. Preserve the selected checkpoint and report failure; do not select an oracle-friendly epoch or silently widen the network.

A precision pass without acquisition supports a conditional refiner only. A power/frozen tie rejects the corresponding phase/re-reading claim and favors the simpler control. Downstream real/audio-only compute remains gated on these decisions. Scientific failure with saved evidence is a completed outcome, not unfinished implementation.
