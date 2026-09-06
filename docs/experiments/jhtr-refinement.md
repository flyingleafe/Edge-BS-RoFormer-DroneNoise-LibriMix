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
`all-recipe-parity.json`. GPU fit and early learning evidence are recorded below.
Final selected-checkpoint precision/capture/oracle measurements remain pending.

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
value/gradient equivalence regression covers the boundary. The subsequent P100
gates below establish fit on that device; the queued A100 profile is separate.

Free Kaggle gate `jhtr-kaggle-cuda-baf7fe` passed **39 tests**, including both CUDA
regressions. Full-model batch-one / 1 s FP16-autocast forward/backward passed on a
Tesla P100 (PyTorch 2.7.0+cu126): **1.732 s**, **212,417,024 bytes** peak allocated,
finite predictions and gradients. This is a small smoke, not a matched-batch fit
or steady-state throughput measurement. Pulled evidence: `cuda-tests.xml` and
`cuda-smoke.json` under `results/jhtr/`. The slim deployment branch
`research/jhtr-kaggle-validation` checks source hashes against implementation
`7a29a301ec79b247d2e7ebcc5042bc5ae80bb826`; publishing that branch was necessary
for the remote omnirun daemon to transport it. Local omnirun has no configured
Kaggle backend. Use Kaggle first for these small CUDA gates, not the cluster queue.

The unchanged **batch-32** profile also passed on Kaggle
(`jhtr-kaggle-profile-9a55a5`): 1 s training used **3.403 GiB** allocated,
4 s training **13.280 GiB** (14.475 GiB reserved), and 8 s validation
**6.453 GiB**. Cold forward/backward times were 5.756/4.628 s and
4.295/14.736 s for the two training sizes; validation forward was 8.500 s.
The portable probe uses the canonical profile loop and resolved configuration,
with only native JHTR construction made dependency-light and completed rows
printed. No optimizer step or data loading is included in those times.
Pulled evidence: `results/jhtr/kaggle-matched-profile.json`.

**User-authorized compute:** at most **$30 combined** for paid S1/S2 pilots.
Paid ablations require further approval. Since the matched profile fits 16 GiB,
24 GiB RTX 4090 offers ($0.22–$0.32/hour) are preferable to A100-80 offers
($1.00–$1.04/hour). Reserve at most **$12.50 per pilot** against a 36 h estimate,
wrap each complete preflight/training command in a **34 h hard runtime cutoff**,
and monitor/release both rentals. The $5 unreserved margin covers setup/teardown;
`omnirun --max-cost` is a placement estimate, **not** a runtime billing cap.
No epochs, patience, sample exposure per validation, or batch settings change.
If a runtime cutoff interrupts inherited training, report incomplete exposure,
not convergence or scientific model failure.

Pilots launched from `1fea49e75d30`: S1 `jhtr-s1-pilot-e21bc1` and S2
`jhtr-s2-pilot-c4f64d`, both RTX 4090. Selected 36 h reservations are
**$11.1200** and **$8.0267** (**$19.1467 combined**); these are estimates, not
billed spend. The omnirun field named `Placement.cost_actual` is populated with
the placement estimate, so it must not be reported as an invoice. Accounting
snapshot: `results/jhtr/paid-pilot-ledger.json`. S1 passed its 39-test CUDA
preflight and canonical matched profile, then entered the real trainer; early
observed throughput is approximately **4.1 s/batch**, not a convergence result.
The now-redundant queued A100 profile was cancelled after the P100 fit passed.

**Live-rental safety:** do not `omnirun pull` an active Vast job: the backend's
`pull_outputs` unconditionally auto-terminates after pulling. Use streamed logs
and W&B during training; pull after terminal state and verify `reaped`.

Canonical evaluator GPU smoke `jhtr-kaggle-evaluator-369641` succeeded for both
pilot checkpoints: **32 examples / 4 groups**, inference batch 32, standard and
oracle cases, seven block states `(32,7,4,251)`, three complete applications plus
input `(32,4,4,251)`, finite arrays and recorded checkpoint hashes. The n32 subset
is explicitly **engineering smoke**, not precision or selection evidence.
`results/jhtr/kaggle-evaluator-smoke.json` records the pulled artifact checks.
The complete canonical evaluator exceeds Kaggle's embedded-source cap; the thin
deployment instead downloads an immutable private R2 source archive, verifies
its SHA-256, installs the original `uv.lock`, and runs the unchanged CLI. Source:
`cf35c33a10f5`, archive hash
`acf00f0d75bab0629a89850f74bf09d79ed509f2b2b014ae10283cf098005fab`.
Kaggle emitted a nonfatal `sitecustomize` missing-`wrapt` warning; both canonical
evaluations still completed successfully. No warning suppression or lock change
was used.

Early learning snapshots are not final results. Complete fixed-set identity
MAE-frame baselines are **1.050398 (S1)** and **0.871646 (S2)**. Epoch-zero
monitor values were 0.876673 and 0.774290; the first ten completed S1 epochs
reached 0.5665, while the first eight S2 epochs reached 0.6941. S1's epoch-zero
optimizer state records **497 actual updates from 500 batches**, with AMP scale
8192. Histories and native baseline arrays are under `results/jhtr/`; no oracle,
signed-offset, fixed-point or mechanism conclusion follows from these monitors.

The existing paper cue-probe helpers originally discarded `rps_cond` when
rebuilding filtered/resampled Frames. Both reproduced `KeyError: rps_cond` with
the native conditional codec. `scripts/rps_cue_probe.py` now retains a supplied
prior without frequency scaling, crops it to the frequency probe's shared
physical span, and preserves the waveform's absolute time origin. Existing
alpha/cutoff grids and readouts are unchanged. Two native-model regressions
(frequency and cutoff, including a 17.25 s time translation) failed before the
fix and passed after it. Ruff passed; direct Pyright completed successfully
after the shorter LSP/CLI checks timed out. This changes evaluation helpers only,
not the running pilots or their data/optimization.

Preliminary cue job `jhtr-kaggle-cues-live-a4d4f1` completed and was pulled.
It uses each pilot's unchanged synthetic fixed set, **not** the paper's real
frequency-probe part: six original cruise-selected clips, all 13 established
alphas, and the existing FIR/n16 cutoff selector. Conditioning stays fixed.
Checkpoint hashes are S1 `a48932ecdeb14432c7d2768589441383e75fe6e2c88ecc3084ba8f76422a873e`
and S2 `5c0c8fd745078cfe10c1eb5a3e8c69416415f050bf2602118287e7d537fbabb5`.
Local/full mean-rate response slopes are **0.625630/0.082057 (S1)** and
**0.000166/−0.000003 (S2)**. Because the selected trajectories are not perfectly
constant, the exact time-warp reference is `alpha * RPS(alpha * t)` at the same
cropped STFT timestamps; its local/full slopes are **0.970929/0.990648 (S1)**
and **1.014426/1.027471 (S2)**. Thus the flat S2 mean response is not explained
by the trajectories' natural time variation. This is a mean-response diagnostic,
not proof that every individual output ignores audio or a converged mechanism
verdict. FIR cutoff 80→10 changes subset MAE 0.701044→0.673373 (S1) and
0.997105→0.998449 (S2); neither exhibits half-rate outputs on this prior-supplied
subset. Arrays/summaries are in `results/jhtr/preliminary-cue-probes.json` and
`cue-physics-reference.json`. Training and monitor selection remain unchanged.

After this probe, S2's epoch-10 monitor improved to **0.689835**, but its
`best.ckpt` multipart R2 upload timed out. An R2 `best.ckpt` URI therefore need
not denote the latest local monitor-selected state. The live worker's e10 file
hash is `4ab55ff5abf6185a91f9037e5ed7635d35acb51c2118880d6fa903f581afbf27`.
Do not equate the earlier cue checkpoint with that newer state. The disposable
evaluation launchers now cache downloaded weights under `results/checkpoints`
so canonical and cue evaluations share identical, preserved checkpoint bytes.
The e10 file was recovered through SSH/rsync without pulling or terminating the
live Vast rental; its downloaded SHA-256 matches the worker. It is preserved at
`r2://ml-data/artifacts/jhtr-campaign/checkpoints/4ab55ff5abf6185a91f9037e5ed7635d35acb51c2118880d6fa903f581afbf27.ckpt`.
S2 resumed epoch 11 without a restart or training change. Both workers' original
Hydra `config.yaml` files were separately recovered: the trainer's ordinary
`results/**` output collection does not include Hydra's default `outputs/**`
directory. Their fully resolved mappings exactly equal the current campaign
configs. Hash-pinned R2 copies and local paths are recorded in
`results/jhtr/as-run-config-parity.json`; the selected snapshot receipt is
`results/jhtr/preliminary-s2-e10-checkpoint.json`.

### Complete fixed-set diagnostics on preliminary selected snapshots

`jhtr-kaggle-full-live-d096ed` completed all **18 cases × 256 examples × two
models**, with 32 original source groups per model, batch 32 and three complete
operator applications per case. Both checkpoint hashes match their cue-run
hashes; all 36 NPZ files have finite trajectories, repeats, targets and metrics,
matching fingerprints/groups, seven block states and exact saved timestamps.
These are full-set measurements, **not completed-training results**. S1 uses
`a48932ec…a873e`; S2 uses the newer `9464a39f…6941b0` snapshot.

| Measurement | S1 | S2 |
|---|---:|---:|
| Standard-corruption MAE-frame, input → output | 1.050398 → 0.566554 | 0.871646 → 0.686291 |
| Applicable signed-offset relative correction, range over all eight offsets | 21.0–74.3% | −4.1–6.2% |
| Oracle-input ordered MAE after one / two / three applications | 0.323 / 0.443 / 0.530 | 0.228 / 0.377 / 0.481 |
| Oracle displacement p95 after one / three applications | 0.987 / 1.758 | 0.715 / 1.493 |
| Local frequency-response slope, same evaluated weights | 0.625630 | 0.000840 |

S1's eight signed-offset mean improvements all have original-group bootstrap
intervals above zero. Thus **S1 learns genuine coarse correction**, not merely
a lower standard-corruption score. S2 does not approach the predeclared 20%
correction target on any signed offset. Oracle deterioration is also measured:
95% original-group bootstrap intervals for one-pass MAE increase are
**[0.263, 0.381] (S1)** and **[0.174, 0.278] (S2)**. Reapplication increases,
rather than removes, this drift. These full-set results do not establish the
separately certified observable-subset precision gate: its required source
metadata remains unavailable, and the evaluator correctly reports
`unestablished`, not a fabricated pass/fail on a favorable subset.

Neither model improves all-four recovery over its input guesses on the half,
double, all-collapse, missing-active or false-active probes. The false-active
probe retains a 100% false-active rate over its applicable stopped rotor-frames.
Duplicate-seed gains are not significant under the recorded group intervals;
the conditional symmetry limitation above remains structural. The untrained
audio-only locator diagnostic is not an audio-only training comparison.
Power/frozen/independent controls have not been trained, so no causal advantage
of phase products, re-reading or joint slots is claimed.

Artifact recovery required the native Kaggle SDK: the first `omnirun pull`
reported an unpublished tar, and the next returned an empty cached directory
despite a complete 558 MB kernel archive. Direct download recovered all data.
Only redundant nested collector copies were omitted from the verified evidence
bundle; no examples, cases or metrics were removed. The 292,783,724-byte bundle
contains both evaluated checkpoints, all NPZ/JSON/config files and verification
receipts, and is preserved at
`r2://ml-data/artifacts/jhtr-campaign/evaluations/2b322d0407309bde79b08e366d264704e9d63f314e4c9459d6d9164eac44cd06.tar.gz`.
Its filename is its SHA-256. Local receipts:
`results/jhtr/full-live-diagnostics.json` and `full-live-evidence.json`.

## Conclusion and failure strategy

Preliminary verdict: S1 is a useful coarse conditional corrector on the tested synthetic set; S2 shows little systematic audio-driven correction. Neither evaluated snapshot demonstrates the intended oracle-preserving precision refinement. Training remains incomplete, so this is not a converged architecture verdict. If a numerical gate fails, inspect interpolation, integration, normalization, masking, padding and gradients before training. If optimization fails, inspect actual processed exposure, gradient finiteness and final-only inherited loss; do not add rescue objectives/curricula. If full-set learning improves but conditional precision drifts, examine noisy local phase, interference, collisions, identity ambiguity and synthetic/real mismatch. Preserve the selected checkpoint and report failure; do not select an oracle-friendly epoch or silently widen the network.

A precision pass without acquisition supports a conditional refiner only. A power/frozen tie rejects the corresponding phase/re-reading claim and favors the simpler control. Downstream real/audio-only compute remains gated on these decisions. Scientific failure with saved evidence is a completed outcome, not unfinished implementation.
