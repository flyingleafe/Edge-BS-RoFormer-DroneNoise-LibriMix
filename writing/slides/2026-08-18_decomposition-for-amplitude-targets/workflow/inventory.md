# Work inventory since last slides

- generated: 2026-08-17T23:47:43+01:00
- boundary artifact: writing/slides/2026-08-04_rps-tracking-results-and-paper-plan
- boundary commit: d6fa3b0 2026-08-11 slides: reassigned high-res spectrogram for the h70 overlays
- HEAD: 84e1e2c 2026-08-17 labels: FLY124/FLY125 refined sidecars + the v4 unified-model design doc

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
84e1e2c labels: FLY124/FLY125 refined sidecars + the v4 unified-model design doc
75b4ee5 paper: the model as a joint distribution, and the derivation to J
e1edf08 labels: refined sidecars for the in_flight_source recordings
9c72c26 paper: preliminaries, full citations, and the experiment inventory
a8c51d6 drivers: full-band stochastic regions, and the refiner opened to every rig
134510a stochastic split: interpolate the floor in time (kill the block seams)
00d37a7 rescore: the telemetry-initialized refined hypothesis
d5f8403 measure: the adaptive floor and the Lorentzian-shaped H
650237e tracking: the objective knows about the stochastic comb
3394752 decomposition: the regime-3 search region is three linewidths
5fdde0c decomposition: regime 3 subtracts to the floor, per bin
728038f tracking: the objective marginalizes the envelopes, it does not only profile them
23dda1e decomposition: regime 3 — the stochastic comb channel
6a60270 campaign: step-5 trajectories packed for the joint rescore
c29d16c tracking: the joint decomposition reports its MAP objective
dc3d037 paper: v4 structure-first draft — joint decomposition with a built-in measure
dc39176 notebooks: rate errors are OU, not a random walk — anchored lines, diffusing phase
d822f91 notebooks: the hand-reimplementation tutorial for the joint decomposition
82d9dc0 scripts: the driver solves with the config it already built
56ae096 docs: the decomposition primitive inventory
b3153c4 tracking: one window geometry, and one Welch
c7132fe tests: the joint composition, the two combinators and the seam guard
a945477 tracking: the joint decomposition as three stages and two combinators
00fd3b5 tests: the v3b joint regression guard, and one fixture builder
ab0524b docs: v3b verdicts — half-order comb retracted (instrument audit #4), stitch metric fixed
bbab5cc docs: v3b re-run results — no half-order peak on either rig
61fc1df docs: v3 design record — the order-cell detrend, the stitch edge, the new knobs
5526a06 tracking: detrend the order-cell instrument, and fix the stitch edge + r2 carrier
5b7411d docs: record v3 in the tracking map and the decomposition experiment log
97c56c3 tests+docs: v3 acceptance tests, the v2 guard, and the design record
8d7589a tracking: the v3 JOINT decomposition — whitened solve, phase split, smooth floor
bab4284 docs: status header for the 2026-08-14 revision
f0aadd9 docs: the ranked best and worst examples of the campaign
dc00c4b docs: room1 measured telemetry gives PIT-MAE 1.0 rev/s; the per-rotor lever does not transfer
5075699 blind-corpus: DREGON room1 units, scored against measured telemetry
810169b docs: the blade-pass mechanism behind every doubling, and the SPCUP one-source rule
0f2bc81 docs: the acceptance gate depends on the arm, and SPCUP is not weak
c0b75a4 blind-corpus: SPCUP19 static units + per-run reports
128652e conf: render twins for the C-series arms
f7cccde docs: the C-series design — combined rigs, amplitude-only propagation
cd9788d tests: the propagation head, the combined pool, and a real v2 round trip
e422b73 docs: bench throttle ladder validates the blind annotator; corpus plan revised
2a71305 blind-corpus: bench / KAIST / room2-per-rotor / SPCUP single-rotor unit artifacts
f01a583 blind-corpus: a recording id can be a path, so the uid must be a slug
f5b62b4 model/conf: amplitude-only propagation head with a learnable per-mic EQ
90f40ed blind-corpus report: the calibrated verdict as a column, and rank inside the gate
0aca8a0 residual attribution: band-power second pass — same verdict, with a mechanism
e965f42 probe: coherence structure of v2 envelopes; withdraw the v2 slot-contrast verification
62e4820 blind-corpus: per-rotor octave instrument + regex/limit recording selection
3cf3870 docs: v2 amplitude arms trained — perrotor 0.818 vs codebook 0.856 val
32de294 docs: v2 targets pinned, scientific amplitude arms launched
67dff82 dload: pin decomp-frames-v2 @ d42834f57d07 (remote derive, job derive-decomp-v2b)
a96db5e derivations: decomp join accepts an integer-multiple decomposition rate
c4d858b conf: gen_a1_amp_v2 / gen_a2_amp_perrotor_v2 — amplitude arms on the v2 targets
1caeb48 docs: Michael's full-recording v2 verification — comb removed above k10
adc5097 docs: OOM misdiagnosis corrected — non-PD banded system, PD-repair fix
d3bd467 tracking: PD-repair retries before the splu fallback
0ff7ec4 docs: room2 calibration — fvk_ratio_double 1.065 separates correct/halved/ramp 17/17
1643689 blind-corpus: re-reference absolute telemetry stamps to the audio start
83625c4 docs: motor bench calibrates the octave gates — half-margin<0 means halve
eb9fd86 blind-corpus: reference join uses PUBLISHED_RPS_KEYS fallback
5c3fb2e docs: bench needs the seedvk arm — vit2dsp is structurally 4-track
3ffbc26 blind-corpus: seedvk arm — single-rotor bench recipe (seed + one VK pass)
6b9f792 docs: full-recording v2 verification — residual comb contrast ~0 in all bands
229ff11 docs: blind-avq-8ch verdict — 8 channels do not resolve the AVQ octave ambiguity
e446ecc docs: name the reported checkpoints (three same-named runs share the R2 prefix)
a4ee812 metrics: an empty MetricSuite is legal, and the test says why
dfc88e3 docs: amplitude-target results (v1 targets) + AGENTS entries
357287c tracking: v2 decomposition — linewidth-matched bandwidth schedule + foreign-tone report
27d04a8 loss: barrier on the harmonics the decomposition never solved
30894f9 docs: amplitude-target campaign doc (v1-target caveat, design, arms)
851b7a6 data/exp: version-parameterized decomp specs + middle held-out block
cd7da9e docs: blind-corpus addendum — alias-penalty lever confirmed, bridge bug located
1c8f078 docs: blind-corpus pilot — AVQ verdicts, mono pseudo-label retraction path, corpus plan
11365db blind-corpus: local validation artifacts + corpus triage
c59a3ba amp-target: loss, codec/task mode, chunk loader, configs, tests
d36a467 blind-corpus: cheap spectrogram triage, no ladder
294974c residual attribution: per-rotor split refuted by geometry null controls
6eafcf3 tracking: promote decomposition + windowed-refine primitives into the Stage framework
da77037 blind-corpus report: make the overlay legible
5d6ff86 blind-corpus: bound the scoring geometry independently of the ladder window
fa93e49 blind-corpus: make the grid worker a module-level callable
7ecd89a data: decomp-frames-v1 derivation (VK envelopes as frames) + amp_stats on the generator
736c82f blind-corpus: label-free blind annotation driver + report
508ffcb vk_decompose: Michael's dataset support (profile follows the spec)
3b9885c docs: VK decomposition campaign — exact split, independence verdict, prior limits
da59c9f tracking: VK maximum-likelihood decomposition instrument
a8787fa lab+docs: full-dataset refined-label arms, comb-aware picks
1331c9e conf: gen_m1_refined / gen_m2_refined_perrotor — full-dataset refined-label arms
3b9c602 docs: seed-43 pair closes the perrotor lottery quantification
4d20872 docs: instrument audit correction — below-null PTF was floor leakage
1a2f8db docs: per-rotor embedding paradox resolved — erosion + selection lottery
8edd6b8 lab: the five label-A/B generators in the comparison notebook
256f3af docs: generator refined-labels A/B on real DREGON — verdict
0f5749f labels: harden the sidecar override at motor stop boundaries
0cbfa6f labels: refined DREGON telemetry sidecar (L-BFGS on F_VK, per-rotor accepted)
79428ca refine_dregon_rps: per-rotor acceptance + flight-frame scale statistics
1d996d6 eval: real-audio per-band comb readout for trained noise generators
213cccc refine_dregon_rps: warm the recording cache before the pool forks
7665afa labels: L-BFGS-refined DREGON telemetry — refiner CLI, loader override, gen_r* arms
9db3482 docs(tutorial): load windows through dload, not only the prep cache
00180de docs: hands-on notebook tutorial for fitness / L-BFGS / methods / step-by-step
```

## Experiment configs (conf/experiment/)

```
  A	conf/experiment/gen_a1_amp.yaml
  A	conf/experiment/gen_a1_amp_render.yaml
  A	conf/experiment/gen_a1_amp_v2.yaml
  A	conf/experiment/gen_a2_amp_perrotor.yaml
  A	conf/experiment/gen_a2_amp_perrotor_render.yaml
  A	conf/experiment/gen_a2_amp_perrotor_v2.yaml
  A	conf/experiment/gen_c1_amp_combined.yaml
  A	conf/experiment/gen_c1_amp_combined_render.yaml
  A	conf/experiment/gen_c2_amp_combined_perrotor.yaml
  A	conf/experiment/gen_c2_amp_combined_perrotor_render.yaml
  A	conf/experiment/gen_m1_refined.md
  A	conf/experiment/gen_m1_refined.yaml
  A	conf/experiment/gen_m2_refined_perrotor.md
  A	conf/experiment/gen_m2_refined_perrotor.yaml
  A	conf/experiment/gen_r1_orig.md
  A	conf/experiment/gen_r1_orig.yaml
  A	conf/experiment/gen_r1_refined.md
  A	conf/experiment/gen_r1_refined.yaml
  A	conf/experiment/gen_r1_scaled.md
  A	conf/experiment/gen_r1_scaled.yaml
  A	conf/experiment/gen_r2_orig_perrotor.md
  A	conf/experiment/gen_r2_orig_perrotor.yaml
  A	conf/experiment/gen_r2_refined_perrotor.md
  A	conf/experiment/gen_r2_refined_perrotor.yaml
```

## Docs (docs/) — excerpts for added files

- MODIFIED: docs/experiments/AGENTS.md
### ADDED: docs/experiments/amplitude-target-training.md
```
# Amplitude-target training: fit the decomposition, not the waveform

**Date**: 2026-08-12 · **Branch**: `amp-target` · **Status**: path built and
trained end to end on **v1 targets**, which makes the numbers below a
PLUMBING VALIDATION, not the scientific comparison — see "Target version". Continuation of `vk-decomposition.md` ("consequence: the
amplitude-target training path") and the erosion verdict of
`generator-perrotor-dynamics.md`.

## The problem this objective removes

The audio-domain multi-scale STFT loss cannot defend mid/high-k harmonics of a
REAL recording, for a measured reason: the lines decohere inside its own
analysis windows (shaft wander σ≈0.6 rev/s → ≈0.24·k rad per 2048-sample
window), each window's band magnitude fluctuates around a low median, and a
log-L1 term fits that median — a persistent downward gradient on every steady
rendered line above k≈25 (`generator-perrotor-dynamics.md` finding 6). The
refined-labels campaign closed the same loop from the data side: order-averaged
tooth contrast along refined tracks is 6.76 / 1.36 / 0.13 / 0.01 dB in the four
bands, so **no arm receives a training signal for sharp teeth above k≈25**
(`generator-refined-labels.md` § CORRECTION).

The Vold-Kalman decomposition removes phase from the problem instead of
fighting it. It is comb-COHERENT demodulation, so it yields one amplitude
ENVELOPE per (rotor, harmonic, microphone) at 100 Hz plus a broadband residual,
and the split is exact by construction. Fitting those envelopes never compares
two realizations of a decohering line.

## Target version (read before the numbers)

The v1 decomposition solved every track at a FLAT 1 Hz envelope bandwidth. Real
