# OT multi-pitch baseline (Björkman & Elvander, TSP 2026) on drone rotor speeds

**Status:** implemented + smoke-tested; adaptation probes running — **Date:** 2026-08-23

Reimplementation of arXiv 2508.02471 ("Inverse Harmonic Clustering for
Multi-Pitch Estimation: An Optimal Transport Approach", stochastic
estimator) as the classical multi-pitch baseline for the wrap-up paper
(§4.1). Code: `src/experiments/otmp_baseline/` (commit c6cb38a); every
implementation choice the paper leaves open is marked `[choice]` in source.

## Fidelity to the paper

Monte-Carlo self-test (their Sec VIII-A, 4 pitches 176/197/240/272 Hz,
5 dB, Table I parameters): **GER 28 %** vs the paper's 8–10 %; median
deviation of FOUND pitches 9.2 cents; 18.4 s/draw. The gap is a detection
failure of few-harmonic pitches (harmonic count uniform 3..10): the
group-sparsity term prices a new pitch column above the transport cost of
absorbing 3 partials into a neighbour. Probes: pitch-grid density matters
(1 Hz grid best of those tried; paper does not state G); more iterations
past ~800 hurt (sparsity keeps eroding weak components after the ranking
settles).

## Out-of-the-box drone result (paper Table II params, grids adapted only)

Frozen valid clips, channel 0, 0.5 s frames: PIT-MAE 38.3 rev/s (cruise
clip) / 45.0 (warm-up) — errors the size of the quantity. Diagnosis on a
clean cruise window: the mass lands on the low-frequency broadband floor
(52–56 Hz), fundamentals come out near half rate. Structural obstacles,
none of them implementation bugs: comb drift within the frame, twin rotor
pairs vs K=4, speech as a competing harmonic source in-band.

## Adaptation probes

(appended by the adaptation round)
