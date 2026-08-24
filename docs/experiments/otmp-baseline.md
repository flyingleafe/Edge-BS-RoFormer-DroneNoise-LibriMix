# OT multi-pitch baseline (Björkman & Elvander, TSP 2026) on drone rotor speeds

**Status:** implemented, smoke-tested, adapted — **Date:** 2026-08-23

**Verdict:** the adaptation is worth 1.5x (cruise PIT-MAE 21.1 -> 14.4 rev/s
with `adapted_drone_config()`) and the method still does not work on this
signal. Quote it as a classical-baseline floor. Do not schedule a
full-protocol run. See "Adaptation probes" below.

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

A bounded study, 2026-08-23, to give the baseline its fair best shot before the
wrap-up paper quotes it. Probe script and every unit JSON:
`.../scratchpad/otmp_adapt/` (`pick_clips.py`, `probe.py`, `inspect_frame.py`,
harness `utils.gridrun`). Package changes: `OTMPConfig.whiten_hz` /
`whiten_n_fft` with `whiten_signal` (`[choice] 8`, off in every preset) and the
new preset `adapted_drone_config()`. `drone_config()` is unchanged.

### Protocol

Four clips of the frozen `DREGON-LM-V4-michaels-valid-full` split, channel 0:

| Probe clip | Sample | Recording | Regime | Targets |
|---|---|---|---|---|
| `cruise_dregon` | `sample_00002` | `free-flight_speech-low_room1` | cruise | 74-87 rev/s, speech in the mixture |
| `cruise_fly124` | `sample_00027` | `michaels_FLY124` | cruise | 73-96 rev/s |
| `warmup` | `sample_00023` | `michaels_FLY124` | warm-up | steady 31-41 rev/s |
| `ground` | `sample_00000` | `free-flight_speech-low_room1` | rotors stopped | 0 |

Screening uses 6 analysis windows per clip, at the same six instants of the clip
for every configuration, so the frame-length lever is not confounded with window
placement. The score is the per-frame Hungarian (PIT) MAE against the
window-averaged `frame["rps"]`, the convention of
`results/m3cur_regime_probe/regime_probe.py`. A report of `K < 4` pitches fills
the four rotor slots first: `K = 2` duplicates each pitch (the twin-pair
reading), `K = 3` repeats the strongest.

The `ground` column measures nothing but the false-positive rate. The pitch grid
starts at 30 rev/s, so the estimator cannot report a stopped rotor and no
configuration can win that column. It is not part of any decision below. The
warm-up clip is `sample_00023`, not the smoke test's `sample_00008`: that one is
a take-off ramp that is half rotors-stopped, which only duplicates `ground`.

These clips are not the smoke test's, so the `base` row below (21.1 rev/s) is
not the 38.3 / 45.0 of the section above. Both are the same estimator with the
same parameters. The smoke test scored a speed ramp (`sample_00001`) and a
half-stopped take-off, whose windows the estimator cannot reach from a 30 rev/s
pitch grid. `base` here is the reference for every row of this study, and the
comparison to make is with it.

### Levers, one at a time (6 windows per clip, PIT-MAE in rev/s)

| Configuration | Change from `drone_config()` | cruise DREGON | cruise FLY124 | warm-up | ground | **cruise mean** |
|---|---|---|---|---|---|---|
| `base` | — | 21.68 | 21.47 | 28.00 | 78.12 | **21.58** |
| `floor120` | band 120-1200 Hz | 24.38 | 21.41 | 14.40 | 79.16 | 22.89 |
| `floor150` | band 150-1200 Hz | 14.46 | 18.34 | 14.89 | 77.35 | **16.40** |
| `floor200` | band 200-1200 Hz | 30.15 | 15.33 | 13.50 | 79.22 | 22.74 |
| `whiten` | `whiten_hz = 150` | 19.03 | 17.63 | 28.19 | 83.81 | **18.33** |
| `frame025` | 0.25 s frame | 29.03 | 25.25 | 25.86 | 74.21 | 27.14 |
| `frame100` | 1.0 s frame | 15.74 | 19.46 | 23.42 | 81.40 | **17.60** |
| `k2` | `n_pitches = 2` | 22.17 | 27.73 | 26.91 | 81.40 | 24.95 |
| `k3` | `n_pitches = 3` | 22.03 | 21.58 | 28.68 | 83.09 | 21.80 |

Four readings:

1. **The band floor is the strongest single lever, and 150 Hz is its optimum.**
   120 Hz is too low to clear the rumble and 200 Hz starts to remove the
   evidence: it is the best of the four on the warm-up clip, whose fundamental
   is 36 rev/s, and the worst of the four on the DREGON cruise clip. The
   warm-up clip prefers any floor at all (28.0 down to 13.5-14.9).
2. **Whitening helps, and for the same reason.** It buys about the same on the
   cruise clips as a 120 Hz floor and nothing on the warm-up clip.
3. **Longer frames win, against the drift argument.** 0.25 s is much worse than
   0.5 s and 1.0 s is better than both. Comb drift inside the window is real,
   but the frequency resolution of a short window costs more than the drift.
4. **Fewer pitches lose.** This refutes the twin-pair reading. The reason is
   visible in the reports: the *strongest-mass* candidate is itself often a
   sub-octave, so cutting the report to the top 2 keeps the junk and discards
   a correct speed. Example, `cruise_dregon`, targets 74.2/74.6/84.3/84.5:
   `K = 4` reports 38.8/41.8/51.2/74.2 and `K = 2` reports 38.8/41.8.

### Best combination, and the regularization touch-up

| Configuration | Change from `drone_config()` | cruise DREGON | cruise FLY124 | warm-up | ground | **cruise mean** |
|---|---|---|---|---|---|---|
| `f150_t100` | 150 Hz floor + 1.0 s frame | 11.40 | 17.68 | 12.34 | 83.62 | **14.54** |
| `f150_wh` | 150 Hz floor + whitening | 20.91 | 14.18 | 23.33 | 78.81 | 17.55 |
| `f150_t100_wh` | all three | 24.12 | 12.71 | 21.63 | 69.33 | 18.41 |
| `reg_b_lo` | `f150_t100`, `beta = 3.4e-3` | 21.13 | 17.38 | 16.46 | 80.57 | 19.25 |
| `reg_b_hi` | `f150_t100`, `beta = 3.4e-1` | 17.00 | 12.15 | 16.39 | 87.23 | 14.58 |
| `reg_ze_lo` | `f150_t100`, `zeta = eta = 1e-2` | 15.41 | 13.40 | 19.81 | 81.54 | 14.40 |
| `reg_ze_hi` | `f150_t100`, `zeta = eta = 1e0` | 9.53 | 18.72 | 13.77 | 74.06 | **14.12** |
| `reg_b_hi_ze_hi` | `f150_t100`, both up | 9.63 | 18.73 | 14.39 | 73.36 | 14.18 |

Whitening does not compose with the raised floor. The two levers do the same
work — they take the prize off the broadband rumble — and the floor does it
better, so adding whitening on top only removes evidence.

The regularization touch-up buys nothing. Five of the eight combinations land
between 14.1 and 14.6, which is one spread of the same result over 12 cruise
frames, and the two that move (`reg_ze_hi`, `reg_b_hi_ze_hi`) trade DREGON
cruise against FLY124 cruise rather than winning both.

### Finalist, re-scored on every window

Every non-overlapping window of all four clips, `base` at 16 windows per clip
and the 1.0 s configurations at 8:

| Configuration | cruise DREGON | cruise FLY124 | warm-up | ground | **cruise mean** |
|---|---|---|---|---|---|
| `base` | 21.01 | 21.16 | 25.76 | 77.20 | **21.09** |
| `f150_t100` | 13.27 | 15.56 | 15.71 | 75.21 | **14.42** |
| `reg_ze_hi` | 12.48 | 19.48 | 14.43 | 74.25 | 15.98 |

`reg_ze_hi`'s screening win does not survive the extra frames, so the
recommendation keeps every one of the paper's regularization parameters:

```python
from experiments.otmp_baseline import adapted_drone_config
cfg = adapted_drone_config()          # == drone_config(freq_lo_hz=150.0, frame_len=16000)
```

Cost is 57 s per frame under a 12-way pool, about 12 s per frame alone.

### Verdict

The adaptation is worth 1.5x — cruise PIT-MAE 21.1 down to 14.4 rev/s, warm-up
25.8 down to 15.7 — and it is not enough. Against targets of 74-96 rev/s at
cruise the finalist still misses by a fifth of the quantity. The learned RPS
predictors of this project score a few rev/s of PIT error on the same split,
and the blind Vold-Kalman tracker scores about 1 rev/s at cruise.

What the adaptation actually fixed is measurable in the reported slots. Of the
four pitches each cruise frame reports, `base` puts 43 % within 3 % of a true
rotor speed, 22 % on a sub-octave of one, and 33 % elsewhere. The finalist puts
60 % on a rotor, 9 % on a sub-octave, and 29 % elsewhere. The band floor
therefore did what the eq-(18) argument predicts: it removed most of the
octave errors. It did not make the estimator find the *set* of speeds. The
finalist covers only about 1.4 of the distinct true speeds per frame, so its
remaining error is not a precision error but a **coverage** error — one or two
rotors found, the other slots filled with candidates that are neither a rotor
nor a sub-octave of one.

Coverage is the same wall the blind comb-search work reports
(`docs/experiments/beat-vk.md`), and here it is structural for the estimator:
the group-sparsity term prices a new pitch column against the transport cost of
absorbing partials into a neighbouring column, and four rotors within a few
percent of each other are exactly the case where absorption is cheaper. The
Monte-Carlo self-test shows the same failure at 28 % GER against the paper's
8-10 %.

**A full-protocol run is not worth scheduling as a competitive baseline.** At
14 rev/s the number carries no ranking information — every learned model and
the classical VK stack beat it by more than 2x, so a wider evaluation cannot
change any conclusion the paper draws. It *is* worth quoting the finalist as a
**classical-baseline floor** in §4.1, with the 4-clip numbers above and the
coverage diagnosis, because that is the honest statement of what an
off-the-shelf multi-pitch estimator gives on this signal. If a full-protocol
number is wanted anyway for completeness, the cost is about 12 s per frame per
core, i.e. roughly 3 CPU-hours for the whole valid-full split at 1.0 s frames,
and it needs no GPU.

## Full-protocol run (2026-08-23, `otmp-protocol-af4da1`)

The adapted finalist over the whole frozen valid
(`dload:DREGON-LM-V4-michaels-valid-full`, 37 clips x 8 channels = 296 units,
`experiments.otmp_baseline.protocol_eval`, per-frame Hungarian PIT vs the
mean telemetry target per 1 s frame, regimes as in the m3cur regime probe).
All 296 units OK; ~10 h wall at 8 CPU workers. Summary
(`results/otmp_protocol/summary.json`):

| pool | MAE (rev/s) | MSE | n frames |
|---|---|---|---|
| flight | 16.29 | 544.3 | 1744 |
| low | 28.27 | 1525.9 | 328 |
| zero | 68.77 | 5437.7 | 296 |
| all | 24.51 | 1291.9 | 2368 |

The full protocol confirms the 4-clip study: cruise ~16 rev/s (study said
14.4 on its subset), and the estimator has no silence model at all (zero-rps
frames score 69 — it always reports four pitches). Quoted in the paper
Table "multipitch" as cruise RMSE sqrt(544.3) = 23.3 rev/s, the
training-free floor row. Ranking unchanged: every learned model and every
tracking method is ~10x better.
