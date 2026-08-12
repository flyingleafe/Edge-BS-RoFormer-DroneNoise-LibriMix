# Amplitude-target training: fit the decomposition, not the waveform

**Date**: 2026-08-12 · **Branch**: `amp-target` · **Status**: built + trained;
numbers below. Continuation of `vk-decomposition.md` ("consequence: the
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

## Step 1 — the decomposition as a dataset

`decomp-frames-v1` (dload, pinned `d2603203d9a5`, 3 samples, 615 MiB), the
three decomposable real recordings:

| recording | drone | span | k_hi | tracks | labels |
|---|---|---|---|---|---|
| `free-flight_nosource_room1` | dregon | 63.9 s | 62 | 248 | refined sidecar |
| `FLY124` | michaels | 112.0 s | 57 | 228 | recalibrated telemetry |
| `FLY125` | michaels | 177.7 s | 57 | 228 | recalibrated telemetry |

Per recording, on ONE time origin (the decomposition span, re-anchored to 0):
`amp` `(mic, rotor, k, time)` at 100 Hz dense to k=80 (zero + `amp_valid=False`
above that recording's `k_hi`, so recordings batch together), `amp_valid`,
`residual` `(mic, time)` at 16 kHz, and `rps` `(rotor, time)` at 16 kHz — the
EXACT carrier the solve used, since an amplitude is only meaningful against the
trajectory it was demodulated with. Audio is deliberately NOT duplicated; the
meta records the source span so the parent frames dataset supplies it.

Michael's decomposition ran as `vk-decompose-michaels-be071d` (uni-cpu, k_hi 57,
228 tracks, 5.3 GB/worker, ~11 min); the artifacts of all three sit at
`r2://ml-data/artifacts/vk-decompose/<recording>/`. The derivation
(`derivations.generate_decomp_frames`, `adopt_only`) joins those artifacts to
the pinned parent frames and was materialized once.

## Step 2 — amplitude-only propagation

`PositionalHarmonicNoiseGen.amp_stats` (`models/generative/`): the emitter's
control curves, read BEFORE the oscillator bank, propagated in amplitude only.

- **Coherent tracks**: `amp[mic, rotor, k, t] = harm_amps[rotor, k, t] ·
  g[mic, rotor]`, `g = ref_distance / dist` — `propagate`'s own 1/r weight
  (`amplitude_gains`, tested against it), with **no delay** (sub-ms delays do
  not move a 100 Hz envelope) and **no cross-rotor sum** (the decomposition
  already separated the rotors — which is exactly why the interference problem
  vanishes).
- **No RPS jitter, structurally**: the OU perturbation is applied to the
  carrier AFTER the amplitude network runs, so this path sees the clean
  amplitudes. That is intended — the target envelopes already carry the
  recording's real linewidth; jitter stays a rendering-time device.
- **Broadband**: per-mic power sum `Σ_r g² · A_r(f,t)²` over the branch's own
  60-band uniform grid.
- **Calibration** (per drone, name-keyed, on the codebook wrapper): a global
  log-gain, a per-microphone log-gain, a separate power-domain constant for the
  broadband branch, and — see below — a per-mic per-band static floor. All are
  applied to EVERY prediction path, so a calibrated model also renders at the
  recording's level instead of needing a post-hoc scale.

**Broadband design, revised by the residual-attribution campaign.** Per-rotor
attribution of the residual is refuted on both arrays; the residual is 70-90 %
per-mic incoherent and DREGON's per-mic residual energy spread is 8.54 dB
against the 1.59 dB four equal 1/r sources can produce
(`residual-attribution.md`). So the broadband branch does not share the
coherent per-mic gain (that would corrupt the array-calibration readout while
failing anyway) and gets an explicit static per-mic per-band floor. The
rotor-propagated term is left to explain only what a propagated source can.

## Step 3 — objective

`losses.AmplitudeTargetLoss`:

    L = mean_valid |log(amp_pred + eps) − log(amp_tgt + eps)|
      + w · mean |log(psd_pred + eps_p) − log(psd_tgt + eps_p)|

- cell weight 1, no k-weighting: the targets already ISOLATE each harmonic, so
  a k=60 line is no longer a hundredth of a band's energy;
- `eps = 1.6e-5` — the decomposition's own floor (1st percentile of valid
  amplitudes; the three recordings give 1.58/1.70/2.20e-5). A v2 could debias
  each track by its noise-equivalent bandwidth (`Envelopes.bw_track` is
  published for exactly that) instead of a fixed floor;
- `w = 0.5`, measured: the raw terms are 7.14 / 7.93 log-units at
  initialization (both dominated by the absolute unit constant), and 1.08 /
  2.54 once one global scalar absorbs that constant, so 0.5 brings the
  broadband term to 1.27 — within 18 % of the amplitude term;
- the 100 Hz targets are resampled onto the emitter's 31.25 Hz control grid; a
  coarse frame counts as valid only if every target frame under it is.

Plumbing is one flag: `task_params.amplitude` widens
`tasks.task.noise_generation` to `{amp_pred, noise_psd}` and makes
`NoiseGenerationCodec` call `amp_stats` instead of rendering — no waveform is
produced at all, which makes a training step ~25 ms on CPU.

## Step 4 — arms and results

| arm | conditioning | objective | data |
|---|---|---|---|
| `gen_a1_amp` | per-drone code | amplitude target | `decomp-frames-v1` |
| `gen_a2_amp_perrotor` | + per-rotor δz | amplitude target | `decomp-frames-v1` |
| `gen_m1_refined` | per-drone code | MSSTFT on audio | DREGON+Michael's stream |
| `gen_m2_refined_perrotor` | + per-rotor δz | MSSTFT on audio | DREGON+Michael's stream |

Both amplitude arms monitor `val_loss` (there is no rendered realization to
score, and a comb-blind scalar is a proven lottery as a selector);
`checkpoint_every=1`, and the final pick is comb-aware and offline, exactly as
the perrotor-dynamics campaign requires.

RESULTS_PLACEHOLDER

## Reading the comb table

`scripts/eval_gen_comb_real.py` renders each arm through the ordinary audio
codec (the `*_render` twin configs exist for exactly this: same parameters, so
an amplitude-trained checkpoint loads strictly) and reads per-k
peak-to-floor and along-track fidelity on the real DREGON recording. Anchors:
6.02 dB is what a perfect but stochastic model scores on `dLogMag`, and −0.78 dB
is the estimator's null for `PTF` (no measurable tooth). Both readings are
self-referenced/paired caveated as in `generator-refined-labels.md`.

## Caveats

- The comb readout covers DREGON only, and its chunks overlap training audio —
  a comb-shape comparison between arms, not a generalization measurement.
- The amplitude objective is scored on the SAME decomposition it is trained on;
  the decomposition's own bias (bandwidth-dependent floor absorption in weak
  bands, `vk-decomposition.md` finding 4) is therefore shared by target and
  metric.
- The per-mic floor is free enough to absorb rps-dependent broadband; the
  broadband branch is not the payload here and is only kept constrained.
