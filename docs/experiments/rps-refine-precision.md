# RPS refinement precision campaign

**Goal (user, 2026-07-30):** make the iterative RPS refinement chain actually
precise — near-perfect on synthetic free-flight (phases locked, exact GT),
much better on Michael's (FLY124). The slider-demo observations that started
it, each confirmed quantitatively on the three 16 s trace windows
(scratchpad `explainer-traces/trace_*.json`, pipeline = fullrange-v2 blind
arm → pi_kalman):

## Diagnosis (2026-07-30)

Per-stage PIT-MAE and per-round trajectory movement (`|Δ|` mean rev/s per
rotor per round):

| window | after ladder | capture (4 rds) | refine (5 rds) | pi_kalman | final MAE |
|---|---|---|---|---|---|
| dregon_ramp | 3.230 | Δ≈0.01–0.05/rd | **Δ≈0.002/rd** | Δ≈0.1 | 3.214 |
| fly124_cruise | 1.195 | Δ≈0.02–0.06/rd | **Δ≈0.003/rd** | Δ≈0.1–0.9 | 1.140 |
| synthetic | 1.266 | Δ≈0.03–0.05/rd | **Δ≈0.004/rd** | Δ≈0.2–0.5 | 0.996 |

1. **Coarse coupling.** The coarse Viterbi solves ONE common trajectory
   c(t); all four tracks are c(t)+offset (identical per-rotor |Δ| at the
   viterbi/gates rows). The ladder splits pair means only, never shapes. On
   dregon_ramp the damage is visible as pred-std ≈ 26 for ALL rotors while
   gt-std is 27/23.5/27/23 — the shallower pair carries ~3.6 MAE it cannot
   shed. Shape-corr is 0.975+ (the common shape is right); the per-rotor
   *deviation* from it is unrepresented.
2. **VK refine is dead.** 5 rounds × ~0.003 rev/s = no-op (fly124 even
   degrades 1.170→1.171). Suspects inside `vk_tracking._freq_update`:
   update_gate=30 periodogram gate, Wiener shrink snr/(1+snr), the
   pentadiagonal smoothing solve, or simply that phase slopes are already
   ~0 at its narrow final bands. To be instrumented.
3. **pi_kalman under-used.** It is the only late stage that moves (one call,
   n_iter=3) and gives the largest late improvements (synthetic
   1.198→0.996). Its capture radius is `band_hz/k` (6 Hz half-band → ±0.75
   rev/s at k=8), so post-ladder per-rotor errors of 1–4 rev/s leave most
   harmonics out-of-band — the likely reason it can't fully individuate
   shapes either. But it re-demodulates per outer iteration, so a
   wide→narrow band schedule + more iterations should progressively
   capture.

Final per-rotor decomposition (PIT-aligned): fly124 biases +0.66/+0.89/+0.84/−0.17
with pred-std ~0.7–1.3 vs gt-std 1.4–2.1 (under-tracking + bias); synthetic
biases ~0 but shape-corr only 0.42–0.80 — wiggle amplitude right, placement
wrong (capture/lag, not smoothing excess).

## WP1 verdicts (2026-07-30, instrumented runs; scratchpad `refine-diagnosis/report.md`)

**A — VK refine is dead because its phase slopes are already ≈0; nothing
downstream eats them.** Across 3 windows × 5 rounds × 4 rotors: the no-comb
gate never fired, max_step never clipped, shrink is a mild ×~2, the
smoothing solve attenuates only ×1.2–1.6. The raw Fisher-fused slope RMS is
0.003–0.008 rev/s against true errors of 0.55–1.5 rev/s — the 100–400×
shortfall exists *before* smoothing. Cause: at final bw 1.5 Hz, **0 of 25
harmonics** have |k·err| in-band anywhere (capture at k=6 needs
|err|<0.125 rev/s), while 61–93% sit inside the ±45 Hz demod brickwall —
which is what the gate tests. Refine has converged in its own narrow-band
terms; the residual is invisible to it. → The stage is dead weight at
current bands; drop or re-band it.

**B — pi_kalman's budget = capture × twin-gating interlock; lag refuted.**
corr(|err|, |dGT/dt|) ≈ 0 (over-smoothing is NOT the problem). The
interlock: post-ladder error is per-rotor *differential*; the low harmonics
(k≤4) whose band admits 0.5–1.5 rev/s errors (63–98% frames in-band) are
exactly where twin combs collide → discarded by the twin gate (fly124 tight
pair loses k=1–3 entirely; dregon twin-gated 80–84%, 2 of 4 rotors skipped
outright in iter 1); high k is resolvable but out-of-band (in-band frac
~0.2–0.3 at k=20–40). Errors are tail-dominated: worst decile of frames
carries 33–60% of total |err|; no-evidence frames carry 1.5–5× the error.
In evidence-covered frames MAE stops improving after iter 1 — and on fly124
evidence frames are *no better* (surviving collided low-k increments
endorse the biased track — the S4 sideband/twin-mixture mechanism).
→ Levers: wide→narrow band_hz schedule with re-demodulating iterations,
`pair_mode="joint"` so low-k differential evidence stops being discarded,
more outer iterations; NOT more smoothing or more of band-6 iterations.

## WP2 probe (2026-07-30, `refine-diagnosis/pk_probe.py` from post-ladder ckpts)

pi_kalman variants from the saved post-ladder state (pooled PIT-MAE):

| window | entry | n3 b6 (cur) | wide_joint n6 | wide_gate n6 | joint n3 | n8 sp4 |
|---|---|---|---|---|---|---|
| synthetic | 1.228 | 1.040 | **0.900** | 0.900 | 1.040 | 0.900 |
| fly124_cruise | 1.176 | 1.147 | **1.024** | 1.075 | 1.085 | 1.088 |
| dregon_ramp | 3.272 | 3.265 | 3.292 | 3.293 | 3.265 | 3.287 |

(wide = band_hz (12,9,6,4,2.5,2.5), k_caps (4,6,8,12,20,40), off_comb 16.)
Verdict: schedule helps ~14%, joint helps on fly124's tight pair, but
**kwargs alone plateau far above target**. Synthetic's rotors are
independent OU trajectories — the ladder's common-shape+offsets hypothesis
cannot represent them, and the residual per-rotor deviation is exactly the
error pi_kalman's collided low-k band cannot see. Decoupling is the
critical path for all three windows.

**WP3 design (two mechanisms, different scales):**
1. *Residual corridor Viterbi* (coarse decoupling, ~0.5 rev/s): coordinate
   descent — per-rotor DP in a ±6–8 rev/s corridor around the current
   track, scoring rotor r's comb on the whitened spec with the other
   rotors' current teeth masked; 2–3 sweeps. Repulsion by masking, not
   rigidity.
2. *Residual-audio pi_kalman* (fine decoupling): per rotor, subtract the
   siblings' VK reconstruction (`vk_envelopes`+`vk_reconstruct` on tracks
   ≠ r) from the audio, run pi_kalman on the residual for rotor r —
   removes the twin-collision interlock entirely, unlocking low-k
   evidence at wide bands.

## WP3 probe (2026-07-30, `refine-diagnosis/decouple_probe.py`)

M1 = residual corridor Viterbi (per-rotor offset DP ±8, 0.25 steps, sibling
teeth masked, surface-quality gate); M2 = residual-audio pi_kalman (sibling
VK reconstruction subtracted). Pooled PIT-MAE:

| chain | synthetic | fly124_cruise | dregon_ramp |
|---|---|---|---|
| entry (post-ladder) | 1.228 | 1.176 | 3.272 |
| WP2 plateau | 0.900 | 1.024 | 3.265 |
| A: M1×3 | 0.835 | 1.130 | no-op (all gated) |
| B: M1→wide_joint pk | 0.772 | 1.065 | 3.292 |
| C: M2 full4 ×2 | 0.772 | **1.027** | recon diverges |
| D: M1→M2 | **0.709** | swaps identities | ≡ C |

- **Synthetic: plateau broken by decoupling** (0.900→0.709, entry −42%),
  both mechanisms contribute. Remaining error = capture-tail frames +
  corridor quantization (0.25 rev/s, 7.8 Hz bins) + M2 sweep-2
  oscillation.
- **fly124: one rotor is comb-invisible** in the coarse whitened spec
  (masked surface quality 0.099 vs 0.28–0.34 siblings) — ungated its DP
  runs away onto a sibling comb; and the tight pair latches onto the
  imperfectly-subtracted twin without the geometric gate. Sibling recon
  quality healthy otherwise (resid/orig 0.54–0.91).
- **dregon_ramp: structurally blocked at this layer** — per-rotor surfaces
  uniformly weak (ramp smear), and sibling VK reconstruction diverges
  (resid/orig 2.6–78; tight twins 0.5/0.8 rev/s split + ~4 rev/s track
  errors → the documented cancelling mode). The ramp needs the upstream
  shape fixed (chirp-matched per-rotor scoring is the future lever:
  during the ramp the local slope is KNOWN from c'(t), so per-rotor teeth
  can be scored by matched summation along the chirp direction).

**Round-2 plan (goal priorities: synthetic → fly124; dregon deferred):**
(a) port M1/M2 into `scripts/rps_refine_lab.py` as chain stages;
(b) alternating M1/M2 rounds with per-round corridor shrink (±4→±2, step
0.1, interpolated sub-bin scoring) + a final narrow-band gate-mode
pi_kalman polish (band (2.5,2,1.5), k→40), convergence-stopped;
(c) fly124: residual re-seed of the comb-invisible rotor — subtract the 3
well-tracked rotors' reconstruction, wide comb re-scan of the residual at
8192-FFT for the weak rotor's true base, then corridor-track it; per-mic
surface selection as fallback.

## WP4 — the synthetic error budget, measured (2026-07-30)

Round-2's `alt_loop` reproduced the WP3 probe at round 1 (0.711) but **degraded
in rounds 2-4** (0.717 → 0.754). Chasing that produced a full accounting of
what actually limits precision. Probes: `refine-diagnosis/{floor,floor2,
iterate,denoise,snr,err_spectrum,ess,alias_check,bandlimited_synth,
converge_clean,sibsub_clean}_probe.py`.

**1. The estimator's floor is 0.05, not 0.4.** Holding an exact-GT init costs
only 0.055 rev/s. Constant offsets up to ±1 rev/s are removed to ~0.1; ±2 is
outside capture. So precision is not intrinsically limited.

**2. Iteration is HARMFUL, not helpful.** Every schedule peaks after ONE
application and then degrades monotonically (0.167 → 0.268 over 5 repeats):
each pass injects ~0.05 rev/s of fresh estimation noise that accumulates
additively in the track. *This is why alt_loop's rounds 2-4 lost ground.*
Design rule: one M1 sweep set + ONE M2 pass, no convergence loop, no extra
polish (the polish pass also only adds noise).

**3. The residual is SNR-independent → not measurement noise.** From a
shape-deficient init: 0 dB 0.413, 10 dB 0.334, 20 dB 0.337, 40 dB 0.336. At
essentially noiseless data the estimator still lands 0.3 from truth.

**4. It is a BANDWIDTH limit + a benchmark artifact.** Error power per band
(40 dB): below 0.5 Hz the estimator recovers 99.9% of GT's power (err/GT power
ratio 0.001-0.006), 2-4 Hz ~0.2, above 8 Hz ~1.0 (nothing recovered). GT holds
70.7% of its power below 0.5 Hz but the residual is 47% in 8-16 Hz — the
frame-grid Nyquist edge. Two causes, neither the estimator's fault: the
`rps_synthesis` OU drive is white to the 250 Hz generation rate (a real rotor's
inertia cannot follow that), and GT is point-sampled onto the 31.25 Hz frame
grid with `np.interp` (no anti-alias), folding it all back in. `ess` was
exonerated (forcing ess=1.0 changes nothing).
*Reality check (`shaft_bw.py`): DREGON's 1 kHz `motors_measured` has only 82%
of AC power below 5 Hz vs the synthetic's 97% — real shafts are, if anything,
rougher, so the synthetic is a fair benchmark; it is the SAMPLING that is
wrong, not the roughness.*

**5. With a physical shaft band-limit the numbers move** (lowpass the OU
trajectory before synthesizing audio AND defining GT): fc=None 0.354 → fc=12
0.277 → fc=8 0.221 → fc=5 0.186 (all 40 dB, one pass).

**6. What remains at high SNR is INTER-ROTOR INTERFERENCE.** On clean
band-limited data an *oracle* M2 (siblings reconstructed from GT) reaches
**0.081** vs plain 0.193 and blind M2-solo 0.157. At 0 dB the oracle (0.201)
equals blind M2-solo (0.203) — additive noise then dominates and no sibling
scheme can do better.

### The floor, stated honestly

| condition | achievable |
|---|---|
| exact init, hold | 0.055 |
| fc=5 Hz, 40 dB, oracle sibling removal | 0.081 |
| fc=5 Hz, 40 dB, blind M2-solo | 0.157 |
| fc=5 Hz, **0 dB** (battery setting), oracle == blind | ~0.20 |
| unphysical OU + point-sampled GT, 0 dB | ~0.35 |

So "<0.1 on synthetic" is **not attainable at 0 dB comb-vs-noise SNR** — the
oracle proves the evidence does not support it. It IS attainable at high SNR
with sibling removal. The honest target for the 0 dB battery is ~0.2-0.3,
against which the current pipeline (0.71 trace window, 2.90 battery) has
**3-10x of real headroom**. Any claim of "near-perfect" must state the SNR and
the GT sampling convention.

### Consequent design (supersedes the Round-2 alt_loop)

`refine_v2`: ladder(capture) → M1 corridor sweeps (per-rotor decoupling,
±8@0.25 then ±4@0.1) → **single** M2-solo pass (sibling-subtracted
pi_kalman, PK_WIDE) → stop. No loop, no polish. Report against both the
point-sampled GT (protocol convention) and a band-limited GT (physical
convention), and always alongside the oracle floor for that window.

## WP5 — does dropping VK refine cost amplitude matching? No. (2026-07-30)

`VKResult` carries two independent products: `r_refined` (the trajectory,
updated by the refine rounds) and `envelopes` (the complex per-(rotor,
harmonic) amplitudes from `vk_envelopes`, a **readout** of *(audio,
trajectory, cfg)*). The refine loop only touches the first. Measured with
`refine-diagnosis/amp_check{,2}.py` — VK reconstruction residual ratio
‖x−x̂‖/‖x‖ under RECON_CFG (k 1..30), lower = the harmonic model explains
more:

| window | entry | after VK refine | after pi_kalman | with GT traj |
|---|---|---|---|---|
| synth_trace (0 dB; perfect-comb removal = 0.7071) | 0.7066 | 0.7068 | **0.6845** | 0.6606 |
| fly124_cruise | 0.5774 | 0.5774 | **0.5300** | 0.5498 |
| dregon_ramp (REFINE_CFG) | 1.8191 | 1.8138 | **0.9898** | — |

- VK refine changes the amplitude model by ~0 (fly124: identical to four
  decimals) — a third independent confirmation that the stage is inert.
- A pi_kalman trajectory **improves** the amplitude model on every window,
  because envelopes are a readout: better trajectory → better envelopes. On
  dregon_ramp the VK reconstruction was *diverging* (ratio 1.82 > 1: it added
  energy rather than explaining it — the WP3 cancelling mode) and a better
  trajectory fixes it outright.
- So the refine rounds can be dropped with no amplitude cost; VK stays in the
  pipeline as the envelope/reconstruction engine (M2's sibling subtraction
  already relies on it).

**Note (fly124): our trajectory explains the audio BETTER than the telemetry
does** (0.5300 vs 0.5498). Suggestive that the 29 Hz telemetry label carries
error we are being scored against — but not conclusive, since our trajectory
is *fitted* to the audio while the telemetry is independent. Clean follow-up:
check whether a smoothed telemetry track fits the audio better than the raw
one; if so, the label noise floor is real and must be quoted with any
"much better on Michael's" claim.

**Next design step (from the user's question):** rather than deleting the
frequency update, replace its estimator — `_freq_update`'s Fisher-weighted
phase slopes + pentadiagonal solve is the same estimator family as pi_kalman
but cruder. The synthesis worth building is **pi_kalman's estimator running on
VK envelopes** instead of on brickwall demod: VK's coupled banded LS (coupling
groups + Tůma bandwidth) separates *close* harmonics better than a brickwall
band, which is exactly the twin-collision starvation WP1-B identified as
pi_kalman's binding constraint on real windows.

## WP6 — `refine_v2` measured; the bottleneck MOVES upstream (2026-07-31)

`refine_v2` (ladder+capture → M1 ±8@0.25×3 → M1 ±4@0.1×2 → ONE M2-solo →
stop) is now in `scripts/rps_refine_lab.py`, with `synthbl*` (physically
band-limited) windows and a per-window `oracle_floor`. Pooled final PIT-MAE:

| group | baseline | alt_loop | refine_v2 | v2 (1 M1 round) | oracle |
|---|---|---|---|---|---|
| synth_trace | 1.038 | 0.752 | 0.753 | **0.711** | 0.397 |
| synth battery (6) | 2.893 | 2.639 | **2.542** | 2.568 | 0.952 |
| synthbl battery (6) | 2.389 | 2.268 | 2.076 | **2.067** | 0.697 |
| real (2) | 2.205 | 2.254 | 2.170 | **2.163** | — |
| all 15 | 2.476 | 2.314 | **2.187** | 2.190 | 0.792 |

- vs baseline: −12% pooled at **0.78× the runtime** (dropping the inert refine
  rounds pays for the new stages), 11 wins / 2 losses / 2 ties.
- vs alt_loop: −0.127 pooled at **0.36× runtime**. alt_loop is *worse than
  baseline* on the real pair — its rounds 2-4 + polish degrade a converged
  track exactly as WP4 predicted.
- Structural win where it matters: on the real pair std-ratio 0.772 → **0.954**
  and shape-corr 0.645 → **0.717** — the WP3 "per-rotor deviation is
  unrepresented" defect is largely fixed.
- "Iteration is harmful" extends to the M1 rounds themselves (round 2 is
  another estimator application); 1-round is better on 9/15 windows and
  strictly better on both real windows. Treat `--v2-rounds 1` as the default
  for real data.
- **Sub-bin corridor scoring: not a lever** (−0.030 mean, sign flips per
  window). `_norm_smooth` + the Viterbi transition cost already smooth over
  the 7.8 Hz bin; the corridor's resolution limit is comb *contrast*, not the
  sampling grid.

### The bottleneck is no longer refinement

The oracle floor is **0.792 pooled vs refine_v2's 2.190 — 2.8×** — but that
gap is *not* sibling interference: 3 of 13 windows (synth01, synth02,
synthbl02) have oracle floors of 1.2-2.1, i.e. their entry tracks already
contain a **wholly mis-assigned rotor**, which perfect sibling removal cannot
repair. The battery mean is failure-dominated. Contrast synth_trace, where the
entry is sound: oracle 0.397 vs 0.711 = 1.8×, genuinely a refinement gap.

`synthbl_hi` (20 dB) sharpens it: on 4/6 windows the oracle floor is
0.126-0.393 against a blind 1.056-2.803 — at good SNR sibling removal is worth
~10× and the blind chain leaves nearly all of it unclaimed. The other 2/6
windows collapse to ~40 rev/s for chain *and* oracle: a seed-stage
octave/subharmonic failure that appears **only** at high SNR (the comb then
dominates the whitened spectrum and the octave check mis-fires).

**Conclusion: further refinement work has low marginal value until capture is
fixed.** Priorities now — (WP7) blind-seed / ladder rotor mis-assignment: the
seeder returns duplicate/twin bases and misses a distinct rotor, and nothing
downstream can invent it. Lever: *residual re-seeding generalized* — subtract
all four tracked combs, scan the residual for an unexplained comb, and
reassign the worst-fitting rotor to it (the lab's `run_reseed` already does the
single comb-invisible-rotor case). (WP8) the high-SNR octave failure.

## WP9 — vs the IAVKF paper: why theirs works and our refine did not (2026-07-31)

Reference: Li, Han & Wang, *"Research on a Signal Separation Method Based on
Vold-Kalman Filter of Improved Adaptive Instantaneous Frequency Estimation"*,
IEEE Access 2020 (DOI 10.1109/access.2020.3002999; in the library, fulltext).

**They do not do what our refine stage does.** Their "adaptive IF estimation"
is *external* to VK: synchrosqueezed wavelet transform → adaptive multiridge
extraction with peak detection → the resulting IF is fed to VK as a **given
parameter**. There is no iterative in-band phase-slope correction of the IF.
A TF-ridge estimator has essentially unlimited capture range (it finds the
ridge wherever it sits in the plane); our `_freq_update` is a *local*
narrowband correction whose capture range is ±bw/(2k) — 0.125 rev/s at the
refine stage. So the two results do not conflict: they report that **VK
separates well when handed a good IF**, which we independently confirm (WP5:
envelopes improve monotonically with trajectory quality).

**Phase noise is NOT the cause — measured and refuted**
(`refine-diagnosis/comb_capture2.py`). Synthetic built with comb and noise
kept separate, VK run with the EXACT trajectory, sweeping shaft jitter:

| jitter (per-rotor rps std) | comb captured @ bw 1.5 Hz | @ 0.5 Hz |
|---|---|---|
| ×0.2 (0.46 rev/s) | 100.0% | 100.0% |
| ×1.0 (2.30 rev/s) | 100.0% | 100.0% |
| ×3.0 (6.90 rev/s) | 100.0% | 100.0% |
| ×1.0 at 0 dB SNR | 97.6% | 99.1% |

Jitter is irrelevant because **VK demodulates *along the trajectory*** (phase
k·∫r dt): correct demodulation removes the FM by construction, so the line is
narrow in the envelope domain no matter how much the shaft wanders. Phase
noise would bite only if we demodulated at a *constant* frequency. The refine
stage's failure is therefore purely capture range (WP1), not linewidth.

**Corollary — a label-error probe.** On FLY124, demodulating along the
*telemetry* GT captures only 36.5% of the recording at bw 1.5 Hz but 67.7% at
bw 5 Hz. Since jitter provably does not widen the line, that shortfall is
residual FM = **error in the telemetry itself**. Independent corroboration of
WP5's observation that our estimated trajectory explains the audio better than
the telemetry does. Quantifying this label floor is a prerequisite for any
"much better on Michael's data" claim.

**Architectural convergence.** IAVKF's wide-capture external IF estimator
feeding VK is structurally the same move as our M1 residual corridor Viterbi
feeding M2 — arrived at independently, from the opposite direction. Their
choice of SWT ridges over our comb-corridor DP is worth a comparison: ridges
are single-component and would need the sibling masking we already built.

## WP7 — the mis-assignment is a SEEDER artefact; M3 repairs half of it (2026-07-31)

### (a) Diagnosis — `split_nudge` duplicates, not octave errors

The entry tracks of the failure windows are wrong because `blind_seed`'s arm R
**deliberately duplicates a base** when it cannot find a 4th distinct comb:

```python
seeds_r.append(anchor + sign * cfg.split_nudge * ((cycle + 1) // 2))   # split_nudge = 0.1
```

18 of the 21 lab windows (battery + real) carry at least one seed pair exactly
0.1 rev/s apart — the fingerprint of that line. Only `synth_trace` and
`synthblhi04` are clean, and `synth_trace` is precisely the window WP6 found to
have a sound entry. Per-window (GT = realized 16 s means, not the configured
OU means):

| window | GT means | seed bases | duplicate pair | unseeded true rotor | class | oracle |
|---|---|---|---|---|---|---|
| synth01 | 72.81 80.87 83.53 86.03 | 72.25 **72.35** 81.4 86.4 | 72.25/72.35 | 83.53 (2.1 from nearest) | duplicate-seed + missing-comb | 1.242 |
| synth02 | 71.02 77.85 79.27 82.62 | 77.15 **77.25** 80.0 **80.1** | ×2 | 71.02, 82.62 | double duplicate-seed | 2.109 |
| synthbl02 | 71.01 77.85 79.25 82.62 | 77.2 **77.3** 80.05 **80.15** | ×2 | 71.01, 82.62 | double duplicate-seed | 1.614 |
| synth04 (control) | 79.76 83.50 90.33 93.21 | 79.3 **79.4** 88.85 93.6 | 79.3/79.4 | 83.50 (4.1) | duplicate-seed, recoverable | 0.470 |
| synthbl03 (control) | 71.18 80.34 84.18 87.26 | 70.8 80.75 84.15 **84.25** | 84.15/84.25 | 87.26 (3.0) | duplicate-seed, recoverable | 0.237 |

**No octave errors and no swaps at 0 dB** — the defect is uniformly
duplicate-seed → missing-comb. The "controls" have the *same* defect; they
differ only in that the mis-assignment stays inside the oracle M2's capture
radius (band 12 Hz at k=1..4 admits 3-4 rev/s), so perfect sibling removal
recovers them and the oracle floor looks healthy.

The error is **unrecoverable from `seed` onwards**: per-stage biases move by
<1 rev/s through `coarse_init → viterbi_c → vit2dsp → capture → M1`, and the
big bias never shrinks (synth01: -8.5 at seed, -7.6 after capture, -7.9 after
M1). Nothing downstream invents a comb.

Why arm R fails: `residual_rescan` requires a new comb to be `>= dedup_rps =
4.0` rev/s from every used base, on a *static, time-averaged* masked spectrum.
The missing rotor is 2.1-3.0 rev/s from a used base in synth01/synthbl03/
synth02 — structurally inadmissible. Re-running `blind_seed` with diagnostics
confirms it: `residual_new = []` for synth02/synthbl02, and in 4 of 5 windows
the true base was *already in the accepted candidate list* (synth02's 70.6 and
82.7, synth04's 84.4, synthbl03's 87.55 at a score comparable to the kept
70.8) and was discarded in favour of duplicating the loudest peaks.

**Upstream lever (WP8 candidate, not taken here):** at cruise a quadrotor's
rotors sit 2-3 rev/s apart, so `dedup_rps = 4.0` forbids resolving them by
construction. Lowering it (or preferring an accepted distinct candidate over a
`split_nudge` duplicate) is a one-line change with battery-wide reach.

### (b) M3 — residual re-seed, generalized (`refine_v3`)

`ladder(capture) → M3 → M1 → M2-solo`. Per iteration (max 3):

1. reconstruct **all four** combs (`vk_envelopes`+`vk_reconstruct`, RECON_CFG)
   and subtract; abort if resid/orig > 1.0 (the dregon cancelling mode);
2. comb-scan the 8192-pt whitened residual (`_reseed_scan_scores`, k=1..12,
   0.05 rev/s grid, on-teeth minus half-teeth) → top-6 peaks with robust z;
3. filter: loose z >= 1.5, >= **2.0 rev/s** from every current track, not a
   p/q ∈ {2,3,4} multiple/submultiple of a track (±1.0), and inside a 1.30
   rotor-band span;
4. the rotor to reassign = **least unique energy explained**, from a *re-fitted*
   leave-one-out residual. Surface quality cannot see duplicates (a duplicate
   sits on a genuine comb and scores well); the removal test can — a duplicate's
   comb is still explained by its twin, so its removal is nearly free
   (synth00: gains 0.081/0.066/**0.008/0.007** for the twin pair);
5. **verify, don't trust the scan**: corridor-track the top 3 proposals on the
   sibling residual (±8@0.25 then ±2@0.1) and keep the one that most reduces the
   4-comb residual, requiring a drop >= max(0.008, 0.25 × median of the other
   rotors' leave-one-out gains).

Design decisions and the measurements behind them:

- **z is a proposer, not a gate.** At the ladder exit the genuinely missing
  rotor scores z = 1.8-3.1 in the residual while comb-residue ridges reach
  3.5-4.0; the seeder's z >= 3.0 both admits junk and rejects the truth.
  Verification by residual drop picks the right comb in synth01 even though it
  was the *lowest*-z of the three proposals (drops 0.019/0.024/**0.027**).
- **The drop threshold is adaptive** because a real repair drops the residual
  by 0.027-0.040 while the spurious combs later iterations keep proposing drop
  it by 0.003-0.006 — an order of magnitude, referenced to the ~0.05-0.08 a
  genuinely-owned rotor contributes.
- **`MIN_SEP = 2.0`, not 1.5.** 1.5 let FLY124 re-seed onto its 93.1 residual
  peak against a track the ladder had dragged to 91.16: two tracks on one
  strong comb, which the coupled VK solve *rewards* (near-degenerate pairs
  cancel, residual drops 0.032) while PIT-MAE collapses 1.13 → 4.80. 2.0 is the
  minimum pairwise rotor separation the battery generator itself assumes.
- **Rotor-band span 1.30.** With the two genuine proposals filtered out, the
  verifier happily took a 66 rev/s noise ridge against tracks at 87-96
  (synth00, 20.4 rev/s error): *any* extra comb with freely fitted VK envelopes
  absorbs some residual energy, so the drop test alone is not self-limiting.
- **No post-track duplicate guard.** Rejecting a repair that lands near a
  sibling was tried and reverted: it killed synth01's correct repair (1.50 gap)
  and synth00's (0.18 gap). M1's sibling-masked coordinate descent is the
  mechanism that separates co-located tracks; that is its job, not M3's.

### (c) Measured, `--v2-rounds 1` throughout (WP6 default)

| group | refine_v2 | refine_v3 | Δ | v2 oracle | v3 oracle | v2 s/win | v3 s/win |
|---|---|---|---|---|---|---|---|
| synth battery (6) | 2.568 | **2.179** | −0.389 | 0.953 | 0.899 | 62 | 83 |
| synthbl battery (6) | 2.067 | **1.742** | −0.325 | 0.695 | 0.827 | 65 | 83 |
| synth_trace | 0.711 | 0.711 | 0.000 | 0.372 | 0.372 | 70 | 52 |
| real (2) | 2.163 | 2.163 | 0.000 | — | — | 136 | 102 |
| **all 15** | **2.190** | **1.904** | **−0.286** | 0.790 | 0.825 | 73 | 83 |
| synthbl_hi (6) | 15.505 | 15.433 | −0.071 | 14.684 | 14.690 | 36 | 57 |
| synthbl_hi (4 non-collapsed) | 1.820 | 1.713 | −0.107 | 0.202 | 0.211 | 34 | 62 |

7 wins / 4 losses / 4 ties over the 15. M3 fires on 13 of 21 windows and
**no-ops on every window that does not need it**: `synth_trace` and both real
windows come back bit-identical (dregon aborts at the recon guard, ratio 110.9;
fly124 stops on the drop test). Of the three WP6 failure windows:

| window | v2 final | v3 final | v2 oracle | v3 oracle | repaired? |
|---|---|---|---|---|---|
| synth01 | 3.168 | **1.340** | 1.242 | **0.585** | yes — both halve |
| synth02 | 3.942 | 4.162 | 2.109 | 2.341 | no |
| synthbl02 | 3.556 | 3.776 | 1.614 | 2.439 | no |

The wins came from windows WP6 did *not* flag, whose duplicate seeds the
oracle had been silently covering: synth00 2.570→1.676, synthbl00 2.177→1.441,
synthbl03 1.348→0.803, synthbl04 1.547→1.021, synthbl05 2.724→2.278.

**seed 102 (synth02/synthbl02) is the residual failure and is structural:** its
two closest true rotors are 1.42 rev/s apart — *below* the 2.0 rev/s separation
both the design and the battery generator assume distinct rotors have — so
every correct proposal is excluded by the MIN_SEP guard, and at aggressiveness
1.4 (per-rotor std 3-4 rev/s) the trajectories cross constantly. It also needs
TWO repairs. Relaxing MIN_SEP to 1.5 recovers it (3.164) but costs FLY124
1.13 → 4.80; the real window wins.

The pooled oracle floor is essentially unchanged (0.790 → 0.825) even though
the chain improved 13%: M3 moves tracks *toward* the truth in a way the oracle
was already able to fake on the recoverable windows, so the remaining
chain-vs-oracle gap (1.904 vs 0.825, 2.3×) is now much more genuinely a
refinement gap than it was at WP6 (2.190 vs 0.790, 2.8×, failure-dominated).

**Cost:** +14% wall on the 15-window set (73 → 83 s/window); ~8 VK envelope
re-fits per M3 iteration (1 full + 4 leave-one-out + up to 3 verification).

## WP8 — the seeder knobs: `dedup_rps` is safe, promotion is not (2026-07-31)

WP7(a)'s "one-line change with battery-wide reach" implemented and measured.
Two `SeedConfig` knobs, both **default-preserving** (verified bit-identical
seeds vs the old code on both real windows, and `chain=baseline` still PASSes
all four trace references 3.262/3.214, 1.148/1.140):

- `dedup_rps` (existing, 4.0): with arms `{"K","R"}` it only gates arm R's
  residual re-scan — how far a NEW comb must sit from every used base.
- `prefer_distinct_candidate` (new, False) + `min_sep_rps` (2.0),
  `promote_z_min` (2.0), `promote_span` (1.30): when a slot would be filled
  with a `split_nudge` duplicate, promote the best *accepted distinct*
  candidate instead. The last two guards are measured necessities — unguarded
  promotion takes a 113.6 rev/s ridge on synth01 and 102.6 on FLY124.

### Seed-stage sweep (21 lab windows, `blind_seed` only, arms `{"R"}`)

Metrics: mean sorted-assignment |seed − true 16 s mean| (`pit_mae`), Σ seed
pairs < 0.5 rev/s apart (`dups`), Σ true rotors with no seed within 2 rev/s.

| config | pit_mae | dups | unseeded | vs default |
|---|---|---|---|---|
| dedup 4.0, off (**shipped**) | 2.566 | 24 | 28 | — |
| dedup 3.0, off | 2.566 | 24 | 28 | identical (no candidate in 3-4) |
| dedup 2.5, off | 2.373 | 21 | 26 | 2 wins / 1 loss |
| dedup 2.0, off | 2.472 | 19 | 26 | **FLY124 collapses 1.23 → 4.70** |
| dedup 1.5, off | 2.472 | 19 | 26 | same collapse |
| dedup 4.0, promote | 1.945 | 12 | 20 | 11 wins / 0 losses |
| **dedup 2.5, promote** | **1.722** | **10** | 19 | **13 wins / 0 losses** |
| dedup 2.5, promote, min_sep 1.5 | 1.657 | 8 | 19 | 14 wins / 0 losses |
| dedup 2.5, promote, z 1.0 | 1.772 | 5 | **17** | 13 wins / 2 losses |

`dedup_rps <= 2.0` reproduces the WP7 MIN_SEP failure **at the seeder**:
FLY124's residual scan then admits a 90.35 comb 2.0 rev/s from the 92.35 base
— two seeds on one comb, exactly the mode that cost M3 1.13 → 4.80.

### End-to-end (`--v2-rounds 1`; "dd" = `dedup_rps 2.5` only)

| group | v2 def | v2 dd | v2 both | v3 def | v3 dd | v3 both | orc v3 def | orc v3 both |
|---|---|---|---|---|---|---|---|---|
| synth battery (6) | 2.568 | 2.191 | **1.870** | 2.179 | 2.211 | **1.680** | 0.899 | 0.729 |
| synthbl battery (6) | 2.067 | 2.067 | **1.765** | 1.742 | 1.742 | **1.654** | 0.827 | 0.626 |
| synth_trace | 0.711 | 0.711 | 0.711 | 0.711 | 0.711 | 0.711 | 0.372 | 0.372 |
| real (2) | 2.163 | 2.163 | 2.163 | 2.163 | 2.163 | 2.163 | — | — |
| **all 15** | 2.190 | 2.039 | **1.790** | 1.904 | 1.917 | **1.669** | 0.825 | **0.654** |
| synthbl_hi (4 non-collapsed) | 1.820 | 1.233 | **1.131** | 1.713 | 1.282 | **1.138** | 0.211 | 0.251 |

v2: 7 wins / 1 loss (+0.08) / 7 ties; v3: 5 wins / 3 losses / 7 ties. The
oracle floor drops with the chain (0.825 → 0.654), i.e. the *entry* really is
better, not just the refinement. Seed 102 (synth02/synthbl02), WP7's
"structural" residual failure, finally moves: v3 4.162 → 2.826 and 3.776 →
2.671, oracle 2.341 → 1.254 — the promoted 82.7 comb is one of its two needed
repairs. Cost: none (v2 52.9 s/window vs 73.4 default; v3 81.7 vs 83.2).

### Why the production default must NOT flip to promotion

Checked at the seed stage on **all 15 cached beatvk real windows** (FLY124
w00-w05 + the three DREGON recordings' w00-w02):

- `dedup_rps = 2.5` alone: **bit-identical seeds on all 15**. Safe.
- promotion on: changes 2 of 15, both for the worse —
  `free-flight_nosource_room1` **w02** 0.56 → 2.37 seed MAE, and end-to-end on
  the production `baseline` chain **1.019 → 4.487**.

Mechanism: DREGON at cruise really is two tight twin pairs (74.8/75.4 and
84.9/86.2), so the `split_nudge` duplicate is the *physically correct* seed and
the "distinct comb" (77.5 rev/s, z 2.25) is junk. Contrast cannot separate the
cases — the good synth02 promotion sits at z 2.57, the bad DREGON one at 2.25.
This is WP7's own lesson restated: **z is a proposer, not a gate**, and the only
working acceptance test is M3's — track the candidate and check the residual
drops. That test needs trajectories, which the seeder does not have.

→ Recommendation: ship `dedup_rps = 2.5` (all-15 v2 2.190 → 2.039, zero real
change); keep `prefer_distinct_candidate = False` as the default and use it for
synthetic/lab work. The real prize (all-15 1.669) needs the promotion to be
*verified* — the natural next step is to let M3 propose from the seeder's
discarded accepted candidates, where the residual-drop test can adjudicate.
FLY124 is untouched by either knob: its missing rotor is 1.2-2.0 rev/s from a
used base and has no accepted candidate at all, so "much better on Michael's"
does not come from here.

## WP12 — M3 made safe, and FLY124 w05 recovered (2026-07-31)

**The M3 regression had two real causes, both worth remembering.**
(1) *Leave-one-out gain cannot distinguish a duplicate seed from a genuine
tight twin* — FLY124 cruise really is `[73.96, 74.85, 80.73, 90.79]`, so the
74/75 pair each look "redundant" (the sibling covers the same teeth). M3
declared a correctly-held track spare. (2) *The residual-drop test is not
self-limiting*: a proposal tracked to 1.94 rev/s from an existing track — same
comb — and the coupled VK solve **rewards** that (near-degenerate pairs
cancel), drop 0.036 vs `min_drop` 0.008. w04 1.087 → 4.743. A third variant
appeared on w02: LOO gains go *negative* for a genuine collapsed pair and the
objective prefers vacating it over the actually-spare pair.

Fix = three changes: **Gate A** (redundancy — fire only if some track's LOO
gain is ≤0 or ≤0.50× the median of the others; duplicates sit at 0.11–0.25 of
it, FLY124's genuine twins at 0.53–0.70); **Gate B** (multiplicity — stop if
the ≤2.0 rev/s neighbour graph has more than one disjoint degenerate group,
since one verified move cannot rank two independent repairs); and a **verified
search over (rotor × proposal)** where LOO only *orders* candidates and the
residual test chooses both the vacated track and its destination.

Result: v3 ≡ v2 on **all six** FLY124 windows (M3 now no-ops on each, at four
different gates), and the synthetic pooled improves to **1.817** (vs WP7's
1.904) — 7 wins / 1 loss / 7 ties, with WP7's synth02/synthbl02 losses
removed. *Fragility to watch: Gate A's margin is thin — w04's minimum is 0.53
against the 0.50 threshold.*

**Cross-window seeding recovers w05: 3.380 → 1.147** (per-rotor 0.72/1.18/
1.57/1.12), putting it in the same band as w03 (1.130) and w04 (1.087).
Bases corroborated by ≥2 other cruise windows of the same recording become
extra proposals. Pooling alone was NOT enough: the drop test picked the wrong
rotor, because correctly placing a track on the comb-invisible 81 rev/s rotor
is worth ~10× *less* residual than mis-placing one beside the strong 91 comb —
the audio can adjudicate the *base* but not the *rotor*. So in the pool branch
the mover is chosen by **cross-window rank evidence** (sorted rank r is stable
at cruise) and the audio only has to not contradict.

Three assumptions, all required, all off by default (`--m3-pool`/`--m3-ref`):
same recording + cruise + near-constant speeds; **sorted-rank identity stable
across windows** (a majority vote — it would have named the wrong rotor on w04,
which is protected only by Gate A); corroboration by ≥2 windows (this is what
stops w02's mis-tracked 72.1 base contaminating w03/w04). w02 is *not*
recovered — its audio actively contradicts the repair, and it is the
takeoff→cruise transition window (GT std 7.4/3.4/3.6/3.8 vs 1.2–2.1).

**FLY124 label lag, measured (feeds WP11):** w02 −61.83, w03 −49.00, w04
−34.61, w05 −31.77 ms (w00/w01 diverge, skipped). OLS vs window-centre time:
**+0.654 ms/s · t − 86.1 ms, R² 0.94**, residual RMS 2.9 ms vs 12.0 ms for a
constant-lag model → dilation confirmed; implied multiplier ×1.000654 on the
shipped 1.001.

## WP11 — Michael's telemetry calibration, packaged for the cluster (2026-07-31)

`scripts/michaels_calib/` is the repo-ified version of the scratchpad probe, so
the full sweep runs in ONE CPU job instead of hand-driven local windows:

- `windows.py` — replays the frozen beat-VK windowing protocol against the raw
  recordings, for FLY125 too (it is the *training* recording and is not in
  `beatvk-valid-raw`). Data root = `$DATA_ROOT` → `<repo>/data` → dload
  `recording_with_motor_speed` via `streams.ensure_local`, so it runs on any
  backend. `selfcheck()` proves the rebuild is bit-identical to the frozen
  FLY124 prep cache when that cache is present.
- `calib.py` — referee (VK recon residual ratio under `RECON_CFG`, k 1..30 —
  the same one-liner as `rps_refine_lab.RECON_CFG`) + the four stages
  `lag` / `val` / `prot` / `post`.
- `fit.py` — lag-vs-time OLS → proposed `time_offset` / `time_dilation`, and
  the additive-vs-multiplicative verdicts.
- `run_sweep.py` — the driver: prep cache once in the parent, then one process
  per (window, stage[, rotor]); restartable (skips units whose JSON exists);
  writes `results/michaels_calib/{manifest,summary}.json` + `raw/*.json`.

Sweep contents: FLY125 lag scan on **every** cruise window + the regression;
FLY124 lag on w03/w04 only, scored against the WP12 numbers above (a
confirmation, not a re-derivation); and, for both recordings, the
additive-vs-multiplicative rev/s test — `val` (matched families, `g = b/80`)
plus the per-rotor `prot` discriminator, which carries the leverage: additive
⇒ the per-rotor optimum is independent of that rotor's mean rps,
multiplicative ⇒ proportional to it (cruise spans 74–91 rev/s, a 23 % spread).

`src/data_processing/michaels.py` is deliberately **untouched** — applying the
correction comes after the calibration lands.

## WP13 — the calibration measured, and APPLIED to the loader (2026-07-31)

Sweep result: `omnirun-outputs/python-519b66/results/michaels_calib/summary.json`
(959 s wall, 16 cores, `uni-cpu`). Referee = the label-free VK reconstruction
residual; only the telemetry is moved.

### Timing — BOTH recordings have a clock-rate (dilation) error

| recording | cruise windows | lag range | OLS slope | intercept | R² | resid RMS | vs constant-lag |
|---|---|---|---|---|---|---|---|
| FLY124 | 4 (w02–w05, centres 40/56/72/88 s) | −61.83 → −31.77 ms | **+0.65356 ms/s** | −86.131 ms | 0.942 | 2.90 ms | 12.04 ms |
| FLY125 | 9 (w01–w09, centres 24–152 s) | −158.35 → −105.54 ms | **+0.37656 ms/s** | −172.086 ms | 0.923 | 4.49 ms | 16.19 ms |

Both are dilation, not offset: the linear fit beats the constant-lag model by
4.1× / 3.6× in residual RMS. FLY124's four lags are WP12's; the cluster job
re-measured w03/w04 independently at −48.51 / −34.56 ms (vs −49.00 / −34.61),
i.e. within 0.5 ms, so the two derivations are the same measurement.

**The algebra** (`scripts/michaels_calib/fit.py:fit_lag`, applied identically to
both recordings). With the old constants the audio prefers the telemetry shifted
by `lag(t) = a + b·t` seconds. The loader maps a raw telemetry stamp τ to
audio-frame time as `W = d·τ + (ts_raw[0] − time_offset)`, so absorbing the
drift means

```
time_dilation_new = time_dilation_old / (1 − b)
time_offset_new   = time_offset_old  − a / (1 − b)
```

FLY124 (b = 6.5356e−4 s/s, a = −0.086131 s, old = (−20.84, 1.001)):

```
1/(1−b)        = 1.00065399
time_dilation  = 1.001   / 0.99934644 = 1.001654644
time_offset    = −20.84 − (−0.086131)/0.99934644 = −20.84 + 0.086187 = −20.753813
```

FLY125 (b = 3.7656e−4 s/s, a = −0.172086 s, old = (−26.51, 1.0048)):

```
1/(1−b)        = 1.000376701
time_dilation  = 1.0048 / 0.99962344 = 1.005178509
time_offset    = −26.51 − (−0.172086)/0.99962344 = −26.51 + 0.172151 = −26.337849
```

### Value — a real ~+0.6 rev/s deficit, of an UNRESOLVED functional form

Telemetry is LOW at cruise on both recordings (DREGON is biased the other way,
so this is a rig property, not a referee bias). The additive-vs-multiplicative
discriminator (`prot`: per-rotor optimal offset regressed on that rotor's own
mean rps, over the 74–91 rev/s cruise spread) is **inconclusive**:

| recording | additive RMS | multiplicative RMS | verdict | margin | free-line R² | predicted spread | observed spread |
|---|---|---|---|---|---|---|---|
| FLY124 | 0.2939 (b = 0.6796) | 0.30557 (g = 0.008387) | additive | 4 % | 0.013 | 0.147 | 0.938 |
| FLY125 | 0.38621 (b = 0.5545) | 0.38607 (g = 0.006904) | multiplicative | 0.04 % | 0.004 | 0.117 | 1.964 |

The two recordings return opposite verdicts, one of them by 0.04 % — a tie. The
free lines explain ~1 % of the variance, and the observed per-rotor spread is
6–17× what *either* model predicts, i.e. the per-rotor optima are dominated by
something else (per-rotor acoustic strength / tracking) and the test has no
statistical power. The matched `val` comparison agrees it cannot separate them
(mean Δ ≈ −3e−4 residual, both families reaching ~0.52).

**Resolved on physical grounds, not statistics: MULTIPLICATIVE.** The two forms
are indistinguishable in the cruise band where they were measured, but the
published frames span the WHOLE recording — ground idle, warm-up, spin-down. An
additive +0.6 rev/s would corrupt a near-stationary rotor's label and
manufacture harmonics at a standstill; a scale correctly vanishes as rps → 0.
Shipped at the time: FLY124 ×1.00839, FLY125 ×1.00690 (+0.671 / +0.552 rev/s at
80 rev/s).

> **SUPERSEDED (WP14, same day).** The *form* — one global multiplicative scale
> per recording — survived; the *magnitudes* did not. These two came from a 2–4
> window per-rotor mean that included the twin-pair rotors, and disagreed by
> 0.15 pp. The refit over all 13 cruise windows, non-twin rotors only, ships
> **FLY124 ×1.00698** and **FLY125 ×1.00706** (0.008 pp apart). Everything below
> in WP13 describes the state as measured then; read WP14 for what is shipped
> now.

### Applied

`src/data_processing/michaels.py`: `MICHAELS_FILES` carries the new
offset/dilation pairs, and a new `MICHAELS_RPS_SCALE` (keyed by CSV stem, with
`rps_scale_for()`) is applied inside `_load_michaels_data_raw`, so **every**
consumer of the loader — `load_michaels_timeframe(s)`, `publish_frame_datasets`,
`create_dregon_librimix`, the online-mix `michaels` source, `vk_blind_sweep`,
`scripts/michaels_calib/windows.py` — gets calibrated labels with no call-site
change. Unknown recordings (the 103 unaligned `new-drone-noises` logs) fall back
to 1.0. `michaels-frames` was republished on top of this; its per-recording
`meta` now records `rps_scale` plus a `provenance.calibration` note, and the raw
`Motor:Speed:*` CSV columns remain **uncalibrated** in the `motor_speed` block
(the same fixed/raw pairing DREGON has with `motors_command_raw`).

Known bypass paths (read the CSV directly, so they never see the scale):
`writing/reports/2026-06-30_synthetic-rps-trajectories/prepare.py` (RC-stick
figure, no rps), the `motor_speed` block above, and
`notebooks/michael_data_analysis.ipynb` (which still hardcodes the OLD
`-20.84/1.001` and `-26.51/1.0048` — it is the historical origin of the
hand-tuned pair, left as a record).

### Validation — the correction closes the gap (job `python-0445f6`, 596 s)

`run_sweep.py --post-shipped` re-runs the `post` stage (residual lag + residual
global offset) on every cruise window of both recordings, rebuilt with whatever
`michaels.py` now ships. Results
(`omnirun-outputs/python-0445f6/results/michaels_calib_post/summary.json`,
13 windows, no edge hits):

| recording | n | resid lag mean | resid lag RMS | resid lag max | resid b mean | resid b max | (was) |
|---|---|---|---|---|---|---|---|
| FLY124 | 4 | −2.47 ms | 2.98 ms | 3.95 ms | −0.178 rev/s | 0.239 | lag −62…−32 ms, b +0.68 |
| FLY125 | 9 | +1.27 ms | 3.29 ms | 8.35 ms | +0.004 rev/s | 0.107 | lag −158…−106 ms, b +0.55 |

Timing is closed on both: the residual lag RMS (3.0 / 3.3 ms) is at the level of
the fit's own residual (2.9 / 4.5 ms), i.e. the dilation model absorbed the whole
drift, and the systematic per-window trend is gone (FLY124 w02→w05 residuals
+0.28/−3.95/−3.64/−2.55 ms, no monotone drift; FLY125 spans ±8 ms with no slope).

FLY125's value error is closed exactly (mean +0.004 rev/s). **FLY124 is left
~0.18 rev/s over-corrected** — consistent by construction: the shipped scale came
from the per-rotor `prot` mean (0.680 at cruise), whereas the global-offset scans
(`val` on w03/w04: 0.513 / 0.608) preferred ~0.56. The gap, 0.1–0.2 rev/s, is
well inside the 0.94 rev/s per-rotor spread that made the additive/multiplicative
test powerless in the first place, and is ~5× smaller than the error it removed
— not worth a second fit on the same four windows. Flagged, not chased.

> **Chased anyway, in WP14** — the 13-window sweep was run for the per-rotor
> question and re-measured the global gain for free. The flag was right:
> FLY124's scale is 0.14 pp too high. Both scales are now refit (see WP14
> § "Applied"); `python-b7e225` is the `--post-shipped` re-validation on the
> corrected labels.

`windows.selfcheck()` now rebuilds explicitly at `FROZEN_BEATVK_CONSTANTS`
(−20.84, 1.001, scale 1.0), since the frozen beat-VK prep cache predates the
calibration and can no longer match the shipped constants by construction.

### Stale artefacts (blast radius — NOTHING regenerated here)

Everything built from Michael's telemetry before `michaels-frames@a7a951b94808`
carries labels late by 31–158 ms and low by ~0.55–0.67 rev/s at cruise. (WP14
later republished as `fdef818432e9` with the refitted rev/s scales — a further
−0.11 rev/s on FLY124 / +0.013 rev/s on FLY125 at cruise. Anything rebuilt
against `a7a951b94808` is *nearly* current; anything older is not.)

| artefact | how stale | to rebuild |
|---|---|---|
| `beatvk-valid-raw` (dload pin `268c7660`) | its FLY124 `rps` was copied from `michaels-frames`; window boundaries also shift (the eval span is derived from the telemetry) | `python scripts/publish_beatvk_valid.py` then `dload pin beatvk-valid-raw` |
| `results/beatvk_vk_arms/` (manifest + 120 MB `prep_cache` + `runs`) | frozen at the old constants; `windows.selfcheck()` is now pinned to `FROZEN_BEATVK_CONSTANTS` for exactly this reason | `scripts/beatvk_vk_arms.py` (re-prep + re-score) |
| `docs/experiments/beat-vk.md` scoreboard | every `fly124_cruise` / steady-FLY124 column, incl. the **1.027 blind-VK bar** and the 1.29/1.33 neural rows, was scored against the old labels; the DREGON columns and the 0.688 bar are unaffected | re-score after the two rows above |
| `writing/papers/2026-07_coupled-vk-blind-rps/main.tex` | the FLY124 columns/rows (1.027, 1.016 → 0.859, the w0–w2 table) and the prose around them | after re-scoring |
| `scripts/rps_predictor_vk_eval.py` (`michaels_FLY124` rows of the hardcoded per-sample table, ~L232–246) | baked GT means/stds from old labels (e.g. cruise 80.374 → ≈ 81.0) | regenerate the table |
| `DREGON-LM-V4-michaels-{train,valid,valid-full}` | offline `rps.npy` baked from the loader. Local copy: **1552 / 9000 train** samples are `michaels_FLY125`; the local valid splits are DREGON-only, but the *published* pins may differ — verify before relying on that | `scripts/create_dregon_librimix.py` (canonical command in `data_processing/AGENTS.md`) |
| online-mix policies using `kind: michaels` / `kind: frames + michaels-frames` (`g1_*`, `g3_gp_aug`, `g5`, `g6`, `g7_ramp`, `e9_real_ft*`, `e12_*`, `beatvk_avq_dload`, `online_mix_v4_michaels_timewarp*`, …) | none pins an explicit `version:`, so they follow `dload.lock` and pick the corrected frames up **automatically** — i.e. models trained before this pin were trained on different labels | nothing to change; note the discontinuity |
| GP egonoise `matrice100` checkpoint (`r2://ml-data/artifacts/gp_egonoise/matrice100`) + the noise generators' michaels codebook entries | fitted against old labels (rps → spectrum mapping shifted by ~0.6 rev/s) | retrain if the ~0.8 % rps shift matters |

Believed **unaffected**: DREGON-only results, `AVQ-egonoise-vkrps` (blind
pseudo-labels from AVQ audio, no michaels telemetry in the product — but the
annotator's *validation* numbers on FLY124 are stale), and every michaels
*audio* artefact (audio itself never changed, only the label clock/scale).

## WP14 — the value model challenged, and settled: ONE global scale, no per-rotor terms (2026-07-31)

WP13's rev/s calibration was fitted from 2–4 windows per recording, and its
`prot` points looked like they carried *per-rotor* structure (FLY124 rotor0
0.50/0.58 vs rotor2 0.91/0.84 — a 4–5× separation against their own scatter,
and *anti*-correlated with rotor speed, which contradicts a multiplicative
scale). The competing hypothesis was that the residual is not a calibration
error at all but a **per-rotor lag** between the ESC-reported speed and the
physical rotor — DREGON's `motors_command` vs `motors_measured` situation.

Both were tested. Sweep `scripts/michaels_calib/run_perrotor.py` (job
`python-ace021`, uni-cpu 16 cores, 22 min, 156 units), verdict tables
`scripts/michaels_calib/analyze_perrotor.py`. Baseline = **raw** rev/s
(`rps_scale=1.0`) with the shipped `time_offset`/`time_dilation`, all **13**
cruise windows (4 FLY124 + 9 FLY125), 4 rotors each. Rotors 1/3
(LFront/RBack) are the near-equal-speed twin pair the audio cannot resolve —
flagged everywhere, excluded from every fit.

### The ESC signal chain, from the CSV alone (no audio)

- Log rate 29.58 / 29.53 Hz (1 ms clock quantisation). `Motor:Speed` is
  **integer RPM** — 1 RPM = 0.0167 rev/s, 40× finer than the disputed 0.6 —
  and is *not* a reciprocal-period lattice.
- **Zero-order hold at ~17.0 Hz**: only 57.2–57.7 % of log frames change
  value, hold runs are essentially all exactly 1 frame. **All four rotors
  update on the SAME frames** (P(j changes | i changes) = 0.986–0.993 vs
  0.574 if independent) → the ESC telemetry block is sampled synchronously,
  there is **no round-robin stagger** and hence no per-rotor sampling-phase
  lag. The ZOH's ~29 ms mean staleness is *common* and already absorbed by
  `time_offset`.
- Command→response lag (`MotorCtrl:PWM` → `Motor:Speed`, z-scored, detrended;
  PWM is duty, so only the *timing* relation is meaningful): FLY124 50/63/58/76 ms,
  FLY125 60/70/83/85 ms (RFront/LFront/LBack/RBack) — **same rank order in both
  recordings**, so a ~25 ms per-rotor spread in the total command→report delay
  is real. It mixes the mechanical time constant, any reporting delay and the
  ZOH; the CSV cannot separate them.
- **The arithmetic that kills the lag hypothesis a priori**: cruise
  `d(rps)/dt` has RMS 15.3–19.2 rev/s² but **mean −0.08…+0.43 rev/s²**. A lag
  τ makes the error `−τ·ṙ`, so at τ = 30 ms it is 0.46–0.57 rev/s **RMS** but
  0.0003–0.013 rev/s in the **mean**. The disputed offsets are +0.5…+0.9 rev/s
  *constant and positive* — 40–600× more than any plausible lag can bias.
- Back-EMF cross-check (`Motor:Volts`·PWM/100 − I·R, per-rotor Kv): **cannot
  resolve a 0.8 % scale error.** Fitted Kv spans 4–8 % between rotors and is
  not even stable between recordings (LFront 345 → 310 RPM/V); residual RMS
  0.57–0.79 rev/s. Its one durable output: with a shared Kv/R the per-rotor
  residual means are reproducible across both recordings (LBack +0.68 always
  highest, RBack −0.79/−0.58 always lowest, ±0.7 rev/s) — i.e. per-rotor
  differences of a few tenths of a rev/s are physically ORDINARY for this rig
  and are not evidence of a telemetry fault. `Motor:V_out` is **not** an
  independent voltage (V_out ≈ 1.32 × PWM in every rotor).

### A — per-rotor offsets are NOT distinguishable (raw baseline, all windows)

| rec | rotor | n | mean rps | mean b | sd b | b/rps |
|---|---|---|---|---|---|---|
| FLY124 | RFront | 4 | 90.59 | 0.551 | 0.059 | 0.608 % |
| FLY124 | LBack | 4 | 80.42 | 0.650 | 0.252 | 0.808 % |
| FLY124 | *LFront (twin)* | 4 | 73.84 | *0.551* | *0.141* | — |
| FLY124 | *RBack (twin)* | 4 | 74.85 | *0.800* | *0.501* | — |
| FLY125 | RFront | 9 | 90.42 | 0.584 | 0.070 | 0.646 % |
| FLY125 | LBack | 9 | 81.16 | 0.635 | 0.148 | 0.782 % |
| FLY125 | *LFront (twin)* | 9 | 73.81 | *0.631* | *0.101* | — |
| FLY125 | *RBack (twin)* | 9 | 74.91 | *0.634* | *0.764* | — |

**Between-rotor spread is SMALLER than within-rotor scatter**: 0.099 vs 0.155
(FLY124), 0.051 vs 0.109 (FLY125) — ratios 0.64 and 0.47. The apparent
per-rotor structure in WP13 was a 2-window artefact. Adding per-rotor constants
buys **0 %** (FLY125) / 4 % (FLY124) of rms over one global constant. The
twin rotors are exactly where the wild scatter still lives (RBack sd 0.50 /
0.76), as expected.

Model rms on the non-twin points: FLY124 additive 0.166 / multiplicative 0.175
/ free line 0.164 / per-rotor const 0.159; FLY125 0.112 / 0.123 / 0.109 /
0.109. After refitting one global scale the residual rms (0.175 / 0.123) equals
the estimator's own per-window scatter — **there is no structure left to model**.

### B/C/D — the lag hypothesis is refuted, three independent ways

- **Acceleration split** (the `off` scan keeps per-frame residuals, so the
  apparent offset is re-minimised on any frame subset). Well-observed rotors,
  b(ṙ>0) − b(ṙ<0): +0.026 / +0.040 (FLY124 RFront/LBack), −0.014 / +0.075
  (FLY125) → implied τ = **−3.6…+0.8 ms**, inconsistent in sign and 20× below
  the measured 60–85 ms command→report delays. b at low |ṙ| vs high |ṙ| is
  flat (0.548 vs 0.557; 0.585 vs 0.576). **A calibration offset is flat in
  this split; it is.**
- **Per-rotor lag scan**: RFront −1.3 ± 1.8 ms (FLY124), +1.0 ± 4.0 ms
  (FLY125); LBack −14.9 ± 19.8 / +8.2 ± 16.1 ms — all consistent with zero.
- **Joint (τ_r, b_r) grid**: from the (0,0) origin, the offset axis alone buys
  Δrecon 0.0104 / 0.0092; the lag axis alone buys **0.0000** (4 dp); both
  together equal the offset alone. `argmin τ = 0` for every well-observed
  rotor. **The offset carries 100 % of the explanatory power.**

### The global magnitude, refitted on 13 windows

| rec | fitted g (non-twin) | scale (now shipped) | was shipped | residual after the OLD scale | after refit |
|---|---|---|---|---|---|
| FLY124 | 0.698 % ± 0.069 | 1.00698 | 1.00839 | mean **−0.117**, rms 0.213 | +0.004, rms 0.175 |
| FLY125 | 0.706 % ± 0.034 | 1.00706 | 1.00690 | mean +0.017, rms 0.124 | +0.003, rms 0.123 |

FLY125's shipped constant is **confirmed** (0.5 σ). FLY124's is **~0.14 pp too
high** — a +0.12 rev/s over-correction at cruise, the same sign and order as
the −0.18 rev/s residual the `--post-shipped` validation already flagged. The
two recordings' refitted gains agree to 0.008 pp, which the shipped pair did
not (0.15 pp apart).

### Additive vs multiplicative: still a tie, now leaning slightly additive

The only lever is RFront (90.5) vs LBack (80.8): multiplicative predicts the
faster rotor needs **more** correction (+0.07 rev/s), additive predicts equal.
Observed contrast is **negative** in both recordings: −0.099 ± 0.129 (−1.3 σ
from multiplicative, −0.8 σ from additive) and −0.051 ± 0.054 (−2.1 σ / −0.9 σ).
Stouffer-combined: −2.4 σ from multiplicative, −1.2 σ from additive. Weak, and
the direction (faster rotor needs *less*) is what *neither* model predicts.

Two arguments keep the multiplicative form anyway:
1. **Warm-up/idle safety** (the WP13 reason, unchanged): a scale vanishes at
   rps → 0, an additive +0.6 rev/s invents motion on a stationary rotor, and
   the published frames cover the ground/warm-up span.
2. **The global error is degenerate with a clock error, which is exactly
   multiplicative.** With εa = audio sample-rate error, εt = telemetry-clock
   error, εr = ESC rev/s scale error: `time_dilation − 1 ≈ εa − εt` and the
   audio-optimal gain `g ≈ −εa − εr`. Measured D−1 = +0.165 % / +0.518 %,
   g = +0.84 % / +0.69 %. Setting εr = 0 gives εa = −0.84 / −0.69 % (same
   device, agree to 0.15 pp) and εt = −1.00 / −1.21 % (agree to 0.21 pp);
   setting εa = 0 gives εr = −0.84 / −0.69 % and εt = −0.17 / −0.52 % (0.35 pp).
   Both self-consistent, not separable — but under **either** the correction is
   exactly multiplicative and vanishes at rps → 0. It also means the constant
   is a *label-for-this-audio* correction, not proof the telemetry is wrong.
   And no clock hypothesis can produce a per-rotor difference: εa is common to
   all four rotors.

### Verdict

**The shipped model FORM is right: one global multiplicative scale per
recording.** Per-rotor constant offsets are not identifiable (between-rotor
spread < within-rotor scatter) and a per-rotor lag is refuted outright. The
only defect is the **magnitude**: FLY124's 1.00839 was 0.14 pp too high
(+0.12 rev/s over-correction at cruise, ~1/5 of the error it removed, inside
the per-window scatter), and FLY125's 1.00690 was confirmed at 0.5 σ but is
worth replacing anyway so both constants come from the *same* fit.

### Applied (2026-07-31, commit on `michaels-label-calibration`)

`src/data_processing/michaels.py`:

| recording | was | now | source |
|---|---|---|---|
| FLY124 | 1.00839 | **1.00698** | 13-window global refit, non-twin rotors (g = 0.698 % ± 0.069) |
| FLY125 | 1.00690 | **1.00706** | same fit (g = 0.706 % ± 0.034) |

Rationale kept in the loader's block comment: the two refits agree to 0.008 pp
(one shared cause), the superseded pair was a 2–4 window per-rotor mean
contaminated by the twin rotors and disagreed by 0.15 pp, and the whole global
gain is degenerate with a sample-clock error — so it is a *label-for-this-audio*
correction, and, since a clock error is common to all four rotors, that same
degeneracy independently supports the global-only verdict.

`michaels-frames` republished at **`fdef818432e9`** (was `a7a951b94808`) and
re-pinned in `dload.lock`. Verified on the published frames: `rps` is
**bitwise-exactly** `(raw Motor:Speed RPM / 60) × scale` on both recordings,
`meta.rps_scale` carries the new value, and CSV completeness is unchanged
(230 columns → 212 numeric channels across 18 per-sensor-block Series + 14
bool/string Series; only `ConvertDatV3` and `4.2.1` absent, both all-empty).

### Re-validated on the corrected labels (job `python-b7e225`, 636 s, uni-cpu)

`run_sweep.py --post-shipped`, 13 cruise windows, no edge hits:

| recording | n | resid lag mean / RMS / max | resid value offset mean | resid value max | (old scales, `python-0445f6`) |
|---|---|---|---|---|---|
| FLY124 | 4 | −2.92 / 3.40 / 4.29 ms | **−0.054** rev/s | 0.120 | −0.178 rev/s |
| FLY125 | 9 | +1.21 / 3.22 / 8.04 ms | **−0.009** rev/s | 0.096 | +0.004 rev/s |

Timing is untouched and still closed (residual lag RMS at the 2.9 / 4.5 ms level
of the dilation fit's own residual; FLY124's per-window residuals
+0.03/−3.29/−4.11/−4.29 ms carry no monotone drift). **FLY124's value error fell
3.3×**, −0.178 → −0.054 rev/s, i.e. ~1/10 of the 0.56 rev/s the calibration
removes; FLY125 is unchanged within noise (+0.004 → −0.009).

The −0.054 rev/s leftover is slightly more than the +0.004 the refit projected,
and the reason is that the two estimators are not the same measurement: the
refit regresses per-rotor `prot` offsets on the **non-twin** rotors, whereas this
scan finds one global offset over **all four**, including the twin pair whose
individual estimates the audio cannot resolve (RBack's sd alone is 0.50 / 0.76
rev/s). Both numbers sit well inside the 0.175–0.213 rev/s per-window scatter,
so there is no residual structure left to chase — consistent with WP14's finding
that the refit residual already equals the estimator's own noise.

The WP13 "stale artefacts" table still applies verbatim — every entry in it now
also predates this second label change, and the two `michaels-frames` versions
differ by −0.14 pp on FLY124 / +0.016 pp on FLY125 (−0.11 / +0.013 rev/s at
cruise), i.e. small next to the 0.55–0.67 rev/s WP13 already moved.

## Work packages

- **WP0 — lab harness** `scripts/rps_refine_lab.py`: repo-ified
  trace_pipeline with configurable stage chains + a synthetic free-flight
  battery (N seeds × aggressiveness), per-stage PIT-MAE + per-rotor
  bias/shape decomposition, JSON out. Fast CPU iteration.
- **WP1 — diagnosis**: instrument `_freq_update` (which factor zeroes the
  step) and pi_kalman (per-iteration capture fractions, in-band offsets).
- **WP2 — aggressive pi_kalman**: sweep n_iter / band_hz schedule
  (wide→narrow) / k_caps / sigma_process / max_step / pair_mode on the
  synthetic battery; target MAE < 0.1 synthetic; transfer to fly124. Decide
  whether VK capture+refine survive at all or the chain becomes
  ladder → iterated pi_kalman.
- **WP3 — per-rotor decoupling**: individuate shapes on dregon_ramp
  (wide-band early pi_kalman rounds; per-rotor corridor Viterbi around
  c(t) if needed). Success = per-rotor pred-std tracks gt-std.
- **WP4 — protocol**: best chain on the fixed beatvk-valid-raw protocol,
  scoreboard row in `docs/experiments/beat-vk.md`.

Success bars: synthetic battery pooled PIT-MAE < 0.1 rev/s; fly124_cruise
well under 1.0 (mind the ~29 Hz telemetry jitter floor of GT — estimate it
by comparing raw vs smoothed telemetry before claiming a floor); dregon_ramp
per-rotor std ratio ≈ 1.
