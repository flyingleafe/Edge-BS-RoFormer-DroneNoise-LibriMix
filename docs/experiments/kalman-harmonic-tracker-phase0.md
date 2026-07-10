# Kalman harmonic tracker — Phase 0 (bet killed at K2)

**Status:** done · 2026-07-10 (single day; Phase-0 budget was 3 days) ·
bet card: [`bets/kalman-harmonic-tracker.md`](./bets/kalman-harmonic-tracker.md) ·
data: `results/kalman_harmonic_phase0{,_joint}/` ·
code: `src/experiments/kalman_harmonic/`

## Motivation

Test whether a causal, streaming, per-harmonic complex Kalman filter driven
by measured RPS (a "complex-KLA" filter — diagonal information filter in
demodulated coordinates) can replace the framed `lstsq_VP_transform` for
harmonic noise subtraction. Two hypotheses: **H1** — at oracle RPS it matches
lstsq within 1 dB SI-SDR (sanity); **H2** — under realistic slow RPS error it
degrades *more gracefully* than lstsq, because process noise `p` absorbs
phase drift (the actual bet; would de-risk Bet 1 / pseudo-RPS). Long-term
intent: make the filter a learnable KLA-style attention layer in the deep
model.

Setup (synthetic-first per the MVP): 4 rotors at ~85–95 RPS with slow wander,
25 harmonics with ±30 % AM at 0.5–1.5 Hz, speech proxy mixed at −10 dB, 4 s
at 16 kHz. RPS error model: mean-zero multiplicative OU drift (τ = 0.5 s,
relative std σ ∈ {0, 0.2, 0.5, 1, 2, 5} %). SI-SDR scored past a 0.5 s
warm-up and the lstsq frame tail, same region for all methods. Runner:
`python -m experiments.kalman_harmonic.phase0 [--joint]`.

## Results

| σ drift | unprocessed | tracker, fixed p | tracker, matched p | lstsq_VP |
|---|---|---|---|---|
| 0 % (oracle) | −9.59 | +3.27 | +4.62 | **+5.60** |
| 0.2 % | −9.59 | −8.12 | −2.83 | **+2.02** |
| 0.5 % | −9.59 | −9.44 | −5.45 | **−4.79** |
| 1 % | −9.59 | −9.41 | −7.22 | −6.79 |
| 2–5 % | −9.59 | ≈ −9.5 | −7.4 … −7.8 | −7.3 … −8.0 |

(dB SI-SDR; "tracker" = joint per-order variant; "matched p" = best
(p_base, p_h2) per σ from the grid — the filter given its error spec.)

- **H1 / K1 — pass, only via the joint update.** The as-drafted *diagonal*
  filter fails oracle parity by 3.5 dB (+2.07 vs +5.60): the four
  near-coincident rotor combs fight over shared energy, exactly the failure
  the card predicted. K1's prescribed escape — a per-order R×R joint update
  (`kalman_harmonic_track_joint`: scalar sample observed through the 4-rotor
  steering vector, rank-1 Sherman–Morrison step; the diagonal filter is this
  minus the off-diagonal covariance) — recovers to within 0.98 dB, causal and
  lstsq-free, and its optimal bandwidth is ~15× wider (the joint solve
  removes the speech-double-counting penalty of bandwidth).
- **H2 / K2 — fail.** The tracker collapses *earlier* than lstsq, not later:
  at σ = 0.2 % it retains 48 % of its oracle gain vs lstsq's 76 %; at 1 %
  both are dead (17 % vs 18 %; H2 required ≥ 70 %). The matched tracker edges
  past lstsq only at σ ≥ 2 %, where both deliver ~2 dB of gain at −10 dB
  SNR — useless territory.
- **The p-knob works but cannot win.** Optimum process noise grows with σ and
  the physically-motivated h²-scaled form beats uniform widening by ~3 dB at
  σ = 0.2 % — yet stays ≥ 5 dB below lstsq. Mechanism (see Conclusion):
  widening amplitude bandwidth is the wrong lever against a systematic
  rotation.
- **Tuning traps found (recorded for any successor):** initial precision must
  be λ₀ = K/(2·var(wav)) — a flat prior lets every channel claim the whole
  signal at t=0, a 1/t over-subtraction transient worth −30 dB on short
  clips; and γ must sit well below the tracking bandwidth k·sr, else the
  amplitude estimate is biased low by k/(1−ā(1−k)) (the drafted γ=10/s gave
  ~0.1× amplitudes).

## Conclusion

Killed per K2, and the failure is structural, not tuning. Any coherent
canceller obeys *span × frequency-error ≲ 1 rad per harmonic* (arrows curl),
while speech leakage scales as 1/span; drift therefore caps the integration
span per harmonic at ~1/(2π·h·f₀·σ) — 18 ms at h=25 for σ=0.2 % — versus the
tracker's speech-optimal ~1 s stare. Within that squeeze the two methods sit
asymmetrically: the causal tracker's window lies strictly in the past
(permanent phase lag Δω/k under sustained offset — 1st-order-PLL result) and
its accumulated phase error is load-bearing state; the framed refit's window
straddles what it reconstructs (spread only, no lag) and re-anchors every
hop, discarding accumulated phase for free. Mean-zero-ness of the RPS error
does not help: what matters is that the error holds its sign longer than the
integration span, and the *phase* error (integral of frequency error) random-
walks regardless. White per-sample jitter integrates away harmlessly — an
error model that would have flattered the tracker and tested nothing.

### Potential mitigations (for any revival — none tested beyond #4/#5)

Ordered by expected leverage:

1. **Comb-coherent drift tracking (one frequency state per rotor).** The
   unexploited structure: RPS error is a *single scalar per rotor*, so phase
   error at harmonic h is exactly h× the fundamental's. Low harmonics stay
   locked with sub-Hz bandwidth even at σ=0.2 % — estimate the drift there
   (where SNR per Hz is best) and correct the demodulation phase of the
   whole comb. Turns 25 per-harmonic bandwidth problems into one per-rotor
   frequency-tracking problem; the high harmonics get drift correction
   without paying any leakage.
2. **Second-order loop / frequency-state augmentation per channel.** A
   type-2 loop has *zero* steady-state phase error against a constant
   frequency offset; the amplitude filter can then return to its
   speech-optimal long span. Nonlinear (EKF/PLL discriminator), so it left
   Phase-0's linear-Gaussian scope — but in the KLA-layer parameterization it
   is naturally learnable: predict corrections to the token-dependent complex
   transition a_t (rotation), not just the gates (p_t, q_t). This is the
   Phase-1 reframe: without learned rotation correction, the layer inherits
   the fragility measured here at 0.2 % RPS error — far below realistic
   tacho/pseudo-RPS accuracy.
3. **Fixed-lag smoothing (~100 ms lookahead).** Strict causality is the
   premise that failed; a fixed-lag smoother converts the tracker's full-span
   lag into an lstsq-like centered spread while staying streaming. Layer
   analog: chunked bidirectional scan. Cheap to test on the existing code.
4. **Per-order joint R×R update** — implemented, mandatory (+2.4 dB at
   oracle, K1 hinges on it). Keep in any successor; rank-1 information
   updates keep it scan-parallelizable.
5. **h²-scaled process noise** — implemented; directionally correct (~3 dB
   under drift) but insufficient alone. Keep as the error-spec knob, not as
   the robustness mechanism.
6. **Periodic re-anchoring (hybrid).** Graft lstsq's amnesia onto the
   recursive filter: block-wise phase re-estimation every ~100 ms while the
   amplitude magnitude keeps its long memory. Crude version of #2; useful as
   a control experiment to separate "lag" from "leakage" contributions.

Known limitations of this record: speech proxy (not LibriSpeech), single
seed/clip, OU error model (real control signals are intermittent — the
project's intermittent RPS model, or a pure constant bias (clock skew,
harsher for the tracker), would be the next error models to test), and the
DREGON-chunk hook from the MVP was never exercised — Phase 0 concluded on
synthetic evidence alone since K2 was already decisive there.
