# Fitting the wind channel by likelihood instead of by realization

## The defect

`conf/loss/multiscale_stft.yaml` compares **one realization** of the generator
against **one realization** of the recording. For the harmonic bank that is
correct — it is deterministic given the shaft phase. For the broadband residual
and the wind channel it is not: a gust is a random process, and the recording
contains one draw of it that no model can reproduce.

Two consequences, both measured rather than argued
(`tests/losses/test_spectral_likelihood.py`):

1. **The minimizer is biased.** For a bin whose content is circular complex
   Gaussian, `|X|` is Rayleigh and the L1 magnitude minimizer is its *median*.
   The fitted power is low by a factor `ln 2` — a systematic **−1.6 dB** on any
   purely stochastic component, at any capacity or training length. The test
   pins this at `ratio == ln 2` to 3%.
2. **The gradient is noise.** Each draw pulls the wind parameters a different
   way. Only the (biased) mean survives averaging, so the stochastic branch
   learns far more slowly than the deterministic one it competes with — and
   competes it does, since both are summed before the loss sees them.

This is sufficient to explain why `gen_v3_wind` was the worst of the four
corrected-geometry variants (free-flight mrstft 3.44 against v1's 5.22) despite
its physics passing a pre-training de-risk with Spearman 0.92.

## The fix

Stop predicting a realization; predict a **distribution**. Per STFT bin, with
the coherent field's complex amplitude `mu` and total incoherent power `sigma2`:

    X = mu * exp(j theta) + CN(0, sigma2)

The absolute phase `theta` is not identifiable — the generator does not know the
recording's initial rotor phases, and per-harmonic phase noise decoheres them
anyway (coupled-VK draft: `theta_k(t) = k phi(t) + b_k(t) + psi_k` with `b_k`
Brownian of rate `q_k`). Marginalizing `theta` under a uniform prior turns the
complex-Gaussian likelihood into a **Rice** likelihood on the magnitude:

    NLL = log(sigma2) + (r - a)^2 / sigma2 - log I0e(2 r a / sigma2)

Both limits are the textbook estimator, which is what makes it trustworthy:

- `a = 0` (pure-noise bin) → `log sigma2 + r^2/sigma2`, the **Whittle**
  likelihood, whose minimizer is `E|X|^2`. Unbiased, unlike the loss it replaces.
- `sigma2 -> 0` (pure tone) → `(r - a)^2`, ordinary magnitude matching, so
  nothing is lost where the old loss was already right.

The same machinery covers partial coherence: a harmonic with coherence `gamma`
contributes mean `gamma*a` and extra variance `(1 - gamma^2) a^2`, conserving
power (`losses.spectral_likelihood.split_coherence`). That is the generator-side
counterpart of the decoherence budget the VK work measures.

**Nothing is sampled during training.** The models gained a `spectral_stats()`
path that returns the coherent mean and the analytic spectral envelope of every
stochastic branch. The wind's gust is marginalized out by Gauss–Hermite
quadrature over its stationary log-normal distribution
(`WindTransduction.expected_mags`), since the level is a nonlinear function of
flow speed and the response at the *mean* gust is not the mean response.

### Making it actually train

The objective as first written diverged to NaN within two epochs on **every**
arm, including the no-wind one. Three separate causes, all found by tracing:

1. **A NaN gradient through an exact zero.** `spectral_stats` returned a
   magnitude (`power.sqrt()`) and the loss squared it back. That round trip is
   the identity analytically, but autograd walks it stepwise: `d(sqrt)/dx` is
   infinite at 0 while `d(x^2)/dx` is 0, so `inf * 0 = NaN` in every bin the
   broadband branch predicts silent — about 12% of them. The loss *value* stayed
   finite, which is why it surfaced only as a NaN gradient on the noise branch
   while the coherent branch's gradients were clean. The interface is now
   **power** (`noise_psd`) end to end, so the pair never exists.
2. **Quadratic blowup at initialization.** From a random init the generator
   emits `|audio| ~ 80` against audio of ~0.05, so `(r-a)^2/sigma2` is ~1e12 at
   step 0 and one Adam step destroys the model. No variance floor fixes this
   (2.6e12 at `floor_rel=1e-6`, still 2.6e7 at 1e-1) because the *mean* is what
   is wrong. Every arm therefore warm-starts from the magnitude-trained
   `gen_v1_recal`, which puts the ratio in the hundreds; the wind model gained a
   load-time key remap (`generator.*` -> `generator.coherent.*`) so it can
   consume that checkpoint at all. Measured after the fix: 48 steps, loss
   6.53 -> 4.82, no divergence.
3. **Gradient imbalance**, addressed with beta-NLL (Seitzer et al.): each bin's
   term is scaled by a detached `sigma2**beta` so low-variance bins stop
   dominating. **It is not a proper scoring rule** — it rescales the loss value,
   preserving the optimum only for a per-bin-flexible sigma and shifting the
   argmin when sigma is shared. Training uses `beta=0.5`; the eval's scoring
   rule and the estimator tests pin `beta=0`.

A side effect worth stating: because every arm now starts from the same
magnitude-trained weights, these are **fine-tunes**, not from-scratch runs. That
is a cleaner comparison for isolating the objective, but it means the numbers
answer "does switching objective improve this model?" rather than "is this
objective better from scratch?".

### Two bugs found on the way

- `ou_envelope` started the OU log-amplitude at `x = 0` rather than from its
  stationary distribution, biasing the opening frames toward `exp(-sigma^2/2)`
  and — more seriously — making the sampler's marginal disagree with the
  stationary marginal the likelihood integrates over. Fixed.
- `spectral_stats` originally resampled the coherent generator's noise envelope
  onto the (coarser) wind grid before summing powers, which discarded resolution
  and lost power. Both envelopes now go onto the finer grid.

## The second defect: identifiability

Fixing the objective is necessary but **not sufficient**. What distinguishes the
wind from the coherent generator's own broadband branch is its *spatial* law: a
wake-gated, per-microphone-incoherent field, against one propagated from the
rotors by `1/r`. Their spectral shapes overlap; only the across-microphone
pattern separates them.

All generator training in the Hydra framework has been **single-microphone**
since the migration (`NoiseGenFrameDataset` took channel 0 and row 0 of the
geometry, and the constructor rejected anything else). At `M = 1` the wind's
only distinguishing feature is invisible: it degenerates into another broadband
shape competing against a far more flexible learned 60-band filter, and it
loses. That is a second, independent reason `v3_wind` could not have worked —
and no amount of optimizer fixing would have addressed it.

`NoiseRPSDataset` gained `channel_policy="all"` and the frame adapter now
renders every microphone, restoring the native multi-observer training the
pre-Hydra trainer did.

## Arms

A 2x2 over the two conditions the channel needs, all under the likelihood, plus
the magnitude-loss baseline:

| Arm | Observers | Wind | Isolates |
|---|---|---|---|
| `gen_v1_recal` | 1 | no | baseline, magnitude loss (trained: **9.149**) |
| `gen_w1_lik_nowind` | 1 | no | the objective alone (vs `v1_recal`) |
| `gen_w2_lik_wind` | 1 | yes | wind where it is *not* identifiable |
| `gen_w3_lik_nowind_mm` | 8 | no | multi-observer alone |
| `gen_w4_lik_wind_mm` | 8 | yes | **the target arm** |

The controls are not optional. The likelihood changes how the generator's *own*
broadband residual is fitted, and rendering 8 microphones changes the fit again,
so without `W1` and `W3` a `W4` improvement could not be attributed to the wind
channel. The design also makes a falsifiable secondary prediction: `W2 - W1`
should be **negligible** (wind unidentifiable at `M = 1`) while `W4 - W3` should
not. If `W2` already captured the gain, the identifiability argument is wrong.

## The third defect: the metric itself

`mrstft` — the monitor every earlier generator round was ranked by — compares
one *realization* of the generator against the recording. A magnitude distance
systematically prefers an **under-dispersed** model. Measured on a stochastic
bin of unit power, mean `|.|` distance to the truth:

| prediction | distance |
|---|---|
| deterministic at the Rayleigh **median** (the −1.6 dB biased optimum) | **0.370** |
| deterministic at the correct RMS | 0.393 |
| correctly calibrated, actually sampling | 0.520 |

So the metric ranks the biased model *first* and the correct model *last*. A
model that fixes the stochastic-power bias can look like a regression under
`mrstft`, and the historical verdict "v3_wind is worst at 3.44" was partly
measuring this rather than fidelity.

`scripts/eval_noise_gen_variants.py` therefore now reports **both**: `mrstft`
for continuity with the earlier reports, and the held-out Rice/Whittle `nll` —
a proper scoring rule, minimized by the true distribution — computed
identically for every variant through `spectral_stats` (every generator has it,
including the magnitude-trained ones). **When the two disagree, `nll` is the
one to believe.**

## Falsifiable expectation

From the pre-training de-risk: the wake gate predicts the DREGON per-mic
low-band floor at Spearman 0.92 against 0.74 for a plain `1/r` control, and
Michael's array sits **out** of the wake (max predicted exposure 0.006 m/s).
Therefore, scored per drone:

- **DREGON** — mics inside the rotor wake, so the channel has something real to
  explain → `W2` should improve on `W1`.
- **Michael's** — array out of the wake, so the channel should be inert →
  `W2` should be **unchanged** against `W1`, not worse.

A wind arm that helps DREGON *at Michael's expense* would mean the channel is
acting as free capacity rather than as the physics it claims to be. The
per-drone split in `scripts/eval_noise_gen_variants.py` is the readout that
distinguishes those two outcomes; a pooled number cannot.

### The expectation is structural, not hopeful

Measured on the real published geometry at **initialization**, with no fitting
(`WindWakeChannel` defaults, RPS 80 rev/s):

| | mic z | flow speed U | wind power | mic-to-mic spread |
|---|---|---|---|---|
| DREGON | ±0.041 (rotors at 0.192) | 0.16–0.35 m/s | 5.0e−4 | **9.5×** |
| Michael's | +0.33 (rotors at 0.0) | 0.0002–0.014 m/s | 6.7e−8 | — |

The channel is **~7500× weaker on Michael's**, whose ring sits above the rotor
plane and therefore upstream of a `−z` downwash. So "inert on Michael's, active
on DREGON" is not something training has to discover — it is forced by the wake
geometry, and the only open question was whether the *optimizer* could fit the
channel where it is active. That is what this objective changes.

Note the wake is a column under each **rotor**, not under the airframe centre:
a microphone at the centre of a quadrotor sees almost nothing. DREGON's
discrimination therefore comes from lateral falloff between mics, matching the
de-risk finding that the win over a `1/r` control is carried by the
perpendicular term.

## Conclusion

_Pending runs._
