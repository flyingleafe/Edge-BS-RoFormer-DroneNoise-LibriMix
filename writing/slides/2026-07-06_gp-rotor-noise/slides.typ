#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [Generated-Noise Augmentation Fails → a Faithful GP Rotor-Noise Model],
  subtitle: [A negative result, then a physically-structured response],
  author: [Dmitrii Mukhutdinov],
  date: [2026-07-06],
)

// ===========================================================================
// Part 1 — The negative result: generated-noise augmentation
// ===========================================================================

= The hypothesis

- *Task:* train an RPS-predictor (4 rotor speeds from noise) at ultra-low SNR.
  Real corpus is tiny: 6 DREGON + 2 Michael's flights (~10 min total).
- *Bet:* a *learned generative model of rotor harmonic noise*
  (PositionalHarmonicNoiseGen: RPS + array → multichannel noise) can synthesise
  *unlimited, perfectly-labelled* training variety → better RPS prediction.

#v(0.4em)
*Three prerequisite stages, all landed:*
+ E2 — the deep generator, trained on DREGON + Michael's, per-drone conditioning.
+ E3 — swapped DREGON split + random-phase + smoothness penalties.
+ E4 — live `GeneratedNoisePool` wired into the online-mix training stream,
  with the synthetic RPS trajectory as its exact label. Plumbing verified
  end-to-end (42 tests).

= The deep generator: real vs. generated

#grid(
  columns: (1fr, 1fr),
  gutter: 0.7em,
  figure(
    image("assets/noise_gen_spec_dregon.png", height: 75%),
    caption: [DREGON — low-BPF comb captured, weak mid-frequency harmonics.],
  ),
  figure(
    image("assets/noise_gen_spec_michaels.png", height: 75%),
    caption: [Michael's — per-drone conditioning works, but the filter-envelope
      fingerprint is fixed across trajectories.],
  ),
)

= E3 — smoothness sweep: a plateau, not a winner

#figure(
  image("assets/e3_smoothness_sweep.png", height: 70%),
  caption: [Best `harm_smooth = 1e-1` (val 5.3506), but the top three sit within
    $approx 0.008$ of the baseline (5.3554). Smoothness does *not* meaningfully
    move the raw spectral val loss.],
)

= E4 — the bet fails: augmentation *degrades* RPS prediction

#grid(
  columns: (1.5fr, 1fr),
  gutter: 0.8em,
  figure(
    image("assets/e4_aug_degradation.png", height: 80%),
    caption: [+1 generated `michaels` source ($approx 1/3$ of noise batches)
      degrades PIT MSE +27% (uni-GRU) and +26% (Transformer).],
  ),
  [
    *Why:*
    - Imperfect mid-freq harmonics + fixed filter-envelope → the predictor
      latches onto the generator's *fingerprint* as a shortcut.
    - RPS-trajectory diversity too narrow.
    - The transformer memorises: train $9.9 arrow.r 3.7$ while val PIT
      $10.6 arrow.r 43.6$ after epoch 9.
    - *Not* a plumbing failure (42 tests pass).
  ],
)

= Part I — take-away

*Generated-noise augmentation, as built, hurts RPS prediction* — a negative
result driven by exploitable generator artifacts + narrow trajectory
diversity.

#v(0.6em)
*Root cause:* the generator's freely-learned representation admits artifacts
the predictor exploits. A model whose *structure encodes the physics*
should resist this.

#v(0.6em)
*Next:* reimplement a *physically-structured* rotor-noise generator —
the Gaussian-Process model of Lee, Ko, Seshadri & Rauleder (JASA 2026).

// ===========================================================================
// Part 2 — The GP rotor-noise model
// ===========================================================================

= The paper

My supervisor attended the Quiet Drones 2026 conference and pointed me at:

#align(center)[
  *Lee, Ko, Seshadri & Rauleder.*
  *Bayesian ML framework for time-domain prediction of multirotor vehicle noise.*
  *JASA* 159(4), 3418–3435 (2026). DOI: 10.1121/10.0043469.
]

*A Gaussian Process that predicts a full 44.1 kHz time-domain acoustic
pressure signal* for a multirotor vehicle at unmeasured flight conditions and
observer locations — not a noise metric, the waveform itself — with calibrated
posterior uncertainty.

*Why this is the right response to Part I's failure:* a model whose tonal
structure is fixed physics has nothing for a downstream predictor to memorise.

= What problem does it solve

- *Most surrogate models* predict an integrated metric (OASPL, a loudness
  number, a few spectral bands). Fine for certification, *useless for auralisation*.
- *Psychoacoustic impact* (annoyance, tonality, roughness) needs the actual
  *waveform* $p(t)$ — you have to listen.
- *Predict the time-series, compute any metric later.*
- *Challenge:* a full-sample-rate GP over audio is $O(N^3)$ per segment; 44.1
  kHz × seconds is enormous. The paper's two design choices make it tractable:
  a *BPF-informed Fourier kernel* and a *broadband-as-likelihood* split.

= Training data: CONA (auralised, not recorded)

- *CONA framework:* Beddoes prescribed-wake + Blade Element Theory +
  Ffowcs Williams–Hawkings (tonal) + Brooks–Pope–Marcolini (broadband) ->
  virtual microphone signals across a grid of (position, $V_1$) pairs.
- *Scale for free:* thousands of perfectly-labelled training waveforms.
- *Example vehicle:* NASA 1-Pax eVTOL quadcopter, $V_1 in {6..10}$ m/s,
  256 ground mics, 44.1 kHz, 112 M data points.

*Two load-bearing pre-processing steps* make this dataset GP-ingestible.

= Pre-processing 1: phase alignment

- Rotor signature is quasi-periodic, but each simulated signal starts at an
  *arbitrary phase* (function of source position + collection window).
- Naive GP over raw samples → *averages phasors* across those arbitrary
  phases → sharp BPF peaks wash out.
- *Fix:* circularly shift every training signal so its first-BPF phase matches a
  reference (mic at $(-30, 0)$ m, $V_1 = 1$ m/s) to within 0.001 rad.
- Pure time-shift — preserves spectrum, only changes the absolute phase reference.
- *We learned the hard way* that omitting it costs ~10 dB at the first harmonic
  — essential, not optional.

= Pre-processing 2: DWT tonal / broadband split

Rotor noise = deterministic *tonal* (BPF harmonics) + stochastic *broadband*
(turbulent-interaction noise). A 4-level Daubechies-4 wavelet split partitions
every waveform:

- *Approximation coefficients* → low-pass-reconstructed *tonal* signal (the GP target).
- *Detail coefficients* → stochastic residual. Their std $sigma_b$ is measured per
  microphone and pushed into the *Gaussian likelihood* as a noise floor.
- They find off-diagonal entries of $bold(R)_b$ are essentially zero, so
  $bold(R)_b approx sigma_b^2 bold(I)$ — *per-mic independent noise*.

*Second load-bearing design choice:* broadband is *not* a structured kernel term,
it is the *likelihood residual*. At synthesis a sample
$epsilon tilde(N)(0, sigma_b^2)$ is added to the tonal prediction — Eq. (3):

$ p(t) = f(t; z) + epsilon, quad epsilon tilde(N)(0, sigma_b^2 bold(I)) $

= The BPF-informed Fourier kernel (Eq. 10–16)

The kernel: $k(z, z') = k_("spatial")((x, y), (x', y')) op k_("tonal")((V_1, t), (V_1', t'))$

- $k_("spatial")$ = Matérn-$5/2$ over 2-D mic position, separate ARD lengthscales
  $ell_x$, $ell_y$ (smoother than RBF; chosen for sharp near-flight-track features).
- $k_("tonal")$ is the novel piece. *Standard kernels (RBF, Matérn) model only
  local smoothness — they cannot represent periodicity.* The Spectral Mixture
  kernel can, but needs too many hyperparameters to span the audible spectrum.
  Instead the paper builds a *known periodic basis*:
  - the first 10 BPF harmonics of front + aft rotors — 20 frequencies, each
    $omega = k dot.op N_("blades") dot.op "RPS"_i$
  - assemble the Fourier design matrix
    $F = [1, sin(omega_1 t), cos(omega_1 t), ...]$.
  - frequencies are *physics-injected* (from CONA's RPM trim), not learned.
- *What is learned:* per-coefficient variance $kappa^2_j$ (prior on the $j$-th
  sine/cosine amplitude) + a Matérn-$5/2$ over $V_1$ so amplitudes/phases vary
  smoothly with flight speed.
- Posterior over Fourier coefficients $bold(w) = (mu, A_k, B_k)$ is
closed-form Gaussian (Eq. 15) — prediction is faster than sampling the
time-domain GP.

= Inference: stochastic variational GP (SVGP)

- Standard GP scales cubically with training points — 112 M points is hopeless.
- *SVGP* (Hensman et al. 2015) reduces cost to $O(N M^2)$, $M << N$:
  - 1000 inducing points, mini-batches of 10 000.
  - Predictive log-likelihood (Jankowiak 2020) instead of plain ELBO — better
    broadband-uncertainty calibration.
  - Training on 661 440 wavelet-compressed points takes a single A100.
  - Synthesis: 0.18 s per 44.1 kHz signal — six orders of magnitude faster than CFD/CAA.

*Three things the Fourier-basis design buys:*
+ Periodicity that ordinary Matérn/RBF cannot capture.
+ Physical interpretability of every hyperparameter (an amplitude per harmonic,
  a spatial lengthscale, a velocity lengthscale).
+ Closed-form posterior over Fourier coefficients — prediction is fast.

= Reimplementation + two refinements

#figure(
  image("assets/gp_overview.png", height: 65%),
  caption: [Pipeline: framing $arrow.r$ DWT split $arrow.r$ phase-align $arrow.r$ least-squares
    on the BPF-injected Fourier design $arrow.r$ SVGP over coefficients $arrow.r$ inverse
    synthesis + broadband-residual sampling.],
)

#v(0.3em)
*Adapted to our data:*
- Conditioning variable is *per-rotor RPS* $"RPS"_i(t)$, not the paper's scalar
  $V_1$ — the actual driver of each rotor's harmonic comb.
- Per-mic diagonal likelihood noise $sigma_b$ from the DWT-detail-std.

= Result — real vs. GP-generated spectra (Michael rec 1)

#figure(
  image("assets/gp_faithful_spectrum.png", height: 78%),
  caption: [Coeff RMSE = 0.10. The BPF comb is reproduced precisely up to
    ~1.7 kHz (H=24). A broadband-residual realisation fills the spectrum per
    Eq. (3). The 2–6 kHz band remains under-covered (H cap; raise $H$ to extend).],
)

= Honest gaps + in-flight work

*Where it still falls short:*
- Spectral coverage capped at $~H dot "BPF"$ — H=24 reaches ~1.7 kHz; real
  noise extends to ~3 kHz. Needs $H approx 60$ → ~8 GiB peak RAM. Streaming
  refactor of the Fourier design is the correct long-term fix.
- Phase uses a circular-mean template; refine for higher auditory realism.

*In progress:*
+ *Train one GP per drone* (`michaels`, `dregon`) on the *same* swapped training
  split the deep generator used — bespoke driver
  `src/experiments/gp_rotor_noise/train_dregon_michaels.py`.
+ *Listening-comparison notebook* extending
  `notebooks/noise_gen_real_vs_generated.ipynb` with GP rows: same real RPS
  trajectory → deep + GP + real audio, user-chosen mic.
+ Both run on the GPU once available; laptop config (H=24, ~8 GiB, ~46 min/drone) measured.

= Cross-thread scoreboard

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    [*Model*], [*Best val*], [*R²*], [*Status*],
    [`PositionalHarmonicNoiseGen` (E3)], [$5.3554$], [—], [trained (deep)],
    [`+ aug → RPS uni-GRU128`], [$9.29$ (PIT)], [0.791], [degraded +27%],
    [`+ aug → RPS transformer`], [$10.63$ (PIT)], [0.762], [degraded +26%],
    [`GP (Lee et al. construct)`], [coeff RMSE $0.10$], [—], [WIP, on DREGON+Michaels],
  ),
  caption: [Deep generator + augmentation is settled negative; the GP model is
    the active response to the failure mode its structure was chosen to avoid.],
)

= Story so far

- *Deep generator + augmentation = negative result.* The predictor exploits
  the generator's artifacts; a freely-learned representation is the wrong
  primitive for augmenting a small-data task.
- *GP rotor-noise model = active response.* Reimplemented from Lee et al. (JASA 2026):
  physics-injected Fourier design + phase alignment + DWT broadband-as-residual.
  Coefficient RMSE $4times$ lower; BPF peaks reproduced exactly.
- *Open question the listening test will answer:* does the physically-structured
  generator produce noise that *sounds* enough like the real recording to help
  (or at least not hurt) the RPS predictor — controlling for trajectory diversity?