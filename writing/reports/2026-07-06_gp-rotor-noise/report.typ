#import "/writing/templates/typst/report.typ": report, author-meta

#let fig(path, caption, width: 100%) = figure(image(path, width: width), caption: caption)

#show: report.with(
  title: [Generated-Noise Augmentation (Negative Result) and a Faithful GP Rotor-Noise Model],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    Two threads are reported here. *First:* a completed, negative-result experiment on
    using a trained generative drone-noise model (`PositionalHarmonicNoiseGen`) to
    *augment* RPS-predictor training — the augmentation degrades held-out PIT MSE by
    +27% on uni-GRU and +26% on Transformer, driven by exploitable generator
    artifacts plus narrow RPS-trajectory diversity, not by the plumbing (verified
    end-to-end). *Second:* a work-in-progress Gaussian-Process rotor-noise model
    reimplemented from Lee, Ko, Seshadri & Rauleder (JASA 2026), with
    physics-injected BPF Fourier design, phase alignment, and a DWT tonal/broadband
    split; on Michael's held-out recordings it reproduces the real per-rotor BPF
    peaks (149/146/156/160 Hz) with a coefficient RMSE of 0.10. Both the deep
    generator and the GP are being trained on identical swapped-split
    DREGON+Michael's data for a same-RPS listening comparison that will close
    this work.
  ],
  keywords: ("generated-noise augmentation", "RPS prediction", "negative result",
             "Gaussian process", "rotor noise", "Bayesian", "Fourier kernel"),
)

#set heading(numbering: "1.")

= Part I — Generated-noise augmentation for RPS prediction (negative result)

#block(fill: luma(240), inset: 8pt, radius: 4pt)[
  _This section is the polished write-up of the experiment log
  `docs/experiments/noise-generation-augmentation.md` (status: done — negative
  result, dates 2026-06-26 to 2026-07-03)._
]

== Motivation

RPS-predictor training consumes a tiny real corpus (six DREGON in-flight
recordings + Michael's two DJI flights, about 10 min total) mixed online with
LibriSpeech at ultra-low SNR. The hypothesis (from the 2026-06-30 supervisor
slides) was that a #emph[learned generative model of rotor harmonic noise] —
the inverse of RPS prediction: given RPS + array geometry, produce multichannel
noise (see @noise_gen_diagram) — could synthesise unlimited, perfectly-labelled
training variety and thereby improve RPS prediction, especially generalisation
to unseen trajectories/drones.

The bet is natural precisely because the inverse task is well-defined: real
flight recordings are RPS-instrumented, so the same data that trains the forward
(RPS -> noise) generator also labels its outputs.

#figure(
  image("assets/noise_gen_diagram.png", width: 70%),
  caption: [The `PositionalHarmonicNoiseGen` — per-rotor harmonic oscillator +
    filtered-noise emitter, differentiable propagation to 8 mics, per-drone
    conditioning codebook.],
) <noise_gen_diagram>

== Implementation (three prerequisite stages)

- #emph[Generator + online pipeline] (E2, 2026-06-26). `PositionalHarmonicNoiseGen`
  — single-rotor harmonic oscillator + filtered-noise emitter with a
  differentiable 1/r attenuation + fractional-delay propagation to 8 mics,
  trained jointly on two drones with per-drone conditioning, streaming DREGON
  `in_flight_noise` + Michael's `FLY125`/`FLY124`.

- #emph[Swapped split + random phase + smoothness] (E3, 2026-07-01). The
  original DREGON split was backwards — trained on 1 room, validated on 5 —
  corrected to train on room2 (5 recs) + FLY125, validate on room1 + FLY124.
  Added random-per-harmonic initial phase at train (zero at eval) and opt-in
  squared-2nd-difference smoothness penalties on harmonic-amplitude curves and
  diffuse-noise filter shape, targeting DREGON's weak mid-frequency harmonics.

- #emph[RPS-training augmentation wiring] (E4, 2026-07-02). A `kind: generated`
  source (`GeneratedNoisePool`): a spawned producer owns the CUDA context for
  a frozen noise-gen checkpoint and renders into a shared-memory ring buffer
  with a seqlock, read lock-free by `DataLoader` workers. Each chunk's
  synthetic RPS trajectory is its exact label. A/B config pair: baseline vs.
  treatment (+ one generated `michaels` source). Plumbing verified by
  `tests/data_processing/test_generated_noise.py` (42 pass).

== Results

=== E3 — noise-gen smoothness sweep (9 runs)

A 1-D sweep over `harm_smooth_weight` and `noise_smooth_weight` (orthogonal
regularisers, swept independently). Harmonic smoothness peaks at $1 / 10$ —
best spectral val `5.3506` — but the top three are within about 0.008 of each
other and *within the noise level* of the no-smoothness baseline (5.3554).
Noise smoothness actively hurts at $1 / 100$ (5.60) and only helps at large
weights ($> 1$).

#fig("assets/e3_smoothness_sweep.png", [E3 — noise-gen smoothness sweep. Best
  `harm_smooth = 1e-1`, but the top three sit within $approx 0.008$ of the
  baseline, i.e. a smoothness plateau, not a clear winner.])

The smoothness penalties do *not* meaningfully move the raw spectral val loss,
but they may yield qualitatively cleaner harmonic/noise *components* — an open
analysis. Checkpoints at `results/noise_gen_sweep/{baseline,harm_*,noise_*}`.

=== E4 — generated-noise augmentation (11 Slurm jobs, 2 usable)

Augmenting the RPS online-mix stream with a live generated `michaels` source
(approximately one third of noise batches, using the no-smoothness baseline
checkpoint) *degrades* RPS prediction versus the no-generator online-mix
baseline:

#table(
  columns: (1.6fr, 1fr, 1fr, 0.8fr),
  align: (left, center, center, center),
  [*Model*], [*Online-mix baseline*], [*+ generated noise*], [*Delta PIT MSE*],
  [`simple_conv_v2_uni_gru128`], [7.33 (R² 0.822)], [*9.29* (R² 0.791)], [*+27%*],
  [`simple_conv_v2_transformer`], [8.46 (R² 0.808)], [*10.63* (R² 0.762)], [*+26%*],
)

#fig("assets/e4_aug_degradation.png", [E4 — generated-noise augmentation
  degrades RPS prediction. Baseline bars are the online-mix real-noise run; red
  bars add a live generated `michaels` noise source ($approx 1 / 3$ of noise
  batches). The transformer shows textbook overfitting: train MSE
  $9.9 -> 3.7$ while val PIT $10.6 -> 43.6$ after epoch 9.])

Only 2 of 11 jobs produced usable checkpoints; the rest died on NaN
divergence, V100 OOM (the producer shares VRAM; batch $<= 8$), or
broken-GPU/multiprocessing node issues. The *transformer* shows textbook
overfitting — global attention lets it memorise the narrow set of synthetic
RPS trajectories and the generator's spectral fingerprint, pivoting to
generator artifacts once the real harmonic structure is exhausted. The causal
uni-GRU is regularised by its frame-by-frame constraint and degrades less, but
still lands below baseline.

=== Why it fails

- (1) #emph[Generator quality is the bottleneck.] The baseline checkpoint has
  imperfect mid-frequency harmonics and a fixed filter-envelope the RPS model
  latches onto as a shortcut feature. Spectrograms of the trained generator
  (@noise_gen_spec_dregon, @noise_gen_spec_michaels) make this visible against
  the real recordings.
- (2) RPS-trajectory diversity is narrow (OU + Poisson manoeuvres).
- (3) The online mixer diversifies only the acoustic dressing (speech, SNR),
  not the RPS curves themselves.

#figure(
  image("assets/noise_gen_spec_dregon.png", width: 80%),
  caption: [Positional harmonic generator — real vs. generated DREGON noise.
    The generator captures the low-BPF comb but imperfectly renders the weak
    mid-frequency harmonics the downstream predictor exploits as a shortcut.],
) <noise_gen_spec_dregon>

#figure(
  image("assets/noise_gen_spec_michaels.png", width: 80%),
  caption: [Same generator — Michael's DJI Matrice 100. Per-drone conditioning
    reproduces the rotor BPF comb, but the filter-envelope fingerprint is fixed
    across trajectories.],
) <noise_gen_spec_michaels>

== Conclusion (Part I)

Generated-noise augmentation, *as built*, hurts RPS prediction — a negative
result driven by exploitable generator artifacts plus insufficient
RPS-trajectory diversity, not by the plumbing (verified end-to-end). This
motivates Part II: a model whose *structure* encodes the physics (rather than
learning it from data alone) is less likely to expose exploitable artifacts.

= Part II — A GP rotor-noise model (work in progress)

== The paper

This thread started from a paper my supervisor recommended after attending the
Quiet Drones 2026 conference:

#align(center)[
  #emph[Lee, Ko, Seshadri & Rauleder, "Bayesian machine learning framework
  for time-domain prediction of multirotor vehicle noise"], #emph[JASA]
  159(4), 3418–3435 (2026). DOI: 10.1121/10.0043469.
]

The paper builds a *Gaussian Process* that predicts a full 44.1 kHz
*time-domain acoustic pressure signal* for a multirotor vehicle at unmeasured
flight conditions and observer locations — not a noise *metric*, the waveform
itself — with calibrated posterior uncertainty. It is the closest external
work to what Part I's negative result motivates: a model whose tonal structure
is fixed physics, so a downstream predictor has nothing to memorise. What
follows is an educational walkthrough of how it works.

=== The problem they solve

Most aeroacoustic surrogate models predict an integrated noise metric (OASPL,
a loudness number, a few spectral bands). That is fine for certification-style
assessment, but #emph[useless for auralisation]: to judge psychoacoustic impact
(annoyance, tonality, roughness) you need to listen, which means you need the
*waveform* $p(t)$ at every observer — not a scalar. The paper's opening move
is effectively: predict the time-series, and every metric can be computed later.

The challenge is tractability. A full-sample-rate GP over audio is $O(N^3)$
per segment — 44.1 kHz × seconds is enormous, and naively GP-fitting that is
hopeless. The paper's contributions are the two design choices that make it
tractable and interpretable: a *BPF-informed Fourier kernel* and a
*broadband-as-likelihood* decomposition.

=== The training data (CONA)

The training set is not recordings but *auralised* noise: a low-fidelity
physically-grounded simulator (the CONA framework — Beddoes prescribed-wake +
Blade Element Theory + Ffowcs Williams–Hawkings for tonal noise; Brooks–Pope–
Marcolini for broadband) generates virtual microphone signals across a grid of
flight velocities and observer positions. The advantage is scale — you can
cheaply synthesise data for thousands of (position, $V_1$) pairs, which becomes
the GP's training corpus. The example vehicle is a quadcopter NASA 1-Pax eVTOL:
forward flight at $V_1 in {6..10}$ m/s, 256 virtual microphones on the ground,
44.1 kHz, totalling 112 M data points.

Two pre-processing steps make this dataset GP-ingestible and both are
load-bearing.

=== Pre-processing 1 — phase alignment

A rotor's acoustic signature is quasi-periodic, but each simulated signal
starts at an arbitrary phase (a function of the source position and the
collection window). A naive GP over raw samples would average phasors across
those arbitrary phases and the sharp BPF peaks would wash out. The paper
therefore *circularly shifts* every training signal so its first-BPF phase
matches a reference (the mic at $(-30, 0)$ m, $V_1 = 1$ m/s) to within 0.001
rad. This is a pure time-shift — it preserves the spectrum and only changes
the absolute phase reference; without it the GP averages phasors across
arbitrary phases and the sharp BPF peaks wash out.

=== Pre-processing 2 — DWT tonal / broadband split

The rotor-noise signal is a sum of two physically distinct components:
*tonal* (deterministic, BPF harmonics) and *broadband* (stochastic,
turbulent-interaction noise). The paper uses a 4-level Daubechies-4 wavelet
decomposition to partition every waveform: the #emph[approximation
coefficients] are a low-pass-reconstructed tonal signal, and the
#emph[detail coefficients] give the stochastic residual. Two decisions follow.

First, the GP only has to model the tonal part — the broadband is not a
structured function you can fit, it is noise. So the broadband std $sigma_b$,
measured per microphone from the detail coefficients, is pushed into the
#emph[Gaussian likelihood] as a noise floor. They find the off-diagonal entries
of $bold(R)_b$ are essentially zero, so $bold(R)_b approx sigma_b^2 bold(I)$ —
per-microphone independent noise. This is the second of the two load-bearing
design choices: #emph[broadband is not a structured kernel term, it is the
likelihood residual.]

Second, at synthesis a sample $epsilon tilde(N)(0, sigma_b^2)$ is drawn and
added to the tonal prediction. The whole waveform is

$ p(t) = f(t; z) + epsilon, quad epsilon tilde(N)(0, sigma_b^2 bold(I)), $

where $f$ is the GP-predicted tonal posterior mean and $epsilon$ carries the
broadband — Eq. (3) of the paper.

=== The BPF-informed Fourier kernel (Eq. 10–16)

This is the heart of the method and its most elegant piece. The kernel is

$ k(z, z') = k_("spatial")((x, y), (x', y')) op k_("tonal")((V_1, t), (V_1', t')) $

where $op$ is the element-wise (Hadamard) product. Both factors have a
physical meaning.

$k_("spatial")$ is a Matérn-$5 / 2$ kernel over the two-dimensional microphone
position, with separate ARD lengthscales $ell_x$, $ell_y$. The Matérn-$5 / 2$
is smoother than the squared-exponential and was chosen for better behaviour
on near-discontinuous features (auralization signals vary sharply near the
flight track). Product along each direction → a 2-D Euclidean Matérn over
$(x, y)$.

$k_("tonal")$ is the novel piece. Standard GP kernels (RBF, Matérn) model only
local smoothness — they cannot represent periodicity. The Spectral Mixture
kernel (Wilson & Adams, 2013) #emph[can], but the paper argues it would need
too many hyperparameters to span the audible spectrum. Instead the paper
builds a #emph[known periodic basis]: for a 4-rotor vehicle, take the first 10
BPF harmonics of the front and aft rotors — 20 frequencies, each of the form
$omega = k dot.op N_("blades") dot.op "RPS"_i$ — and assemble the Fourier
design matrix

$ F = [1, sin(omega_1 t), cos(omega_1 t), ..., sin(omega_n t), cos(omega_n t)] $

These frequencies are *not learned*. They are injected from CONA's trim
solves (the rotor RPM at each flight velocity is known). What #emph[is]
learned is a per-coefficient variance $kappa^2_j$ (one per Fourier column,
assembled into a diagonal matrix $D = "diag"(kappa^2_1, ..., kappa^2_(2n+1))$
that acts as the prior on the $j$-th sine/cosine amplitude), plus a
Matérn-$5 / 2$ over $V_1$ that lets the amplitudes and phases vary smoothly
with flight speed. The tonal kernel is then

$ k_("tonal") = F^T D_s(t) D_s(t') F' dot.op k_("Matérn 5/2")(V_1, V_1') $

— a linear-model-coregionalisation kernel whose posterior over the Fourier
coefficient vector $bold(w) = (mu, A_k, B_k)$ is closed-form Gaussian
(Eq. 15). Three things this design buys: (i) periodicity that ordinary
Matérn/RBF cannot capture; (ii) physical interpretability of every
hyperparameter (an amplitude per harmonic, a spatial lengthscale, a velocity
lengthscale); (iii) closed-form posterior over the Fourier coefficients,
which makes prediction faster than sampling the time-domain GP directly.

=== Inference: SVGP

A standard GP scales cubically with training points. The paper uses a
stochastic variational GP (SVGP, Hensman et al. 2015) — 1000 inducing points,
mini-batches of 10 000, predictive log-likelihood (Jankowiak 2020) instead of
the plain ELBO for better broadband-uncertainty calibration. Training on the
CONA database (661 440 points after wavelet compression) takes a single A100,
and synthesis is 0.18 s per 44.1 kHz sample — six orders of magnitude faster
than a CFD/CAA solve.

== Reimplementation

We reimplement the Lee et al. construct in
`src/experiments/gp_rotor_noise/gp_rotor_noise.py`, with two refinements to fit
our data:

- #emph[Per-rotor RPS, not aggregate $V_1$.] Our conditioning variable is the
  actual driver of each rotor's harmonic comb, $"RPS"_i (t)$, rather than the
  paper's scalar flight velocity. This generalises the Fourier design to four
  independent comb families.
- #emph[Diagonal per-mic likelihood noise $sigma_b$] initialised to the measured
  DWT-detail-std per microphone (approximating the paper's multitask likelihood).

#figure(
  image("assets/gp_overview.png", width: 90%),
  caption: [The GP pipeline: framing -> DWT split -> phase-alignment ->
    least-squares on the BPF-injected Fourier design -> SVGP over coefficients ->
    inverse synthesis with broadband-residual sampling.],
)

== Held-out reconstruction (Michael recording 1)

Training on 8 frames and holding out 6, the GP's predicted posterior mean over
the Fourier coefficients is multiplied by the per-frame design matrix and
overlap-added back to a waveform. The generated spectrogram matches the real
per-rotor BPF peaks *exactly*.

#fig("assets/gp_faithful_spectrum.png", [GP on Michael recording 1
  (H=24, 8 train / 6 held-out frames). Top: STFT of real held-out audio vs.
  the GP-generated tonal posterior mean. Bottom-left: average spectra; the
  generated mean matches the BPF comb precisely. Bottom-right: a second
  realisation samples $cal(N)(0, sigma_b^2)$ broadband residual, per Eq. (3)
  of the paper. Coefficient RMSE = 0.10. The 2–6 kHz band remains
  under-covered (H=24 caps at about 1.5 kHz); raising $H$ reaches it but at
  proportionate memory cost.])

== Honest gaps and next steps

The generation reproduces the BPF harmonic comb up to about 1.7
kHz (H=24). Real Michael noise extends to about 3 kHz; reaching it needs
$H approx 60$, which lifts peak RAM from about 2 GiB to about 8 GiB and
lengthens the SVGP optimisation proportionately. A streaming refactor of the
per-frame Fourier design (frame-at-a-time instead of materialising the full
`(F, 2n+1, W)`) is the *correct* long-term fix.

We are currently in the middle of:

+ *Training one GP per drone* (`michaels`, `dregon`) on the *same* swapped
  training split the deep `PositionalHarmonicNoiseGen` trained on — bespoke
  driver `src/experiments/gp_rotor_noise/train_dregon_michaels.py`. The GP-fit
  shape does not suit `train.py`'s chunk-dataloader / epoch / early-stopping
  machinery (its "training" is dominated by closed-form DWT/phase-align/least-
  squares preprocessing followed by a short full-batch ELBO optimisation), so
  it is kept as a thin bespoke driver that reuses the project's real loaders
  (`load_dregon_timeframes`, `load_michaels_timeframes`) and the same
  `val_at_start=true` swapped-split definition.
+ *Building a listening-comparison notebook* extending
  `notebooks/noise_gen_real_vs_generated.ipynb` to add GP rows: an 8-second
  slice from Michael's valid recording and an 8-second slice from a DREGON
  valid recording, both with #emph[the same real RPS trajectory] fed into the
  deep model and the corresponding GP, with the user-chosen microphone. Audio
  players for real, deep-generated, and GP-generated.

Both will run on the GPU once available; the laptop-friendly config (H=24,
about 8 GiB peak, about 46 min per drone for 400 iters) is measured and ready.

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    [*Model*], [*Best val*], [*R²*], [*Status*],
    [`PositionalHarmonicNoiseGen` (E3)], [$5.3554$], [—], [trained (deep)],
    [`+ aug -> RPS uni-GRU128`], [$9.29$ (PIT)], [0.791], [degraded +27%],
    [`+ aug -> RPS transformer`], [$10.63$ (PIT)], [0.762], [degraded +26%],
    [`GP (Lee et al. construct)`], [coeff RMSE $0.10$], [—], [WIP, on DREGON+Michaels],
  ),
  caption: [Cross-thread scoreboard. The deep generator + augmentation is a
    settled negative result; the GP model is the active response to the
    failure mode its structure was chosen to avoid.],
)

= Discussion

The two parts form a coherent story. The deep generator failed *because* its
freely-learned representation admits exploitable artifacts and its trajectory
diversity is narrow — a model-collapse-like failure w.r.t. the predictor. The
GP *deliberately* removes both degrees of freedom: its tonal space is fixed
physics (the BPF combs are $k times B times f$ with blade count $B$ and RPS rate $f$),
so nothing to memorise, no artifacts to latch onto, and the broadband is
explicitly modelled as *residual*, not as a learned filter envelope. The cost
is spectral coverage (capped by H) and inference speed; the benefit is
interpretability and insensitivity to the small-data memorisation that sank the
deep generator.

Whether the GP produces noise that *sounds* enough like the real recording to
help (or at least to not hurt) the RPS predictor under the same augmentation
protocol is precisely the open question the in-progress listening test will
answer. The notebook comparison #emph[with the same RPS trajectory] feeding
both generators is the right instrument for that judgement: it controls for the
trajectory-diversity confound that defeated E4 and isolates acoustic realism.