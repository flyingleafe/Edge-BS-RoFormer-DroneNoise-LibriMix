# Koopman operators & Vold–Kalman literature — ideas

**Status:** literature survey, no implementation commitment · **Date:** 2026-07-18

Consolidates a multi-session literature exploration (via the `bib` bibliography
MCP/CLI — all papers below are tagged `harmonic-noise-suppression` there;
`bib search "..."` or `search_library` will resurface them with full
abstracts). Motivating question: **can we get a single latent state of the
"rotating + buzzing" system (motor+propeller) that's conditionable by either
audio or RPS and can predict either, both directions, in one model** — rather
than a one-way RPS→noise generator or a one-way audio→RPS estimator bolted
together. Two literature branches were explored for this; a third
(Vold–Kalman) turned out to double as a concrete, already-being-implemented
mechanism (see `docs/vk-order-tracking-design.md`) rather than just background
reading.

---

## 1 · Candidate architectures for the bidirectional audio↔RPS latent

Ranked by how directly each matches "one shared latent, either modality in,
either modality out":

1. **Multimodal Mixture/Product-of-Experts VAE** — Shi et al.,
   *"Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep
   Generative Models"* (2019, arXiv:1911.03393). The literal pattern: per-
   modality encoders feed a **shared latent** via PoE/MoE combination; a
   decoder per modality reconstructs from that shared z. Supply audio-only,
   RPS-only, or both; infer/generate the other. Originally vision+language,
   not audio — the framework is domain-agnostic. **Bonus**: PoE handles
   missing modalities per-example naturally, which matches "no RPS telemetry
   for most clips" directly (those clips just train through the audio-only
   branch).
2. **Neural Source-Filter (NSF)** — Wang, Takaki & Yamagishi (2019,
   10.1109/taslp.2019.2956145) + **Harmonic-Net** — (2023,
   10.1109/taslp.2023.3275032). Decompose synthesis into a harmonic source
   driven by F0 (→ RPS) plus a learned filter/noise texture — exactly the
   "rotating part" (periodic source) vs. "buzzing part" (filtered texture)
   split, natively F0-controllable. Best candidate decoder to plug into the
   shared latent above.
3. **PESTO** — Riou et al., *"Pitch Estimation With Self-Supervised
   Transposition-Equivariant Objective"* (2023, arXiv:2309.02265; journal
   version 2025, 10.5334/tismir.251). Modern replacement for SPICE's
   CQT-based approach (flagged as insufficient for our data) — Siamese
   architecture on VQT, real-time-capable, self-supervised.
4. **RAVE** — Caillon & Esling, *"A variational autoencoder for fast and
   high-quality neural audio synthesis"* (2021, arXiv:2111.05011). Not
   multimodal, but the best-in-class controllable, real-time, bidirectional
   audio↔latent VAE — candidate audio encoder/decoder pair to slot into the
   PoE-VAE shell rather than building spectrogram encoders from scratch.
5. **A-JEPA** — *"Joint-Embedding Predictive Architecture Can Listen"** (2023,
   arXiv:2311.15830). Strong self-supervised audio representation learner
   (predicts masked spectrogram regions in latent space) — **caveat**: JEPA
   architectures have no decoder by design, so useful for pretraining a rich
   audio encoder, not for generating audio back out on its own.

---

## 2 · Koopman operators

**Simple-terms recap**: for a nonlinear dynamical system `x(t+1)=f(x(t))`,
Koopman's theorem says there's *always* some linear operator `K` acting on a
(possibly huge) set of observable functions `g(x)` such that
`g(x(t+1)) = K·g(x(t))` — i.e. the nonlinearity gets absorbed into choosing
the right features, and everything afterward is linear (eigendecomposition,
mode superposition, stable long-horizon prediction). "Deep Koopman" networks
learn `g` (encoder), `K` (one matrix), and a decoder, instead of hand-picking
features. For a genuinely oscillatory/rotating system, the natural Koopman
eigenfunctions *are* the vibration modes (frequency + damping per eigenvalue)
— a rotating motor is close to the textbook case.

**The field has bifurcated into two streams that haven't fully met:**

**Stream A — ML/control theory (mature, applied, not audio).** Lusch, Kutz &
Brunton, *"Learning Deep Neural Network Representations for Koopman Operators
of Nonlinear Dynamical Systems"* (2017/2019, arXiv:1708.06850) is the
foundational deep-Koopman paper; its direct citers skew toward
physics/power-grids/fluid-dynamics/spacecraft, not audio. Current state of
the field (2023–2025):
- *"Koopman Operators in Robot Learning"* (2024, arXiv:2408.04200) — survey.
- *"Vehicular Applications of Koopman Operator Theory"* (2023,
  10.1109/access.2023.3257109) — survey.
- *"Deep Koopman operators for causal discovery"* (2025,
  10.1038/s42005-025-02426-1) — reads causal structure off the operator's
  eigenstructure, not just forward prediction.
- Also relevant: *"Physics-Informed Koopman Network"* (Zakwan & Rodrigues,
  2022, arXiv:2211.09419) — lightly cited, thin citation graph.

**Stream B — physical-modelling audio synthesis (very recent, same idea,
different vocabulary).** This is the actual convergence point: Zheleznov,
Bilbao, Wright & King, *"Learning Nonlinear Dynamics in Physical Modelling
Synthesis using Neural ODEs"* (2025, arXiv:2505.10511) and its direct
follow-up *"Stable Differentiable Modal Synthesis for Learning Nonlinear
Dynamics"* (2026, arXiv:2601.10453 — swaps the original's MLP nonlinear term
for a gradient-network parametrization compatible with scalar-auxiliary-
variable stability guarantees). Both decompose a vibrating physical object
into oscillatory **modes** (frequency + damping per mode — literally
Koopman/DMD's formulation) and learn the nonlinear inter-mode coupling with a
neural ODE; the mode state is both the audio-synthesis readout and the
object's physical configuration. Swap "vibrating string" for "rotating motor
+ propeller" and this is close to a template for the bidirectional
audio↔RPS ask — **for a rotating machine, the fundamental mode frequency is
RPS by construction**, not an assumption to add on top.

**The explicit VAE↔dynamics bridge**, if a more ML-flavored version is wanted
over physical modal synthesis: *"Finding Low-Dimensional Dynamical Structure
Through Variational Auto-Encoding Dynamic Mode Decomposition"* (Kaneko, 2019,
10.1109/mlsp.2019.8918765) — puts a VAE around DMD (the standard estimator
for the Koopman operator) for a probabilistic, learned latent state with mode
structure. Small/early (1 citation) but the clearest existing precedent.

**Bottom line**: the modal-synthesis line (2025–2026) is already solving the
structural problem for vibrating strings; its mode-frequency state is a
natural place to inject/read out RPS directly, combined with a differentiable
modal-synthesis decoder driven by a learned Koopman/DMD-style latent — a
two-way audio↔RPS model in one framework, natively self-supervised (fit the
ODE/modes from audio alone; RPS becomes a directly-readable coordinate rather
than a separately-learned proxy).

---

## 3 · Vold–Kalman order tracking

Directly relevant beyond background reading: `docs/vk-order-tracking-design.md`
already implements a **coupled outer-loop VK** (§2.3 there: alternate
envelope-solve given frequency tracks, then a phase-slope frequency update) —
which is *exactly* the "VK in reverse" mechanism surveyed below, arrived at
independently. Worth cross-checking that design against the literature here.

**Origin (1993–2005)**: Vold & Leuridan's Kalman-filter order tracking,
extended by Feldbauer/Höldrich, *"Further exploration of Vold–Kalman-filtering
order tracking with shaft-speed information — I/II"* (2005,
10.1016/j.ymssp.2005.01.005 / .007) to add coupled/decoupled handling of
*crossing orders* (orders that briefly share instantaneous frequency — a
naive filter can't separate them). This coupling machinery is VKF's answer to
multiple simultaneous harmonic sources overlapping in frequency, i.e. exactly
several rotors with sometimes-coincident fundamentals.

**Active aeroacoustics community (audio-relevant, not just mechanical
vibration diagnosis) — running right through 2024–2025:**
- *"Propeller Source Noise Separation from Flight Test Measurements of the
  Joby Aviation Aircraft"* + *"Identification and Computation of Individual
  Propeller Acoustics..."* (both 2024, 10.2514/6.2024-3231 / -3232) — real
  eVTOL flight-test data, about as close to "real multi-rotor drone, real
  flight conditions" as published work gets.
- *"Characterizing the noise patterns of overlapping propellers in forward
  flight"* (2025, 10.1121/10.0036453) — the crossing/coupled-order problem
  for multiple propellers, current year.
- *"Noise reduction and aerodynamics of isolated multi-copter rotors with
  serrated trailing edges"* (2020, 10.1016/j.jsv.2020.115688).
- *"Aerodynamic noise separation of helicopter main and tail rotors using a
  cascade filter with Vold-Kalman filter and cyclic Wiener filter"* (2022,
  10.1016/j.apacoust.2022.108751) — a literal *coupled/cascaded* VKF for two
  simultaneous rotor harmonic series; methodologically the closest published
  match to "coupled Vold-Kalman for audio."
- *"The space-time structure of sound produced by stacked rotors in hover
  using Vold-Kalman filters and proper orthogonal decomposition"* (2023,
  10.1177/1475472x231199186) — coaxial/stacked rotors, hover.
- *"Standalone Extraction of Tonal Components from Aeroacoustic Signals"* +
  *"Stand-Alone Extraction of Cyclostationary Broadband Components..."* (both
  2021, 10.2514/1.j060288 / .j060289) — companion papers; conceptually the
  closest match to this project's actual goal (tonal/harmonic vs.
  broadband/turbulent split of an aeroacoustic recording).
- *"Acoustic and psychoacoustic characterisation of small-scale
  contra-rotating propellers"* (2023, 10.1016/j.jsv.2023.117971) —
  drone-scale, includes a perceptual angle.

**"VK in reverse" — estimating the time-varying order/instantaneous
frequency from the signal itself, rather than requiring it as known input**
(the specific mechanism `vk-order-tracking-design.md`'s outer frequency loop
implements independently):
- **Direct hit**: Li, Han & Wang, *"Research on a Signal Separation Method
  Based on Vold-Kalman Filter of Improved Adaptive Instantaneous Frequency
  Estimation"* (2020, 10.1109/access.2020.3002999). Uses a synchrosqueezing-
  wavelet-transform adaptive multi-ridge IF estimator to derive instantaneous
  frequency from the raw signal, then feeds that into VKF as the reference
  frequency — VKF's usual forward envelope-extraction role stays the same,
  but the "known" order frequency is no longer external/tacho input.
- **Ambiguous, flagged not confirmed** (no OpenAlex abstract, Elsevier
  doesn't always release them — would need the actual PDF): Pan & Wu,
  *"Adaptive Vold–Kalman filtering order tracking"* (2007,
  10.1016/j.ymssp.2007.06.002, 69 citations) and Chu, Le & Pan, *"Adaptive
  angular-velocity Vold–Kalman filter order tracking"* (2016,
  10.1016/j.ymssp.2016.03.013, 40 citations — TLDR mentions decoupling
  crossing orders from "multi-axial reference rotating speeds," which reads
  more like known-multiple-references than blind estimation, but unconfirmed
  without full text).
- **Broader pattern, not VKF-specific**: most "tacholess order tracking"
  work uses a *different* method for the frequency-estimation step (nonlinear
  mode decomposition, generalized demodulation, synchrosqueezing ridge
  extraction) and only optionally hands off to VKF afterward — e.g. Wang,
  Tang, Meng & Hou, *"Adaptive Estimation of Instantaneous Angular Speed for
  Wind Turbine Planetary Gearbox Fault Detection"* (2019,
  10.1109/access.2019.2908192, full text in library) uses NMD instead of VKF
  for stage one. *"Tacholess Speed Estimation in Order Tracking: A Review"*
  (2019, in library as stub) surveys this whole landscape.

---

## 4 · Cross-cutting note

Koopman/DMD (§2) and Vold–Kalman (§3) are, at bottom, doing the same kind of
thing from two different communities: both decompose a periodic/quasi-
periodic signal into a small number of oscillatory modes with a
frequency+envelope each. The modal-synthesis Koopman papers (§2, Stream B)
and the aeroacoustic coupled-VKF papers (§3) could plausibly be read side by
side as two numerically-different solvers for structurally the same
"multi-rotor harmonic decomposition" problem — one from neural-ODE/Koopman
theory, one from Kalman-filter order tracking. Worth keeping in mind if the
VK outer-loop design ever needs a differentiable/learnable upgrade path: the
modal-synthesis line already has one.

---

## References (DOI/arXiv, for quick lookup outside `bib`)

| Ref | Link |
|---|---|
| Shi et al. 2019, MMVAE | arxiv.org/abs/1911.03393 |
| Wang et al. 2019, NSF | doi.org/10.1109/taslp.2019.2956145 |
| Harmonic-Net 2023 | doi.org/10.1109/taslp.2023.3275032 |
| PESTO 2023 / 2025 | arxiv.org/abs/2309.02265 · doi.org/10.5334/tismir.251 |
| RAVE 2021 | arxiv.org/abs/2111.05011 |
| A-JEPA 2023 | arxiv.org/abs/2311.15830 |
| Lusch et al. 2017, deep Koopman | arxiv.org/abs/1708.06850 |
| Koopman Operators in Robot Learning 2024 | arxiv.org/abs/2408.04200 |
| Vehicular Koopman survey 2023 | doi.org/10.1109/access.2023.3257109 |
| Deep Koopman causal discovery 2025 | doi.org/10.1038/s42005-025-02426-1 |
| Physics-Informed Koopman Network 2022 | arxiv.org/abs/2211.09419 |
| Modal synthesis + neural ODE 2025 | arxiv.org/abs/2505.10511 |
| Stable differentiable modal synthesis 2026 | arxiv.org/abs/2601.10453 |
| VAE-DMD 2019 | doi.org/10.1109/mlsp.2019.8918765 |
| Vold–Kalman I/II 2005 | doi.org/10.1016/j.ymssp.2005.01.005 · .007 |
| Joby propeller separation 2024 (x2) | doi.org/10.2514/6.2024-3231 · -3232 |
| Overlapping propellers 2025 | doi.org/10.1121/10.0036453 |
| Multi-copter serrated rotors 2020 | doi.org/10.1016/j.jsv.2020.115688 |
| Cascade VKF helicopter 2022 | doi.org/10.1016/j.apacoust.2022.108751 |
| Stacked rotors hover 2023 | doi.org/10.1177/1475472x231199186 |
| Tonal/cyclostationary extraction 2021 (x2) | doi.org/10.2514/1.j060288 · .j060289 |
| Contra-rotating propellers 2023 | doi.org/10.1016/j.jsv.2023.117971 |
| Adaptive IF-estimation VKF 2020 | doi.org/10.1109/access.2020.3002999 |
| Adaptive VKF order tracking 2007 | doi.org/10.1016/j.ymssp.2007.06.002 |
| Adaptive angular-velocity VKF 2016 | doi.org/10.1016/j.ymssp.2016.03.013 |
| Tacholess IAS estimation (WT gearbox) 2019 | doi.org/10.1109/access.2019.2908192 |
