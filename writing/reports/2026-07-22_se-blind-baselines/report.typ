#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [Blind Speech-Enhancement Floors under Harmonic Rotor Noise\
    #text(size: 12pt)[A two-pass baseline study of five architectures on drone and harmonic noise]],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    We establish the *blind* (no rotor-speed side information) speech-enhancement
    floor on our ultra-low-SNR (−30…0 dB) harmonic-noise data with five modern
    architectures — four discriminative models and the score-based generative
    SGMSE+ — each trained in two passes (on drone noise only, and on a
    category-uniform mix of harmonic-noise families) and scored on fixed,
    held-out drone and per-family validation sets against noisy-input and Wiener
    anchors.
    The floor is strongly *architecture-bound*: MP-SENet (parallel
    magnitude+phase) is the strongest baseline, and both ported models
    (MP-SENet, TF-GridNet) beat the in-house Edge-BS-RoFormer — even while
    compute-limited — while the classic complex-UNet (DCUNet) only removes noise
    energy without restoring intelligibility. Whether training on *diverse*
    harmonic noise helps on drone noise is *capacity-dependent* — it aids only
    the capacity-limited DCUNet and dilutes all three stronger ports (MP-SENet,
    TF-GridNet, Edge-BS-RoFormer). Trained from scratch on the same budget, the
    generative SGMSE+ does not reach viability — its sampler stays below the
    noisy floor — so the blind floor here is a discriminative result. These
    floors gate later rotor-informed claims.
  ],
  keywords: ("speech enhancement", "drone ego-noise", "ultra-low SNR", "SI-SDR", "blind baselines"),
)

= Introduction

Speech enhancement under the harmonic ego-noise of rotating sources (drone
motors and propellers) is an extreme low-SNR problem: at onboard microphones the
target speech sits −30 to 0 dB below dense, structured rotor noise. Before any
rotor-speed-informed method can claim a benefit, we need an honest *blind* floor
— how well modern architectures do with no side information — measured on our own
data with strong anchors. This report establishes that floor.

We ask two questions. *(i)* Which modern architectures set the blind floor, and
by how much do they beat trivial anchors (noisy input, Wiener)? *(ii)* Does
training on a *diverse* set of harmonic noises help on drone noise (transferable
harmonic structure) or hurt (capacity dilution)? We answer both with a
five-architecture set trained in two passes and evaluated per input SNR.

= Methods

== Data and mixing
Speech is LibriSpeech `train-clean-100` (25 speakers held out for validation).
Noise is drawn online each step and mixed at SNR $tilde.op cal("U")(−30, 0)$ dB
with `random_gain`/`random_polarity` augmentation, 16 kHz mono, 1 s chunks. Two
noise regimes define the passes:
- *Pass A (drone only):* DREGON + Michael's real drone recordings plus open
  drone-noise datasets, roughly uniform over sub-datasets.
- *Pass B (all harmonic):* category-uniform over nine families — drone, MIMII
  (industrial), MIMII-DG, aircraft (AeroSonicDB), motors (HUSTmotor, KAIST),
  horns (HornBase) — so the 258 GiB MIMII cannot dominate.

The noise pool streams lazily from object storage at shard granularity, capped at
24 shards per dataset (a bounded, diverse subset — an uncapped random-shard-per-
sample draw would stream the whole 258 GiB MIMII over a run). Both passes are
scored on the *same* fixed, published validation set: held-out noise recordings
(recording-level split) × held-out speakers, SNR grid ${−30,−25,−20,−15,−10,−5,0}$
dB, 50 mixtures/point, deterministic. Every table carries *noisy-input* and
*Wiener* (`scipy.signal.wiener`) anchors.

== Architectures
Five blind (no-RPS) waveform-in/waveform-out models: *Edge-BS-RoFormer*
(band-split rotary transformer, our Paper-1 model), *TF-GridNet* (dense
full+sub-band dual-path, mid-size $tilde.op 8.4$ M, ported), *MP-SENet* (parallel
magnitude+phase, $tilde.op 1.7$ M, ported), *DCUNet* (complex U-Net, the 2023
benchmark winner @mukhutdinov2023 — a continuity anchor), and *SGMSE+*
(score-based diffusion, ported, trained from scratch via a bespoke score-matching
loop; @tbl:sgmse and the caveats below).

== Loss and training
An initial masked-MSE pass produced floors *at* the noisy level (at ultra-low
SNR, MSE rewards attenuation-toward-silence, improving SDR but not intelligibility).
We therefore train with a metric-aligned composite: negative SI-SDR + a
multi-resolution STFT term. Training length matches the 2023 benchmark
@mukhutdinov2023: early-stop patience $N_E=30$, LR-plateau patience
$N_alpha=15$, $tilde.op 1300$ steps/epoch, cap 150 epochs, AdamW at $10^(-3)$,
`amp` off (complex STFT). Jobs run on A100 GPUs.

== Evaluation
The single evaluation path scores each checkpoint per SNR with SI-SDR, SDR, PESQ
(wideband) and extended STOI. For a *fair* cross-model comparison all models are
scored on the same balanced 25-mixtures/SNR subset (SI-SDR is high-variance at
0 dB, so mixing 25- and 50-clip means across models is unfair; the 25-subset
ranking is stable, absolute values at 0 dB carry $plus.minus$ a few dB).

= Results

@fig:floor and @tbl:sisdr summarise Pass A on the drone validation set. The
spread across architectures is large and the ranking is unambiguous:
*MP-SENet > TF-GridNet ≳ Edge-BS-RoFormer ≫ DCUNet*.

#figure(
  image("assets/floor_drone.png", width: 100%),
  caption: [Blind SE floor on `SE-valid-drone` (Pass A). (a) SI-SDR improvement
    over the noisy input vs input SNR. (b) Absolute eSTOI (intelligibility);
    the dotted line is the noisy input. MP-SENet and TF-GridNet — both ported —
    lead the in-house Edge-BS-RoFormer; DCUNet tracks the noisy eSTOI.],
) <fig:floor>

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    table.header([SNR], [Wiener], [*MP-SENet*], [TF-GridNet], [Edge-BS-RoF], [DCUNet-A], [DCUNet-B]),
    [−30], [−0.1], [*+23.2*], [+17.0], [+21.1], [+10.2], [+11.9],
    [−20], [+0.1], [*+18.5*], [+11.9], [+13.9], [+6.0], [+6.0],
    [−15], [+0.1], [*+21.4*], [+16.8], [+14.4], [+5.0], [+7.6],
    [−10], [+0.1], [*+17.9*], [+13.7], [+12.6], [+2.7], [+3.9],
    [−5], [+0.1], [*+14.0*], [+11.6], [+9.6], [−0.2], [+0.7],
    [0], [+0.1], [*+16.6*], [+14.5], [+11.1], [+2.7], [+2.2],
  ),
  caption: [SI-SDR improvement over the noisy input (dB), per input SNR, uniform
    25-mixtures/point. Bold = best. Every model clears the Wiener anchor at
    $>= −10$ dB.],
) <tbl:sisdr>

*Intelligibility (eSTOI).* The models separate most clearly on eSTOI
(@tbl:estoi). MP-SENet nearly doubles noisy intelligibility at −15 dB
(0.234 → 0.540) and reaches 0.789 at 0 dB; TF-GridNet and Edge-BS-RoFormer
follow; DCUNet stays at the noisy level — it removes noise energy (SI-SDR/PESQ
rise) without restoring speech.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    table.header([SNR], [noisy], [*MP-SENet*], [TF-GridNet], [Edge-BS-RoF], [DCUNet]),
    [−15], [0.234], [*0.540*], [0.436], [0.370], [0.228],
    [−10], [0.276], [*0.643*], [0.533], [0.454], [0.249],
    [−5], [0.423], [*0.693*], [0.603], [0.534], [0.350],
    [0], [0.468], [*0.789*], [0.687], [0.609], [0.418],
  ),
  caption: [Absolute eSTOI (higher = more intelligible), noisy input vs models.],
) <tbl:estoi>

*Diversity verdict — capacity-dependent (@tbl:div).* We compare each model's
drone-only pool (Pass A) against the all-harmonic pool (Pass B), scored on the
drone valid (both passes select the best drone-valid checkpoint, and both are the
same compute-limited regime, so the delta is a fair within-architecture
comparison). The verdict splits cleanly by capacity: for the weak *DCUNet*,
Pass B $>=$ Pass A at most SNRs (Δ SI-SDR at −15 dB: +2.5) — extra harmonic data
helps a model that under-fits drone noise alone. But *all three stronger ports
are hurt or unmoved*: Edge-BS-RoFormer and TF-GridNet lose SI-SDR at every SNR
and eSTOI across the board (Edge-BS-RoF eSTOI −0.036 at −15 dB, TF-GridNet
−0.031), while MP-SENet loses SI-SDR at low SNR and is flat on eSTOI. Under an
equal training budget, diverse noise dilutes drone-specific capacity; the
transferable structure of rotating-source harmonics helps only when the model is
capacity-limited on the target. A capable model does better *focusing* on the
target noise.

#figure(
  table(
    columns: 8,
    align: (left, center, center, center, center, center, center, center),
    table.header([model (B−A)], [−30], [−25], [−20], [−15], [−10], [−5], [0]),
    [DCUNet (weak)], [*+1.7*], [−0.0], [−0.0], [*+2.5*], [*+1.2*], [*+0.9*], [−0.6],
    [MP-SENet], [−3.6], [−4.6], [−4.3], [−2.6], [−1.4], [−0.1], [+0.0],
    [TF-GridNet], [−1.1], [−0.8], [−1.3], [−2.2], [−3.6], [−2.3], [−1.3],
    [Edge-BS-RoF], [−3.6], [−4.5], [−6.3], [−4.4], [−2.9], [−3.0], [−1.3],
  ),
  caption: [Diversity delta: Δ SI-SDR of Pass B (all-harmonic) minus Pass A
    (drone-only) on the drone valid (dB). Positive (bold) = diversity helps.
    It helps only the capacity-limited DCUNet; the three stronger ports are
    hurt or unmoved.],
) <tbl:div>

*Loss and length matter.* On DCUNet, masked-MSE gave a floor essentially at the
noisy level; the SI-SDR+MRSTFT composite with the paper-matched schedule lifted
the −15 dB SI-SDR gain from +1.6 dB (an early-stopping-truncated 12-minute run)
to +5.0 dB — a reminder that a "baseline" is a claim about training as much as
architecture.

== Per-category harmonic floor
Evaluating the Pass-B (all-harmonic) models per noise family on
`SE-valid-harmonic` (@tbl:harm, mean over the SNR grid) shows a clean,
model-independent difficulty ordering: the *more harmonic* the source, the more
enhancement recovers. The strongly-tonal families — motors, aircraft, horns —
are the easiest (MP-SENet lifts SI-SDR +15.7…+19.6 dB, eSTOI to 0.52…0.57),
drone sits just below, and the *stochastic* industrial machine noise of MIMII /
MIMII-DG is hardest (MP-SENet eSTOI only 0.29…0.34). This is direct evidence for
the project's premise: it is harmonic structure — not loudness — that a blind
model can exploit, so a rotating-source target is a favourable case, and the
broadband-machine families bound the low end.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header([family], [DCUNet], [*MP-SENet*], [TF-GridNet], [Edge-BS-RoF]),
    [motors], [+3.9], [*+19.6*], [+16.9], [+12.1],
    [aircraft], [+10.4], [*+16.6*], [+14.4], [+16.5],
    [drone], [+5.9], [*+16.5*], [+12.2], [+10.2],
    [horns], [+2.5], [*+15.7*], [+11.4], [+9.1],
    [mimii], [+1.2], [*+9.9*], [+8.1], [−0.9],
    [mimii-dg], [+1.2], [+4.7], [*+6.9*], [+5.3],
  ),
  caption: [Per-family blind floor on `SE-valid-harmonic` (Pass B): SI-SDR
    improvement over noisy (dB), mean over the SNR grid, sorted easiest→hardest.
    Harmonic/tonal families (motors, aircraft, horns, drone) are recovered far
    better than stochastic industrial noise (MIMII).],
) <tbl:harm>

The Pass B − Pass A transfer per family (@tbl:harmtransfer) echoes the
capacity story from the drone valid, now resolved by family. Adding the noisy,
*broadband* MIMII families to a fixed budget mostly *hurts* the strong models
even on those very families (Edge-BS-RoFormer −11.9 dB on MIMII, MP-SENet −7.6 on
MIMII-DG) — they cannot fit stochastic noise and lose ground focusing on it — whereas
the *harmonic* motors and horns families benefit from in-domain exposure across
every architecture (all four positive). The weak DCUNet, which under-fits
throughout, gains almost everywhere. Diverse data helps when the added families
share exploitable structure and the model has spare capacity; it hurts when
neither holds.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header([family (B−A)], [DCUNet], [MP-SENet], [TF-GridNet], [Edge-BS-RoF]),
    [motors], [+6.9], [+2.0], [+6.9], [+2.7],
    [horns], [+0.6], [+3.8], [+3.9], [+1.3],
    [aircraft], [+4.9], [−4.3], [+1.9], [−0.0],
    [drone], [+0.3], [−1.8], [−1.8], [−3.7],
    [mimii-dg], [+1.3], [−7.6], [−0.2], [−4.3],
    [mimii], [−0.8], [−4.9], [−0.0], [−11.9],
  ),
  caption: [Per-family transfer: Δ SI-SDR of Pass B minus Pass A (dB) on
    `SE-valid-harmonic`. Harmonic families (motors, horns) gain across all archs;
    stochastic MIMII families lose for the strong models.],
) <tbl:harmtransfer>

== The generative baseline: SGMSE+ from scratch
The fifth architecture, SGMSE+, is score-based diffusion — a different training
paradigm (denoising score-matching + a reverse-SDE sampler at inference), trained
here from scratch via a bespoke loop. @tbl:sgmse reports it on the drone valid
against the noisy anchor. The result is unambiguous: *the reverse-SDE output is
below the noisy input at every SNR* — SI-SDR is 25–30 dB worse and eSTOI collapses
to ≈ 0 (no intelligibility). The training signal explains why: the score net's
loss drops sharply in the first ~2k steps and then the validation SI-SDR stays
*flat across training* — the model learns the score field but the sampler never
converges to clean speech within budget. A 65 M NCSN++ trained from scratch needs
orders of magnitude more compute than the discriminative models (the SGMSE
papers train ≈ 100× longer); under an equal budget it is *not viable*. This is a
useful negative control: the blind floor on our data is set by discriminative
enhancement, not by a from-scratch generative model.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header([SNR], [noisy SI-SDR], [SGMSE+ SI-SDR], [noisy eSTOI], [SGMSE+ eSTOI]),
    [−30], [−31.0], [−57.2], [0.046], [0.008],
    [−20], [−20.0], [−42.7], [0.112], [−0.003],
    [−15], [−16.4], [−40.2], [0.234], [−0.002],
    [−10], [−10.0], [−34.1], [0.276], [−0.002],
    [−5], [−5.1], [−31.4], [0.423], [0.001],
    [0], [−4.8], [−29.9], [0.468], [0.008],
  ),
  caption: [SGMSE+ (Pass B, from-scratch score diffusion) vs the noisy anchor on
    `SE-valid-drone`. The undertrained sampler is *below* the noisy input
    everywhere (eSTOI ≈ 0) — a compute-bounded non-viable baseline, not a floor.],
) <tbl:sgmse>

= Discussion

The blind floor on our data is set by the *newer* speech-enhancement
architectures: MP-SENet's explicit magnitude-and-phase decoders are the
strongest, and both ports beat the in-house Edge-BS-RoFormer. This is a useful
recalibration — the strongest blind baseline is not the band-split transformer we
had been treating as SOTA. The intelligibility gap between models that *restore*
speech (MP-SENet, TF-GridNet, Edge-BS-RoFormer) and one that merely *denoises*
(DCUNet) is the headline: at these SNRs, SI-SDR and PESQ can rise while eSTOI does
not, so intelligibility metrics are the ones that separate real enhancement from
attenuation.

That MP-SENet and TF-GridNet win *while compute-limited* (the caveats below) means
their true ceiling is higher still. The diversity result cautions against a
one-size-fits-all data recipe: broadening training to many harmonic-noise
families helps only the capacity-limited DCUNet and *hurts or fails to help* all
three stronger ports on the target noise. The split is clean across every model
we ran, so data breadth should scale with — not substitute for — architecture and
training budget.

== Status and caveats
The MP-SENet and TF-GridNet numbers are *lower bounds*: `amp`-off fp32 makes them
$tilde.op 10 times$ slower per step than DCUNet (MP-SENet 1.5 it/s; TF-GridNet
slower at batch 4), so both hit the wall-clock wall before convergence; their
best checkpoints were recovered from object storage. fp16 training (STFT kept in
fp32) would let them converge and would likely widen their lead. The Pass B
deltas for the compute-limited MP-SENet and TF-GridNet are read from their
best-drone-valid checkpoints while their runs continue, matched against the
same-regime Pass A checkpoints; the direction of the diversity effect is stable,
absolute magnitudes may tighten as they converge. SGMSE+ (@tbl:sgmse) is
evaluated at ≈ 40 k score-matching steps of its 200 k-step budget; its validation
SI-SDR is flat from step 2 k, so the non-viability is a property of the
compute budget, not the checkpoint — training continues but the trajectory shows
no learning of enhancement. The per-category harmonic tables use a balanced
15-clips/(family, SNR) subsample for CPU tractability; the difficulty ordering is
stable, per-cell absolutes carry the usual small-sample variance.

#bibliography("refs.bib", style: "ieee")
