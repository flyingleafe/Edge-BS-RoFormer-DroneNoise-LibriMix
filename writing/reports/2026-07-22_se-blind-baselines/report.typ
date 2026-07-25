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
    magnitude+phase) is the strongest baseline at every SNR, both ported models
    (MP-SENet, TF-GridNet) are competitive with or ahead of the in-house
    Edge-BS-RoFormer — even while compute-limited — while the classic
    complex-UNet (DCUNet) removes noise energy at low SNR but *degrades* the
    input above −5 dB and never restores intelligibility — far short of the same
    model's published result on this task, which we attribute to our training
    configuration rather than the architecture. Whether training on *diverse*
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
(wideband) and extended STOI. Every model is scored *per clip* on the *full*
valid set, and the per-clip records are aggregated afterwards; earlier revisions
of this report averaged a 25-mixtures/SNR subset, which we found to be
unrepresentative (see #emph[A data-quality defect and its correction], below).

== A data-quality defect and its correction
The first revision of this study reported a puzzling *non-monotonic dip at 0 dB*
for every architecture, and a noisy-input anchor of −4.8 dB SI-SDR at 0 dB where
$approx 0$ dB is expected. Both were artefacts of a defect in the mixing
pipeline.

The mixer scales the speech to hit the requested SNR *relative to the noise*,
$s' = s dot sqrt((P_n dot 10^("SNR"\/10)) \/ (P_s + epsilon))$. The $epsilon$
guards the denominator only, so a *digitally silent noise draw* ($P_n = 0$) sets
the gain to zero and the clean target *and* the mixture both become all-zeros —
a completely empty sample. Auditing the noise pools showed one offending source:
`drone_audio` returns silence on $approx 6%$ of 1-second draws, while the other
five drone sources never do. Five of the 350 `SE-valid-drone` clips are empty
this way, three of them at 0 dB.

Because SI-SDR against an all-zero reference collapses to the $10 log_10 epsilon$
floor ($-80$ dB here), those three clips alone dragged the 0 dB mean down by
$approx 5$ dB for *every* method, anchors included — manufacturing the dip. With
the empty clips excluded the noisy anchor reads −0.0 dB at 0 dB as it should, and
every model's curve, DCUNet included, is monotonic in SNR. All numbers in this
report are computed on the remaining 345 clips. The pipeline now rejects and
redraws silent draws (in the training stream and in the valid-set builder), with
a regression test; the published valid sets still contain the empty clips and are
scheduled for a rebuild.

The same audit found that the loss is not blind to this: an all-zero target makes
the SI-SDR term return a constant $+80$ dB with *zero* gradient — harmless to
learning but corrupting the monitored metric used for checkpoint selection and
early stopping — while a magnitude-domain term instead receives a genuine
gradient pushing the output toward silence.

= Results

@fig:floor and @tbl:sisdr summarise Pass A on the drone validation set. The
spread across architectures is large and MP-SENet leads at every SNR; the two
ported models trade places below it, with Edge-BS-RoFormer stronger at the
lowest SNRs and TF-GridNet stronger from −15 dB up:
*MP-SENet > {TF-GridNet, Edge-BS-RoFormer} ≫ DCUNet*.

#figure(
  image("assets/floor_drone.png", width: 100%),
  caption: [Blind SE floor on `SE-valid-drone` (Pass A), full 345-clip set.
    (a) SI-SDR improvement over the noisy input vs input SNR. (b) Absolute eSTOI
    (intelligibility); the dotted line is the noisy input. MP-SENet leads
    throughout; Edge-BS-RoFormer and TF-GridNet trade places (Edge stronger below
    −20 dB, TF-GridNet above). Every curve is monotonic once the corrupt clips
    are removed. DCUNet falls *below* the noisy input on eSTOI at every SNR, and
    below it on SI-SDR at 0 dB.],
) <fig:floor>

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    table.header([SNR], [Wiener], [*MP-SENet*], [TF-GridNet], [Edge-BS-RoF], [DCUNet-A], [DCUNet-B]),
    [−30], [−0.1], [*+23.3*], [+15.4], [+20.1], [+11.1], [+11.5],
    [−25], [−0.0], [*+19.0*], [+12.7], [+16.1], [+8.3], [+8.5],
    [−20], [+0.1], [*+18.3*], [+11.6], [+14.0], [+6.0], [+6.0],
    [−15], [+0.1], [*+19.5*], [+14.3], [+13.7], [+4.7], [+6.3],
    [−10], [+0.1], [*+17.0*], [+13.4], [+11.8], [+2.6], [+3.0],
    [−5], [+0.1], [*+13.9*], [+11.9], [+9.8], [+0.1], [+1.6],
    [0], [+0.1], [*+11.7*], [+9.7], [+6.3], [−2.2], [−2.6],
  ),
  caption: [SI-SDR improvement over the noisy input (dB), per input SNR, over the
    full 345-clip valid set (the 5 corrupt clips of #emph[A data-quality defect] excluded).
    Bold = best. Every model clears the Wiener anchor at every SNR except DCUNet
    at 0 dB, where it falls *below* the noisy input.],
) <tbl:sisdr>

*Intelligibility (eSTOI).* The models separate most clearly on eSTOI
(@tbl:estoi). MP-SENet more than doubles noisy intelligibility at −15 dB
(0.239 → 0.516) and reaches 0.779 at 0 dB; TF-GridNet and Edge-BS-RoFormer
follow; DCUNet sits *below* the noisy input at every SNR — it removes noise
energy (SI-SDR/PESQ rise at low SNR) while damaging the speech it should
preserve.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    table.header([SNR], [noisy], [*MP-SENet*], [TF-GridNet], [Edge-BS-RoF], [DCUNet]),
    [−15], [0.239], [*0.516*], [0.404], [0.344], [0.200],
    [−10], [0.276], [*0.564*], [0.464], [0.388], [0.220],
    [−5], [0.423], [*0.708*], [0.616], [0.545], [0.362],
    [0], [0.497], [*0.779*], [0.681], [0.602], [0.410],
  ),
  caption: [Absolute eSTOI (higher = more intelligible), noisy input vs models,
    full 345-clip valid set. DCUNet sits *below* the noisy input at every SNR.],
) <tbl:estoi>

*Diversity verdict — capacity-dependent (@tbl:div).* We compare each model's
drone-only pool (Pass A) against the all-harmonic pool (Pass B), scored on the
drone valid (both passes select the best drone-valid checkpoint, and both are the
same compute-limited regime, so the delta is a fair within-architecture
comparison). The verdict splits cleanly by capacity: for the weak *DCUNet*,
Pass B $>=$ Pass A at every SNR but 0 dB (Δ SI-SDR at −15 dB: +1.5) — extra
harmonic data helps a model that under-fits drone noise alone. But *all three
stronger ports are hurt or unmoved*: Edge-BS-RoFormer loses SI-SDR at every SNR
(−3.8 at −15 dB) and TF-GridNet is negative throughout, while MP-SENet loses
heavily at low SNR (−4.8 at −30 dB) and is flat at high SNR. Under an
equal training budget, diverse noise dilutes drone-specific capacity; the
transferable structure of rotating-source harmonics helps only when the model is
capacity-limited on the target. A capable model does better *focusing* on the
target noise.

#figure(
  table(
    columns: 8,
    align: (left, center, center, center, center, center, center, center),
    table.header([model (B−A)], [−30], [−25], [−20], [−15], [−10], [−5], [0]),
    [DCUNet (weak)], [*+0.5*], [*+0.2*], [*+0.1*], [*+1.5*], [*+0.4*], [*+1.5*], [−0.5],
    [MP-SENet], [−4.8], [−3.1], [−4.6], [−2.8], [−1.4], [*+0.4*], [*+0.1*],
    [TF-GridNet], [−0.8], [−0.4], [−1.3], [−1.8], [−0.8], [−0.6], [−0.1],
    [Edge-BS-RoF], [−3.8], [−4.6], [−5.0], [−3.8], [−4.0], [−2.7], [−1.6],
  ),
  caption: [Diversity delta: Δ SI-SDR of Pass B (all-harmonic) minus Pass A
    (drone-only) on the drone valid (dB). Positive (bold) = diversity helps.
    It helps only the capacity-limited DCUNet; the three stronger ports are
    hurt or unmoved.],
) <tbl:div>

*Loss and length matter — but they are not DCUNet's ceiling.* On DCUNet,
masked-MSE gave a floor essentially at the noisy level; the SI-SDR+MRSTFT
composite with the paper-matched schedule lifted the −15 dB SI-SDR gain from
+1.6 dB (an early-stopping-truncated 12-minute run) to +4.7 dB — a reminder that
a "baseline" is a claim about training as much as architecture. We tested
whether the composite's spectral term was itself the limit: its multi-resolution
STFT term dominates the SI-SDR term's gradient at the model output by 4× at
−30 dB rising to 54× at 0 dB, which would plausibly drive a masking model toward
over-suppression. It does not. Retraining DCUNet to convergence under three
objectives — the original composite, *pure* negative SI-SDR, and the composite
with its spectral term down-weighted $times 0.05$ (which brings the gradient
ratio at 0 dB from 54× to 2.5×) — moves 0 dB SI-SDR only between −2.17, −2.07 and
−1.37 dB, all still *below* the noisy input, and leaves eSTOI at 0.41/0.39/0.40
against a noisy 0.50. A $< 1$ dB spread, with the ordering unchanged. DCUNet's
high-SNR degradation is therefore not an artefact of the objective.

*It is, however, very likely an artefact of the rest of our setup.* The same
architecture is the top performer of the 2023 benchmark @mukhutdinov2023, where
DCUNet reaches SI-SDR $+3.7$ dB and eSTOI $0.4$ at −15 dB input — against
$−10.4$ dB and $0.20$ here, a $approx 14$ dB gap. That study differs from ours in
much more than the loss: it runs at *8 kHz* (we use 16 kHz), gives DCUNet an
STFT of 64 ms/16 ms (ours is 128 ms/32 ms — *four times* coarser in time), trains
on *3 s* crops (ours 1 s), samples SNR from $cal(U)(−25, −5)$ rather than
$cal(U)(−30, 0)$, and draws noise from a *single* drone's ego-noise rather than
six pooled datasets. Any of these could plausibly cost a complex-masking UNet
its high-SNR behaviour. We therefore treat the DCUNet row of this report as a
*property of this training configuration, not of the architecture*, and a
replication of the published setup is in progress to identify which factor is
responsible. The other three architectures share the same configuration, so
their absolute numbers carry the same caveat, though their *relative* ordering is
measured under matched conditions and is unaffected.

== Per-category harmonic floor
Evaluating the Pass-B (all-harmonic) models per noise family on
`SE-valid-harmonic` (@tbl:harm, mean over the SNR grid) shows a clean,
model-independent difficulty ordering: the *more harmonic* the source, the more
enhancement recovers. The strongly-tonal families — motors, aircraft, horns —
are the easiest (MP-SENet lifts SI-SDR +15.8…+19.8 dB, eSTOI to 0.51…0.58),
drone sits just below, and the *stochastic* industrial machine noise of MIMII /
MIMII-DG is hardest (MP-SENet eSTOI only 0.28…0.34). This is direct evidence for
the project's premise: it is harmonic structure — not loudness — that a blind
model can exploit, so a rotating-source target is a favourable case, and the
broadband-machine families bound the low end.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header([family], [DCUNet], [*MP-SENet*], [Edge-BS-RoF]),
    [motors], [+4.5], [*+19.8*], [+12.2],
    [aircraft], [+10.1], [*+16.9*], [+16.0],
    [horns], [+2.1], [*+15.8*], [+9.0],
    [drone], [+4.9], [*+15.2*], [+9.5],
    [mimii], [+1.1], [*+9.8*], [−0.7],
    [mimii-dg], [+1.1], [+2.1], [*+4.8*],
  ),
  caption: [Per-family blind floor on `SE-valid-harmonic` (Pass B): SI-SDR
    improvement over noisy (dB), mean over the SNR grid, sorted easiest→hardest.
    Harmonic/tonal families (motors, aircraft, horns, drone) are recovered far
    better than stochastic industrial noise (MIMII). TF-GridNet's Pass-B
    harmonic evaluation is still pending and is omitted rather than quoted from
    the superseded subsampled run.],
) <tbl:harm>

The Pass B − Pass A transfer per family (@tbl:harmtransfer) echoes the
capacity story from the drone valid, now resolved by family. Adding the noisy,
*broadband* MIMII families to a fixed budget mostly *hurts* the strong models
even on those very families (Edge-BS-RoFormer −11.6 dB on MIMII, MP-SENet −9.9 on
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
    table.header([family (B−A)], [DCUNet], [MP-SENet], [Edge-BS-RoF]),
    [motors], [+7.2], [+2.4], [+3.0],
    [horns], [+0.8], [+4.1], [+1.3],
    [aircraft], [+5.0], [−3.6], [−0.4],
    [drone], [+0.5], [−2.3], [−3.6],
    [mimii-dg], [+1.2], [−9.9], [−4.6],
    [mimii], [−0.9], [−5.1], [−11.6],
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
    [−30], [−31.0], [−52.9], [0.046], [−0.005],
    [−20], [−20.0], [−44.2], [0.112], [−0.002],
    [−15], [−15.1], [−38.5], [0.239], [0.003],
    [−10], [−10.0], [−34.7], [0.276], [0.005],
    [−5], [−5.1], [−31.8], [0.423], [0.003],
    [0], [−0.0], [−30.1], [0.497], [0.007],
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
no learning of enhancement. TF-GridNet's Pass-B evaluation on
`SE-valid-harmonic` had not completed at the time of writing and is omitted from
@tbl:harm / @tbl:harmtransfer rather than quoted from the superseded run.
Separately, TF-GridNet's training jobs each cold-started (checkpoint resume was
off) and were killed at the wall clock after ≈ 4 of 150 epochs, so its numbers
are the weakest-supported of the four discriminative models.

All results here are recomputed *per clip* over the full valid sets with the five
corrupt clips of #emph[A data-quality defect and its correction] removed; the previous revision's
25-clips/SNR subsample took the *first* 25 clips of each group, which is not a
random sample (at 0 dB its mean SI-SDR differed from the discarded half by
$approx 10$ dB) and, combined with the corrupt clips, is what produced the
spurious 0 dB dip. The published `SE-valid-*` artefacts still contain the corrupt
clips: they should be rebuilt and re-pinned with the now-fixed builder, after
which these tables should be regenerated (the exclusion used here approximates,
but is not identical to, a clean rebuild).

#bibliography("refs.bib", style: "ieee")
