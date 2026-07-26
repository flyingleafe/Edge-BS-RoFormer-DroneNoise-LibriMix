#import "/writing/templates/typst/report.typ": report, author-meta

#let seen(body) = text(fill: rgb("#1f77b4"), weight: "bold", body)
#let unseen(body) = text(fill: rgb("#d62728"), weight: "bold", body)

#show: report.with(
  title: [Why DCUNet Wins Some Drone-Speech Benchmarks and Loses Ours\
    #text(size: 12pt)[Noise-recording leakage, measured and quantified]],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    DCUNet came out *weakest* of five architectures in our blind speech-enhancement
    study, which sat badly against two prior results in this project's own
    lineage: the 2023 IEEE Access drone-SE survey, whose 12-model benchmark
    DCUNet *won*, and our Paper-1 (Edge-BS-RoFormer) replication on DN-LM, where
    DCUNet again ranked *first*. This report resolves the contradiction.
    We reproduce the survey's DCUNet result essentially exactly under the
    survey's own protocol (SI-SDR +3.82 dB against the published +3.7; eSTOI
    0.408 against 0.4, at −15 dB input), which rules out any defect in our
    training or evaluation pipeline. We then show, by a single controlled
    manipulation, that this result is a *seen-noise* result: holding out two of
    the five ego-noise recordings — of the *same drone*, differing only by
    recording session — costs *12.9 dB SI-SDR and 0.17 eSTOI*, while a control
    model trained on all five scores the two halves within 0.3 dB of each other.
    The same mechanism accounts for our DN-LM result. Our re-creation of DN-LM
    generates its train and validation splits from an identical specification
    differing only in random seed, so *99.2%* of validation clips reuse a noise
    recording seen in training and speaker overlap is total. We are careful about
    attribution here: the original Edge-BS-RoFormer repository publishes neither
    the dataset nor a builder for it, and our re-creation *inverts* the paper's
    headline comparison (Edge-BS-RoFormer 3.55 dB behind DCUNet at −15 dB, where
    the paper reports it 2.2 dB ahead), so the leak is demonstrably ours and the
    original split is unknown. DCUNet is first of four on our leaked
    re-creation and last of four on our held-out benchmark, where it pushes
    eSTOI *below* the unprocessed input.
    Mechanically, the failure is a split between two axes usually assumed to move
    together: on unseen noise DCUNet still removes 3–7 dB of noise *energy* while
    recovering essentially no *intelligibility*. This is specific to DCUNet, not
    to the data — MP-SENet, trained on a pool containing none of this drone's
    audio, gains +0.342 eSTOI on it, generalising to an unheard drone better than
    DCUNet generalises between two sessions of a drone it was trained on.
  ],
  keywords: ("speech enhancement", "drone noise", "generalization", "data leakage", "DCUNet"),
)

// Keep figures in reading order: the base journal template floats them, which
// puts a figure above the section that introduces it.
#set figure(placement: none)

= Summary

#block(fill: luma(245), inset: 10pt, radius: 4pt, width: 100%)[
  1. *The survey's DCUNet result reproduces here.* Under the survey's own
     protocol we measure SI-SDR *+3.82 dB* (published: +3.7) and eSTOI *0.408*
     (published: 0.4) at −15 dB input. Our pipeline is not at fault.

  2. *It is a seen-noise result.* Holding out 2 of 5 ego-noise recordings of the
     same drone costs *12.9 dB SI-SDR* and *0.17 eSTOI*. A control trained on
     all five scores the two halves within *0.3 dB / 0.016 eSTOI*, so the held-out
     half is not intrinsically harder.

  3. *The survey's benchmark leaks noise by design*, reusing its five recordings
     between train and validation. *Our DN-LM re-creation leaks it too* — both
     splits come from one specification differing only in seed, so *99.2%* of
     validation clips reuse a training noise clip. The *original* DN-LM ships no
     dataset and no builder, and our re-creation inverts the paper's headline
     comparison, so that leak is ours and theirs is unknown.

  4. *The ranking inverts with leakage.* DCUNet is *1st of 4* on our leaked
     DN-LM re-creation and *last of 4* on our held-out benchmark, where its
     eSTOI (0.193) falls *below* the unprocessed input (0.233).

  5. *The failure mode is a metric split, not a broken output.* On unseen noise
     DCUNet still gains 3–7 dB SI-SDR while gaining $approx 0$ eSTOI: it removes noise
     energy without recovering speech.

  6. *It is DCUNet-specific.* MP-SENet, trained without a single sample of this
     drone, gains *+0.342 eSTOI* on it.
]

= Why this investigation happened

Our blind-baseline study (batch `f1-se-blind-baselines`) ranked five
architectures on ultra-low-SNR harmonic noise and put DCUNet last by a wide
margin — it removed noise energy at low SNR but degraded eSTOI at every SNR
tested. That is an uncomfortable result to publish, for two reasons internal to
this project:

- The *2023 IEEE Access survey* @survey (this project's own prior work) reports
  DCUNet as the *best* of twelve models on drone ego-noise, reaching SI-SDR
  +3.7 dB, eSTOI 0.4 and PESQ 1.9 at −15 dB input.
- Our *Paper-1 replication*, on our own re-creation of the DN-LM dataset, put
  DCUNet *first* of four models on SI-SDR and STOI, ahead of the
  Edge-BS-RoFormer that study was built around — the reverse of what that paper
  reports.

Either our pipeline was broken, or the three benchmarks were not measuring the
same thing. This report establishes that it is the latter, and identifies
exactly which property of the benchmarks does the work.

= What each benchmark actually holds out

This is the whole story in one table, so it comes first.

#figure(
  table(
    columns: (1.05fr, 0.85fr, 0.85fr, 1.9fr),
    align: (left + horizon, center + horizon, center + horizon, left),
    stroke: 0.5pt + luma(180),
    table.header([*Benchmark*], [*Noise held out?*], [*Speech held out?*], [*How the split is made*]),
    [2023 survey @survey], [*No* — by design], [Yes (90/10 utterances)],
    [The same 5 ego-noise recordings are used for train *and* validation; only the speech is split.],
    [DN-LM — *our re-creation*], [*No*], [*No*],
    [`DN-LM-train` and `DN-LM-valid` are the *same generator spec*, differing only in `seed: 42 → 43`. Both draw uniformly from the same file lists. The *original* dataset is unpublished (no data, no builder), so its split is unknown.],
    [SE-valid-drone / -harmonic (F1)], [*Yes* — whole recordings], [Yes (25 of 246 speakers)],
    [Per-dataset shard holdout: validation draws recordings that training never sees.],
    [SE-valid-avq-split (this report)], [*Both, separately*], [Yes (same 25 speakers)],
    [Two categories in one set: `avq_ego_s1` (recordings the model trained on) and `avq_ego_s2` (recordings it never saw).],
  ),
  caption: [What the three benchmarks in this project's lineage hold out. Only the
    F1 sets isolate the model from the specific noise recordings it is scored on.],
)

The last row is the instrument built for this report: a validation set whose two
halves differ *only* in whether the model was trained on those recordings, with
the same held-out speakers on both sides.

= Result 1 — the survey's result reproduces

We re-ran the survey's DCUNet arm as faithfully as the data allows: DCUNet-10,
3.0 s crops, STFT window 64 ms / hop 16 ms, SI-SDR-only loss, Adam 1e-3, plateau
×0.1 patience 5, early stop 10, batch 32, train SNR $cal(U)(-25, -5)$ dB mixed on
the fly, and — per the survey — *the same five AVQ ego-noise recordings for
training and validation*, splitting only the speech.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    table.header([*Metric at −15 dB input*], [*Unprocessed*], [*This work*], [*Survey @survey*]),
    [SI-SDR (dB)], [−15.05], [*+3.82*], [+3.7],
    [eSTOI], [0.126], [*0.408*], [0.4],
    [PESQ (wideband, 16 kHz)], [1.061], [1.201], [—],
    [PESQ (narrowband, 8 kHz)], [1.196], [1.538], [1.9],
  ),
  caption: [The replication. SI-SDR and eSTOI land on the published values.],
)

SI-SDR and eSTOI — the two metrics carrying the survey's claim — reproduce
essentially exactly. PESQ does not, and we state the accounting honestly rather
than rounding it away:

- Our metric code selects *wideband* PESQ at ≥16 kHz, whereas the survey ran at
  8 kHz and therefore reported *narrowband* PESQ. Measured on real LibriSpeech
  speech, PESQ-NB scores *+0.64 to +1.54 above* PESQ-WB on identical audio.
- Scoring like-for-like moves us from 1.201 to *1.538*, closing roughly half the
  gap to 1.9. *A residual ≈0.36 remains unexplained.* Candidates — none of them
  measured here — are the speech corpus (LibriSpeech vs TIMIT), the drone (AVQ vs
  the survey's AS recordings), our audio being 16 kHz decimated to 8 kHz rather
  than natively 8 kHz, and this run having been stopped while still improving.

The important consequence is negative: *there is no defect in our training or
evaluation pipeline*, and the F1 DCUNet result is not an artefact of it. Whatever
separates the benchmarks lies in the data, not the code.

= Result 2 — how DCUNet fails to generalize

Take the replication above and change *exactly one thing*: train on AVQ session 1
only (`S1_seq1/2/3`), holding session 2 (`S2_seq1/2`) out of training entirely.
Everything else — architecture, loss, sample rate, crop length, SNR range,
optimiser, schedule, epoch size, held-out speakers — is identical. Then score
both halves.

#figure(
  image("assets/seen_unseen.png", width: 100%),
  caption: [The generalization failure, isolated. #seen[Blue] is noise the model
    trained on; #unseen[red] is noise from the same drone that it never heard.
    Grey is the control trained on all five recordings — it lies on top of the
    blue curve for *both* halves, proving session 2 is not the harder half. The
    red curve barely leaves the unprocessed input.],
)

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + luma(180),
    table.header([*At −15 dB input*], [*SI-SDR*], [*ΔSI-SDR*], [*eSTOI*], [*ΔeSTOI*], [*corr*]),
    [`avq_ego_s1` — seen in training], [*+3.60*], [+18.53], [*0.339*], [*+0.225*], [0.823],
    [`avq_ego_s2` — never seen], [*−9.30*], [+5.67], [*0.168*], [*+0.046*], [0.344],
    table.hline(stroke: 0.5pt),
    [*gap*], [*12.9 dB*], [], [*0.171*], [], [],
  ),
  caption: [Two recording sessions of the same drone, scored by one model.],
)

Three things make this measurement tight:

+ *The halves are equally hard unprocessed.* Noisy SI-SDR is −14.93 (seen) vs
  −14.97 (unseen); noisy eSTOI is 0.114 vs 0.122; 250 clips each.
+ *The control removes the alternative explanation.* The step-1 model, trained on
  all five recordings, scores +4.42 dB / 0.391 eSTOI on `s1` and +4.12 dB / 0.375
  on `s2` — a gap of *0.3 dB and 0.016 eSTOI*. Session 2 is not intrinsically
  harder; it is only hard for a model that did not train on it.
+ *On the seen half, the probe reproduces the replication* (+3.60 dB vs +3.82 dB),
  so nothing else about the reduced training set broke the model.

The correlation column is the clearest statement of what "fails to generalize"
means here: on seen noise the output is strongly aligned with the clean target
(`corr` 0.823); on unseen noise from the same aircraft it is not (0.344).

= Result 3 — "training-pool breadth" is the same effect

Our first hypothesis was that DCUNet was defeated by the *breadth* of our
training noise. We tested it as a ladder: identical model, loss and validation
clips, with only the training noise pool widened from AVQ-only, to AVQ plus all
drone sources, to AVQ plus all harmonic sources — each pool a strict superset of
the last, so AVQ is never removed, only diluted (from 100% of training samples to
$approx 14%$ to $approx 2%$).

#figure(
  image("assets/ladder.png", width: 100%),
  caption: [Widening the training pool destroys the intelligibility gain (left)
    while leaving a 3–7 dB energy gain intact (right).],
)

ΔeSTOI at −15 dB falls *+0.276 → +0.006 → −0.002*. But the ladder and the
seen/unseen experiment are not two findings — they are one. The probe's *unseen*
half (−9.30 dB SI-SDR, ΔeSTOI +0.046) lands on top of the broad-pool arms
(all-drone: −9.64 dB, +0.006). What governs DCUNet's score is not how broad the
pool was but *how much of its training was spent on the specific recording it is
being tested on*. Breadth matters only because it dilutes that share.

This also disposes of a weaker explanation we initially entertained and had to
retract: that the broad arms were merely *specialising elsewhere* and so looked
bad on an off-distribution test. They are not — each broad arm was also scored on
the validation set matching its own training pool, and the picture is unchanged
(all-drone on `SE-valid-drone`: ΔeSTOI 0.000 / −0.001 / +0.004 / −0.014 / −0.020 /
−0.018 / −0.041 across the SNR grid).

= Result 4 — what the failure looks like mechanically

Speech enhancement is usually reported with metrics assumed to move together.
They come apart here, and the split is the substance of the failure.

#figure(
  image("assets/energy_vs_intelligibility.png", width: 88%),
  caption: [Every DCUNet condition without heavy exposure to the test recording
    sits in the shaded band: several dB of noise energy removed, no
    intelligibility recovered.],
)

On unseen noise DCUNet behaves as an *energy* suppressor rather than a speech
*recoverer*. It is worth being precise about what it is not doing, because we got
this wrong once during the investigation:

#block(fill: rgb("#fff6f6"), inset: 9pt, radius: 4pt, width: 100%, stroke: 0.5pt + rgb("#d62728"))[
  *A retracted reading.* We initially diagnosed the F1 DCUNet as having
  *collapsed to a near-null output*, on the evidence that its `sdr` sat near
  0 dB at every input SNR — which is what an all-but-silent estimate produces,
  since $||s - hat(s)||^2 -> ||s||^2$. That inference is invalid. The pair
  $(mono("sdr"), mono("si_sdr"))$ admits *two* solutions for the output level —
  a near-null one and an over-loud one — and both reproduce any given pair. Direct
  measurement of the output/target energy ratio settles it: at −10 dB the
  all-harmonic arm has `sdr` = −0.39 (the supposedly diagnostic value) with
  `gain_db` = −0.22, i.e. output energy *at target level*. The models sit on the
  over-loud residual-noise root. (Relatedly: `SI-SDR ≥ SDR` is *false*; the
  correct bound is $"SDR"_"lin" <= "SI-SDR"_"lin" + 1$.) Our per-clip eval now
  records `gain_db` and `corr` so this question is measured, never inferred.
]

= Result 5 — the control: this is DCUNet's limitation

If broad, held-out rotor noise were simply unlearnable, no model would do better.
One already does, and the comparison is stronger than it first appears:
the F1 drone training pool (`conf/online_mix/se_drone_only.yaml`) contains *no
AVQ audio whatsoever* — AVQ was added to this project after F1 ran. So when
MP-SENet is scored on the AVQ clips, *all* of that drone is unseen noise for it.

#figure(
  image("assets/control.png", width: 72%),
  caption: [MP-SENet generalises to a drone it has never heard better than
    DCUNet generalises across two sessions of a drone it trained on.],
)

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    table.header([*At −15 dB, unseen rotor noise*], [*AVQ in training?*], [*ΔeSTOI*], [*SI-SDR*]),
    [MP-SENet (F1, broad drone pool)], [*never*], [*+0.342*], [+3.11],
    [DCUNet (this report's probe)], [same drone, other session], [*+0.046*], [−9.30],
  ),
  caption: [The control that localises the limitation to the architecture.],
)

= Why the prior benchmarks flattered DCUNet

== The survey

The survey's protocol states it plainly: the five ego-noise sequences are used
for both training and validation, and only the speech is split 90/10. Every
validation mixture therefore contains noise the model was trained on. By the
measurement in Result 2, that is worth roughly *13 dB SI-SDR and 0.17 eSTOI* to
DCUNet.

== DN-LM — *our re-creation*, and an important caveat

Here the leak is not a protocol choice but a property of the generator — and,
crucially, of *our* generator. `DN-LM-train` and `DN-LM-valid` are declared by
the *same* derived-dataset specification in
`src/data_processing/derivations.py` — identical LibriSpeech pin, identical
`drone_audio` pin, identical subpaths and mixing parameters. The only differences
are `seed: 42 → 43` and `num_samples: 6480 → 720`. Both splits then draw,
uniformly and with replacement, from the same two file lists.

#figure(
  table(
    columns: 2,
    align: (left, right),
    stroke: 0.5pt + luma(180),
    table.header([*DN-LM leakage (our re-creation)*], [*Value*]),
    [distinct noise clips in the pool], [1332],
    [distinct underlying drone recordings], [257],
    [training draws / validation draws], [6480 / 720],
    [validation clips whose noise clip is *also* in train], [*714 / 720 (99.2%)*],
    [validation clips reusing an *exact* train utterance], [149 / 720 (20.7%)],
    [speaker overlap], [$approx 100%$ (no speaker split)],
  ),
  caption: [Leakage arithmetic for our DN-LM re-creation. With 6480 draws from
    1332 clips, the probability that a given clip is never drawn in training is
    $(1 - 1\/1332)^6480 = 0.0077$; all 257 underlying recordings appear in
    training with near-certainty.],
)

#block(fill: rgb("#fffdf0"), inset: 9pt, radius: 4pt, width: 100%, stroke: 0.5pt + rgb("#b58900"))[
  *This leak is ours, and must not be attributed to the published work.* The
  original Edge-BS-RoFormer repository ships *neither the DN-LM dataset nor any
  script that builds it* — `datasets/` is in its `.gitignore`, there are no
  releases and no dataset link, and the repository is 78 KB of code. Its
  `dataset.py` simply takes a training directory and a validation directory,
  so the split is entirely a property of a data-preparation step that was never
  published. Our DN-LM is therefore an independent re-creation
  (`scripts/create_dataset.py`), and the 99.2% figure above characterises *it*.

  There is positive evidence that the two differ. The paper reports
  Edge-BS-RoFormer *beating* DCUNet by *2.2 dB* SI-SDR at −15 dB input. On our
  re-creation, at the same operating point, Edge-BS-RoFormer comes in *3.55 dB
  behind* DCUNet — a *5.75 dB swing with the sign reversed*:

  #align(center, table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    table.header([*At −15 dB input*], [*DCUNet*], [*Edge-BS-RoFormer*], [*Δ*]),
    [our DN-LM re-creation (n = 128)], [−7.90], [−11.45], [*−3.55 dB*],
    [paper's reported claim], [—], [—], [*+2.20 dB*],
  ))

  We did not reproduce the paper's headline comparison, so our re-creation is
  demonstrably not their dataset. The *direction* of the discrepancy is what
  leakage predicts — leakage flatters the memorisation-prone model most, and
  DCUNet is exactly that model by the measurement in Result 2 — but this is
  suggestive, not proof, and the original dataset's split remains *unknown*
  rather than shown to be sound or unsound.
]

The practical consequence is confined but real: *conclusions this project has
drawn from its own DN-LM re-creation are unsafe*, DCUNet's first-place finish
there most of all.

== The ranking inverts with leakage

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, left, center),
    stroke: 0.5pt + luma(180),
    table.header(
      [*Model*], [*DN-LM SI-SDR*], [*DN-LM STOI*], [], [*SE-valid-drone SI-SDR*],
    ),
    [*DCUNet*], [*−8.09 (1st)*], [*0.541 (1st)*], [], [*−10.88 (last)*],
    [Edge-BS-RoFormer], [−9.94 (2nd)], [0.529 (2nd)], [], [−2.13 (2nd)],
    [HTDemucs], [−10.10 (3rd)], [0.503 (3rd)], [], [—],
    [DPTNet], [−33.39 (4th)], [0.302 (4th)], [], [—],
    [MP-SENet], [—], [—], [], [*+2.27 (1st)*],
    [TF-GridNet], [—], [—], [], [−2.57 (3rd)],
  ),
  caption: [Left: the leaked DN-LM benchmark, where DCUNet ranks first. Right: the
    held-out-noise F1 benchmark, where it ranks last. (The two columns use
    different valid sets and are not comparable in absolute value — only the
    ordering is the point.)],
)

On the held-out benchmark DCUNet's eSTOI is *0.193* against an unprocessed input
of *0.233*: it is not merely unhelpful but actively harmful to intelligibility.

= What this does and does not establish

*Established.*
- The survey's DCUNet numbers are reproducible under the survey's protocol, so
  our pipeline is sound.
- Holding out noise recordings of the same drone costs DCUNet 12.9 dB SI-SDR and
  0.17 eSTOI, with a control excluding "the held-out half is harder".
- *Our re-creation of* DN-LM shares 99.2% of noise clips and $approx 100%$ of
  speakers between train and valid, measured from the generator specification and
  the source file inventory.
- At least one modern architecture (MP-SENet) does generalise to unheard rotor
  noise on this data, so the task is learnable.

*Not established.*
- *Why* DCUNet specifically fails to generalise. We localise it to the
  architecture but do not identify the mechanism — capacity (2.81 M parameters),
  the complex-mask formulation, and the absence of any attention or
  cross-band mixing are all untested candidates.
- Whether the survey's *absolute* PESQ is reachable: ≈0.36 of the gap survives the
  wideband/narrowband correction and is unexplained.
- *Whether the original DN-LM dataset was leaked.* Its data and construction are
  unpublished. Our re-creation is leaked and does not reproduce the paper's
  headline ranking, which shows the two differ but says nothing about which way
  the original was split.
- Whether DCUNet's broad-pool arms would improve with much longer training. They
  early-stopped or plateaued with validation degrading while training loss fell,
  which argues against it, but no long-run was done.
- The probe's numbers come from its epoch-7 checkpoint (still improving when
  scored). The 12.9 dB gap is far too large for that to change its sign or order
  of magnitude, but the exact figures would shift slightly at convergence.

= Recommendations

+ *Report DCUNet as a baseline, but never quote the survey's absolute numbers as
  a target for held-out-noise work.* They are seen-noise numbers; the honest
  comparison point on unseen noise is roughly +5 dB SI-SDR and ≈0 eSTOI gain.
+ *Do not spend further effort "fixing" DCUNet.* The F2 recipe (16 kHz, 64 ms /
  16 ms STFT, SI-SDR-only loss) already recovered everything that configuration
  could recover — it lifted ΔSI-SDR from erratic-and-sometimes-negative to a
  consistent +3…+7.7 dB and stopped eSTOI being actively degraded. The residue is
  generalisation, not configuration.
+ *Treat noise-recording holdout as a first-class property of any benchmark we
  build or cite*, and state it explicitly in every results table. Two of the
  three benchmarks in this project's own lineage leak it, and the leak is worth
  more than the difference between architectures.
+ *Re-examine any conclusion drawn from our DN-LM re-creation*, since its
  leakage affects every model measured on it, not only DCUNet — though it should
  flatter memorisation-prone models most. Rebuilding it with a recording-level
  and speaker-level holdout is cheap and would make it usable again.

= Reproducing this

All numbers come from per-clip CSVs under `results/f2_perclip/`, produced by
`scripts/eval_se_perclip.py` (one row per validation clip, with `category` and
`input_snr` metadata, and `gain_db`/`corr`/`pesq_nb` columns). Figures are
regenerated by `prepare.py` in this directory; tables by
`scripts/f2_ladder_table.py --by-category`.

#table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt + luma(180),
  table.header([*Artefact*], [*Identifier*]),
  [Batch documentation], [`docs/experiments/f2-survey-replication.md`],
  [Replication arm], [`conf/experiment/f2_dcunet_avq_survey.yaml`],
  [Memorisation probe], [`conf/experiment/f2_dcunet_avq_heldout.yaml`],
  [Ladder arms], [`f2_dcunet_alldrone.yaml`, `f2_dcunet_allharmonic.yaml`],
  [Noise dataset], [`AVQ-egonoise` (5 sequences, ch. 0, 16 kHz mono)],
  [Split valid set], [`SE-valid-avq-split@681bf90cf1c6` (500 clips)],
  [Survey valid set], [`SE-valid-avq-survey` (250 clips)],
  [DN-LM specs], [`src/data_processing/derivations.py`, `SPECS["DN-LM-{train,valid}"]`],
)

#bibliography("refs.bib", style: "ieee")
