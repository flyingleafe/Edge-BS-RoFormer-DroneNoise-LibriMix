#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

// Notes-visible build: `make notes` (typst --input notes=1). Default build is clean.
#let notes-mode = sys.inputs.at("notes", default: none)

#show: hns-slides.with(
  title: [Why DCUNet wins some benchmarks and loses ours],
  subtitle: [Noise-recording leakage, measured and quantified],
  author: [Dmitrii Mukhutdinov],
  date: [2026-07-27],
  show-notes: if notes-mode != none { bottom } else { none },
)

#let good(body) = text(fill: rgb("#1f77b4"), weight: "bold", body)
#let bad(body) = text(fill: rgb("#d62728"), weight: "bold", body)

= The contradiction

DCUNet came out *last of five* in our blind-baseline study — below the noisy
input on eSTOI at every SNR. But in this project's own lineage it *wins twice*:

#v(0.6em)

- *2023 IEEE Access survey* — DCUNet is best of *twelve* models on drone
  ego-noise: SI-SDR +3.7 dB, eSTOI 0.4, PESQ 1.9 at −15 dB.
- *Paper-1 / DN-LM* (Edge-BS-RoFormer) — DCUNet ranks *first of four* on SI-SDR
  and STOI.

#v(0.8em)

#align(center, text(size: 1.15em)[
  Either our pipeline is broken, or the three benchmarks\
  are not measuring the same thing.
])

#speaker-note[
  This is the awkward result that started the investigation. Our F1 study is the
  one that says DCUNet is bad; two prior results in our own lineage say it is
  excellent. We cannot publish the F1 ranking without resolving this.
]

= Step 1 — the survey's result reproduces exactly

We re-ran the survey's DCUNet arm under the survey's own protocol: same model,
loss, crop, SNR range, schedule — and, as the survey does, *the same five
ego-noise recordings for training and validation*.

#v(0.5em)

#align(center, table(
  columns: 4,
  align: (left, center, center, center),
  stroke: 0.5pt + luma(180),
  table.header([*at −15 dB input*], [*do nothing*], [*ours*], [*published*]),
  [SI-SDR (dB)], [−15.05], [*+3.82*], [+3.7],
  [eSTOI], [0.126], [*0.408*], [0.4],
  [PESQ (narrowband)], [1.196], [1.538], [1.9],
))

#v(0.6em)

*Our pipeline is not the problem.* Whatever separates the benchmarks is in the
data.

#speaker-note[
  SI-SDR and eSTOI — the two metrics carrying the claim — land on the published
  values. PESQ is short by ~0.36 even after correcting wideband vs narrowband
  scoring (the survey ran at 8 kHz, we run at 16 kHz); that residual is honestly
  unexplained. Candidates: TIMIT vs LibriSpeech, a different drone, decimated
  vs native 8 kHz.
]

= The one experiment that explains everything

Same replication, *one* change: train on AVQ *session 1 only*, hold session 2
out. Same drone. Same held-out speakers. Score both halves.

#v(0.4em)

#align(center, image("assets/seen_unseen.png", width: 96%))

#speaker-note[
  Everything else is identical — architecture, loss, sample rate, crop, SNR
  range, optimiser, schedule, epoch size. The only difference between the blue
  and red curves is whether those recordings were in the training set.

  Red barely leaves the do-nothing line. At −15 dB: +3.60 dB vs −9.30 dB, and
  eSTOI 0.339 vs 0.168.
]

= Session 2 is not the harder half

Score the *step-1* model — which trained on all five recordings — on the same
split. If session 2 were simply harder, its gap would persist.

#v(0.3em)

#align(center, image("assets/control.png", width: 94%))

#align(center)[
  Unprocessed, the halves are identical: SI-SDR −14.93 vs −14.97, eSTOI 0.114 vs
  0.122, 250 clips each.
]

#speaker-note[
  This is the control that closes the argument. The left panel is indifferent to
  which half it is scored on — 0.3 dB. The right panel falls off a cliff on the
  half it never saw — 12.9 dB. Training exposure is the only difference between
  the two panels.
]

= "Pool breadth" was the same effect all along

Our first hypothesis was that DCUNet was defeated by the *breadth* of our noise.
Widening the pool (AVQ only #sym.arrow +all drone #sym.arrow +all harmonic)
dilutes AVQ from 100% to ~14% to ~2% of training.

#v(0.3em)

#align(center, image("assets/ladder.png", width: 100%))

#speaker-note[
  ΔeSTOI falls +0.276 → +0.006 → −0.002. But the probe's *unseen* half
  (−9.30 dB) lands right on the broad-pool arms (−9.64 dB). One effect, not two:
  what governs DCUNet's score is how much of its training was spent on the
  recording under test. Breadth matters only because it dilutes that share.
]

= What the failure actually is

On unseen noise DCUNet still removes *3–7 dB of noise energy* while recovering
*essentially no intelligibility* — two axes usually assumed to move together.

Not a collapsed or silent output: measured output energy sits at or above target
level. The estimate is *decorrelated* from the speech.

#v(0.6em)

#align(center, table(
  columns: 4,
  align: (left, center, center, center),
  stroke: 0.5pt + luma(180),
  table.header([*at −15 dB*], [*ΔSI-SDR*], [*ΔeSTOI*], [*corr*]),
  [seen noise], [+18.5 dB], [*+0.225*], [0.823],
  [unseen noise], [+5.7 dB], [*+0.046*], [0.344],
))

#speaker-note[
  Worth stating because we got it wrong first: we initially read `sdr` sitting
  near 0 dB as proof of collapse to a near-null output. That inference is
  invalid — the quadratic has two roots and the models sit on the over-loud
  residual-noise root. We now measure gain_db and corr rather than infer.

  Also: SI-SDR >= SDR is false. The bound is SDR_lin <= SI-SDR_lin + 1.
]

= It is DCUNet's limitation, not the data's

F1's drone pool contains *no AVQ audio at all* — so MP-SENet had *never heard
this drone*.

#v(0.3em)

#align(center, image("assets/mpsenet.png", width: 56%))

#align(center)[
  MP-SENet generalises to an *unheard* drone better than DCUNet generalises\
  between two sessions of a drone it *trained on*.
]

#speaker-note[
  +0.342 eSTOI vs +0.046 at −15 dB. This is the control that localises the
  limitation to the architecture rather than to the task. The broad harmonic
  noise problem is learnable; DCUNet does not learn it.
]

= Why the prior benchmarks flattered DCUNet

*The survey* reuses its five noise recordings between train and validation by
design, splitting only the speech.

#v(0.5em)

*DN-LM* leaks by its published protocol (Liu et al., Drones 2025, §3.5):

#quote(block: true)[
  "randomly selected from LibriSpeech and DroneAudioDataset … a 2 h synthetic
  dataset was constructed *and partitioned into training and validation sets at
  a 9:1 ratio*"
]

Mixtures are synthesised from the *full* pools first, then the *mixtures* are
split. No speaker or recording holdout is described — and a random split of one
pool cannot create one.

#speaker-note[
  2 h at 1 s per clip is 7200 clips; 9:1 is 6480/720 — exactly our re-creation's
  split sizes, so our rebuild follows the description faithfully.
]

= DN-LM leakage, measured

#align(center, table(
  columns: 2,
  align: (left, right),
  stroke: 0.5pt + luma(180),
  table.header([*on the same source corpora*], [*value*]),
  [distinct noise clips / recordings], [1332 / 257],
  [train draws / valid draws], [6480 / 720],
  [valid clips whose noise clip is *also* in train], [*714 / 720 (99.2%)*],
  [valid clips reusing an *exact* train utterance], [149 / 720 (20.7%)],
  [speaker overlap], [$approx 100%$],
))

#v(0.6em)

More leaked than the survey: it shares the noise recordings, the speaker set,
*and* a fifth of the exact utterances.

#speaker-note[
  Caveat to state if asked: the repo ships neither the dataset nor a builder, so
  we assess the protocol as published. An undocumented source-level partition
  would void this.
]

= The ranking inverts with leakage

#align(center, table(
  columns: 3,
  align: (left, center, center),
  stroke: 0.5pt + luma(180),
  table.header([*model*], [*DN-LM (leaked)*], [*F1 (held-out noise)*]),
  [*DCUNet*], [*−8.09 dB — 1st*], [*−10.88 dB — last*],
  [Edge-BS-RoFormer], [−9.94 dB — 2nd], [−2.13 dB — 2nd],
  [HTDemucs], [−10.10 dB — 3rd], [—],
  [DPTNet], [−33.39 dB — 4th], [—],
  [MP-SENet], [—], [*+2.27 dB — 1st*],
  [TF-GridNet], [—], [−2.57 dB — 3rd],
))

#v(0.5em)

On the held-out benchmark DCUNet's eSTOI is #bad[0.193] against an unprocessed
#good[0.233] — actively harmful to intelligibility.

= What this means

+ *Both published DCUNet wins are seen-noise results.* Neither is wrong; they
  measure a different thing from what we need.
+ *Do not quote the survey's absolute numbers as a held-out-noise target.* On
  unseen noise the honest figure is ≈+5 dB SI-SDR and ≈0 eSTOI gain.
+ *Stop trying to fix DCUNet.* The F2 recipe already recovered everything
  configuration could recover. The residue is generalisation.
+ *Treat noise-recording holdout as a first-class benchmark property* and state
  it in every results table.
+ *The F1 ranking stands.* MP-SENet and TF-GridNet are the architectures that
  generalise across rotor noise.

= Not established

- *Why* DCUNet specifically fails to generalise. Capacity (2.81 M params), the
  complex-mask formulation, no cross-band mixing — all untested candidates.
- Whether the *released* DN-LM data matches its published description (no data,
  no builder shipped).
- *Why our replication inverts the paper's ranking* — Edge-BS-RoFormer 3.55 dB
  behind DCUNet where the paper puts it 2.2 dB ahead. Same protocol, so probably
  our compute-limited training. Unresolved.
- Probe figures come from an epoch-7 checkpoint; the 12.9 dB gap cannot change
  sign, but exact values would shift at convergence.

#speaker-note[
  Being explicit about these matters more than the positive results — the
  headline claim is a leakage claim, so the limits of what we checked should be
  on the record.
]

= Backup — the retracted reading

We first diagnosed F1 DCUNet as having *collapsed to a near-null output*, because
its `sdr` sat near 0 dB at every input SNR. That inference is *invalid*.

For $hat(s) = alpha s + e$:

$ "SI-SDR" = alpha^2 ||s||^2 / ||e||^2 quad "SDR" = ||s||^2 / ((1-alpha)^2 ||s||^2 + ||e||^2) $

Both a near-null *and* an over-loud estimate reproduce any given pair. Measured:
the all-harmonic arm has `sdr` = −0.39 at −10 dB with `gain_db` = −0.22 — output
*at target level*.

Also `SI-SDR ≥ SDR` is *false*; the bound is $"SDR"_"lin" <= "SI-SDR"_"lin" + 1$.
