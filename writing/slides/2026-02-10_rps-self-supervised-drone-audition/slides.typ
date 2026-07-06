#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [RPS & Self-Supervised Drone Audition],
  subtitle: [Progress & research direction],
  author: [Dmitrii Mukhutdinov],
  date: [2026-02-10],
)

= Disclaimer

- *Time has been very tight*
- Full-time project at work with a *March deadline* is pressing
- I could spend *less than one working day per week* on this research
- Hence, I failed to perform all I intended last week
- However, I have formulated a better idea of what to focus on.

= Current idea — Overview

We want to use *rotor speed (RPS)* to improve drone noise modelling, but we face a *data imbalance*:

- *Very little* data with ground-truth RPS
- *Much more* drone audio without RPS (including a lot of ground-recorded material)

*Proposed direction*: self-supervised learning so we can use *both* kinds of data and get a model that benefits from RPS when available and still works without it.

The next slides unpack the motivation, the data situation, and the concrete idea.

= Motivation: Rotor speeds help

From the paper:

*"Enhancing drone audition with rotor-conditioned deep models"*

- Introducing *rotor speeds (RPS)* as a conditioning signal improves model performance
- Rotation-induced noise is physically tied to rotor RPM
- Conditioning on RPS lets the model use this structure explicitly

*Implication*: We want models that can use RPS when we have it.

= The data gap

- We have *very little* labelled RPS data; *much more* without (or speed-agnostic).
- *Goal*: Use both — supervised where we have RPS, self-supervision where we don't.

#table(
  columns: 2,
  table.header([Data type], [Dataset / source]),
  table.cell(rowspan: 2)[*With RPS* \ ~8.5 min],
  [DREGON ~5.5 min;],
  [Michaels' recordings ~3 min],
  table.cell(rowspan: 5)[*Without RPS* \ ~30 h],
  [Drone Audio Dataset — Alemadi, S. (2019). GitHub. #link("https://github.com/saraalemadi/DroneAudioDataset/tree/master")[github.com/saraalemadi/DroneAudioDataset]],
  [SPCup 19 Egonoise Dataset — Inria (2019). #link("http://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/")[dregon.inria.fr/.../the-spcup19-egonoise-dataset]],
  [DroneNoise Database — Ramos-Romero, C., et al. (2024). Salford. #link("https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411")[salford.figshare.com/.../22133411]],
  [AUDROK Drone Sound Data — AUDROK (2023). Mobilithek. #link("https://mobilithek.info/offers/605778370199691264")[mobilithek.info/offers/605778370199691264]],
  [Sound-Based Drone Fault Classification (MTL) — Yi, W., Choi, J.-W., & Lee, J.-W. (2023). Zenodo. #link("https://doi.org/10.5281/zenodo.7779574")[doi.org/10.5281/zenodo.7779574]],
)

= Core idea: Encoder predicts speed

- Use an *encoder–decoder* style architecture (e.g. built on RPS-DCCRN or similar).
- The *encoder* is tasked with *predicting rotor speed(s)* from the mixture (or from a representation of it).
- *Decoder* is conditioned on speed: when *ground-truth RPS* is available, we *replace* the decoder's speed input with it; otherwise we use the *predicted* RPS.
- Predicting RPS is an *auxiliary task* that encourages the model to internalise rotation-related structure.

When we have *ground-truth RPS*: train the encoder with a *supervised loss* (e.g. regression or classification over RPS).

When we *don't have RPS*: use a *regularization loss* that encourages the predicted speed to be *temporally smooth* (e.g. not change too much across short windows).

= High-level architecture

#text(size: 8pt)[
```mermaid
flowchart LR
  subgraph Input
    M["Noisy mixture"]
  end

  subgraph Model["Encoder–decoder"]
    E["Encoder"]
    R["Predicted RPS"]
    GT["Ground truth RPS"]
    D["Decoder"]
    E --> R
    R --> D
    GT -.->|"when available"| D
  end

  subgraph Output
    O["Enhanced audio"]
  end

  M --> E
  D --> O

  subgraph Losses["Loss (depends on data)"]
    direction TB
    L_enh["Enhancement loss (both cases)"]
    subgraph WithRPS["With RPS"]
      L_sup["Supervised RPS loss"]
    end
    subgraph NoRPS["Without RPS"]
      L_smooth["Smoothness regularization"]
    end
  end

  R -.->|"when labels"| L_sup
  R -.->|"when no labels"| L_smooth
  O --> L_enh
```
]

Decoder receives *ground-truth RPS* when available, else *predicted* RPS. With RPS → supervise encoder; without → regularize for smoothness.

= Loss design (summary)

#table(
  columns: 3,
  table.header([Data], [Loss], [Example]),
  [*With RPS*], [*Supervised loss* — match predicted speed(s) to ground truth], [L2: $norm(hat(r) - r^*)^2$],
  [*Without RPS*], [*Regularization loss* — temporal smoothness (first- & second-order diff.)], [$lambda_1 norm(Delta hat(r))^2 + lambda_2 norm(Delta^2 hat(r))^2$],
)

- Same encoder sees both types of data.
- With RPS: learn to predict real RPM.
- Without RPS: learn a stable, physically plausible "pseudo-speed" that still helps the decoder.
- _Notation:_ $hat(r)$ predicted speed, $r^*$ ground truth; $Delta hat(r)$ first-order diff., $Delta^2 hat(r)$ second-order (weights $lambda_1$, $lambda_2$).

= Why this could help

+ *Single model for both settings*
  - Can be used *with* RPS (conditioning on measured or predicted speed).
  - Can be used *without* RPS (encoder predicts speed; decoder uses it or we use a default).
  - Deployable in scenarios where RPS is sometimes available and sometimes not.

+ *Better use of physics*
  - Encoder is encouraged to explain the mixture in terms of rotation.
  - Internal representation of "speed" should be closer to the physical cause of rotor noise.
  - May lead to *better reduction of rotation-induced noise* than a model that never sees or predicts RPS.

= Next steps (concise)

+ *Replicate the paper with RPS-only data*
  - Reproduce results from "Enhancing drone audition with rotor-conditioned deep models" (or the chosen RPS-DCCRN baseline) using only the ~8.5 min of RPS-labelled data.
  - Establishes a baseline and validates the pipeline.

+ *Implement the self-supervised RPS model*
  - Implement the encoder (predict RPS) + decoder (enhancement) with:
    - Supervised RPS loss when labels exist.
    - Temporal smoothness (or similar) regularization when they don't.
  - Train on the *mixed* dataset (RPS + no-RPS).
  - Compare to the RPS-only baseline to see if mixed training improves performance.

= Summary

- *Disclaimer*: Limited time (work deadline in March); progress is incremental.
- *Motivation*: RPS conditioning helps; we want to exploit it despite limited RPS labels.
- *Idea*: Encoder predicts speed; supervised loss when RPS is available, regularization (e.g. smoothness) when it isn't; single model for both regimes.
- *Next*: Replicate paper on RPS-only data, then implement and train the self-supervised RPS model on mixed data and evaluate.

Questions and suggestions welcome.
