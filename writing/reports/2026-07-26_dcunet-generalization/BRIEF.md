# Slide brief — the DCUNet generalization result

**Feed this file to a slide-editing session.** It is the single source of truth for
the speech-enhancement / DCUNet portion of any deck: every number with its
provenance, the epistemic status of every claim, the things that must *not* be
over-claimed, and three findings that were retracted mid-investigation and must
not be reintroduced.

Companion sources, in order of usefulness:

| what | where |
|---|---|
| Full narrative + figures | `writing/reports/2026-07-26_dcunet-generalization/report.typ` |
| Existing 13-slide deck | `writing/slides/2026-07-27_dcunet-generalization/slides.typ` |
| Experimental record, protocol, deviations | `docs/experiments/f2-survey-replication.md` |
| Per-clip data behind every number | `results/f2_perclip/*.csv` |
| Interactive probe (audio + spectrograms) | `notebooks/generalization_explorer.ipynb` |

---

## 1. The one-sentence result

DCUNet's published wins on drone speech enhancement are **seen-noise results**:
it scores well on noise recordings it was trained on and collapses on noise it
was not, whereas the architectures that beat it on our benchmark generalize.

## 2. The headline numbers

All at **−15 dB input SNR** unless stated. "Δ" is always against the unprocessed
mixture on the *same* clips.

### The replication (F2 step 1, `f2_dcunet_avq_survey`)

| metric | unprocessed | ours | published (survey) |
|---|---|---|---|
| SI-SDR | −15.05 dB | **+3.82 dB** | +3.7 dB |
| eSTOI | 0.126 | **0.408** | 0.4 |
| PESQ (wideband, 16 kHz) | 1.061 | 1.201 | — |
| PESQ (narrowband, 8 kHz) | 1.196 | 1.538 | 1.9 |

SI-SDR and eSTOI reproduce essentially exactly. **PESQ is short by ≈0.36 even
after the wideband→narrowband correction, and that residual is unexplained.**
Do not present PESQ as replicated.

### The mechanism (F2 step 1b, `f2_dcunet_avq_heldout`)

Train on AVQ session 1 only, hold session 2 out, score both. Same drone, same
held-out speakers, 250 clips per half.

| category | SI-SDR | ΔSI-SDR | eSTOI | ΔeSTOI | corr |
|---|---|---|---|---|---|
| `avq_ego_s1` — **seen in training** | **+3.60** | +18.53 | **0.339** | **+0.225** | 0.823 |
| `avq_ego_s2` — **never seen** | **−9.30** | +5.67 | **0.168** | **+0.046** | 0.344 |
| **gap** | **12.9 dB** | | **0.171** | | |

### Two controls that make the gap mean something

1. **The halves are equally hard unprocessed**: noisy SI-SDR −14.93 vs −14.97;
   noisy eSTOI 0.114 vs 0.122.
2. **The step-1 model (trained on all five recordings) scores them evenly**:
   +4.42 dB / eSTOI 0.391 on `s1` vs +4.12 dB / 0.375 on `s2` — a gap of
   **0.3 dB / 0.016**. So `s2` is not the harder half; it is the unseen half.

Without control (2) the whole argument is refutable. Any slide showing the
12.9 dB gap should show or mention it.

### Breadth is the same effect, not a second one

Same model/loss/valid clips, only the training pool widens (AVQ share of
training in brackets):

| arm | ΔeSTOI @−15 | ΔSI-SDR |
|---|---|---|
| AVQ only (100 %) | **+0.276** | +18.8 |
| + all drone (~14 %) | **+0.006** | +5.4 |
| + all harmonic (~2 %) | **−0.002** | +3.3 |

The probe's *unseen* half (−9.30 dB) lands on the broad-pool arms (−9.64 dB).
What governs the score is exposure to the **specific recording under test**;
breadth matters only because it dilutes that exposure.

### The architecture control

F1's drone training pool contains **no AVQ audio at all**, so MP-SENet had never
heard that drone:

| model | AVQ in training? | ΔeSTOI @−15 | SI-SDR |
|---|---|---|---|
| **MP-SENet** (broad drone pool) | **never** | **+0.342** | +3.11 |
| **DCUNet** (probe) | same drone, other session | **+0.046** | −9.30 |

MP-SENet generalizes to an unheard drone better than DCUNet generalizes between
two sessions of a drone it trained on. On `SE-valid-drone`, ΔeSTOI is
**MP-SENet +0.234, Edge-BS-RoFormer +0.100, DCUNet −0.040**.

### The ranking inverts with leakage

| model | DN-LM (leaked) | SE-valid-drone (held-out noise) |
|---|---|---|
| **DCUNet** | **−8.09 dB / STOI 0.541 — 1st** | **−10.88 dB — last** |
| Edge-BS-RoFormer | −9.94 / 0.529 — 2nd | −2.13 — 2nd |
| HTDemucs | −10.10 / 0.503 — 3rd | — |
| DPTNet | −33.39 / 0.302 — 4th | — |
| MP-SENet | — | **+2.27 — 1st** |
| TF-GridNet | — | −2.57 — 3rd |

On the held-out benchmark DCUNet's eSTOI is **0.193 vs 0.233 unprocessed** — it
is actively harmful to intelligibility, not merely unhelpful.

## 3. Why the prior benchmarks flattered DCUNet

**The 2023 survey** reuses its five ego-noise recordings between train and
validation by design, splitting only the speech.

**DN-LM** leaks by its published protocol. Liu et al., *Drones* 2025, §3.5:

> "speech and UAV noise samples were **randomly selected from** LibriSpeech and
> DroneAudioDataset …"
>
> "a **2 h synthetic dataset** … was constructed **and partitioned into training
> and validation sets at a 9:1 ratio**"

Mixtures are synthesised from the *full* pools first; the *mixtures* are then
split 9:1. No speaker or recording holdout is described, and a random split of a
single pool cannot create one. Their Figure 3 presents the two splits as
statistically identical (train −16.4 dB, valid −16.5 dB mean SNR), which is what
a random split yields. 2 h at 1 s = **7200 clips**; 9:1 = **6480/720** — our
re-creation's exact split sizes.

Measured on the same corpora: **714/720 (99.2 %)** of validation clips reuse a
training noise clip; all **257** underlying recordings appear in training;
**149/720 (20.7 %)** reuse an exact training utterance; speaker overlap ~100 %.

**Two caveats that must travel with this claim:**
1. We assess the *described protocol*, not the released data — the repo ships
   neither the dataset nor a builder (`datasets/` gitignored, no releases).
2. **We did not reproduce their ranking.** The paper reports Edge-BS-RoFormer
   **+2.2 dB** over DCUNet at −15 dB; on our re-creation it is **3.55 dB behind**
   — a 5.75 dB sign-reversed swing. Same protocol means this is probably our
   compute-limited Edge-BS-RoFormer training, not a dataset difference.
   **Unresolved.**

## 4. Retracted — do not reintroduce

Three claims were made during the investigation and then refuted by measurement.
They are wrong; if a slide says any of them, it is a regression.

1. **"SI-SDR ≥ SDR"** — false. The correct bound is
   `SDR_lin ≤ SI-SDR_lin + 1`.
2. **"F1 DCUNet collapsed to a near-null / silent output"** — refuted. `sdr ≈ 0`
   is reachable from *two* roots (near-null and over-loud); measured `gain_db`
   puts the output **at or above target level**. The failure is
   **decorrelation**, not silence. (`eval_se_perclip.py` now emits
   `gain_db`/`corr` so this is measured, never inferred.)
3. **"The noise pool is not the cause"** (the pre-flight diagnostic) — invalid
   inference. It scored a model already trained on the broad pool against a
   narrow valid set, measuring test-set difficulty rather than training-pool
   breadth.

A fourth, subtler one: an early framing said DN-LM's leak was *ours* rather than
the published protocol's. The paper's §3.5 settles it — the protocol leaks. But
caveat (1) in §3 above still applies.

## 5. Findings that postdate the report

From `notebooks/generalization_explorer.ipynb` (added after the report was
written — **not in report.typ**):

- **Pass-A checkpoints cannot demonstrate "fits in-distribution, fails out."**
  Those models trained on the *broad* pool, so their in-distribution condition is
  itself broad, and DCUNet's ΔeSTOI is **negative in all three conditions**
  (−0.058 / −0.044 / −0.027 at n=1). That is "weak everywhere". Only a
  **narrowly** trained model shows memorization.
- The notebook therefore has a **Part B** using `f2_dcunet_avq_heldout` on
  `avq_ego_s1` vs `avq_ego_s2`. Verified on CPU at n=2: eSTOI
  **0.449 → 0.202**, SI-SDR **+2.39 → −12.10 dB** — reproducing the report's gap.
- A new **`drone_seen`** category in `scripts/build_se_valid.py` mirrors the F1
  Pass-A *training* pool, since no published valid set has an in-distribution
  arm.

## 6. Claims to avoid

- ❌ "PESQ replicates" — it does not; ≈0.36 unexplained.
- ❌ "Edge-BS-RoFormer and MP-SENet don't memorize" — **not tested**. There is no
  narrowly-trained checkpoint for either. What *is* established is that both
  still improve intelligibility on held-out noise while DCUNet degrades it.
- ❌ "We know why DCUNet fails to generalize" — we localized it to the
  architecture but did **not** identify the mechanism. Capacity (2.81 M params),
  the complex-mask formulation, and the absence of cross-band mixing are all
  untested candidates.
- ❌ "The published papers are wrong" — they are not. They measure a different
  thing (seen noise). Both results are correct within their protocols.
- ❌ Quoting the survey's absolute numbers as a held-out-noise target. On unseen
  noise the honest figure is ≈+5 dB SI-SDR and ≈0 eSTOI gain.

## 7. Suggested slide arc (as built)

The contradiction → the exact replication → **the seen/unseen experiment** →
**the control** → breadth-as-dilution → the energy/intelligibility split → the
MP-SENet control → why the benchmarks leaked → the ranking inversion → what it
means → what is not established.

**Slides 3 and 4 are the argument.** Slide 3 shows the gap; slide 4 is the
control that makes it mean something. Dropping the control leaves the whole
claim open to "session 2 is just harder".

## 8. Figures available

In `writing/slides/2026-07-27_dcunet-generalization/assets/` (regenerate with
`python prepare.py`; PNGs are gitignored):

| file | shows |
|---|---|
| `seen_unseen.png` | the 12.9 dB gap, SI-SDR + eSTOI panels |
| `control.png` | trained-on-all-5 vs trained-on-S1, side by side |
| `ladder.png` | ΔeSTOI collapse vs ΔSI-SDR survival across pool breadth |
| `mpsenet.png` | MP-SENet vs DCUNet on unseen noise |

The report's `assets/energy_vs_intelligibility.png` (scatter at −15 dB) is also
useful and has no slide equivalent yet.
