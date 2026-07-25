---
experiment: f2_dcunet_avq_heldout
training_config: conf/experiment/f2_dcunet_avq_heldout.yaml
batch: docs/experiments/f2-survey-replication.md
---

# `f2_dcunet_avq_heldout`

## Motivation

Step 1 of this batch reproduces the survey's DCUNet result (`val/si_sdr` +3.2 dB
against the paper's +3.7 at −15 dB). Steps 2 and 3 show that broadening the
training noise pool destroys it. But there is a third possibility the ladder as
built cannot see, and it is the one that matters for whether the survey number
transfers to *our* problem at all:

**the survey trains and validates on the same 5 ego-noise recordings**, splitting
only the speech. So its DCUNet may have learned those particular recordings
rather than a denoiser that generalises to ego-noise it has not heard. Our F1
setup, by contrast, holds noise recordings out. If the replication depends on
noise reuse, then "DCUNet wins the benchmark" and "DCUNet is weakest on our data"
are not in conflict at all — they are the same model measured with and without
noise leakage, and no amount of configuration work on our side will close the
gap.

## Design

Byte-for-byte the step-1 experiment (same model `f2_dcunet_survey`, same
`si_sdr` loss, same 16 kHz / 3.0 s crops, same SNR range U(−25,−5), same Adam
1e-3 / plateau ×0.1 patience 5 / early stop 10 / batch 32 /
`samples_per_validation: 41580`) with exactly one change: **training noise is
AVQ session 1 only** (`S1_seq1`, `S1_seq2`, `S1_seq3` — policy
`conf/online_mix/se_avq_s1.yaml`), holding session 2 out of training entirely.

Validation is `SE-valid-avq-split`, built with the same seed, SNR grid,
duration and held-out speakers as `SE-valid-avq-survey` but carrying **two
categories**:

| category | ego-noise recordings | seen in training? |
|---|---|---|
| `avq_ego_s1` | `S1_seq1`, `S1_seq2`, `S1_seq3` | **yes** |
| `avq_ego_s2` | `S2_seq1`, `S2_seq2` | **no** |

Both halves are scored for the *same* model in the *same* eval, and the speech
is held-out on both sides, so the seen/unseen difference isolates the
memorisation term and nothing else. (The two sessions are different recording
sessions of the same drone, so "unseen" also means "different session" — that is
precisely the generalisation the F1 protocol demands.)

## What each outcome means

- **Small gap** (`avq_ego_s2` ≈ `avq_ego_s1`): DCUNet genuinely learned to
  suppress this drone's ego-noise. The survey number transfers, and the F1 floor
  is fully attributable to noise-pool breadth (steps 2–3).
- **Large gap** (`avq_ego_s2` collapses toward the noisy anchor): the survey's
  headline depends on train/valid noise reuse. Then the honest conclusion is that
  the published DCUNet result is not reachable under a held-out-noise protocol,
  and the F1 comparison was never unfair to DCUNet — it was simply stricter.

Either way the answer is quantified in dB rather than argued.

## Conclusion

_Pending — built and validated (`validate_only` passes); queued behind the
step 1–3 chains._
