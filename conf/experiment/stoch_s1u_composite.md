---
experiment: stoch_s1u_composite
training_config: conf/experiment/stoch_s1u_composite.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1u_composite`

## Motivation

The composite of the two changes this campaign has measured to work, aimed at
per-regime parity with the real-trained target (zero 2.87, low 3.48,
flight 2.49).

**Both families in one stream, from arm S.** The analytic comb owns the stopped
rotors (4.73 rev/s, the best of any synthetic-only model) and the stochastic
family owns cruise (2.60, level with real training). Every arm before S replaced
one with the other and inherited its weakness with its strength. Running both
gives the best ramp cell the campaign has produced — 8.94 against a previous
best of 16.20, taking that cell from 4.66x the target to 2.57x — and from S's
epoch-2 checkpoint, so it is a floor rather than a converged value.

**Silence from inside those families, from arm T.** `scripts/zero_probe.py` fed
six inputs at one level to each model. The stochastic arms read the silence pool
correctly and then called their own combless floor a 39 to 46 rev/s rotor,
indistinguishable from the same floor at 12 rev/s:

| model | digital silence | own combless floor | same at 12 rev/s | silence pool |
|---|---|---|---|---|
| `m3abl_comb_scv2_s1` | -0.23 | **0.65** | 1.97 | -3.14 |
| `stoch_s1g_scv2` | 2.03 | **39.43** | 38.44 | 1.23 |
| `r4hb_scv2`, real | -1.26 | -1.25 | -1.40 | -1.00 |

They never learned that no comb means stopped; they learned that stopped means
the silence pool's texture, because nothing else in the stream resembled it. A
real stopped-rotor clip is the same room and the same microphones as the cruise
clips around it — the drone family with its comb switched off.

So the `kind: silence` source is removed. Every zero-labelled window is now one
of the two drone families with its rotors stopped: the comb family's is digital
silence, the stochastic family's is its own floor at the recording chain's level
through `floor_static_rel`. Absence of a comb is the only cue left.

Data `stoch_s1u`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1u_composite`.

## Conclusion

PENDING — the run has not finished.
