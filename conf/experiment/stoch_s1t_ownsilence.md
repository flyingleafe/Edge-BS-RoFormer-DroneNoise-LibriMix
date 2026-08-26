---
experiment: stoch_s1t_ownsilence
training_config: conf/experiment/stoch_s1t_ownsilence.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1t_ownsilence`

## Motivation

Every zero-labelled window comes from the stochastic family itself, with the
rotors stopped. The separate silence generator is removed.

The campaign could not explain why the analytic comb family reads a stopped
rotor at 4.73 rev/s while seeing 1.0% zero frames and carrying no silence
source, when the stochastic arms see about 19% zeros WITH a dedicated silence
pool and read the same clips at 20 to 28. `scripts/zero_probe.py` answers it by
feeding six inputs at one level and asking what each model calls them:

| model | digital silence | stoch floor | stoch 12 rev/s | silence pool | real zero |
|---|---|---|---|---|---|
| `m3abl_comb_scv2_s1` | -0.23 | **0.65** | 1.97 | -3.14 | 0.03 |
| `stoch_s1g_scv2` | 2.03 | **39.43** | 38.44 | 1.23 | 2.55 |
| `stoch_s1h_scv2` | 2.90 | **46.19** | 46.06 | 1.89 | 3.28 |
| `r4hb_scv2`, real-trained | -1.26 | -1.25 | -1.40 | -1.00 | -0.78 |

The stochastic models read the silence pool correctly and digital silence
correctly, and call their own combless floor a 39 to 46 rev/s rotor —
indistinguishable from the same floor at 12 rev/s. They never learned that no
comb means stopped. They learned that stopped means the silence pool's texture:
room tone, colored noise, low-frequency rumble, and nothing else in the stream
resembles it.

A real stopped-rotor clip is the same room and the same microphones as the
cruise clips around it — the drone family with its comb switched off, which is
precisely the input those models misread by 40 rev/s.

So `floor_static_rel` keeps the pool's own zero windows audible at the recording
chain's floor rather than at digital zero, and the `kind: silence` source is
removed. A model trained here can only tell a stopped rotor by the absence of a
comb, which is the cue that survives a room it has never heard.

Data `stoch_s1t`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1t_ownsilence`.

## Conclusion

PENDING — the run has not finished.
