---
experiment: hb_sal_multif0_l4
training_config: conf/experiment/hb_sal_multif0_l4.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hb_sal_multif0_l4`

## Motivation

Level L2 of the LateDeep row in Block S of the paper regime matrix (see the
batch doc § "Block S: the adaptation ladder of the multi-pitch baselines"):
per-rotor Gaussian layers on the 0-150 rev/s grid with the CRF readout,
ORIGINAL INPUT. Its L0 is the old `hb_sal_multif0` and its L1 the old
`hb_sal_multif0_nsr`; the regime is R4 as those rows ran it, the
`hb_silence_dload.yaml` pool.

`hb_sal_multif0` reads ONE SHARED salience map and decodes it by threshold plus
Hungarian assignment. `models.salience_crf` encoded real training telemetry
into that map and decoded it back — a PERFECT target, no model involved — and
got the trajectory back 8.24 rev/s away on average, with 39-45% of frames more
than half a bin off. Gaussian per-rotor layers read by a CRF plus a
log-parabolic vertex fit return 2.22e-16. The old number is an oracle floor,
thus the row measured the representation as much as the architecture. L2
removes that ceiling and leaves the architecture question intact.

Architecture: `multif0_salience_l4` — the LateDeep CNN over an HCQT front end,
`fmin` 32.7 Hz, 6 octaves, over-sample 5. The INPUT is `hb_sal_multif0`'s
exactly: same front end, same trunk, same [16000, 256] time grid. Only
`LateDeep`'s final 1x1 convolution and the super-resolution head widen to four
channels, which is 603 more parameters, and the output becomes four per-rotor
layers on a LINEAR 0-150 rev/s grid of 300 bins (`superres_out: true`; the CRF
band and the vertex fit are both defined on a uniform axis, and the HCQT grid
is log-spaced).

One known limit, and it belongs to L2's "original input" rule rather than to
this row. `FreqSuperResHead` clamps outside the input span, thus every output
bin below the input `fmin` reads the same input bin: the band under 32.7 rev/s
carries no independent evidence and the convolution stack has to separate it
on context alone. `conf/model/multif0_salience_nsr_hb.yaml` reports 4.9% of
the frozen split's rotor-frames nonzero and under 55 rev/s, which bounds that
share from above. `hb_sal_multif0_nsr` (L1) buys the band by moving the input;
L2 may not, because it isolates the output representation. L3, the comb gather
on the linear STFT, is the level that fixes it, and LateDeep has no L3.

Loss and metrics: `salience_layers_r150_h256` — the permutation-invariant BCE
over four Gaussian layers, and its validation twin plus `rps_mae`. They are the
hop-256 twins of `salience_layers_r150`, and the declared frame rate is the
only value that differs: this trunk emits salience at hop 256, the harmonic
ports at hop 512, and the hop is an input.

Monitor: `rps_mae`, minimized — the layers read in rev/s. `hf0_r4_l4` was
BCE-selected and had to be rerun as `hf0_r4_l4_v2` for that reason, so this row
selects on the rev/s quantity from the start. At eval the decode is
`models.harmonic_ports.layer_readout.LayerCRFReadout`: one CRF best path per
layer, no threshold and no Hungarian step.

Data: unchanged from `hb_sal_multif0` — `conf/online_mix/hb_silence_dload.yaml`,
validated on the frozen full-envelope real split
`dload:DREGON-LM-V4-michaels-valid-full`.

Train with `python train.py experiment=hb_sal_multif0_l4`.

Full batch context: [The paper regime matrix](../../docs/experiments/paper-regime-matrix.md).

## Conclusion

This arm's outcome is recorded in the batch write-up: [The paper regime matrix](../../docs/experiments/paper-regime-matrix.md).
