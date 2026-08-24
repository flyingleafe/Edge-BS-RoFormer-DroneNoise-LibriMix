---
experiment: hb_tr_ssq
training_config: conf/experiment/hb_tr_ssq.yaml
batch: docs/experiments/honest-base-frontends.md
---

# `hb_tr_ssq`

## Motivation

One cell of the HB grid (honest base + front-end comparison; see the batch
doc for the full design). Architecture: Transformer head (simple_conv_v2_transformer). Front-end: `stft_ssq`.
Voicing gate: yes. Data: the `hb` regime
(`conf/online_mix/hb_silence_dload.yaml`) — the fs_v2 real pool plus a
zero-labeled silence arm (16.7% of chunks: room tone, colored floors up to
flight level, LF rumble) and an SNR reference floor
(`snr_ref_floor_rms: 0.02`) that keeps speech audible on quiet chunks.
The grid answers two questions with one protocol: which front-end wins per
architecture, and does the honest regime + gate close the zero-regime gap.
Validation: the fixed full-envelope real split
`dload:DREGON-LM-V4-michaels-valid-full`. Train:
`python train.py experiment=hb_tr_ssq`.

## Conclusion

Pending.
