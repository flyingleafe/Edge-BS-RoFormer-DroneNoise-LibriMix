# Harmonic multi-pitch architectures ported to the linear STFT

**Configs:** `conf/experiment/{hf0,hppnet,hft}_*.yaml`
**Design note:** [`docs/harmonic-ports-design.md`](../harmonic-ports-design.md)
**Code:** `src/models/harmonic_ports/`

## Why this batch exists

The project's rotor-speed regressors read a spectrogram and emit four numbers.
The multi-pitch literature solves a structurally identical problem — several
simultaneous harmonic sources, unknown count, dense frame labels — with
architectures built around the harmonic structure itself. Three were ported:

| port | source | the harmonic device |
|---|---|---|
| `hf0` | [HarmoF0](https://github.com/WX-Wei/HarmoF0) | `MRDConv`, multi-rate dilated convolution |
| `hppnet` | [HPPNet](https://github.com/WX-Wei/HPPNet) | `HarmonicDilatedConv` + `FreqGroupLSTM` |
| `hft` | [hFT-Transformer](https://github.com/sony/hFT-Transformer) | per-note cross-attention over harmonics |

**All three assume a log-frequency axis, and this project's axis is linear.**
Their harmonic device is a fixed dilation pattern that only lands on harmonics
when frequency is logarithmic. Rather than resample the STFT onto a log grid —
which undersamples high harmonics badly (separating them would need 11,443 bins)
— the ports replace that device with the campaign's comb gather, which reads the
spectrum at `k * r` directly on the linear axis. The gather IS a
harmonic-dilated convolution on a linear axis, so this is the same idea in the
representation the data actually has.

## The cells

Each trunk runs on three curricula: static comb only (`_comb`), stochastic comb
only (`_stoch`), and the real-data curriculum R4 (`_r4`). The `_l4` suffix marks
the four-per-rotor Gaussian salience layers with a CRF readout, which replaced
the shared triangular-kernel map after that pair was measured to lose 8.24 rev/s
on a PERFECT target (`models.salience_crf`). Rows without `_l4` predate the
measurement and are kept for the comparison.

## Conclusion

Both convolutional ports beat the regressor baselines on the two synthetic
families; the transformer did not converge. Read those numbers with the caveat
that they were produced on the OLD synthetic streams, whose training and
validation distributions differ (`freq_scale` on every training sample against
an unaugmented validation set) and whose validation set was 12 clips from ONE
trajectory. The rebuilt streams and the grid that replaces this batch are
[SALV2](./salv2-speech-and-objective-grid.md).
