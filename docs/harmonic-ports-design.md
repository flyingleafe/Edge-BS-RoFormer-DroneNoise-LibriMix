# Porting HarmoF0, HPPNet and hFT-Transformer to rotor-speed estimation

Design note, 2026-08-31. Read before touching `src/models/harmonic_ports/`.

## The one substitution

All three architectures gather harmonic evidence, and all three do it on a
LOG-FREQUENCY axis, where harmonic `k` of a fundamental sits at a fixed offset
`log2(k) * B` bins and the gather is therefore a SHIFT — expressible as a
dilated convolution with weight sharing across pitch.

Read from the sources:

* HarmoF0 `MRDConv` — `dilation_list = round(log(k)/log(2^(1/B)))` for
  `k = 1..n_har`; per harmonic a 1x1 convolution, a shift along frequency, and a
  sum. At `B = 48` that is `[0, 48, 76, 96, 111, 124, 135, 144, ...]`.
* HPPNet `HarmonicDilatedConv` — eight `Conv2d(kernel=[1,3], dilation=[1,d])`
  with `d` in `48, 76, 96, 111, 124, 135, 144, 152`, i.e. the same offsets for
  `k = 2..9`.
* hFT-Transformer — no explicit harmonic prior; the decoder holds one output
  token per note and CROSS-ATTENDS to the frequency tokens, learning the
  harmonic pattern from data (`attention = [batch, frame, heads, n_note, n_bin]`).

Both harmonic models obtain the log axis the same way, and it is worth naming
because it is exactly the operation this note rejects: a uniform STFT
LINEARLY INTERPOLATED onto log-spaced frequencies (`WaveformToLogSpecgram`,
`log_idxs = fmin * 2**(idxs/bins_per_octave) / fre_resolution`), not a true CQT.

**The substitution: keep the linear STFT, and replace the shift with an explicit
gather at `k*r`.** `models.comb_salience.CombGather` already does this — it reads
the spectrogram at `k * r_g` for every (harmonic, candidate rate) pair, with the
read positions computed rather than assumed. The property that makes the log axis
attractive survives unchanged: the offsets are PROPORTIONAL to the hypothesis, so
one set of downstream weights serves every candidate rate. Weight sharing across
rate does not require a log axis; it requires read positions that scale with the
hypothesis, and indexing gives that directly.

## Why the log axis cannot be kept here

A log grid's spacing is `f * (2^(1/B) - 1)` — fine at the bottom of the band,
coarse at the top — while the discriminating information in this task sits at
HIGH harmonics. Two rotors `D` rev/s apart have their `k`-th harmonics `k*D` Hz
apart at frequency `k*r`, so under a log grid the separation-to-bandwidth ratio
is `D / (r * (2^(1/B) - 1))` — **`k` cancels**. Separability is the same at every
harmonic, and a rotor pair is either resolved everywhere or nowhere.

| B (bins/oct) | spacing at 30 Hz | at 2325 Hz | at 8 kHz | bins to 8 kHz |
|---|---|---|---|---|
| 48 (HPPNet) | 0.44 Hz | 33.8 Hz | 116 Hz | 387 |
| 208 | 0.100 | 7.76 | 26.7 | 1676 |
| 400 | 0.052 | 4.03 | 13.9 | 3224 |
| 1420 | 0.015 | 1.14 | 3.91 | 11443 |

DREGON's tightest cruise pair is 0.13 rev/s, whose 31st harmonics are 4.03 Hz
apart at 2325 Hz. Preserving that needs `B > 400`; never undersampling the linear
STFT anywhere needs `B >= 1420`, i.e. **11443 bins, 5.6x the 2049 linear bins**,
and every extra bin at the bottom is interpolated filler. Log resampling is lossy
below that threshold and wasteful above it. There is no setting at which it wins.

A uniform STFT does the opposite: separability improves LINEARLY with `k`, and
one 0.256 s window (n_fft 4096 at 16 kHz, 3.906 Hz bins) suffices everywhere —
0.85 rev/s separates from `k >= 5`, 0.39 from `k >= 11`, 0.13 from `k >= 31`.

**Corollary — do not use a log RATE grid either.** Its step is proportional to
the rate, so at 208 bins/octave it gives 0.100 rev/s at 30 rev/s and 0.501 at
150. DREGON's close pairs live at 74-86 rev/s, the coarse end: a log rate grid
spends its resolution where nothing needs it. Use a linear rate grid.

## Output grid: 0.5 rev/s is ample, and the reason is not obvious

With Gaussian-blurred targets and three-point log-parabolic peak interpolation,
the grid step contributes almost nothing, because the log of a Gaussian is a
parabola and three samples locate its vertex exactly. RMS recovery error is
`~ h * sigma_bins * 10^(-SNR/20)`, linear in all three:

| step h | 40 dB | 30 dB | 20 dB |
|---|---|---|---|
| 0.1 rev/s | 0.003 | 0.008 | 0.025 |
| 0.5 | 0.013 | 0.039 | 0.126 |
| 1.0 | 0.025 | 0.079 | 0.252 |

0.5 rev/s over 0-150 (300 bins) holds discretization at 0.013 rev/s at 40 dB and
0.13 at 20 dB — under the campaign's 0.2 rev/s honest floor and far under
DREGON's own +-0.6 rev/s label jitter. Use `sigma ~ 1 bin`, not 2: the error is
linear in blur width. These figures are the SINGLE-BUMP case, which is the
argument for per-rotor maps — on a shared map two rotors 0.13 rev/s apart merge
under any trainable blur and the interpolated centre is biased to their midpoint,
which is the regression fan reappearing in the output layer.

## What each port becomes

Shared front end for all three: STFT power (n_fft 4096, hop 512, linear) ->
running-median floor -> `CombGather` at `k*r` over a linear rate grid ->
`(B, K, G, T)`.

* **HarmoF0.** `MRDConv` -> gather plus a learned per-harmonic weight (its 1x1
  convolutions ARE that weight). Blocks 2-4 keep their structure but their
  dilations now run along the RATE axis, where an octave is not a fixed offset —
  use plain dilated context convolutions and say so. Head: 1x1 convs to R
  per-rotor maps instead of one monophonic map.
* **HPPNet.** Same front-end substitution; keep `FreqGroupLSTM` unchanged, now
  grouping over candidate rates rather than pitches — its job (a recurrence
  shared across the output axis) is unaffected.
* **hFT-Transformer.** The decoder already cross-attends from per-note output
  tokens to frequency tokens. Make the output tokens CANDIDATE RATES and use the
  gather as a structured sparsity prior on that attention: each rate token
  attends to its own K harmonics only. This is also what makes the model
  affordable here — 300 rate tokens x 32 harmonics is 9600 reads against 2049^2
  = 4.2M for full frequency self-attention over a linear STFT.

## Interface

The framework already has what is needed: task `salience_rps` (`forward(audio) ->
(B, F, T)` logits, `outputs_salience = True`, BCE training, Hungarian tracking at
eval), and `models.multif0.utils.linear_freq_grid` for a linear output axis.
Per-rotor maps need either R stacked into the frequency axis or a task variant;
start with ONE map and multi-hot targets, which is multi-pitch estimation and
needs no framework change, then add per-rotor maps as a variant.

## Sources

Read from the official repositories on 2026-08-31: `WX-Wei/HarmoF0`
(`harmof0/layers.py`, `harmof0/network.py`), `WX-Wei/HPPNet` (`hppnet/layers.py`,
`hppnet/nets.py`, `hppnet/constants.py`), `sony/hFT-Transformer`
(`model/model_spec2midi.py`). Both HarmoF0 and HPPNet already run at 16 kHz.
