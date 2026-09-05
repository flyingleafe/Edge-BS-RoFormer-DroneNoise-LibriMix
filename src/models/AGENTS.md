# src/models/ — Model Implementations

All neural network models live here.  This directory is an editable-installed
package (mapped to ``models`` in ``pyproject.toml`` hatch packages), so
``from models.X import Y`` works from anywhere in the project.

## Directory layout

```
src/models/
  __init__.py           (none — namespace package)
  frontends/            Pluggable spectral front-ends
  multif0/              Multi-F0 HCQT + CNN (Cuesta et al. ISMIR 2020)
  basic_pitch/          Basic Pitch note transcription, PyTorch port (Bittner et al. ICASSP 2022)
  harmonic_ports/       Multi-pitch architectures ported to rotor-rate salience on a LINEAR STFT
  rps_predictor.py      SimpleConv* family + DCUNet/DCCRN encoders (RPS)
  ckla.py               Complex Kalman Linear Attention head + SimpleConvV2CKLA (docs/ckla-design.md)
  dcunet.py             DCUNet (speech enhancement)
  dccrn.py              DCCRN
  dcunet_refactored.py  DCUNetRefactored, DCCRNRefactored
  demucs4ht.py          HTDemucs
  diffusion_buffer.py   DiffusionBufferModel
  dptnet/               DPTNet
  edge_bs_rof/          BSRoformer, MelBandRoformer
  generative/           RPS → noise generative models (DroneNoiseGen, …)
```

## Spectral front-ends

Located in `frontends/`.  Every front-end implements:

```python
class SpectralFrontEnd(nn.Module):
    out_channels: int
    def forward(self, audio: Tensor) -> Tensor:   # (B, N) → (B, C, F, T)
    def num_frames(self, n_samples: int) -> int:
```

Build via: `from models.frontends import build_frontend; fe = build_frontend(name, **kw)`

| Key | Class | `out_channels` | F (typical) | Description |
|-----|-------|----------------|-------------|-------------|
| `stft_mag` | STFTMag | 1 | n_fft//2+1 | log₁₊ magnitude |
| `stft_magphase` | STFTMagPhase | 3 | n_fft//2+1 | log mag + cos(θ) + sin(θ) |
| `stft_mag_if` | STFTMagIF | 2 | n_fft//2+1 | log₁₊ mag + instantaneous-frequency deviation (fractional bins; all-torch wrap, no numpy unwrap) |
| `stft_ssq` | STFTSSQMag | 1 | n_fft//2+1 | log₁₊ of the synchrosqueezed magnitude — STFT power scattered along frequency onto the rounded instantaneous-frequency bin (`stft.if_deviation_bins`), then `sqrt`. Same grid as `stft_mag`, energy-conserving, no parameters; sharper comb ridges at low frequency, and a 1-channel alternative to `stft_mag_if` |
| `hcqt` | HCQTFrontEnd | H or 2H | 360·bpo/60 | Harmonic CQT (librosa). `phase=True`→2H |
| `pyramid_if` | PyramidIFFrontEnd | 2 (8 if `collapse_bands=False`) | 340 (log-f rows) | Multi-resolution STFT pyramid: 4 bands (n_fft 8192/4096/2048/1024, 30-250/250-1000/1-2k/2-4k Hz), per-band log1p-mag + IF, fixed-interp onto a geometric log-f axis + hop-512 grid; dense band-sum default (G8a2), per-band concat = dead G8a |
| `comb_if` | CombIFFrontEnd | 4 (3 if `coord_channel=False`) | 361 (f0 rows) | Whitened comb matched-filter + Fisher-weighted IF consensus + occupancy + CoordConv row-f0 channel (G4b) over a 30..120 rev/s ×0.25 candidate-f0 grid (VK-scan analogue, linear grid, teeth ≤1200 Hz) |

### Using a front-end

```python
from models.frontends import build_frontend

fe = build_frontend("stft_mag", n_fft=2048, hop_length=512)
features = fe(audio)  # (B, 1, F, T)

# HCQT with custom resolution (120 bins/octave = 10¢)
fe = build_frontend("hcqt", phase=True, over_sample=10, harmonics=[1,2,3,4,5])
features = fe(audio)  # (B, 10, 720, T)
```

### Adding a new front-end

1. Subclass `SpectralFrontEnd`, set `key` and `out_channels`.
2. Decorate with `@register_frontend`.
3. Import the module from `frontends/__init__.py::_ensure_imported`.
4. It's instantly available via `build_frontend(key)`.

## Model type registry (enhancement models)

From `models.registry.LEGACY_MODEL_BUILDERS` (the dict-ified legacy `model_type` dispatch; the unified listing across all registries is `models.registry.model_types()`).

| Key | Model | File | RPS support |
|-----|-------|------|-------------|
| `edge_bs_rof` | Edge-BS-RoFormer | `edge_bs_rof/` | No |
| `dcunet` | DCUNet | `dcunet.py` | Yes |
| `dcunet_refactored` | DCUNetRefactored | `dcunet_refactored.py` | Yes |
| `dccrn` | DCCRN | `dccrn.py` | Yes |
| `dccrn_refactored` | DCCRNRefactored | `dcunet_refactored.py` | Yes |
| `dptnet` | DPTNet | `dptnet/` | No |
| `htdemucs` | HTDemucs | `demucs4ht.py` | No |
| `diffusion_buffer` | DiffusionBufferModel | `diffusion_buffer.py` | No |

## RPS prediction models

Registered in `rps_predictor.py::RPS_MODEL_REGISTRY` (also re-exported, plus the
salience/multif0 variants, from `registry.py::RPS_MODEL_REGISTRY` — the single
`build_model(name, **params)` entry point used by the unified `train.py`/`eval.py`).

| Key | Class / description |
|-----|---------------------|
| `simple_conv` | Baseline — 5-block encoder + Conv1d head |
| `simple_conv_v2` | Residual + SE + attention pool + BiGRU |
| `simple_conv_v2_tcn` | `simple_conv_v2` encoder/pool + symmetric dilated TCN head |
| `simple_conv_v2_causal_tcn` | `simple_conv_v2` encoder/pool + left-padded dilated TCN head (head-only causal) |
| `simple_conv_v2_smol_tcn` | `simple_conv_v2` encoder/pool + SMoLnet-style frequency-dilated refinement + symmetric TCN head |
| `simple_conv_v2_smol_causal_tcn` | `simple_conv_v2` encoder/pool + SMoLnet-style refinement + left-padded TCN head |
| `smolnet_rps_tcn` | SMoLnet-style compressed re/im STFT backbone + attention frequency pool + symmetric TCN head |
| `smolnet_rps_simple_head` | SMoLnet-style compressed re/im STFT backbone + SimpleConv-style mean frequency pool and shallow Conv1d head |
| `smolnet_rps_causal_tcn` | SMoLnet-style compressed re/im STFT backbone with left-padded late layers + left-padded TCN head |
| `simple_conv_v2_uni_gru` | `simple_conv_v2` encoder/pool + unidirectional causal GRU head (head-only causal) |
| `simple_conv_v2_uni_gru128` | `simple_conv_v2_uni_gru` with hidden size 128 to match BiGRU output width |
| `simple_conv_v2_uni_gru128_norm` | `simple_conv_v2_uni_gru128` with GroupNorm after the causal Conv1d prenet |
| `simple_conv_v2_uni_gru128_norm_do03` | `simple_conv_v2_uni_gru128_norm` with stronger head dropout (`0.3`) |
| `simple_conv_v2_uni_gru96_norm_do03` | Normalized unidirectional GRU head with hidden size 96 and dropout `0.3` |
| `simple_conv_v2_uni_gru96_norm_do02` | Normalized unidirectional GRU head with hidden size 96 and dropout `0.2` |
| `simple_conv_v2_uni_gru64_norm_do03` | Normalized unidirectional GRU head with hidden size 64 and dropout `0.3` |
| `simple_conv_v2_causal_gru` | Time-causal STFT framing + left-padded temporal conv encoder + unidirectional GRU |
| `simple_conv_v2_causal_gru96` | `simple_conv_v2_causal_gru` with wider unidirectional GRU (`hidden_ch=96`) |
| `simple_conv_v2_transformer` | `simple_conv_v2` encoder/pool + Transformer temporal head |
| `simple_conv_v2_transformer_hcqt` | `simple_conv_v2_transformer` trunk on a harmonic-stacked HCQT front-end (nnAudio, 16 kHz native, fmin 32.7, harmonics [1,2,3] ⇒ 6 ch; hop-256 features time-interpolated onto the hop-512 output grid) — G2a VK-parity front-end arm |
| `simple_conv_v2_transformer_if` | `simple_conv_v2_transformer` trunk on the `stft_mag_if` front-end (log-mag + IF deviation, 2 ch, same STFT grid) — G2b VK-parity front-end arm |
| `simple_conv_v2_transformer_pyramid` | `simple_conv_v2_transformer` trunk on the `pyramid_if` multi-resolution pyramid front-end (dense 2 ch × 340 log-f rows, zero front-end params; `collapse_bands=False` → 8-ch dead-G8a variant) — G8a/G8a2 VK-parity arm |
| `simple_conv_v2_transformer_comb` | `simple_conv_v2_transformer` trunk on the `comb_if` front-end (whitened comb score + IF consensus + occupancy + coord channel, 4 ch × 361 f0 rows; `coord_channel=False` → 3-ch G4a variant) — G4/G4b VK-parity front-end arm |
| `simple_conv_v2_ckla` | `simple_conv_v2_transformer` trunk with the temporal head replaced by a complex-KLA scan head (`ckla.py::TemporalCKLAHead` — flat complex-OU Kalman linear attention, input-dependent per-slot rotation; docs/ckla-design.md) on the `stft_mag_if` front-end |
| `simple_conv_v2_ckla_mag` | `simple_conv_v2_ckla` on the plain `stft_mag` front-end (1 ch) — front-end interaction ablation (design §5 ladder item 5) |
| `simple_conv_v2_ckla_phaseonly_cond` | Conditional RPS **refiner** (`ckla.py::SimpleConvV2CKLACond`): the phase-only CKLA backbone with a corrupted-RPS conditioning input — `(4, F)` track MLP-embedded per frame, concatenated to the pooled trunk features before the temporal head — and a bounded residual output `cond + max_delta·tanh(head)`. `forward(audio, cond)`; output rotor order == conditioning order, trained with plain non-PIT MSE (`losses.RPSMSELoss`) on `(audio, corrupt(GT)) → GT` pairs from `data_processing/rps_corruption.py` |
| `simple_conv_v2_fkla` | `simple_conv_v2_ckla` wrapper with the temporal mixer replaced by the vendored plain-KLA flat layer (`fkla/` — kla-loglinear@11e5a39, real OU state, no rotation by construction) — cross-implementation ablation companion to the `_norot` controls |
| `simple_conv_v2_local_attn` | `simple_conv_v2` encoder/pool + local-window Transformer temporal head |
| `simple_conv_v2_multires` | `simple_conv_v2` with concatenated long/short-window STFT magnitude inputs |
| `simple_conv_v2_dwt` | `simple_conv_v2` with a lightweight Haar-like temporal wavelet branch |
| `simple_conv_v2_magphase` | `simple_conv_v2` using log-magnitude plus cosine/sine phase STFT channels |
| `simple_conv_v2_dual_pool` | `simple_conv_v2` concatenating attention and mean frequency pooling before BiGRU |
| `simple_conv_v2_gru96` | `simple_conv_v2` with a wider BiGRU temporal head (`hidden_ch=96`) |
| `simple_conv_wide` | Wider/deeper baseline |
| `simple_conv_tcn` | TCN head (dilated convs) |
| `simple_conv_multiscale` | FPN-style multi-scale fusion |
| `simple_conv_bigru` | Baseline encoder + BiGRU head |
| `simple_conv_bigru_v2` | Deeper + BiGRU head |
| `simple_conv_magphase_bigru` | 3-channel (mag+phase) + BiGRU |
| `simple_conv_attn_pool` | Attention frequency pooling |
| `simple_conv_se_next` | SE + residual + deeper |
| `dcunet_enc_rps` | DCUNet complex-conv encoder + `RPSPredictionHead` (from `models.dcunet`) |
| `dccrn_enc_rps` | DCCRN complex-conv encoder + `RPSPredictionHead` (from `models.dcunet`) |
| `multif0_rps` | Multi-F0 LateDeep CNN + soft-centroid RPS |
| `multif0_salience` | LateDeep CNN → salience-map logits; BCE-trained, Hungarian-tracked to RPS at eval (`salience_rps.py`) |
| `basic_pitch_salience` | Basic Pitch contour branch → salience-map logits; same BCE+tracking path, native 16 kHz (`salience_rps.py`) |
| `harmof0_rps` | HarmoF0 (Wei et al. ISMIR 2022) with its log-frequency harmonic SHIFT replaced by a gather at `k*r` on the linear STFT → salience logits on a linear CANDIDATE-RATE grid (`harmonic_ports/harmof0_rps.py`) |
| `hppnet_rps` | HPPNet (Wei et al. ISMIR 2022) with `HarmonicDilatedConv` (eight log-axis dilated branches) replaced by the same gather at `k*r`; `CNNTrunk` and `FreqGroupLSTM` kept, MPE head only (`harmonic_ports/hppnet_rps.py`) |
| `hft_rps` | hFT-Transformer (Toyama et al. ISMIR 2023) with its per-note decoder tokens made CANDIDATE RATES and its cross-attention hard-masked to each rate's own harmonics, read at `k*r`; MPE head only (`harmonic_ports/hft_rps.py`) |
| `harmof0_orig` | HarmoF0 (Wei et al. ISMIR 2022) UNMODIFIED — its own log-interpolated STFT front end, `MRDConv`, the octave-dilated blocks 2-4, and a 352-bin log salience map at 48 bins/octave from 27.5 Hz. The CONTROL for `harmof0_rps` (`harmonic_ports/harmof0_orig.py`) |
| `hppnet_orig` | HPPNet (Wei et al. ISMIR 2022) UNMODIFIED — nnAudio CQT, `HarmonicDilatedConv`, `CNNTrunk`, `FreqGroupLSTM`, frame head only, on the same 352-bin log grid. The CONTROL for `hppnet_rps` (`harmonic_ports/hppnet_orig.py`) |

All SimpleConv* models now accept a `frontend=` kwarg.  Old checkpoints are
loadable via automatic `window` → `frontend.window` remap.

### Voicing gate and string front-ends (the honest-base grid)

The three grid architectures — `simple_conv_v2` (`BiGRUHead`),
`simple_conv_v2_uni_gru128` (`CausalGRUHead`) and
`simple_conv_v2_transformer` (`TemporalTransformerHead`) — take two more
constructor keywords, both reachable from a `conf/model` `params:` block.
`frontend=` also accepts a front-end registry **key as a string**
(`frontend: stft_mag_if`), which the model builds with its own
`n_fft`/`hop_length`; the first encoder block then adapts its input width to
the front-end's `out_channels`, so the default 1-channel `stft_mag` case stays
weight-identical to older checkpoints. `voicing_gate=True` replaces the head's
final `nn.Linear` with `GatedProjection` (`rps_predictor.py`), which emits
`speed * sigmoid(gate_logit)` from one `Linear` to `2*num_rotors`: a stopped
rotor becomes a classification decision instead of an MSE-mean regression to a
false hover. `voicing_gate=False` (the default) keeps the attribute name
`head.proj` and its `weight`/`bias` keys, so existing checkpoints load
unchanged; the gated variant nests them under `head.proj.linear.*` and is not
weight-compatible. Ten grid configs use them:
`conf/model/hb_{scv2,tr,gru}_{mag,if,ssq}.yaml` (architecture × front-end, gate
on) plus `conf/model/hb_scv2_mag_nogate.yaml` (the gate control). Tests:
`tests/models/test_voicing_gate.py`.

Causal RPS gotcha from autoresearch session `20260617-012233`: simply swapping
`BiGRUHead` for a unidirectional GRU was unstable/poor. The best causal-head
variant in that sweep was `simple_conv_v2_uni_gru96_norm_do03` (GroupNorm +
dropout 0.3), still worse than `simple_conv_v2`. Fully time-causal STFT +
left-padded temporal conv variants underfit badly, likely due alignment/latency
and loss of future context; treat them as a separate front-end/alignment problem,
not just a head replacement. The external SMoLnet reference
(`../drone-audition/drone_audition/models/smolnet.py`) is frequency-dilated in
its early `(kernel, 1)` Conv2d layers and uses symmetric time padding in late
square layers, so it is not strictly causal as written. When adapting a new
backbone such as SMoLnet to RPS prediction, run the cleanest body-only ablation
first (body + SimpleConv-style mean-pool Conv1d head) before adding stronger
TCN/GRU/attention heads; otherwise body and head effects are confounded.

### Harmonic ports (`harmonic_ports/`)

Paper multi-pitch architectures, each with ONE organ replaced: the
log-frequency harmonic **shift** becomes an explicit **gather** at `k*r` on the
linear STFT (`models.comb_salience.CombGather`). The measurements that reject
the log axis for this task live in `docs/harmonic-ports-design.md` — read it
before touching this package; the short version is that a log grid's
separation-to-bandwidth ratio for two rotors `D` apart is
`D / (r * (2^(1/B) - 1))`, in which the harmonic index cancels, so a rotor pair
is resolved at every harmonic or at none, while a uniform STFT improves
linearly with `k`. The output axis is therefore the CANDIDATE RATE, not
frequency, and it is **linear** (a log rate grid spends its resolution at the
coarse end, where nothing needs it).

**Per-rotor layers (`n_maps: 4`) are the default for new rows.** The framework's
shared salience map is not a lossless encoding of this task: `models.salience_crf`
encodes real training telemetry and decodes it back — a PERFECT target, no model
involved — and returns it **8.24 rev/s** away on average, with 39-45% of frames
more than half a bin off, against **2.22e-16** for Gaussian per-rotor layers read
by a CRF plus log-parabolic fit. Three causes: one map cannot hold four rotors
whose pairs sit inside one 0.5 rev/s bin; a triangular kernel has no exact
sub-bin readout; and `active = rps_grid > 0.1` encodes a stopped rotor as an
ABSENCE, which is what forces a decode threshold. The `_l4` model configs
(`conf/model/{harmof0,hppnet,hft}_rps_l4.yaml`) set `n_maps: 4` and must be paired
with `conf/loss/salience_layers_r150.yaml` and
`conf/metrics/salience_layers_r150.yaml`; `models.harmonic_ports.layer_readout`
then overrides `predict_rps` with one CRF best path per layer — no threshold, no
Hungarian step. The layers ride the codec's `(batch, freq, time)` wire format
stacked along the output axis (width `4 * 300`), because a 4-D model output does
not type-check through `SalienceRPSCodec`. With `n_maps: 1` everything falls back
to the old shared-map path unchanged.

They satisfy the ordinary `salience_rps` contract — `forward(audio) -> (B, G, T)`
logits, `outputs_salience = True`, BCE through `losses.SalienceRPSBCELoss`,
Hungarian tracking through the inherited `predict_rps` — by declaring the rate
grid as `SalienceRPSPredictor.out_freqs`, the hook that already exists for a
salience axis decoupled from a log-spaced input CQT. Nothing in the task, the
codec, the loss or the tracker changes.

**The two controls.** `harmof0_orig` and `hppnet_orig` are the same two papers
with NOTHING replaced — the published harmonic device, the published front end,
and a 352-bin log grid at 48 bins/octave from 27.5 Hz — wired into the same
`salience_rps` task. They exist because every port row so far has been read
against the direct REGRESSORS, which share neither the trunk nor the output
representation, so no measurement yet separates the substitution from the trunk.
Both emit a bit-identical grid, so `conf/loss/salience_bce_orig.yaml` and
`conf/metrics/salience_bce_orig.yaml` serve both arms; the experiments are
`hb_sal_{hf0,hppnet}_orig` and the batch doc is
`docs/experiments/paper-regime-matrix.md` § "Block S", where they are level L0
of the multi-pitch adaptation ladder and the `*_rps` ports are level L3. Their harmonic blocks are
checked bit-identical against the upstream source in
`tests/models/test_harmonic_orig.py`; the remaining deviations are seam-level
(the hop-512 frame grid, logits instead of a sigmoid, HPPNet's piano-specific
heads and its two pools) and are listed in the two module docstrings. Under
`f0 = rps` that log grid spans 27.5-4371 rev/s, of which rotors occupy bins
0-118 of 352, at 1.45% of the rate per bin — which is the cost the pair is there
to measure.

| Model | Paper | What was replaced |
|-------|-------|-------------------|
| `harmof0_rps` | HarmoF0, Wei et al. ISMIR 2022 | `MRDConv` (a 1x1 conv, a `round(log2(k)*B)`-bin shift, and a sum, per harmonic) → `CombGather` at `k*r` times a learned per-harmonic weight. Its blocks 2-4 keep their shape but their octave-sized dilations become plain dilated context convolutions along RATE, where an octave is not a fixed offset — a deliberate deviation, documented in the module docstring |
| `hft_rps` | hFT-Transformer, Toyama et al. ISMIR 2023 | Nothing structural — hFT already holds ONE DECODER TOKEN PER NOTE that cross-attends to the frequency tokens (`attention = [batch, frame, heads, n_note, n_bin]`). The tokens become CANDIDATE RATES and the gather becomes a hard mask on that attention: a rate token sees only its own K harmonics. hFT's frequency self-attention ENCODER is deleted, because nothing else is visible; `attn_mode: bias` restores it over a pooled 256-token spectrum with the gather as a learned additive bias instead (variant (ii) of the design note). Only the frame/MPE head survives — onset, offset and velocity are piano-specific |

Two traps this package has already paid for, both specific to a rate grid that
reaches 0 and both handled inside `harmof0_rps.py`:

1. **The near-DC gather.** At 1.5 rev/s every harmonic lands inside the STFT
   window's DC mainlobe, reads far above a median floor computed from that same
   mainlobe, and wins. `f_min` (30 Hz by default) drops those reads. The
   classical scan never needed this because it searches 30-100 rev/s.
2. **Count normalization.** The classical head divides the summed evidence by
   the number of in-band harmonics, which is right in a narrow search band and a
   trap on a 0-150 grid: a candidate with three surviving harmonics wins on one
   lucky hit. Dividing by the constant `k_max` instead took the untrained
   score from 3/8 to 7/8 on a synthetic single-comb probe.
3. **The unresolvable candidate.** Consecutive harmonics of a rate are that
   rate apart in Hz, so below one STFT bin (3.906 Hz at n_fft 4096) they land
   in the same bin and the hypothesis is not a comb — its K reads are a handful
   of bins inside one strong low line's skirt. `f_min` does not catch it: those
   reads sit at 30-48 Hz, outside the DC mainlobe. `hft_rps.py`'s `r_min`
   (default `sr / n_fft`) drops every harmonic of such a candidate, and with it
   the untrained read of a 37 rev/s comb moves from 1.51 rev/s to 37.12; 45, 60,
   84.5 and 120 also land within one grid bin.

### Salience-map RPS baselines (`salience_rps.py`)

`multif0_salience` and `basic_pitch_salience` are *multi-pitch* baselines: they
output per-bin salience **logits** `(B, n_bins, T)` (flagged `outputs_salience=True`),
not RPS directly. The unified trainer (`train.py`) routes them to a BCE path by
loss selection — pick a `conf/loss` entry that targets salience (BCE-on-salience,
`src/losses/salience.py`) instead of the PIT-MSE loss used by the direct-RPS
models. `rps_to_salience()` builds the per-bin target (precomputed/cached in the
dataset; blurred via a blur-bins parameter), trained with `BCEWithLogitsLoss`
(pos-weight parameter).
At eval, `predict_rps()` does `sigmoid → salience_to_rps_segmented` (Hungarian
tracking, `--track_threshold`) → STFT grid, so the existing global-PIT metrics
(PIT MSE/RMSE/MAE/R²) apply unchanged and stay comparable to the SimpleConv family.
Both run natively at 16 kHz. The RPS↔salience helpers live in `multif0/utils.py`.

**The zero convention (both directions).** A stopped rotor (`rps <= 0.1`) is the
only case the *target* leaves dark — a rotor that is slow but running is
quantized ONTO the lowest bin, not dropped, so a grid whose `fmin` sits above
the ramp speeds teaches a false speed there rather than losing the frame. On the
*decode* side a frame with no peak above `track_threshold` emits **0 rev/s for
every rotor**: silence == zero rotor speed, never a hold-over of the last speed
and never NaN (`_hungarian_tracking` / `_track_rotors`; tests in
`tests/models/test_salience_rps.py`). Track identity survives a dark frame — only
the emitted value is zeroed — so a momentary dropout does not restart the tracks.

`multif0_salience`'s HCQT `fmin` defaults to **27.5 Hz (A0)** — matching
basic-pitch, low enough to cover rotor fundamentals below C1 — and is settable
via `--hcqt_fmin`. The grid descriptor is read back from the front-end, so
changing `fmin` auto-reshapes the salience target and the tracker. (At 16 kHz,
`fmin=27.5` auto-derives 4 harmonics `[1,2,3,4]`; lower `fmin` → more.)

`--fused_branches` runs LateDeep's two identical mag/phase branches as a single
grouped (`groups=2`) stack (`LateDeep(fused_branches=True)`): mathematically
identical (verified to float32 precision in the since-removed `test_multif0.py` smoke test), one kernel launch
per layer instead of two, and the channel concat becomes free. Checkpoints
convert between the two layouts transparently via a `load_state_dict` pre-hook on
`LateDeep`, so a model trained either way loads either way. Same FLOPs — the win
is launch overhead, so benchmark on GPU (`scripts/bench.py --target grouped_branches`) before
relying on it; `groups=2` can regress on some cuDNN versions.

`--stacked_hcqt` (`LateDeepSalience(stacked=True)`, which rides through to
`build_frontend("hcqt", stacked=True)` → `HCQTFrontEnd(stacked=True)`) uses
`HCQTStacked_nnAudio`: **one** CQT (extra high bins) + harmonic freq-shifts of mag
and phase, instead of one CQT per harmonic. ~2× faster front-end on GPU
(`scripts/bench.py --target cqt`), same `(mag, dphase)` contract and grid. It is a **lossy
approximation** at higher harmonics (h=3 mag corr ~0.977), so the features differ
— **train from scratch**, do not load a non-stacked checkpoint into it. Composes
with `--fused_branches`.

## Multi-F0 (Cuesta et al. ISMIR 2020)

Located in `multif0/`.  Pure PyTorch reimplementation.

| Class | Description |
|-------|-------------|
| `EarlyShallow` | Early-fusion, shallow CNN |
| `EarlyDeep` | Early-fusion, deep CNN |
| `LateDeep` | Late-fusion, deep CNN (best reported) |
| `LateDeepNoPhase` | LateDeep without phase input |
| `MultiF0RPSPredictor` | Wraps `LateDeep` for RPS prediction (HCQT→CNN→soft-centroid→RPS) |

HCQT is in `multif0/hcqt.py` (librosa-based, reference implementation).
A GPU-compatible version via nnAudio is in `multif0/nnaudio_cqt.py`.

## Generative models (RPS → noise)

In `generative/`.  Not registered in `get_model_from_config`.  Full sync of
`drone_audition.models` (no `env.settings`; sample rate is an explicit arg,
default 16 kHz).  `DroneNoisePlusFilterGen` was used by the deleted `train_noise_gen.py`
(no unified-framework `conf/model`/training entry point for it yet);
`VP_transform` is used by `src/utils/align_rps.py`.

| Class | File | Purpose |
|-------|------|---------|
| `DroneNoiseGen` | `harmonic_noise_gen.py` | Per-rotor harmonic oscillator bank |
| `PropellerNoiseGen` / `PolynomialRegression` / `PolyWithExpLog` | `harmonic_noise_gen.py` | Single-rotor bank + scalar gain regressors |
| `DroneNoisePlusFilterGen` | `filtered_noise.py` | DroneNoiseGen + RPS-conditioned filtered-noise residual (port-only) |
| `HarmonicTransformModule` + `VP_transform` / `lstsq_VP_transform` / `inverse_VP_transform` / `harmonic_VP_transform` | `harmonic_transform.py` | VP-transform harmonic analysis/synthesis (dot-product **and** least-squares projection; zero-guard + per-frame antialias + `center`) |
| `HarmonicNoiseGenNew` | `harmonic_gen_new.py` | End-to-end RPS→audio: NN predicts harmonic amps + noise mags → oscillator bank + filtered noise |
| `JointAmplitudePredictor` / `ConstantAmplitudePredictor` / `DirectionalOutputHead` / `SpeedsPostprocessingWrapper` / `LearnableTimeShift` | `harmonic_gen_new.py` | Amplitude predictors + helpers for `HarmonicNoiseGenNew` |
| `PositionalHarmonicNoiseGen` + `propagate` / `fractional_delay` | `positional_harmonic_gen.py` | Position-aware generator: single-rotor `HarmonicNoiseGenNew` (rotor folded into batch) **emits** per-rotor sources, then **propagates** to observation point(s) with 1/r attenuation + fractional delay (`r/c`, c=343). Native multi-observer — rotors summed in the rfft domain, so M mics cost R fwd + M inv transforms. Differentiable w.r.t. position. Isotropic point source (distance-only). Per-drone conditioning is **external** (`cond_dim=d` FiLM-conditions the emitter on a code `z (B,d)` passed to `forward`; the `name→z` table is a separate `tasks.noise_generation.DroneCodebook`, so model params never resize with drone count and an unseen drone is few-shot-adaptable by freezing the model and fitting just its code). `amp_stats(rps, rel_pos, z)` returns the per-`(mic, rotor, harmonic)` amplitude ENVELOPES (1/r gains only — no delay, no rotor sum, no synthesis, and jitter-free by construction) plus the power-summed per-mic broadband envelope: the training path of the Vold-Kalman amplitude objective, and ~100x cheaper than a render. `build_noise_gen_model(amp_calibration=True, n_mics=…, noise_floor_bands=…)` adds the per-drone absolute-level gains it needs (global, per-mic, a separate power-domain constant and a static per-mic per-band floor for the broadband branch — per-rotor attribution of the residual is refuted, see docs/experiments/residual-attribution.md); they apply to EVERY prediction path, so a calibrated model also renders at the recording's level. `forward(..., return_dict=True)` also exposes the emitter's per-rotor control curves (`harm_amps` `[B,R,O,H,t]`, `noise_amps` `[B,R,F,t]`) — the inputs to the Stage-2 smoothness regularisers (`losses.smoothness_penalty`, squared 2nd difference; harmonic amps over time, noise shape over time+freq). Was wired via the deleted `train_noise_generation.py` (`--harm_smooth_weight`/`--noise_smooth_weight`, both default 0 = off; validation stayed pure spectral); now routes through the unified `train.py` — `conf/experiment/e2_noise_gen_dregon_michaels.yaml` (no smoothness) / `e3_noise_gen_swapped_smoothness.yaml` (`losses.SmoothnessPenalty` on `harm_amps`/`noise_amps`, exposed via `tasks.codecs.NoiseGenerationCodec(return_dict=True)`) — see `src/tasks/noise-generation/AGENTS.md` and REPLICATION.md § E2/E3. `src/models/registry.py::build_noise_gen_model`/`build_noise_gen_loss` remain the model/loss factories (now also used by `conf/model/positional_harmonic_gen*.yaml`'s `_target_`); report/notebook figure scripts still call them directly to reload a trained checkpoint outside the training loop. **Initial harmonic phases** (`HarmonicNoiseGenNew.forward`): random per-harmonic in **train** mode (phase augmentation), **zero** in **eval** mode (deterministic) — so always `model.eval()` for inference/rendering. Overridable via `forward(..., initial_phases=[B,R,H])` for reproducible/pinned synthesis (distinct from the heavier `use_random_phases` spectral-phase randomiser, which is off by default). |
| `MicEQ` | `propagation.py` | The frequency-dependent half of the amplitude-only propagation head: `A_obs[r,k,c](t) = A_src[r,k](t) * (1/dist_{r,c}) * EQ_c(f_k(t))`. `EQ_c` is a learnable smooth magnitude response — `n_knots` (<=16) control points log-spaced in frequency, holding the log gain, linear between knots, held (not extrapolated) outside the span — one curve per **(rig, microphone)**, **shared across rotors** (room + capsule belong to the receiver). Zero init = unity, so an untrained head is the plain 1/r law. Built by `build_noise_gen_model(mic_eq_knots=…, eq_f_min=…, eq_f_max=…)`, which then does NOT build the frequency-flat `log_mic_gain` (a flat EQ *is* that scalar). Applied on every prediction path: on `amp_stats`'s cells at their own `f = k * rps_r(t)` (which is why `amp_stats` also returns `freq`), and on rendered audio / `spectral_stats`' `coherent` as a zero-phase rfft-domain multiply, so a checkpoint renders with the response it was fitted with. The knot curve rides out as the `mic_eq` prediction entry, so its curvature penalty is an ordinary `losses.SmoothnessPenalty` composite term. Rationale + arms: `docs/experiments/amplitude-target-training.md` § C-series. |
| `WindWakeChannel` + `wake_flow_speed` / `QuadDynamics` / `WindTransduction` | `wind_wake_gen.py` | **Additive, incoherent wind-noise channel** — the flow-noise (pseudo-sound) that the coherent `PositionalHarmonicNoiseGen` propagation path structurally cannot make. Same `rps+geometry → [B,M,T]` contract; its output is **summed** at each mic. Physics places the air, only the mic response is learned: **A** `QuadDynamics` (grey-box quad, RPS→`V_rel`, hover-anchored; skipped at `V_rel=0`), **B** `wake_flow_speed` (closed-form bent-wake-column gate → per-mic flow speed `U_m`; aero constants `k,α,β` only), **C** `WindTransduction` (learned `U→½ρU² level ·` low-pass `H(f/f_c)`, OU gust envelope, independent filtered noise per mic ⇒ incoherent by construction). `rps=0 → silence`; differentiable through positions/params. Not yet wired to `train.py` (no `conf/*` yet). **Pre-training de-risk** `scripts/wind_wake_validation.py` (CPU, DREGON single-motor): the geometric gate predicts the per-mic low-band floor at Spearman **0.92** (Pearson 0.97), **beating a 1/r-proximity control (0.74)** → wake-specific, not just closeness; Michael's out-of-wake array → max exposure **0.006 m/s** ≈ 0 (generalization negative test passes). Tests: `tests/test_wind_wake_gen.py`. |
| `SimpleHarmonicNoiseGen` / `PropellerAmplitudePredictor` | `harmonic_gen_new.py` | DEPRECATED random-phase synthesiser + per-prop predictor |
| `CausalConv1d` / `CausalConv1dBlock` / `ResNet` / `RnnSandwich` / … | `nn.py` | Shared building blocks (used by the predictors) |

`good_lstsq` (in `harmonic_transform.py`) picks `gelsd` on CPU / `gels` on CUDA;
`iterative_lstsq_minimize` needs the optional `torchmin` package (imported
lazily).  The `harmonic_gen_new` predictors run natively at 16 kHz; older
44.1 kHz checkpoints from `drone_audition` are not weight-compatible.

## Checkpoint compatibility

SimpleConv* models produced before the front-end refactor (pre-0.13) stored
the Hann window as ``"window"``.  After the refactor it is ``"frontend.window"``.
All model classes define ``load_state_dict`` that automatically remaps.
Existing checkpoints load without user intervention.

To verify a legacy checkpoint loads correctly:

```python
from models.rps_predictor import SimpleConv
model = SimpleConv(n_fft=2048, hop_length=512, num_rotors=4)
ckpt = torch.load("old_checkpoint.pt")
model.load_state_dict(ckpt["state_dict"], strict=True)  # remap is automatic
```
