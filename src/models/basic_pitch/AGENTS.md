# src/models/basic_pitch/ — Basic Pitch (PyTorch port)

Faithful PyTorch port of Spotify's **Basic Pitch** note-transcription model
(Bittner et al., ICASSP 2022) — https://github.com/spotify/basic-pitch.
Verified **numerically identical** to the original (released ICASSP-2022
weights), max abs diff ~3e-6 vs the official ONNX export.

## Why it exists

Basic Pitch is a lightweight, instrument-agnostic multi-pitch / note model.
Ported here as a multi-F0 baseline (cf. `src/models/multif0/`) and a source of
pretrained pitch-contour features.

## Files (mirror the upstream layout)

| File | Upstream | Contents |
|------|----------|----------|
| `model.py` | `basic_pitch/models.py` | `BasicPitch` nn.Module, `Conv2dSame` (TF 'SAME' padding), TF-weight loader |
| `cqt.py` | `layers/nnaudio.py` (+ `get_cqt`) | `CQTFrontEnd`: nnAudio `CQT2010v2` + `NormalizedLog` |
| `nn.py` | `basic_pitch/nn.py` | `HarmonicStacking`, `flatten_freq_ch` (shape-only) |
| `signal.py` | `layers/signal.py` | `NormalizedLog` (per-sample dB min-max norm) |
| `weights/icassp_2022.pt` | converted from the TF checkpoint | committed torch state_dict |

The upstream `layers/nnaudio.py` is a TF re-port of the PyTorch **nnAudio**
library — already a project dependency — so we call `nnAudio.features.CQT2010v2`
directly instead of re-porting 250 lines of TF. Its defaults match basic-pitch
exactly (`norm=True, basis_norm=1, hann, reflect, earlydownsample`).

## Interface

```python
from models.basic_pitch import BasicPitch
model = BasicPitch.from_pretrained()           # no TF/onnx needed
out = model(audio)                              # audio: (B, 43844) @ 22050 Hz
# out: {"contour": (B,172,264), "note": (B,172,88), "onset": (B,172,88)}  sigmoids in [0,1]
```

Fixed input length is `AUDIO_N_SAMPLES = 22050*2 - 256 = 43844` samples (the
upstream window). `from_pretrained_tf(weights_npz)` rebuilds from raw TF
checkpoint tensors (keyed `layer_with_weights-N/{kernel,bias,gamma,...}`),
matched to layers by shape.

## Key fidelity details (don't "fix" these)

- **Layout.** TF is channels-last `(B,time,freq,ch)`; we use NCHW
  `(B,ch,time,freq)` with H=time, W=freq. Conv weights ported as
  `(kh,kw,in,out) → (out,in,kh,kw)`.
- **BatchNorm eps = 1e-3** (Keras default), not PyTorch's 1e-5.
- **`Conv2dSame`** replicates TF 'SAME' padding incl. the strided `(1,3)`
  note/onset convs (PyTorch `padding='same'` rejects stride>1).
- **Contour conv-block** before `contour_out` is intentionally a single
  `(3,39)` conv — upstream commented out the first contour conv (a documented
  quirk of the shipped checkpoint).

## Tests

`tests/test_basic_pitch.py` (+ fixture `tests/basic_pitch_ref.npz`) asserts the
port matches the original ONNX outputs to `<2e-3`. No TF/onnx needed at test
time. `pytest tests/test_basic_pitch.py`.
