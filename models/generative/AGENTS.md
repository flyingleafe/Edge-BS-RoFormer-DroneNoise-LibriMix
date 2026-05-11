# models/generative/ — Drone-noise generative models from RPS telemetry

Ported (and simplified) from the legacy `drone-audition` project. Synthesise
real-sounding drone propeller noise *from rotor-speed telemetry alone* —
useful as

- a learned physical model of the ego-noise generation process,
- a guidance signal / regulariser for the enhancement networks, and
- training data augmentation when paired with clean speech.

All modules drop the legacy `env.settings` dependency. **Sample rate is an
explicit constructor argument** on every module (default 16 kHz to match
DREGON-LM, DN-LM, etc.).

## Modules

| File | What's in it |
|------|--------------|
| `dsp.py`               | `oscillator_bank`, `harmonic_oscillator_bank`, `freqs_to_phasors`, `frequency_filter`, `fft_convolve` |
| `math_utils.py`        | hz↔midi, `exp_sigmoid`, `overlap_and_add`, `signal_frame`, `safe_log`, etc. |
| `harmonic_noise_gen.py`| `PropellerNoiseGen`, `DroneNoiseGen` — sinusoidal harmonic synthesis from RPS |
| `filtered_noise.py`    | `FilteredNoiseSynth`, `RPSFilterNet`, `DroneNoisePlusFilterGen` |
| `harmonic_transform.py`| `VP_transform`, `inverse_VP_transform`, `HarmonicTransformModule` (variable-phasor analysis/synthesis) |
| `losses.py`            | `MultiScaleSTFT` (DDSP-style multi-scale spectral loss, lin + log mag) |

## Forward API summary

```python
from models.generative import DroneNoisePlusFilterGen, MultiScaleSTFT

gen = DroneNoisePlusFilterGen(
    n_motors=4, n_harmonics=50, sample_rate=16000,
    filter_n_freqs=65, filter_n_frames=64,
)
# RPS must be in Hz (rev/s) at the AUDIO sample rate, shape [B, n_motors, T]
out = gen(rps_audio_rate)        # phase_shifts default to zero
out["audio"]      # [B, T]  full prediction
out["harmonic"]   # [B, T]  harmonic-only branch
out["noise"]      # [B, T]  filtered-noise residual
out["filter_mags"]# [B, F_frames, K_freqs]
```

## Architecture relationships

```
DroneNoisePlusFilterGen
├── DroneNoiseGen
│   └── PropellerNoiseGen
│       └── harmonic_oscillator_bank   (dsp.py)
├── RPSFilterNet               (1-D Conv stack: rps → per-frame mags)
└── FilteredNoiseSynth
    └── frequency_filter       (dsp.py: zero-phase FIR via overlap-add)

HarmonicTransformModule
├── VP_transform               (audio → harmonic projections)
├── <user net>                 (operates on projections)
└── inverse_VP_transform       (projections → audio)
```

## Training

Use `train_noise_gen.py` at the repo root + `configs/noise_gen.yaml`. The
trainer pairs the generator with `MultiScaleSTFT` against real recorded
drone noise from DREGON + Michael's set (see
`data_processing/noise_rps_dataset.py`).

## Differences vs. legacy `drone_audition`

- `env.settings.SAMPLE_RATE` removed everywhere — pass `sample_rate=` explicitly.
- `signal_frame(pad_end=True)` no longer adds a spurious extra frame when the
  signal length is exactly aligned (legacy bug that broke `fft_convolve` on
  power-of-2 chunk sizes).
- Loss is implemented on `torch.stft` rather than Asteroid filterbanks, so the
  module has no extra Asteroid dependency.
- `RPSFilterNet` is new: replaces the legacy DDSP `ResnetSinusoidalEncoder`'s
  noise-magnitude head with a much smaller direct-from-RPS conv net, suitable
  for the generator-only task. The DDSP `InverseSynthesis` path is not ported
  (it was for audio→params→audio, not the goal here).
