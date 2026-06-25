# Task: Noise Generation

Generate the drone noise observed at each microphone from rotor speeds (RPS) and
the array geometry. This is the **inverse of RPS prediction**: RPS prediction
maps *audio → rotor speeds*; noise generation maps *rotor speeds + geometry →
multichannel noise*.

## Model interface

```python
class NoiseGenerator(nn.Module):
    def forward(self, rps: Tensor, rel_pos: Tensor) -> Tensor:
        """
        rps:     (B, R, T)      per-rotor speed (Hz) at audio rate
        rel_pos: (B, M, R, 3)   vector rotor_r -> mic_m (metres)
        returns: (B, M, T)      noise at each of the M microphones
        """
```

- **Input rate**: 16 kHz; `rps` is upsampled to the audio grid (not the STFT grid).
- **Geometry as input**: microphone/rotor positions are non-temporal array
  metadata, carried in `TimeFrame.global_data` (`mic_positions (M,3)`,
  `rotor_positions (R,3)`). `tasks.noise_generation.geometry_to_rel_pos` turns
  them into `rel_pos[m, r] = mic[m] - rotor[r]`. Geometry is fixed per array:
  `data_processing.dregon.get_geometry()` (DREGON 8-mic) and
  `data_processing.michaels.get_geometry()` (Michael's circular 8-mic ring on a
  DJI Matrice 100 — derived from the rig photos in
  `data/recording_with_motor_speed/`; rotor rows ordered RFront, LFront, LBack,
  RBack to match the telemetry). Select with `train_noise_generation.py
  --geometry {dregon,michaels}`.

  **Mixed-geometry datasets** (DREGON + Michael's chunks together) are not yet
  supported: the dataset attaches one geometry to all chunks, so train per
  source for now. Supporting mixtures means persisting positions **per chunk**
  (in `global_data`) at dataset-creation time.
- **Multichannel**: all M mics are rendered **jointly** (native multi-observer),
  *not* flattened into the batch like RPS prediction. The reference model sums
  rotors in the rfft domain → M mics cost R forward + M inverse transforms.
- **Output**: clean drone noise `(B, M, T)` (no speech).

## Training integration

- **Script**: `train_noise_generation.py`.
- **Registry**: add the model class to `MODEL_REGISTRY` there; the factory is
  `get_model(name, sample_rate, n_harmonics, use_diff_noise)`.
- **Dataset**: `DREGONNoiseGenDataset` reuses the **same on-disk format as RPS
  prediction** (DREGON-LM `sample_*` chunks) but:
  - target is the clean `noise.wav` (configurable via `--target_file`), **not**
    `mixture.wav` — no speech mixing;
  - positions are attached from the recording geometry (`get_geometry`).
  Yields `(rps_audio_rate (R,T), rel_pos (M,R,3), target (M,T))`.
  Requires chunks that keep a clean-noise target (a non `--real_valid`
  multichannel set; `--real_valid` valid chunks have only `mixture.wav`).
- **Loss**: multi-scale STFT (`models.generative.MultiScaleSTFT`), the mic axis
  folded into the batch. Magnitude loss is blind to a *common* delay but sees
  inter-rotor delay differences (the geometric signal).

## Code placement

- **Model**: `src/models/generative/positional_harmonic_gen.py`
  (`PositionalHarmonicNoiseGen` = single-rotor `HarmonicNoiseGenNew` emitter +
  `propagate`).
- **Task module**: `src/tasks/noise_generation.py` (`NoiseGenerator` protocol,
  `geometry_to_rel_pos`, `load_input_set` TimeFrame loader).

## Existing implementations

| Model | Key | Approach |
|-------|-----|----------|
| PositionalHarmonicNoiseGen | `positional_harmonic_gen` | Per-rotor harmonic + filtered-noise emitter, propagated (1/r + fractional delay) to every mic and summed |

## Checklist for a new noise-generation model

1. [ ] Implement core model in `src/models/generative/`.
2. [ ] Satisfy `forward(rps (B,R,T), rel_pos (B,M,R,3)) -> (B,M,T)`.
3. [ ] Register in `train_noise_generation.py::MODEL_REGISTRY`.
4. [ ] Smoke test: shapes + one forward/backward (`tests/train/test_noise_generation.py`).
5. [ ] One-epoch run to verify gradient flow and loss decrease.
