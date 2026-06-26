# Task: Noise Generation

Generate the drone noise observed at each microphone from rotor speeds (RPS) and
the array geometry. This is the **inverse of RPS prediction**: RPS prediction
maps *audio → rotor speeds*; noise generation maps *rotor speeds + geometry →
multichannel noise*.

## Model interface

```python
class NoiseGenerator(nn.Module):
    def forward(self, rps: Tensor, rel_pos: Tensor, z: Tensor | None = None) -> Tensor:
        """
        rps:     (B, R, T)      per-rotor speed (Hz) at audio rate
        rel_pos: (B, M, R, 3)   vector rotor_r -> mic_m (metres)
        z:       (B, d)         external per-drone conditioning code (or None)
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

### Per-drone conditioning — external codebook (different drones = different source)

DREGON and Michael's are different drones (different harmonics/broadband/dynamics),
so the source is conditioned on a learned per-drone code. The code is **external**
to the model, by the same logic that keeps geometry external:

- The generator takes the code `z (B, d)` as an **input**, not a `drone_id`:
  `PositionalHarmonicNoiseGen(cond_dim=d)` builds the emitter with
  `JointAmplitudePredictor(film=True)` so `z → (γ,β)` modulate the CNN features
  (per-drone spectral envelope *and* RPS→sound dynamics). `cond_dim == 0`
  disables conditioning. FiLM starts near-identity (γ≈1, β≈0) with a small
  non-zero weight so the code gets gradient from step 1.
- The `name → z` table is a separate `tasks.noise_generation.DroneCodebook(d, names)`:
  a **name-keyed** `nn.ParameterDict`. Crucially the model owns `d`
  (architectural — it sizes the FiLM generator) but **not** `K` (the number of
  drones, a data property). Adding a drone never resizes model weights; codes
  load by name with `strict=False`, so no index drift between datasets.
- **Few-shot adaptation to an unseen drone** = freeze the generator
  (`--freeze_emitter`), warm-start from a trained bundle (`--init_checkpoint`),
  and optimise just the new drone's `d`-vector. This is the payoff of the
  external/`z`-input design and is not clean with an in-model table.

`train_noise_generation.py --cond_dim d --drone_name NAME`; the codebook is
bundled with the model in the checkpoint (`save_bundle`: `{"model", "codebook",
"cond_dim", "drone_names"}`). **Multi-drone training in one run** needs per-chunk
`drone_name` + geometry (same per-chunk metadata follow-on as mixed datasets);
the model + codebook already support arbitrary `K`, only the dataset attaches one
name today.

## Training integration

- **Script**: `train_noise_generation.py`.
- **Registry**: add the model class to `MODEL_REGISTRY` there; the factory is
  `get_model(name, sample_rate, n_harmonics, use_diff_noise)`.
- **Dataset**: `DREGONNoiseGenDataset` reuses the **same on-disk format as RPS
  prediction** (DREGON-LM `sample_*` chunks) but:
  - target is the clean `noise.wav` (configurable via `--target_file`), **not**
    `mixture.wav` — no speech mixing;
  - positions are attached from the recording geometry (`get_geometry`).
  Yields `(rps_audio_rate (R,T), rel_pos (M,R,3), target (M,T), drone_name str)`.
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
  `geometry_to_rel_pos`, `load_input_set` TimeFrame loader, `DroneCodebook` —
  the external name-keyed per-drone code table).

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
