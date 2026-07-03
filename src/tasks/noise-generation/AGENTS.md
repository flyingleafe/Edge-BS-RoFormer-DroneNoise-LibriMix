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
  metadata, carried as Frame entries `mic_pos (M,3)`/`rotor_pos (R,3)` (dims
  `("mic", None)`/`("rotor", None)`). `tasks.noise_generation.geometry_to_rel_pos`
  turns them into `rel_pos[m, r] = mic[m] - rotor[r]` — it now has **two
  dispatch paths**: the original unbatched-numpy `(M,3),(R,3) -> (M,R,3)`
  (report/notebook figure scripts, `data_processing.generated_noise`) and a
  batched-torch `(B,M,3),(B,R,3) -> (B,M,R,3)` path (differentiable,
  on-device), used by `tasks.codecs.NoiseGenerationCodec` — which now
  correctly builds `rel_pos` from a training batch's `mic_pos`/`rotor_pos`
  entries before calling the model (the fix for the codec/model signature
  mismatch REPLICATION.md § E2/E3 used to document as an open bug). Geometry
  is fixed per array: `data_processing.dregon.get_geometry()` (DREGON 8-mic)
  and `data_processing.michaels.get_geometry()` (Michael's circular 8-mic ring
  on a DJI Matrice 100 — derived from the rig photos in
  `data/recording_with_motor_speed/`; rotor rows ordered RFront, LFront, LBack,
  RBack to match the telemetry). Geometry selection (`{dregon,michaels}`) was a
  `--geometry` flag on the deleted `train_noise_generation.py`; the unified
  framework now resolves geometry per-chunk via the data source instead — see
  "Training integration" below (`conf/model/positional_harmonic_gen{,_conditioned}.yaml`,
  `conf/data/noise_rps_dregon_michaels{,_swapped}.yaml`,
  `conf/experiment/e2_noise_gen_dregon_michaels.yaml`/
  `e3_noise_gen_swapped_smoothness.yaml` — REPLICATION.md § E2/E3).
  Report/notebook scripts (e.g. `notebooks/noise_gen_real_vs_generated.ipynb`)
  still use `src/models/registry.py::build_noise_gen_model` directly to
  reconstruct a generator against a chosen geometry outside the training loop.

  **Mixed-geometry datasets** (DREGON + Michael's chunks together) ARE now
  supported for training: `data_processing.frame_datasets.NoiseGenFrameDataset`
  wraps `data_processing.noise_rps_dataset.NoiseRPSDataset` (whose chunks
  already carry a per-draw `origin`, `"dregon"`/`"michaels"`) and attaches
  that origin's geometry + a `meta.drone` name per sample — so DREGON and
  Michael's chunks stream together in one dataset, each with its own
  geometry. **Caveat**: `NoiseRPSDataset` reduces each draw to one selected
  audio channel without reporting which physical index was picked, so
  `NoiseGenFrameDataset` only supports `channel_policy="first"` (single mic,
  not the full 8-mic array) — see REPLICATION.md § E2/E3 for the deviation
  from the historical online trainer's native multi-observer rendering.
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

The deleted `train_noise_generation.py` took this as `--cond_dim d --drone_name
NAME`, with the codebook bundled alongside the model in a checkpoint file
(`save_bundle`: `{"model", "codebook", "cond_dim", "drone_names"}`) — external
to "the model" in `models.registry`'s sense, with its own optimizer param
group. The unified `training.loop.run_training` doesn't support that (one
`optimizer = get_optimizer(model, ...)` over `model.parameters()`, one
checkpoint = `model.state_dict()`), so `models.registry.build_noise_gen_model(...,
cond_dim=d, drone_names=[...])` now instead returns a composite
`_CodebookConditionedNoiseGen(generator, codebook)` — the codebook is a
genuine submodule, so its params are trained and checkpointed through the
normal single-model path. `tasks.codecs.NoiseGenerationCodec(conditioned=True)`
resolves each sample's `drone_names` from `meta.drone` (see "Mixed-geometry
datasets" above) and calls `model(rps, rel_pos, drone_names)`; the model
resolves `z` from its own codebook. **Multi-drone training in one run** now
works out of the box: `NoiseGenFrameDataset` already attaches per-chunk
`meta.drone` + geometry from whichever source (`dregon`/`michaels`) each
draw came from.

## Training integration

- **Script**: the dedicated `train_noise_generation.py` is deleted; training
  routes through the unified `train.py` + `conf` —
  `conf/model/positional_harmonic_gen{,_conditioned}.yaml`,
  `conf/data/noise_rps_dregon_michaels{,_swapped}.yaml`,
  `conf/loss/multiscale_stft{,_smoothness}.yaml`,
  `conf/metrics/noise_gen_spectral.yaml`,
  `conf/experiment/e2_noise_gen_dregon_michaels.yaml`/
  `e3_noise_gen_swapped_smoothness.yaml` (REPLICATION.md § E2/E3; E1 remains
  an intentional dead end, its model class isn't registered).
- **Registry**: register the model class in
  `src/models/registry.py::NOISE_GEN_MODEL_REGISTRY`; the factory is
  `build_noise_gen_model(name, sample_rate, n_harmonics, use_diff_noise,
  cond_dim, drone_names)` (verbatim port of the former
  `train_noise_generation.py::get_model`, plus the `drone_names` ->
  `_CodebookConditionedNoiseGen` wrapping described above).
- **Dataset**: `data_processing.frame_datasets.NoiseGenFrameDataset` wraps
  `data_processing.noise_rps_dataset.NoiseRPSDataset`/
  `build_noise_rps_datasets` (DREGON `in_flight_noise` + Michael's, reused
  verbatim — not the on-disk DREGON-LM `sample_*` chunk format
  `DREGONNoiseGenDataset` used historically). Emits Frames with
  `rps (rotor,time)` at audio rate, `audio (mic,time)` (the clean target —
  single mic only, see the `channel_policy="first"` caveat above),
  `mic_pos`/`rotor_pos`, `meta.drone`.
- **Loss**: multi-scale STFT (`losses.MultiScaleSTFTLoss`, `pred_key=
  target_key="audio"`), the mic axis folded into the batch by
  `losses.spectral._flatten_to_2d`. Magnitude loss is blind to a *common*
  delay but sees inter-rotor delay differences (the geometric signal). E3's
  Stage-2 smoothness regularisers: `losses.SmoothnessPenalty(entry=
  "harm_amps"|"noise_amps", series_dims=..., series_time=None)` acting on
  the extra pred entries `tasks.codecs.NoiseGenerationCodec(return_dict=True)`
  exposes.

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
3. [ ] Register in `src/models/registry.py::NOISE_GEN_MODEL_REGISTRY`.
4. [ ] Smoke test: shapes + one forward/backward — `tests/tasks/test_noise_generation.py` covers the codec/task-composition layer generically (geometry_to_rel_pos, NoiseGenerationCodec, build_noise_gen_model's DroneCodebook wrapping); `tests/models/test_positional_harmonic_gen.py` is the model-specific pattern to mirror for a new model.
5. [ ] One-epoch run to verify gradient flow and loss decrease.
