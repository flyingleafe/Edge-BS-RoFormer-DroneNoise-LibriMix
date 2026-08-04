# Task: Speech Enhancement

Recover clean speech from a mixture of speech and harmonic rotor noise at
ultra-low SNR (0 to −30 dB). Input is the noisy waveform, output is the
enhanced waveform on the **same** grid. Two options widen the contract: RPS
conditioning on the input side (`use_rps`) and an auxiliary RPS-prediction head
on the output side (`predict_rps`).

## Frame contract (`tasks.task.speech_enhancement`)

The task is a `Task(name, input_spec, output_spec)` over **batched** Frames
(leading `"batch"` dim). Built by the factory from `conf/model/<name>.yaml`'s
`task_params`:

```python
speech_enhancement(
    n_channels=None,        # None -> mono dims (batch, time); else (batch, mic, time)
    use_rps=False,          # add an "rps" input entry
    predict_rps=False,      # add an "rps_pred" output entry
    sr=(16000, 1),          # audio rate as an exact (num, den) pair
    rps_rate=None,          # RPS frame rate, normally (sr, hop) e.g. (16000, 512)
)
```

| Side | Entry | Dims | Time / rate | When |
|------|-------|------|-------------|------|
| in | `mixture` | `(batch, time)` or `(batch, mic, time)` | grid @ `sr` | always |
| in | `rps` | `(batch, rotor, time)` | grid @ `rps_rate` | `use_rps=True` |
| out | `enhanced` | same dims as `mixture` | grid @ `sr` | always |
| out | `rps_pred` | `(batch, rotor, time)` | grid @ `rps_rate` | `predict_rps=True` |

The training target lives in the batch Frame as `target` (the clean speech,
same shape as `mixture`); it is a *loss/metric* entry, not part of the task's
output spec — see `conf/loss/si_sdr_mrstft.yaml` (`target_key: target`).

## Model interface

Models are plain `nn.Module`s; the codec adapts them, so no model has to know
about Frames:

```python
class Enhancer(nn.Module):
    def forward(self, x, rps=None):
        """
        x:   (B, 1, T) mono — the codec adds the channel axis when
             n_channels is None — or (B, C, T) multichannel
        rps: (B, 4, T_frames) when use_rps, else absent
        returns: enhanced (B, 1, T)  — or a tuple (enhanced, rps_pred)
                 when predict_rps
        """
```

`tasks.codecs.SpeechEnhancementCodec` is the only seam:

- `to_inputs(batch)` — pulls `"mixture"` (and `"rps"` when `use_rps`).
- `call_model(model, inputs)` — **unsqueezes mono to `(B, 1, T)`** (the
  DCUNet/DCCRN convention `train.py` always used), then calls
  `model(x)` or `model(x, rps=...)`.
- `to_frame(outputs, batch)` — splits a bare tensor or an
  `(enhanced, rps_pred)` tuple via `_split_model_output`, and wraps the
  primary output as `"enhanced"` (plus `"rps_pred"` when `predict_rps` **and**
  the model actually returned an auxiliary tensor).

`zoo.FrameModel` bundles `(model, codec, task)` into one `td.Frame → td.Frame`
callable — that is what `eval.py`, `scripts/se_eval.py` and the notebooks use.

## Training integration

### Config
`conf/model/<name>.yaml` declares the task and its params; the model itself is
either a native `_target_` factory (`models.tfgridnet.build_tfgridnet`,
`models.mpsenet.build_mpsenet`, `models.htdemucs_ft.build_htdemucs_ft`) or a
legacy ZFTurbo-style tree inlined under `params.config` and built by
`models.registry.build_legacy_inline`:

```yaml
task: speech_enhancement
task_params:
  n_channels: null
  use_rps: true
  predict_rps: true
  sr: [16000, 1]
  rps_rate: [16000, 512]
_target_: models.registry.build_legacy_inline
params: {model_type: dcunet, config: {...}}
```

`training.config.build_task_and_codec(cfg.model)` builds the `Task` and the
`Codec` from that one `task` + `task_params` pair — they can never drift apart.

### Data
- Train: `data_processing.frame_datasets.OnlineMixFrameDataset.from_yaml`
  over an online-mix policy with `task: speech_enhancement`
  (`conf/online_mix/se_*.yaml`) — the policy emits `mixture` + `target`
  (+ `rps` when the noise source carries telemetry). See
  `src/data_processing/AGENTS.md` § "Speech-enhancement target mode".
- Valid: `SEValidFrameDataset` over a frozen published split
  (`SE-valid-drone`, `SE-valid-harmonic`) — fixed, so early stopping compares
  like with like across models.
- Legacy offline: `DNLMFrameDataset` / `DregonLMFrameDataset`.

### Loss and metrics
- Loss: `conf/loss/si_sdr_mrstft.yaml` — negative SI-SDR (metric-aligned)
  plus a multi-resolution STFT magnitude term. **Do not use plain masked MSE**
  at ultra-low SNR: it rewards attenuation toward silence
  (`docs/experiments/f1-se-blind-baselines.md`).
- Metrics: `conf/metrics/separation_basic.yaml` / `separation_full.yaml`, or
  `separation_plus_rps.yaml` for the `predict_rps` families. Monitor
  `si_sdr` with `optim.monitor_mode: max`.

## Code placement

- Core model: `src/models/<name>.py` or `src/models/<subdir>/`.
- Registration: a native `_target_` factory is enough — nothing to add to a
  registry. Legacy model-type keys live in
  `models.registry.LEGACY_MODEL_BUILDERS`; `models.registry.model_types()`
  lists everything either way.
- RPS conditioning (`use_rps`): the fusion strategies (`bottleneck`, `gru`,
  `hierarchical`) are documented in `src/models/AGENTS.md`.

## Existing implementations

| Model | Config | Notes |
|-------|--------|-------|
| DCUNet / DCCRN (+RPS, +PredRPS) | `conf/model/b1_*`, `b2_*` | The RPS-conditioning families (Paper 2) |
| Edge-BS-RoFormer | `conf/model/a1_edge_bs_rof_*` | Paper 1 baseline |
| TF-GridNet | `conf/model/f1_tfgridnet.yaml` | Native `_target_` port |
| MP-SENet | `conf/model/f1_mpsenet.yaml` | Trains **fp32** — bf16 NaN-poisons |
| SGMSE+ | `conf/model/f1_sgmse.yaml` | Score-based diffusion |
| HTDemucs | `conf/model/f1_htdemucs.yaml` | Fine-tuned from the Meta checkpoint |

## Evaluation

`python eval.py` (the only evaluation entry point) or `scripts/se_eval.py`
for the generic per-checkpoint metric table. Metrics: SI-SDR, SDR, PESQ,
(e)STOI — one implementation each, in `src/metrics`.

---

## Checklist for a new SE model

1. [ ] Read the paper, find official source, note the native sample rate.
2. [ ] Implement in `src/models/`; keep `forward(x, rps=None) -> Tensor` (or
       the `(enhanced, rps_pred)` tuple) — nothing Frame-aware inside.
3. [ ] Write `conf/model/<name>.yaml` with `task: speech_enhancement` +
       `task_params`.
4. [ ] Smoke test the conf-yaml build path through
       `training.config.instantiate_model` (mirror `tests/models/test_tfgridnet.py`).
5. [ ] Check the codec round trip: `zoo.FrameModel(model, codec, task)(frame)`
       returns an `enhanced` entry with the mixture's shape.
6. [ ] One-epoch run to verify gradient flow, then the full experiment.
