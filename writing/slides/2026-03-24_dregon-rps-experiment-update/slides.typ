#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [DREGON Experiment Update],
  subtitle: [Rotor-Conditioned Speech Enhancement],
  author: [Dmitrii Mukhutdinov],
  date: [2026-03-24],
)

DCUNet and DCCRN partial replication of:
_Enhancing drone audition with rotor-conditioned deep models_

= 1) Completed Experiment Set (Batch 1)

*Objective*
- Compare baseline vs rotor-conditioned variants on DREGON-LibriMix
- Models: `DCUNet` and `DCCRN`
- Inputs: noisy waveform (always), optional RPS (`use_rps`)
- Configs:
  - `configs/7b_DCUNet_baseline_DREGON.yaml`
  - `configs/7a_DCUNet_RPS_DREGON.yaml`
  - `configs/10a_DCCRN_baseline_DREGON.yaml`
  - `configs/10b_DCCRN_RPS_DREGON.yaml`

= Batch 1 Setup Notes

- Same dataset split and training schedule across baseline/RPS pairs
- RPS-enabled runs use `num_rotors: 4`, `rps_length: 8500`
- Target: vocals enhancement at ultra-low SNR
- Purpose: partial replication of rotor-conditioned gains from DREGON paper

= DCUNet: without RPS vs with RPS

#grid(
  columns: 2,
  gutter: 1em,
  [
    *Without RPS*

    ```mermaid
    flowchart TB
      A1["Noisy waveform"] --> A2["STFT"]
      A2 --> A3["Complex U-Net encoder x5"]
      A3 --> A4["Bottleneck"]
      A4 --> A5["Complex decoder with skip connections"]
      A5 --> A6["Mask x input STFT"]
      A6 --> A7["ISTFT to enhanced waveform"]
    ```

    - No rotor path — `dcunet_num_encoder_layers: 5`
  ],
  [
    *With RPS (bottleneck fusion)*

    ```mermaid
    flowchart TB
      B1["Noisy waveform"] --> B2["STFT"]
      B2 --> B3["Complex U-Net encoder x5"]
      B3 --> B4["Bottleneck features"]
      R1["RPS: 4-rotor timeseries"] --> R2["RotorEncoder: Conv1d then Conv1d"]
      R2 --> R3["Mean over time then linear projection"]
      R3 --> B4
      B4 --> B5["Complex decoder with skip connections"]
      B5 --> B6["Mask x input STFT"]
      B6 --> B7["ISTFT to enhanced waveform"]
    ```

    - `dcunet_rps_fusion: bottleneck`
  ],
)

= DCCRN: without RPS vs with RPS

#grid(
  columns: 2,
  gutter: 1em,
  [
    *Without RPS*

    ```mermaid
    flowchart TB
      C1["Noisy waveform"] --> C2["STFT"]
      C2 --> C3["Complex encoder x6"]
      C3 --> C4["Flatten bottleneck"]
      C4 --> C5["BiGRU x2, hidden 256"]
      C5 --> C6["Linear projection to bottleneck"]
      C6 --> C7["Complex decoder with skip"]
      C7 --> C8["Mask x input STFT"]
      C8 --> C9["ISTFT to enhanced waveform"]
    ```

    - No rotor path
  ],
  [
    *With RPS (pre-GRU concat)*

    ```mermaid
    flowchart TB
      D1["Noisy waveform"] --> D2["STFT"]
      D2 --> D3["Complex encoder x6"]
      D3 --> D4["Flatten bottleneck"]
      R4["RPS: 4-rotor timeseries"] --> R5["RotorEncoder: Conv1d then Conv1d"]
      R5 --> R6["Time-align to bottleneck T"]
      R6 --> D4
      D4 --> D5["BiGRU x2, hidden 256"]
      D5 --> D6["Linear projection"]
      D6 --> D7["Complex decoder with skip"]
      D7 --> D8["Mask x input STFT"]
      D8 --> D9["ISTFT to enhanced waveform"]
    ```

    - RPS concatenated before bottleneck GRU
  ],
)

= 2) Running Experiment Set: + PredRPS Auxiliary Loss

*New idea*
- Keep RPS-conditioned enhancement path
- Add auxiliary task: predict rotor RPS from encoder features (`predict_rps: true`)
- Joint objective: enhancement loss + weighted RPS prediction loss (`rps_aux_weight`)

*Configs currently running*
- `configs/7c_DCUNet_RPS_PredRPS_DREGON.yaml`
- `configs/10d_DCCRN_RPS_PredRPS_DREGON.yaml`

*Loss weighting in current configs*
- DCUNet: `rps_aux_weight: 0.5`
- DCCRN: `rps_aux_weight: 2.0`

= PredRPS in DCUNet (Running)

```mermaid
flowchart TB
  E1["Encoder levels x5"] --> E2["RPSPredictionHead"]
  E2 --> E3["Predicted RPS per STFT frame"]
  E1 --> E4["RPS-conditioned enhancement path"]
  E4 --> E5["Enhanced waveform"]
  GT1["Ground-truth RPS"] --> E6["Auxiliary RPS loss"]
  E3 --> E6
  E5 --> E7["Enhancement loss"]
```

- Prediction head is FPN-like over encoder scales
- Config: `configs/7c_DCUNet_RPS_PredRPS_DREGON.yaml`
- Total training objective combines enhancement + auxiliary losses

= PredRPS in DCCRN (Running)

```mermaid
flowchart TB
  F1["Encoder levels x6"] --> F2["RPSPredictionHead"]
  F2 --> F3["Predicted RPS"]
  F1 --> F4["Pre-GRU RPS concat enhancement path"]
  F4 --> F5["Enhanced waveform"]
  GT2["Ground-truth RPS"] --> F6["Auxiliary RPS loss"]
  F3 --> F6
  F5 --> F7["Enhancement loss"]
```

- Config: `configs/10d_DCCRN_RPS_PredRPS_DREGON.yaml`
- Goal: encourage rotor-aware latent representations

= 3) Current Results (Batch 1)

*Per-SNR comparison*

#figure(
  image("assets/per_snr_comparison.png", height: 62%),
)

Source:
- `assets/per_snr_comparison.png`
- `assets/dregon_per_snr_comparison.csv`

= 3) Current Results (Batch 1): Overall Metrics

*Overall metrics from CSV (600 samples)*

#table(
  columns: 4,
  align: (left, right, right, right),
  [Model], [SI-SDR], [eSTOI], [PESQ],
  [DCUNet baseline], [-9.3224], [0.2401], [1.1462],
  [DCUNet + RPS], [-7.5920], [0.2374], [1.1379],
  [DCUNet + RPS + PredRPS\*], [-25.3389], [0.1323], [1.3298],
  [DCCRN baseline], [-5.6759], [0.2666], [1.1246],
  [DCCRN + RPS], [-7.9758], [0.2419], [1.1369],
  [DCCRN + RPS + PredRPS\*], [-8.4733], [0.2255], [1.1145],
)

*Notes*
- DCUNet: RPS improves SI-SDR (+1.73 dB), but PredRPS\* currently collapses SI-SDR
- DCCRN: RPS and PredRPS\* both underperform baseline on SI-SDR in this batch
- `*` PredRPS runs likely need further investigation (see next slide)

= Next Steps

- This week: finish investigating `PredRPS` runs:
  - Test exact encoder parts of DCUNet and DCCRN on the task of only predicting motor speeds. Compare with previous simple RPS predictor model. Achieve good motor speed prediction performance.
  - Investigate possible training setup discrepancies / errors in loss scaling
  - Re-run the training; also train DCCRN+RPS for more epochs to achieve parity in the number of steps between training runs.

- Next week:
  - Schedule more experiments (include diffusion models?)
  - Fix the list of models and the dataset compositions to fully cover throughout April
  - Establish more frequent sync iterations.
