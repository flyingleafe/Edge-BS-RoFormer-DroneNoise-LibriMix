#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [RPS predictions & RPS-informed DCUNet],
  subtitle: [DREGON noises · Baseline vs conditioned],
  author: [Dmitrii Mukhutdinov],
  date: [2026-02-24],
)

= Outline

+ *RPS predictions from noisy mixture (DREGON noises)*
+ *RPS-informed DCUNet vs baseline DCUNet*

= Section 1: RPS predictions from noisy mixture (DREGON noises)

= Can we reliably extract motor speeds from noisy mixtures?

*Goal:* Predict rotor speeds (RPS) from *noisy mixtures* of speech + drone noise (DREGON), so that downstream enhancement can be conditioned on inferred motor state.

- *Input:* Raw mono waveform — speech + DREGON drone noise at 16 kHz.
- *Output:* Four motor RPS values *per STFT frame* (aligned with enhancement models).
- *Why it matters:* If we can estimate RPS from the mixture, we can use it to condition a separator (e.g. DCUNet) even when ground-truth RPS is unavailable at test time.

= RPS predictor architecture

#[
#set text(size: 9pt)
```mermaid
flowchart LR
  subgraph In["Input"]
    A["Raw audio<br/>16 kHz mono"]
  end
  subgraph Front["Front-end"]
    B["STFT"]
    C["Log-mag<br/>spectrogram"]
    B --> C
  end
  subgraph Enc["Encoder (DCUNet-like)"]
    D["5 conv blocks<br/>freq strides only"]
    E["BatchNorm +<br/>LeakyReLU"]
    D --> E
  end
  subgraph Head["Head"]
    F["Pool over freq"]
    G["1D conv<br/>90→64→4"]
    F --> G
  end
  subgraph Out["Output"]
    H["(B, 4, T)<br/>RPS/frame"]
  end
  A --> B
  C --> D
  E --> F
  G --> H
```
]

Real-valued encoder on log-mag; time axis preserved for per-frame RPS. Head: AdaptiveAvgPool over frequency → 1D conv → 4 rotors.

= Training set and setup

*Dataset (DREGON-LM):*
- *Train / valid:* `datasets/DREGON-LM/{train,valid}` — each sample = `mixture.wav` + `rps.npy` (4 motors, original time grid).
- RPS resampled to STFT frame grid via linear interpolation (hop 512, n_fft 2048).

*Training:*
- *Loss:* MSE between predicted and target RPS (per frame).
- *Optimizer:* AdamW (lr 1e-3, weight decay 1e-4); ReduceLROnPlateau (factor 0.5, patience 5).
- *Regularization:* Grad clip 5.0; mixed precision (AMP).
- *Baseline:* Naive predictor = constant train-set mean RPS per frame (MSE ≈ 365, R² ≈ -2.6 on val).

= Training dynamics

#figure(image("assets/training_curves.png", height: 40%), caption: [RPS predictor training curves])

- *MSE (log):* Train and val MSE drop well below naive baseline (365.1); best val MSE *5.51* at epoch 58.
- *R²:* Val R² recovers from early instability and reaches *0.946* at best — model explains ~95% of variance in RPS.
- Early epochs show val fluctuation; convergence is stable by mid–late training. Early stopping (patience 10) would retain the best checkpoint.

= RPS prediction samples

One slide per sample: spectrogram (log-scale frequency), target motor speeds, predicted RPS.

= RPS sample: dregon_lm_sample_00000

#figure(image("assets/dregon_lm_sample_00000.png", height: 78%), caption: [dregon_lm_sample_00000])

= RPS sample: dregon_lm_sample_00149

#figure(image("assets/dregon_lm_sample_00149.png", height: 78%), caption: [dregon_lm_sample_00149])

= RPS sample: dregon_lm_sample_00299

#figure(image("assets/dregon_lm_sample_00299.png", height: 78%), caption: [dregon_lm_sample_00299])

= RPS sample: dregon_lm_sample_00449

#figure(image("assets/dregon_lm_sample_00449.png", height: 78%), caption: [dregon_lm_sample_00449])

= RPS sample: dregon_lm_sample_00599

#figure(image("assets/dregon_lm_sample_00599.png", height: 78%), caption: [dregon_lm_sample_00599])

= RPS sample: ext_recording_1_124_chunk00

#figure(image("assets/ext_recording_1_124_chunk00.png", height: 78%), caption: [ext_recording_1_124_chunk00])

= RPS sample: ext_recording_1_124_chunk01

#figure(image("assets/ext_recording_1_124_chunk01.png", height: 78%), caption: [ext_recording_1_124_chunk01])

= RPS sample: ext_recording_1_124_chunk02

#figure(image("assets/ext_recording_1_124_chunk02.png", height: 78%), caption: [ext_recording_1_124_chunk02])

= RPS sample: ext_recording_1_124_chunk03

#figure(image("assets/ext_recording_1_124_chunk03.png", height: 78%), caption: [ext_recording_1_124_chunk03])

= RPS sample: ext_recording_1_124_chunk04

#figure(image("assets/ext_recording_1_124_chunk04.png", height: 78%), caption: [ext_recording_1_124_chunk04])

= RPS sample: ext_recording_2_125_chunk00

#figure(image("assets/ext_recording_2_125_chunk00.png", height: 78%), caption: [ext_recording_2_125_chunk00])

= RPS sample: ext_recording_2_125_chunk01

#figure(image("assets/ext_recording_2_125_chunk01.png", height: 78%), caption: [ext_recording_2_125_chunk01])

= RPS sample: ext_recording_2_125_chunk02

#figure(image("assets/ext_recording_2_125_chunk02.png", height: 78%), caption: [ext_recording_2_125_chunk02])

= RPS sample: ext_recording_2_125_chunk03

#figure(image("assets/ext_recording_2_125_chunk03.png", height: 78%), caption: [ext_recording_2_125_chunk03])

= RPS sample: ext_recording_2_125_chunk04

#figure(image("assets/ext_recording_2_125_chunk04.png", height: 78%), caption: [ext_recording_2_125_chunk04])

= RPS sample: sample_00000

#figure(image("assets/sample_00000.png", height: 78%), caption: [sample_00000])

= RPS sample: sample_00149

#figure(image("assets/sample_00149.png", height: 78%), caption: [sample_00149])

= RPS sample: sample_00299

#figure(image("assets/sample_00299.png", height: 78%), caption: [sample_00299])

= RPS sample: sample_00449

#figure(image("assets/sample_00449.png", height: 78%), caption: [sample_00449])

= RPS sample: sample_00599

#figure(image("assets/sample_00599.png", height: 78%), caption: [sample_00599])

= Section 2: RPS-informed DCUNet vs baseline DCUNet

*TBD* — training in progress. Final comparison and metrics will be added once training completes.

= RPS-informed DCUNet vs baseline DCUNet (training in progress)

#figure(image("assets/dcunet_sdr_training.png", height: 65%), caption: [DCUNet+RPS vs DCUNet baseline — SDR over steps])

Validation SDR over training steps: *DCUNet+RPS (DREGON)* vs *DCUNet baseline (DREGON)*. Full results TBD.

= Summary / Conclusion

- *Prediction of motor speeds from noisy mixtures* seems to be quite an easy task.
- After the *RPS-informed DCUNet ablation* experiment is done, I'll move to *joint training* for denoising and motor speeds prediction on the DREGON dataset and see how it goes.
- Overall, *positive evidence* towards the idea of using predicted motor speeds when ground-truth motor speeds are unavailable.
