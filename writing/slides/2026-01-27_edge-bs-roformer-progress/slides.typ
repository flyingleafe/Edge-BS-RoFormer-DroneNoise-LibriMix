#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [Edge-BS-RoFormer Progress Update],
  subtitle: [UAV Speech Enhancement Research],
  author: [Dmitrii Mukhutdinov],
  date: [2026-01-27],
)

= What Was Done This Week

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    == Edge-BS-RoFormer Fixes

    - Investigated and fixed several training issues
    - Experimented with different hyperparameter configurations
    - *Result*: Fixes did not lead to significant improvement
  ],
  [
    == Diffusion Buffer Implementation

    - Reimplemented *Diffusion Buffer* model from scratch
    - Based on: _"Diffusion Buffer: Online Diffusion-based Speech Enhancement"_
    - Applied to the DroneNoise-LibriMix dataset
    - Currently debugging training issues
  ],
)

= Diffusion Buffer: Key Idea

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Problem*: Standard diffusion models are *too slow*
    - Multiple score model calls per frame required
    - RTF >> 1 → real-time processing impossible

    *Diffusion Buffer solution*:
    - Align diffusion time-steps with physical time
    - Introduce a *buffer of B frames*
    - Frames closer to present → more noise
    - Frames further in past → progressively denoised
    - *Only one score model call* per input frame!
  ],
  [
    *Trade-off*: Buffer size B controls latency vs quality
    - Latency = hop\_size × B
    - Achievable: *320-960ms* latency

    *Key insight*: By aligning diffusion steps with time, we amortize the cost of denoising across multiple frames
  ],
)

= Diffusion Buffer: Concept Diagram

#align(center)[
  #block(inset: 1em)[
    *Noisy Audio Stream*

    Past → ... → Current

    #v(0.8em)
    *Diffusion Buffer (B frames)*

    Frame 1 (🟢 Clean) → Frame 2 → ... → Frame B (🔴 Noisy)

    #v(0.8em)
    *Output*

    Enhanced

    #v(1em)
    Current — "Add noise" → Frame B

    Frame 1 — "Pop" → Enhanced
  ]
]

= Diffusion Buffer: Training & Inference

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    == Training Process

    + Sample clean/noisy pair from dataset
    + Pad with K-1 leading zeros (init)
    + Randomly crop K=128 frames (~2s)
    + Sample ascending time-steps t⃗
    + Compute perturbed input V^t⃗
    + Optimize denoising score matching loss
  ],
  [
    == Online Inference

    + Initialize empty buffer V^t⃗
    + For each new frame R in stream:
      - Pop first frame → output
      - Append R + σ\_tB·Z to buffer end
      - Run *one reverse step* for all frames
    + Output delay = hop\_size × B
  ],
)

#v(1em)
#align(center)[
  *Key advantage*: Score model called only *once* per hop → real-time processing
]

= Diffusion Buffer: Architecture Flow

#align(center)[
  #block(inset: 1em)[
    *Input*

    Noisy 16kHz → STFT → Compress

    #v(0.8em)
    *Diffusion Buffer*

    Compress → Buffer B → +Noise → NCSN++ → Reverse → (back to Buffer B)

    #v(0.8em)
    *Output*

    Buffer B → Pop → ISTFT → Enhanced
  ]
]

= BBED SDE Configuration

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1.5em,
  [
    *SDE Parameters*

    #table(
      columns: 2,
      [Param], [Value],
      [Type], [BBED],
      [c], [0.08],
      [k], [2.6],
      [T], [0.8],
      [t\_eps], [0.03],
    )

    *Audio*: SR=16kHz, Win=510, Hop=256
  ],
  [
    *NCSN++ Network*

    #table(
      columns: 3,
      [Param], [Orig], [Red],
      [Ch], [128], [96],
      [Blocks], [6], [4],
      [Res], [2], [1],
      [Params], [65M], [18M],
    )

    *Training*: Adam, LR=1e-4, Batch=32
  ],
  [
    *Key Design Choices*

    - Reduced network → faster inference
    - Single score call per frame
    - Buffer size B controls latency
    - BBED SDE: fewer steps needed

    *Latency*: B=20→320ms, B=60→960ms
  ],
)

= Training Issue: Metrics Degrading

#align(center)[
  *Problem*: Validation metrics decrease during training
]

#v(1em)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  align(center)[*Train Loss* ↓ Decreasing normally],
  align(center)[*SI-SDR* ↓ Going down (bad!)],
  align(center)[*SDR* ↓ Also degrading],
)

#v(1em)
#align(center)[
  *Investigation*: Loss↓ but metrics↓ → overfitting or loss-metric mismatch?
]

= Model Comparison: Metrics

#align(center)[
  *SI-SDR*, *STOI*, *PESQ* across SNR levels (-30 to -5 dB) for all models
]

= Audio Sample Comparison: Spectrograms

#align(center)[
  Waveform and spectrogram comparison for *sample_00033* across all models
]

= Audio Sample Comparison: Listen

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  align(center)[*Edge-BS-RoFormer*],
  align(center)[*DCUNet*],
  align(center)[*DPTNet*],
)

#v(1em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  align(center)[*HTDemucs*],
  align(center)[*Diffusion-Buffer*],
)

= Next Steps

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    == Diffusion Buffer

    - _Results promising even with broken train loop!_
    - Debug the training issue
    - Check loss-metric alignment
    - Verify data preprocessing pipeline
    - Compare with paper's training curves
    - Try different buffer sizes (B=5, 10, 30, 60)
    - Scale the model size!
  ],
  [
    == Edge-BS-RoFormer

    - Explore alternative training strategies
    - Consider ensemble approaches
    - Investigate combining with diffusion?
  ],
)

= Questions?
