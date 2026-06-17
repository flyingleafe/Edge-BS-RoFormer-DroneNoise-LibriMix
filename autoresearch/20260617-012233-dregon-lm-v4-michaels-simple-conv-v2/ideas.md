# Autoresearch Ideas — 20260617-012233-dregon-lm-v4-michaels-simple-conv-v2

Dataset: `DREGON-LM-V4-michaels`  
Baseline: `simple_conv_v2`  
Target metrics: PIT MSE (lower is better), R^2 (higher is better, 1.0 is max)

## Initial user idea seed

Start near simple_conv_v2. Prefer small, comparable variants that can smoke-test with one forward pass and train within gpushort. Keep the first batch hypothesis-diverse rather than many hyperparameter-only tweaks. Example hypotheses: use transformer layer instead of BiGRUs, experiment with sliding window attention; also experiment with input features, multi-resolution STFTs, discrete wavelet transforms or something similar.

## Hypothesis log

Add entries before implementation.

| ID | Status | Model key | Hypothesis | Expected mechanism | Risk |
|----|--------|-----------|------------|--------------------|------|
| H0 | completed | simple_conv_v2 | Baseline reference trained with identical parameters. | Establish comparable score floor. | Short gpushort run may undertrain. |
| H1 | completed — worse than baseline | simple_conv_v2_transformer | Replace the BiGRU temporal head in `simple_conv_v2` with a small Transformer encoder. | Self-attention can model rotor-speed trajectories and frame-to-frame harmonic continuity without recurrent bottlenecks. | Underperformed badly: PIT MSE 43.5184, R² -0.6571; likely lacks useful recurrence/local smoothness bias or needs different tuning. |
| H2 | completed — worse than baseline | simple_conv_v2_local_attn | Use a local/sliding-window temporal attention head instead of global recurrence. | RPS varies smoothly, so bounded context may capture useful dynamics with lower compute and stronger locality bias. | PIT MSE 18.5846, R² 0.5213; local attention improved over global Transformer but still much worse than BiGRU. |
| H3 | completed — near baseline, worse | simple_conv_v2_multires | Fuse two STFT magnitude resolutions before the existing encoder/head. | Rotor harmonics benefit from high frequency resolution while transients/changes benefit from shorter windows. | PIT MSE 8.9704, R² 0.8088; close but not better, extra resolution did not beat baseline. |
| H4 | completed — near baseline, worse | simple_conv_v2_dwt | Add a lightweight wavelet-like temporal frontend/branch alongside STFT magnitude. | Wavelet/time-scale features may expose periodic structure and RPS changes not obvious in a fixed STFT grid. | PIT MSE 8.8957, R² 0.8133; closest candidate but still worse than baseline. |
