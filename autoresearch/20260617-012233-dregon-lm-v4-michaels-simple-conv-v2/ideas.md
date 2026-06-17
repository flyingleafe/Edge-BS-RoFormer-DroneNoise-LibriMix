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
| H5 | completed — worse than baseline | simple_conv_v2_magphase | Keep the winning `simple_conv_v2` encoder/pool/BiGRU pattern but use log-magnitude plus phase (`cos`, `sin`) input. | Phase may expose frame-to-frame harmonic motion while preserving the BiGRU temporal bias that beat attention variants. | PIT MSE 10.4266, R² 0.7466; phase input hurt under this setup. |
| H6 | completed — worse than baseline | simple_conv_v2_dual_pool | Concatenate attention frequency pooling with simple mean frequency pooling before the BiGRU head. | Attention may focus on harmonics while mean pooling preserves broad energy/context lost by attention-only pooling. | PIT MSE 9.8217, R² 0.7462; dual pooling did not help. |
| H7 | completed — mixed, not better primary metric | simple_conv_v2_gru96 | Increase BiGRU hidden size from 64 to 96 while keeping the rest of `simple_conv_v2` unchanged. | If baseline is capacity-limited in temporal tracking, modest GRU capacity may improve without changing inductive bias. | PIT MSE 8.6612, R² 0.8216; best candidate and slightly higher R² than baseline but worse primary PIT MSE. |
| H8 | submitted/running | simple_conv_v2_uni_gru | Replace only the BiGRU temporal head with a unidirectional GRU head, preserving the current STFT and encoder. | Tests how much score depends on future recurrence; should be closest to baseline while making the recurrent part streamable. | Existing symmetric STFT/encoder temporal padding still uses lookahead; this is causal-head-only, not fully waveform causal. |
| H9 | submitted/pending | simple_conv_v2_causal_gru | Use causal STFT framing, left-padded temporal convolutions, and a unidirectional GRU. | A truly streaming neural stack may trade some accuracy for deployability; frequency attention is per-frame so remains causal in time. | Smoke-tested; latest job remained pending after >10 s, so submissions stopped before H10. |
| H10 | smoke-tested, not submitted | simple_conv_v2_causal_gru96 | Same causal stack as H9, but widen the unidirectional GRU to 96 hidden units. | H7 suggests extra GRU capacity can recover signal; widening may offset the loss of bidirectional context. | Implemented and smoke-tested, but not submitted because H9 remained pending after >10 s. |
