# comb_floor_learned — phase-aware front-end on the comb floor

`comb_floor_base` with one change: the log-magnitude STFT front-end is replaced
by `learned_conv`, a free time-domain filterbank whose responses (real,
imaginary, log-magnitude) give the trunk access to phase. `init="stft"` starts
at the windowed DFT basis, so the arm begins at the baseline representation.

Read against `comb_floor_base` 2.535 and `comb_floor_deep` 2.155.
