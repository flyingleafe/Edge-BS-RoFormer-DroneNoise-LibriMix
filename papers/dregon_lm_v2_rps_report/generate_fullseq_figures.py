#!/usr/bin/env python3
"""
Replaced — DREGON-LM v2 RPS full-sequence figures via the unified API.

The original 338-line script evaluated multiple model variants on full-sequence
DREGON recordings and generated per-model 3-panel figures (spectrogram + RPS + MSE).

Now produced by::

    make-plot --type=rps_prediction.full_sequence <audio+rps arrays via Python API>

See legacy/generate_fullseq_figures.py for the original.
"""
if __name__ == "__main__":
    print(__doc__)
