#!/usr/bin/env python3
"""
Replaced — DREGON-LM v2 RPS cross-evaluation figures via the unified API.

Original generated:
  - fig_cross_eval.pdf     — cross-eval bar chart
  - fig_degradation.pdf    — degradation by domain shift
  - fig_pit_std_gap.pdf    — PIT vs standard gap
  - fig_v3_training_curves.pdf

Now produced by evaluate-rps + make-plot::

    evaluate-rps --input-set datasets/DREGON-LM/valid -m A@... -m B@... -o results.json
    make-plot --type=rps_prediction.summary_metrics --results results.json
    make-plot --type=rps_prediction.training_curves --log results/rps_exp/training_log.csv

See legacy/generate_figures_v2.py for the original 239-line script.
"""
if __name__ == "__main__":
    print(__doc__)
