"""Classical (non-learned) per-rotor RPS baselines and their protocol evaluation.

The five estimators of the 2026-05-29 study — pyin, cepstrum, harmonic product
spectrum, a matched comb-template bank, and harmonic-dictionary NMF — restored
from commit ``00753c4`` (report:
``writing/reports/2026-05-29_classical-baselines/``), plus a driver that scores
them on the frozen full-envelope validation split with the same per-frame
Hungarian (PIT) convention as every modern model.

Layout:

* :mod:`~experiments.classical_rps.predictors` — the five estimators and the
  ``CLASSICAL_TRACKERS`` dictionary.
* :mod:`~experiments.classical_rps.valid_eval` — the gridrun evaluation driver.

Run the driver with::

    PYTHONPATH=src python -m experiments.classical_rps.valid_eval \\
        --out results/classical_valid_eval --jobs 8
"""

from experiments.classical_rps.predictors import (
    CLASSICAL_TRACKERS,
    N_ROTORS,
    cepstral_tracker,
    evaluate_predictions,
    hps_tracker,
    matched_filter_tracker,
    nmf_tracker,
    pyin_single_f0,
)

__all__ = [
    "CLASSICAL_TRACKERS",
    "N_ROTORS",
    "cepstral_tracker",
    "evaluate_predictions",
    "hps_tracker",
    "matched_filter_tracker",
    "nmf_tracker",
    "pyin_single_f0",
]
