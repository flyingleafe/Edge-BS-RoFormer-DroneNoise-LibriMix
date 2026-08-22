"""Classical multi-pitch baseline: inverse harmonic clustering by optimal transport.

A NumPy/SciPy reimplementation of the *stochastic* estimator (``Proposed_c``)
of A. Björkman and F. Elvander, "Inverse Harmonic Clustering for Multi-Pitch
Estimation: An Optimal Transport Approach", IEEE TSP 2026 (arXiv:2508.02471),
kept here as a baseline for blind 4-rotor drone speed estimation.

Layout:

* :mod:`~experiments.otmp_baseline.cost` — grids and the eq-(18) ground cost.
* :mod:`~experiments.otmp_baseline.solver` — the Bregman proximal-gradient
  outer loop (Prop. 1), the block-coordinate inner dual (Prop. 2) with its
  water-filling step, and the debiasing program eq (37).
* :mod:`~experiments.otmp_baseline.estimate` — analysis front end, per-frame
  and per-clip estimation, and the parameter presets of Tables I / II.
* :mod:`~experiments.otmp_baseline.simulation` — the Sec. VIII-A Monte-Carlo
  self-test (gross error rate).
* :mod:`~experiments.otmp_baseline.drone_smoke` — the same estimator on real
  4-rotor clips of the frozen DREGON-LM validation split, scored with per-frame
  Hungarian (PIT) matching against the rotor-speed targets.

Both of the last two are runnable::

    PYTHONPATH=src python -m experiments.otmp_baseline.simulation --draws 50
    PYTHONPATH=src python -m experiments.otmp_baseline.drone_smoke --clips 2
"""

from experiments.otmp_baseline.cost import ground_cost, linear_grid
from experiments.otmp_baseline.estimate import (
    FrameEstimate,
    OTMPConfig,
    drone_config,
    estimate_clip,
    estimate_frame,
    real_config,
    simulated_config,
)

__all__ = [
    "FrameEstimate",
    "OTMPConfig",
    "drone_config",
    "estimate_clip",
    "estimate_frame",
    "ground_cost",
    "linear_grid",
    "real_config",
    "simulated_config",
]
