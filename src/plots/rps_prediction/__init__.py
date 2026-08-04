# src/plots/rps_prediction/__init__.py — RPS-prediction plot registry.
"""Plot types for the RPS-prediction task.

Each plot function is registered by a dotted name::

    make-plot --type=rps_prediction.sample_comparison ...
"""

from plots import register as _register

from .full_sequence import plot_full_sequence
from .per_snr import plot_per_snr
from .salience_comparison import plot_salience_comparison
from .sample_comparison import plot_sample_comparison
from .summary_metrics import plot_summary_metrics
from .training_curves import plot_training_curves

PLOT_TYPES = {
    "rps_prediction.sample_comparison": plot_sample_comparison,
    "rps_prediction.salience_comparison": plot_salience_comparison,
    "rps_prediction.summary_metrics": plot_summary_metrics,
    "rps_prediction.per_snr": plot_per_snr,
    "rps_prediction.training_curves": plot_training_curves,
    "rps_prediction.full_sequence": plot_full_sequence,
}

for name, fn in PLOT_TYPES.items():
    _register(name, fn)
