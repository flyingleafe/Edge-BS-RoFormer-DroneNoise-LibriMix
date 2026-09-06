"""Repository-wide test configuration.

CPU cap (user rule, 2026-09-06): the CRF and comb-slot tests are CPU-bound
and torch takes every core by default. Leave half the cores free.
``HNS_TEST_THREADS`` overrides the cap.
"""

from __future__ import annotations

import os

_cap = int(os.environ.get("HNS_TEST_THREADS", max(1, (os.cpu_count() or 2) // 2)))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(_cap))
try:
    import torch

    torch.set_num_threads(_cap)
except Exception:  # torch absent in a docs-only environment
    pass
