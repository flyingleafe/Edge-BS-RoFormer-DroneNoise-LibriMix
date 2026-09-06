"""Put this directory on sys.path so the shared fixture builder imports by name.

``--import-mode=importlib`` (``pyproject.toml``) does not add a test file's own
directory, and ``_joint_fixture`` is shared by two test modules — the joint
acceptance test and the refactoring guard — so it is imported, not copied.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
