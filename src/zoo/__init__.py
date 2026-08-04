"""Model zoo (docs/refactor-2026-08-plan.md § 3.3).

Discover trained checkpoints on the R2 artifact store and load any of them
as a Frame → Frame callable::

    import zoo

    zoo.model_types()          # every registered model type, merged
    zoo.refresh()              # (re)list the R2 artifact store into the cache
    zoo.checkpoints()          # cached rows: experiment, files, metrics, ...
    fm = zoo.load("rps_simple_conv_v2_v4")   # FrameModel, ready to call
    pred = fm(sample_frame)                  # td.Frame in, td.Frame out

The cache is a gitignored ``<repo-root>/.checkpoints-cache.json``; R2 stays
the source of truth. See ``zoo.cache`` for the schema and refresh mechanics.
"""

from models.registry import model_types
from zoo.cache import CACHE_FILENAME, REPO_ROOT, CacheInfo, checkpoints, refresh
from zoo.frame_model import FrameModel, load

__all__ = [
    "model_types",
    "refresh",
    "checkpoints",
    "load",
    "FrameModel",
    "CacheInfo",
    "CACHE_FILENAME",
    "REPO_ROOT",
]
